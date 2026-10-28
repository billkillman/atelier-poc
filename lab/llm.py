"""Appels headless à Claude Code (`claude -p`) et Codex (`codex exec`), derrière une interface commune (Agent).

Sécurité : l'agent lit du contenu web non fiable. Chaque appel est donc verrouillé :
- Claude : --restricted (aucun outil d'exécution de code, outils fichiers confinés au
  dossier de travail, réglages utilisateur ignorés) + --safe-mode (ni CLAUDE.md, ni hooks,
  ni plugins) + liste blanche explicite d'outils + refus automatique de toute permission.
- Codex : sandbox en lecture seule, configuration utilisateur ignorée, session éphémère.
Les jetons d'authentification restent dans l'environnement, jamais dans le dossier de travail.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

_QUOTA_RE = re.compile(r"usage limit|rate limit|limit reached|quota|too many requests|\b429\b|overloaded", re.I)
_AUTH_RE = re.compile(r"not logged in|unauthori[sz]ed|\b401\b|login required|invalid api key|authentication", re.I)


class LLMError(RuntimeError):
    pass


class QuotaError(LLMError):
    pass


class AuthError(LLMError):
    pass


class RefusalError(LLMError):
    """Le filtre de politique d'usage a refusé la requête : inutile de la renvoyer à l'identique."""


_REFUS_RE = re.compile(r"can't help with this|cannot help with this|legal/aup|usage policy", re.I)


class Comptabilite:
    """Trace chaque appel (durée, coût équivalent API, tours, recherches web) pour piloter le quota."""

    def __init__(self):
        self.appels: list[dict] = []
        self._lock = threading.Lock()

    def ajouter(self, **kw):
        with self._lock:
            self.appels.append(kw)

    def resume(self) -> dict:
        par_etape: dict[str, dict] = {}
        for a in self.appels:
            e = par_etape.setdefault(f'{a["moteur"]}:{a["etape"]}', {"appels": 0, "echecs": 0, "secondes": 0.0, "cout_equivalent_usd": 0.0, "recherches_web": 0})
            e["appels"] += 1
            e["echecs"] += 0 if a.get("ok") else 1
            e["secondes"] = round(e["secondes"] + a.get("secondes", 0), 1)
            e["cout_equivalent_usd"] = round(e["cout_equivalent_usd"] + (a.get("cout_usd") or 0), 4)
            e["recherches_web"] += a.get("recherches_web", 0)
        return par_etape


COMPTA = Comptabilite()


_CIBLES_WINDOWS = {
    "claude": ["node_modules/@anthropic-ai/claude-code/bin/claude.exe"],
    "codex": ["node_modules/@openai/codex/bin/codex.js"],
}


def _exe(nom: str) -> list[str]:
    """Commande à lancer. Sous Windows, contourne les shims npm `.cmd` : cmd.exe
    réinterprète les guillemets et caractères spéciaux des arguments JSON."""
    chemin = shutil.which(nom)
    if not chemin:
        raise LLMError(f"exécutable introuvable : {nom}")
    if chemin.lower().endswith(".cmd"):
        for rel in _CIBLES_WINDOWS.get(nom, []):
            cible = Path(chemin).parent / rel
            if cible.exists():
                return [str(cible)] if cible.suffix == ".exe" else [shutil.which("node") or "node", str(cible)]
    return [chemin]


# --------------------------------------------------------------------------- Claude Code

def claude_json(prompt: str, schema: dict, *, etape: str, model: str, cwd: Path,
                tools: tuple[str, ...] = (), effort: str | None = None, timeout: int = 900,
                tentatives: int = 3) -> dict:
    """Exécute `claude -p` et renvoie la sortie structurée validée par --json-schema."""
    outils = ",".join(tools)
    options = [
        "-p",
        "--model", model,
        "--restricted", "--safe-mode", "--strict-mcp-config",
        "--tools", outils,
        "--permission-mode", "dontAsk",
        "--no-session-persistence",
        "--output-format", "json",
        "--json-schema", json.dumps(schema, ensure_ascii=False),
    ]
    if outils:
        options += ["--allowedTools", outils]
    if effort:
        options += ["--effort", effort]
    # Pas de mise à jour automatique pendant une exécution : elle remplace le binaire en cours de route.
    env = {**os.environ, "DISABLE_AUTOUPDATER": "1"}
    derniere = None
    for essai in range(tentatives):
        if essai:
            time.sleep(30 * essai)  # laisse passer une mise à jour ou une indisponibilité passagère
        t0 = time.time()
        try:
            # Résolu à chaque essai : le chemin du binaire peut changer après une mise à jour.
            proc = subprocess.run([*_exe("claude"), *options], input=prompt.encode("utf-8"), cwd=cwd,
                                  capture_output=True, timeout=timeout, env=env)
        except (OSError, LLMError) as e:  # binaire absent le temps d'une mise à jour
            derniere = LLMError(f"{etape} : lancement impossible — {e}")
            log.warning("%s (essai %d/%d)", derniere, essai + 1, tentatives)
            continue
        except subprocess.TimeoutExpired:
            COMPTA.ajouter(moteur="claude", etape=etape, ok=False, secondes=timeout, erreur="timeout")
            derniere = LLMError(f"{etape} : délai dépassé ({timeout}s)")
            continue
        dt = time.time() - t0
        sortie = proc.stdout.decode("utf-8", "replace")
        erreur = proc.stderr.decode("utf-8", "replace")
        try:
            d = json.loads(sortie)
        except json.JSONDecodeError:
            d = {"is_error": True, "result": (sortie or erreur)[-2000:]}
        usage = d.get("usage", {}) or {}
        COMPTA.ajouter(
            moteur="claude", etape=etape, modele=model, ok=not d.get("is_error"), secondes=round(dt, 1),
            cout_usd=d.get("total_cost_usd"), tours=d.get("num_turns"),
            recherches_web=(usage.get("server_tool_use") or {}).get("web_search_requests", 0),
        )
        if not d.get("is_error") and d.get("structured_output") is not None:
            return d["structured_output"]
        message = f'{d.get("result", "")} {d.get("api_error_status") or ""} {erreur[-500:]}'
        if _REFUS_RE.search(message):
            raise RefusalError(f"{etape} : requête refusée par le filtre de politique d'usage")
        if _QUOTA_RE.search(message):
            raise QuotaError(f"{etape} : quota Claude atteint — {message[:300]}")
        if _AUTH_RE.search(message):
            raise AuthError(f"{etape} : authentification Claude — {message[:300]}")
        derniere = LLMError(f"{etape} : {d.get('subtype') or 'erreur'} — {message[:500]}")
        log.warning("%s (essai %d/%d)", derniere, essai + 1, tentatives)
    raise derniere


# --------------------------------------------------------------------------- Codex

# Mots-clés refusés par le mode strict des sorties structurées d'OpenAI : retirés à l'appel de Codex.
# Les bornes correspondantes sont rétablies par le pipeline sur la réponse (scores bornés, listes tronquées).
_HORS_STRICT = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minItems", "maxItems",
                "minLength", "maxLength", "pattern", "format", "default"}


def schema_strict(schema):
    """Copie du schéma sans les mots-clés hors du mode strict (les noms de propriétés sont préservés)."""
    if isinstance(schema, list):
        return [schema_strict(x) for x in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k in _HORS_STRICT:
            continue
        if k in ("properties", "$defs", "definitions") and isinstance(v, dict):
            out[k] = {nom: schema_strict(s) for nom, s in v.items()}
        else:
            out[k] = schema_strict(v)
    return out


def codex_json(prompt: str, schema: dict, *, etape: str, model: str, recherche: bool = False,
               effort: str | None = None, timeout: int = 900, tentatives: int = 2) -> dict:
    """`codex exec` en lecture seule, dans un dossier vide (tout ce qu'il doit lire est dans le prompt).
    Lève QuotaError / AuthError / LLMError comme claude_json."""
    if shutil.which("codex") is None:
        raise AuthError(f"{etape} : codex introuvable")
    derniere = None
    for essai in range(tentatives):
        if essai:
            time.sleep(20 * essai)
        with tempfile.TemporaryDirectory() as tmp:
            schema_path, sortie_path = Path(tmp) / "schema.json", Path(tmp) / "sortie.json"
            schema_path.write_text(json.dumps(schema_strict(schema), ensure_ascii=False), encoding="utf-8")
            vide = Path(tmp) / "travail"
            vide.mkdir()
            cmd = _exe("codex") + (["--search"] if recherche else []) + [
                "exec", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config", "--ignore-rules",
                "--sandbox", "read-only", "--color", "never",
                # Dossier vide : Codex ne voit aucun fichier du projet. (Sous Windows, le sandbox lecture
                # seule refuse de toute façon toute commande, y compris la lecture de fichiers.)
                "-C", str(vide), "--output-schema", str(schema_path), "-o", str(sortie_path), "-m", model,
            ]
            if effort:
                cmd += ["-c", f'model_reasoning_effort="{effort}"']
            cmd.append("-")
            t0 = time.time()
            try:
                proc = subprocess.run(cmd, input=prompt.encode("utf-8"), capture_output=True, timeout=timeout,
                                      env={**os.environ, "NO_COLOR": "1"})
            except subprocess.TimeoutExpired:
                COMPTA.ajouter(moteur="codex", etape=etape, modele=model, ok=False, secondes=timeout, erreur="timeout")
                derniere = LLMError(f"{etape} : délai dépassé ({timeout}s)")
                continue
            dt = time.time() - t0
            texte = sortie_path.read_text(encoding="utf-8") if sortie_path.exists() else ""
        erreur = (proc.stderr + proc.stdout).decode("utf-8", "replace")[-3000:]
        ok = proc.returncode == 0 and bool(texte.strip())
        COMPTA.ajouter(moteur="codex", etape=etape, modele=model, ok=ok, secondes=round(dt, 1))
        if ok:
            try:
                return json.loads(texte)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", texte, re.S)
                if m:
                    return json.loads(m.group(0))
                derniere = LLMError(f"{etape} : sortie illisible")
                continue
        dernier_mot = (erreur.strip().splitlines() or [""])[-1][:300]
        if _QUOTA_RE.search(erreur):
            raise QuotaError(f"{etape} : quota Codex atteint — {dernier_mot}")
        if _AUTH_RE.search(erreur):
            raise AuthError(f"{etape} : authentification Codex — {dernier_mot}")
        derniere = LLMError(f"{etape} : codex en échec (code {proc.returncode}) — {dernier_mot}")
        log.warning("%s (essai %d/%d)", derniere, essai + 1, tentatives)
    raise derniere


# --------------------------------------------------------------------------- agents

_OUTILS_WEB = {"WebSearch", "WebFetch"}


class Agent:
    """Un modèle d'un SDK, derrière une interface commune. `lit_fichiers` : l'agent peut ouvrir les fichiers
    du dossier de travail (Claude Code, outils Read/Grep) ; sinon, le pipeline place les sources dans le prompt."""

    def __init__(self, sdk: str, modele: str):
        self.sdk, self.modele = sdk, modele
        self.lit_fichiers = sdk == "claude"

    def json(self, prompt: str, schema: dict, *, etape: str, cwd: Path, outils: tuple[str, ...] = (),
             effort: str | None = None, timeout: int = 900, tentatives: int | None = None) -> dict:
        if self.sdk == "claude":
            return claude_json(prompt, schema, etape=etape, model=self.modele, cwd=cwd, tools=outils, effort=effort,
                               timeout=timeout, tentatives=tentatives or 3)
        if self.sdk == "codex":
            return codex_json(prompt, schema, etape=etape, model=self.modele, recherche=bool(set(outils) & _OUTILS_WEB),
                              effort=effort, timeout=timeout, tentatives=tentatives or 2)
        raise LLMError(f"SDK inconnu : {self.sdk}")
