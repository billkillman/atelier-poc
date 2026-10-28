"""Échelle complète d'un POC sur un GPU Google Colab (CLI officiel google-colab-cli, compte Colab Pro de l'utilisateur).

- Seulement pour un notebook déjà validé dans le sandbox (exécuté de bout en bout sur CPU, phases publique et banque) :
  phase publique, TAILLE « complete ». Un échec ici ne demande aucune correction au modèle : le POC est publié à
  l'échelle réduite du sandbox, avec la raison.
- Aucun secret sur la machine Colab : le jeton OAuth reste sur le runner (~/.config/colab-cli/token.json, écrit par le
  workflow depuis le secret COLAB_TOKEN) ; le notebook n'y a jamais accès. Le CLI tourne sans les autres secrets.
- Exécution détachée sur la machine Colab, suivie par de courtes requêtes : le jeton d'une session expire au bout
  d'une heure et une connexion longue ne le renouvelle pas. Délai global, puis arrêt de la machine dans tous les cas
  (et, en cas d'arrêt brutal du moteur, par le workflow, depuis le journal des sessions ouvertes).
- Solde d'unités vérifié avant chaque lancement : jamais sur le palier gratuit, où les conditions de Colab proscrivent
  l'usage hors de l'interface des notebooks.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from .store import read_json

# Le CLI remplace un nom de GPU inconnu par une A100 (la plus chère) : liste fermée.
GPUS = ("T4", "L4")
JETON = "~/.config/colab-cli/token.json"

# Lancé sur la machine Colab, détaché du noyau : installe, exécute, puis écrit fin.code (97 : installation impossible).
# Le notebook tourne sans les variables d'identifiants du noyau Colab (propagation des identifiants, jetons).
_LANCEUR = """
import importlib.util, os, pathlib, re, subprocess, sys
for k in [k for k in os.environ if re.search(r"TBE_|CRED|TOKEN|SECRET|KEY|PASSWORD|AUTH", k, re.I)]:
    del os.environ[k]
r = pathlib.Path({racine!r})
paquets = [l.strip() for l in (r / "dependances.txt").read_text().splitlines() if l.strip()]
if importlib.util.find_spec("nbclient") is None:
    paquets.append("nbclient")
if paquets:
    with open(r / "installation.log", "w") as log:
        if subprocess.call([sys.executable, "-m", "pip", "install", "-q", *paquets], cwd=r, stdout=log, stderr=subprocess.STDOUT):
            (r / "fin.code").write_text("97")
            sys.exit()
with open(r / "execution.log", "w") as log:
    code = subprocess.call([sys.executable, "executer.py", "--entree", "poc.ipynb", "--sortie", "execute_complete.ipynb",
                            "--rapport", "rapport_complete.json", "--phase", "publique", "--taille", "complete",
                            "--timeout-cellule", "{delai_cellule}"], cwd=r, stdout=log, stderr=subprocess.STDOUT)
(r / "fin.code").write_text(str(code))
"""

_DEMARRER = """
import subprocess, sys
subprocess.Popen([sys.executable, "-c", {lanceur!r}], cwd={racine!r}, start_new_session=True,
                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("POC_LANCE")
"""

_ETAT = """
import json, pathlib
r = pathlib.Path({racine!r})
lire = lambda n: (r / n).read_text(errors="replace").strip() if (r / n).exists() else None
print("POC_ETAT " + json.dumps({{"fin": lire("fin.code"), "progres": lire("progres_publique.json")}}))
"""


def _env_sans_secret() -> dict:
    return {k: v for k, v in os.environ.items() if not re.search(r"TOKEN|KEY|SECRET|PASSWORD", k, re.I)}


class Colab:
    def __init__(self, conf: dict, journal_sessions: Path):
        self.conf = conf
        self.gpu = str(conf.get("gpu", "T4")).upper()
        self.cli = [*shlex.split(os.environ.get("VEILLE_COLAB_CLI", "colab")), "--auth", "oauth2"]
        self.racine = conf.get("racine", "/content/poc")
        self.journal_sessions = journal_sessions
        self.solde: float | None = None
        self.raison = self._diagnostic()  # None : prêt

    @property
    def pret(self) -> bool:
        return self.raison is None

    @property
    def materiel(self) -> str:
        return f"GPU {self.gpu} (Google Colab)"

    def _cmd(self, *args: str, entree: str | None = None, delai: int = 120) -> subprocess.CompletedProcess:
        try:
            return subprocess.run([*self.cli, *args], input=entree, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=delai, env=_env_sans_secret())
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(args, -1, "", f"délai dépassé ({delai} s) : colab {args[0]}")
        except OSError as e:
            return subprocess.CompletedProcess(args, -1, "", f"CLI Colab inutilisable : {e}")

    def _exec(self, session: str, code: str, delai: int = 90) -> subprocess.CompletedProcess:
        return self._cmd("exec", "-s", session, "--timeout", str(delai), entree=code, delai=delai + 60)

    def _diagnostic(self) -> str | None:
        if not self.conf.get("actif", True):
            return "désactivée (settings.yaml, poc.colab.actif)"
        if self.gpu not in GPUS:
            return f"GPU {self.gpu!r} non autorisé (autorisés : {', '.join(GPUS)})"
        if not Path(os.environ.get("VEILLE_COLAB_JETON", JETON)).expanduser().exists():
            return "jeton Colab absent (secret COLAB_TOKEN)"
        if not shutil.which(self.cli[0]):
            return "CLI Colab absent (pipx install google-colab-cli)"
        p = self._cmd("usage", delai=120)
        m = re.search(r"Current balance:\s*([\d.]+)", p.stdout)
        if p.returncode or not m:
            return f"Colab injoignable : {(p.stderr or p.stdout).strip()[-200:]}"
        self.solde = float(m.group(1))
        seuil = self.conf.get("solde_min_unites", 5)
        if self.solde < seuil:
            return f"solde Colab insuffisant ({self.solde} unités, minimum {seuil}) : jamais sur le palier gratuit"
        return None

    def executer(self, dossier: Path, outils: Path, delai: int | None = None) -> dict:
        """Phase publique, TAILLE « complete », sur GPU. Rapport au format de l'exécuteur, plus « materiel ».
        `delai` : délai global imposé par l'atelier (temps restant du job), jamais au-delà du réglage."""
        sortie = dossier / "complete"
        shutil.rmtree(sortie, ignore_errors=True)
        sortie.mkdir(parents=True)
        delai, intervalle = min(self.conf.get("timeout", 2700), delai or 10**9), self.conf.get("intervalle", 60)
        t0, nom = time.time(), f"poc-{uuid.uuid4().hex[:8]}"
        base = {"phase": "publique", "taille": "complete", "ok": False, "materiel": self.materiel, "resultats": None}

        def echec(nature: str, message: str = "", **kw) -> dict:
            return {**base, "secondes": round(time.time() - t0), **kw,
                    "erreur": {"nature": nature, "cellule": kw.pop("cellule", None), "section": kw.pop("section", None),
                               "source": "", "message": message[-3000:]}}

        with open(self.journal_sessions, "a", encoding="utf-8") as f:
            f.write(nom + "\n")
        try:
            p = self._cmd("new", "-s", nom, "--gpu", self.gpu, delai=600)
            if p.returncode:
                return echec("allocation du GPU Colab impossible", p.stderr + p.stdout)
            p = self._exec(nom, f"import os; os.makedirs({self.racine!r}, exist_ok=True); print('POC_PRET')")
            if "POC_PRET" not in p.stdout:
                return echec("machine Colab inutilisable", p.stderr + p.stdout)
            for f in (dossier / "poc.ipynb", dossier / "dependances.txt", outils / "executer.py"):
                p = self._cmd("upload", "-s", nom, str(f), f"{self.racine}/{f.name}", delai=300)
                if p.returncode:
                    return echec(f"envoi de {f.name} vers Colab impossible", p.stderr + p.stdout)
            lanceur = _LANCEUR.format(racine=self.racine, delai_cellule=self.conf.get("timeout_cellule", 1800))
            p = self._exec(nom, _DEMARRER.format(lanceur=lanceur, racine=self.racine))
            if "POC_LANCE" not in p.stdout:
                return echec("lancement sur Colab impossible", p.stderr + p.stdout)
            fin, progres, ratees = None, None, 0
            while fin is None:
                if time.time() - t0 > delai:
                    break
                time.sleep(intervalle)
                p = self._exec(nom, _ETAT.format(racine=self.racine))
                m = re.search(r"POC_ETAT (\{.*\})", p.stdout)
                if not m:
                    ratees += 1
                    if ratees >= 5:
                        return echec("machine Colab injoignable pendant l'exécution", p.stderr + p.stdout)
                    continue
                ratees, etat = 0, json.loads(m.group(1))
                fin, progres = etat["fin"], etat["progres"] or progres
            if fin is None:
                cel = json.loads(progres) if progres else {}
                return echec(f"délai global dépassé sur Colab ({delai // 60} min)", cellule=cel.get("cellule"), section=cel.get("section"))
            for distant, local in (("rapport_complete.json", "rapport.json"), ("execute_complete.ipynb", "execute_complete.ipynb"),
                                   ("installation.log", "installation.log"), ("execution.log", "execution.log")):
                self._cmd("download", "-s", nom, f"{self.racine}/{distant}", str(sortie / local), delai=300)
            if fin == "97":
                journal = (sortie / "installation.log").read_text(encoding="utf-8", errors="replace") if (sortie / "installation.log").exists() else ""
                return echec("installation des dépendances impossible sur Colab", journal)
            rapport = read_json(sortie / "rapport.json", None)
            if not isinstance(rapport, dict):
                journal = (sortie / "execution.log").read_text(encoding="utf-8", errors="replace") if (sortie / "execution.log").exists() else ""
                return echec(f"exécuteur arrêté sur Colab (code {fin})", journal)
            return {**rapport, "materiel": self.materiel, "secondes_totales": round(time.time() - t0)}
        finally:
            self._cmd("stop", "-s", nom, delai=180)
