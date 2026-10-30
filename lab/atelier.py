"""Atelier des preuves de concept (POC) : exécute une commande de POC et en publie le livrable dans un dépôt public.

Ce module est synchronisé, avec ce dont il dépend (llm, store, colab, tools/poc, prompts et schémas des POC), dans le
dépôt public de l'atelier, qui exécute les POC sur ses propres minutes GitHub Actions. Une commande ne contient que
du contenu public et neutre : le sujet (contenu public de la source), le plan technique, les faits vérifiés et les
modèles à utiliser. Le choix des sujets et leur intérêt pour l'équipe qui les commande restent chez elle.

Déroulé d'une commande (`python -m lab.atelier executer commandes/<jour>/<sujet>.json` dans l'atelier) :
1. écriture : le modèle rédige les cellules ; l'atelier les encadre (livrable.py : en-tête, mise en route, paramètres,
   configuration des données internes, initialisation, titres numérotés, bilan) ; règles de lisibilité vérifiées et
   corrigées avant toute exécution (controler_lisibilite, deux passes de mise en forme au plus) ;
2. contrôle statique, puis sandbox CPU à échelle réduite (SCALE « reduced ») : phase publique, puis phase privée
   (données factices, faux service à la convention OpenAI) ; erreurs renvoyées au modèle (`iterations_max`) ;
3. échelle complète (SCALE « full », phase publique), sur le matériel décidé par l'atelier (`materiel_complet`) :
   GPU Colab pour tout entraînement PyTorch ou TensorFlow, ou quand l'analyse le justifie ; CPU du sandbox sinon ;
4. revue de méthode (fuite, équité, validité) sur les résultats les plus complets ; un problème bloquant repart en
   correction ;
5. livraison (dossier `livraison/` : noms anglais, README illustré, figures, contrôle de neutralité bloquant) ; puis `publier` crée le dépôt daté du POC et y pousse le livrable (le jeton
   de publication n'est présent qu'à cette étape, jamais pendant l'exécution du code écrit par le modèle).

Sécurité : le modèle n'a aucun outil. Le code qu'il écrit s'exécute seulement dans un conteneur Docker sans secret,
sans privilège, aux ressources bornées, ou sur une machine Colab sans secret (voir colab.py). Sans Docker, rien ne
s'exécute (sauf en développement, ATELIER_SANS_ISOLATION=1, à réserver à du code relu).
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import unicodedata
from pathlib import Path
from string import Template

import nbformat
import requests
import yaml

from . import livrable, llm
from .colab import Colab
from .livrable import (NOTEBOOK, SECTIONS, SECTIONS_REQUISES, _nb_md, api_aide, assembler,
                       bilan_markdown, env_exemple, modeles_par_taille, nombre,
                       script_telechargement)
from .store import read_json, write_json

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
REPRISES = ROOT / "reprises"   # reprises/<jour>/<sujet>/<étape>.json : étapes rédigées lors d'un run précédent
OUTILS = ROOT / "tools" / "poc"


def _prompt(nom: str, **valeurs) -> str:
    return Template((ROOT / "prompts" / f"{nom}.md").read_text(encoding="utf-8")).safe_substitute(**valeurs)


def _schema(nom: str) -> dict:
    return json.loads((ROOT / "schemas" / f"{nom}.schema.json").read_text(encoding="utf-8"))


def charger_reglages() -> dict:
    """config/atelier.yaml dans le dépôt de l'atelier ; à défaut (dépôt qui commande les POC), le bloc poc de settings.yaml."""
    atelier = ROOT / "config" / "atelier.yaml"
    if atelier.exists():
        return yaml.safe_load(atelier.read_text(encoding="utf-8"))
    s = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    return {"poc": s["poc"], "timeouts_secondes": s.get("timeouts_secondes", {})}


ROUTES_SERVICE = {"llmaas": "/chat/completions", "sttaas": "/audio/transcriptions", "embeddings": "/embeddings"}


_SECRET = re.compile(r"""(sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{30,}|(api[_-]?key|token|secret)\s*=\s*["'][^"']{12,}["'])""", re.I)


_VARIABLES_ENV = r"(?:LLMAAS_|STTAAS_)[A-Z_]*|PRIVATE_DATA|LOCAL_MODELS_DIR"
_ECRIT_ENV = re.compile(rf"""os\.environ\s*(\[\s*["']({_VARIABLES_ENV})["']\s*\]\s*=|\.(update|setdefault)\s*\(.*({_VARIABLES_ENV}))""")


LIGNES_MAX_CELLULE = 40
MISES_EN_FORME = 2   # passes de mise en forme par itération, avant exécution


_INSTALL = re.compile(r"^\s*[!%]\s*pip\b|pip\s+install|conda\s+install", re.M)


def materiel_complet(plan: dict, faits: dict, conf: dict) -> tuple[str, str]:
    """Matériel de l'échelle complète, décidé par l'atelier et non par le modèle : (« gpu » | « cpu », raison).
    Règle : tout entraînement PyTorch ou TensorFlow sur GPU ; sinon, GPU seulement si l'analyse le justifie
    (modèle trop gros pour le CPU, durée estimée hors du budget CPU)."""
    for role in ("reference", "nouvelle"):
        if plan[role].get("entrainement") == "pytorch_tensorflow":
            return "gpu", f"entraînement PyTorch ou TensorFlow (approche {ROLES[role]} : {plan[role]['nom']})"
    for role in ("reference", "nouvelle"):
        for m in plan[role]["modeles_hf"]:
            p = (faits.get("modeles", {}).get(m) or {}).get("parametres") or 0
            if p > conf["params_max_milliards"] * 1e9:
                return "gpu", f"{m} ({p / 1e9:.1f} Md de paramètres) trop gros pour le CPU du sandbox"
    budget = conf["timeouts"]["complete_cpu"] // 60
    if plan["duree_estimee_minutes"] > budget:
        return "gpu", f"échelle complète estimée à {plan['duree_estimee_minutes']} min sur CPU, au-delà du budget de {budget} min"
    return "cpu", plan.get("analyse_materiel") or "ni entraînement PyTorch ou TensorFlow, ni modèle ou durée hors du budget CPU"


def preinstalles() -> list[str]:
    """Paquets préinstallés dans l'image du sandbox (lus dans le Dockerfile : source unique)."""
    texte = (OUTILS / "Dockerfile").read_text(encoding="utf-8").replace("\\\n", " ")  # lignes de continuation jointes
    paquets = []
    for instruction in texte.splitlines():
        if not instruction.startswith("RUN"):
            continue
        for bloc in re.findall(r"pip install([^&]*)", instruction):
            paquets += [m for m in bloc.split() if not m.startswith(("-", "http")) and re.fullmatch(r"[A-Za-z0-9][\w.-]*", m)]
    return sorted(set(paquets))


# Mots français fréquents dans des noms de variables ou de fonctions : le code publié est écrit en anglais.
_MOTS_FRANCAIS = set("""donnee donnees resultat resultats modele modeles entrainement entrainer entraine etiquette etiquettes
classe jeu jeux taille graine chemin chemins fichier fichiers sortie sorties appel appels liste nombre valeur valeurs seuil
seuils lot lots exemple exemples approche approches metrique metriques reponse reponses texte textes colonne colonnes ligne
lignes calcul calculer charger chargement evaluer nouvelle nouveau probabilite probabilites apprentissage motif motifs
niveau niveaux erreur erreurs mesure mesures duree debut moyenne ecart tirage melange melanger choisir sens vrai faux oui
non tache taches rappel appareil lire ecrire afficher tableau courbe locuteur locuteurs enregistrer telecharger verifier
construire preparer normaliser transcrire decouper decoupage echantillon echantillons poids couche couches jeton jetons
consigne pli plis bloc blocs confiance vraie vraies predite predites predit""".split())


def _identifiants(source: str) -> set[str]:
    """Noms définis par le code : fonctions, classes, arguments, variables affectées."""
    src = "\n".join("pass  # magique" if l.lstrip().startswith(("%", "!")) else l for l in source.splitlines())
    try:
        arbre = ast.parse(src)
    except SyntaxError:
        return set()
    noms = set()
    for n in ast.walk(arbre):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            noms.add(n.name)
        elif isinstance(n, ast.arg):
            noms.add(n.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            noms.add(n.id)
    return noms


def _nom_francais(nom: str) -> bool:
    if not nom.isascii():
        return True
    morceaux = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", nom)
    return any(m.lower() in _MOTS_FRANCAIS for m in morceaux)


def controler_neutralite(code: dict) -> list[str]:
    """Termes qui rattacheraient le dépôt public à un secteur ou à son origine (contrôle bloquant)."""
    erreurs = []
    for i, c in enumerate(code["cellules"]):
        for terme in livrable.termes_interdits(c["source"]):
            erreurs.append(f"cellule {i} ({c['section']}) : terme « {terme} » interdit dans un dépôt public (dire « données "
                           "internes », « services financiers », « expérience »)")
    textes = [code.get("protocole"), code.get("ecarts_sandbox"), code.get("resume_simple"), *(code.get("limites") or []),
              *[f"{g.get('terme')} {g.get('definition')}" for g in code.get("glossaire") or []],
              *[f"{m.get('libelle')} {m.get('explication')}" for m in code.get("lecture_des_metriques") or []]]
    for terme in sorted({x for s in textes for x in livrable.termes_interdits(s or "")}):
        erreurs.append(f"champs du livrable (protocole, limites, glossaire, métriques) : terme « {terme} » interdit")
    return erreurs


def controler_lisibilite(code: dict, plan: dict | None = None) -> list[str]:
    """Règles de lisibilité du notebook publié : cellules courtes, chacune introduite par une explication, fonctions
    documentées, noms en anglais, sections dans l'ordre, figures, métriques et glossaire expliqués."""
    erreurs, cellules = [], code["cellules"]
    ordre = list(dict.fromkeys(SECTIONS.values()))
    rang_max, francais = -1, set()
    for i, c in enumerate(cellules):
        rang = ordre.index(SECTIONS[c["section"]]) if c["section"] in SECTIONS else -1
        if rang < rang_max:
            erreurs.append(f"cellule {i} ({c['section']}) : section hors de l'ordre {list(SECTIONS)}")
        rang_max = max(rang_max, rang)
        if c["type"] != "code":
            continue
        n = len(c["source"].strip().splitlines())
        if n > LIGNES_MAX_CELLULE:
            erreurs.append(f"cellule {i} ({c['section']}) : {n} lignes (au plus {LIGNES_MAX_CELLULE}) : découpe-la, chaque "
                           "morceau introduit par une cellule Markdown")
        if i == 0 or cellules[i - 1]["type"] != "markdown":
            erreurs.append(f"cellule {i} ({c['section']}) : cellule de code sans cellule Markdown juste avant pour dire ce "
                           "qu'elle fait et pourquoi")
        try:
            arbre = ast.parse("\n".join("pass" if l.lstrip().startswith(("%", "!")) else l for l in c["source"].splitlines()))
            sans_doc = [n.name for n in ast.walk(arbre) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not ast.get_docstring(n)]
            erreurs += [f"cellule {i} ({c['section']}) : `{nom}` sans docstring" for nom in sans_doc]
        except SyntaxError:
            pass
        francais |= {nom for nom in _identifiants(c["source"]) if _nom_francais(nom)}
    if francais:
        erreurs.append("noms de variables ou de fonctions en français (le code publié est en anglais ; docstrings et "
                       f"commentaires restent en français) : {', '.join(sorted(francais)[:15])}")
    sources = lambda *s: "\n".join(c["source"] for c in cellules if c["type"] == "code" and c["section"] in s)
    if not any(c["section"] == "context" and c["type"] == "markdown" for c in cellules):
        erreurs.append("section « context » absente : explique la question et l'idée de la nouvelle approche en Markdown")
    if "save_figure(" not in sources("public_data", "private_data"):
        erreurs.append("aucune figure sur les données (save_figure dans public_data) : répartition des classes, longueurs, durées…")
    if "plot_comparison(" not in sources("evaluation"):
        erreurs.append("évaluation : appelle plot_comparison(results) après save_results")
    if "save_figure(" not in sources("analysis"):
        erreurs.append("aucune figure d'analyse (save_figure dans analysis) : erreurs par classe, matrice de confusion, fiabilité…")
    if plan and not code.get("lecture_des_metriques"):
        erreurs.append("lecture_des_metriques vide : chaque métrique doit avoir un libellé et une explication")
    if len(code.get("glossaire") or []) < 3:
        erreurs.append("glossaire : au moins 3 termes techniques expliqués simplement")
    if not (code.get("resume_simple") or "").strip():
        erreurs.append("resume_simple vide : 3 à 5 phrases sur la question et l'idée, pour un lecteur non spécialiste")
    return erreurs


def controler_cellules(code: dict, faits: dict, plan: dict | None = None) -> list[str]:
    """Contrôle statique avant exécution : syntaxe, sections, secrets, installation, code distant non épinglé, modèles
    du plan (vérifiés) bien utilisés, substituts du sandbox compris."""
    erreurs = []
    codes = [c for c in code["cellules"] if c["type"] == "code"]
    presentes = {c["section"] for c in codes}
    erreurs += [f"section inconnue « {c['section']} »" for c in code["cellules"] if c["section"] not in SECTIONS]
    erreurs += [f"section de code « {s} » absente" for s in SECTIONS_REQUISES if s not in presentes]
    tout = "\n".join(c["source"] for c in codes)
    for i, c in enumerate(code["cellules"]):
        if c["type"] != "code":
            continue
        src = "\n".join("pass  # magique" if l.lstrip().startswith(("%", "!")) else l for l in c["source"].splitlines())
        try:
            ast.parse(src)
        except SyntaxError as e:
            erreurs.append(f"cellule {i} ({c['section']}) : erreur de syntaxe ligne {e.lineno} : {e.msg}")
        if _SECRET.search(c["source"]):
            erreurs.append(f"cellule {i} ({c['section']}) : secret écrit en dur (clé, jeton)")
        if _ECRIT_ENV.search(c["source"]):
            erreurs.append(f"cellule {i} ({c['section']}) : écrit une variable LLMAAS_/STTAAS_/PRIVATE_DATA/LOCAL_MODELS_DIR (elles viennent de l'environnement)")
        if _INSTALL.search(c["source"]):
            erreurs.append(f"cellule {i} ({c['section']}) : installation de paquets dans le code (utilise `dependances`)")
        if re.search(r"def\s+(save_results|llm_chat|load_private_data|bootstrap_ci|wer|save_figure|plot_comparison|model_path|configure)\s*\(", c["source"]):
            erreurs.append(f"cellule {i} ({c['section']}) : redéfinit une fonction de l'aide commune")
        if re.search(r"PRIVATE_CONFIG\s*(\[[^\]]*\]\s*)*=(?!=)|PRIVATE_CONFIG\.(update|setdefault|pop)\(", c["source"]):
            erreurs.append(f"cellule {i} ({c['section']}) : modifie PRIVATE_CONFIG (cellule de configuration générée, laissée à l'utilisateur)")
        if re.search(r"PRIVATE_DATA|LOCAL_MODELS_DIR", c["source"]):
            erreurs.append(f"cellule {i} ({c['section']}) : lit la configuration des données internes directement (passe par load_private_data et model_path)")
        if re.search(r"^\s*(from\s+helpers\s+import|import\s+helpers)", c["source"], re.M):
            erreurs.append(f"cellule {i} ({c['section']}) : réimporte helpers (déjà importé par la cellule d'initialisation)")
        if re.search(r"google\.colab|drive\.mount|authenticate_user|google\.auth", c["source"]):
            erreurs.append(f"cellule {i} ({c['section']}) : accès au compte Google interdit (google.colab, Drive, authentification)")
        if "trust_remote_code=True" in c["source"].replace(" ", "") and "revision=" not in c["source"]:
            erreurs.append(f"cellule {i} ({c['section']}) : trust_remote_code=True sans revision épinglée")
    if "save_results(" not in tout:
        erreurs.append("aucun appel à save_results")
    if "PHASE" not in tout:
        erreurs.append("le code ne distingue pas les phases (variable PHASE)")
    if "load_private_data(" not in tout:
        erreurs.append("la phase private ne charge pas ses données par load_private_data(...)")
    for role in ("reference", "nouvelle") if plan else ():
        substituts = plan[role].get("modeles_hf_sandbox") or []
        erreurs += [f"modèle {m} du plan ({ROLES[role]}) absent du code" for m in [*plan[role]["modeles_hf"], *substituts] if m not in tout]
        if substituts and "by_scale(" not in tout:
            erreurs.append(f"substituts de l'échelle réduite ({', '.join(substituts)}) non choisis par by_scale(reduced=..., full=...)")
    return erreurs + controler_neutralite(code)


class Sandbox:
    """Exécute un notebook dans un conteneur Docker sans secret (ou localement en développement, sur demande explicite)."""

    def __init__(self, conf: dict):
        self.image = os.environ.get("ATELIER_IMAGE", conf.get("image", "atelier-sandbox:latest"))
        self.mode = None
        if shutil.which("docker") and subprocess.run(["docker", "image", "inspect", self.image], capture_output=True).returncode == 0:
            self.mode = "docker"
        elif os.environ.get("ATELIER_SANS_ISOLATION") == "1":
            self.mode = "local"
        cpus, ram = os.cpu_count() or 2, None
        try:
            ram = int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1)) / 2 ** 20
        except (OSError, AttributeError):
            pass
        s = conf.get("sandbox", {})
        self.cpus = min(cpus, s.get("cpus_max", 4))
        self.ram_go = round(min(ram * 0.8 if ram else s.get("ram_go", 6), s.get("ram_go_max", 12)), 1)

    def description(self, conf: dict, gpu: str = "") -> str:
        t = conf["timeouts"]
        return (f"Sandbox de validation : Linux, CPU seul ({self.cpus} cœurs, {self.ram_go} Go de RAM), aucun GPU. Accès à "
                f"Internet pour pip et Hugging Face, sans jeton (rien d'accès restreint). Python 3.12 ; ffmpeg et libsndfile "
                f"présents. Le notebook y est exécuté en SCALE = « reduced » (échelle réduite), de la première à la dernière "
                f"cellule, sans intervention : phase publique en {t['publique'] // 60} min au plus, téléchargements compris "
                f"({t['cellule'] // 60} min par cellule), puis phase private (données factices) en {t['privee'] // 60} min au plus.\n"
                f"Échelle complète (SCALE = « full », phase public seulement), exécutée une fois le code validé, sur le "
                f"matériel choisi par l'atelier : GPU NVIDIA T4 (16 Go, CUDA, Google Colab, {conf['colab']['timeout'] // 60} min "
                f"au plus) pour tout entraînement PyTorch ou TensorFlow, pour un modèle de plus de {conf['params_max_milliards']} Md "
                f"de paramètres ou pour une durée CPU estimée au-delà de {t['complete_cpu'] // 60} min ; sinon ce même CPU, "
                f"en {t['complete_cpu'] // 60} min au plus.{(' ' + gpu) if gpu else ''}")

    def verifier_reseau(self) -> str | None:
        """Le conteneur résout-il et joint-il Hugging Face et PyPI ? Sinon, aucun notebook ne peut réussir : panne du
        sandbox, à ne jamais renvoyer au modèle (le 2026-10-26, un pare-feu trop strict a coupé le DNS des conteneurs)."""
        if self.mode != "docker":
            return None
        test = ("import socket, urllib.request\n"
                "for h in ('huggingface.co', 'pypi.org', 'files.pythonhosted.org', 'datasets-server.huggingface.co'):\n"
                "    socket.getaddrinfo(h, 443)\n"
                "urllib.request.urlopen('https://huggingface.co/api/models/almanach/camembert-base', timeout=20).read(1)\n")
        proc = subprocess.run(["docker", "run", "--rm", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "1000:1000",
                               self.image, "python", "-c", test], capture_output=True, timeout=180)
        return None if proc.returncode == 0 else (proc.stderr or proc.stdout).decode("utf-8", "replace").strip().splitlines()[-1][:300]

    def executer(self, dossier: Path, phase: str, delai: int, delai_cellule: int, taille: str = "sandbox",
                 modeles: list[str] | None = None) -> dict:
        nom_sortie = phase if taille == "sandbox" else "complete"  # l'échelle complète n'écrase pas l'échelle réduite
        args = ["--entree", "poc.ipynb", "--sortie", f"execute_{nom_sortie}.ipynb", "--rapport", f"rapport_{nom_sortie}.json",
                "--phase", phase, "--taille", taille, "--timeout-cellule", str(min(delai_cellule, delai))]
        if phase == "publique" and taille == "sandbox":
            args += ["--freeze", "requirements_sandbox.txt"]
        if phase == "privee" and modeles:  # servis hors ligne, comme dans un environnement cible sans Internet
            args += ["--modeles", ",".join(modeles)]
        rapport_chemin = dossier / f"rapport_{nom_sortie}.json"
        rapport_chemin.unlink(missing_ok=True)
        t0 = time.time()
        if self.mode == "docker":
            for chemin in [dossier, *dossier.rglob("*")]:  # l'utilisateur du conteneur (uid 1000) doit pouvoir écrire
                try:
                    chemin.chmod(0o777 if chemin.is_dir() else 0o666)
                except OSError:
                    pass
            nom = f"poc-{uuid.uuid4().hex[:10]}"
            script = ('export PATH="$PYTHONUSERBASE/bin:$PATH"; '
                      'if [ -s dependances.txt ]; then pip install --user --quiet --no-warn-script-location -r dependances.txt '
                      '> installation.log 2>&1 || { tail -n 60 installation.log; exit 97; }; fi; '
                      f'exec python /outils/executer.py {" ".join(args)}')
            cmd = ["docker", "run", "--rm", "--name", nom, "--cpus", str(self.cpus), "--memory", f"{self.ram_go}g",
                   "--memory-swap", f"{self.ram_go}g", "--shm-size", "1g", "--pids-limit", "4096", "--cap-drop", "ALL",
                   "--security-opt", "no-new-privileges", "--user", "1000:1000",
                   "-e", "HOME=/travail/.maison", "-e", "PYTHONUSERBASE=/travail/.local", "-e", "HF_HOME=/travail/.cache/huggingface",
                   "-e", "PIP_CACHE_DIR=/travail/.cache/pip", "-v", f"{dossier}:/travail", "-v", f"{OUTILS}:/outils:ro",
                   "-w", "/travail", self.image, "sh", "-c", script]
        elif self.mode == "local":
            env = {k: v for k, v in os.environ.items() if not re.search(r"TOKEN|KEY|SECRET|PASSWORD|AUTH|LLMAAS|STTAAS|PRIVATE_DATA|LOCAL_MODELS_DIR", k, re.I)}
            nom, cmd = None, [sys.executable, str(OUTILS / "executer.py"), *args]
        else:
            raise RuntimeError("aucun sandbox : Docker et l'image du sandbox sont requis (ATELIER_SANS_ISOLATION=1 en développement)")
        try:
            proc = subprocess.run(cmd, cwd=dossier, capture_output=True, timeout=delai + 300,
                                  env=env if self.mode == "local" else None)
            sortie = (proc.stdout + proc.stderr).decode("utf-8", "replace")
        except subprocess.TimeoutExpired:
            if nom:
                subprocess.run(["docker", "kill", nom], capture_output=True)
            progres = read_json(dossier / f"progres_{phase}.json", {})
            return {"phase": phase, "taille": taille, "ok": False, "secondes": round(time.time() - t0),
                    "erreur": {"nature": f"délai global dépassé ({delai // 60} min)", "cellule": progres.get("cellule"),
                               "section": progres.get("section"), "source": progres.get("source", ""), "message": ""}}
        if rapport_chemin.exists():
            return read_json(rapport_chemin, {})
        nature = "installation des dépendances impossible" if proc.returncode == 97 else f"le sandbox s'est arrêté (code {proc.returncode})"
        return {"phase": phase, "taille": taille, "ok": False, "secondes": round(time.time() - t0),
                "erreur": {"nature": nature, "cellule": None, "section": None, "source": "", "message": sortie[-3000:]}}


def compte_rendu(rapport: dict) -> str:
    """Résumé d'une exécution pour le modèle (erreur exacte, fin des sorties)."""
    if not rapport:
        return "(pas d'exécution)"
    echelle = "échelle complète" if rapport.get("taille") == "complete" else "échelle réduite"
    lignes = [f"Phase {rapport.get('phase')} ({echelle}, {rapport.get('materiel') or 'CPU du sandbox'}) : "
              f"{'succès' if rapport.get('ok') else 'échec'} en {rapport.get('secondes')} s."]
    e = rapport.get("erreur")
    if e:
        lignes += [f"Erreur : {e.get('nature')} — cellule {e.get('cellule')} (section {e.get('section')}).",
                   "Source de la cellule :", "```python", (e.get("source") or "")[:3000], "```",
                   "Message :", "```", (e.get("message") or "")[-2500:], "```"]
    for s in (rapport.get("sorties") or [])[-6:]:
        if s.get("texte"):
            lignes += [f"Sortie de la cellule {s['cellule']} ({s.get('section')}) :", "```", s["texte"][-1200:], "```"]
    if rapport.get("appels_service"):
        lignes.append(f"Appels au faux service : {rapport['appels_service']}")
    return "\n".join(lignes)


_PANNE_RESEAU = re.compile(r"Temporary failure in name resolution|Name or service not known|Network is unreachable|"
                           r"No route to host|nodename nor servname|getaddrinfo failed|Could not resolve host", re.I)


def panne_infrastructure(rapport: dict | None) -> str | None:
    """Erreur due au sandbox (réseau coupé) plutôt qu'au code : à ne pas renvoyer au modèle."""
    e = (rapport or {}).get("erreur") or {}
    m = _PANNE_RESEAU.search(f"{e.get('message', '')} {e.get('nature', '')}")
    return f"réseau du sandbox indisponible : {m.group(0)}" if m else None


ECHELLES_REVUE = {
    "complete": "Les résultats de la phase publique viennent de l'échelle complète (tout le jeu, réglages complets) : juge-les "
                "pleinement, chiffres compris.",
    "prealable": "Revue préalable : seule l'échelle réduite a tourné (sous-échantillon, entraînement écourté, parfois un modèle "
                 "substitut). L'échelle complète tournera ensuite si tu ne relèves rien de bloquant, et sera revue à son tour. "
                 "Les chiffres de l'échelle réduite testent la mécanique : un niveau faible, même proche du hasard, est attendu et "
                 "n'est jamais bloquant en soi. Est bloquant ce qui fausserait aussi l'échelle complète : bug de calcul ou de mesure "
                 "(probabilités qui ne somment pas à 1, métrique hors de ses bornes, statistique exactement nulle ou constante qui "
                 "trahit un bogue), fuite de données, comparaison inéquitable par construction, approche qui n'est pas celle de la fiche.",
    "reduite": "Seule l'échelle réduite a tourné, et l'échelle complète ne tournera pas ({raison}) : le POC sera publié à cette "
               "échelle, avec ses réserves. Un niveau faible, même proche du hasard, est attendu à cette échelle et n'est jamais "
               "bloquant en soi. Est bloquant ce qui fausserait aussi l'échelle complète (bug de calcul ou de mesure, fuite, "
               "comparaison inéquitable par construction, approche qui n'est pas celle de la fiche) ; la conclusion doit dire que "
               "les chiffres ne permettent pas de trancher, sans les prendre pour un résultat.",
}


def meme_metrique(a: str, b: str) -> bool:
    """Même métrique, à la graphie près (« macro_f1 » et « macro-F1 ») : la forme ne doit pas coûter une itération."""
    norme = lambda s: re.sub(r"[^0-9a-z]", "", unicodedata.normalize("NFKD", str(s)).lower())
    return norme(a) == norme(b)


def valider_execution(pub: dict, ban: dict | None, plan: dict, min_exemples: int) -> list[str]:
    erreurs = []
    r = pub.get("resultats")
    if not pub.get("ok"):
        return ["la phase publique a échoué"]
    if not r:
        return ["la phase publique n'a pas enregistré de résultats (save_results)"]
    if r["dataset"]["n_examples"] < min_exemples:
        erreurs.append(f"évaluation sur {r['dataset']['n_examples']} exemples seulement (au moins {min_exemples} à l'échelle réduite)")
    if not meme_metrique(r["primary_metric"], plan["metrique_principale"]):
        erreurs.append(f"métrique principale {r['primary_metric']!r} au lieu de {plan['metrique_principale']!r} (plan)")
    if ban is None:
        return erreurs
    if not ban.get("ok"):
        return erreurs + ["la phase private (données factices, faux service) a échoué"]
    if not ban.get("resultats"):
        erreurs.append("la phase private n'a pas enregistré de résultats")
    appels = ban.get("appels_service") or {}
    for role in ("reference", "nouvelle"):
        service = plan[role]["service_phase2"]
        if service != "aucun" and not any(rt.endswith(ROUTES_SERVICE[service]) for rt in appels.get("routes", [])):
            erreurs.append(f"phase private : l'approche {role} devait appeler le service {service} ({ROUTES_SERVICE[service]}), aucun appel reçu")
    if appels.get("refus"):
        erreurs.append(f"phase private : {appels['refus']} appel(s) refusé(s) par le faux service (authentification, format)")
    return erreurs


def valider_complete(complet: dict, reduit: dict, plan: dict) -> list[str]:
    """L'échelle complète compte si elle a tourné jusqu'au bout et mesure la même chose que l'échelle réduite validée."""
    if not complet.get("ok"):
        e = complet.get("erreur") or {}
        return [f"{e.get('nature') or 'échec'}" + (f" (cellule {e['cellule']}, section {e.get('section')})" if e.get("cellule") is not None else "")]
    r, ref = complet.get("resultats"), reduit["resultats"]
    if not r:
        return ["aucun résultat enregistré"]
    erreurs = []
    if not meme_metrique(r["primary_metric"], plan["metrique_principale"]):
        erreurs.append(f"métrique principale {r['primary_metric']!r} au lieu de {plan['metrique_principale']!r}")
    cles = lambda res: sorted((a["role"], tuple(sorted(a["metrics"]))) for a in res["approaches"])
    if cles(r) != cles(ref):
        erreurs.append("approches ou métriques différentes de celles de l'échelle réduite")
    if r["dataset"]["n_examples"] < ref["dataset"]["n_examples"]:
        erreurs.append(f"{r['dataset']['n_examples']} exemples d'évaluation, moins qu'à l'échelle réduite ({ref['dataset']['n_examples']})")
    return erreurs


ROLES = {"reference": "référence", "nouvelle": "nouvelle"}


def resume_poc(plan: dict, etat: dict, res: dict, revue: dict) -> dict:
    mp = res["primary_metric"]
    vals = {a["role"]: a["metrics"][mp] for a in res["approaches"]}
    complet = etat.get("complete") or {}
    echelle = "complete" if etat.get("resultats_complets") else "reduite"
    ou = complet.get("materiel") if echelle == "complete" else "échelle réduite, CPU du sandbox"
    synthese = (f"{mp} : {plan['nouvelle']['nom']} {nombre(vals.get('new'))} contre "
                f"{nombre(vals.get('reference'))} pour {plan['reference']['nom']} ({res['dataset']['n_examples']} exemples, {ou}).")
    return {"sujet_id": etat["sujet_id"], "titre": plan["titre"], "statut": "valide", "niveau_sandbox": plan["niveau_sandbox"],
            "echelle": echelle, "materiel": complet.get("materiel") if echelle == "complete" else "CPU du sandbox",
            "echelle_complete": {c: complet.get(c) for c in ("statut", "materiel_requis", "raison_materiel", "raison")},
            "iterations": len(etat["iterations"]), "metrique_principale": mp, "sens": res["direction"],
            "approches": [{"nom": a["name"], "role": "nouvelle" if a["role"] == "new" else a["role"], "valeur": a["metrics"][mp],
                           "ic95": (a.get("ci95") or {}).get(mp)} for a in res["approaches"]],
            "n_exemples": res["dataset"]["n_examples"], "jeu": res["dataset"]["name"], "synthese": synthese,
            "conclusion": revue["conclusion"], "confiance": revue["confiance"], "reproduit_la_source": revue["reproduit_la_source"],
            # Aperçu dans l'application : verdict court, libellé lisible de la métrique, réserves écrites pour le lecteur.
            "en_bref": revue.get("en_bref") or "",
            "libelle_metrique": (livrable.metriques_lisibles(etat.get("code") or {}, res).get(mp) or {}).get("libelle")
                                or (res.get("metric_labels") or {}).get(mp) or mp,
            "reserves": livrable.pour_le_lecteur(revue, "majeur")}


def cause_echec(etat: dict) -> str:
    """« technique » (panne du sandbox) : relancé automatiquement ; « methode » (code, validation, revue) : définitif."""
    it = etat["iterations"][-1] if etat["iterations"] else {}
    return "technique" if it.get("infrastructure") else "methode"


def raison_echec(etat: dict) -> str:
    it = etat["iterations"][-1] if etat["iterations"] else {}
    if etat["statut"] == "non_execute":
        return "non exécuté (mode inspection)"
    if it.get("infrastructure"):
        return f"panne du sandbox ({it['infrastructure']}) : à relancer"
    if it.get("controle"):
        return "contrôle avant exécution : " + "; ".join(it["controle"])[:240]
    if it.get("validation"):
        return "; ".join(it["validation"])[:240]
    if it.get("revue", {}).get("bloquants"):
        return f"la revue de méthode a relevé {it['revue']['bloquants']} problème(s) bloquant(s)"
    return (etat.get("dernier_compte_rendu") or "échec")[:240]



# --------------------------------------------------------------------------- exécution d'une commande

class LivrableNonNeutre(Exception):
    """Le livrable contient un terme interdit dans un dépôt public (livrable.TERMES_INTERDITS)."""


class Redaction:
    """Appels aux modèles de l'atelier : SDK, modèle et effort fixés par la commande, par rôle. Chaque résultat est
    gardé dans le dossier de travail : une relance reprend où l'exécution s'était arrêtée."""

    def __init__(self, moteurs: dict, timeouts: dict, dossier: Path):
        self.moteurs, self.timeouts = moteurs, timeouts
        self.etapes = dossier / "etapes"
        self.etapes.mkdir(parents=True, exist_ok=True)
        self.journal: list[str] = []

    def note(self, msg: str):
        log.info(msg)
        self.journal.append(f"{time.strftime('%H:%M:%S')} {msg}")

    def etape(self, nom: str, fn):
        chemin = self.etapes / f"{nom}.json"
        if chemin.exists():
            self.note(f"[{nom}] reprise du résultat existant")
            return read_json(chemin, None)
        t = time.time()
        self.note(f"[{nom}] début")
        res = fn()
        write_json(chemin, res)
        self.note(f"[{nom}] fin en {time.time() - t:.0f}s")
        return res

    def rediger(self, prompt: str, schema: dict, *, etape: str, role: str) -> dict:
        m = self.moteurs[role]
        with tempfile.TemporaryDirectory() as tmp:
            return llm.Agent(m["sdk"], m["modele"]).json(prompt, schema, etape=etape, cwd=Path(tmp), effort=m.get("effort"),
                                                       timeout=self.timeouts.get(role, 1800))


class Atelier:
    """Exécute une commande : écriture, sandbox, échelle complète, revue de méthode, livraison."""

    def __init__(self, commande: dict, travail: Path, reglages: dict | None = None, colab: Colab | None = None):
        reglages = reglages or charger_reglages()
        self.c, self.conf = commande, reglages["poc"]
        self.sid, self.jour = commande["sujet_id"], commande["jour"]
        self.fiche, self.plan, self.faits = commande["fiche"], commande["plan"], commande["faits"]
        self.dir = travail
        self.dir.mkdir(parents=True, exist_ok=True)
        # Reprise : étapes rédigées lors d'un run précédent (code, revues), déposées par le dépôt qui commande ; relues du
        # cache au lieu d'être redemandées aux modèles, les exécutions sont rejouées (2026-10-28 : POC validé, livraison en échec).
        reprise = REPRISES / self.jour / self.sid
        if reprise.is_dir():
            shutil.copytree(reprise, self.dir / "etapes", dirs_exist_ok=True)
        self.r = Redaction(commande["moteurs"], reglages.get("timeouts_secondes", {}), travail)
        self.sandbox = Sandbox(self.conf)
        self.colab = colab
        self.t0 = time.time()
        self.duree_complete: float | None = None   # durée mesurée d'une échelle complète réussie de ce POC
        self.regles = _prompt("poc_regles", api_aide=api_aide(), preinstalles=", ".join(preinstalles()))
        # Termes interdits dans le dépôt publié : empreintes reçues de config/atelier.yaml (ou calculées dans le dépôt privé).
        livrable.configurer_empreintes(self.conf.get("empreintes_interdites")
                                       or [livrable.empreinte_terme(x) for x in self.conf.get("termes_livrable", [])])

    def note(self, msg: str):
        self.r.note(msg)

    def min_exemples(self) -> int:
        """Taille minimale de l'évaluation : en exemples courts, ou en enregistrements longs (appels, réunions, documents)."""
        if self.plan.get("type_exemples") == "enregistrements_longs":
            return self.conf.get("min_enregistrements_evaluation", 20)
        return self.conf["min_exemples_evaluation"]

    def temps_restant(self) -> float:
        return self.conf["duree_max_minutes"] * 60 - (time.time() - self.t0)

    def sandbox_txt(self) -> str:
        if self.colab is None or self.colab.pret:
            return self.sandbox.description(self.conf)
        return self.sandbox.description(self.conf, f"Aujourd'hui, le GPU est indisponible ({self.colab.raison}) : une échelle "
                                                   "complète qui le demande ne sera pas exécutée, le POC sera publié à l'échelle réduite.")

    # ---------------------------------------------------------------- étapes LLM

    def _json(self, x) -> str:
        return json.dumps(x, ensure_ascii=False, indent=1)

    def ecrire(self) -> dict:
        prompt = _prompt("poc_code", jour=self.jour, fiche=self._json(self.fiche), plan=self._json(self.plan),
                         verifications=self._json(self.faits), sandbox=self.sandbox_txt(), regles=self.regles)
        return self.r.rediger(prompt, _schema("poc_code"), etape=f"poc_code:{self.sid}", role="poc_code")

    def reparer(self, code: dict, motif: str, compte: str) -> dict:
        prompt = _prompt("poc_reparation", motif=motif, plan=self._json(self.plan), verifications=self._json(self.faits),
                         sandbox=self.sandbox_txt(), cellules=self._json(code["cellules"]), compte_rendu=compte, regles=self.regles)
        return self.r.rediger(prompt, _schema("poc_code"), etape=f"poc_reparation:{self.sid}", role="poc_reparation")

    def mettre_en_forme(self, code: dict, defauts: list[str]) -> dict:
        motif = ("il n'est pas encore assez lisible pour être publié : un lecteur doit pouvoir le suivre comme un tutoriel. "
                 "Corrige seulement la présentation (découpage des cellules, explications, noms, docstrings, figures, "
                 "glossaire, lecture des métriques), sans changer ce que calcule le code.")
        compte = "Règles de lisibilité non respectées :\n" + "\n".join(f"- {d}" for d in defauts)
        return self.reparer(code, motif, compte)

    def revoir(self, code: dict, pub: dict, ban: dict | None, reduit: dict | None = None, echelle: str = "complete",
               raison: str = "") -> dict:
        """`echelle` : « complete » (pub est l'échelle complète), « prealable » (échelle réduite, l'échelle complète suivra
        si rien n'est bloquant) ou « reduite » (le POC sera publié à l'échelle réduite, pour `raison`)."""
        publique = compte_rendu(pub) + "\nRésultats enregistrés :\n" + json.dumps(pub.get("resultats"), ensure_ascii=False, indent=1)[:6000]
        if reduit:  # pub est l'échelle complète : l'échelle réduite du sandbox en regard
            r = reduit["resultats"]
            publique += (f"\n\nPour mémoire, échelle réduite validée dans le sandbox (CPU, {r['dataset']['n_examples']} exemples) :\n"
                         + json.dumps({a["name"]: a["metrics"] for a in r["approaches"]}, ensure_ascii=False))
        prompt = _prompt("poc_revue", fiche=self._json(self.fiche), plan=self._json(self.plan), echelle=ECHELLES_REVUE[echelle].format(raison=raison),
                         cellules=self._json(code["cellules"]), publique=publique, privee=compte_rendu(ban), api_aide=api_aide())
        return self.r.rediger(prompt, _schema("poc_revue"), etape=f"poc_revue:{self.sid}", role="poc_revue")

    # ---------------------------------------------------------------- échelle complète

    def budget_complete(self, materiel: str) -> int:
        """Temps à réserver pour l'échelle complète : la durée déjà mesurée sur ce POC (+30 %, +5 min), sinon le délai
        maximal du réglage (le 2026-10-28, 45 min réservées pour un run de 25 min ont empêché de relancer la version corrigée)."""
        plafond = self.conf["colab"]["timeout"] if materiel == "gpu" else self.conf["timeouts"]["complete_cpu"]
        return min(plafond, int(1.3 * self.duree_complete) + 300) if self.duree_complete else plafond

    def empechement_complete(self) -> tuple[dict, str | None]:
        """Matériel de l'échelle complète, et ce qui empêcherait de l'exécuter maintenant (None : exécutable)."""
        materiel, raison = materiel_complet(self.plan, self.faits, self.conf)
        info = {"materiel_requis": materiel, "raison_materiel": raison}
        if materiel == "gpu" and not (self.colab and self.colab.pret):
            pourquoi = self.colab.raison if self.colab else "non configuré"
            return info, f"GPU nécessaire ({raison}) mais indisponible : {pourquoi}"
        if self.temps_restant() < self.budget_complete(materiel) + 600:
            return info, "budget de temps épuisé"
        return info, None

    def echelle_complete(self, reduit: dict) -> dict:
        """Phase publique à l'échelle complète, une fois l'échelle réduite validée, sur le matériel décidé par l'atelier.
        Un échec ne repart pas en correction : le POC reste publiable à l'échelle réduite, avec la raison."""
        info, empechement = self.empechement_complete()
        if empechement:
            return {**info, "statut": "non_execute", "raison": empechement}
        materiel, raison = info["materiel_requis"], info["raison_materiel"]
        # Délai réel : tout le temps restant du job (moins la marge de publication), jamais au-delà du réglage.
        delai = int(self.temps_restant() - 600)
        self.note(f"[poc] {self.sid} : échelle complète sur {'GPU' if materiel == 'gpu' else 'CPU du sandbox'} ({raison})")
        if materiel == "gpu":
            rapport = self.colab.executer(self.dir, OUTILS, delai=delai)
            notebook = self.dir / "complete" / "execute_complete.ipynb"
        else:
            budget = min(self.conf["timeouts"]["complete_cpu"], delai)
            rapport = self.sandbox.executer(self.dir, "publique", budget, budget, taille="complete")
            rapport.setdefault("materiel", f"CPU, {self.sandbox.cpus} cœurs")
            notebook = self.dir / "execute_complete.ipynb"
        erreurs = valider_complete(rapport, reduit, self.plan)
        if not erreurs and rapport.get("secondes"):
            self.duree_complete = float(rapport["secondes"])
        self.note(f"[poc] {self.sid} : échelle complète {'validée' if not erreurs else 'en échec (' + '; '.join(erreurs) + ')'} "
                  f"en {rapport.get('secondes')} s")
        return {**info, "statut": "echec" if erreurs else "valide", "raison": "; ".join(erreurs)[:300], "materiel": rapport.get("materiel"),
                "secondes": rapport.get("secondes"), "rapport": rapport, "notebook": str(notebook)}

    # ---------------------------------------------------------------- un POC

    def un_poc(self, sans_execution: bool = False) -> dict:
        sid, plan, faits, travail = self.sid, self.plan, self.faits, self.dir
        etat = {"sujet_id": sid, "titre": plan["titre"], "niveau_sandbox": plan["niveau_sandbox"], "iterations": [], "statut": "echec",
                "faits": faits}
        code = self.r.etape("code_1", self.ecrire)
        for k in range(1, self.conf["iterations_max"] + 1):
            it = {"numero": k, "diagnostic": code.get("diagnostic", "")}
            etat["iterations"].append(it)
            motif, compte, pub, ban = None, "", {}, None
            # Lisibilité d'abord (deux passes au plus, sans exécution) : le notebook publié doit se lire comme un tutoriel.
            for j in range(1, MISES_EN_FORME + 1):
                defauts = controler_lisibilite(code, plan)
                if not defauts or controler_cellules(code, faits, plan):
                    break
                self.note(f"[poc] {sid} itération {k} : mise en forme {j} ({len(defauts)} défaut(s) de lisibilité)")
                code = self.r.etape(f"mise_en_forme_{k}_{j}", lambda: self.mettre_en_forme(code, defauts))
            it["lisibilite"] = controler_lisibilite(code, plan)
            statiques = controler_cellules(code, faits, plan)
            nb = assembler(self.fiche, plan, faits, code, self.jour)
            nbformat.write(nb, travail / "poc.ipynb")
            shutil.copyfile(OUTILS / "helpers.py", travail / "helpers.py")
            (travail / "dependances.txt").write_text("\n".join(sorted(set(code["dependances"]))) + "\n", encoding="utf-8")
            if statiques:
                it["controle"] = statiques
                motif = "il ne respecte pas les règles de l'atelier (contrôle avant exécution)."
                compte = "Contrôle statique :\n" + "\n".join(f"- {e}" for e in statiques)
            elif sans_execution:
                etat["statut"] = "non_execute"
                break
            else:
                t = self.conf["timeouts"]
                pub = self.sandbox.executer(travail, "publique", t["publique"], t["cellule"])
                it["publique"] = {"ok": pub.get("ok"), "secondes": pub.get("secondes"), "erreur": (pub.get("erreur") or {}).get("nature")}
                self.note(f"[poc] {sid} itération {k} : phase publique {'réussie' if pub.get('ok') else 'en échec'} ({pub.get('secondes')} s)")
                erreurs = valider_execution(pub, None, plan, self.min_exemples())
                if not erreurs:
                    ban = self.sandbox.executer(travail, "privee", t["privee"], t["cellule"], modeles=modeles_par_taille(plan)["reduced"])
                    it["privee"] = {"ok": ban.get("ok"), "secondes": ban.get("secondes"), "erreur": (ban.get("erreur") or {}).get("nature"),
                                    "appels": ban.get("appels_service")}
                    self.note(f"[poc] {sid} itération {k} : phase privée {'réussie' if ban.get('ok') else 'en échec'} ({ban.get('secondes')} s)")
                    erreurs = valider_execution(pub, ban, plan, self.min_exemples())
                panne = panne_infrastructure(pub) or panne_infrastructure(ban)
                if panne:  # le réseau du sandbox, pas le notebook : inutile (et trompeur) de demander une correction
                    it["infrastructure"] = panne
                    etat["dernier_compte_rendu"] = f"panne du sandbox ({panne}) : POC à relancer"
                    self.note(f"[poc] {sid} : panne du sandbox ({panne}), pas de correction demandée")
                    break
                if erreurs:
                    it["validation"] = erreurs
                    motif = "son exécution dans le sandbox a échoué ou ne remplit pas les conditions de validation."
                    compte = "\n".join(f"- {e}" for e in erreurs) + "\n\n" + compte_rendu(pub) + ("\n\n" + compte_rendu(ban) if ban else "")
                else:
                    # Revue préalable, à l'échelle réduite, avant d'engager l'échelle complète (GPU) : un défaut de code ou de
                    # protocole se corrige sans brûler l'échelle complète (le 2026-10-28, un Brier de 672 découvert après 25 min de GPU).
                    prealables = []
                    if self.empechement_complete()[1] is None:
                        pre = self.r.etape(f"revue_prealable_{k}", lambda: self.revoir(code, pub, ban, echelle="prealable"))
                        prealables = [x for x in pre["problemes"] if x["gravite"] == "bloquant"]
                        it["revue_prealable"] = {"bloquants": len(prealables), "confiance": pre["confiance"]}
                        if prealables:
                            etat["revue"] = pre
                    complet = self.echelle_complete(pub) if not prealables else None
                    if complet:
                        it["complete"] = {c: complet.get(c) for c in ("statut", "materiel_requis", "materiel", "secondes", "raison")}
                        revue_sur = complet["rapport"] if complet["statut"] == "valide" else pub
                        revue = self.r.etape(f"revue_{k}", lambda: self.revoir(
                            code, revue_sur, ban, reduit=pub if revue_sur is not pub else None,
                            echelle="complete" if revue_sur is not pub else "reduite", raison=complet.get("raison") or ""))
                        bloquants = [x for x in revue["problemes"] if x["gravite"] == "bloquant"]
                        it["revue"] = {"bloquants": len(bloquants), "confiance": revue["confiance"]}
                    else:
                        bloquants = prealables
                    if not bloquants:
                        etat.update(statut="valide", code=code, publique=pub, privee=ban, revue=revue,
                                    complete={c: v for c, v in complet.items() if c != "rapport"},
                                    resultats_complets=complet["rapport"]["resultats"] if complet["statut"] == "valide" else None)
                        break
                    motif = ("la revue de méthode a relevé des problèmes bloquants : la conclusion serait trompeuse."
                             + (" (Revue préalable à l'échelle réduite : l'échelle complète n'a pas été lancée.)" if not complet else ""))
                    compte = "Problèmes bloquants relevés par la revue :\n" + "\n".join(
                        f"- {x['description']} — correction attendue : {x['correction']}" for x in bloquants)
                    if complet:
                        etat["revue"] = revue
            etat["dernier_compte_rendu"] = compte[-4000:]
            if k == self.conf["iterations_max"]:
                break
            if self.temps_restant() < self.conf["timeouts"]["publique"] + self.conf["timeouts"]["privee"] + 600:
                etat["dernier_compte_rendu"] = "budget de temps épuisé. " + etat["dernier_compte_rendu"]
                break
            if it["lisibilite"]:  # la correction traite aussi ce que la mise en forme n'a pas réglé
                compte += ("\n\nRègles de lisibilité encore non respectées (à corriger aussi) :\n"
                           + "\n".join(f"- {d}" for d in it["lisibilite"]))
            code = self.r.etape(f"code_{k + 1}", lambda: self.reparer(code, motif, compte))
        etat["code_final"] = code
        write_json(travail / "etat.json", {k: v for k, v in etat.items() if k not in ("code", "code_final")})
        return etat

    # ---------------------------------------------------------------- livraison

    def livrer(self, etat: dict) -> dict:
        """Dossier livraison/ : ce qui sera publié tel quel dans le dépôt (noms anglais, aucun terme interdit)."""
        dest = self.dir / "livraison"
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        complet = etat.get("resultats_complets")
        source = Path(etat["complete"]["notebook"]) if complet else self.dir / "execute_publique.ipynb"
        nb = nbformat.read(source, as_version=4)
        nb.cells = [c for c in nb.cells if "injected-parameters" not in c.get("metadata", {}).get("tags", [])]
        nb.metadata.pop("poc", None)
        res, revue, code = complet or etat["publique"]["resultats"], etat["revue"], etat["code"]
        figures = livrable.extraire_figures(nb, dest)
        for c in nb.cells:
            if c.get("metadata", {}).get("section") == "header":
                lisibles = livrable.metriques_lisibles(code, res)
                c.source = livrable.entete_notebook(self.plan, livrable.en_bref(self.plan, res, revue, bool(complet), lisibles), bool(complet),
                                                    lisibles.get(res["primary_metric"], {}).get("libelle", ""))
            for o in c.get("outputs", []):  # sorties volumineuses retirées (dépôt git)
                for k in [k for k in (o.get("data") or {}) if k.startswith("image/") and len(str(o["data"][k])) > 400_000]:
                    del o["data"][k]
                if o.get("output_type") == "stream" and len(o.get("text", "")) > 20_000:
                    o["text"] = "[…]\n" + o["text"][-20_000:]
        nb.cells.append(_nb_md(bilan_markdown(self.plan, etat, res, revue, code, figures), section="summary"))
        with open(dest / NOTEBOOK, "w", encoding="utf-8", newline="\n") as f:
            nbformat.write(nb, f)
        shutil.copyfile(OUTILS / "helpers.py", dest / "helpers.py")
        (dest / "results").mkdir()
        write_json(dest / "results" / "results.json", res)
        if complet:
            write_json(dest / "results" / "results_reduced_scale.json", etat["publique"]["resultats"])
        gel = self.dir / "requirements_sandbox.txt"
        (dest / "requirements.txt").write_text(gel.read_text(encoding="utf-8") if gel.exists() else "", encoding="utf-8", newline="\n")
        (dest / "private.env.example").write_text(env_exemple(self.plan), encoding="utf-8", newline="\n")
        (dest / "download_models.py").write_text(script_telechargement(self.plan, self.faits), encoding="utf-8", newline="\n")
        fichiers = [(NOTEBOOK, "le notebook : question, données, deux approches, évaluation, analyse, bilan (sorties incluses)"),
                    ("helpers.py", "fonctions communes : phases et échelles, services internes, lecture des données, mesures, figures"),
                    ("results/results.json", "résultats publiés" + (" (échelle complète)" if complet else " (échelle réduite)")
                     + " : métriques, intervalles, latences, versions et machine")]
        if complet:
            fichiers.append(("results/results_reduced_scale.json", "résultats de l'échelle réduite (validation de la mécanique, CPU)"))
        if figures:
            fichiers.append(("figures/", "figures du notebook reprises dans ce README"))
        fichiers += [("requirements.txt", "versions exactes des bibliothèques utilisées"),
                     ("private.env.example", "variables d'environnement pour exécuter le notebook sur vos données"),
                     ("download_models.py", "téléchargement des poids des modèles, pour un environnement sans Internet")]
        (dest / "README.md").write_text(livrable.readme(self.fiche, self.plan, self.faits, etat, res, revue, code, figures, fichiers),
                                        encoding="utf-8", newline="\n")
        fuites = livrable.verifier_neutralite(dest)
        if fuites:
            raise LivrableNonNeutre("; ".join(fuites[:8]))
        return resume_poc(self.plan, etat, res, revue)

    def executer(self, sans_execution: bool = False) -> dict:
        if not sans_execution:
            if self.sandbox.mode is None:
                raise SystemExit("aucun sandbox disponible : construire l'image (docker build -t atelier-sandbox tools/poc) ; "
                                 "aucun code produit par un modèle n'est exécuté hors sandbox")
            self.note(f"[poc] sandbox : {self.sandbox.mode}, {self.sandbox.cpus} cœurs, {self.sandbox.ram_go} Go")
            panne = self.sandbox.verifier_reseau()
            if panne:
                raise SystemExit(f"sandbox sans accès à Hugging Face ou PyPI ({panne}) : POC non tenté")
            if self.colab is None:
                self.colab = Colab(self.conf["colab"], self.dir / "colab_sessions.txt")
            self.note(f"[poc] GPU : {self.colab.materiel}, solde {self.colab.solde} unités" if self.colab.pret
                      else f"[poc] GPU indisponible : {self.colab.raison}")
        base = {"jour": self.jour, "sujet_id": self.sid, "depot": self.c["depot"], "titre": self.plan["titre"]}
        try:
            etat = self.un_poc(sans_execution)
            if etat["statut"] == "valide":
                resultat = {**base, **self.livrer(etat)}
            else:
                resultat = {**base, "statut": etat["statut"], "iterations": len(etat["iterations"]), "raison": raison_echec(etat),
                            "cause": cause_echec(etat)}
        except LivrableNonNeutre as e:  # garde-fou : jamais publié avec un terme qui rattacherait le dépôt à son origine
            self.note(f"[poc] livrable non publiable : {e}")
            resultat = {**base, "statut": "echec", "cause": "methode", "raison": f"livrable non neutre : {str(e)[:220]}"}
        except llm.QuotaError as e:
            self.note(f"[poc] quota du modèle atteint : {e}")
            resultat = {**base, "statut": "non_traite", "cause": "technique", "raison": "quota du modèle atteint : à relancer"}
        except llm.LLMError as e:  # modèle indisponible, authentification, réponse illisible après les tentatives
            self.note(f"[poc] modèle en échec : {e}")
            resultat = {**base, "statut": "echec", "cause": "technique", "raison": f"modèle en échec : {str(e)[:200]}"}
        resultat.update(duree_minutes=round((time.time() - self.t0) / 60, 1), consommation=llm.COMPTA.resume())
        write_json(self.dir / "resultat.json", resultat)
        (self.dir / "journal.log").write_text("\n".join(self.r.journal) + "\n", encoding="utf-8")
        return resultat


# --------------------------------------------------------------------------- publication et index

API = "https://api.github.com"


def _git(*args: str, cwd: Path) -> None:
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"git {args[0]} : {(p.stderr or p.stdout).strip()[-300:]}")


def publier(travail: Path, token: str) -> dict:
    """Crée (ou reprend) le dépôt public daté du POC et y pousse le livrable. Seule étape qui détient le jeton."""
    res = read_json(travail / "resultat.json", None)
    if not res or res.get("statut") != "valide":
        return res or {}
    h = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    moi = requests.get(f"{API}/user", headers=h, timeout=30)
    moi.raise_for_status()
    proprietaire = moi.json()["login"]
    r = requests.post(f"{API}/user/repos", headers=h, timeout=30, json={
        "name": res["depot"], "description": res["titre"][:340], "private": False,
        "has_issues": False, "has_projects": False, "has_wiki": False})
    if r.status_code != 201 and not (r.status_code == 422 and "already exists" in r.text):
        raise RuntimeError(f"création du dépôt {res['depot']} impossible : HTTP {r.status_code} {r.text[:200]}")
    if r.status_code == 422:  # dépôt existant : sa description suit le livrable
        requests.patch(f"{API}/repos/{proprietaire}/{res['depot']}", headers=h, timeout=30,
                       json={"description": res["titre"][:340]})
    with tempfile.TemporaryDirectory() as tmp:
        depot = Path(tmp)
        shutil.copytree(travail / "livraison", depot, dirs_exist_ok=True)
        _git("init", "-q", "-b", "main", cwd=depot)
        _git("add", "-A", cwd=depot)
        _git("-c", f"user.name={proprietaire}", "-c", f"user.email={proprietaire}@users.noreply.github.com",
             "commit", "-q", "-m", res["titre"][:200], cwd=depot)
        # Le livrable est régénéré à chaque exécution de la commande : il remplace le précédent.
        _git("push", "-q", "--force", f"https://x-access-token:{token}@github.com/{proprietaire}/{res['depot']}.git", "main", cwd=depot)
    res["url"] = f"https://github.com/{proprietaire}/{res['depot']}"
    write_json(travail / "resultat.json", res)
    return res


def maj_index(racine: Path, res: dict) -> None:
    """index.json de l'atelier : un élément par commande traitée, du plus récent au plus ancien."""
    chemin = racine / "index.json"
    index = read_json(chemin, None) or {"version": 1, "pocs": []}
    index["pocs"] = [p for p in index["pocs"] if (p["jour"], p["sujet_id"]) != (res["jour"], res["sujet_id"])] + [res]
    index["pocs"].sort(key=lambda p: (p["jour"], p["sujet_id"]), reverse=True)
    write_json(chemin, index)


def a_relancer(entree: dict, reprises: int) -> bool:
    """Échec technique (panne, interruption, quota, publication) : relancé `reprises` fois au plus ; échec de méthode
    (code, validation, revue bloquante, livrable non neutre) : jamais."""
    technique = entree.get("statut") == "non_traite" or entree.get("cause") == "technique"
    return entree.get("statut") != "valide" and technique and int(entree.get("tentatives") or 1) <= reprises


def lister(racine: Path, materiel: str | None = None) -> list[str]:
    """Commandes à exécuter : sans résultat dans l'index, ou en échec technique encore relançable. `materiel` (« gpu » |
    « cpu ») : seulement celles dont l'échelle complète demande ce matériel (`materiel_complet`)."""
    conf = charger_reglages()["poc"]
    reprises = int(conf.get("reprises_techniques", 1))
    faits = {(p["jour"], p["sujet_id"]) for p in (read_json(racine / "index.json", None) or {}).get("pocs", [])
             if not a_relancer(p, reprises)}
    attente = []
    for c in sorted(racine.glob("commandes/*/*.json")):
        d = read_json(c, {})
        if (d.get("jour"), d.get("sujet_id")) in faits:
            continue
        if materiel and _materiel_commande(d, conf) != materiel:
            continue
        attente.append(c.relative_to(racine).as_posix())
    return attente


def _materiel_commande(commande: dict, conf: dict) -> str:
    """« gpu » ou « cpu » ; une commande illisible va avec le GPU (exécution une à une : le cas le plus prudent)."""
    try:
        return materiel_complet(commande["plan"], commande.get("faits") or {}, conf)[0]
    except (KeyError, TypeError):
        return "gpu"


def _travail(commande: dict) -> Path:
    return ROOT / "travail" / commande["jour"] / commande["sujet_id"]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(prog=f"python -m {__package__}.atelier")
    sous = ap.add_subparsers(dest="action", required=True)
    sous.add_parser("lister", help="commandes en attente (JSON)").add_argument("--materiel", choices=("gpu", "cpu"))
    for nom in ("executer", "publier", "indexer"):
        p = sous.add_parser(nom)
        p.add_argument("commande", help="chemin de la commande (commandes/<jour>/<sujet>.json)")
    a = ap.parse_args(argv)
    if a.action == "lister":
        print(json.dumps(lister(ROOT, a.materiel)))
        return 0
    commande = read_json(ROOT / a.commande, None)
    if not commande:
        raise SystemExit(f"commande illisible : {a.commande}")
    if a.action == "executer":
        res = Atelier(commande, _travail(commande)).executer()
        log.info("POC %s : %s", commande["sujet_id"], res["statut"] + (f" ({res.get('raison')})" if res.get("raison") else ""))
        return 0
    jeton = os.environ.get("POC_PUBLICATION_TOKEN")
    travail = _travail(commande)
    res = read_json(travail / "resultat.json", None) or {
        "jour": commande["jour"], "sujet_id": commande["sujet_id"], "depot": commande["depot"], "titre": commande["plan"]["titre"],
        "statut": "non_traite", "cause": "technique",
        "raison": "exécution interrompue avant son terme (délai du job ou panne) : à relancer"}
    if a.action == "indexer":  # après publication : réécrit l'entrée sur l'index le plus récent (jobs parallèles)
        maj_index(ROOT, res)
        return 0
    if res.get("statut") == "valide":
        if not jeton:
            raise SystemExit("POC_PUBLICATION_TOKEN absent : publication impossible")
        try:
            res = publier(travail, jeton)
        except (RuntimeError, requests.RequestException) as e:  # le livrable ne survit pas au job : tout sera rejoué
            res = {**res, "statut": "non_traite", "cause": "technique", "raison": f"publication impossible : {str(e)[:200]}"}
    # Tentatives : celles de l'entrée précédente (index lu au début du job), plus celle-ci ; au-delà, échec définitif.
    precedente = next((p for p in (read_json(ROOT / "index.json", None) or {}).get("pocs", [])
                       if (p["jour"], p["sujet_id"]) == (res["jour"], res["sujet_id"])), None)
    res["tentatives"] = int(precedente.get("tentatives") or 1) + 1 if precedente else 1
    reprises = int(charger_reglages()["poc"].get("reprises_techniques", 1))
    if res.get("statut") != "valide" and res.get("cause") == "technique" and res["tentatives"] > reprises:
        res.update(statut="echec", raison=f"{res.get('raison', '')} (abandon après {res['tentatives']} tentatives)"[:300])
    travail.mkdir(parents=True, exist_ok=True)
    write_json(travail / "resultat.json", res)  # relu par « indexer » après la remise à niveau de l'index
    maj_index(ROOT, res)
    log.info("POC %s : %s%s", commande["sujet_id"], res["statut"], f" → {res['url']}" if res.get("url") else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
