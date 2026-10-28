"""Atelier des preuves de concept (POC) : exécute une commande de POC et en publie le livrable dans un dépôt public.

Ce module est synchronisé, avec ce dont il dépend (llm, store, colab, tools/poc, prompts et schémas des POC), dans le
dépôt public de l'atelier, qui exécute les POC sur ses propres minutes GitHub Actions. Une commande ne contient que
du contenu public et neutre : le sujet (contenu public de la source), le plan technique, les faits vérifiés et les
modèles à utiliser. Le choix des sujets et leur intérêt pour l'équipe qui les commande restent chez elle.

Déroulé d'une commande (`python -m veille.atelier executer commandes/<jour>/<sujet>.json`) :
1. écriture : le modèle rédige les cellules ; l'atelier les encadre (en-tête, paramètres, aide commune, bilan) ;
2. contrôle statique, puis sandbox CPU à échelle réduite (TAILLE « sandbox ») : phase publique, puis phase banque
   (données factices, faux service à la convention OpenAI) ; erreurs renvoyées au modèle (`iterations_max`) ;
3. échelle complète (TAILLE « complete », phase publique), sur le matériel décidé par l'atelier (`materiel_complet`) :
   GPU Colab pour tout entraînement PyTorch ou TensorFlow, ou quand l'analyse le justifie ; CPU du sandbox sinon ;
4. revue de méthode (fuite, équité, validité) sur les résultats les plus complets ; un problème bloquant repart en
   correction ;
5. livraison (dossier `livraison/`) ; puis `publier` crée le dépôt daté du POC et y pousse le livrable (le jeton
   de publication n'est présent qu'à cette étape, jamais pendant l'exécution du code écrit par le modèle).

Sécurité : le modèle n'a aucun outil. Le code qu'il écrit s'exécute seulement dans un conteneur Docker sans secret,
sans privilège, aux ressources bornées, ou sur une machine Colab sans secret (voir colab.py). Sans Docker, rien ne
s'exécute (sauf en développement, VEILLE_POC_SANS_ISOLATION=1, à réserver à du code relu).
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
from datetime import date
from pathlib import Path
from string import Template

import nbformat
import requests
import yaml

from . import llm
from .colab import Colab
from .store import read_json, write_json

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
REPRISES = ROOT / "reprises"   # reprises/<jour>/<sujet>/<étape>.json : étapes rédigées lors d'un run précédent
OUTILS = ROOT / "tools" / "poc"
MOIS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre")


def _prompt(nom: str, **valeurs) -> str:
    return Template((ROOT / "prompts" / f"{nom}.md").read_text(encoding="utf-8")).safe_substitute(**valeurs)


def _schema(nom: str) -> dict:
    return json.loads((ROOT / "schemas" / f"{nom}.schema.json").read_text(encoding="utf-8"))


def date_courte(jour: str) -> str:
    d = date.fromisoformat(jour)
    return f"{d.day} {MOIS[d.month - 1]} {d.year}"


def charger_reglages() -> dict:
    """config/atelier.yaml dans le dépôt de l'atelier ; à défaut (dépôt qui commande les POC), le bloc poc de settings.yaml."""
    atelier = ROOT / "config" / "atelier.yaml"
    if atelier.exists():
        return yaml.safe_load(atelier.read_text(encoding="utf-8"))
    s = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    return {"poc": s["poc"], "timeouts_secondes": s.get("timeouts_secondes", {})}


SECTIONS_REQUISES = ("donnees_publiques", "donnees_banque", "reference", "nouvelle", "evaluation")


ROUTES_SERVICE = {"llmaas": "/chat/completions", "sttaas": "/audio/transcriptions", "embeddings": "/embeddings"}


_SECRET = re.compile(r"""(sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{30,}|(api[_-]?key|token|secret)\s*=\s*["'][^"']{12,}["'])""", re.I)


_ECRIT_ENV = re.compile(r"""os\.environ\s*(\[\s*["'](LLMAAS|STTAAS|BANQUE)_[A-Z_]*["']\s*\]\s*=|\.(update|setdefault)\s*\(.*(LLMAAS|STTAAS|BANQUE)_)""")


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


def api_aide() -> str:
    """Signatures et première ligne de documentation des fonctions de tools/poc/aide.py (source unique)."""
    arbre = ast.parse((OUTILS / "aide.py").read_text(encoding="utf-8"))
    lignes = []
    for n in arbre.body:
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_"):
            sig = ast.unparse(n.args)
            doc = " ".join((ast.get_docstring(n) or "").split("\n\n")[0].split())
            lignes.append(f"- `{n.name}({sig})` : {doc}")
    lignes.append("- `JOURNAL_APPELS` : liste des appels aux services (durée, modèle, succès), résumée dans les résultats.")
    return "\n".join(lignes)


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


def _nb_code(source: str, **meta):
    return nbformat.v4.new_code_cell(source, metadata=meta)


def _nb_md(source: str, **meta):
    return nbformat.v4.new_markdown_cell(source, metadata=meta)


def services_du_plan(plan: dict) -> list[str]:
    return sorted({plan[r]["service_phase2"] for r in ("reference", "nouvelle")} - {"aucun"})


def modeles_par_taille(plan: dict) -> dict[str, list[str]]:
    """Modèles chargés à chaque échelle : substituts du sandbox (sinon les mêmes) et modèles de l'échelle complète."""
    sandbox, complete = [], []
    for r in ("reference", "nouvelle"):
        complete += plan[r]["modeles_hf"]
        sandbox += plan[r].get("modeles_hf_sandbox") or plan[r]["modeles_hf"]
    return {"sandbox": sorted(set(sandbox)), "complete": sorted(set(complete))}


def colonnes_chemins(plan: dict) -> list[str]:
    return [c["colonne"] for c in plan["phase2"]["donnees"]
            if re.search(r"chemin|path|fichier|audio|wav|image", f"{c['type']} {c['colonne']}", re.I)]


def cellule_config_banque(plan: dict) -> str:
    """Seule cellule à adapter dans l'environnement de la banque : générée depuis le plan (colonnes attendues,
    colonnes de chemins, modèles, services), valeurs par défaut lues dans l'environnement, aucune clé."""
    services = services_du_plan(plan)
    colonnes = {c["colonne"]: c["colonne"] for c in plan["phase2"]["donnees"]}
    l = ["# Configuration de l'environnement bancaire : seule cellule à adapter pour la phase banque (PHASE = \"banque\").",
         "# Valeurs par défaut lues dans l'environnement (modèle : banque.env.exemple). Aucune clé ici : elle vient de",
         "# <SERVICE>_API_KEY, d'un fichier de secret monté (<SERVICE>_API_KEY_FILE) ou d'une saisie masquée.",
         "import os",
         "",
         "CONFIG_BANQUE = {",
         "    # Jeu bancaire : fichier (csv, tsv, parquet, jsonl, json, xlsx) ou dossier de fichiers de même structure.",
         '    "donnees": os.environ.get("BANQUE_DONNEES", ""),',
         "    # Colonne attendue par le notebook -> nom de la colonne dans votre jeu (à modifier à droite seulement).",
         f'    "colonnes": {colonnes!r},',
         "    # Colonnes de chemins de fichiers (relatifs au fichier de données qui les cite, ou absolus).",
         f'    "chemins": {colonnes_chemins(plan)!r},',
         '    "max_lignes": None,                      # plafond facultatif (échantillon aléatoire reproductible)',
         '    "csv": {"sep": None, "encodage": None},  # None : séparateur et encodage devinés (« ; », Windows-1252…)',
         "    # Poids copiés localement (org--nom/, script telecharger_modeles.py) pour travailler sans Internet ; vide sinon.",
         '    "modeles_dir": os.environ.get("BANQUE_MODELES_DIR", ""),',
         f'    "modeles": {modeles_par_taille(plan)!r},',
         "    # Autorité de certification d'entreprise (fichier PEM) pour les appels HTTPS internes ; vide sinon.",
         '    "certificats": os.environ.get("SSL_CERT_FILE", ""),']
    if "llmaas" in services or "embeddings" in services:
        l += ['    "llmaas": {"base_url": os.environ.get("LLMAAS_BASE_URL", ""), "model": os.environ.get("LLMAAS_MODEL", ""),']
        if "embeddings" in services:
            l += ['               "embeddings_model": os.environ.get("LLMAAS_EMBEDDINGS_MODEL", ""),']
        l += ['               "cle_fichier": os.environ.get("LLMAAS_API_KEY_FILE", ""),',
              '               "en_tetes": {}},  # en-têtes propres à votre passerelle, si elle en exige']
    if "sttaas" in services:
        l += ['    "sttaas": {"base_url": os.environ.get("STTAAS_BASE_URL", ""), "model": os.environ.get("STTAAS_MODEL", ""),',
              '               "cle_fichier": os.environ.get("STTAAS_API_KEY_FILE", ""),',
              '               "en_tetes": {}},']
    return "\n".join(l + ["}"])


def env_exemple(plan: dict) -> str:
    services = services_du_plan(plan)
    l = ["# Phase banque : variables à définir dans l'environnement du notebook (projet, job ou fichier .env non versionné),",
         "# ou directement dans la cellule CONFIG_BANQUE. Jamais de clé dans le notebook ni dans un dépôt.",
         "BANQUE_DONNEES=/chemin/vers/le_jeu_ou_son_dossier",
         "BANQUE_MODELES_DIR=/chemin/vers/les_modeles      # facultatif : sans accès à Internet (telecharger_modeles.py)"]
    if "llmaas" in services or "embeddings" in services:
        l += ["LLMAAS_BASE_URL=https://llmaas.example.internal/v1",
              "LLMAAS_MODEL=nom-du-modele",
              "LLMAAS_API_KEY_FILE=/chemin/du/secret/monte     # ou LLMAAS_API_KEY"]
        if "embeddings" in services:
            l += ["LLMAAS_EMBEDDINGS_MODEL=nom-du-modele-d-embeddings"]
        l += ['# LLMAAS_EN_TETES={"X-Tenant": "..."}            # facultatif : en-têtes exigés par la passerelle']
    if "sttaas" in services:
        l += ["STTAAS_BASE_URL=https://sttaas.example.internal/v1",
              "STTAAS_MODEL=nom-du-modele",
              "STTAAS_API_KEY_FILE=/chemin/du/secret/monte     # ou STTAAS_API_KEY"]
    l += ["# HTTPS_PROXY=http://proxy.example.internal:8080",
          "# SSL_CERT_FILE=/chemin/autorite-entreprise.pem",
          "# PIP_INDEX_URL=https://miroir-pypi.example.internal/simple",
          "# POC_TIMEOUT_SECONDES=120 et POC_TENTATIVES=3 : délai et tentatives par appel de service"]
    return "\n".join(l) + "\n"


def script_telechargement(plan: dict, faits: dict) -> str:
    """Script livré avec le POC : poids des modèles, aux révisions vérifiées, à copier dans l'environnement de la banque."""
    modeles = modeles_par_taille(plan)
    revisions = {m: (faits.get("modeles", {}).get(m) or {}).get("sha") for m in modeles["sandbox"] + modeles["complete"]}
    return f'''"""Télécharge les poids des modèles du POC, aux révisions vérifiées lors de sa validation, dans un dossier à
copier dans l'environnement de la banque (BANQUE_MODELES_DIR), pour l'exécuter sans accès à Internet.

Usage : python telecharger_modeles.py DOSSIER [--taille sandbox|complete]   (pip install huggingface_hub)
"""
import argparse
from pathlib import Path

MODELES = {modeles!r}
REVISIONS = {revisions!r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dossier")
    ap.add_argument("--taille", choices=sorted(MODELES), default="complete")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    for m in MODELES[a.taille]:
        cible = Path(a.dossier) / m.replace("/", "--")
        snapshot_download(m, revision=REVISIONS.get(m), local_dir=cible)
        print(f"{{m}} -> {{cible}}")


if __name__ == "__main__":
    main()
'''


def entete(fiche: dict, plan: dict, faits: dict, jour: str) -> str:
    sources = "\n".join(f"- [{s['titre']}]({s['url']})" for s in fiche.get("sources", [])[:6])
    services = services_du_plan(plan)
    variables = ["| `BANQUE_DONNEES` | jeu bancaire : fichier (csv, tsv, parquet, jsonl, json, xlsx) ou dossier de fichiers ; `factice` pour tester le code |",
                 "| `BANQUE_MODELES_DIR` | (facultatif) dossier des poids copiés localement (`org--nom/`), sans accès à Internet |"]
    if "llmaas" in services or "embeddings" in services:
        variables += ["| `LLMAAS_BASE_URL`, `LLMAAS_MODEL`, `LLMAAS_API_KEY` ou `LLMAAS_API_KEY_FILE` | service LLM à la convention OpenAI (clé par variable ou fichier de secret monté) |"]
    if "embeddings" in services:
        variables += ["| `LLMAAS_EMBEDDINGS_MODEL` | modèle d'embeddings du service |"]
    if "sttaas" in services:
        variables += ["| `STTAAS_BASE_URL`, `STTAAS_MODEL`, `STTAAS_API_KEY` ou `STTAAS_API_KEY_FILE` | service de transcription à la convention OpenAI |"]
    if services:
        variables += ["| `<SERVICE>_EN_TETES` | (facultatif) en-têtes HTTP exigés par la passerelle, objet JSON |"]
    modeles = modeles_par_taille(plan)
    colonnes = "\n".join(f"| `{c['colonne']}` | {c['type']} | {c['description']} |" for c in plan["phase2"]["donnees"])
    licences = "; ".join(f"{m} : {f.get('licence', 'inconnue')}" for m, f in faits.get("modeles", {}).items() if f)
    return f"""# POC — {plan['titre']}

*Preuve de concept du {date_courte(jour)} · sujet : « {fiche['titre']} » · niveau sandbox : {plan['niveau_sandbox']}*

**Hypothèse.** {plan['hypothese']}

| | Approche | Mise en œuvre |
|---|---|---|
| Référence | {plan['reference']['nom']} — {plan['reference']['description']} | {plan['reference']['implementation']} |
| Nouvelle | {plan['nouvelle']['nom']} — {plan['nouvelle']['description']} | {plan['nouvelle']['implementation']} |

**Données publiques.** `{plan['jeu_public']['hub_id']}` ({plan['jeu_public']['config'] or 'configuration par défaut'}) — {plan['jeu_public']['representativite']} Écarts avec les données de la banque : {plan['jeu_public']['ecarts_banque']}

**Métriques.** {', '.join(plan['metriques'])} (principale : {plan['metrique_principale']}, {'à maximiser' if plan['sens'] == 'max' else 'à minimiser'}).

**Sources.**
{sources}

## Mode d'emploi
1. **Phase publique** (`PHASE = "publique"`) : rejoue la comparaison sur le jeu public. `TAILLE = "sandbox"` reprend l'échelle réduite validée par l'atelier sur CPU ; `TAILLE = "complete"` utilise tout le jeu et les modèles de l'échelle complète (GPU nécessaire dès qu'un modèle est entraîné avec PyTorch ou TensorFlow).
2. **Phase banque** (`PHASE = "banque"`) : même protocole sur un jeu de données bancaire, dans l'environnement de la banque. Validée dans le sandbox de l'atelier sur un jeu factice écrit comme un vrai export (dossier de fichiers, CSV « ; » en Windows-1252, colonnes à faire correspondre), contre un faux service à la convention OpenAI (clé lue dans un fichier de secret), modèles servis hors ligne. Pour l'adapter :
   1. **Modèles** : sans accès à Internet, `python telecharger_modeles.py <dossier> --taille complete` sur un poste connecté (révisions vérifiées : {', '.join(f'`{m}`' for m in modeles['complete']) or 'aucun modèle local'}), puis copier le dossier et le désigner par `BANQUE_MODELES_DIR` ;
   2. **Variables** : renseigner l'environnement du notebook à partir de `banque.env.exemple` (ou directement la cellule `CONFIG_BANQUE`, juste après les paramètres). **Jamais de clé dans le notebook** : `<SERVICE>_API_KEY`, un fichier de secret monté (`<SERVICE>_API_KEY_FILE`) ou la saisie masquée ;
   3. **Colonnes** : dans `CONFIG_BANQUE["colonnes"]`, faire correspondre chaque colonne attendue (ci-dessous) au nom de la colonne de votre jeu ;
   4. **Lancer** avec `PHASE = "banque"` (et `TAILLE` voulue) : la cellule d'initialisation affiche une liste de contrôle (jeu, colonnes, modèles, services) et s'arrête sur ce qui manque, avant tout calcul.

| Variable | Rôle |
|---|---|
{chr(10).join(variables)}

Jeu bancaire attendu ({plan['phase2']['volume_conseille']}) :

| Colonne | Type | Description |
|---|---|---|
{colonnes}

Métriques métier : {plan['phase2']['metriques_metier']}

Garde-fous : {plan['phase2']['garde_fous']} En phase banque, le notebook refuse les services publics (OpenAI, Anthropic, Mistral, Hugging Face…) et le HTTP non chiffré, n'affiche aucune donnée brute et désactive la télémétrie ; avec `BANQUE_MODELES_DIR`, il travaille sans accès à Internet.

3. **Dépendances** : `pip install -r requirements.txt` (versions exactes du sandbox), ou `POC_INSTALLER=1` pour installer depuis le notebook. Proxy et certificats d'entreprise : `HTTPS_PROXY`, `SSL_CERT_FILE`, `PIP_INDEX_URL`.

Licences des modèles : {licences or 'voir les cartes des modèles'}. Les résultats (`resultats_<phase>.json`) consignent versions, machine, graine et empreinte des données, pour la traçabilité.
"""


def assembler(fiche: dict, plan: dict, faits: dict, code: dict, jour: str):
    nb = nbformat.v4.new_notebook()
    nb.metadata.update({"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                        "language_info": {"name": "python"}, "poc": {"sujet_id": fiche["sujet_id"], "jour": jour}})
    deps = sorted(set(code["dependances"]))
    cellules = [
        _nb_md(entete(fiche, plan, faits, jour), section="entete"),
        _nb_code('PHASE = "publique"     # "publique" : jeu public ; "banque" : jeu bancaire\n'
                 'TAILLE = "sandbox"     # "sandbox" : échelle réduite (CPU) ; "complete" : tout le jeu, modèles complets\n'
                 'GRAINE = 42\nPOC_SORTIE = "."      # dossier des résultats', tags=["parameters"], section="parametres"),
        _nb_code(cellule_config_banque(plan), section="config_banque"),
        _nb_code("# Dépendances (en plus de requirements.txt). Installées par le notebook seulement si POC_INSTALLER=1.\n"
                 f"DEPENDANCES = {deps!r}\nimport os, subprocess, sys\n"
                 'if DEPENDANCES and os.environ.get("POC_INSTALLER") == "1":\n'
                 '    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *DEPENDANCES])', section="installation"),
        _nb_code((OUTILS / "aide.py").read_text(encoding="utf-8"), section="aide", jupyter={"source_hidden": True}),
        _nb_code("appliquer_config_banque(CONFIG_BANQUE)  # phase banque : liste de contrôle, arrêt si quelque chose manque\n"
                 "preparer_environnement()\nfixer_graine(GRAINE)\nafficher_configuration()", section="initialisation"),
    ]
    for c in code["cellules"]:
        cellules.append((_nb_md if c["type"] == "markdown" else _nb_code)(c["source"], section=c["section"]))
    nb.cells = cellules
    return nb


def controler_cellules(code: dict, faits: dict, plan: dict | None = None) -> list[str]:
    """Contrôle statique avant exécution : syntaxe, sections, secrets, installation, code distant non épinglé, modèles
    du plan (vérifiés) bien utilisés, substituts du sandbox compris."""
    erreurs = []
    codes = [c for c in code["cellules"] if c["type"] == "code"]
    presentes = {c["section"] for c in codes}
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
            erreurs.append(f"cellule {i} ({c['section']}) : écrit une variable LLMAAS_/STTAAS_/BANQUE_ (elles viennent de l'environnement)")
        if _INSTALL.search(c["source"]):
            erreurs.append(f"cellule {i} ({c['section']}) : installation de paquets dans le code (utilise `dependances`)")
        if re.search(r"def\s+(enregistrer_resultats|llm_chat|donnees_banque|ic_bootstrap|wer)\s*\(", c["source"]):
            erreurs.append(f"cellule {i} ({c['section']}) : redéfinit une fonction de l'aide commune")
        if re.search(r"CONFIG_BANQUE\s*(\[[^\]]*\]\s*)*=(?!=)|CONFIG_BANQUE\.(update|setdefault|pop)\(", c["source"]):
            erreurs.append(f"cellule {i} ({c['section']}) : modifie CONFIG_BANQUE (cellule de configuration générée, laissée à l'utilisateur)")
        if re.search(r"BANQUE_DONNEES|BANQUE_MODELES_DIR", c["source"]):
            erreurs.append(f"cellule {i} ({c['section']}) : lit la configuration bancaire directement (passe par donnees_banque et chemin_modele)")
        if re.search(r"google\.colab|drive\.mount|authenticate_user|google\.auth", c["source"]):
            erreurs.append(f"cellule {i} ({c['section']}) : accès au compte Google interdit (google.colab, Drive, authentification)")
        if "trust_remote_code=True" in c["source"].replace(" ", "") and "revision=" not in c["source"]:
            erreurs.append(f"cellule {i} ({c['section']}) : trust_remote_code=True sans revision épinglée")
    if "enregistrer_resultats(" not in tout:
        erreurs.append("aucun appel à enregistrer_resultats")
    if "PHASE" not in tout:
        erreurs.append("le code ne distingue pas les phases (variable PHASE)")
    if "donnees_banque(" not in tout:
        erreurs.append("la phase banque ne charge pas ses données par donnees_banque(...)")
    for role in ("reference", "nouvelle") if plan else ():
        substituts = plan[role].get("modeles_hf_sandbox") or []
        erreurs += [f"modèle {m} du plan ({ROLES[role]}) absent du code" for m in [*plan[role]["modeles_hf"], *substituts] if m not in tout]
        if substituts and "selon_taille(" not in tout:
            erreurs.append(f"substituts du sandbox ({', '.join(substituts)}) non choisis par selon_taille(sandbox=..., complete=...)")
    return erreurs


class Sandbox:
    """Exécute un notebook dans un conteneur Docker sans secret (ou localement en développement, sur demande explicite)."""

    def __init__(self, conf: dict):
        self.image = os.environ.get("VEILLE_POC_IMAGE", conf.get("image", "veille-poc:latest"))
        self.mode = None
        if shutil.which("docker") and subprocess.run(["docker", "image", "inspect", self.image], capture_output=True).returncode == 0:
            self.mode = "docker"
        elif os.environ.get("VEILLE_POC_SANS_ISOLATION") == "1":
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
                f"présents. Le notebook y est exécuté en TAILLE « sandbox » (échelle réduite), de la première à la dernière "
                f"cellule, sans intervention : phase publique en {t['publique'] // 60} min au plus, téléchargements compris "
                f"({t['cellule'] // 60} min par cellule), puis phase banque (données factices) en {t['banque'] // 60} min au plus.\n"
                f"Échelle complète (TAILLE « complete », phase publique seulement), exécutée une fois le code validé, sur le "
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
        if phase == "banque" and modeles:  # servis hors ligne, comme dans un environnement bancaire sans Internet
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
            env = {k: v for k, v in os.environ.items() if not re.search(r"TOKEN|KEY|SECRET|PASSWORD|AUTH|LLMAAS|STTAAS|BANQUE", k, re.I)}
            nom, cmd = None, [sys.executable, str(OUTILS / "executer.py"), *args]
        else:
            raise RuntimeError("aucun sandbox : Docker et l'image du sandbox sont requis (VEILLE_POC_SANS_ISOLATION=1 en développement)")
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
        return ["la phase publique n'a pas enregistré de résultats (enregistrer_resultats)"]
    if r["jeu"]["n_exemples"] < min_exemples:
        erreurs.append(f"évaluation sur {r['jeu']['n_exemples']} exemples seulement (au moins {min_exemples} en sandbox)")
    if not meme_metrique(r["metrique_principale"], plan["metrique_principale"]):
        erreurs.append(f"métrique principale {r['metrique_principale']!r} au lieu de {plan['metrique_principale']!r} (plan)")
    if ban is None:
        return erreurs
    if not ban.get("ok"):
        return erreurs + ["la phase banque (données factices, faux service) a échoué"]
    if not ban.get("resultats"):
        erreurs.append("la phase banque n'a pas enregistré de résultats")
    appels = ban.get("appels_service") or {}
    for role in ("reference", "nouvelle"):
        service = plan[role]["service_phase2"]
        if service != "aucun" and not any(rt.endswith(ROUTES_SERVICE[service]) for rt in appels.get("routes", [])):
            erreurs.append(f"phase banque : l'approche {role} devait appeler le service {service} ({ROUTES_SERVICE[service]}), aucun appel reçu")
    if appels.get("refus"):
        erreurs.append(f"phase banque : {appels['refus']} appel(s) refusé(s) par le faux service (authentification, format)")
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
    if not meme_metrique(r["metrique_principale"], plan["metrique_principale"]):
        erreurs.append(f"métrique principale {r['metrique_principale']!r} au lieu de {plan['metrique_principale']!r}")
    cles = lambda res: sorted((a["role"], tuple(sorted(a["metriques"]))) for a in res["approches"])
    if cles(r) != cles(ref):
        erreurs.append("approches ou métriques différentes de celles de l'échelle réduite")
    if r["jeu"]["n_exemples"] < ref["jeu"]["n_exemples"]:
        erreurs.append(f"{r['jeu']['n_exemples']} exemples d'évaluation, moins qu'à l'échelle réduite ({ref['jeu']['n_exemples']})")
    return erreurs


def nombre(v, chiffres: int = 3) -> str:
    """Nombre à la française (virgule décimale)."""
    return f"{v:.{chiffres}g}".replace(".", ",") if isinstance(v, (int, float)) and not isinstance(v, bool) else "—"


def _valeur(a: dict, m: str) -> str:
    ic = (a.get("ic95") or {}).get(m)
    return nombre(a["metriques"].get(m), 4) + (f" [{nombre(ic[0])} ; {nombre(ic[1])}]" if ic else "")


ROLES = {"reference": "référence", "nouvelle": "nouvelle"}


def bilan_markdown(plan: dict, etat: dict, res: dict, revue: dict) -> str:
    mets = list(res["approches"][0]["metriques"])
    tete = "| Approche | Rôle | " + " | ".join(mets) + " | Latence p50 (ms) |"
    lignes = [tete, "|" + "---|" * (len(mets) + 3)]
    for a in res["approches"]:
        lignes.append(f"| {a['nom']} | {ROLES.get(a['role'], a['role'])} | " + " | ".join(_valeur(a, m) for m in mets)
                      + f" | {nombre((a.get('latence_ms') or {}).get('p50'))} |")
    pb = [x for x in revue["problemes"] if x["gravite"] == "majeur"]
    mineurs = [x for x in revue["problemes"] if x["gravite"] == "mineur"]
    code = etat.get("code") or {}
    services = sorted({plan[r]["service_phase2"] for r in ("reference", "nouvelle")} - {"aucun"})
    banque = ("Phase banque exécutée sur données factices, contre un faux service à la convention OpenAI "
              f"({', '.join(services)}) : le code de la phase 2 est validé, pas ses résultats." if services else
              "Phase banque exécutée sur données factices : le code de la phase 2 est validé, pas ses résultats.")
    complet, reduit = etat.get("complete") or {}, etat["publique"]["resultats"]
    jeu = f"{res['jeu']['n_exemples']} exemples d'évaluation de `{res['jeu']['nom']}` ({res['jeu'].get('split')})"
    if etat.get("resultats_complets"):
        mp = reduit["metrique_principale"]
        rappel = " ; ".join(f"{a['nom']} {nombre(a['metriques'].get(mp))}" for a in reduit["approches"])
        intro = (f"*Échelle complète exécutée sur {complet.get('materiel')} ({complet.get('raison_materiel')}), graine "
                 f"{res.get('graine')} : {jeu}. Le code a d'abord été validé de bout en bout dans le sandbox de l'atelier (CPU, "
                 f"échelle réduite : {reduit['jeu']['n_exemples']} exemples, {mp} : {rappel}). {banque} "
                 f"{len(etat['iterations'])} itération(s) d'écriture et de correction.*")
    else:
        statut = {"echec": "en échec", "non_execute": "non exécutée"}.get(complet.get("statut"), "non exécutée")
        intro = (f"*Exécuté de bout en bout dans le sandbox de l'atelier (CPU, échelle réduite, graine {res.get('graine')}) : {jeu}. "
                 f"{banque} {len(etat['iterations'])} itération(s) d'écriture et de correction.*\n\n"
                 f"**Échelle complète {statut}** : {complet.get('raison') or '—'}. Les chiffres ci-dessous sont ceux de "
                 f"l'échelle réduite : indicatifs, à confirmer en `TAILLE = \"complete\"`.")
    textes = ["## Bilan", "", intro, "",
              "\n".join(lignes), "", f"**Conclusion de la revue de méthode** (confiance {revue['confiance']}) : {revue['conclusion']}", ""]
    if pb:
        textes += ["**Réserves majeures :**"] + [f"- {x['description']}" for x in pb] + [""]
    if code.get("ecarts_sandbox") or plan.get("adaptation_sandbox"):
        textes += [f"**Échelle réduite du sandbox (écarts avec l'échelle complète) :** {code.get('ecarts_sandbox') or plan['adaptation_sandbox']}", ""]
    if code.get("limites"):
        textes += ["**Limites :**"] + [f"- {x}" for x in code["limites"]] + [""]
    if mineurs:
        textes += ["**Améliorations possibles :**"] + [f"- {x['description']}" for x in mineurs[:5]] + [""]
    textes += ["Pour conclure pour la banque : rejouer en `PHASE = \"banque\"`, `TAILLE = \"complete\"` (voir le mode d'emploi en tête)."]
    return "\n".join(textes)


def resume_poc(plan: dict, etat: dict, res: dict, revue: dict) -> dict:
    mp = res["metrique_principale"]
    vals = {a["role"]: a["metriques"][mp] for a in res["approches"]}
    complet = etat.get("complete") or {}
    echelle = "complete" if etat.get("resultats_complets") else "reduite"
    ou = complet.get("materiel") if echelle == "complete" else "échelle réduite, CPU du sandbox"
    synthese = (f"{mp} : {plan['nouvelle']['nom']} {nombre(vals.get('nouvelle'))} contre "
                f"{nombre(vals.get('reference'))} pour {plan['reference']['nom']} ({res['jeu']['n_exemples']} exemples, {ou}).")
    return {"sujet_id": etat["sujet_id"], "titre": plan["titre"], "statut": "valide", "niveau_sandbox": plan["niveau_sandbox"],
            "echelle": echelle, "materiel": complet.get("materiel") if echelle == "complete" else "CPU du sandbox",
            "echelle_complete": {c: complet.get(c) for c in ("statut", "materiel_requis", "raison_materiel", "raison")},
            "iterations": len(etat["iterations"]), "metrique_principale": mp, "sens": res["sens"],
            "approches": [{"nom": a["nom"], "role": a["role"], "valeur": a["metriques"][mp], "ic95": (a.get("ic95") or {}).get(mp)} for a in res["approches"]],
            "n_exemples": res["jeu"]["n_exemples"], "jeu": res["jeu"]["nom"], "synthese": synthese,
            "conclusion": revue["conclusion"], "confiance": revue["confiance"], "reproduit_la_source": revue["reproduit_la_source"],
            "reserves": [x["description"] for x in revue["problemes"] if x["gravite"] == "majeur"]}


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

    def note(self, msg: str):
        self.r.note(msg)

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

    def revoir(self, code: dict, pub: dict, ban: dict | None, reduit: dict | None = None, echelle: str = "complete",
               raison: str = "") -> dict:
        """`echelle` : « complete » (pub est l'échelle complète), « prealable » (échelle réduite, l'échelle complète suivra
        si rien n'est bloquant) ou « reduite » (le POC sera publié à l'échelle réduite, pour `raison`)."""
        publique = compte_rendu(pub) + "\nRésultats enregistrés :\n" + json.dumps(pub.get("resultats"), ensure_ascii=False, indent=1)[:6000]
        if reduit:  # pub est l'échelle complète : l'échelle réduite du sandbox en regard
            r = reduit["resultats"]
            publique += (f"\n\nPour mémoire, échelle réduite validée dans le sandbox (CPU, {r['jeu']['n_exemples']} exemples) :\n"
                         + json.dumps({a["nom"]: a["metriques"] for a in r["approches"]}, ensure_ascii=False))
        prompt = _prompt("poc_revue", fiche=self._json(self.fiche), plan=self._json(self.plan), echelle=ECHELLES_REVUE[echelle].format(raison=raison),
                         cellules=self._json(code["cellules"]), publique=publique, banque=compte_rendu(ban))
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
            rapport.setdefault("materiel", f"CPU du sandbox ({self.sandbox.cpus} cœurs)")
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
            statiques = controler_cellules(code, faits, plan)
            nb = assembler(self.fiche, plan, faits, code, self.jour)
            nbformat.write(nb, travail / "poc.ipynb")
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
                erreurs = valider_execution(pub, None, plan, self.conf["min_exemples_evaluation"])
                if not erreurs:
                    ban = self.sandbox.executer(travail, "banque", t["banque"], t["cellule"], modeles=modeles_par_taille(plan)["sandbox"])
                    it["banque"] = {"ok": ban.get("ok"), "secondes": ban.get("secondes"), "erreur": (ban.get("erreur") or {}).get("nature"),
                                    "appels": ban.get("appels_service")}
                    self.note(f"[poc] {sid} itération {k} : phase banque {'réussie' if ban.get('ok') else 'en échec'} ({ban.get('secondes')} s)")
                    erreurs = valider_execution(pub, ban, plan, self.conf["min_exemples_evaluation"])
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
                        etat.update(statut="valide", code=code, publique=pub, banque=ban, revue=revue,
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
            if self.temps_restant() < self.conf["timeouts"]["publique"] + self.conf["timeouts"]["banque"] + 600:
                etat["dernier_compte_rendu"] = "budget de temps épuisé. " + etat["dernier_compte_rendu"]
                break
            code = self.r.etape(f"code_{k + 1}", lambda: self.reparer(code, motif, compte))
        etat["code_final"] = code
        write_json(travail / "etat.json", {k: v for k, v in etat.items() if k not in ("code", "code_final")})
        return etat

    # ---------------------------------------------------------------- livraison

    def livrer(self, etat: dict) -> dict:
        """Dossier livraison/ : ce qui sera publié tel quel dans le dépôt du POC."""
        dest = self.dir / "livraison"
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        complet = etat.get("resultats_complets")
        source = Path(etat["complete"]["notebook"]) if complet else self.dir / "execute_publique.ipynb"
        nb = nbformat.read(source, as_version=4)
        nb.cells = [c for c in nb.cells if "injected-parameters" not in c.get("metadata", {}).get("tags", [])]
        res, revue = complet or etat["publique"]["resultats"], etat["revue"]
        bilan = bilan_markdown(self.plan, etat, res, revue)
        nb.cells.append(_nb_md(bilan, section="bilan"))
        for c in nb.cells:  # sorties volumineuses retirées (dépôt git)
            for o in c.get("outputs", []):
                for k in [k for k in (o.get("data") or {}) if k.startswith("image/") and len(str(o["data"][k])) > 400_000]:
                    del o["data"][k]
                if o.get("output_type") == "stream" and len(o.get("text", "")) > 20_000:
                    o["text"] = "[…]\n" + o["text"][-20_000:]
        with open(dest / "poc.ipynb", "w", encoding="utf-8", newline="\n") as f:
            nbformat.write(nb, f)
        write_json(dest / "resultats.json", res)
        if complet:
            write_json(dest / "resultats_sandbox.json", etat["publique"]["resultats"])
        gel = self.dir / "requirements_sandbox.txt"
        (dest / "requirements.txt").write_text(gel.read_text(encoding="utf-8") if gel.exists() else "", encoding="utf-8", newline="\n")
        (dest / "banque.env.exemple").write_text(env_exemple(self.plan), encoding="utf-8", newline="\n")
        (dest / "telecharger_modeles.py").write_text(script_telechargement(self.plan, self.faits), encoding="utf-8", newline="\n")
        fichiers = ("Notebook : `poc.ipynb` · résultats : `resultats.json`"
                    + (" (échelle complète) · échelle réduite du sandbox : `resultats_sandbox.json`" if complet else "")
                    + " · dépendances exactes du sandbox : `requirements.txt` · environnement bancaire : `banque.env.exemple`, "
                    "`telecharger_modeles.py` (poids à copier pour travailler sans Internet).")
        (dest / "README.md").write_text(
            entete(self.fiche, self.plan, self.faits, self.jour) + "\n" + bilan + f"\n\n---\n\n{fichiers}\n\n"
            "*Notebook écrit par un modèle de langage, exécuté de bout en bout dans un sandbox (puis à l'échelle complète), "
            "relu pour sa méthode par un second passage. À relire avant toute décision.*\n", encoding="utf-8", newline="\n")
        return resume_poc(self.plan, etat, res, revue)

    def executer(self, sans_execution: bool = False) -> dict:
        if not sans_execution:
            if self.sandbox.mode is None:
                raise SystemExit("aucun sandbox disponible : construire l'image (docker build -t veille-poc tools/poc) ; "
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
                resultat = {**base, "statut": etat["statut"], "iterations": len(etat["iterations"]), "raison": raison_echec(etat)}
        except llm.QuotaError as e:
            self.note(f"[poc] quota du modèle atteint : {e}")
            resultat = {**base, "statut": "non_traite", "raison": "quota du modèle atteint : à relancer"}
        except llm.LLMError as e:  # modèle indisponible, authentification, réponse illisible après les tentatives
            self.note(f"[poc] modèle en échec : {e}")
            resultat = {**base, "statut": "echec", "raison": f"modèle en échec : {str(e)[:200]}"}
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
        "name": res["depot"], "description": f"Preuve de concept : {res['titre']}"[:340], "private": False,
        "has_issues": False, "has_projects": False, "has_wiki": False})
    if r.status_code != 201 and not (r.status_code == 422 and "already exists" in r.text):
        raise RuntimeError(f"création du dépôt {res['depot']} impossible : HTTP {r.status_code} {r.text[:200]}")
    with tempfile.TemporaryDirectory() as tmp:
        depot = Path(tmp)
        shutil.copytree(travail / "livraison", depot, dirs_exist_ok=True)
        _git("init", "-q", "-b", "main", cwd=depot)
        _git("add", "-A", cwd=depot)
        _git("-c", "user.name=atelier-poc", "-c", f"user.email={proprietaire}@users.noreply.github.com",
             "commit", "-q", "-m", f"Preuve de concept du {date_courte(res['jour'])} : {res['titre']}", cwd=depot)
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


def lister(racine: Path) -> list[str]:
    """Commandes à exécuter : sans résultat définitif dans l'index (valide ou échec)."""
    faits = {(p["jour"], p["sujet_id"]) for p in (read_json(racine / "index.json", None) or {}).get("pocs", [])
             if p.get("statut") in ("valide", "echec")}
    attente = []
    for c in sorted(racine.glob("commandes/*/*.json")):
        d = read_json(c, {})
        if (d.get("jour"), d.get("sujet_id")) not in faits:
            attente.append(c.relative_to(racine).as_posix())
    return attente


def _travail(commande: dict) -> Path:
    return ROOT / "travail" / commande["jour"] / commande["sujet_id"]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(prog="python -m veille.atelier")
    sous = ap.add_subparsers(dest="action", required=True)
    sous.add_parser("lister", help="commandes en attente (JSON)")
    for nom in ("executer", "publier"):
        p = sous.add_parser(nom)
        p.add_argument("commande", help="chemin de la commande (commandes/<jour>/<sujet>.json)")
    a = ap.parse_args(argv)
    if a.action == "lister":
        print(json.dumps(lister(ROOT)))
        return 0
    commande = read_json(ROOT / a.commande, None)
    if not commande:
        raise SystemExit(f"commande illisible : {a.commande}")
    if a.action == "executer":
        res = Atelier(commande, _travail(commande)).executer()
        log.info("POC %s : %s", commande["sujet_id"], res["statut"] + (f" ({res.get('raison')})" if res.get("raison") else ""))
        return 0
    jeton = os.environ.get("POC_PUBLICATION_TOKEN")
    res = read_json(_travail(commande) / "resultat.json", None) or {
        "jour": commande["jour"], "sujet_id": commande["sujet_id"], "depot": commande["depot"], "titre": commande["plan"]["titre"],
        "statut": "non_traite", "raison": "exécution interrompue avant son terme (délai du job ou panne) : à relancer"}
    if res.get("statut") == "valide":
        if not jeton:
            raise SystemExit("POC_PUBLICATION_TOKEN absent : publication impossible")
        res = publier(_travail(commande), jeton)
    maj_index(ROOT, res)
    log.info("POC %s : %s%s", commande["sujet_id"], res["statut"], f" → {res['url']}" if res.get("url") else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
