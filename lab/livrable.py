"""Livrable d'une expérience : cadre du notebook, README, figures et contrôle de neutralité.

Le dépôt publié ne dit rien de son origine : pas de date de commande, pas de mention d'un rédacteur automatique ni
d'une relecture automatique, aucun secteur nommé (« services financiers » seulement). Il se lit seul : le README
explique la question, le protocole, les résultats illustrés et la façon de rejouer l'expérience, sur le jeu public
puis sur des données internes ; le notebook suit la même structure numérotée.

Noms du livrable (anglais) : experiment.ipynb, helpers.py, download_models.py, private.env.example, requirements.txt,
results/, figures/. Variables du notebook : PHASE (« public » | « private »), SCALE (« reduced » | « full »), SEED,
OUTPUT_DIR, PRIVATE_CONFIG.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import nbformat

OUTILS = Path(__file__).resolve().parent.parent / "tools" / "poc"
NOTEBOOK = "experiment.ipynb"

# Termes qui ne doivent jamais apparaître dans un dépôt publié (ils le rattacheraient à son commanditaire ou à son
# origine). La liste reste privée : l'atelier n'en reçoit que les empreintes (config/atelier.yaml, empreintes_interdites),
# comparées à celles des mots et groupes de mots des textes. Les noms de jeux (« banking77_fr ») ne sont pas des mots.
EMPREINTES: set[str] = set()


def configurer_empreintes(empreintes) -> None:
    EMPREINTES.clear()
    EMPREINTES.update(empreintes or [])


def _mots(texte: str) -> list[str]:
    t = unicodedata.normalize("NFKD", texte or "").encode("ascii", "ignore").decode().lower()
    return re.findall(r"[a-z0-9_]+", t)


def empreinte_terme(terme: str) -> str:
    return hashlib.sha256(" ".join(_mots(terme)).encode()).hexdigest()[:16]


def termes_interdits(texte: str) -> list[str]:
    """Mots ou groupes de trois mots au plus du texte dont l'empreinte est interdite."""
    if not EMPREINTES:
        return []
    mots, trouves = _mots(texte), set()
    for n in (1, 2, 3):
        for i in range(len(mots) - n + 1):
            groupe = " ".join(mots[i:i + n])
            if hashlib.sha256(groupe.encode()).hexdigest()[:16] in EMPREINTES:
                trouves.add(groupe)
    return sorted(trouves)


_FIN_DE_PHRASE = re.compile(r"(?<=[.!?…])\s+")


def neutre(texte: str) -> str:
    """Texte repris dans le livrable (revue, limites, plan) : phrases contenant un terme interdit retirées."""
    if not isinstance(texte, str) or not termes_interdits(texte):
        return texte
    return " ".join(p for p in _FIN_DE_PHRASE.split(texte) if not termes_interdits(p))


def verifier_neutralite(dossier: Path) -> list[str]:
    """Termes interdits dans les fichiers du livrable (vide : publiable). Sorties des cellules exclues : elles peuvent
    citer le contenu des jeux publics."""
    trouves = []
    for f in sorted(dossier.rglob("*")):
        if f.is_file() and f.suffix in (".md", ".ipynb", ".py", ".json", ".txt", ".example", ".yaml"):
            texte = f.read_text(encoding="utf-8", errors="ignore")
            if f.suffix == ".ipynb":
                nb = json.loads(texte)
                texte = "\n".join("".join(c.get("source", "")) for c in nb["cells"]) + json.dumps(nb.get("metadata", {}))
            trouves += [f"{f.relative_to(dossier).as_posix()} : « {t} »" for t in termes_interdits(texte)]
    return trouves


# --------------------------------------------------------------------------- formats

def nombre(v, chiffres: int = 3) -> str:
    """Nombre à la française (virgule décimale)."""
    return f"{v:.{chiffres}g}".replace(".", ",") if isinstance(v, (int, float)) and not isinstance(v, bool) else "—"


def _cellule(texte) -> str:
    """Texte sûr dans une cellule de tableau Markdown."""
    return neutre(str(texte or "—")).replace("|", "\\|").replace("\n", " ")


def _duree(secondes) -> str:
    if not isinstance(secondes, (int, float)):
        return "—"
    return f"{max(1, round(secondes / 60))} min" if secondes >= 60 else f"{round(secondes)} s"


ROLES = {"reference": "Référence", "new": "Nouvelle approche"}


def metriques_lisibles(code: dict, res: dict) -> dict[str, dict]:
    """Par clé de métrique : libellé, explication, sens (« max » / « min » / « »), d'après le code et les résultats."""
    out = {}
    for m in (code or {}).get("lecture_des_metriques") or []:
        out[m["cle"]] = {"libelle": m["libelle"], "explication": m["explication"], "sens": m.get("sens", "")}
    for cle, libelle in (res.get("metric_labels") or {}).items():
        out.setdefault(cle, {"libelle": libelle, "explication": "", "sens": ""})
    for a in res.get("approaches", []):
        for cle in a["metrics"]:
            out.setdefault(cle, {"libelle": cle, "explication": "", "sens": ""})
    p = res.get("primary_metric")
    if p in out and not out[p]["sens"]:
        out[p]["sens"] = res.get("direction", "")
    return out


# --------------------------------------------------------------------------- configuration des données internes

def services_du_plan(plan: dict) -> list[str]:
    return sorted({plan[r]["service_phase2"] for r in ("reference", "nouvelle")} - {"aucun"})


def modeles_par_taille(plan: dict) -> dict[str, list[str]]:
    """Modèles chargés à chaque échelle (noms du notebook) : substituts de l'échelle réduite (sinon les mêmes), puis ceux
    de l'échelle complète."""
    reduced, full = [], []
    for r in ("reference", "nouvelle"):
        full += plan[r]["modeles_hf"]
        reduced += plan[r].get("modeles_hf_sandbox") or plan[r]["modeles_hf"]
    return {"reduced": sorted(set(reduced)), "full": sorted(set(full))}


def colonnes_chemins(plan: dict) -> list[str]:
    return [c["colonne"] for c in plan["phase2"]["donnees"]
            if re.search(r"chemin|path|fichier|file|audio|wav|image", f"{c['type']} {c['colonne']}", re.I)]


def cellule_config_privee(plan: dict) -> str:
    """Seule cellule à adapter pour exécuter le notebook sur des données internes. Générée depuis le plan (colonnes
    attendues, colonnes de chemins, modèles, services) ; valeurs lues dans l'environnement ; aucune clé."""
    services = services_du_plan(plan)
    colonnes = {c["colonne"]: c["colonne"] for c in plan["phase2"]["donnees"]}
    l = ["# Configuration pour vos données (PHASE = \"private\") : seule cellule à adapter.",
         "# Les valeurs par défaut viennent de l'environnement (modèle : private.env.example).",
         "# Aucune clé ici : elle vient de <SERVICE>_API_KEY, d'un fichier de secret monté (<SERVICE>_API_KEY_FILE) ou d'une saisie masquée.",
         "import os",
         "",
         "PRIVATE_CONFIG = {",
         "    # Jeu de données : un fichier (csv, tsv, parquet, jsonl, json, xlsx) ou un dossier de fichiers de même structure.",
         '    "data": os.environ.get("PRIVATE_DATA", ""),',
         "    # Colonne attendue par le notebook -> nom de la colonne dans votre jeu (ne modifier que la partie droite).",
         f'    "columns": {colonnes!r},',
         "    # Colonnes qui contiennent des chemins de fichiers (relatifs au fichier de données qui les cite, ou absolus).",
         f'    "path_columns": {colonnes_chemins(plan)!r},',
         '    "max_rows": None,                         # plafond facultatif (échantillon aléatoire reproductible)',
         '    "csv": {"sep": None, "encoding": None},   # None : séparateur et encodage devinés (« ; », Windows-1252…)',
         "    # Poids des modèles copiés localement (dossiers org--nom/, script download_models.py), pour travailler sans Internet.",
         '    "models_dir": os.environ.get("LOCAL_MODELS_DIR", ""),',
         f'    "models": {modeles_par_taille(plan)!r},',
         "    # Autorité de certification d'entreprise (fichier PEM) pour les appels HTTPS internes ; vide sinon.",
         '    "ca_bundle": os.environ.get("SSL_CERT_FILE", ""),']
    if "llmaas" in services or "embeddings" in services:
        l += ['    "llmaas": {"base_url": os.environ.get("LLMAAS_BASE_URL", ""), "model": os.environ.get("LLMAAS_MODEL", ""),']
        if "embeddings" in services:
            l += ['               "embeddings_model": os.environ.get("LLMAAS_EMBEDDINGS_MODEL", ""),']
        l += ['               "api_key_file": os.environ.get("LLMAAS_API_KEY_FILE", ""),',
              '               "headers": {}},  # en-têtes exigés par votre passerelle, s\'il y en a']
    if "sttaas" in services:
        l += ['    "sttaas": {"base_url": os.environ.get("STTAAS_BASE_URL", ""), "model": os.environ.get("STTAAS_MODEL", ""),',
              '               "api_key_file": os.environ.get("STTAAS_API_KEY_FILE", ""),',
              '               "headers": {}},']
    return "\n".join(l + ["}"])


def variables_environnement(plan: dict) -> list[tuple[str, str]]:
    services = services_du_plan(plan)
    v = [("PRIVATE_DATA", "jeu de données : fichier ou dossier (csv, tsv, parquet, jsonl, json, xlsx) ; `fake` pour tester le code sur un jeu factice"),
         ("LOCAL_MODELS_DIR", "(facultatif) dossier des poids copiés localement (`org--nom/`), pour travailler sans Internet")]
    if "llmaas" in services or "embeddings" in services:
        v += [("LLMAAS_BASE_URL, LLMAAS_MODEL", "service LLM interne à la convention OpenAI (URL en `/v1`, nom du modèle)"),
              ("LLMAAS_API_KEY ou LLMAAS_API_KEY_FILE", "clé du service, par variable ou par fichier de secret monté")]
    if "embeddings" in services:
        v += [("LLMAAS_EMBEDDINGS_MODEL", "modèle d'embeddings du service")]
    if "sttaas" in services:
        v += [("STTAAS_BASE_URL, STTAAS_MODEL", "service de transcription interne à la convention OpenAI"),
              ("STTAAS_API_KEY ou STTAAS_API_KEY_FILE", "clé du service de transcription")]
    if services:
        v += [("<SERVICE>_HEADERS", "(facultatif) en-têtes HTTP exigés par la passerelle, objet JSON")]
    v += [("HTTPS_PROXY, SSL_CERT_FILE, PIP_INDEX_URL", "(facultatif) proxy, autorité de certification et miroir PyPI d'entreprise"),
          ("SERVICE_TIMEOUT_S, SERVICE_MAX_RETRIES", "(facultatif) délai (120 s) et tentatives (3) par appel de service")]
    return v


def env_exemple(plan: dict) -> str:
    services = services_du_plan(plan)
    l = ["# Variables pour exécuter experiment.ipynb sur vos données (PHASE = \"private\").",
         "# À définir dans l'environnement du notebook (projet, job, ou fichier .env non versionné), ou directement dans la",
         "# cellule PRIVATE_CONFIG. Jamais de clé dans le notebook ni dans un dépôt.",
         "PRIVATE_DATA=/chemin/vers/le_jeu_ou_son_dossier      # ou « fake » pour tester le code sur un jeu factice",
         "LOCAL_MODELS_DIR=/chemin/vers/les_modeles            # facultatif : sans accès à Internet (download_models.py)"]
    if "llmaas" in services or "embeddings" in services:
        l += ["LLMAAS_BASE_URL=https://llm.example.internal/v1",
              "LLMAAS_MODEL=nom-du-modele",
              "LLMAAS_API_KEY_FILE=/chemin/du/secret/monte          # ou LLMAAS_API_KEY"]
        if "embeddings" in services:
            l += ["LLMAAS_EMBEDDINGS_MODEL=nom-du-modele-d-embeddings"]
        l += ['# LLMAAS_HEADERS={"X-Tenant": "..."}                  # facultatif : en-têtes exigés par la passerelle']
    if "sttaas" in services:
        l += ["STTAAS_BASE_URL=https://stt.example.internal/v1",
              "STTAAS_MODEL=nom-du-modele",
              "STTAAS_API_KEY_FILE=/chemin/du/secret/monte          # ou STTAAS_API_KEY"]
    l += ["# HTTPS_PROXY=http://proxy.example.internal:8080",
          "# SSL_CERT_FILE=/chemin/autorite-entreprise.pem",
          "# PIP_INDEX_URL=https://pypi-mirror.example.internal/simple",
          "# SERVICE_TIMEOUT_S=120 et SERVICE_MAX_RETRIES=3 : délai et tentatives par appel de service"]
    return "\n".join(l) + "\n"


def script_telechargement(plan: dict, faits: dict) -> str:
    """download_models.py : poids des modèles, aux révisions vérifiées, à copier dans l'environnement cible."""
    modeles = modeles_par_taille(plan)
    revisions = {m: (faits.get("modeles", {}).get(m) or {}).get("sha") for m in modeles["reduced"] + modeles["full"]}
    return f'''"""Télécharge les poids des modèles de l'expérience, aux révisions exactes utilisées pour les résultats publiés.

À lancer sur un poste connecté à Internet ; copier ensuite le dossier dans l'environnement cible et le désigner par
LOCAL_MODELS_DIR : le notebook charge alors les modèles sans aucun accès réseau.

Usage :
    pip install huggingface_hub
    python download_models.py DOSSIER [--scale reduced|full]
"""
import argparse
from pathlib import Path

# Modèles chargés à chaque échelle (SCALE dans le notebook).
MODELS = {modeles!r}
# Révision (commit Hugging Face) de chaque modèle : les mêmes poids que pour les résultats publiés.
REVISIONS = {revisions!r}


def main() -> None:
    """Télécharge chaque modèle de l'échelle demandée dans DOSSIER/org--nom."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", help="dossier de destination")
    parser.add_argument("--scale", choices=sorted(MODELS), default="full", help="échelle (défaut : full)")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    for model_id in MODELS[args.scale]:
        target = Path(args.folder) / model_id.replace("/", "--")
        snapshot_download(model_id, revision=REVISIONS.get(model_id), local_dir=target)
        print(f"{{model_id}} -> {{target}}")


if __name__ == "__main__":
    main()
'''


# --------------------------------------------------------------------------- notebook

# Sections du code rédigé, dans l'ordre ; le moteur insère le titre numéroté de chaque groupe.
SECTIONS = {
    "context": "1. La question",
    "public_data": "2. Les données", "private_data": "2. Les données",
    "reference": "3. L'approche de référence",
    "new_approach": "4. La nouvelle approche",
    "evaluation": "5. Évaluation",
    "analysis": "6. Analyse des résultats",
}
SECTIONS_REQUISES = ("public_data", "private_data", "reference", "new_approach", "evaluation", "analysis")
SOMMAIRE = ["0. Mise en route", *dict.fromkeys(SECTIONS.values()), "7. Bilan"]


def _nb_code(source: str, **meta):
    return nbformat.v4.new_code_cell(source, metadata=meta)


def _nb_md(source: str, **meta):
    return nbformat.v4.new_markdown_cell(source, metadata=meta)


def api_helpers() -> dict[str, str]:
    """Fonctions publiques de helpers.py et la première phrase de leur documentation (source unique)."""
    arbre = ast.parse((OUTILS / "helpers.py").read_text(encoding="utf-8"))
    out = {}
    for n in arbre.body:
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_"):
            out[n.name] = " ".join((ast.get_docstring(n) or "").split("\n\n")[0].split())
    return out


def api_aide() -> str:
    """Signatures et documentation des fonctions de helpers.py, pour le modèle qui écrit le notebook."""
    arbre = ast.parse((OUTILS / "helpers.py").read_text(encoding="utf-8"))
    lignes = []
    for n in arbre.body:
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_"):
            doc = " ".join((ast.get_docstring(n) or "").split("\n\n")[0].split())
            lignes.append(f"- `{n.name}({ast.unparse(n.args)})` : {doc}")
    lignes.append("- `SERVICE_CALLS` : liste des appels aux services (durée, modèle, succès), résumée dans les résultats.")
    lignes.append("- `COLORS` : couleurs communes (`reference`, `new`, `accent`, `good`, `bad`) ; `COLORS['new']` pour la nouvelle approche.")
    return "\n".join(lignes)


def fonctions_utilisees(code: dict) -> list[tuple[str, str]]:
    """Fonctions de helpers.py appelées par le code rédigé, avec leur rôle (tableau de la mise en route)."""
    api = api_helpers()
    tout = "\n".join(c["source"] for c in code.get("cellules", []) if c["type"] == "code")
    return [(nom, doc) for nom, doc in api.items() if re.search(rf"\b{nom}\s*\(", tout)]


def _modeles_md(ids: list[str]) -> str:
    return ", ".join(f"[`{m}`](https://huggingface.co/{m})" for m in ids) or "—"


def entete_notebook(plan: dict, en_bref: str = "", complet: bool = False, libelle_metrique: str = "") -> str:
    j = plan["jeu_public"]
    sens = "à maximiser" if plan["sens"] == "max" else "à minimiser"
    sortie = ("Les sorties affichées sont celles de l'échelle complète (`SCALE = \"full\"`)." if complet else
              "Les sorties affichées sont celles de l'échelle réduite (`SCALE = \"reduced\"`).")
    lignes = [f"# {plan['titre']}", ""]
    if en_bref:
        lignes += [f"> **En bref.** {en_bref}", ""]
    lignes += [
        "| | |", "|---|---|",
        f"| **Question** | {_cellule(plan['hypothese'])} |",
        f"| **Référence** | {_cellule(plan['reference']['nom'])} — {_cellule(plan['reference']['description'])} |",
        f"| **Nouvelle approche** | {_cellule(plan['nouvelle']['nom'])} — {_cellule(plan['nouvelle']['description'])} |",
        f"| **Données publiques** | [`{j['hub_id']}`](https://huggingface.co/datasets/{j['hub_id']})"
        f"{' (' + j['config'] + ')' if j.get('config') else ''} |",
        f"| **Métrique principale** | {_cellule(libelle_metrique or plan['metrique_principale'])} ({sens}) |",
        "",
        "**Sommaire** — " + " · ".join(SOMMAIRE),
        "",
        "**Trois façons d'exécuter ce notebook** (cellule de paramètres, section 0) :",
        "",
        "| Réglage | À quoi ça sert |", "|---|---|",
        "| `PHASE = \"public\"`, `SCALE = \"reduced\"` | vérifier que tout fonctionne : quelques minutes sur CPU, sur un échantillon |",
        "| `PHASE = \"public\"`, `SCALE = \"full\"` | reproduire les résultats publiés, sur tout le jeu public |",
        "| `PHASE = \"private\"` | rejouer le même protocole sur vos données, dans votre environnement (voir le README) |",
        "",
        sortie,
    ]
    return "\n".join(lignes)


def _mise_en_route(code: dict) -> str:
    fonctions = fonctions_utilisees(code)
    lignes = ["## 0. Mise en route", "",
              "Trois réglages suffisent : la **phase** (quelles données), l'**échelle** (combien de données et quels modèles) "
              "et la **graine** (mêmes tirages, mêmes résultats). Le même code sert aux deux phases : seule la source des "
              "données change.", "",
              "- `PHASE = \"public\"` charge le jeu public ; `PHASE = \"private\"` lit vos données d'après `PRIVATE_CONFIG` "
              "(cellule suivante) et appelle vos services internes. En phase `private`, aucune donnée brute n'est affichée.",
              "- `SCALE = \"reduced\"` prend un petit échantillon et, si besoin, des modèles plus petits de la même famille : "
              "c'est un test de la mécanique, ses chiffres ne valent pas résultat. `SCALE = \"full\"` utilise tout le jeu.",
              "- Les fonctions communes (chargement des données internes, appels de services, intervalles de confiance, "
              "figures, enregistrement des résultats) sont dans `helpers.py`, à côté du notebook."]
    if fonctions:
        lignes += ["", "Fonctions de `helpers.py` utilisées ici :", "", "| Fonction | Rôle |", "|---|---|"]
        lignes += [f"| `{nom}` | {_cellule(doc)} |" for nom, doc in fonctions]
    return "\n".join(lignes)


def cadre(plan: dict, code: dict) -> list:
    """Cellules placées par le moteur avant le code rédigé : en-tête, mise en route, paramètres, configuration des
    données internes, dépendances, initialisation."""
    deps = sorted(set(code.get("dependances") or []))
    return [
        _nb_md(entete_notebook(plan), section="header"),
        _nb_md(_mise_en_route(code), section="setup"),
        _nb_code('PHASE = "public"        # "public" : jeu public ; "private" : vos données, dans votre environnement\n'
                 'SCALE = "reduced"       # "reduced" : échantillon, quelques minutes sur CPU ; "full" : tout le jeu\n'
                 'SEED = 42               # graine : mêmes tirages, mêmes résultats\n'
                 'OUTPUT_DIR = "outputs"  # résultats (results_<phase>.json) et figures', tags=["parameters"], section="parameters"),
        _nb_md("### Configuration pour vos données\n\nÀ modifier seulement pour `PHASE = \"private\"`. Chaque entrée est "
               "expliquée en commentaire ; les valeurs par défaut viennent des variables d'environnement décrites dans "
               "`private.env.example`. Au lancement, une liste de contrôle vérifie le jeu, les colonnes, les modèles et les "
               "services, et s'arrête sur ce qui manque avant tout calcul.", section="setup"),
        _nb_code(cellule_config_privee(plan), section="private_config"),
        _nb_md("### Dépendances et initialisation\n\nLes versions exactes utilisées pour les résultats publiés sont dans "
               "`requirements.txt`. Les paquets propres à cette expérience s'installent depuis le notebook avec "
               "`INSTALL_DEPENDENCIES=1`.", section="setup"),
        _nb_code("# Paquets propres à cette expérience (en plus de requirements.txt).\n"
                 f"EXTRA_PACKAGES = {deps!r}\n"
                 "import os, subprocess, sys\n"
                 'if EXTRA_PACKAGES and os.environ.get("INSTALL_DEPENDENCIES") == "1":\n'
                 '    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *EXTRA_PACKAGES])', section="install"),
        _nb_code("from helpers import *  # fonctions communes (voir le tableau ci-dessus)\n\n"
                 "configure(phase=PHASE, scale=SCALE, seed=SEED, output_dir=OUTPUT_DIR)\n"
                 "apply_private_config(PRIVATE_CONFIG)  # phase private : liste de contrôle, arrêt si quelque chose manque\n"
                 "prepare_environment()                 # télémétrie coupée ; hors ligne si les modèles sont locaux\n"
                 "use_plot_style()\n"
                 "show_configuration()", section="init"),
    ]


def _meme_titre(ligne: str, groupe: str) -> bool:
    """« ### La question » redit « 1. La question » : même texte, numéro, casse et ponctuation mis à part."""
    def norme(t):
        return re.sub(r"[\W\d_]+", " ", t.lower()).strip()
    return ligne.startswith("#") and norme(ligne) == norme(groupe)


def assembler(fiche: dict, plan: dict, faits: dict, code: dict, jour: str = ""):
    """Notebook complet : cadre du moteur, puis le code rédigé avec un titre numéroté par groupe de sections."""
    nb = nbformat.v4.new_notebook()
    nb.metadata.update({"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                        "language_info": {"name": "python"}})
    cellules, titre = cadre(plan, code), None
    for c in code["cellules"]:
        groupe, source = SECTIONS.get(c["section"]), c["source"]
        if groupe and groupe != titre:
            titre = groupe
            if c["type"] == "markdown":  # le titre de section ouvre la première cellule de texte, sans la redire
                premiere, _, suite = source.lstrip().partition("\n")
                if _meme_titre(premiere, groupe):
                    source = suite.lstrip("\n")
                source = f"## {groupe}\n\n{source}".rstrip()
            else:
                cellules.append(_nb_md(f"## {groupe}", section=c["section"]))
        cellules.append((_nb_md if c["type"] == "markdown" else _nb_code)(source, section=c["section"]))
    nb.cells = cellules
    return nb


# --------------------------------------------------------------------------- figures

def extraire_figures(nb, dossier: Path) -> list[dict]:
    """Figures enregistrées par save_figure (sorties marquées « figure ») : écrites dans dossier/figures, dans l'ordre."""
    figures, vues = [], set()
    for c in nb.cells:
        for o in c.get("outputs", []):
            meta = (o.get("metadata") or {})
            info = meta.get("figure") or (meta.get("image/png") or {}).get("figure")
            png = (o.get("data") or {}).get("image/png")
            if not (info and png) or info.get("name") in vues or not re.fullmatch(r"[a-z0-9_]{2,40}", str(info.get("name"))):
                continue
            vues.add(info["name"])
            (dossier / "figures").mkdir(parents=True, exist_ok=True)
            (dossier / "figures" / f"{info['name']}.png").write_bytes(base64.b64decode("".join(png) if isinstance(png, list) else png))
            figures.append({"nom": info["name"], "legende": neutre(str(info.get("caption") or "")),
                            "fichier": f"figures/{info['name']}.png", "section": c.get("metadata", {}).get("section")})
    return figures


# --------------------------------------------------------------------------- synthèse des résultats

def _valeur(a: dict, m: str) -> str:
    ci = (a.get("ci95") or {}).get(m)
    return nombre(a["metrics"].get(m), 3) + (f" [{nombre(ci[0])} ; {nombre(ci[1])}]" if ci else "")


def _numerique(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def tableau_resultats(res: dict, lisibles: dict) -> str:
    """Tableau Markdown : une ligne par approche ; la meilleure valeur de chaque métrique orientée en gras. Une métrique
    sans objet pour une approche (None) s'affiche « — » et n'entre pas dans la comparaison."""
    cles = list(dict.fromkeys(m for a in res["approaches"] for m in a["metrics"]))
    meilleures = {}
    for m in cles:
        sens = lisibles.get(m, {}).get("sens")
        vals = [v for a in res["approaches"] if _numerique(v := a["metrics"].get(m))]
        if sens in ("max", "min") and len(vals) == len(res["approaches"]) and len(set(vals)) > 1:
            meilleures[m] = max(vals) if sens == "max" else min(vals)
    tete = ["Approche", *[lisibles.get(m, {}).get("libelle", m) + (" ★" if m == res["primary_metric"] else "")
                          + {"max": " ↑", "min": " ↓"}.get(lisibles.get(m, {}).get("sens"), "") for m in cles]]
    latence = any(a.get("latency_ms") for a in res["approaches"])
    if latence:
        tete.append("Latence médiane (ms) ↓")
    lignes = ["| " + " | ".join(_cellule(t) for t in tete) + " |", "|" + "---|" * len(tete)]
    for a in res["approaches"]:
        cellules = [f"{_cellule(a['name'])} ({ROLES.get(a['role'], a['role']).lower()})"]
        for m in cles:
            v = _valeur(a, m)
            cellules.append(f"**{v}**" if m in meilleures and a["metrics"].get(m) == meilleures[m] else v)
        if latence:
            cellules.append(nombre((a.get("latency_ms") or {}).get("p50")))
        lignes.append("| " + " | ".join(cellules) + " |")
    lignes += ["", "★ métrique principale · ↑ plus haut = meilleur · ↓ plus bas = meilleur · entre crochets : intervalle de "
                   "confiance à 95 % (bootstrap) · en gras : la meilleure valeur."]
    return "\n".join(lignes)


def en_bref(plan: dict, res: dict, revue: dict, echelle_complete: bool, lisibles: dict | None = None) -> str:
    mp = res["primary_metric"]
    libelle = (lisibles or {}).get(mp, {}).get("libelle") or (res.get("metric_labels") or {}).get(mp) or mp
    vals = {a["role"]: a for a in res["approaches"]}
    nouv, ref = vals.get("new"), vals.get("reference")
    verdict = neutre(revue.get("en_bref") or " ".join(_FIN_DE_PHRASE.split(revue["conclusion"])[:2]))
    if not (nouv and ref):
        return verdict
    echelle = "échelle complète" if echelle_complete else "échelle réduite, chiffres indicatifs"
    chiffres = (f"{libelle} : {_valeur(nouv, mp)} pour {neutre(nouv['name'])}, contre {_valeur(ref, mp)} pour "
                f"{neutre(ref['name'])} ({res['dataset']['n_examples']} exemples de test, {echelle}).")
    return f"{verdict} {chiffres}"


def _points(titre: str, items: list[str]) -> list[str]:
    return [f"**{titre}**", "", *[f"- {neutre(x)}" for x in items], ""] if items else []


def pour_le_lecteur(revue: dict, gravite: str) -> list[str]:
    """Problèmes de la revue reformulés pour le lecteur des résultats ; une revue plus ancienne n'a que `description`."""
    return [x.get("pour_le_lecteur", x["description"]) for x in revue["problemes"]
            if x["gravite"] == gravite and x.get("pour_le_lecteur", x["description"]).strip()]


def bilan_markdown(plan: dict, etat: dict, res: dict, revue: dict, code: dict | None = None, figures: list[dict] | None = None) -> str:
    """Dernière section du notebook : verdict, tableau, réserves, limites, suite."""
    code = code or etat.get("code") or {}
    lisibles = metriques_lisibles(code, res)
    complet = bool(etat.get("resultats_complets"))
    info = etat.get("complete") or {}
    reduit = etat["publique"]["resultats"]
    if complet:
        echelle = (f"*Échelle complète : {res['dataset']['n_examples']} exemples de test de `{res['dataset']['name']}` "
                   f"({_materiel(info)}). La mécanique avait d'abord été validée "
                   f"à l'échelle réduite ({reduit['dataset']['n_examples']} exemples, CPU).*")
    else:
        echelle = (f"*Échelle réduite seulement ({res['dataset']['n_examples']} exemples, CPU) : "
                   f"{neutre(info.get('raison') or 'échelle complète non exécutée')}. Ces chiffres valident la mécanique ; "
                   "ils sont indicatifs et à confirmer avec `SCALE = \"full\"`.*")
    lignes = ["## 7. Bilan", "", f"> **Verdict** (confiance {revue['confiance']}). {neutre(revue['conclusion'])}", "", echelle, "",
              tableau_resultats(res, lisibles), ""]
    lignes += _points("Réserves", pour_le_lecteur(revue, "majeur"))
    lignes += _points("Limites", code.get("limites") or [])
    if code.get("ecarts_sandbox") or plan.get("adaptation_sandbox"):
        lignes += _points("Échelle réduite : ce qui diffère de l'échelle complète", [code.get("ecarts_sandbox") or plan["adaptation_sandbox"]])
    lignes += _points("Améliorations possibles", pour_le_lecteur(revue, "mineur")[:5])
    lignes += ["**Et ensuite ?** Rejouer le même protocole sur vos données : `PHASE = \"private\"`, `SCALE = \"full\"`, après avoir "
               "préparé le jeu et la configuration décrits dans le README (section « Sur vos données »)."]
    return "\n".join(lignes)


# --------------------------------------------------------------------------- README

def _etiquette_mermaid(texte: str, n: int = 60) -> str:
    t = re.sub(r"[\"\[\]{}()<>|#;`]", " ", neutre(str(texte)))
    t = re.sub(r"\s+([,.;:…])", r"\1", re.sub(r"\s+", " ", t)).strip()
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


ACCES = {"hub_libre": "Hugging Face, libre", "telechargement_libre": "téléchargement libre", "compte_requis": "compte requis",
         "sur_demande": "sur demande", "payant": "payant", "restreint": "accès restreint"}


def _texte_alternatif(legende: str) -> str:
    """Texte alternatif d'une figure : le début de sa légende, sans ponctuation qui casse le Markdown."""
    debut = re.split(r"[(;:]", legende, maxsplit=1)[0]
    return _etiquette_mermaid(debut, 90) or "figure"


def _figures_md(figures: list[dict]) -> list[str]:
    lignes = []
    for f in figures:
        lignes += [f"![{_texte_alternatif(f['legende'])}]({f['fichier']})", "", f"*{f['legende']}*", ""]
    return lignes


def _licence(licence: str | None) -> str:
    """Licence déclarée sur le Hub ; « other » ou « unknown » cachent souvent des restrictions (usage commercial)."""
    if not licence:
        return "non précisée : lire la fiche du modèle"
    return f"{licence} (licence propre au modèle : lire ses conditions, notamment d'usage commercial)" \
        if licence in ("other", "unknown") else licence


def _materiel(info: dict) -> str:
    """« GPU T4 (Google Colab), 25 min » ou « CPU, 4 cœurs, 8 s »."""
    return f"{info.get('materiel') or 'GPU'}, {_duree(info.get('secondes'))}"


def diagramme_protocole(plan: dict, res: dict, lisibles: dict) -> str:
    j = plan["jeu_public"]
    metriques = ", ".join(lisibles.get(m, {}).get("libelle", m) for m in list(res["approaches"][0]["metrics"])[:4])
    return "\n".join([
        "```mermaid",
        "flowchart LR",
        f'    D["Jeu public<br/>{_etiquette_mermaid(j["hub_id"])}"] --> S["Même découpage<br/>{res["dataset"]["n_examples"]} exemples de test"]',
        f'    S --> R["Référence<br/>{_etiquette_mermaid(plan["reference"]["nom"])}"]',
        f'    S --> N["Nouvelle approche<br/>{_etiquette_mermaid(plan["nouvelle"]["nom"])}"]',
        f'    R --> M["Mêmes métriques<br/>{_etiquette_mermaid(metriques, 70)}"]',
        "    N --> M",
        '    M --> C["Comparaison<br/>intervalles de confiance à 95 %"]',
        "```",
    ])


DIAGRAMME_PHASES = "\n".join([
    "```mermaid",
    "flowchart LR",
    '    A["1. Jeu public<br/>se faire une conviction"] --> B["2. Vos données<br/>même protocole, même code"]',
    '    B --> C["3. Décision<br/>sur vos métriques métier"]',
    "```",
])


def readme(fiche: dict, plan: dict, faits: dict, etat: dict, res: dict, revue: dict, code: dict, figures: list[dict],
           fichiers: list[tuple[str, str]]) -> str:
    """README du dépôt : question, protocole, résultats illustrés, enseignements, rejouer (jeu public, puis données
    internes), contenu, références, glossaire."""
    lisibles = metriques_lisibles(code, res)
    complet = bool(etat.get("resultats_complets"))
    info = etat.get("complete") or {}
    j, p2 = plan["jeu_public"], plan["phase2"]
    reduit = etat["publique"]
    L = [f"# {plan['titre']}", "", f"> **En bref.** {en_bref(plan, res, revue, complet, lisibles)}", ""]
    figures_donnees = [f for f in figures if f.get("section") in ("public_data", "private_data")]
    figures_resultats = ([f for f in figures if f["nom"] == "comparison"]
                         + [f for f in figures if f["nom"] != "comparison" and f not in figures_donnees])
    L += ["## Sommaire", "", "1. [La question](#la-question)", "2. [Le protocole](#le-protocole)", "3. [Les résultats](#les-résultats)",
          "4. [Ce qu'il faut retenir](#ce-quil-faut-retenir)", "5. [Rejouer l'expérience](#rejouer-lexpérience)",
          "6. [Contenu du dépôt](#contenu-du-dépôt)", "7. [Références](#références)", "8. [Glossaire](#glossaire)", ""]

    # 1. La question
    L += ["## La question", ""]
    if code.get("resume_simple"):
        L += [neutre(code["resume_simple"]), ""]
    L += [f"**Hypothèse testée.** {neutre(plan['hypothese'])}", "",
          "| | Référence | Nouvelle approche |", "|---|---|---|",
          f"| **Approche** | {_cellule(plan['reference']['nom'])} | {_cellule(plan['nouvelle']['nom'])} |",
          f"| **Principe** | {_cellule(plan['reference']['description'])} | {_cellule(plan['nouvelle']['description'])} |",
          f"| **Mise en œuvre** | {_cellule(plan['reference']['implementation'])} | {_cellule(plan['nouvelle']['implementation'])} |",
          f"| **Modèles** | {_modeles_md(plan['reference']['modeles_hf'])} | {_modeles_md(plan['nouvelle']['modeles_hf'])} |", ""]

    # 2. Le protocole
    L += ["## Le protocole", "", diagramme_protocole(plan, res, lisibles), "",
          f"- **Données** : [`{j['hub_id']}`](https://huggingface.co/datasets/{j['hub_id']})"
          f"{' (configuration `' + j['config'] + '`)' if j.get('config') else ''}, évaluation sur le découpage "
          f"`{j.get('split_evaluation') or res['dataset'].get('split')}` ({res['dataset']['n_examples']} exemples). {neutre(j['representativite'])}",
          f"- **Écarts avec des données réelles** : {neutre(j.get('ecarts_donnees_reelles') or j.get('ecarts_banque', ''))}",
          *([f"- **Déroulé** : {neutre(code.get('protocole') or plan.get('protocole'))}"] if code.get("protocole") or plan.get("protocole") else []),
          f"- **Deux échelles, un seul code** : l'échelle réduite ({reduit['resultats']['dataset']['n_examples']} exemples, CPU, "
          f"{_duree(reduit.get('secondes'))}) valide la mécanique ; "
          + (f"l'échelle complète ({_materiel(info)}) donne les résultats."
             if complet else f"l'échelle complète n'a pas été exécutée ({neutre(info.get('raison') or '—')}) : résultats indicatifs."), ""]
    L += _figures_md(figures_donnees[:2])
    if lisibles:
        L += ["**Les métriques**", "", "| Métrique | Ce qu'elle mesure | Meilleur si |", "|---|---|---|"]
        L += [f"| {_cellule(v['libelle'])}{' ★' if k == res['primary_metric'] else ''} | {_cellule(v['explication'])} | "
              f"{ {'max': 'plus haut', 'min': 'plus bas'}.get(v['sens'], '—') } |" for k, v in lisibles.items()]
        L += ["", "★ métrique principale : c'est elle qui tranche la question.", ""]

    # 3. Les résultats
    L += ["## Les résultats", "", tableau_resultats(res, lisibles), ""]
    L += _figures_md(figures_resultats[:4])
    L += [f"**Lecture** (confiance {revue['confiance']}). {neutre(revue['conclusion'])}", ""]

    # 4. Ce qu'il faut retenir
    L += ["## Ce qu'il faut retenir", ""]
    L += _points("Réserves", pour_le_lecteur(revue, "majeur"))
    L += _points("Limites", code.get("limites") or [])
    L += _points("Pistes d'amélioration", pour_le_lecteur(revue, "mineur")[:5])

    # 5. Rejouer l'expérience
    L += ["## Rejouer l'expérience", "", DIAGRAMME_PHASES, "",
          "### Sur le jeu public", "",
          "```bash", "pip install -r requirements.txt", "jupyter lab experiment.ipynb", "```", "",
          "Dans la cellule de paramètres : `PHASE = \"public\"`, puis `SCALE = \"reduced\"` (quelques minutes sur CPU, pour "
          "vérifier) ou `SCALE = \"full\"` (résultats publiés"
          + (f" ; {_materiel(info)})." if complet else ").") + " Les résultats sont "
          "écrits dans `outputs/results_public.json`, les figures dans `outputs/figures/` (les résultats publiés en sont une "
          "copie, dans `results/` et `figures/`).", "",
          "### Sur vos données (environnement de services financiers)", "",
          "Le même protocole, sur vos données, dans votre environnement : seule la cellule `PRIVATE_CONFIG` s'adapte. Le code "
          "de cette phase a été exécuté de bout en bout sur un jeu factice écrit comme un vrai export (dossier de fichiers, "
          "CSV « ; » en Windows-1252, colonnes à faire correspondre), contre un service simulé à la convention OpenAI (clé lue "
          "dans un fichier de secret), modèles chargés hors ligne.", "",
          "**1. Préparer le jeu de données.** Colonnes attendues (les noms peuvent différer : on les fait correspondre "
          "dans `PRIVATE_CONFIG[\"columns\"]`) :", "",
          "| Colonne attendue | Type | Contenu |", "|---|---|---|"]
    L += [f"| `{c['colonne']}` | {_cellule(c['type'])} | {_cellule(c['description'])} |" for c in p2["donnees"]]
    L += ["", f"Volume conseillé : {neutre(p2['volume_conseille'])}", ""]
    L += _points("Conditions pour que la comparaison soit concluante", p2.get("prescriptions") or [])
    if p2.get("preparation"):
        L += [f"**Constituer l'extraction.** {neutre(p2['preparation'])}", ""]
    L += ["**2. Rendre les modèles disponibles hors ligne** (si l'environnement n'a pas accès à Internet) :", "",
          "```bash", "python download_models.py ./models --scale full   # sur un poste connecté", "```", "",
          "puis copier le dossier dans l'environnement et le désigner par `LOCAL_MODELS_DIR`. Les révisions téléchargées sont "
          "celles des résultats publiés.", "",
          "**3. Configurer** l'environnement du notebook à partir de `private.env.example` (ou directement la cellule "
          "`PRIVATE_CONFIG`). Jamais de clé dans le notebook : une variable, un fichier de secret monté ou la saisie masquée.", "",
          "| Variable | Rôle |", "|---|---|"]
    L += [f"| `{n}` | {d} |" for n, d in variables_environnement(plan)]
    L += ["", "**4. Lancer** avec `PHASE = \"private\"` (et `SCALE = \"full\"`). La cellule d'initialisation affiche une liste de "
          "contrôle (jeu, colonnes, modèles, services) et s'arrête sur ce qui manque, avant tout calcul. Les résultats sont "
          "écrits dans `outputs/results_private.json` (avec versions, machine, graine et empreinte des données).", "",
          f"**5. Interpréter.** Métriques métier à suivre : {neutre(p2['metriques_metier'])}", "",
          f"**Garde-fous.** {neutre(p2['garde_fous'])} En phase `private`, le notebook refuse les services publics (OpenAI, "
          "Anthropic, Mistral, Hugging Face…) et le HTTP non chiffré, n'affiche aucune donnée brute et coupe la télémétrie "
          "des bibliothèques ; avec `LOCAL_MODELS_DIR`, il travaille sans accès à Internet.", ""]

    # 6. Contenu du dépôt
    L += ["## Contenu du dépôt", "", "| Fichier | Contenu |", "|---|---|"]
    L += [f"| `{n}` | {d} |" for n, d in fichiers]
    L += [""]

    # 7. Références
    L += ["## Références", ""]
    sources = [s for s in fiche.get("sources", [])[:8] if s.get("url")]
    if sources:
        L += ["**Sources de l'approche**", ""] + [f"- [{_cellule(s.get('titre') or s['url'])}]({s['url']})" for s in sources] + [""]
    autres = [x for x in j.get("jeux_explores") or [] if x.get("decision") != "retenu"]
    if autres:
        L += ["**Autres jeux de données examinés**", "", "| Jeu | Accès | Intérêt | Pourquoi pas ici |", "|---|---|---|---|"]
        L += [(f"| [{_cellule(x['nom'])}]({x['url']})" if x.get("url") else f"| {_cellule(x['nom'])}")
              + f" | {ACCES.get(x['acces'], _cellule(x['acces']))} | {_cellule(x['adequation'])} | {_cellule(x['raison'])} |" for x in autres]
        L += [""]
    licences = [f"`{m}` : {_licence(f.get('licence'))}" for m, f in (faits.get("modeles") or {}).items() if f]
    jeu = (faits.get("jeu") or {}).get("licence")
    L += ["**Licences**", "", f"- Jeu de données `{j['hub_id']}` : {jeu or 'voir sa fiche'}"]
    L += [f"- {x}" for x in licences] + ["- Vérifier la licence de chaque modèle avant tout usage au-delà de l'évaluation.", ""]

    # 8. Glossaire
    gloss = code.get("glossaire") or []
    if gloss:
        L += ["## Glossaire", "", "| Terme | Définition |", "|---|---|"]
        L += [f"| **{_cellule(g['terme'])}** | {_cellule(g['definition'])} |" for g in gloss] + [""]
    return "\n".join(L).rstrip() + "\n"
