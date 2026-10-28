"""Exécute un notebook de POC dans le sandbox et rend compte (lancé dans le conteneur, sans aucun secret).

- Paramètres injectés après la cellule marquée « parameters » (comme papermill) : PHASE, TAILLE, GRAINE, POC_SORTIE.
- Phase « banque » : démarre le faux service à la convention OpenAI (faux_service.py) et pointe les variables
  LLMAAS_* / STTAAS_* dessus (clé par fichier de secret), avec BANQUE_DONNEES=factice : le code de la phase 2 est
  réellement exécuté, modèles servis hors ligne depuis BANQUE_MODELES_DIR (--modeles).
- Rapport JSON : succès, durée, cellule en erreur (section, exception, trace), fin des sorties des dernières cellules.

Usage : python executer.py --entree poc.ipynb --sortie execute.ipynb --phase publique --rapport rapport.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError

ICI = Path(__file__).resolve().parent
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _texte_sorties(cellule, limite: int = 1500) -> str:
    morceaux = []
    for o in cellule.get("outputs", []):
        if o.get("output_type") == "stream":
            morceaux.append(o.get("text", ""))
        elif o.get("output_type") in ("execute_result", "display_data"):
            morceaux.append((o.get("data") or {}).get("text/plain", ""))
        elif o.get("output_type") == "error":
            morceaux.append(f"{o.get('ename')}: {o.get('evalue')}\n" + "\n".join(o.get("traceback", [])[-6:]))
    t = _ANSI.sub("", "".join(m if isinstance(m, str) else "".join(m) for m in morceaux))
    return t if len(t) <= limite else "[…]\n" + t[-limite:]


def injecter_parametres(nb, valeurs: dict) -> None:
    idx = next((i for i, c in enumerate(nb.cells) if "parameters" in c.get("metadata", {}).get("tags", [])), None)
    source = "# Paramètres injectés par le sandbox\n" + "\n".join(f"{k} = {v!r}" for k, v in valeurs.items())
    cellule = nbformat.v4.new_code_cell(source, metadata={"tags": ["injected-parameters"]})
    nb.cells.insert(0 if idx is None else idx + 1, cellule)


def modeles_hors_ligne(modeles: list[str], dossier: Path) -> dict:
    """Phase banque comme dans un environnement sans Internet : les modèles de la phase publique sont copiés de son
    cache vers BANQUE_MODELES_DIR (org--nom), puis le noyau démarre hors ligne avec un cache Hugging Face vide. Un
    modèle chargé autrement que par chemin_modele(...) échoue alors ici comme il échouerait à la banque. Si un
    modèle manque au cache (jamais chargé en phase publique), le test hors ligne n'est pas appliqué, et le rapport le dit."""
    import shutil
    from huggingface_hub import snapshot_download
    racine, copies, absents = dossier / "modeles_banque", [], []
    for m in modeles:
        try:
            instantane = snapshot_download(m, local_files_only=True)  # dossier du cache (liens vers les blobs)
            shutil.copytree(instantane, racine / m.replace("/", "--"), symlinks=False, dirs_exist_ok=True)
            copies.append(m)
        except Exception:  # noqa: BLE001 — absent du cache, quelle qu'en soit la forme
            absents.append(m)
    if absents:
        return {"applique": False, "copies": copies, "absents": absents}
    os.environ.update({"BANQUE_MODELES_DIR": str(racine), "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                       "TRANSFORMERS_OFFLINE": "1", "HF_HOME": str(dossier / ".cache" / "hf_banque_vide")})
    os.environ.pop("HF_HUB_CACHE", None)
    return {"applique": True, "copies": copies, "absents": []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entree", required=True)
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--rapport", required=True)
    ap.add_argument("--phase", choices=["publique", "banque"], default="publique")
    ap.add_argument("--taille", choices=["sandbox", "complete"], default="sandbox")
    ap.add_argument("--graine", type=int, default=42)
    ap.add_argument("--timeout-cellule", type=int, default=1800)
    ap.add_argument("--freeze", help="fichier où écrire `pip freeze` après l'exécution")
    ap.add_argument("--modeles", default="", help="phase banque : modèles (org/nom, séparés par des virgules) à servir hors ligne")
    a = ap.parse_args()

    entree = Path(a.entree).resolve()
    dossier = entree.parent
    nb = nbformat.read(entree, as_version=4)
    injecter_parametres(nb, {"PHASE": a.phase, "TAILLE": a.taille, "GRAINE": a.graine, "POC_SORTIE": str(dossier)})
    journal = dossier / f"appels_service_{a.phase}.jsonl"
    hors_ligne = None
    if a.phase == "banque":
        sys.path.insert(0, str(ICI))
        import faux_service
        journal.unlink(missing_ok=True)
        serveur = faux_service.demarrer(0, str(journal))  # port libre choisi par le système
        url = f"http://127.0.0.1:{serveur.server_address[1]}/v1"
        secrets = dossier / ".secrets"
        secrets.mkdir(exist_ok=True)
        (secrets / "cle_service").write_text(faux_service.CLE_ATTENDUE, encoding="utf-8")
        for p in ("LLMAAS", "STTAAS"):  # clé par fichier de secret monté, comme en entreprise
            os.environ.pop(f"{p}_API_KEY", None)
            os.environ.update({f"{p}_BASE_URL": url, f"{p}_API_KEY_FILE": str(secrets / "cle_service"), f"{p}_MODEL": "modele-factice"})
        os.environ["LLMAAS_EMBEDDINGS_MODEL"] = "embeddings-factice"
        os.environ["BANQUE_DONNEES"] = "factice"
        if a.modeles:
            hors_ligne = modeles_hors_ligne([m for m in a.modeles.split(",") if m], dossier)
    os.environ.update({"POC_PHASE": a.phase, "POC_TAILLE": a.taille, "POC_SORTIE": str(dossier), "MPLBACKEND": "Agg"})

    rapport = {"phase": a.phase, "taille": a.taille, "ok": False, "erreur": None, "hors_ligne": hors_ligne}
    t0 = time.time()
    client = NotebookClient(nb, timeout=a.timeout_cellule, kernel_name="python3", allow_errors=False,
                            resources={"metadata": {"path": str(dossier)}})
    progres = dossier / f"progres_{a.phase}.json"

    def debut_cellule(cell, cell_index, **_):  # en cas d'arrêt brutal (délai global), dit quelle cellule tournait
        progres.write_text(json.dumps({"cellule": cell_index, "section": cell.get("metadata", {}).get("section"),
                                       "source": (cell.get("source") or "")[:3000]}, ensure_ascii=False), encoding="utf-8")

    client.on_cell_start = debut_cellule
    try:
        client.execute()
        rapport["ok"] = True
    except (CellExecutionError, CellTimeoutError, DeadKernelError) as e:
        en_erreur = next((i for i, c in enumerate(nb.cells) if c.get("cell_type") == "code"
                          and any(o.get("output_type") == "error" for o in c.get("outputs", []))), None)
        if isinstance(e, CellTimeoutError):
            nature = f"délai dépassé ({a.timeout_cellule} s pour une cellule)"
        elif isinstance(e, DeadKernelError):
            nature = "noyau arrêté (mémoire saturée ou plantage natif)"
        else:
            nature = "exception"
        if en_erreur is None:  # délai ou noyau mort : la cellule en cours est la dernière exécutée
            en_erreur = max((i for i, c in enumerate(nb.cells) if c.get("cell_type") == "code" and c.get("execution_count")), default=None)
        cel = nb.cells[en_erreur] if en_erreur is not None else {}
        rapport["erreur"] = {
            "nature": nature, "cellule": en_erreur, "section": cel.get("metadata", {}).get("section"),
            "source": (cel.get("source") or "")[:4000], "message": _ANSI.sub("", str(e))[-3000:],
        }
    rapport["secondes"] = round(time.time() - t0, 1)
    executees = [(i, c) for i, c in enumerate(nb.cells) if c.get("cell_type") == "code" and c.get("outputs")]
    rapport["sorties"] = [{"cellule": i, "section": c.get("metadata", {}).get("section"), "texte": _texte_sorties(c)} for i, c in executees[-12:]]
    resultats = dossier / f"resultats_{a.phase}.json"
    rapport["resultats"] = json.loads(resultats.read_text(encoding="utf-8")) if rapport["ok"] and resultats.exists() else None
    if journal.exists():
        appels = [json.loads(l) for l in journal.read_text(encoding="utf-8").splitlines() if l.strip()]
        rapport["appels_service"] = {"n": len(appels), "routes": sorted({x["route"] for x in appels}),
                                     "refus": sum(1 for x in appels if x["code"] >= 400)}
    nbformat.write(nb, a.sortie)
    if a.freeze:
        gel = subprocess.run([sys.executable, "-m", "pip", "freeze", "--exclude-editable"], capture_output=True, text=True)
        Path(a.freeze).write_text(gel.stdout, encoding="utf-8")
    Path(a.rapport).write_text(json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: rapport[k] for k in ("phase", "ok", "secondes")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
