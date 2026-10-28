"""État persistant entre les exécutions (JSON, versionnable dans git)."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path


def read_json(path: Path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:  # mêmes octets sous Windows et Linux (sha256 du manifeste)
        json.dump(data, f, ensure_ascii=False, indent=1)
    tmp.replace(path)


class State:
    """
    seen.json      clé d'élément -> date ISO où il a été vu par le tri (évite de le retrier)
    sujets.json    sujets publiés (mémoire éditoriale pour le suivi et l'anti-répétition)
    sources.json   statistiques par source (collectés, retenus, publiés, erreurs, référence html)
    meta.json      divers (dernière collecte arXiv réussie…)
    """

    def __init__(self, state_dir: Path):
        self.dir = state_dir
        self.seen: dict[str, str] = read_json(state_dir / "seen.json", {})
        self.sujets: list[dict] = read_json(state_dir / "sujets.json", [])
        self.sources: dict[str, dict] = read_json(state_dir / "sources.json", {})
        self.meta: dict = read_json(state_dir / "meta.json", {})

    def source_stat(self, source_id: str) -> dict:
        return self.sources.setdefault(source_id, {
            "collectes": 0, "candidats": 0, "retenus": 0, "publies": 0,
            "erreurs": 0, "derniere_erreur": None, "reference_html": [],
        })

    def sujets_recents(self, jour: date, jours: int = 14) -> list[dict]:
        # Le jour même est exclu : une relance ne doit pas prendre ses propres sujets pour des redites.
        limite, auj = (jour - timedelta(days=jours)).isoformat(), jour.isoformat()
        return [s for s in self.sujets if limite <= s.get("date", "") < auj]

    def save(self, jour: date) -> None:
        limite_seen = (jour - timedelta(days=45)).isoformat()
        self.seen = {k: v for k, v in self.seen.items() if v >= limite_seen}
        limite_sujets = (jour - timedelta(days=120)).isoformat()
        self.sujets = [s for s in self.sujets if s.get("date", "") >= limite_sujets]
        write_json(self.dir / "seen.json", self.seen)
        write_json(self.dir / "sujets.json", self.sujets)
        write_json(self.dir / "sources.json", self.sources)
        write_json(self.dir / "meta.json", self.meta)
