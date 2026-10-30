"""Fonctions communes du notebook : paramètres d'exécution, appels aux services, données internes, mesures,
résultats et figures.

Le notebook les importe (`from helpers import *`) : son code ne contient que ce qui est propre à l'expérience.

Conventions :
- deux phases, un seul code : `public` (jeu public, modèles locaux) et `private` (données internes, dans
  l'environnement cible, services internes à la convention OpenAI) ;
- deux échelles : `reduced` (quelques minutes sur CPU, pour valider la mécanique) et `full` (tout le jeu) ;
- aucune clé ni URL dans le notebook : tout vient de variables d'environnement ou de la cellule `PRIVATE_CONFIG` ;
- en phase `private`, aucune donnée brute n'est affichée, les services publics et le HTTP non chiffré sont refusés.

Dépendances : bibliothèque standard ; numpy, pandas, matplotlib, openai et soundfile seulement pour les fonctions
qui en ont besoin.
"""
from __future__ import annotations

import contextlib
import getpass
import hashlib
import io
import json
import math
import os
import platform
import random
import re
import statistics
import sys
import tempfile
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

__all__ = [
    "PHASES", "SCALES", "SERVICE_CALLS", "configure", "current_phase", "current_scale", "set_seed", "device",
    "sample_size", "by_scale", "check_endpoint", "service_config", "openai_client", "show_configuration",
    "prepare_environment", "model_path", "apply_private_config", "check_private_environment", "llm_chat", "embed_texts",
    "wav_bytes", "transcribe", "decode_audio", "resample", "telephone_channel", "load_private_data", "preview",
    "fingerprint", "timer", "latency_stats", "bootstrap_ci", "normalize_fr", "wer", "environment_info", "save_results",
    "results_table", "use_plot_style", "save_figure", "plot_comparison", "COLORS",
]

PHASES = ("public", "private")
SCALES = ("reduced", "full")
_SERVICES = {"LLMAAS": "service LLM (convention OpenAI)", "STTAAS": "service de transcription (convention OpenAI)"}
# Points d'accès publics : jamais de données internes vers eux, sauf autorisation explicite (ALLOW_PUBLIC_ENDPOINT=1).
_PUBLIC_ENDPOINTS = ("api.openai.com", "openai.azure.com", "api.anthropic.com", "api.mistral.ai",
                     "generativelanguage.googleapis.com", "api-inference.huggingface.co", "router.huggingface.co",
                     "api.together.xyz", "api.groq.com", "openrouter.ai")
SERVICE_CALLS: list[dict] = []   # un élément par appel de service : durée, modèle, succès (résumé dans les résultats)
_RUN = {"phase": "public", "scale": "reduced", "seed": 42, "output_dir": Path(".")}


# --------------------------------------------------------------------------- paramètres d'exécution

def configure(phase: str, scale: str, seed: int = 42, output_dir: str = ".") -> None:
    """Fixe la phase, l'échelle, la graine et le dossier des résultats pour tout le notebook (cellule d'initialisation).

    `phase` : « public » (jeu public) ou « private » (données internes) ; `scale` : « reduced » ou « full »."""
    if phase not in PHASES:
        raise ValueError(f"PHASE doit valoir {PHASES}, pas {phase!r}")
    if scale not in SCALES:
        raise ValueError(f"SCALE doit valoir {SCALES}, pas {scale!r}")
    _RUN.update(phase=phase, scale=scale, seed=int(seed), output_dir=Path(output_dir))
    _RUN["output_dir"].mkdir(parents=True, exist_ok=True)
    set_seed(int(seed))


def current_phase() -> str:
    """Phase en cours : « public » ou « private »."""
    return _RUN["phase"]


def current_scale() -> str:
    """Échelle en cours : « reduced » ou « full »."""
    return _RUN["scale"]


def set_seed(seed: int = 42) -> None:
    """Graine unique pour random, numpy et torch (s'ils sont présents) : résultats reproductibles."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    with contextlib.suppress(ImportError):
        import numpy as np
        np.random.seed(seed)
    with contextlib.suppress(ImportError):
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def device() -> str:
    """Matériel de calcul disponible : « cuda » (GPU NVIDIA), « mps » (GPU Apple) ou « cpu »."""
    with contextlib.suppress(ImportError):
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def sample_size(reduced: int, full: int | None = None) -> int | None:
    """Nombre d'exemples à utiliser : `reduced` à l'échelle réduite, `full` (None = tout le jeu) à l'échelle complète."""
    return reduced if current_scale() == "reduced" else full


def by_scale(reduced, full):
    """Valeur qui dépend de l'échelle : modèle substitut, nombre d'époques, de pas ou d'exemples."""
    return reduced if current_scale() == "reduced" else full


# --------------------------------------------------------------------------- configuration et sécurité

def _env(name: str, required: bool = True, secret: bool = False) -> str | None:
    """Variable d'environnement ; pour un secret, aussi <NOM>_FILE (fichier de secret monté), puis saisie masquée."""
    value = os.environ.get(name)
    if value:
        return value
    if secret and os.environ.get(f"{name}_FILE"):
        secret_file = Path(os.environ[f"{name}_FILE"]).expanduser()
        if not secret_file.is_file():
            raise RuntimeError(f"{name}_FILE désigne un fichier absent : {secret_file}")
        return secret_file.read_text(encoding="utf-8").strip()
    if required and secret and sys.stdin and sys.stdin.isatty():
        return getpass.getpass(f"{name} : ")
    if required:
        raise RuntimeError(f"Variable d'environnement {name} absente{' (ni ' + name + '_FILE)' if secret else ''}. "
                           "Renseigne-la dans la cellule PRIVATE_CONFIG ou dans l'environnement du notebook (variables du "
                           "projet ou du job) ; une clé, jamais dans le notebook.")
    return None


def check_endpoint(url: str) -> str:
    """Refuse en phase « private » un point d'accès public ou non chiffré (sauf ALLOW_PUBLIC_ENDPOINT=1)."""
    host = (urlparse(url).hostname or "").lower()
    if not host:
        raise ValueError(f"URL de service invalide : {url!r}")
    private = current_phase() == "private"
    if private and urlparse(url).scheme != "https" and host not in ("localhost", "127.0.0.1"):
        raise ValueError(f"En phase private, le service doit être en HTTPS : {host}")
    if private and any(host == p or host.endswith("." + p) for p in _PUBLIC_ENDPOINTS) \
            and os.environ.get("ALLOW_PUBLIC_ENDPOINT") != "1":
        raise PermissionError(f"{host} est un service public : refusé en phase private (données sensibles).")
    return url


def service_config(prefix: str = "LLMAAS") -> dict:
    """URL, clé et modèle d'un service à la convention OpenAI : variables <PREFIX>_BASE_URL, _API_KEY, _MODEL."""
    url = check_endpoint(_env(f"{prefix}_BASE_URL"))
    return {"base_url": url.rstrip("/"), "api_key": _env(f"{prefix}_API_KEY", secret=True), "model": _env(f"{prefix}_MODEL")}


def openai_client(prefix: str = "LLMAAS"):
    """Client `openai` vers le service <PREFIX>. Proxy et certificats d'entreprise : HTTPS_PROXY, SSL_CERT_FILE ;
    en-têtes exigés par une passerelle : <PREFIX>_HEADERS (objet JSON)."""
    from openai import OpenAI
    config = service_config(prefix)
    headers = json.loads(os.environ.get(f"{prefix}_HEADERS") or "{}")
    return OpenAI(base_url=config["base_url"], api_key=config["api_key"],
                  timeout=float(os.environ.get("SERVICE_TIMEOUT_S", "120")),
                  max_retries=int(os.environ.get("SERVICE_MAX_RETRIES", "3")), default_headers=headers or None)


def show_configuration() -> None:
    """Affiche la configuration de l'exécution, sans jamais montrer une clé."""
    print(f"PHASE={current_phase()}  SCALE={current_scale()}  SEED={_RUN['seed']}  device={device()}  "
          f"python={platform.python_version()}")
    for prefix, label in _SERVICES.items():
        url, model = os.environ.get(f"{prefix}_BASE_URL"), os.environ.get(f"{prefix}_MODEL")
        key = "renseignée" if os.environ.get(f"{prefix}_API_KEY") or os.environ.get(f"{prefix}_API_KEY_FILE") else "absente"
        print(f"{label:45} {prefix}_BASE_URL={urlparse(url).hostname if url else '—'}  {prefix}_MODEL={model or '—'}  clé {key}")
    data = os.environ.get("PRIVATE_DATA")
    print(f"données internes : {'factices' if data == 'fake' else ('renseignées' if data else '—')}"
          f"  modèles locaux : {os.environ.get('LOCAL_MODELS_DIR') or '—'}")


def prepare_environment() -> None:
    """Désactive la télémétrie des bibliothèques ; en phase « private » avec modèles locaux, interdit les téléchargements."""
    for name in ("HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK", "ANONYMIZED_TELEMETRY"):
        os.environ.setdefault(name, "1")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import logging
    import warnings
    warnings.filterwarnings("ignore", message=r"(?s).*(HF_TOKEN|unauthenticated requests)")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")  # relu par huggingface_hub à son import, qui réinitialise son niveau
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    if current_phase() == "private" and os.environ.get("LOCAL_MODELS_DIR"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def model_path(hub_id: str) -> str:
    """Chemin à passer à `from_pretrained`, dans les deux phases : l'identifiant Hugging Face, ou son dossier local
    sous LOCAL_MODELS_DIR (org--nom ou org/nom), seul accessible dans un environnement sans Internet."""
    root = os.environ.get("LOCAL_MODELS_DIR")
    if root:
        for folder in (Path(root) / hub_id.replace("/", "--"), Path(root) / hub_id):
            if folder.is_dir():
                return str(folder)
        if current_phase() == "private":
            raise FileNotFoundError(f"{hub_id} absent de LOCAL_MODELS_DIR ({root}) : copie ses poids dans "
                                    f"{Path(root) / hub_id.replace('/', '--')} (script download_models.py du dépôt).")
    return hub_id


# --------------------------------------------------------------------------- environnement cible (phase private)

_CONFIG: dict = {}   # PRIVATE_CONFIG appliquée : correspondance des colonnes, chemins, modèles, options de lecture
_SERVICE_VARIABLES = {"base_url": "BASE_URL", "model": "MODEL", "embeddings_model": "EMBEDDINGS_MODEL",
                      "api_key_file": "API_KEY_FILE", "headers": "HEADERS"}


def apply_private_config(config: dict) -> list[str]:
    """Applique la cellule PRIVATE_CONFIG (seule à adapter), puis vérifie l'environnement. Sans effet en phase « public ».

    Rend la liste des points à vérifier (vide si tout est prêt) ; arrête le notebook sur un manque bloquant, en disant
    quoi corriger."""
    _CONFIG.clear()
    _CONFIG.update(config)
    if current_phase() != "private":
        return []
    for key, variable in (("data", "PRIVATE_DATA"), ("models_dir", "LOCAL_MODELS_DIR"), ("ca_bundle", "SSL_CERT_FILE")):
        if config.get(key):
            os.environ[variable] = str(config[key])
    for prefix in _SERVICES:
        for key, suffix in _SERVICE_VARIABLES.items():
            value = (config.get(prefix.lower()) or {}).get(key)
            if value:
                os.environ[f"{prefix}_{suffix}"] = json.dumps(value) if isinstance(value, dict) else str(value)
    issues = check_private_environment()
    if any(p.startswith("BLOQUANT") for p in issues):
        raise RuntimeError("Environnement incomplet :\n" + "\n".join(issues))
    return issues


def check_private_environment() -> list[str]:
    """Liste de contrôle avant la phase « private » : jeu lisible, modèles présents en local si LOCAL_MODELS_DIR est
    renseigné, services joignables avec leur clé. N'affiche aucun contenu."""
    issues, lines = [], []

    def ok(message):
        lines.append(f"  [ok] {message}")

    def missing(message, blocking=True):
        issues.append(("BLOQUANT : " if blocking else "à vérifier : ") + message)
        lines.append(f"  [{'MANQUE' if blocking else 'à vérifier'}] {message}")

    source = os.environ.get("PRIVATE_DATA", "")
    if not source:
        missing("PRIVATE_DATA non renseigné (fichier ou dossier du jeu, ou « fake » pour tester le code)")
    elif source == "fake":
        ok("données factices (test du code, aucun résultat interprétable)")
    else:
        try:
            files = _data_files(Path(source).expanduser())
            ok(f"jeu : {len(files)} fichier(s) dans {Path(source).name}")
        except (FileNotFoundError, ValueError) as e:
            missing(str(e))
    models = (_CONFIG.get("models") or {}).get(current_scale(), [])
    root = os.environ.get("LOCAL_MODELS_DIR")
    if root:
        absent = [m for m in models if not any((Path(root) / x).is_dir() for x in (m.replace("/", "--"), m))]
        if absent:
            missing(f"modèles absents de {root} : {absent} (script download_models.py)")
        else:
            ok(f"{len(models)} modèle(s) présents en local, sans accès à Internet")
    elif models:
        missing(f"LOCAL_MODELS_DIR vide : les modèles {models} seront téléchargés depuis Hugging Face", blocking=False)
    for prefix, label in _SERVICES.items():
        if prefix.lower() not in _CONFIG:
            continue
        try:
            config = service_config(prefix)
        except (RuntimeError, ValueError, PermissionError) as e:
            missing(f"{label} : {e}")
            continue
        try:
            openai_client(prefix).models.list()
            ok(f"{label} joignable ({urlparse(config['base_url']).hostname}, modèle {config['model']})")
        except Exception as e:  # certaines passerelles n'exposent pas /models : à vérifier, pas bloquant
            missing(f"{label} : liste des modèles indisponible ({type(e).__name__}) ; le premier appel le dira", blocking=False)
    print("Environnement cible :\n" + "\n".join(lines))
    return issues


# --------------------------------------------------------------------------- appels aux services (convention OpenAI)

def _log_call(service: str, start: float, **details) -> None:
    SERVICE_CALLS.append({"service": service, "seconds": round(time.time() - start, 3), **details})


def _lenient_json(text: str):
    text = (text or "").strip()
    with contextlib.suppress(json.JSONDecodeError):
        return json.loads(text)
    match = re.search(r"\{.*\}|\[.*\]", text, re.S)
    if match:
        with contextlib.suppress(json.JSONDecodeError):
            return json.loads(match.group(0))
    return None


def _is_strict_schema(schema) -> bool:
    """Vrai si le schéma respecte le mode strict d'OpenAI (objets fermés, toutes les propriétés requises)."""
    if isinstance(schema, list):
        return all(_is_strict_schema(x) for x in schema)
    if not isinstance(schema, dict):
        return True
    props = schema.get("properties", {})
    if (schema.get("type") == "object" or props) and (
            schema.get("additionalProperties") is not False or set(schema.get("required", [])) != set(props)):
        return False
    children = [*props.values(), *schema.get("$defs", {}).values(), *schema.get("anyOf", [])]
    if "items" in schema:
        children.append(schema["items"])
    return all(_is_strict_schema(c) for c in children)


def llm_chat(messages: list[dict], *, schema: dict | None = None, temperature: float = 0.0, max_tokens: int = 512,
             prefix: str = "LLMAAS", client=None, model: str | None = None):
    """Appel chat/completions. Avec `schema` (JSON Schema), rend l'objet décodé (None si la sortie est illisible) :
    sortie structurée si le service la prend en charge (mode strict si le schéma s'y prête), sinon consigne dans le
    prompt et décodage tolérant."""
    client = client or openai_client(prefix)
    model = model or service_config(prefix)["model"]
    request = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    start = time.time()
    if schema is not None:
        try:
            response = client.chat.completions.create(**request, response_format={"type": "json_schema", "json_schema": {
                "name": "answer", "schema": schema, "strict": _is_strict_schema(schema)}})
        except Exception as e:  # service sans sortie structurée : consigne explicite et décodage tolérant
            if not re.search(r"response_format|json_schema|unsupported|not supported|400", str(e), re.I):
                raise
            instruction = {"role": "system", "content": "Réponds uniquement par un objet JSON conforme à ce schéma : "
                                                        + json.dumps(schema, ensure_ascii=False)}
            response = client.chat.completions.create(**{**request, "messages": [instruction, *messages]})
        parsed = _lenient_json(response.choices[0].message.content or "")
        _log_call(prefix, start, model=model, ok=parsed is not None,
                  tokens=getattr(response.usage, "total_tokens", None) if response.usage else None)
        return parsed
    response = client.chat.completions.create(**request)
    _log_call(prefix, start, model=model, ok=True, tokens=getattr(response.usage, "total_tokens", None) if response.usage else None)
    return response.choices[0].message.content or ""


def embed_texts(texts: list[str], *, prefix: str = "LLMAAS", model: str | None = None, batch_size: int = 64, client=None):
    """Embeddings par lots (modèle : <PREFIX>_EMBEDDINGS_MODEL par défaut). Rend une liste de vecteurs."""
    client = client or openai_client(prefix)
    model = model or _env(f"{prefix}_EMBEDDINGS_MODEL")
    vectors = []
    for i in range(0, len(texts), batch_size):
        start = time.time()
        response = client.embeddings.create(model=model, input=texts[i:i + batch_size])
        vectors += [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
        _log_call(f"{prefix}:embeddings", start, model=model, ok=True, n=len(texts[i:i + batch_size]))
    return vectors


def wav_bytes(audio, sample_rate: int) -> bytes:
    """Tableau numpy (mono, flottant) -> octets d'un fichier WAV 16 bits."""
    import wave
    import numpy as np
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes((clipped * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def transcribe(audio, sample_rate: int, *, language: str = "fr", prefix: str = "STTAAS", client=None,
               model: str | None = None) -> str:
    """Transcription d'un signal numpy par le service de transcription (route audio/transcriptions)."""
    client = client or openai_client(prefix)
    model = model or service_config(prefix)["model"]
    start = time.time()
    response = client.audio.transcriptions.create(model=model, file=("audio.wav", wav_bytes(audio, sample_rate), "audio/wav"),
                                                  language=language)
    _log_call(prefix, start, model=model, ok=True, audio_seconds=round(len(audio) / sample_rate, 2))
    return getattr(response, "text", response if isinstance(response, str) else "")


def decode_audio(value, target_rate: int | None = 16000):
    """Signal mono float32 et fréquence d'échantillonnage, depuis une colonne audio de `datasets` (dict array /
    sampling_rate, dict bytes / path, décodeur torchcodec) ou un chemin de fichier ; rééchantillonné à `target_rate`
    si demandé. Sans soundfile, les WAV sont lus par la bibliothèque standard."""
    import numpy as np
    if hasattr(value, "get_all_samples"):  # datasets >= 4 : décodeur torchcodec
        samples = value.get_all_samples()
        audio, rate = samples.data.numpy(), int(samples.sample_rate)
    elif isinstance(value, dict) and value.get("array") is not None:
        audio, rate = np.asarray(value["array"]), int(value["sampling_rate"])
    elif isinstance(value, (dict, str, Path)) and (not isinstance(value, dict) or value.get("bytes") or value.get("path")):
        if isinstance(value, dict):
            source = io.BytesIO(value["bytes"]) if value.get("bytes") else value["path"]
        else:
            source = value
        try:
            import soundfile as sf
            audio, rate = sf.read(source, dtype="float32")
        except ImportError:  # sans libsndfile (environnement verrouillé) : WAV PCM par la bibliothèque standard
            import wave
            with wave.open(source if isinstance(source, io.BytesIO) else str(source), "rb") as w:
                rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
                raw = np.frombuffer(w.readframes(w.getnframes()), dtype={1: np.uint8, 2: "<i2", 4: "<i4"}[width])
            audio = (raw.astype(np.float32) - (128 if width == 1 else 0)) / float(2 ** (8 * width - 1))
            audio = audio.reshape(-1, channels).T if channels > 1 else audio
    else:
        raise TypeError(f"format audio non reconnu : {type(value)}")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0 if audio.shape[0] < audio.shape[-1] else -1)
    if target_rate and rate != target_rate:
        audio, rate = resample(audio, rate, target_rate), target_rate
    return audio, rate


def resample(audio, rate: int, target_rate: int):
    """Rééchantillonnage (scipy s'il est présent, sinon interpolation linéaire)."""
    import numpy as np
    if rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    try:
        from scipy.signal import resample_poly
        g = math.gcd(int(rate), int(target_rate))
        return resample_poly(audio, target_rate // g, rate // g).astype(np.float32)
    except ImportError:
        n = int(round(len(audio) * target_rate / rate))
        return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


def telephone_channel(audio, sample_rate: int):
    """Simule un canal téléphonique : 8 kHz, bande 300-3400 Hz, compression µ-law 8 bits, puis retour à `sample_rate`.

    Les corpus publics sont enregistrés en large bande : sans cette dégradation, un modèle de transcription y paraît
    meilleur qu'au téléphone."""
    import numpy as np
    x = resample(audio, sample_rate, 8000)
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / 8000)
    spectrum[(freqs < 300) | (freqs > 3400)] = 0
    x = np.clip(np.fft.irfft(spectrum, n=len(x)), -1, 1)
    mu = 255.0
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y = np.round((y + 1) / 2 * mu) / mu * 2 - 1  # quantification 8 bits
    x = np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu)) / mu)
    return resample(x.astype(np.float32), 8000, sample_rate)


# --------------------------------------------------------------------------- données internes

_FORMATS = (".csv", ".tsv", ".txt", ".parquet", ".jsonl", ".json", ".xlsx")


def _data_files(path: Path) -> list[Path]:
    """Le fichier, ou les fichiers de données d'un dossier (récursivement, même structure attendue)."""
    if path.is_file():
        if path.suffix.lower() not in _FORMATS:
            raise ValueError(f"format non pris en charge : {path.suffix} (attendus : {', '.join(_FORMATS)})")
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _FORMATS and not p.name.startswith("."))
        if not files:
            raise ValueError(f"aucun fichier de données ({', '.join(_FORMATS)}) dans {path}")
        return files
    raise FileNotFoundError(f"jeu introuvable : {path}")


def _read_file(path: Path, csv: dict):
    import pandas as pd
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".xlsx":
        return pd.read_excel(path)  # openpyxl requis
    # Export d'entreprise : UTF-8 puis Windows-1252 ; séparateur compté sur la ligne d'en-tête (« ; » fréquent), « , »
    # si aucun (une seule colonne : le détecteur de la bibliothèque standard s'y trompe).
    for encoding in ([csv["encoding"]] if csv.get("encoding") else ["utf-8-sig", "cp1252"]):
        try:
            with open(path, encoding=encoding) as f:
                header = f.readline()
            sep = csv.get("sep") or ("\t" if suffix == ".tsv" else max((";", ",", "\t", "|"), key=header.count) if any(
                x in header for x in ";,\t|") else ",")
            return pd.read_csv(path, sep=sep, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path.name} : encodage non reconnu (indique PRIVATE_CONFIG['csv']['encoding'])")


def _write_fake_dataset(df, folder: Path, path_columns: list[str], sample_rate: int) -> dict:
    """Mode factice : le jeu synthétique est écrit comme un vrai export (dossier de deux fichiers : CSV « ; » en
    Windows-1252, ou parquet si une colonne est composite ; colonnes renommées ; audio en fichiers WAV à chemins
    relatifs), puis relu par le code de production : la lecture, la correspondance des colonnes et la résolution des
    chemins sont ainsi exercées. Rend la correspondance des colonnes à utiliser."""
    import numpy as np
    import pandas as pd
    (folder / "audio").mkdir(parents=True, exist_ok=True)
    df = df.copy()
    for column in path_columns:
        if column in df.columns:
            def to_file(value, i):
                if isinstance(value, (bytes, bytearray)):
                    content = bytes(value)
                elif isinstance(value, np.ndarray):
                    content = wav_bytes(value, sample_rate)
                else:
                    return value
                (folder / "audio" / f"{column}_{i}.wav").write_bytes(content)
                return f"audio/{column}_{i}.wav"
            df[column] = [to_file(v, i) for i, v in enumerate(df[column])]
    df = df.rename(columns={c: f"src_{c}" for c in df.columns})
    half = len(df) // 2
    composite = df.apply(lambda col: col.map(lambda v: isinstance(v, (list, dict, tuple, np.ndarray))).any()).any()
    for i, part in enumerate((df.iloc[:half], df.iloc[half:]), 1):
        part = pd.DataFrame(part).reset_index(drop=True)
        if composite:
            part.to_parquet(folder / f"part_{i}.parquet", index=False)
        else:
            part.to_csv(folder / f"part_{i}.csv", sep=";", index=False, encoding="cp1252", errors="replace")
    return {c[4:]: c for c in df.columns}


def load_private_data(columns: dict[str, str], make_fake_data, *, n_fake: int = 60):
    """Jeu de la phase « private » : DataFrame aux colonnes attendues, lu dans PRIVATE_CONFIG['data'] (ou PRIVATE_DATA),
    fichier ou dossier (csv, tsv, parquet, jsonl, json, xlsx).

    Avec « fake », le jeu vient de `make_fake_data(n)` (mêmes colonnes ; audio en octets WAV ou tableaux numpy à 16 kHz),
    écrit puis relu comme un vrai export. `columns` : {nom attendu: description} ; PRIVATE_CONFIG['columns'] fait
    correspondre les noms du vrai jeu. Lignes incomplètes retirées, chemins relatifs résolus, échantillon borné par
    PRIVATE_CONFIG['max_rows']. N'affiche jamais de contenu."""
    import pandas as pd
    source = _env("PRIVATE_DATA")
    mapping = dict(_CONFIG.get("columns") or {})
    path_columns = list(_CONFIG.get("path_columns") or [])
    if source == "fake":
        df = make_fake_data(n_fake)
        df = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        folder = Path(tempfile.mkdtemp(prefix="fake_dataset_"))
        mapping = _write_fake_dataset(df, folder, path_columns, int(_CONFIG.get("fake_sample_rate", 16000)))
        source = str(folder)
    files = _data_files(Path(source).expanduser())
    parts = []
    for file in files:
        part = _read_file(file, _CONFIG.get("csv") or {})
        for column in path_columns:  # chemins relatifs au fichier qui les cite
            name = mapping.get(column, column)
            if name in part.columns:
                part[name] = [str((file.parent / v).resolve()) if isinstance(v, str) and v and not Path(v).is_absolute() else v
                              for v in part[name]]
        parts.append(part)
    df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    renaming = {mapping.get(c, c): c for c in columns}
    missing = [name for name in renaming if name not in df.columns]
    if missing:
        raise ValueError(f"colonnes absentes du jeu : {missing}. Colonnes disponibles : {list(df.columns)}. Fais-les "
                         f"correspondre dans PRIVATE_CONFIG['columns'] (nom attendu -> nom du jeu) ; attendues : {columns}")
    df = df.rename(columns=renaming)[list(columns)]
    before = len(df)
    df = df.dropna(subset=list(columns)).reset_index(drop=True)
    cap = _CONFIG.get("max_rows")
    if cap and len(df) > int(cap):
        df = df.sample(n=int(cap), random_state=_RUN["seed"]).reset_index(drop=True)
    print(f"données internes : {len(df)} lignes retenues sur {before} ({len(files)} fichier(s)) ; colonnes {list(columns)} ; "
          f"empreinte {fingerprint(df)}")
    return df


def preview(df, n: int = 3):
    """Premières lignes en phase « public » ; en phase « private », seulement la structure (sauf SHOW_EXAMPLES=1)."""
    if current_phase() == "private" and os.environ.get("SHOW_EXAMPLES") != "1":
        print(f"{len(df)} lignes ; colonnes : {list(getattr(df, 'columns', []))} (contenu masqué en phase private)")
        return None
    return df.head(n) if hasattr(df, "head") else df[:n]


def fingerprint(obj) -> str:
    """Empreinte courte (sha256) d'un jeu de données, pour la traçabilité des résultats."""
    h = hashlib.sha256()
    if hasattr(obj, "to_csv"):
        h.update(obj.to_csv(index=False).encode("utf-8"))
    else:
        h.update(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- mesures

@contextlib.contextmanager
def timer(durations_ms: list | None = None):
    """Mesure une durée en millisecondes : `with timer(liste): ...` ajoute la durée à `liste`."""
    start = time.perf_counter()
    yield
    if durations_ms is not None:
        durations_ms.append((time.perf_counter() - start) * 1000)


def latency_stats(durations_ms: list[float]) -> dict:
    """Médiane (p50) et 95e centile (p95) d'une liste de durées en millisecondes."""
    if not durations_ms:
        return {}
    s = sorted(durations_ms)
    return {"p50": round(statistics.median(s), 2), "p95": round(s[min(len(s) - 1, int(math.ceil(0.95 * len(s))) - 1)], 2),
            "n": len(s)}


def bootstrap_ci(y_true, y_pred, metric, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Valeur de `metric(y_true, y_pred)` et son intervalle de confiance à 95 % par bootstrap : (valeur, bas, haut).

    Le bootstrap rééchantillonne les exemples de test avec remise : l'intervalle dit à quel point le chiffre dépend
    des exemples tirés."""
    y_true, y_pred = list(y_true), list(y_pred)
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true et y_pred doivent avoir la même longueur non nulle")
    rng = random.Random(seed)
    m = len(y_true)
    values = []
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        values.append(metric([y_true[i] for i in idx], [y_pred[i] for i in idx]))
    values.sort()
    return float(metric(y_true, y_pred)), float(values[int(alpha / 2 * n)]), float(values[min(n - 1, int((1 - alpha / 2) * n))])


def normalize_fr(text: str) -> str:
    """Normalisation pour le WER en français : minuscules, apostrophes et tirets unifiés, ponctuation retirée."""
    t = unicodedata.normalize("NFC", (text or "").lower())
    t = re.sub(r"[’`´]", "'", t)
    t = re.sub(r"(?<=\w)'(?=\w)", "' ", t)          # l'appel -> l' appel
    t = re.sub(r"[^\w\s'-]|_", " ", t)
    t = re.sub(r"(?<!\w)-|-(?!\w)", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def wer(references: list[str], hypotheses: list[str], normalize=normalize_fr) -> float:
    """Taux d'erreur mots (WER) agrégé sur le corpus : (substitutions + suppressions + insertions) / mots de référence."""
    errors = total = 0
    for ref, hyp in zip(references, hypotheses, strict=True):
        r, h = normalize(ref).split(), normalize(hyp).split()
        previous = list(range(len(h) + 1))
        for i, word_ref in enumerate(r, 1):
            current = [i] + [0] * len(h)
            for j, word_hyp in enumerate(h, 1):
                current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (word_ref != word_hyp))
            previous = current
        errors += previous[-1]
        total += len(r)
    return errors / total if total else 0.0


# --------------------------------------------------------------------------- résultats

def environment_info() -> dict:
    """Machine et versions des bibliothèques, enregistrées avec les résultats (traçabilité)."""
    from importlib import metadata
    versions = {}
    for package in ("torch", "transformers", "datasets", "sentence-transformers", "scikit-learn", "numpy", "pandas", "openai", "xgboost"):
        with contextlib.suppress(metadata.PackageNotFoundError):
            versions[package] = metadata.version(package)
    ram = None
    with contextlib.suppress(Exception):
        ram = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1)
    gpu = None
    with contextlib.suppress(ImportError, RuntimeError):
        import torch
        if torch.cuda.is_available():
            gpu = {"name": torch.cuda.get_device_name(0),
                   "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)}
    return {"python": platform.python_version(), "system": platform.platform(), "cpu": os.cpu_count(), "ram_gb": ram,
            "device": device(), "gpu": gpu, "scale": current_scale(), "versions": versions}


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def save_results(results: dict) -> dict:
    """Valide puis écrit results_<phase>.json dans le dossier des résultats. Structure attendue :

    {"primary_metric": str, "direction": "max" | "min",
     "metric_labels": {clé: libellé lisible} (facultatif),
     "dataset": {"name", "config", "split", "n_examples"},
     "approaches": [{"name", "role": "reference" | "new", "metrics": {clé: nombre, ou None si la métrique est sans objet
                                                                     pour cette approche (jamais une valeur « par construction »)},
                     "ci95": {clé: [bas, haut]} (facultatif), "latency_ms": {"p50", "p95"} (facultatif), "notes": str}]}"""
    errors = []
    primary = results.get("primary_metric")
    if results.get("direction") not in ("max", "min"):
        errors.append("direction doit valoir « max » ou « min »")
    dataset = results.get("dataset") or {}
    if not isinstance(dataset.get("n_examples"), int) or dataset["n_examples"] <= 0:
        errors.append("dataset.n_examples doit être un entier positif")
    approaches = results.get("approaches") or []
    roles = [a.get("role") for a in approaches]
    if "reference" not in roles or "new" not in roles:
        errors.append("il faut au moins une approche « reference » et une « new »")
    for a in approaches:
        metrics = a.get("metrics") or {}
        if not _is_number(metrics.get(primary)):
            errors.append(f"{a.get('name')} : métrique principale {primary!r} absente ou non numérique")
        errors += [f"{a.get('name')} : {k} = {v!r} n'est pas un nombre fini" for k, v in metrics.items()
                   if v is not None and not _is_number(v)]
        for k, ci in (a.get("ci95") or {}).items():
            if not (isinstance(ci, (list, tuple)) and len(ci) == 2 and all(_is_number(x) for x in ci) and ci[0] <= ci[1]):
                errors.append(f"{a.get('name')} : ci95[{k}] invalide")
    keys = [set((a.get("metrics") or {})) for a in approaches]
    if keys and any(k != keys[0] for k in keys):
        errors.append("toutes les approches doivent rapporter les mêmes métriques")
    if errors:
        raise ValueError("résultats invalides : " + " ; ".join(errors))
    output = {**results, "phase": current_phase(), "scale": current_scale(), "seed": _RUN["seed"],
              "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "environment": environment_info(),
              "service_calls": {"n": len(SERVICE_CALLS), "failures": sum(1 for c in SERVICE_CALLS if not c.get("ok", True)),
                                "seconds": round(sum(c["seconds"] for c in SERVICE_CALLS), 1)}}
    path = _RUN["output_dir"] / f"results_{current_phase()}.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"résultats enregistrés : {_shown_path(path)}")
    return output


def _label(results: dict, key: str) -> str:
    return (results.get("metric_labels") or {}).get(key) or key


def _shown_path(path: Path) -> str:
    """Chemin affiché dans les sorties : relatif au dossier courant quand c'est possible."""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def results_table(results: dict):
    """Tableau comparatif : une ligne par approche, chaque métrique avec son intervalle de confiance."""
    rows = []
    for a in results["approaches"]:
        row = {"approche": a["name"], "rôle": "référence" if a["role"] == "reference" else "nouvelle"}
        for key, value in a["metrics"].items():
            ci = (a.get("ci95") or {}).get(key)
            row[_label(results, key)] = ("—" if value is None else f"{value:.4g}") + (f" [{ci[0]:.4g} ; {ci[1]:.4g}]" if ci else "")
        if a.get("latency_ms"):
            row["latence p50 (ms)"] = a["latency_ms"].get("p50")
        rows.append(row)
    try:
        import pandas as pd
        return pd.DataFrame(rows).set_index("approche")
    except ImportError:
        return rows


# --------------------------------------------------------------------------- figures

COLORS = {"reference": "#7c8591", "new": "#2563eb", "accent": "#f59e0b", "good": "#16a34a", "bad": "#dc2626"}


def use_plot_style() -> None:
    """Style commun des figures : lisible sur écran et dans le README (police, grille légère, sans cadre superflu)."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 130, "figure.figsize": (8, 4.5), "font.size": 11, "axes.titlesize": 13,
        "axes.titleweight": "bold", "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
        "axes.prop_cycle": plt.cycler(color=[COLORS["reference"], COLORS["new"], COLORS["accent"], "#9333ea", "#0891b2"]),
    })


def save_figure(fig, name: str, caption: str) -> None:
    """Enregistre la figure (figures/<name>.png), l'affiche avec sa légende et la ferme ; ne renvoie rien.

    `name` : identifiant court en minuscules (lettres, chiffres, « _ ») ; `caption` : ce que la figure montre et comment
    la lire, en une ou deux phrases. Les figures enregistrées ainsi illustrent aussi le README du dépôt."""
    import matplotlib.pyplot as plt
    if not re.fullmatch(r"[a-z0-9_]{2,40}", name):
        raise ValueError(f"nom de figure invalide : {name!r} (minuscules, chiffres et « _ »)")
    folder = _RUN["output_dir"] / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    try:
        from IPython.display import Image, Markdown, display
        display(Image(filename=str(path)), metadata={"figure": {"name": name, "caption": caption}})
        display(Markdown(f"*Figure — {caption}*"))
    except ImportError:
        print(f"figure enregistrée : {_shown_path(path)} — {caption}")


def plot_comparison(results: dict, metrics: list[str] | None = None, name: str = "comparison",
                    caption: str | None = None) -> None:
    """Figure de synthèse : une barre par approche et par métrique, avec l'intervalle de confiance à 95 % quand il
    existe. Par défaut, la métrique principale puis jusqu'à trois autres métriques. Une métrique sans objet pour une
    approche (None) n'a pas de barre pour elle."""
    import matplotlib.pyplot as plt
    primary = results["primary_metric"]
    all_keys = list(results["approaches"][0]["metrics"])
    keys = metrics or [primary, *[k for k in all_keys if k != primary][:3]]
    use_plot_style()
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys) + 1, 4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        shown = [a for a in results["approaches"] if _is_number(a["metrics"].get(key))]
        names = [a["name"] for a in shown]
        values = [a["metrics"][key] for a in shown]
        colors = [COLORS["new"] if a["role"] == "new" else COLORS["reference"] for a in shown]
        errors = [[0, 0]] * len(values)
        cis = [(a.get("ci95") or {}).get(key) for a in shown]
        if any(cis):
            errors = [[v - ci[0], ci[1] - v] if ci else [0, 0] for v, ci in zip(values, cis)]
        bars = ax.bar(range(len(values)), values, color=colors, yerr=list(zip(*errors)), capsize=5)
        ax.bar_label(bars, labels=[f"{v:.3g}" for v in values], padding=3, fontsize=10)
        ax.set_xticks(range(len(values)), [n if len(n) <= 18 else n[:17] + "…" for n in names], rotation=15, ha="right")
        title = _label(results, key)
        ax.set_title(title + (" ★" if key == primary else ""), fontsize=11)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Comparaison des approches (barres : intervalle de confiance à 95 %)", fontsize=12, y=1.02)
    fig.tight_layout()
    default = (f"Valeur de chaque métrique par approche (★ : métrique principale, "
               f"{'à maximiser' if results['direction'] == 'max' else 'à minimiser'}) ; les barres d'erreur donnent "
               f"l'intervalle de confiance à 95 % sur {results['dataset']['n_examples']} exemples de test.")
    save_figure(fig, name, caption or default)
