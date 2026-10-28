# Aide commune des notebooks de POC de la veille IA (insérée telle quelle par le moteur : ne pas modifier ici).
# Dépendances : bibliothèque standard ; numpy, pandas, openai, soundfile seulement pour les fonctions qui en ont besoin.
# Convention : aucune clé ni URL dans le notebook. Tout vient de variables d'environnement (voir afficher_configuration).
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
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

PHASES = ("publique", "banque")
TAILLES = ("sandbox", "complete")
_PREFIXES = {"LLMAAS": "service LLM (convention OpenAI)", "STTAAS": "service STT (convention OpenAI)"}
# Points d'accès publics : jamais de données bancaires vers eux, sauf autorisation explicite.
_PUBLICS = ("api.openai.com", "openai.azure.com", "api.anthropic.com", "api.mistral.ai", "generativelanguage.googleapis.com",
            "api-inference.huggingface.co", "router.huggingface.co", "api.together.xyz", "api.groq.com", "openrouter.ai")
JOURNAL_APPELS: list[dict] = []


def _phase() -> str:
    p = globals().get("PHASE", os.environ.get("POC_PHASE", "publique"))
    if p not in PHASES:
        raise ValueError(f"PHASE doit valoir {PHASES}, pas {p!r}")
    return p


def _taille() -> str:
    t = globals().get("TAILLE", os.environ.get("POC_TAILLE", "sandbox"))
    if t not in TAILLES:
        raise ValueError(f"TAILLE doit valoir {TAILLES}, pas {t!r}")
    return t


def fixer_graine(graine: int = 42) -> None:
    """Graine unique pour random, numpy et torch (si présents) : résultats reproductibles."""
    random.seed(graine)
    os.environ["PYTHONHASHSEED"] = str(graine)
    with contextlib.suppress(ImportError):
        import numpy as np
        np.random.seed(graine)
    with contextlib.suppress(ImportError):
        import torch
        torch.manual_seed(graine)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(graine)


def appareil() -> str:
    """« cuda », « mps » ou « cpu » selon ce qui est disponible."""
    with contextlib.suppress(ImportError):
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def taille_echantillon(sandbox: int, complete: int | None = None) -> int | None:
    """Nombre d'exemples à utiliser : `sandbox` en TAILLE « sandbox », `complete` (None = tout) sinon."""
    return sandbox if _taille() == "sandbox" else complete


def selon_taille(sandbox, complete):
    """Valeur selon TAILLE : modèle substitut, nombre d'époques, de pas ou d'exemples (`sandbox` en « sandbox »)."""
    return sandbox if _taille() == "sandbox" else complete


# --------------------------------------------------------------------------- configuration et sécurité

def _variable(nom: str, obligatoire: bool = True, secrete: bool = False) -> str | None:
    """Variable d'environnement ; pour un secret, aussi <NOM>_FILE (fichier de secret monté), puis saisie masquée."""
    v = os.environ.get(nom)
    if v:
        return v
    if secrete and os.environ.get(f"{nom}_FILE"):
        fichier = Path(os.environ[f"{nom}_FILE"]).expanduser()
        if not fichier.is_file():
            raise RuntimeError(f"{nom}_FILE désigne un fichier absent : {fichier}")
        return fichier.read_text(encoding="utf-8").strip()
    if obligatoire and secrete and sys.stdin and sys.stdin.isatty():
        return getpass.getpass(f"{nom} : ")
    if obligatoire:
        raise RuntimeError(f"Variable d'environnement {nom} absente{' (ni ' + nom + '_FILE)' if secrete else ''}. Renseigne-la "
                           "dans la cellule CONFIG_BANQUE ou dans l'environnement du notebook (variables du projet ou du job) ; "
                           "une clé, jamais dans le notebook.")
    return None


def verifier_point_acces(url: str) -> str:
    """Refuse en phase « banque » un point d'accès public (données bancaires) sauf POC_AUTORISER_ENDPOINT_PUBLIC=1."""
    hote = (urlparse(url).hostname or "").lower()
    if not hote:
        raise ValueError(f"URL de service invalide : {url!r}")
    if urlparse(url).scheme != "https" and hote not in ("localhost", "127.0.0.1") and _phase() == "banque":
        raise ValueError(f"En phase banque, le service doit être en HTTPS : {hote}")
    if _phase() == "banque" and any(hote == p or hote.endswith("." + p) for p in _PUBLICS) \
            and os.environ.get("POC_AUTORISER_ENDPOINT_PUBLIC") != "1":
        raise PermissionError(f"{hote} est un service public : refusé en phase banque (données sensibles).")
    return url


def config_service(prefixe: str = "LLMAAS") -> dict:
    """URL, clé et modèle d'un service à la convention OpenAI : variables <PREFIXE>_BASE_URL, _API_KEY, _MODEL."""
    url = verifier_point_acces(_variable(f"{prefixe}_BASE_URL"))
    return {"base_url": url.rstrip("/"), "api_key": _variable(f"{prefixe}_API_KEY", secrete=True),
            "model": _variable(f"{prefixe}_MODEL")}


def client_openai(prefixe: str = "LLMAAS"):
    """Client `openai` vers le service <PREFIXE> (proxy et certificats d'entreprise : HTTPS_PROXY, SSL_CERT_FILE ;
    en-têtes propres à une passerelle : <PREFIXE>_EN_TETES, objet JSON)."""
    from openai import OpenAI
    c = config_service(prefixe)
    en_tetes = json.loads(os.environ.get(f"{prefixe}_EN_TETES") or "{}")
    return OpenAI(base_url=c["base_url"], api_key=c["api_key"], timeout=float(os.environ.get("POC_TIMEOUT_SECONDES", "120")),
                  max_retries=int(os.environ.get("POC_TENTATIVES", "3")), default_headers=en_tetes or None)


def afficher_configuration() -> None:
    """Affiche la configuration utile, sans jamais montrer une clé."""
    print(f"PHASE={_phase()}  TAILLE={_taille()}  appareil={appareil()}  python={platform.python_version()}")
    for p, nom in _PREFIXES.items():
        url, modele = os.environ.get(f"{p}_BASE_URL"), os.environ.get(f"{p}_MODEL")
        cle = "renseignée" if os.environ.get(f"{p}_API_KEY") else "absente"
        print(f"{nom:34} {p}_BASE_URL={urlparse(url).hostname if url else '—'}  {p}_MODEL={modele or '—'}  clé {cle}")
    print(f"données banque : {'factices' if os.environ.get('BANQUE_DONNEES') == 'factice' else ('renseignées' if os.environ.get('BANQUE_DONNEES') else '—')}"
          f"  modèles locaux : {os.environ.get('BANQUE_MODELES_DIR') or '—'}")


def preparer_environnement() -> None:
    """Désactive la télémétrie ; en phase banque avec modèles locaux, interdit les téléchargements."""
    for k in ("HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK", "ANONYMIZED_TELEMETRY"):
        os.environ.setdefault(k, "1")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if _phase() == "banque" and os.environ.get("BANQUE_MODELES_DIR"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def chemin_modele(hub_id: str) -> str:
    """À utiliser pour charger tout modèle, dans les deux phases : l'identifiant Hugging Face, ou son dossier local
    sous BANQUE_MODELES_DIR (org--nom ou org/nom), seul accessible dans un environnement sans Internet."""
    racine = os.environ.get("BANQUE_MODELES_DIR")
    if racine:
        for d in (Path(racine) / hub_id.replace("/", "--"), Path(racine) / hub_id):
            if d.is_dir():
                return str(d)
        if _phase() == "banque":
            raise FileNotFoundError(f"{hub_id} absent de BANQUE_MODELES_DIR ({racine}) : copie ses poids dans "
                                    f"{Path(racine) / hub_id.replace('/', '--')} (script telecharger_modeles.py du dépôt).")
    return hub_id


# --------------------------------------------------------------------------- environnement bancaire

_CONFIG: dict = {}   # CONFIG_BANQUE appliquée (correspondance des colonnes, chemins, modèles, options de lecture)
_VARIABLES_SERVICE = {"base_url": "BASE_URL", "model": "MODEL", "embeddings_model": "EMBEDDINGS_MODEL",
                      "cle_fichier": "API_KEY_FILE", "en_tetes": "EN_TETES"}


def appliquer_config_banque(config: dict) -> list[str]:
    """Applique la cellule CONFIG_BANQUE (générée avec le notebook, seule à adapter) puis vérifie l'environnement.
    En phase publique : ne fait rien. Rend la liste des problèmes (vide si tout est prêt) ; arrête le notebook sur un
    problème bloquant, avec ce qu'il faut corriger."""
    _CONFIG.clear()
    _CONFIG.update(config)
    if _phase() != "banque":
        return []
    for cle, variable in (("donnees", "BANQUE_DONNEES"), ("modeles_dir", "BANQUE_MODELES_DIR"), ("certificats", "SSL_CERT_FILE")):
        if config.get(cle):
            os.environ[variable] = str(config[cle])
    for prefixe in _PREFIXES:
        for cle, suffixe in _VARIABLES_SERVICE.items():
            v = (config.get(prefixe.lower()) or {}).get(cle)
            if v:
                os.environ[f"{prefixe}_{suffixe}"] = json.dumps(v) if isinstance(v, dict) else str(v)
    problemes = verifier_environnement_banque()
    if any(p.startswith("BLOQUANT") for p in problemes):
        raise RuntimeError("Environnement bancaire incomplet :\n" + "\n".join(problemes))
    return problemes


def verifier_environnement_banque() -> list[str]:
    """Diagnostic avant de lancer la phase banque : jeu lisible et colonnes trouvées, modèles présents en local si
    BANQUE_MODELES_DIR est renseigné, services joignables avec leur clé. Affiche une liste de contrôle, sans contenu."""
    problemes, lignes = [], []

    def ok(message):
        lignes.append(f"  [ok] {message}")

    def ko(message, bloquant=True):
        problemes.append(("BLOQUANT : " if bloquant else "à vérifier : ") + message)
        lignes.append(f"  [{'MANQUE' if bloquant else 'à vérifier'}] {message}")

    src = os.environ.get("BANQUE_DONNEES", "")
    if not src:
        ko("BANQUE_DONNEES non renseigné (fichier ou dossier du jeu, ou « factice »)")
    elif src == "factice":
        ok("données factices (test du code, aucun résultat interprétable)")
    else:
        try:
            fichiers = _fichiers_jeu(Path(src).expanduser())
            ok(f"jeu : {len(fichiers)} fichier(s) dans {Path(src).name}")
        except (FileNotFoundError, ValueError) as e:
            ko(str(e))
    modeles = (_CONFIG.get("modeles") or {}).get(_taille(), [])
    racine = os.environ.get("BANQUE_MODELES_DIR")
    if racine:
        absents = [m for m in modeles if not any((Path(racine) / x).is_dir() for x in (m.replace("/", "--"), m))]
        if absents:
            ko(f"modèles absents de {racine} : {absents} (script telecharger_modeles.py)")
        else:
            ok(f"{len(modeles)} modèle(s) présents en local, sans accès à Internet")
    elif modeles:
        ko(f"BANQUE_MODELES_DIR vide : les modèles {modeles} seront téléchargés depuis Hugging Face", bloquant=False)
    for prefixe, nom in _PREFIXES.items():
        if prefixe.lower() not in _CONFIG:
            continue
        try:
            c = config_service(prefixe)
        except (RuntimeError, ValueError, PermissionError) as e:
            ko(f"{nom} : {e}")
            continue
        try:
            client_openai(prefixe).models.list()
            ok(f"{nom} joignable ({urlparse(c['base_url']).hostname}, modèle {c['model']})")
        except Exception as e:  # certaines passerelles n'exposent pas /models : à vérifier, pas bloquant
            ko(f"{nom} : liste des modèles indisponible ({type(e).__name__}) ; le premier appel le dira", bloquant=False)
    print("Environnement bancaire :\n" + "\n".join(lignes))
    return problemes


# --------------------------------------------------------------------------- appels aux services (convention OpenAI)

def _noter(service: str, debut: float, **kw) -> None:
    JOURNAL_APPELS.append({"service": service, "secondes": round(time.time() - debut, 3), **kw})


def _json_tolerant(texte: str):
    texte = (texte or "").strip()
    with contextlib.suppress(json.JSONDecodeError):
        return json.loads(texte)
    m = re.search(r"\{.*\}|\[.*\]", texte, re.S)
    if m:
        with contextlib.suppress(json.JSONDecodeError):
            return json.loads(m.group(0))
    return None


def _schema_strict(schema) -> bool:
    """Vrai si le schéma respecte le mode strict d'OpenAI (objets fermés, toutes les propriétés requises)."""
    if isinstance(schema, list):
        return all(_schema_strict(x) for x in schema)
    if not isinstance(schema, dict):
        return True
    props = schema.get("properties", {})
    if (schema.get("type") == "object" or props) and (
            schema.get("additionalProperties") is not False or set(schema.get("required", [])) != set(props)):
        return False
    enfants = [*props.values(), *schema.get("$defs", {}).values(), *schema.get("anyOf", [])]
    if "items" in schema:
        enfants.append(schema["items"])
    return all(_schema_strict(e) for e in enfants)


def llm_chat(messages: list[dict], *, schema: dict | None = None, temperature: float = 0.0, max_tokens: int = 512,
             prefixe: str = "LLMAAS", client=None, modele: str | None = None):
    """Appel chat/completions. Avec `schema` (JSON Schema), renvoie l'objet décodé (None si la sortie est illisible) :
    sortie structurée si le service la prend en charge (mode strict si le schéma s'y prête), sinon consigne dans le
    prompt et décodage tolérant."""
    client = client or client_openai(prefixe)
    modele = modele or config_service(prefixe)["model"]
    kw = {"model": modele, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    t = time.time()
    if schema is not None:
        try:
            r = client.chat.completions.create(**kw, response_format={"type": "json_schema", "json_schema": {
                "name": "reponse", "schema": schema, "strict": _schema_strict(schema)}})
        except Exception as e:  # service sans sortie structurée : consigne explicite et décodage tolérant
            if not re.search(r"response_format|json_schema|unsupported|not supported|400", str(e), re.I):
                raise
            consigne = {"role": "system", "content": "Réponds uniquement par un objet JSON conforme à ce schéma : " + json.dumps(schema, ensure_ascii=False)}
            r = client.chat.completions.create(**{**kw, "messages": [consigne, *messages]})
        texte = r.choices[0].message.content or ""
        obj = _json_tolerant(texte)
        _noter(prefixe, t, modele=modele, ok=obj is not None, jetons=getattr(r.usage, "total_tokens", None) if r.usage else None)
        return obj
    r = client.chat.completions.create(**kw)
    _noter(prefixe, t, modele=modele, ok=True, jetons=getattr(r.usage, "total_tokens", None) if r.usage else None)
    return r.choices[0].message.content or ""


def embeddings(textes: list[str], *, prefixe: str = "LLMAAS", modele: str | None = None, lot: int = 64, client=None):
    """Embeddings par lots (modèle : <PREFIXE>_EMBEDDINGS_MODEL par défaut). Renvoie une liste de vecteurs."""
    client = client or client_openai(prefixe)
    modele = modele or _variable(f"{prefixe}_EMBEDDINGS_MODEL")
    out = []
    for i in range(0, len(textes), lot):
        t = time.time()
        r = client.embeddings.create(model=modele, input=textes[i:i + lot])
        out += [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
        _noter(f"{prefixe}:embeddings", t, modele=modele, ok=True, n=len(textes[i:i + lot]))
    return out


def audio_wav(audio, sr: int) -> bytes:
    """Tableau numpy (mono, flottant) -> octets WAV 16 bits."""
    import numpy as np
    import wave
    a = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes((a * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def stt_transcrire(audio, sr: int, *, langue: str = "fr", prefixe: str = "STTAAS", client=None, modele: str | None = None) -> str:
    """Transcription par audio/transcriptions (convention OpenAI) d'un signal numpy."""
    client = client or client_openai(prefixe)
    modele = modele or config_service(prefixe)["model"]
    t = time.time()
    r = client.audio.transcriptions.create(model=modele, file=("audio.wav", audio_wav(audio, sr), "audio/wav"), language=langue)
    _noter(prefixe, t, modele=modele, ok=True, duree_audio=round(len(audio) / sr, 2))
    return getattr(r, "text", r if isinstance(r, str) else "")


def decoder_audio(x, sr_cible: int | None = 16000):
    """Signal mono float32 et fréquence depuis une colonne audio de `datasets` (dict array/sampling_rate,
    dict bytes/path, ou décodeur torchcodec) ou un chemin de fichier (jeu bancaire), rééchantillonné à `sr_cible` si
    demandé. Sans soundfile, les WAV sont lus par la bibliothèque standard."""
    import numpy as np
    if hasattr(x, "get_all_samples"):  # datasets >= 4 : torchcodec AudioDecoder
        s = x.get_all_samples()
        a, sr = s.data.numpy(), int(s.sample_rate)
    elif isinstance(x, dict) and x.get("array") is not None:
        a, sr = np.asarray(x["array"]), int(x["sampling_rate"])
    elif isinstance(x, (dict, str, Path)) and (not isinstance(x, dict) or x.get("bytes") or x.get("path")):
        source = io.BytesIO(x["bytes"]) if isinstance(x, dict) and x.get("bytes") else (x["path"] if isinstance(x, dict) else x)
        try:
            import soundfile as sf
            a, sr = sf.read(source, dtype="float32")
        except ImportError:  # sans libsndfile (environnement verrouillé) : WAV PCM par la bibliothèque standard
            import wave
            with wave.open(source if isinstance(source, io.BytesIO) else str(source), "rb") as w:
                sr, largeur, canaux = w.getframerate(), w.getsampwidth(), w.getnchannels()
                brut = np.frombuffer(w.readframes(w.getnframes()), dtype={1: np.uint8, 2: "<i2", 4: "<i4"}[largeur])
            a = (brut.astype(np.float32) - (128 if largeur == 1 else 0)) / float(2 ** (8 * largeur - 1))
            a = a.reshape(-1, canaux).T if canaux > 1 else a
    else:
        raise TypeError(f"format audio non reconnu : {type(x)}")
    a = np.asarray(a, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(axis=0 if a.shape[0] < a.shape[-1] else -1)
    if sr_cible and sr != sr_cible:
        a, sr = reechantillonner(a, sr, sr_cible), sr_cible
    return a, sr


def reechantillonner(audio, sr: int, sr_cible: int):
    """Rééchantillonnage (scipy si présent, sinon interpolation linéaire)."""
    import numpy as np
    if sr == sr_cible:
        return np.asarray(audio, dtype=np.float32)
    try:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), int(sr_cible))
        return resample_poly(audio, sr_cible // g, sr // g).astype(np.float32)
    except ImportError:
        n = int(round(len(audio) * sr_cible / sr))
        return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


def telephonie(audio, sr: int):
    """Simule un canal téléphonique : 8 kHz, bande 300-3400 Hz, compression µ-law 8 bits, puis retour à `sr`.
    Les corpus publics sont en large bande : sans cette dégradation, le STT y paraît meilleur qu'en centre d'appels."""
    import numpy as np
    x = reechantillonner(audio, sr, 8000)
    spectre = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / 8000)
    spectre[(f < 300) | (f > 3400)] = 0
    x = np.clip(np.fft.irfft(spectre, n=len(x)), -1, 1)
    mu = 255.0
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y = np.round((y + 1) / 2 * mu) / mu * 2 - 1  # quantification 8 bits
    x = np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu)) / mu)
    return reechantillonner(x.astype(np.float32), 8000, sr)


# --------------------------------------------------------------------------- données

_FORMATS = (".csv", ".tsv", ".txt", ".parquet", ".jsonl", ".json", ".xlsx")


def _fichiers_jeu(chemin: Path) -> list[Path]:
    """Le fichier, ou les fichiers de données d'un dossier (récursivement, même structure attendue)."""
    if chemin.is_file():
        if chemin.suffix.lower() not in _FORMATS:
            raise ValueError(f"format non pris en charge : {chemin.suffix} (attendus : {', '.join(_FORMATS)})")
        return [chemin]
    if chemin.is_dir():
        fichiers = sorted(p for p in chemin.rglob("*") if p.is_file() and p.suffix.lower() in _FORMATS and not p.name.startswith("."))
        if not fichiers:
            raise ValueError(f"aucun fichier de données ({', '.join(_FORMATS)}) dans {chemin}")
        return fichiers
    raise FileNotFoundError(f"jeu introuvable : {chemin}")


def _lire_fichier(p: Path, csv: dict):
    import pandas as pd
    s = p.suffix.lower()
    if s == ".parquet":
        return pd.read_parquet(p)
    if s == ".jsonl":
        return pd.read_json(p, lines=True)
    if s == ".json":
        return pd.read_json(p)
    if s == ".xlsx":
        return pd.read_excel(p)  # openpyxl requis
    # CSV d'entreprise : encodage UTF-8 puis Windows-1252 ; séparateur compté sur la ligne d'en-tête (« ; » fréquent),
    # « , » si aucun (une seule colonne : le détecteur de la bibliothèque standard s'y trompe).
    for encodage in ([csv["encodage"]] if csv.get("encodage") else ["utf-8-sig", "cp1252"]):
        try:
            with open(p, encoding=encodage) as f:
                entete = f.readline()
            sep = csv.get("sep") or ("\t" if s == ".tsv" else max((";", ",", "\t", "|"), key=entete.count) if any(
                x in entete for x in ";,\t|") else ",")
            return pd.read_csv(p, sep=sep, encoding=encodage)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{p.name} : encodage non reconnu (indique CONFIG_BANQUE['csv']['encodage'])")


def _materialiser_factice(df, dossier: Path, chemins: list[str], sr: int):
    """Mode factice : le jeu synthétique est écrit comme le serait un vrai jeu (dossier de deux fichiers : CSV « ; »
    en Windows-1252, ou parquet si une colonne est composite ; colonnes renommées ; audio en fichiers WAV à chemins
    relatifs), puis relu par le code de production : la phase banque du sandbox exerce ainsi la lecture, la
    correspondance des colonnes et la résolution des chemins."""
    import numpy as np
    import pandas as pd
    (dossier / "audio").mkdir(parents=True, exist_ok=True)
    df = df.copy()
    for c in chemins:
        if c in df.columns:
            def ecrire(v, i):
                if isinstance(v, (bytes, bytearray)):
                    donnees = bytes(v)
                elif isinstance(v, np.ndarray):
                    donnees = audio_wav(v, sr)
                else:
                    return v
                (dossier / "audio" / f"{c}_{i}.wav").write_bytes(donnees)
                return f"audio/{c}_{i}.wav"
            df[c] = [ecrire(v, i) for i, v in enumerate(df[c])]
    df = df.rename(columns={c: f"src_{c}" for c in df.columns})
    moitie = len(df) // 2
    # Export bancaire typique : CSV « ; » en Windows-1252 ; colonnes non scalaires (listes, tableaux) : parquet.
    composite = df.apply(lambda col: col.map(lambda v: isinstance(v, (list, dict, tuple, np.ndarray))).any()).any()
    for i, part in enumerate((df.iloc[:moitie], df.iloc[moitie:]), 1):
        part = pd.DataFrame(part).reset_index(drop=True)
        if composite:
            part.to_parquet(dossier / f"partie_{i}.parquet", index=False)
        else:
            part.to_csv(dossier / f"partie_{i}.csv", sep=";", index=False, encoding="cp1252", errors="replace")
    return {c[4:]: c for c in df.columns}


def donnees_banque(colonnes: dict[str, str], generateur_factice, *, n_factice: int = 60):
    """Jeu bancaire de la phase 2 (DataFrame aux colonnes attendues). Source : CONFIG_BANQUE['donnees'] ou
    BANQUE_DONNEES, fichier ou dossier (csv, tsv, parquet, jsonl, json, xlsx), ou « factice » (jeu produit par
    `generateur_factice(n)` : mêmes colonnes ; audio en octets WAV ou tableaux numpy à 16 kHz). `colonnes` :
    {nom attendu: description} ; CONFIG_BANQUE['colonnes'] fait correspondre les noms du vrai jeu. Lignes vides
    retirées, chemins relatifs résolus, échantillon borné par CONFIG_BANQUE['max_lignes']. N'affiche jamais de contenu."""
    import pandas as pd
    src = _variable("BANQUE_DONNEES")
    correspondance = dict(_CONFIG.get("colonnes") or {})
    chemins = list(_CONFIG.get("chemins") or [])
    if src == "factice":
        df = generateur_factice(n_factice)
        df = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        import tempfile
        dossier = Path(tempfile.mkdtemp(prefix="banque_factice_"))
        correspondance = _materialiser_factice(df, dossier, chemins, int(_CONFIG.get("sr_factice", 16000)))
        src = str(dossier)
    racine = Path(src).expanduser()
    fichiers = _fichiers_jeu(racine)
    morceaux = []
    for f in fichiers:
        m = _lire_fichier(f, _CONFIG.get("csv") or {})
        for c in chemins:  # chemins relatifs au fichier qui les cite
            source = correspondance.get(c, c)
            if source in m.columns:
                m[source] = [str((f.parent / v).resolve()) if isinstance(v, str) and v and not Path(v).is_absolute() else v
                             for v in m[source]]
        morceaux.append(m)
    df = pd.concat(morceaux, ignore_index=True) if len(morceaux) > 1 else morceaux[0]
    renommage = {correspondance.get(c, c): c for c in colonnes}
    manquantes = [source for source, c in renommage.items() if source not in df.columns]
    if manquantes:
        raise ValueError(f"colonnes absentes du jeu bancaire : {manquantes}. Colonnes disponibles : {list(df.columns)}. "
                         f"Fais-les correspondre dans CONFIG_BANQUE['colonnes'] (nom attendu -> nom du jeu) ; attendues : {colonnes}")
    df = df.rename(columns=renommage)[list(colonnes)]
    avant = len(df)
    df = df.dropna(subset=list(colonnes)).reset_index(drop=True)
    plafond = _CONFIG.get("max_lignes")
    if plafond and len(df) > int(plafond):
        df = df.sample(n=int(plafond), random_state=int(globals().get("GRAINE", 42))).reset_index(drop=True)
    print(f"données banque : {len(df)} lignes retenues sur {avant} ({len(fichiers)} fichier(s)) ; colonnes {list(colonnes)} ; "
          f"empreinte {empreinte(df)}")
    return df


def apercu(df, n: int = 3):
    """Exemples en phase publique ; en phase banque, seulement la structure (sauf POC_AFFICHER_EXEMPLES=1)."""
    if _phase() == "banque" and os.environ.get("POC_AFFICHER_EXEMPLES") != "1":
        print(f"{len(df)} lignes ; colonnes : {list(getattr(df, 'columns', []))} (contenu masqué en phase banque)")
        return None
    return df.head(n) if hasattr(df, "head") else df[:n]


def empreinte(obj) -> str:
    """Empreinte courte (sha256) d'un jeu de données, pour la traçabilité des résultats."""
    h = hashlib.sha256()
    if hasattr(obj, "to_csv"):
        h.update(obj.to_csv(index=False).encode("utf-8"))
    else:
        h.update(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- mesures

@contextlib.contextmanager
def chrono(mesures: list | None = None):
    """Mesure une durée en millisecondes : `with chrono(liste): ...` ajoute la durée à `liste`."""
    t = time.perf_counter()
    yield
    if mesures is not None:
        mesures.append((time.perf_counter() - t) * 1000)


def latences(mesures_ms: list[float]) -> dict:
    """p50 et p95 (millisecondes) d'une liste de durées."""
    if not mesures_ms:
        return {}
    s = sorted(mesures_ms)
    return {"p50": round(statistics.median(s), 2), "p95": round(s[min(len(s) - 1, int(math.ceil(0.95 * len(s))) - 1)], 2), "n": len(s)}


def ic_bootstrap(y_vrai, y_pred, metrique, n: int = 1000, graine: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Valeur de `metrique(y_vrai, y_pred)` et intervalle de confiance bootstrap à 95 % : (valeur, bas, haut)."""
    y_vrai, y_pred = list(y_vrai), list(y_pred)
    if len(y_vrai) != len(y_pred) or not y_vrai:
        raise ValueError("y_vrai et y_pred doivent avoir la même longueur non nulle")
    r = random.Random(graine)
    m = len(y_vrai)
    vals = []
    for _ in range(n):
        idx = [r.randrange(m) for _ in range(m)]
        vals.append(metrique([y_vrai[i] for i in idx], [y_pred[i] for i in idx]))
    vals.sort()
    return float(metrique(y_vrai, y_pred)), float(vals[int(alpha / 2 * n)]), float(vals[min(n - 1, int((1 - alpha / 2) * n))])


def normaliser_fr(texte: str) -> str:
    """Normalisation pour le WER français : minuscules, apostrophes et tirets unifiés, ponctuation retirée."""
    t = unicodedata.normalize("NFC", (texte or "").lower())
    t = re.sub(r"[’`´]", "'", t)
    t = re.sub(r"(?<=\w)'(?=\w)", "' ", t)          # l'appel -> l' appel
    t = re.sub(r"[^\w\s'-]|_", " ", t)
    t = re.sub(r"(?<!\w)-|-(?!\w)", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def wer(references: list[str], hypotheses: list[str], normaliser=normaliser_fr) -> float:
    """Taux d'erreur mots (WER) agrégé sur le corpus : substitutions + suppressions + insertions / mots de référence."""
    erreurs = total = 0
    for ref, hyp in zip(references, hypotheses, strict=True):
        r, h = normaliser(ref).split(), normaliser(hyp).split()
        precedente = list(range(len(h) + 1))
        for i, mr in enumerate(r, 1):
            courante = [i] + [0] * len(h)
            for j, mh in enumerate(h, 1):
                courante[j] = min(precedente[j] + 1, courante[j - 1] + 1, precedente[j - 1] + (mr != mh))
            precedente = courante
        erreurs += precedente[-1]
        total += len(r)
    return erreurs / total if total else 0.0


# --------------------------------------------------------------------------- résultats

def infos_environnement() -> dict:
    """Machine et versions des bibliothèques, enregistrées avec les résultats (traçabilité, risque de modèle)."""
    from importlib import metadata
    versions = {}
    for p in ("torch", "transformers", "datasets", "sentence-transformers", "scikit-learn", "numpy", "pandas", "openai", "xgboost"):
        with contextlib.suppress(metadata.PackageNotFoundError):
            versions[p] = metadata.version(p)
    ram = None
    with contextlib.suppress(Exception):
        ram = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1)
    gpu = None
    with contextlib.suppress(ImportError, RuntimeError):
        import torch
        if torch.cuda.is_available():
            gpu = {"nom": torch.cuda.get_device_name(0), "memoire_go": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)}
    return {"python": platform.python_version(), "systeme": platform.platform(), "cpu": os.cpu_count(), "ram_go": ram,
            "appareil": appareil(), "gpu": gpu, "taille": _taille(), "versions": versions}


def _nombre_valide(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def enregistrer_resultats(resultats: dict) -> dict:
    """Valide et écrit resultats_<phase>.json (dossier POC_SORTIE). Structure attendue :
    {"metrique_principale": str, "sens": "max"|"min", "jeu": {"nom", "config", "split", "n_exemples"},
     "approches": [{"nom", "role": "reference"|"nouvelle", "metriques": {nom: nombre},
                    "ic95": {nom: [bas, haut]} (facultatif), "latence_ms": {"p50", "p95"} (facultatif), "notes": str}]}"""
    erreurs = []
    mp = resultats.get("metrique_principale")
    if resultats.get("sens") not in ("max", "min"):
        erreurs.append("sens doit valoir « max » ou « min »")
    jeu = resultats.get("jeu") or {}
    if not isinstance(jeu.get("n_exemples"), int) or jeu["n_exemples"] <= 0:
        erreurs.append("jeu.n_exemples doit être un entier positif")
    approches = resultats.get("approches") or []
    roles = [a.get("role") for a in approches]
    if "reference" not in roles or "nouvelle" not in roles:
        erreurs.append("il faut au moins une approche « reference » et une « nouvelle »")
    for a in approches:
        m = a.get("metriques") or {}
        if mp not in m:
            erreurs.append(f"{a.get('nom')} : métrique principale {mp!r} absente")
        erreurs += [f"{a.get('nom')} : {k} = {v!r} n'est pas un nombre fini" for k, v in m.items() if not _nombre_valide(v)]
        for k, ic in (a.get("ic95") or {}).items():
            if not (isinstance(ic, (list, tuple)) and len(ic) == 2 and all(_nombre_valide(x) for x in ic) and ic[0] <= ic[1]):
                erreurs.append(f"{a.get('nom')} : ic95[{k}] invalide")
    cles = [set((a.get("metriques") or {})) for a in approches]
    if cles and any(c != cles[0] for c in cles):
        erreurs.append("toutes les approches doivent rapporter les mêmes métriques")
    if erreurs:
        raise ValueError("résultats invalides : " + " ; ".join(erreurs))
    sortie = {**resultats, "phase": _phase(), "taille": _taille(), "graine": globals().get("GRAINE"),
              "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "environnement": infos_environnement(),
              "appels_services": {"n": len(JOURNAL_APPELS), "echecs": sum(1 for a in JOURNAL_APPELS if not a.get("ok", True)),
                                  "secondes": round(sum(a["secondes"] for a in JOURNAL_APPELS), 1)}}
    chemin = Path(globals().get("POC_SORTIE", os.environ.get("POC_SORTIE", "."))) / f"resultats_{_phase()}.json"
    chemin.write_text(json.dumps(sortie, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"résultats enregistrés : {chemin}")
    return sortie


def tableau_resultats(resultats: dict):
    """Tableau comparatif (pandas si présent) : une ligne par approche, métriques et intervalles."""
    lignes = []
    for a in resultats["approches"]:
        l = {"approche": a["nom"], "rôle": a["role"]}
        for k, v in a["metriques"].items():
            ic = (a.get("ic95") or {}).get(k)
            l[k] = f"{v:.4g}" + (f" [{ic[0]:.4g} ; {ic[1]:.4g}]" if ic else "")
        if a.get("latence_ms"):
            l["latence p50 (ms)"] = a["latence_ms"].get("p50")
        lignes.append(l)
    try:
        import pandas as pd
        return pd.DataFrame(lignes)
    except ImportError:
        return lignes
