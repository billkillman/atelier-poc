"""Faux service LLMaaS / STTaaS à la convention OpenAI, pour exécuter la phase « banque » d'un notebook dans le sandbox.

Il ne juge rien : il vérifie que le code appelle le service comme le ferait un vrai (authentification Bearer, routes
/v1/chat/completions, /v1/embeddings, /v1/audio/transcriptions, /v1/models) et renvoie des réponses factices mais
valides (y compris un objet conforme au JSON Schema demandé). Chaque appel est journalisé (une ligne JSON par appel).

Usage : python faux_service.py --port 8765 --journal appels.jsonl (dans les tests : demarrer(0, journal))
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CLE_ATTENDUE = "cle-factice-sandbox"


def instance(schema: dict, defs: dict | None = None, profondeur: int = 0):
    """Objet minimal conforme à un JSON Schema (enum, const, types, objets, tableaux, $ref, anyOf)."""
    defs = defs if defs is not None else (schema.get("$defs") or schema.get("definitions") or {})
    if profondeur > 12:
        return None
    if "$ref" in schema:
        return instance(defs.get(schema["$ref"].rsplit("/", 1)[-1], {}), defs, profondeur + 1)
    if "const" in schema:
        return schema["const"]
    if schema.get("enum"):
        return schema["enum"][0]
    for k in ("anyOf", "oneOf", "allOf"):
        if schema.get(k):
            return instance(schema[k][0], defs, profondeur + 1)
    t = schema.get("type", "object")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "null")
    if t == "object":
        props = schema.get("properties", {})
        requis = schema.get("required", list(props))
        return {k: instance(props[k], defs, profondeur + 1) for k in requis if k in props}
    if t == "array":
        n = max(1, schema.get("minItems", 1))
        return [instance(schema.get("items", {"type": "string"}), defs, profondeur + 1) for _ in range(n)]
    if t == "string":
        return "factice"
    if t == "integer":
        return int(schema.get("minimum", 1))
    if t == "number":
        return float(schema.get("minimum", 0.5))
    if t == "boolean":
        return True
    return None


def vecteur(texte: str, dim: int = 64) -> list[float]:
    """Embedding déterministe (même texte, même vecteur), normé."""
    octets = hashlib.sha256(texte.encode("utf-8")).digest() * (dim // 32 + 1)
    v = [(b - 127.5) / 127.5 for b in octets[:dim]]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class Gestionnaire(BaseHTTPRequestHandler):
    journal_chemin = None
    verrou = threading.Lock()

    def log_message(self, *args):  # silencieux : le journal JSON suffit
        pass

    def _repondre(self, code: int, corps: dict | str, type_contenu: str = "application/json"):
        donnees = corps.encode("utf-8") if isinstance(corps, str) else json.dumps(corps).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", type_contenu)
        self.send_header("Content-Length", str(len(donnees)))
        self.end_headers()
        self.wfile.write(donnees)

    def _noter(self, route: str, code: int, **kw):
        if not self.journal_chemin:
            return
        with self.verrou, open(self.journal_chemin, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": round(time.time(), 3), "route": route, "code": code, **kw}) + "\n")

    def _autorise(self, route: str) -> bool:
        if self.headers.get("Authorization", "") != f"Bearer {CLE_ATTENDUE}":
            self._noter(route, 401)
            self._repondre(401, {"error": {"message": "clé API absente ou invalide", "type": "invalid_request_error"}})
            return False
        return True

    def do_GET(self):
        route = self.path.split("?")[0].rstrip("/")
        if not self._autorise(route):
            return
        if route.endswith("/models"):
            self._noter(route, 200)
            return self._repondre(200, {"object": "list", "data": [{"id": "modele-factice", "object": "model", "owned_by": "sandbox"}]})
        self._noter(route, 404)
        self._repondre(404, {"error": {"message": f"route inconnue : {route}"}})

    def do_POST(self):
        route = self.path.split("?")[0].rstrip("/")
        longueur = int(self.headers.get("Content-Length") or 0)
        brut = self.rfile.read(longueur) if longueur else b""
        if not self._autorise(route):
            return
        if route.endswith("/chat/completions"):
            return self._chat(route, brut)
        if route.endswith("/embeddings"):
            return self._embeddings(route, brut)
        if route.endswith("/audio/transcriptions"):
            return self._transcription(route, brut)
        self._noter(route, 404)
        self._repondre(404, {"error": {"message": f"route inconnue : {route}"}})

    def _json(self, route: str, brut: bytes) -> dict | None:
        try:
            return json.loads(brut or b"{}")
        except json.JSONDecodeError:
            self._noter(route, 400, erreur="corps JSON invalide")
            self._repondre(400, {"error": {"message": "corps JSON invalide"}})
            return None

    def _chat(self, route: str, brut: bytes):
        req = self._json(route, brut)
        if req is None:
            return
        if not req.get("model") or not isinstance(req.get("messages"), list) or not req["messages"]:
            self._noter(route, 400, erreur="model ou messages absents")
            return self._repondre(400, {"error": {"message": "model et messages sont obligatoires"}})
        fmt = req.get("response_format") or {}
        if fmt.get("type") == "json_schema":
            contenu = json.dumps(instance((fmt.get("json_schema") or {}).get("schema") or {}), ensure_ascii=False)
        elif fmt.get("type") == "json_object":
            contenu = "{}"
        else:
            contenu = "Réponse factice du service de test."
        logprobs = None
        if req.get("logprobs"):  # comme vLLM : premier jeton « A » et alternatives (lettres) par logprob décroissante
            n_alt = max(1, min(int(req.get("top_logprobs") or 1), 20))
            alternatives = [{"token": t, "logprob": -0.2 - i, "bytes": [ord(t)]} for i, t in enumerate("ABCDEFGHIJKLMNOPQRST"[:n_alt])]
            if fmt.get("type") not in ("json_schema", "json_object"):
                contenu = "A"
            logprobs = {"content": [{"token": contenu[:1], "logprob": -0.2, "bytes": list(contenu[:1].encode()), "top_logprobs": alternatives}]}
        n_entree = sum(len(str(m.get("content", ""))) // 4 for m in req["messages"])
        self._noter(route, 200, modele=req["model"], structure=fmt.get("type"), flux=bool(req.get("stream")), logprobs=bool(logprobs))
        if req.get("stream"):
            morceaux = [{"id": "chatcmpl-factice", "object": "chat.completion.chunk", "created": int(time.time()), "model": req["model"],
                         "choices": [{"index": 0, "delta": {"role": "assistant", "content": contenu}, "finish_reason": None}]},
                        {"id": "chatcmpl-factice", "object": "chat.completion.chunk", "created": int(time.time()), "model": req["model"],
                         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
            corps = "".join(f"data: {json.dumps(m)}\n\n" for m in morceaux) + "data: [DONE]\n\n"
            return self._repondre(200, corps, "text/event-stream")
        self._repondre(200, {
            "id": "chatcmpl-factice", "object": "chat.completion", "created": int(time.time()), "model": req["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": contenu}, "logprobs": logprobs,
                         "finish_reason": "length" if req.get("max_tokens") == 1 else "stop"}],
            "usage": {"prompt_tokens": n_entree, "completion_tokens": len(contenu) // 4, "total_tokens": n_entree + len(contenu) // 4},
        })

    def _embeddings(self, route: str, brut: bytes):
        req = self._json(route, brut)
        if req is None:
            return
        entrees = req.get("input")
        if isinstance(entrees, str):
            entrees = [entrees]
        if not req.get("model") or not isinstance(entrees, list) or not entrees:
            self._noter(route, 400, erreur="model ou input absents")
            return self._repondre(400, {"error": {"message": "model et input sont obligatoires"}})
        self._noter(route, 200, modele=req["model"], n=len(entrees))
        self._repondre(200, {"object": "list", "model": req["model"],
                             "data": [{"object": "embedding", "index": i, "embedding": vecteur(str(t))} for i, t in enumerate(entrees)],
                             "usage": {"prompt_tokens": len(entrees), "total_tokens": len(entrees)}})

    def _transcription(self, route: str, brut: bytes):
        type_contenu = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in type_contenu:
            self._noter(route, 400, erreur="multipart attendu")
            return self._repondre(400, {"error": {"message": "multipart/form-data attendu"}})
        champs = dict(re.findall(rb'name="([^"]+)"\r\n\r\n([^\r]*)\r\n', brut))
        if b"model" not in champs or b'name="file"' not in brut:
            self._noter(route, 400, erreur="model ou file absents")
            return self._repondre(400, {"error": {"message": "model et file sont obligatoires"}})
        texte = "transcription factice du service de test"
        self._noter(route, 200, modele=champs[b"model"].decode("utf-8", "replace"), octets=len(brut))
        fmt = champs.get(b"response_format", b"json").decode()
        if fmt == "text":
            return self._repondre(200, texte, "text/plain")
        corps = {"text": texte}
        if fmt == "verbose_json":
            corps.update({"language": "fr", "duration": 1.0, "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": texte}]})
        self._repondre(200, corps)


def demarrer(port: int = 0, journal: str | None = None) -> ThreadingHTTPServer:
    """Démarre le service en arrière-plan ; port 0 : port libre choisi par le système (server_address[1])."""
    Gestionnaire.journal_chemin = journal
    serveur = ThreadingHTTPServer(("127.0.0.1", port), Gestionnaire)
    threading.Thread(target=serveur.serve_forever, daemon=True).start()
    return serveur


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--journal")
    a = ap.parse_args()
    demarrer(a.port, a.journal)
    threading.Event().wait()
