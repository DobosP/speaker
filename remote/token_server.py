"""FastAPI token server + web host + text /chat for remote access.

Run (after ``pip install -r requirements-remote.txt`` and setting LIVEKIT_* in
the environment / .env):

    uvicorn remote.token_server:app --host 0.0.0.0 --port 8080
    # or: python -m remote.token_server

Endpoints:
    GET  /healthz                  -> {"ok": true}
    GET  /token?identity=&room=    -> {"token","url","room","identity"}
    POST /chat {"message": "..."}  -> {"reply": "..."}   (a core LLM turn)
    GET  /                         -> serves web/ client when present

LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET come from the environment.
NOTE: as written this mints a token for any identity/room -- put real
authentication in front of /token before exposing it.

The live-voice path is handled by the LiveKit worker (``remote.worker``) driving
:class:`core.engines.livekit.LiveKitEngine`; /chat here is a lightweight text
turn through the same local LLM for the web UI's text box.

Pure string helpers live at module scope and are import-safe without fastapi or
livekit installed; everything else lazy-imports those deps.
"""
from __future__ import annotations

import os
import re
from typing import Optional

_ROOM_RE = re.compile(r"[^a-z0-9_-]+")
_IDENT_BAD = re.compile(r"[^A-Za-z0-9_\-]+")


def sanitize_room_name(name: Optional[str], default: str = "assistant") -> str:
    cleaned = _ROOM_RE.sub("-", (name or "").strip().lower()).strip("-")
    return cleaned or default


def sanitize_identity(identity: Optional[str], default: str = "user") -> str:
    cleaned = _IDENT_BAD.sub("", re.sub(r"\s+", "-", (identity or "").strip()))
    return cleaned or default


def create_access_token(identity: str, room: str, ttl_seconds: int = 3600) -> str:
    """Mint a LiveKit JWT. Requires livekit-api + LIVEKIT_API_KEY/SECRET."""
    from livekit import api  # lazy

    key = os.environ.get("LIVEKIT_API_KEY")
    secret = os.environ.get("LIVEKIT_API_SECRET")
    if not key or not secret:
        raise RuntimeError("LIVEKIT_API_KEY / LIVEKIT_API_SECRET are not set")
    token = (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room))
    )
    try:
        from datetime import timedelta

        token = token.with_ttl(timedelta(seconds=ttl_seconds))
    except Exception:
        pass
    return token.to_jwt()


def _load_config() -> dict:
    try:
        import json

        with open("config.json") as f:
            return json.load(f)
    except Exception:
        return {}


def _make_llm(config: dict):
    """Build a core LLM client from the ``llm`` config block (lazy, local)."""
    from core.llm import EchoLLM, LlamaCppLLM, OllamaLLM

    llm_cfg = config.get("llm") or {}
    backend = llm_cfg.get("backend", "ollama")
    if backend == "llamacpp":
        path = llm_cfg.get("main_model_path")
        if not path:
            return EchoLLM()
        return LlamaCppLLM(
            path,
            n_ctx=llm_cfg.get("n_ctx", 4096),
            n_threads=llm_cfg.get("n_threads"),
            n_gpu_layers=llm_cfg.get("n_gpu_layers", 0),
            chat_format=llm_cfg.get("chat_format"),
            options=llm_cfg.get("options"),
        )
    model = llm_cfg.get("main_model") or config.get("llm_model", "gemma3:12b")
    return OllamaLLM(
        model=model,
        host=llm_cfg.get("host"),
        options=llm_cfg.get("options"),
        keep_alive=llm_cfg.get("keep_alive"),
    )


def create_app(config: Optional[dict] = None):
    """Build the FastAPI app (lazy import so this module loads without fastapi)."""
    from fastapi import Body, FastAPI, HTTPException

    config = config if config is not None else _load_config()
    app = FastAPI(title="Speaker remote")
    _holder = {"llm": None}

    def llm():
        if _holder["llm"] is None:
            _holder["llm"] = _make_llm(config)
        return _holder["llm"]

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/token")
    def token(identity: str = "user", room: str = "assistant"):
        ident = sanitize_identity(identity)
        rm = sanitize_room_name(room)
        try:
            jwt = create_access_token(ident, rm)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return {"token": jwt, "url": os.environ.get("LIVEKIT_URL", ""), "room": rm, "identity": ident}

    @app.post("/chat")
    def chat(message: str = Body(..., embed=True)):
        text = (message or "").strip()
        if not text:
            return {"reply": ""}
        try:
            reply = llm().generate(text)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return {"reply": (reply or "").strip()}

    web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
    if os.path.isdir(web_dir):
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app


# Module-level ASGI app for `uvicorn remote.token_server:app`. Guarded so that
# importing this module without fastapi installed leaves app = None instead of
# raising (keeps the base test suite import-safe).
try:  # pragma: no cover
    app = create_app()
except Exception:  # pragma: no cover
    app = None


def main():  # pragma: no cover
    import uvicorn

    cfg = _load_config()
    port = int((cfg.get("remote") or {}).get("token_server_port", 8080))
    uvicorn.run(create_app(cfg), host="0.0.0.0", port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
