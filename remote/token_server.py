"""FastAPI token server + web host + text /chat for remote access.

Run (after `pip install -r requirements-remote.txt` and setting LIVEKIT_* in
the environment / .env):

    uvicorn remote.token_server:app --host 0.0.0.0 --port 8080
    # or: python -m remote.token_server

Endpoints:
    GET  /healthz                  -> {"ok": true}
    GET  /token?identity=&room=    -> {"token","url","room","identity"}
    POST /chat {"message": "..."}  -> {"reply": "..."}   (reuses the pipeline)
    GET  /                         -> serves web/ client when present

LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET come from the environment
(.env is auto-loaded by utils.llm). NOTE: as written this mints a token for any
identity/room -- put real authentication in front of /token before exposing it.

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


def create_app(config: Optional[dict] = None):
    """Build the FastAPI app (lazy import so this module loads without fastapi)."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    config = config if config is not None else _load_config()
    app = FastAPI(title="Speaker remote")
    _holder = {"session": None}

    def session():
        if _holder["session"] is None:
            from remote.pipeline_bridge import RemoteSession

            _holder["session"] = RemoteSession(config)
        return _holder["session"]

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

    class ChatIn(BaseModel):
        message: str

    @app.post("/chat")
    def chat(body: ChatIn):
        reply = "".join(session().respond(body.message)).strip()
        return {"reply": reply}

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
