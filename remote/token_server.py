"""FastAPI token server + web host + text /chat for remote access.

Run (after ``pip install -r requirements-remote.txt`` and setting LIVEKIT_* in
the environment / .env):

    uvicorn remote.token_server:app --host 127.0.0.1 --port 8080
    # or: python -m remote.token_server

Endpoints:
    GET  /healthz                  -> {"ok": true}
    GET  /token?identity=&room=    -> {"token","url","room","identity"}
    POST /chat {"message": "..."}  -> {"reply": "..."}   (a core LLM turn)
    GET  /                         -> serves web/ client when present

LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET come from the environment.

Auth (env-only, never hard-coded): /token and /chat require an
``Authorization: Bearer <token>`` header matching ``SPEAKER_REMOTE_TOKEN``. If
that env var is unset, the server refuses remote auth (every protected request
401s) UNLESS ``SPEAKER_REMOTE_ALLOW_NOAUTH=1`` is set as an explicit dev opt-in
(which logs a clear WARNING). The bind defaults to 127.0.0.1; set
``SPEAKER_REMOTE_BIND_ALL=1`` to bind 0.0.0.0 (logs a prominent WARNING when
bound publicly without a configured token).

The live-voice path is handled by the LiveKit worker (``remote.worker``) driving
:class:`core.engines.livekit.LiveKitEngine`; /chat here is a lightweight text
turn through the same local LLM for the web UI's text box.

Pure string helpers live at module scope and are import-safe without fastapi or
livekit installed; everything else lazy-imports those deps.
"""
from __future__ import annotations

import logging
import os
import re
import time
from collections import deque
from typing import Deque, Dict, Optional

logger = logging.getLogger(__name__)

_ROOM_RE = re.compile(r"[^a-z0-9_-]+")
_IDENT_BAD = re.compile(r"[^A-Za-z0-9_\-]+")

# Safety limits for the public /chat surface.
MAX_CHAT_BYTES = 16 * 1024  # 16 KiB request body cap
RATE_LIMIT_MAX = 30  # requests ...
RATE_LIMIT_WINDOW = 60.0  # ... per this many seconds, per client


def sanitize_room_name(name: Optional[str], default: str = "assistant") -> str:
    cleaned = _ROOM_RE.sub("-", (name or "").strip().lower()).strip("-")
    return cleaned or default


def sanitize_identity(identity: Optional[str], default: str = "user") -> str:
    cleaned = _IDENT_BAD.sub("", re.sub(r"\s+", "-", (identity or "").strip()))
    return cleaned or default


def _expected_token() -> Optional[str]:
    # A whitespace-only value is treated as unset (not "configured but
    # unmatchable"), so the posture logging stays accurate.
    tok = (os.environ.get("SPEAKER_REMOTE_TOKEN") or "").strip()
    return tok or None


def _allow_noauth() -> bool:
    return os.environ.get("SPEAKER_REMOTE_ALLOW_NOAUTH", "") == "1"


def _bind_all() -> bool:
    return os.environ.get("SPEAKER_REMOTE_BIND_ALL", "") == "1"


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    """Pull the token out of an ``Authorization: Bearer <token>`` header value."""
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def check_auth(authorization: Optional[str]) -> bool:
    """Return True iff the request is allowed by the env-driven auth policy.

    Pure and import-safe so it can be unit-tested without fastapi.

    Policy:
    - ``SPEAKER_REMOTE_TOKEN`` set  -> require a matching ``Bearer`` token.
    - unset + ``SPEAKER_REMOTE_ALLOW_NOAUTH=1`` -> allow (dev opt-in; caller
      logs a WARNING).
    - unset + no opt-in            -> refuse (deny by default).
    """
    expected = _expected_token()
    if expected is None:
        return _allow_noauth()
    presented = _extract_bearer(authorization)
    if presented is None:
        return False
    # Constant-time compare to avoid leaking the token via timing.
    import hmac

    return hmac.compare_digest(presented, expected)


class RateLimiter:
    """Tiny in-process fixed-window-ish sliding rate limiter.

    Not distributed and intentionally simple — a first line of defence on the
    public /chat surface, not a substitute for an upstream gateway.
    """

    def __init__(self, max_requests: int = RATE_LIMIT_MAX, window: float = RATE_LIMIT_WINDOW):
        self.max_requests = max_requests
        self.window = window
        self._hits: Dict[str, Deque[float]] = {}

    def allow(self, key: str, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        dq = self._hits.get(key)
        if dq is None:
            dq = deque()
            self._hits[key] = dq
        cutoff = now - self.window
        while dq and dq[0] <= cutoff:
            dq.popleft()
        if len(dq) >= self.max_requests:
            return False
        dq.append(now)
        return True


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


def _log_auth_posture() -> None:
    """Emit a WARNING describing the current (insecure) auth/bind posture."""
    if _expected_token() is None:
        if _allow_noauth():
            logger.warning(
                "SPEAKER_REMOTE_TOKEN is unset and SPEAKER_REMOTE_ALLOW_NOAUTH=1: "
                "/token and /chat are UNAUTHENTICATED. Dev use only -- never expose this."
            )
        else:
            logger.warning(
                "SPEAKER_REMOTE_TOKEN is unset: /token and /chat will refuse all "
                "requests (401). Set SPEAKER_REMOTE_TOKEN to enable remote auth."
            )
    if _bind_all() and _expected_token() is None:
        logger.warning(
            "SPEAKER_REMOTE_BIND_ALL=1 binds 0.0.0.0 with NO token configured -- "
            "the token server is publicly reachable without authentication."
        )


def create_app(config: Optional[dict] = None):
    """Build the FastAPI app (lazy import so this module loads without fastapi)."""
    from fastapi import Depends, FastAPI, Header, HTTPException, Request

    # Make `Request` resolvable as a module global so FastAPI can evaluate the
    # PEP 563 string annotations (`from __future__ import annotations`) on the
    # route handlers below -- they are declared `request: Request`, but the
    # import is local to keep this module import-safe without fastapi. Without
    # this, FastAPI cannot resolve the forward ref and mis-binds `request` as a
    # query param, 422-ing every /chat call before it reaches the handler.
    globals()["Request"] = Request

    config = config if config is not None else _load_config()
    app = FastAPI(title="Speaker remote")
    _holder = {"llm": None}
    rate_limiter = RateLimiter()

    _log_auth_posture()

    def llm():
        if _holder["llm"] is None:
            _holder["llm"] = _make_llm(config)
        return _holder["llm"]

    def require_auth(authorization: Optional[str] = Header(default=None)):
        if check_auth(authorization):
            return
        if _expected_token() is None and not _allow_noauth():
            # Misconfigured / locked down: make the cause explicit.
            raise HTTPException(
                status_code=401,
                detail="remote auth not configured (set SPEAKER_REMOTE_TOKEN)",
            )
        raise HTTPException(
            status_code=401,
            detail="missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    def _client_key(request: Request) -> str:
        client = request.client
        return client.host if client else "unknown"

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/token")
    def token(
        identity: str = "user",
        room: str = "assistant",
        _: None = Depends(require_auth),
    ):
        ident = sanitize_identity(identity)
        rm = sanitize_room_name(room)
        try:
            jwt = create_access_token(ident, rm)
        except Exception:
            # Log the real cause server-side; return a generic message so we
            # don't leak config/secret details (e.g. a missing LIVEKIT_API_SECRET
            # surfaces in the exception text) to the client.
            logger.exception("token mint failed for identity=%r room=%r", ident, rm)
            raise HTTPException(status_code=500, detail="failed to mint access token")
        return {"token": jwt, "url": os.environ.get("LIVEKIT_URL", ""), "room": rm, "identity": ident}

    @app.post("/chat")
    async def chat(
        request: Request,
        _: None = Depends(require_auth),
    ):
        if not rate_limiter.allow(_client_key(request)):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        raw = await request.body()
        if len(raw) > MAX_CHAT_BYTES:
            raise HTTPException(status_code=413, detail="request body too large")
        try:
            import json

            payload = json.loads(raw or b"{}")
            message = payload.get("message", "") if isinstance(payload, dict) else ""
        except Exception:
            raise HTTPException(status_code=400, detail="invalid JSON body")
        text = (message or "").strip()
        if not text:
            return {"reply": ""}
        try:
            reply = llm().generate(text)
        except Exception:
            # Log server-side; the backend error can carry host/path detail
            # (Ollama URL, model path), so the client gets a generic message.
            logger.exception("chat LLM turn failed")
            raise HTTPException(status_code=500, detail="chat backend error")
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
    host = "0.0.0.0" if _bind_all() else "127.0.0.1"
    if host == "0.0.0.0":
        logger.warning(
            "Binding token server to 0.0.0.0:%s (SPEAKER_REMOTE_BIND_ALL=1) -- "
            "publicly reachable on this network.",
            port,
        )
    else:
        logger.info("Binding token server to 127.0.0.1:%s (set SPEAKER_REMOTE_BIND_ALL=1 to expose).", port)
    uvicorn.run(create_app(cfg), host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
