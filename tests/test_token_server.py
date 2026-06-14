"""Tests for the token server's pure helpers (no fastapi/livekit needed).

The endpoint-level tests below additionally require fastapi/starlette and
self-skip when those are absent (keeping the Tier-0 suite import-safe).
"""
import pytest

from remote.token_server import sanitize_identity, sanitize_room_name


def test_sanitize_room_name():
    assert sanitize_room_name("My Room!") == "my-room"
    assert sanitize_room_name("  Assistant Room 1 ") == "assistant-room-1"
    assert sanitize_room_name("") == "assistant"
    assert sanitize_room_name(None) == "assistant"
    assert sanitize_room_name("***") == "assistant"


def test_sanitize_identity():
    assert sanitize_identity("Alice Smith") == "Alice-Smith"
    assert sanitize_identity("") == "user"
    assert sanitize_identity(None) == "user"
    assert sanitize_identity("  bob  ") == "bob"
    assert sanitize_identity("a/b\\c!") == "abc"


def _client(monkeypatch):
    """Build a TestClient over create_app with auth disabled (dev opt-in).

    Self-skips when fastapi/starlette are not installed.
    """
    pytest.importorskip("fastapi")
    from starlette.testclient import TestClient

    import remote.token_server as ts

    monkeypatch.setenv("SPEAKER_REMOTE_ALLOW_NOAUTH", "1")
    monkeypatch.delenv("SPEAKER_REMOTE_TOKEN", raising=False)
    app = ts.create_app({})
    return ts, TestClient(app, raise_server_exceptions=False)


def test_token_error_detail_is_generic(monkeypatch):
    """A token-mint failure must not leak the exception text (e.g. a secret-
    bearing config error) to the client."""
    ts, client = _client(monkeypatch)

    def _boom(*_a, **_k):
        raise RuntimeError("LIVEKIT_API_SECRET=supersecret missing")

    monkeypatch.setattr(ts, "create_access_token", _boom)
    r = client.get("/token")
    assert r.status_code == 500
    detail = r.json()["detail"]
    assert detail == "failed to mint access token"
    assert "supersecret" not in detail
    assert "LIVEKIT_API_SECRET" not in detail


def test_chat_error_detail_is_generic(monkeypatch):
    """A chat backend failure must not leak host/path detail to the client."""
    ts, client = _client(monkeypatch)

    class _Boom:
        def generate(self, *_a, **_k):
            raise RuntimeError("ollama at http://10.0.0.5:11434 refused")

    monkeypatch.setattr(ts, "_make_llm", lambda _cfg: _Boom())
    r = client.post("/chat", json={"message": "hi"})
    assert r.status_code == 500
    detail = r.json()["detail"]
    assert detail == "chat backend error"
    assert "10.0.0.5" not in detail
    assert "ollama" not in detail
