"""Tests for the token server's pure helpers (no fastapi/livekit needed).

The endpoint-level tests below additionally require fastapi/starlette and
self-skip when those are absent (keeping the Tier-0 suite import-safe).
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from core.llm_threads import resolve_llamacpp_thread_pair
from remote.token_server import (
    _make_llm,
    create_access_token,
    sanitize_identity,
    sanitize_room_name,
)


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


def test_llamacpp_text_server_uses_bounded_thread_pair():
    llm = _make_llm({"llm": {"backend": "llamacpp", "main_model_path": "model.gguf"}})
    expected = resolve_llamacpp_thread_pair()

    assert (llm.n_threads, llm.n_threads_batch) == (
        expected.n_threads,
        expected.n_threads_batch,
    )


def test_llamacpp_text_server_preserves_explicit_batch_override():
    llm = _make_llm(
        {
            "llm": {
                "backend": "llamacpp",
                "main_model_path": "model.gguf",
                "n_threads": 2,
                "n_threads_batch": 3,
            }
        }
    )

    assert (llm.n_threads, llm.n_threads_batch) == (2, 3)


def _fake_livekit_api(monkeypatch, *, ttl_error=None):
    calls = []

    class VideoGrants:
        def __init__(self, **kwargs):
            calls.append(("grants", kwargs))

    class AccessToken:
        def __init__(self, key, secret):
            calls.append(("token", key, secret))

        def with_identity(self, identity):
            calls.append(("identity", identity))
            return self

        def with_name(self, name):
            calls.append(("name", name))
            return self

        def with_grants(self, grants):
            assert isinstance(grants, VideoGrants)
            calls.append(("with_grants",))
            return self

        def with_ttl(self, ttl):
            calls.append(("ttl_seconds", ttl.total_seconds()))
            if ttl_error is not None:
                raise ttl_error
            return self

        def to_jwt(self):
            calls.append(("to_jwt",))
            return "test-token"

    livekit = ModuleType("livekit")
    livekit.api = SimpleNamespace(AccessToken=AccessToken, VideoGrants=VideoGrants)
    monkeypatch.setitem(sys.modules, "livekit", livekit)
    return calls


def test_access_token_uses_reviewed_api_chain_and_exact_ttl(monkeypatch):
    calls = _fake_livekit_api(monkeypatch)
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret")

    assert create_access_token("publisher", "room", ttl_seconds=3600) == "test-token"
    assert calls == [
        ("token", "test-key", "test-secret"),
        ("identity", "publisher"),
        ("name", "publisher"),
        ("grants", {"room_join": True, "room": "room"}),
        ("with_grants",),
        ("ttl_seconds", 3600.0),
        ("to_jwt",),
    ]


def test_access_token_ttl_failure_is_fail_closed(monkeypatch):
    calls = _fake_livekit_api(monkeypatch, ttl_error=RuntimeError("ttl rejected"))
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret")

    with pytest.raises(RuntimeError, match="ttl rejected"):
        create_access_token("publisher", "room", ttl_seconds=3600)
    assert calls[-1] == ("ttl_seconds", 3600.0)
    assert ("to_jwt",) not in calls


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
