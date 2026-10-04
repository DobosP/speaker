"""Retained Python worker SDK mint contracts; HTTP contracts moved to Go.

See docs/go_serving_boundary.md for the original-test mapping.
"""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from remote.token_server import create_access_token


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
