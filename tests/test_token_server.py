"""Tests for the token server's pure helpers (no fastapi/livekit needed)."""
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
