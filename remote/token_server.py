"""Retained Python LiveKit-worker token helper; no HTTP listener.

The optional rollback web boundary is the Go binary in ``remote/serving``.
This module retains only the exact reviewed LiveKit SDK mint chain used by
``remote.worker``. It does not activate trusted-LAN or grant action authority.
See ADR-0224 and docs/go_serving_boundary.md for the boundary and qualification.
"""
from __future__ import annotations

import os


def create_access_token(identity: str, room: str, ttl_seconds: int = 3600) -> str:
    """Mint the retained worker's JWT; exact TTL assignment is fail closed."""
    from datetime import timedelta

    from livekit import api

    key = os.environ.get("LIVEKIT_API_KEY")
    secret = os.environ.get("LIVEKIT_API_SECRET")
    if not key or not secret:
        raise RuntimeError("LIVEKIT_API_KEY / LIVEKIT_API_SECRET are not set")
    return (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .with_ttl(timedelta(seconds=ttl_seconds))
        .to_jwt()
    )


if __name__ == "__main__":
    raise SystemExit("Python HTTP serving was removed; see docs/go_serving_boundary.md")
