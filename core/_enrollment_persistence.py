"""Admission for enrollment mutation before private IO or capture.

Windows needs a qualified native ACL/handle/publication/lock implementation;
POSIX mode bits and uid placeholders cannot supply that contract (ADR-0227).
"""
from __future__ import annotations

import os
import sys


class EnrollmentPersistenceUnavailable(ValueError):
    """The host has no qualified private enrollment transaction backend."""


def require_enrollment_persistence() -> None:
    """Refuse unsupported hosts without inspecting any supplied path."""
    if sys.platform == "win32":
        raise EnrollmentPersistenceUnavailable(
            "private enrollment persistence is unavailable on Windows: "
            "the native ACL, handle, durability and locking backend is not "
            "qualified; see docs/windows_enrollment_persistence.md (ADR-0227)"
        )
    if (
        os.name != "posix"
        or not callable(getattr(os, "fchmod", None))
        or not callable(getattr(os, "getuid", None))
    ):
        raise EnrollmentPersistenceUnavailable(
            "private enrollment persistence requires a qualified POSIX "
            "ownership/permission backend"
        )
