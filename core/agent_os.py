"""Display-server preflight for Open Interpreter "OS mode" (desktop GUI control).

OS mode drives the mouse/keyboard and reads the screen (pyautogui + OCR). On
Linux this needs an X11 session: pyautogui's synthetic input and screenshots
generally do NOT work under Wayland without XWayland/portals. This module only
detects the situation and returns human-readable warnings; it never blocks.

Pure functions with injectable env/platform/which so they are unit-testable.
Ported from the legacy ``utils/agent_os.py`` onto the new ``core`` runtime.
"""
from __future__ import annotations

import os
import shutil
import sys
from typing import Callable, Mapping, Optional


def detect_display_server(
    env: Optional[Mapping[str, str]] = None,
    platform: Optional[str] = None,
) -> str:
    """Return one of: macos, windows, wayland, x11, headless."""
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    if platform == "darwin":
        return "macos"
    if platform.startswith("win"):
        return "windows"
    session = (env.get("XDG_SESSION_TYPE") or "").lower()
    if env.get("WAYLAND_DISPLAY") or session == "wayland":
        return "wayland"
    if env.get("DISPLAY") or session == "x11":
        return "x11"
    return "headless"


def os_mode_preflight(
    env: Optional[Mapping[str, str]] = None,
    platform: Optional[str] = None,
    which: Optional[Callable[[str], Optional[str]]] = None,
) -> list[str]:
    """Return warnings about desktop GUI control readiness (possibly empty)."""
    which = shutil.which if which is None else which
    server = detect_display_server(env=env, platform=platform)
    warnings: list[str] = []

    if server == "wayland":
        warnings.append(
            "Wayland session detected: mouse/keyboard control and screenshots "
            "usually fail under Wayland. Use an X11 session, or tools like "
            "ydotool / portal-based capture."
        )
    elif server == "headless":
        warnings.append(
            "No display detected (headless): desktop GUI control will not work; "
            "code/shell and web actions still do."
        )
    elif server == "macos":
        warnings.append(
            "macOS: grant Accessibility and Screen Recording permissions to the "
            "terminal/app running this, or GUI control is silently blocked."
        )

    if server in ("x11", "wayland"):
        if not which("tesseract"):
            warnings.append(
                "tesseract not found: on-screen text reading (OCR) will be "
                "limited (e.g. apt install tesseract-ocr)."
            )
        if not any(which(tool) for tool in ("scrot", "gnome-screenshot", "import", "spectacle")):
            warnings.append(
                "No screenshot tool found (scrot/gnome-screenshot): screen "
                "capture may fail (e.g. apt install scrot)."
            )

    return warnings
