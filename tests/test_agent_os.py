"""Tests for the OS-mode (desktop GUI) display-server preflight."""
from utils.agent_os import detect_display_server, os_mode_preflight


def test_detect_wayland():
    assert detect_display_server(env={"WAYLAND_DISPLAY": "wayland-0"}, platform="linux") == "wayland"
    assert detect_display_server(env={"XDG_SESSION_TYPE": "wayland"}, platform="linux") == "wayland"


def test_detect_x11():
    assert detect_display_server(env={"DISPLAY": ":0"}, platform="linux") == "x11"
    assert detect_display_server(env={"XDG_SESSION_TYPE": "x11"}, platform="linux") == "x11"


def test_detect_headless():
    assert detect_display_server(env={}, platform="linux") == "headless"


def test_detect_macos_and_windows():
    assert detect_display_server(env={}, platform="darwin") == "macos"
    assert detect_display_server(env={}, platform="win32") == "windows"


def test_preflight_wayland_warns():
    warns = os_mode_preflight(
        env={"WAYLAND_DISPLAY": "wayland-0"}, platform="linux", which=lambda x: "/usr/bin/" + x
    )
    assert any("wayland" in w.lower() for w in warns)


def test_preflight_headless_warns():
    warns = os_mode_preflight(env={}, platform="linux", which=lambda x: None)
    assert any("headless" in w.lower() for w in warns)


def test_preflight_x11_missing_tools_warns():
    warns = os_mode_preflight(env={"DISPLAY": ":0"}, platform="linux", which=lambda x: None)
    assert any("tesseract" in w.lower() for w in warns)
    assert any("screenshot" in w.lower() for w in warns)


def test_preflight_x11_with_tools_is_clean():
    warns = os_mode_preflight(env={"DISPLAY": ":0"}, platform="linux", which=lambda x: "/usr/bin/" + x)
    assert warns == []


def test_preflight_macos_permissions():
    warns = os_mode_preflight(env={}, platform="darwin", which=lambda x: None)
    assert any("permission" in w.lower() for w in warns)
