"""Scoped window-capture seam for the watch/monitor capability (`core/watch.py`).

The single net-new capture primitive: observe ONE owner-granted application window
-- never the whole screen. Mirrors the ``AudioEngine`` / ``ui_grounding.CaptureFn``
seam so the whole watch feature is Tier-0 testable with a fake source (no real
display, no ``mss``/Xlib/tesseract).

Security invariant (INV-2): there is NO code path here that falls back to a
monitor-wide grab. A window that cannot be resolved (not open, ambiguous, or an
unsupported display server) yields ``None`` -- a benign no-match, not a full-screen
capture. The geometry is re-resolved immediately before each grab so a moved/closed
window is never mis-captured.

X11-only in v1: Wayland / headless / macOS / Windows resolve to an "unavailable"
source (``observe`` always ``None``). A Wayland-portal / Quartz / win32 backend is a
documented follow-up behind this same seam.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from always_on_agent.origin import Origin

log = logging.getLogger("speaker.watch_source")

_HAYSTACK_CAP = 4096  # bound display-derived text before any user regex runs on it


def _is_redos(pattern: str) -> bool:
    """Heuristic: flag a nested-quantifier pattern (``(a+)+``, ``(a*)*``, ``(\\d+){2,}``)
    -- the classic catastrophic-backtracking shapes. A group that is itself quantified
    AND contains a quantifier inside is rejected."""
    for m in re.finditer(r"\(([^()]*)\)(?:[+*]|\{\d+,?\d*\})", pattern or ""):
        if re.search(r"[+*]|\{\d+,?\d*\}", m.group(1)):
            return True
    return False


def wm_class_matches(wanted: str, instance: str, klass: str) -> bool:
    """True iff the granted ``wanted`` wm_class EXACTLY equals (case-insensitive) the
    window's instance or class. Exact, not substring: WM_CLASS is an attacker-settable
    identifier, so a substring match would let a window classed ``Signal-Phisher``
    satisfy a grant for ``signal``."""
    w = (wanted or "").strip().lower()
    if not w:
        return False
    return w == (instance or "").strip().lower() or w == (klass or "").strip().lower()


def safe_search(pattern: str, text: str, *, flags: int = re.I):
    """Run an OWNER-supplied regex against ATTACKER-influenceable display text without
    risking a ReDoS on the shared poller thread: reject nested-quantifier patterns and
    bound the haystack length. Returns the match or None (None also means rejected/
    invalid -- fail-closed: a footgun pattern disables that match, never freezes)."""
    if not pattern or _is_redos(pattern):
        return None
    try:
        return re.search(pattern, (text or "")[:_HAYSTACK_CAP], flags)
    except re.error:
        return None

# An OCR function: image bytes -> word boxes (same shape as ui_grounding.ocr_words).
OcrFn = Callable[[bytes], "list[dict]"]
# A grabber: a resolved window rect -> encoded image bytes (or None when unavailable).
BboxGrabber = Callable[["WindowRect"], Optional[bytes]]


@dataclass(frozen=True)
class WindowRect:
    """The live geometry + identity of the single granted window to capture."""
    left: int
    top: int
    width: int
    height: int
    win_id: str = ""
    title: str = ""
    wm_class: str = ""
    monitor: int = 1


@dataclass(frozen=True)
class Observation:
    """Result of ONE scoped capture. ``origin`` is always ``Origin.SCREEN`` (the
    capture is attacker-controllable DATA, never an action-trusted channel -- see
    always_on_agent.origin). The captured frame bytes are NOT held here: only the
    extracted ``text`` + word boxes survive, and even those stay inside one
    ``WatchManager.tick`` call (INV-3, ephemeral)."""
    text: str
    words: tuple[dict, ...] = ()
    rect: Optional[WindowRect] = None
    origin: str = Origin.SCREEN.value


class WindowResolver(Protocol):
    def resolve(self, app: dict) -> Optional[WindowRect]:
        """``app`` = a grant's app descriptor ``{backend, wm_class, title_pattern?}``.
        Returns the geometry of the SINGLE matching live window, or ``None`` when it
        is not open, ambiguous (more than one match -> fail-closed), or the display
        server is unsupported. NEVER returns a monitor-wide rect."""


class WatchSource(Protocol):
    def observe(self, app: dict) -> Optional[Observation]:
        """Resolve -> bbox-grab -> OCR, scoped to the granted window only. ``None``
        when the window is not resolvable (benign; NOT an error, NEVER a full-screen
        fallback)."""


# --- resolver-backed source (real + injectable) -----------------------------

class _ResolverSource:
    """Composes a resolver + a bbox grabber + an OCR fn into a WatchSource. All
    three are injected so tests use deterministic fakes and no real display."""

    def __init__(self, resolver: WindowResolver, grabber: BboxGrabber, ocr_fn: Optional[OcrFn]):
        self._resolver = resolver
        self._grabber = grabber
        self._ocr = ocr_fn

    def observe(self, app: dict) -> Optional[Observation]:
        try:
            rect = self._resolver.resolve(app or {})
        except Exception:  # noqa: BLE001 - resolution must never crash the poller
            log.debug("watch resolve failed", exc_info=True)
            return None
        if rect is None:
            return None  # not open / ambiguous / unsupported -> NEVER a full grab
        try:
            image = self._grabber(rect)
        except Exception:  # noqa: BLE001 - capture must never crash the poller
            log.debug("watch grab failed", exc_info=True)
            return None
        if not image:
            return None
        words: tuple[dict, ...] = ()
        if self._ocr is not None:
            try:
                words = tuple(self._ocr(image))
            except Exception:  # noqa: BLE001 - OCR is best-effort
                log.debug("watch ocr failed", exc_info=True)
                words = ()
        text = " ".join(str(w.get("text", "")) for w in words).strip()
        # The image bytes go out of scope here -- never persisted (INV-3).
        return Observation(text=text, words=words, rect=rect, origin=Origin.SCREEN.value)


class _UnavailableSource:
    """A WatchSource that can never observe -- the fail-closed default for any
    display server we do not support. Logs the reason once."""

    def __init__(self, reason: str):
        self._reason = reason
        self._warned = False

    def observe(self, app: dict) -> Optional[Observation]:
        if not self._warned:
            log.info("watch unavailable: %s", self._reason)
            self._warned = True
        return None


def _unavailable_source(reason: str) -> WatchSource:
    return _UnavailableSource(reason)


def _grab_bbox(rect: WindowRect) -> Optional[bytes]:
    """mss-backed grab of EXACTLY ``rect`` (a window bbox, not a monitor). Lazy
    import -- mss is an optional dep, only touched when a real watch runs."""
    import mss  # noqa: PLC0415 - optional dep

    from .screen_capture import ScreenCaptureConfig, _default_encoder  # noqa: PLC0415

    if rect.width <= 0 or rect.height <= 0:
        return None
    with mss.mss() as sct:
        shot = sct.grab({
            "left": int(rect.left), "top": int(rect.top),
            "width": int(rect.width), "height": int(rect.height),
            "mon": int(rect.monitor),
        })
        return _default_encoder(ScreenCaptureConfig())(bytes(shot.rgb), (shot.width, shot.height))


class _X11Resolver:
    """Best-effort X11 window resolver (python-xlib, lazy). Matches a grant's
    ``wm_class`` by EXACT case-insensitive equality against the window's instance OR
    class (WM_CLASS values are identifiers, not free text -- a substring match would
    let an attacker window classed ``Signal-Phisher`` satisfy a grant for ``signal``)
    and, when given, a ``title_pattern`` regex (ReDoS-guarded). Returns the geometry of
    the UNIQUE match; zero or multiple matches -> ``None`` (fail-closed, never guess).
    Needs real hardware to validate (Tier-2/3); on any error it degrades to ``None``."""

    def resolve(self, app: dict) -> Optional[WindowRect]:
        wm_class = str(app.get("wm_class", "") or "").strip().lower()
        if not wm_class:
            return None  # a grant with no app identity matches nothing (fail-closed)
        title_pat = app.get("title_pattern")

        from Xlib import X, display  # noqa: PLC0415 - optional dep
        from Xlib.error import XError  # noqa: PLC0415

        dsp = display.Display()
        try:
            root = dsp.screen().root
            net_client = dsp.intern_atom("_NET_CLIENT_LIST")
            prop = root.get_full_property(net_client, X.AnyPropertyType)
            if prop is None:
                return None
            matches: list[WindowRect] = []
            for win_id in prop.value:
                try:
                    win = dsp.create_resource_object("window", win_id)
                    cls = win.get_wm_class()  # (instance, class)
                    if not cls:
                        continue
                    if not wm_class_matches(wm_class, cls[0] or "", cls[1] or ""):
                        continue  # EXACT identity match -- not substring (anti-spoof)
                    name = win.get_wm_name() or ""
                    # ReDoS-guarded + length-bounded match of the owner's title pattern
                    # against the attacker-settable _NET_WM_NAME.
                    if title_pat and safe_search(str(title_pat), str(name)) is None:
                        continue
                    geom = win.get_geometry()
                    # Translate to root (absolute) coordinates for the bbox grab.
                    coords = root.translate_coords(win, 0, 0)
                    matches.append(WindowRect(
                        left=int(coords.x), top=int(coords.y),
                        width=int(geom.width), height=int(geom.height),
                        win_id=str(int(win_id)), title=str(name),
                        wm_class=str(cls[1] or cls[0] or ""),
                    ))
                except XError:
                    continue
            if len(matches) != 1:
                # 0 -> not open; >1 -> ambiguous. Either way: do not capture.
                return None
            return matches[0]
        finally:
            try:
                dsp.close()
            except Exception:  # noqa: BLE001
                pass


def _x11_window_source(ocr_fn: Optional[OcrFn] = None) -> WatchSource:
    ocr = ocr_fn
    if ocr is None:
        from .ui_grounding import ocr_words  # noqa: PLC0415 - default OCR backend
        ocr = ocr_words
    return _ResolverSource(_X11Resolver(), _grab_bbox, ocr)


def make_watch_source(
    *,
    ocr_fn: Optional[OcrFn] = None,
    display_server: Optional[str] = None,
) -> WatchSource:
    """Pick the capture backend for the current display server. X11 -> a real,
    window-scoped source; everything else -> an unavailable source (never a
    full-screen fallback). ``display_server`` is injectable for tests."""
    server = display_server
    if server is None:
        try:
            from .agent_os import detect_display_server  # noqa: PLC0415
            server = detect_display_server()
        except Exception:  # noqa: BLE001
            server = "headless"
    if server == "x11":
        return _x11_window_source(ocr_fn)
    return _unavailable_source(f"display server {server!r} not supported for window-scoped capture (X11 only in v1)")
