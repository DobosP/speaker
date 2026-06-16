"""Tier-0 tests for the watch capture seam (``core/watch_source.py``).

Focus: INV-2 -- scoped, window-only capture. There is NO code path that falls
back to a monitor-wide grab. A window that cannot be resolved yields a benign
``None`` (never a full-screen capture), and a resolved window is grabbed at
EXACTLY its rect.

Self-contained + fakes only: no real display, no mss/Xlib/tesseract/model. The
resolver, grabber, and OCR fn are all injected as deterministic fakes.
"""
from __future__ import annotations

import pytest

from core.watch_source import (
    Observation,
    WindowRect,
    _ResolverSource,
    _UnavailableSource,
    make_watch_source,
)
from always_on_agent.origin import Origin


# --- tiny self-contained fakes ----------------------------------------------

class FakeResolver:
    """A duck-typed WindowResolver: returns whatever rect it was seeded with."""

    def __init__(self, rect):
        self._rect = rect
        self.calls = 0
        self.last_app = None

    def resolve(self, app):
        self.calls += 1
        self.last_app = app
        return self._rect


class RaisingResolver:
    """A resolver that blows up -- the source must degrade to None, not crash."""

    def __init__(self, exc):
        self._exc = exc
        self.calls = 0

    def resolve(self, app):
        self.calls += 1
        raise self._exc


class SpyGrabber:
    """Records every rect it is asked to grab; returns fixed image bytes."""

    def __init__(self, image=b"img"):
        self._image = image
        self.calls = 0
        self.rects = []

    def __call__(self, rect):
        self.calls += 1
        self.rects.append(rect)
        return self._image


def _ocr_fixed(words):
    """An OCR fn returning a fixed list of word boxes regardless of image bytes."""

    def _fn(image_bytes):
        assert isinstance(image_bytes, (bytes, bytearray))
        return list(words)

    return _fn


# --- INV-2: unresolved window => None, grabber NEVER touched -----------------

def test_unresolved_window_returns_none_and_never_grabs():
    """resolver.resolve -> None means: no Observation AND no capture at all.
    This is the core of INV-2: there is no full-screen fallback when the
    target window is absent/ambiguous/unsupported."""
    resolver = FakeResolver(None)
    grabber = SpyGrabber()
    src = _ResolverSource(resolver, grabber, _ocr_fixed([{"text": "x"}]))

    result = src.observe({"wm_class": "firefox"})

    assert result is None
    assert resolver.calls == 1
    # The single most important INV-2 assertion: nothing was captured.
    assert grabber.calls == 0
    assert grabber.rects == []


def test_resolver_exception_returns_none_and_never_grabs():
    """A crashing resolver must fail-closed to None (benign no-match) and must
    NOT fall through to a grab."""
    resolver = RaisingResolver(RuntimeError("boom"))
    grabber = SpyGrabber()
    src = _ResolverSource(resolver, grabber, _ocr_fixed([{"text": "x"}]))

    assert src.observe({"wm_class": "firefox"}) is None
    assert resolver.calls == 1
    assert grabber.calls == 0


# --- INV-2: resolved window => grab EXACTLY that rect -----------------------

def test_resolved_window_grabs_exact_rect_and_builds_observation():
    """resolver.resolve -> a WindowRect means the grabber is called with EXACTLY
    that rect (the scoped bbox, not a monitor), and the OCR output becomes the
    Observation's words/text. Origin is always SCREEN (untrusted DATA)."""
    rect = WindowRect(left=10, top=20, width=300, height=200,
                      win_id="42", title="Inbox", wm_class="firefox")
    resolver = FakeResolver(rect)
    grabber = SpyGrabber(image=b"frame-bytes")
    ocr = _ocr_fixed([{"text": "hello"}, {"text": "world"}])
    src = _ResolverSource(resolver, grabber, ocr)

    obs = src.observe({"wm_class": "firefox"})

    assert isinstance(obs, Observation)
    # Grabbed exactly the resolved window rect -- no widening to a monitor.
    assert grabber.calls == 1
    assert grabber.rects == [rect]
    assert grabber.rects[0] is rect
    # OCR words flow into the Observation; text is the joined words.
    assert obs.words == ({"text": "hello"}, {"text": "world"})
    assert obs.text == "hello world"
    assert obs.rect is rect
    # The capture is attacker-controllable DATA, never an action-trusted channel.
    assert obs.origin == Origin.SCREEN.value
    assert obs.origin == "screen"


def test_resolved_window_passes_app_to_resolver():
    """The grant's app descriptor is handed verbatim to the resolver (so the
    resolver -- not a wide screengrab -- decides what window matches)."""
    rect = WindowRect(left=0, top=0, width=100, height=100, wm_class="code")
    resolver = FakeResolver(rect)
    grabber = SpyGrabber()
    src = _ResolverSource(resolver, grabber, _ocr_fixed([]))

    app = {"wm_class": "code", "title_pattern": "main.py"}
    src.observe(app)

    assert resolver.last_app == app


def test_observation_text_empty_when_no_ocr_words():
    """No OCR words -> empty text, still a valid scoped Observation (the grab
    happened, just nothing legible)."""
    rect = WindowRect(left=1, top=2, width=3, height=4, wm_class="term")
    src = _ResolverSource(FakeResolver(rect), SpyGrabber(), _ocr_fixed([]))

    obs = src.observe({"wm_class": "term"})

    assert isinstance(obs, Observation)
    assert obs.words == ()
    assert obs.text == ""
    assert obs.origin == Origin.SCREEN.value


# --- make_watch_source: only x11 is a real source; everything else None ------

@pytest.mark.parametrize("server", ["wayland", "headless", "windows", "macos"])
def test_unsupported_display_servers_are_unavailable(server):
    """Any non-X11 display server resolves to an unavailable source whose
    observe() is ALWAYS None -- the fail-closed default (never full-screen)."""
    src = make_watch_source(display_server=server)

    assert isinstance(src, _UnavailableSource)
    # Repeated calls, varied apps: never anything but None.
    assert src.observe({"wm_class": "firefox"}) is None
    assert src.observe({}) is None
    assert src.observe({"wm_class": "anything", "title_pattern": ".*"}) is None


def test_x11_source_is_a_real_source_not_the_unavailable_stub():
    """display_server='x11' returns a real, window-scoped source -- NOT the
    unavailable stub. We do not call observe() (it would need real Xlib/mss);
    we only assert it is the resolver-backed type."""
    src = make_watch_source(display_server="x11")

    assert isinstance(src, _ResolverSource)
    assert not isinstance(src, _UnavailableSource)


def test_x11_source_degrades_gracefully_when_deps_missing():
    """The X11 source's real resolver lazily imports Xlib; with Xlib absent (the
    Tier-0 env) resolve() raises ImportError, which _ResolverSource swallows ->
    observe() returns None gracefully (and, per INV-2, never grabs)."""
    src = make_watch_source(display_server="x11")
    assert isinstance(src, _ResolverSource)

    # Real resolver, no display -> ImportError/XError internally -> None.
    # (If Xlib happened to be installed, resolve still fail-closes to None for a
    # bogus wm_class against no live window.)
    result = src.observe({"wm_class": "no-such-window-zzz"})
    assert result is None
