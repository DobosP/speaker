"""Tier-0 tests for the optional screen-capture feed (core/screen_capture.py).

Pure logic: the grabber + encoder are injected, so the cadence + plumbing are
verified with no real display, no mss, no Pillow. The feed is OFF by default.
"""
from __future__ import annotations

import time

from core.screen_capture import (
    ScreenCaptureConfig,
    ScreenFrameFeed,
    build_screen_feed,
)


def test_disabled_by_default():
    assert ScreenCaptureConfig.from_dict({}).enabled is False
    assert ScreenCaptureConfig.from_dict(None).enabled is False


def test_config_from_dict_reads_fields():
    cfg = ScreenCaptureConfig.from_dict(
        {"enabled": True, "interval_sec": 0.5, "monitor": 0, "max_width": 800, "format": "PNG"}
    )
    assert cfg.enabled and cfg.interval_sec == 0.5 and cfg.monitor == 0
    assert cfg.max_width == 800 and cfg.format == "png"
    # interval is floored so a bad 0 can't spin a hot loop.
    assert ScreenCaptureConfig.from_dict({"interval_sec": 0}).interval_sec >= 0.1


def test_capture_once_grabs_encodes_and_feeds():
    frames: list = []
    feed = ScreenFrameFeed(
        frames.append,
        ScreenCaptureConfig(enabled=True),
        grabber=lambda: (b"RAWPIXELS", (4, 2)),
        encoder=lambda raw, size: b"IMG:" + raw + str(size).encode(),
    )
    out = feed.capture_once()
    assert out == b"IMG:RAWPIXELS(4, 2)"
    assert frames == [b"IMG:RAWPIXELS(4, 2)"]


def test_capture_error_is_swallowed_and_feeds_nothing():
    frames: list = []

    def boom():
        raise RuntimeError("no display")

    feed = ScreenFrameFeed(frames.append, grabber=boom)
    assert feed.capture_once() is None
    assert frames == []  # a failed grab feeds no frame


def test_loop_feeds_frames_then_stop_clears_the_frame():
    frames: list = []
    feed = ScreenFrameFeed(
        frames.append,
        ScreenCaptureConfig(enabled=True, interval_sec=0.1),
        grabber=lambda: (b"R", (1, 1)),
        encoder=lambda raw, size: b"FRAME",
    )
    feed.start()
    deadline = time.monotonic() + 2.0
    while not frames and time.monotonic() < deadline:
        time.sleep(0.01)
    feed.stop()  # joins the thread, then clears the frame
    assert b"FRAME" in frames, "the loop never fed a frame"
    # stop() drops the stale frame so the model stops seeing the old screen.
    assert frames[-1] is None


def test_start_is_idempotent():
    feed = ScreenFrameFeed(
        lambda f: None,
        ScreenCaptureConfig(enabled=True, interval_sec=0.1),
        grabber=lambda: (b"R", (1, 1)),
        encoder=lambda raw, size: b"F",
    )
    feed.start()
    t1 = feed._thread
    feed.start()  # second start is a no-op while alive
    assert feed._thread is t1
    feed.stop()


class _FakeRuntime:
    def __init__(self):
        self.frames: list = []

    def set_current_frame(self, image):
        self.frames.append(image)


def test_build_screen_feed_off_by_default_returns_none():
    assert build_screen_feed({}, _FakeRuntime()) is None
    assert build_screen_feed({"screen_capture": {"enabled": False}}, _FakeRuntime()) is None


def test_build_screen_feed_wires_to_runtime_set_current_frame():
    rt = _FakeRuntime()
    feed = build_screen_feed({"screen_capture": {"enabled": True}}, rt)
    assert feed is not None
    # The built feed is wired to the runtime's setter (inject a grabber/encoder so
    # capture_once doesn't touch a real display).
    feed._grab = lambda: (b"X", (1, 1))
    feed._encode = lambda raw, size: b"WIRED"
    feed.capture_once()
    assert rt.frames == [b"WIRED"]


def test_build_screen_feed_none_when_runtime_cannot_take_frames():
    class _NoSetter:
        pass

    assert build_screen_feed({"screen_capture": {"enabled": True}}, _NoSetter()) is None
