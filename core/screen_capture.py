"""Optional screen-capture feed → the model's visual context. OFF BY DEFAULT.

A background loop grabs the screen every ``interval_sec``, encodes one frame, and
hands it to ``set_frame`` (wire this to :meth:`core.runtime.VoiceRuntime.set_current_frame`),
so the assistant has *ambient* context of what is on screen — the latest frame
rides along on every assistant turn until cleared.

Disabled unless ``config.screen_capture.enabled`` is true, so a default run never
captures the screen. When enabled it needs ``mss`` (`pip install mss`) and,
optionally, ``Pillow`` to downscale + JPEG-encode (without Pillow it sends a
full-resolution PNG via ``mss.tools.to_png``). The grabber + encoder are
injectable so the cadence + plumbing are unit-tested with no real display.

Privacy: this continuously reads your screen while enabled. It stays local — an
image-bearing turn is forced to the on-device/main tier and treated as PRIVATE
(``core/capabilities.py``), so a frame never rides a public cloud chain (§9.7).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

log = logging.getLogger("speaker.screen")

# A grabber returns (raw_RGB_bytes, (width, height)); an encoder turns that into
# the image bytes the multimodal LLM accepts (JPEG/PNG).
Grabber = Callable[[], Tuple[bytes, Tuple[int, int]]]
Encoder = Callable[[bytes, Tuple[int, int]], bytes]
SetFrame = Callable[[Optional[object]], None]


@dataclass
class ScreenCaptureConfig:
    enabled: bool = False
    interval_sec: float = 2.0
    monitor: int = 1          # mss monitor index: 1 = primary, 0 = all monitors
    max_width: int = 1280     # downscale the longest edge to this (0 = full; needs Pillow)
    format: str = "jpeg"      # "jpeg" (needs Pillow) | "png"
    jpeg_quality: int = 70

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "ScreenCaptureConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            interval_sec=max(0.1, float(data.get("interval_sec", 2.0) or 2.0)),
            monitor=int(data.get("monitor", 1)),
            max_width=int(data.get("max_width", 1280) or 0),
            format=str(data.get("format", "jpeg") or "jpeg").lower(),
            jpeg_quality=int(data.get("jpeg_quality", 70) or 70),
        )


def _default_grabber(monitor: int) -> Grabber:
    """An ``mss``-backed grabber (lazy import — mss is an optional dep)."""
    def grab() -> Tuple[bytes, Tuple[int, int]]:
        import mss  # noqa: PLC0415 - optional dep, imported only when enabled

        with mss.mss() as sct:
            mons = sct.monitors
            idx = monitor if 0 <= monitor < len(mons) else 1
            shot = sct.grab(mons[idx])
            # shot.rgb is tightly-packed RGB; shot.size is (width, height).
            return bytes(shot.rgb), (shot.width, shot.height)

    return grab


def _default_encoder(cfg: ScreenCaptureConfig) -> Encoder:
    """Downscale + JPEG via Pillow when available; else a full-res PNG via mss."""
    warned = {"no_pil": False}

    def encode(raw: bytes, size: Tuple[int, int]) -> bytes:
        w, h = size
        try:
            import io

            from PIL import Image  # noqa: PLC0415 - optional dep

            img = Image.frombytes("RGB", (w, h), raw)
            if cfg.max_width and w > cfg.max_width:
                nh = max(1, round(h * cfg.max_width / w))
                img = img.resize((cfg.max_width, nh))
            buf = io.BytesIO()
            if cfg.format == "png":
                img.save(buf, format="PNG")
            else:
                img.save(buf, format="JPEG", quality=cfg.jpeg_quality)
            return buf.getvalue()
        except ImportError:
            if not warned["no_pil"]:
                log.warning(
                    "Pillow not installed: sending a full-resolution PNG (large). "
                    "`pip install Pillow` to downscale + JPEG-encode frames."
                )
                warned["no_pil"] = True
            import mss.tools  # noqa: PLC0415

            return mss.tools.to_png(raw, size)

    return encode


class ScreenFrameFeed:
    """Background loop: grab → encode → ``set_frame`` every ``interval_sec``.

    Start/stop are idempotent; ``stop()`` clears the frame so the model doesn't
    keep seeing a stale screen. ``capture_once()`` does one grab+encode+feed and is
    the unit-test seam (inject ``grabber``/``encoder`` to avoid a real display)."""

    def __init__(
        self,
        set_frame: SetFrame,
        config: Optional[ScreenCaptureConfig] = None,
        *,
        grabber: Optional[Grabber] = None,
        encoder: Optional[Encoder] = None,
        observer: Optional[Callable[[Optional[bytes]], None]] = None,
    ) -> None:
        self._set_frame = set_frame
        self._cfg = config or ScreenCaptureConfig()
        self._grab = grabber or _default_grabber(self._cfg.monitor)
        self._encode = encoder or _default_encoder(self._cfg)
        # Optional second consumer of each frame (e.g. the visual memorizer). Cheap
        # + best-effort; it must never block or break the capture loop.
        self._observer = observer
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def config(self) -> ScreenCaptureConfig:
        return self._cfg

    def capture_once(self) -> Optional[bytes]:
        """Grab + encode one frame and feed it. Returns the bytes, or None on error
        (a transient grab/encode failure must never crash the feed)."""
        try:
            raw, size = self._grab()
            frame = self._encode(raw, size)
        except Exception:  # noqa: BLE001 - a flaky display/encoder skips one frame
            log.exception("screen capture failed; skipping this frame")
            return None
        self._set_frame(frame)
        if self._observer is not None:
            try:
                self._observer(frame)
            except Exception:  # noqa: BLE001 - a second consumer must never break capture
                log.exception("screen-frame observer failed; ignoring")
        return frame

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="screen-feed", daemon=True)
        self._thread.start()
        log.info(
            "screen-capture feed ON (every %.1fs, monitor %d, %s, max_width %d)",
            self._cfg.interval_sec, self._cfg.monitor, self._cfg.format, self._cfg.max_width,
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.capture_once()
            # wait() returns early when stop() fires, so teardown is prompt.
            self._stop.wait(self._cfg.interval_sec)

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.0)
        self._thread = None
        # Drop the stale frame so the assistant stops "seeing" the last screen.
        try:
            self._set_frame(None)
        except Exception:  # noqa: BLE001
            pass


def build_screen_feed(
    config: dict, runtime, *, observer: Optional[Callable[[Optional[bytes]], None]] = None
) -> Optional[ScreenFrameFeed]:
    """Build a feed wired to ``runtime.set_current_frame`` from the
    ``screen_capture`` config block, or ``None`` when it is disabled (the default).
    Returns None if the runtime can't accept frames (no set_current_frame).
    ``observer`` (e.g. the visual memorizer's ``observe``) gets each frame too."""
    cfg = ScreenCaptureConfig.from_dict(config.get("screen_capture"))
    if not cfg.enabled:
        return None
    setter = getattr(runtime, "set_current_frame", None)
    if not callable(setter):
        log.warning("screen_capture.enabled but the runtime has no set_current_frame; skipping")
        return None
    return ScreenFrameFeed(setter, cfg, observer=observer)
