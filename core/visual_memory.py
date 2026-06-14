"""Persistent visual memory: turn screen frames into recallable text. OFF BY DEFAULT.

When the screen-capture feed is on AND ``screen_capture.memorize`` is true, each new
frame is — **off the hot path**, on a background worker — turned into a short
multimodal CAPTION plus an OCR snippet and ingested as a ``vision`` memory, so the
assistant can later RECALL what was on screen through the same smart-recall layer
(``always_on_agent/recall.py`` → ``build_block``).

Privacy (``docs/target_architecture.md`` §9.7): captioning uses the **local**
multimodal model only; the persisted trace is tagged ``vision`` so a turn that
recalls it floats sensitivity PRIVATE (``core/capabilities.py``) and never rides a
public cloud chain; the owner can purge it (``MemoryManager.clear_observations``).

Everything is injectable (``caption_fn``/``ocr_fn``/``ingest``) so the cadence,
OCR, and captioning are unit-tested with **no display, no model, no tesseract,
no DB**. Missing optional deps (pytesseract/Pillow) or model degrade to an empty
component, never a crash.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Callable, Optional

from always_on_agent.untrusted import redact_pii

log = logging.getLogger("speaker.visual_memory")

CaptionFn = Callable[[bytes], str]
OcrFn = Callable[[bytes], str]
IngestFn = Callable[[str], None]

# Structured-ish, entity-rich one-liner: naming the app + the activity + key visible
# items makes the stored caption match far better on recall (keyword overlap +
# embeddings) than bare prose, and gives the PII redactor more to catch. Still one
# line + directive so a small local multimodal model returns something usable.
_CAPTION_PROMPT = (
    "Describe this screen in ONE short line, as: <app or website> -- <what the user "
    "is doing> -- <key visible items or names>. Be specific and concise. No preamble."
)


def _dhash(frame: bytes, size: int = 8) -> Optional[int]:
    """64-bit perceptual difference-hash of a frame, or ``None`` if Pillow is
    absent / the bytes don't decode.

    Decode -> grayscale -> resize to ``(size+1) x size`` -> compare each pixel to
    its right neighbour (row-major) -> one bit each = ``size*size`` bits. Tiny
    visual changes (cursor blink, clock tick, 1px scroll) leave the gradient almost
    unchanged, so near-identical frames hash to a small Hamming distance -- unlike
    the exact SHA1, which any single changed byte defeats. Stdlib + optional Pillow
    (already the OCR dep); degrades to ``None`` -> the caller falls back to SHA1."""
    try:
        import io  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415 - optional dep (shared with OCR)

        # Area-averaging downscale (LANCZOS) so each of the tiny output pixels
        # reflects its source block -- more stable on text-heavy screens than the
        # BICUBIC default for an extreme >100x downscale. Falls back to the default
        # filter on a very old Pillow without the enum.
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", None)
        img = Image.open(io.BytesIO(frame)).convert("L").resize((size + 1, size), resample)
    except Exception:  # noqa: BLE001 - missing dep / undecodable frame -> no phash
        return None
    px = list(img.getdata())
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | (1 if px[base + col] > px[base + col + 1] else 0)
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass
class VisualMemoryConfig:
    """Knobs for visual memory, read from the ``screen_capture`` config block.

    Default OFF (``memorize`` false) so a default run — even with the screen feed
    on — persists nothing. ``min_interval_sec`` is the floor between stored
    observations (captioning is a multimodal LLM call; we do NOT run it every
    frame); ``on_change`` skips a frame byte-identical to the last stored one."""

    enabled: bool = False
    min_interval_sec: float = 30.0
    on_change: bool = True
    caption: bool = True
    ocr: bool = True
    max_chars: int = 280
    ocr_max_chars: int = 200
    redact_pii: bool = True
    phash_threshold: int = 4

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "VisualMemoryConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("memorize", False)),
            min_interval_sec=max(0.0, float(data.get("memorize_min_interval_sec", 30.0) or 0.0)),
            on_change=bool(data.get("memorize_on_change", True)),
            caption=bool(data.get("memorize_caption", True)),
            ocr=bool(data.get("memorize_ocr", True)),
            max_chars=int(data.get("memorize_max_chars", 280) or 280),
            ocr_max_chars=int(data.get("memorize_ocr_max_chars", 200) or 200),
            # §9.7 image-security: scrub cards/SSN/keys/email/phone from OCR'd screen
            # text BEFORE it becomes a durable 'vision' memory. Default ON.
            redact_pii=bool(data.get("memorize_redact_pii", True)),
            # on_change near-dup gate: skip a frame within this Hamming distance of
            # the last stored one's perceptual hash (0 = require an exact perceptual
            # match; larger = more aggressive dedup). Only used when Pillow is present.
            phash_threshold=max(0, int(data.get("memorize_phash_threshold", 4) or 0)),
        )


def default_ocr_fn(max_chars: int = 200) -> OcrFn:
    """An OCR callable backed by pytesseract+Pillow (lazy, optional). Returns ``''``
    on any missing-dep / runtime error so visual memory degrades to caption-only."""
    warned = {"no_dep": False}

    def ocr(frame: bytes) -> str:
        try:
            import io  # noqa: PLC0415

            import pytesseract  # noqa: PLC0415 - optional dep
            from PIL import Image  # noqa: PLC0415 - optional dep

            text = pytesseract.image_to_string(Image.open(io.BytesIO(frame)))
        except ImportError:
            if not warned["no_dep"]:
                log.warning("pytesseract/Pillow not installed: visual memory OCR disabled "
                            "(caption-only). `pip install pytesseract pillow` + tesseract.")
                warned["no_dep"] = True
            return ""
        except Exception:  # noqa: BLE001 - a flaky OCR pass must never crash the worker
            return ""
        return " ".join(text.split())[:max_chars]

    return ocr


def llm_caption_fn(llm) -> CaptionFn:
    """A caption callable backed by the LOCAL multimodal model (§9.7). Returns
    ``''`` on any error (e.g. a text-only/cloud model) so OCR can still carry."""
    def caption(frame: bytes) -> str:
        try:
            out = llm.generate(_CAPTION_PROMPT, images=[frame])
        except Exception:  # noqa: BLE001 - captioning is best-effort, never fatal
            return ""
        return " ".join((out or "").split())

    return caption


class VisualMemorizer:
    """Throttled, on-change, off-hot-path producer of ``vision`` memories.

    ``observe(frame)`` is called from the capture loop and is CHEAP: it fingerprints
    + throttles + enqueues (bounded, drop-oldest), so the real-time loop never waits
    on OCR/captioning. The background worker does the slow work and ingests the
    composed trace. Start/stop are idempotent."""

    def __init__(
        self,
        *,
        ingest: IngestFn,
        caption_fn: Optional[CaptionFn] = None,
        ocr_fn: Optional[OcrFn] = None,
        config: Optional[VisualMemoryConfig] = None,
    ) -> None:
        self._ingest = ingest
        self._cfg = config or VisualMemoryConfig()
        self._caption_fn = caption_fn
        self._ocr_fn = ocr_fn
        self._q: "Queue[bytes]" = Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_fp: Optional[str] = None
        self._last_phash: Optional[int] = None
        self._last_at = 0.0

    @property
    def config(self) -> VisualMemoryConfig:
        return self._cfg

    def observe(self, frame: Optional[bytes]) -> None:
        """On-change + throttle gate, then enqueue. Never blocks the caller."""
        if not frame or self._stop.is_set():
            return
        now = time.monotonic()
        # Cheap throttle pre-check (atomic float read, no lock): a frame inside the
        # min_interval is discarded regardless, so don't pay the perceptual decode
        # for it. The authoritative throttle check below (under the lock) still
        # decides; this only avoids wasted work on the common throttled frame.
        last_at = self._last_at
        if last_at and (now - last_at) < self._cfg.min_interval_sec:
            return
        fp = hashlib.sha1(frame).hexdigest()
        # Perceptual hash for the near-dup gate (None when Pillow is absent /
        # undecodable -> the exact-SHA1 path is used instead). A full-frame decode
        # (~tens of ms at 1280px) -- fine on the seconds-cadence screen-feed thread,
        # off the real-time ASR/TTS path.
        phash = _dhash(frame) if self._cfg.on_change else None
        with self._lock:
            if self._last_at and (now - self._last_at) < self._cfg.min_interval_sec:
                return
            if self._cfg.on_change:
                # Skip a frame that is VISUALLY ~identical to the last STORED one
                # (cursor blink, clock tick, 1px scroll) so we don't burn a caption
                # call + a memory row on noise. Perceptual when available, else the
                # original exact-SHA1 behaviour.
                if phash is not None and self._last_phash is not None:
                    if _hamming(phash, self._last_phash) <= self._cfg.phash_threshold:
                        return
                elif phash is None and fp == self._last_fp:
                    return
            self._last_fp = fp
            # Track the phash of the LAST STORED frame unconditionally (None for an
            # undecodable/no-Pillow frame), so a stale hash from an earlier frame can
            # never mis-gate a later one -- it stays consistent with _last_fp.
            self._last_phash = phash
            self._last_at = now
        # Bounded, drop-oldest: a slow worker must never stall the capture loop.
        try:
            self._q.put_nowait(frame)
        except Full:
            try:
                self._q.get_nowait()
            except Empty:
                pass
            try:
                self._q.put_nowait(frame)
            except Full:
                pass

    @staticmethod
    def _safe(fn: Optional[Callable[[bytes], str]], frame: bytes) -> str:
        if fn is None:
            return ""
        try:
            return (fn(frame) or "").strip()
        except Exception:  # noqa: BLE001 - one component failing must not lose the other
            return ""

    def compose(self, frame: bytes) -> str:
        """Caption + OCR snippet → one trimmed trace line (``''`` if both empty).

        Each component is guarded independently: if captioning fails, the OCR
        snippet still carries (and vice versa)."""
        caption = self._safe(self._caption_fn, frame) if self._cfg.caption else ""
        ocr_raw = self._safe(self._ocr_fn, frame) if self._cfg.ocr else ""
        # §9.7 image-security: scrub PII (cards/SSN/keys/email/phone) from BOTH the
        # caption AND the OCR before any trim/persist. The multimodal caption can
        # transcribe a visible secret just as the OCR can ("a form showing card
        # 4111..."), so both screen-pixels->record paths must be redacted; doing it
        # on the FULL text before the trims means a number split by a trim can't
        # survive half-redacted.
        if self._cfg.redact_pii:
            if caption:
                caption = redact_pii(caption)
            if ocr_raw:
                ocr_raw = redact_pii(ocr_raw)
        ocr = ocr_raw[: self._cfg.ocr_max_chars].strip() if self._cfg.ocr else ""
        parts = []
        if caption:
            parts.append(caption)
        if ocr:
            parts.append(f"on-screen text: {ocr}")
        return " | ".join(parts)[: self._cfg.max_chars].strip()

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._q.get(timeout=0.5)
            except Empty:
                continue
            try:
                trace = self.compose(frame)
                if trace:
                    self._ingest(trace)
            except Exception:  # noqa: BLE001 - a bad frame must never kill the worker
                log.exception("visual memory: produce/ingest failed; skipping a frame")

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name="visual-memory", daemon=True)
        self._thread.start()
        log.info(
            "visual memory ON (min_interval %.0fs, caption=%s, ocr=%s)",
            self._cfg.min_interval_sec, self._cfg.caption, self._cfg.ocr,
        )

    def stop(self) -> None:
        """Stop the worker. Idempotent. Queued-but-undrained frames are dropped
        (best-effort — visual memory is ambient, not transactional); an in-flight
        caption may run briefly after this returns."""
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.0)
        # Only forget the worker if it actually exited; if a slow caption blew the
        # join timeout, keep the handle so start()'s is_alive() guard refuses to
        # spawn a duplicate over the still-running worker.
        if t is None or not t.is_alive():
            self._thread = None


def build_visual_memorizer(config: dict, runtime, llm) -> Optional[VisualMemorizer]:
    """Build a memorizer from the ``screen_capture`` block, or ``None`` when
    ``memorize`` is off / nothing can produce a trace. Wire its ``observe`` as the
    screen feed's observer (``build_screen_feed(..., observer=...)``)."""
    cfg = VisualMemoryConfig.from_dict(config.get("screen_capture"))
    if not cfg.enabled:
        return None
    memory = getattr(runtime, "memory", None)
    if memory is None:
        log.warning("screen_capture.memorize on but runtime has no memory; skipping")
        return None
    # §9.7 defense-in-depth: caption ONLY on a bare local handle. Prefer the
    # factory-stamped local_main / HedgeLLM.local; if the resolved client is still
    # a cloud-capable wrapper (a caller forgot to unwrap), hard-skip captioning
    # (OCR-only) rather than risk encoding a raw frame to a cloud chain.
    local_llm = getattr(llm, "local_main", None) or getattr(llm, "local", None) or llm
    if local_llm is not None and type(local_llm).__name__ in ("HedgeLLM", "SensitivityRouterLLM"):
        log.warning("visual memory: main LLM is cloud-capable and no local handle was found; "
                    "captioning DISABLED (OCR-only) to keep screen frames on-device (§9.7)")
        local_llm = None
    caption_fn = llm_caption_fn(local_llm) if (cfg.caption and local_llm is not None) else None
    ocr_fn = default_ocr_fn(cfg.ocr_max_chars) if cfg.ocr else None
    if caption_fn is None and ocr_fn is None:
        log.warning("screen_capture.memorize on but both caption and ocr are off; skipping")
        return None

    def ingest(text: str) -> None:
        memory.add(text, tags=("vision",))

    return VisualMemorizer(ingest=ingest, caption_fn=caption_fn, ocr_fn=ocr_fn, config=cfg)
