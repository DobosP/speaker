"""Read-only on-screen element identification ("find / identify things on screen")
for the computer-use capability. READ-ONLY by construction: it locates candidate UI
elements + their screen coordinates and returns them as DATA; it never moves the
cursor, types, or clicks.

Safety (the computer-use threat model): every label it surfaces is
attacker-controllable SCREEN text, so it is (a) PII-redacted (a visible password /
card must not be echoed back), (b) ``wrap_untrusted()``-fenced as untrusted DATA,
and (c) tagged ``Origin.SCREEN`` -- which means under the action gate
(:mod:`always_on_agent.origin`) a screen label can MATCH the owner's spoken target
but can NEVER originate or parameterize a real action. All grounding (OCR; a11y is
a future backend) runs on-device; raw pixels never leave (§9.7) -- the capability
is ``egress="local"`` and marks its result PRIVATE so routing pins it on-device.

The matching logic (:func:`find_targets`) is pure + stdlib so it unit-tests with
hand-built word boxes -- no display, no models, no tesseract.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from always_on_agent.capabilities import CapabilityRegistry, CapabilityResult, CapabilitySpec
from always_on_agent.origin import Origin
from always_on_agent.text import keywords, normalize_text
from always_on_agent.untrusted import redact_pii, wrap_untrusted

log = logging.getLogger("speaker.ui_grounding")

# A screenshot grabber returns encoded image bytes (or None when unavailable); an
# OCR backend returns word boxes. Both injectable so the capability unit-tests with
# fakes (no display, no tesseract).
CaptureFn = Callable[[], Optional[bytes]]
OcrFn = Callable[[bytes], "list[dict]"]


@dataclass(frozen=True)
class Element:
    """One located on-screen element. ``label`` is untrusted screen text."""

    label: str
    left: int
    top: int
    width: int
    height: int
    score: float
    source: str = "ocr"

    @property
    def center(self) -> tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)


def find_targets(
    words: Sequence[dict], query: str, *, limit: int = 8, min_conf: float = 40.0
) -> list[Element]:
    """Rank OCR/a11y word boxes against the spoken ``query`` target. Pure + testable.

    ``words`` items: ``{text,left,top,width,height,conf}``. With a query, keep only
    boxes overlapping its tokens (ranked by overlap, conf as tiebreak); with an
    empty query ("what's on screen"), return the highest-confidence boxes. Low-
    confidence noise (< ``min_conf``) is dropped."""
    # Stopword-filter the spoken target so decorative words ("where is the ... button")
    # don't tie/outrank the real target -- same term extraction the rest of the brain
    # uses. Falls back to the all-boxes path when the query is only stopwords.
    q = set(keywords(query))
    scored: list[Element] = []
    for w in words:
        text = str(w.get("text", "") or "").strip()
        if not text:
            continue
        try:
            conf = float(w.get("conf", 0) or 0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < min_conf:
            continue
        toks = set(normalize_text(text).split())
        if not toks:
            continue
        overlap = len(q & toks)
        if q and not overlap:
            continue  # a target was named but this box doesn't match it
        score = overlap + min(conf, 100.0) / 1000.0  # overlap primary, conf tiebreak
        scored.append(Element(
            text,
            int(w.get("left", 0) or 0), int(w.get("top", 0) or 0),
            int(w.get("width", 0) or 0), int(w.get("height", 0) or 0),
            score, str(w.get("source", "ocr") or "ocr"),
        ))
    scored.sort(key=lambda e: e.score, reverse=True)
    return scored[: max(1, limit)]


def render_elements(elements: Sequence[Element]) -> str:
    """Render located elements as an untrusted, PII-redacted, fenced DATA block --
    safe to hand to the LLM. ``''``-safe via a placeholder so the model gets a
    definite "nothing matched" rather than an empty injection envelope."""
    if not elements:
        body = "(no matching on-screen elements found)"
    else:
        body = "\n".join(
            f'"{redact_pii(e.label)}" at screen ({e.center[0]},{e.center[1]})'
            for e in elements
        )
    return wrap_untrusted(body, source="screen")


def ocr_words(image_bytes: bytes) -> list[dict]:
    """Word boxes via pytesseract ``image_to_data`` (lazy/optional). Returns ``[]``
    on any missing-dep / decode error so identification degrades, never crashes."""
    try:
        import io  # noqa: PLC0415

        import pytesseract  # noqa: PLC0415 - optional dep (shared with OCR)
        from PIL import Image  # noqa: PLC0415 - optional dep

        data = pytesseract.image_to_data(
            Image.open(io.BytesIO(image_bytes)), output_type=pytesseract.Output.DICT
        )
    except Exception:  # noqa: BLE001 - missing dep / undecodable -> no grounding
        return []
    out: list[dict] = []
    for i in range(len(data.get("text", []))):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        out.append({
            "text": text, "left": data["left"][i], "top": data["top"][i],
            "width": data["width"][i], "height": data["height"][i], "conf": data["conf"][i],
        })
    return out


def _default_capture(monitor: int = 1) -> CaptureFn:
    """A screenshot grabber reusing core/screen_capture's mss+encoder. Returns None
    when mss/capture is unavailable (read-only identification then degrades)."""
    def capture() -> Optional[bytes]:
        try:
            from .screen_capture import ScreenCaptureConfig, _default_encoder, _default_grabber  # noqa: PLC0415

            cfg = ScreenCaptureConfig(monitor=monitor)
            raw, size = _default_grabber(monitor)()
            return _default_encoder(cfg)(raw, size)
        except Exception:  # noqa: BLE001 - no display / no mss -> unavailable
            return None
    return capture


def attach_computer_use_capability(
    registry: CapabilityRegistry,
    *,
    enabled: bool = False,
    monitor: int = 1,
    capture_fn: Optional[CaptureFn] = None,
    ocr_fn: Optional[OcrFn] = None,
) -> CapabilityRegistry:
    """Register the READ-ONLY ``screen.identify`` capability (default-OFF -- only
    when ``enabled``). It captures the screen, locates elements matching the spoken
    target, and returns a fenced, PII-redacted, PRIVATE result -- it never actuates.

    No actuating ``gui.*`` capability is registered here: input control is a
    separate, harder-gated slice (owner speaker-ID + the action chokepoint in
    :mod:`always_on_agent.origin`) and must not ship on this read-only path."""
    if not enabled:
        return registry
    capture = capture_fn or _default_capture(monitor)
    ocr = ocr_fn or ocr_words

    def screen_identify(query: str, context: dict[str, object]) -> CapabilityResult:
        # sensitivity=private marks intent that screen content stays on-device.
        # egress=False is intentional (NOT egress="local"): the react.py /
        # capabilities.py egress-truthiness check decides whether to EXTRA-wrap a
        # tool result as untrusted; we skip that because render_elements already
        # wrap_untrusted-fences the screen text at the source (do not "fix" to True
        # -> double-wrap). NOTE: this sensitivity flag is advisory until the
        # actuator slice wires the §9.7 float; that is why the spec is
        # planner_tool=False (kept off the cloud-routable planner) for now.
        data = {"sensitivity": "private", "egress": False, "origin": Origin.SCREEN.value}
        try:
            image = capture()
        except Exception:  # noqa: BLE001 - read-only, never fatal
            log.exception("screen.identify: capture failed")
            image = None
        if not image:
            return CapabilityResult(
                True, wrap_untrusted("(screen capture unavailable)", source="screen"),
                data={**data, "count": 0},
            )
        words = ocr(image)
        elements = find_targets(words, query or "")
        return CapabilityResult(
            True, render_elements(elements), data={**data, "count": len(elements)},
        )

    registry.register(
        "screen.identify", screen_identify,
        spec=CapabilitySpec(
            name="screen.identify",
            summary="Identify what is on the screen and where UI elements are (read-only).",
            when_to_use="When the user asks what is on their screen, or where a button/field/link is. Read-only: it locates things, it does not click or type.",
            # planner_tool=False: keep screen text OFF the ReAct planner (which can
            # route to a cloud tier) until the §9.7 sensitivity-float for screen
            # content is actually enforced -- otherwise OCR'd screen text could
            # egress via planner synthesis. The actuator slice wires the float +
            # a local-tier pin, then this can become a planner tool.
            egress="local", speaks=True, side_effecting=False, planner_tool=False, user_facing=True,
        ),
    )
    log.info("computer-use: screen.identify (read-only) registered")
    return registry
