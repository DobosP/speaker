"""Opt-in TTS expressive markup -> per-utterance (speaker_id, speed).

When its system prompt teaches the grammar (an explicit, per-deployment opt-in),
the LLM may prefix a spoken sentence with a small leading bracket tag carrying
voice / emotion / rate directives, e.g.

    [emotion:calm voice:warm] Here is a gentle explanation.
    [voice:narrator rate:1.05] And now the exciting part!

:func:`parse_tts_markup` strips that one leading tag and returns the directives;
:func:`resolve_tts_params` maps them to a concrete ``(speaker_id, speed)`` for
the synthesizer. This is the whole "emotion + voice diversity" capability and it
is deliberately CHEAP: no extra model, no added latency -- a regex scan plus two
dict lookups per sentence. Kokoro's only realistic expressivity levers via
sherpa-onnx are *which voice* (sid, 103 of them) and *how fast* (speed); there is
no latent style vector exposed, so emotion is modelled as voice-choice +
rate-as-affect.

Design rules (both functions are pure, stdlib-only, and never raise on bad input
-- they must never corrupt speech):

* A leading bracket is treated as a directive ONLY when it contains at least one
  KNOWN key, so ordinary text that merely starts with ``[`` (footnote markers
  like ``[1]``, markdown, code) passes through untouched.
* Unknown voice names / emotions / absurd rates fall back to the configured
  defaults; speed is clamped to a safe band and the speaker id to the model's
  real range.

This is sherpa/desktop-only and OFF by default (gated by ``SherpaConfig.tts_markup``).
It is intentionally NOT part of :mod:`core.contract` -- the cross-language brain
contract the Dart mobile loop mirrors -- because mobile does not share this path.
"""
from __future__ import annotations

import re
from typing import Mapping, Optional

# One leading bracketed tag, <=120 chars of non-bracket content. Only the FIRST
# leading tag is consumed (the directive for THIS utterance); any bracket later
# in the sentence is real text and never touched.
_TAG_RE = re.compile(r"^\s*\[([^\[\]]{1,120})\]\s*")

# The directive keys understood. A leading bracket lacking ALL of these is NOT a
# TTS tag, so "[1]" / "[see note]" pass through as ordinary spoken text.
KNOWN_KEYS = ("emotion", "voice", "rate", "speed")

# Safe synthesis-speed band: a runaway "[rate:50]" must never produce unusable
# audio. Matches the dataclass defaults (SherpaConfig.tts_speed_min/max).
DEFAULT_SPEED_MIN = 0.5
DEFAULT_SPEED_MAX = 2.0


def parse_tts_markup(text: str) -> tuple[str, dict[str, str]]:
    """Split a leading ``[emotion:.. voice:.. rate:..]`` tag off ``text``.

    Returns ``(clean_text, directives)``. With no leading tag -- or a leading
    bracket holding none of :data:`KNOWN_KEYS` -- returns the original text and an
    empty dict (a byte-identical pass-through). Accepts ``k:v`` or ``k=v`` pairs
    separated by spaces and/or commas. Never raises."""
    if not text:
        return text, {}
    try:
        m = _TAG_RE.match(text)
        if not m:
            return text, {}
        directives: dict[str, str] = {}
        for tok in m.group(1).replace(",", " ").split():
            sep = ":" if ":" in tok else ("=" if "=" in tok else "")
            if not sep:
                continue
            key, _, val = tok.partition(sep)
            key = key.strip().lower()
            val = val.strip()
            if key in KNOWN_KEYS and val:
                directives[key] = val
        if not directives:
            return text, {}  # a non-directive bracket ("[1]") -> leave intact
        return text[m.end():], directives
    except Exception:  # pragma: no cover - defensive; speech must survive bad markup
        return text, {}


def resolve_tts_params(
    directives: Optional[Mapping[str, str]],
    *,
    default_sid: int,
    default_speed: float,
    voice_map: Optional[Mapping[str, int]] = None,
    emotion_speed_map: Optional[Mapping[str, float]] = None,
    num_speakers: int = 0,
    speed_min: float = DEFAULT_SPEED_MIN,
    speed_max: float = DEFAULT_SPEED_MAX,
) -> tuple[int, float]:
    """Map parsed ``directives`` to a concrete ``(speaker_id, speed)``.

    ``voice`` -> sid via ``voice_map``; ``emotion`` -> a speed multiplier via
    ``emotion_speed_map``; an explicit ``rate``/``speed`` -> a further multiplier.
    Empty/absent directives return the defaults unchanged (byte-identical). Every
    lookup is fail-soft: an unknown voice keeps ``default_sid``, an unknown emotion
    contributes no multiplier, a non-numeric rate is ignored. The final speed is
    clamped to ``[speed_min, speed_max]`` and the sid validated against
    ``num_speakers`` (when known, >0) -- an out-of-range sid falls back to the
    default. Never raises."""
    sid = int(default_sid)
    speed = float(default_speed)
    if not directives:
        return sid, speed
    voice_map = voice_map or {}
    emotion_speed_map = emotion_speed_map or {}

    vname = directives.get("voice")
    if vname and vname in voice_map:
        try:
            sid = int(voice_map[vname])
        except (TypeError, ValueError):
            pass

    emo = directives.get("emotion")
    if emo and emo in emotion_speed_map:
        try:
            speed *= float(emotion_speed_map[emo])
        except (TypeError, ValueError):
            pass

    rate = directives.get("rate") or directives.get("speed")
    if rate:
        try:
            speed *= float(rate)
        except (TypeError, ValueError):
            pass

    speed = min(max(speed, speed_min), speed_max)
    if num_speakers and not (0 <= sid < num_speakers):
        sid = int(default_sid)
    return sid, speed


def build_markup_guidance(
    voices: Optional[list] = None, emotions: Optional[list] = None
) -> str:
    """A system-prompt snippet teaching the LLM the expressive-markup grammar.

    Returns a short instruction listing the configured ``voices`` / ``emotions``
    and the leading-tag format, or ``""`` when neither is offered (so the LLM
    stays tag-unaware -- the default). The runtime appends the result to the
    system prompt only when ``SherpaConfig.tts_markup`` is on, so a deployment
    enables the whole capability by setting that flag plus the maps -- no manual
    prompt edits."""
    voices = [str(v) for v in (voices or []) if v]
    emotions = [str(e) for e in (emotions or []) if e]
    if not voices and not emotions:
        return ""
    parts = [
        "Expressive voice: you may begin a sentence with ONE optional tag in "
        "square brackets to set how it is spoken. Put it at the very start; it is "
        "removed before speaking, so the listener never hears it. Use it sparingly "
        "-- only when emotion or a voice change genuinely helps; most sentences "
        "need no tag."
    ]
    if emotions:
        parts.append("Emotions: " + ", ".join(emotions) + ".")
    if voices:
        parts.append("Voices: " + ", ".join(voices) + ".")
    example = "["
    if emotions:
        example += f"emotion:{emotions[0]}"
    if voices:
        example += (" " if emotions else "") + f"voice:{voices[0]}"
    example += "]"
    parts.append(
        f"Format example: '{example} Your sentence here.' Use ONLY the names "
        "listed above; anything else is ignored."
    )
    return " ".join(parts)
