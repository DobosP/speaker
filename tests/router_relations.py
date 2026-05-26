"""Shared helpers for router-intelligence tests.

Pure perturbation operators (used by metamorphic relations and to fuzz golden
rows) plus thin invocation wrappers over the two deterministic decision layers:

* :class:`utils.conversation_router.ConversationRouter` -- ``route`` helpers
* :class:`always_on_agent.speech_analyzer.LiveSpeechAnalyzer` -- ``analyze``

These layers are pure (text + context -> enum, no I/O), so everything here is
deterministic and requires no audio/LLM backends.
"""
from __future__ import annotations

from utils.conversation_router import (
    ConversationRouter,
    RouteContext,
    RouteDecision,
)
from always_on_agent.events import Mode
from always_on_agent.models import IntentDecision, SpeechObservation
from always_on_agent.speech_analyzer import LiveSpeechAnalyzer

DEFAULT_CAPS: tuple[str, ...] = ("system.time", "debug.echo")

_ROUTER = ConversationRouter()


# ── ConversationRouter wrappers ─────────────────────────────────────────────
def route(text: str, *, caps: tuple[str, ...] = DEFAULT_CAPS) -> RouteDecision:
    """Route a finalized transcript."""
    return _ROUTER.route(
        RouteContext(transcript=text, available_capabilities=tuple(caps))
    )


def route_partial(text: str, *, caps: tuple[str, ...] = DEFAULT_CAPS) -> RouteDecision:
    """Route a partial transcript (only high-confidence control is allowed)."""
    return _ROUTER.route_partial(
        RouteContext(
            transcript=text, is_partial=True, available_capabilities=tuple(caps)
        )
    )


# ── LiveSpeechAnalyzer wrappers ─────────────────────────────────────────────
def observe(text: str, *, is_final: bool = True) -> SpeechObservation:
    return LiveSpeechAnalyzer().observe(text, is_final=is_final)


def analyze(
    text: str, *, mode: Mode = Mode.PASSIVE, is_final: bool = True
) -> IntentDecision:
    analyzer = LiveSpeechAnalyzer()
    obs = analyzer.observe(text, is_final=is_final)
    return analyzer.decide(obs, mode)


# ── Perturbation operators (oracle-free metamorphic inputs) ─────────────────
def case_variants(text: str) -> list[str]:
    """Same utterance in different letter cases (normalization should erase this)."""
    return [text, text.upper(), text.lower(), text.title()]


def punctuation_variants(text: str) -> list[str]:
    """Trailing/embedded punctuation the normalizer strips to spaces."""
    return [text, f"{text}.", f"{text}!", f"{text}?", f"{text},", f"{text} ..."]


def whitespace_variants(text: str) -> list[str]:
    """Leading/trailing/doubled whitespace the normalizer collapses."""
    return [text, f"  {text}  ", f"\t{text}\n", text.replace(" ", "   ")]


def with_trailing_filler(text: str) -> list[str]:
    """Disfluencies appended after a control word (router prefix-match tolerates)."""
    return [text, f"{text} uh", f"{text} um", f"{text} now", f"{text} please"]


def with_leading_filler(text: str) -> list[str]:
    """Disfluencies *before* a control word (a known gap surface for both layers)."""
    return [f"uh {text}", f"um {text}", f"please {text}"]


def asr_repeat_words(text: str) -> str:
    """Duplicate every token, mimicking an ASR stutter (router collapses dups)."""
    out: list[str] = []
    for word in text.split():
        out.extend([word, word])
    return " ".join(out)
