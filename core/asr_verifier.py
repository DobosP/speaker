"""Pure, evidence-constrained final-ASR verification.

The production baseline is a fallback and rendering candidate, never an
additional vote.  Only the independent streaming, offline, and verifier
recognizers contribute support.  A text change requires an exact normalized
quorum of at least two recognizers and may not change controller-owned STOP,
confirm/deny, or mode-switch semantics. Two decoded-empty independent models
may suppress only a one-word, non-control streaming-only hypothesis.

No transcript is logged or included in routine object representations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import unicodedata

from always_on_agent.speech_analyzer import exact_control_class


class AsrConsensusOutcome(str, Enum):
    """Aggregate-safe result of acoustic consensus resolution."""

    CONSENSUS = "consensus"
    NO_QUORUM = "no_quorum"
    TIE = "tie"
    CONTROL_GUARD = "control_guard"
    EMPTY_VETO = "empty_veto"
    ERROR = "error"


class AsrConsensusSource(str, Enum):
    """Aggregate-safe source whose existing rendering was returned."""

    BASELINE = "baseline"
    OFFLINE = "offline"
    VERIFIER = "verifier"
    STREAMING = "streaming"


@dataclass(frozen=True)
class AsrConsensusDecision:
    """One final choice with transcript text excluded from ``repr``."""

    chosen: str = field(repr=False)
    outcome: AsrConsensusOutcome
    source: AsrConsensusSource
    support: int
    changed: bool


def _exact_tokens(text: str) -> tuple[str, ...]:
    """Unicode-preserving, case/punctuation-insensitive word tokens.

    The WER helper is intentionally ASCII-only, which is appropriate for the
    current English aggregate gate but unsafe for authority-bearing consensus:
    dropping non-ASCII words could make two different hypotheses appear equal.
    Keep every Unicode letter/number/mark and every symbol; ignore only spacing
    and punctuation separators. Curly apostrophes normalize to ASCII inside a
    word so ordinary typography does not prevent an otherwise exact quorum.
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            tokens.append("".join(current))
            current.clear()

    for character in normalized:
        category = unicodedata.category(character)
        if character.isalnum() or category.startswith("M"):
            current.append(character)
        elif character in {"'", "’"} and current:
            current.append("'")
        elif category.startswith("S"):
            flush()
            tokens.append(character)
        else:
            flush()
    flush()
    return tuple(tokens)


def _fallback(
    baseline: str,
    outcome: AsrConsensusOutcome,
    *,
    support: int = 0,
) -> AsrConsensusDecision:
    return AsrConsensusDecision(
        chosen=baseline,
        outcome=outcome,
        source=AsrConsensusSource.BASELINE,
        support=support,
        changed=False,
    )


def verify_asr_consensus(
    *,
    baseline_selected: str,
    streaming: str,
    verifier: str,
    offline: str | None = None,
) -> AsrConsensusDecision:
    """Choose one existing rendering after exact independent-source quorum.

    Normalization is punctuation/case-insensitive but, unlike the English-only
    aggregate WER helper, preserves Unicode words and symbols. Empty normalized
    hypotheses do not vote. When a winning transcript exists, rendering
    priority is baseline, offline, verifier, then streaming. One deliberately
    narrow exception lets independent decoded-empty offline and verifier
    results veto a one-word, non-control streaming-only hypothesis. This removes
    a measured false endpoint without allowing either model to invent text or
    erase STOP/confirm/deny/mode meaning. Any invalid input or unexpected
    failure keeps a valid string baseline with safe metadata and no exception
    detail.
    """

    baseline = baseline_selected if isinstance(baseline_selected, str) else ""
    if (
        not isinstance(baseline_selected, str)
        or not isinstance(streaming, str)
        or not isinstance(verifier, str)
        or (offline is not None and not isinstance(offline, str))
    ):
        return _fallback(baseline, AsrConsensusOutcome.ERROR)

    try:
        evidence = (
            (AsrConsensusSource.STREAMING, streaming),
            (AsrConsensusSource.OFFLINE, offline),
            (AsrConsensusSource.VERIFIER, verifier),
        )
        normalized: dict[AsrConsensusSource, tuple[str, ...]] = {}
        groups: dict[tuple[str, ...], list[AsrConsensusSource]] = {}
        for source, text in evidence:
            if text is None:
                continue
            tokens = _exact_tokens(text)
            normalized[source] = tokens
            if tokens:
                groups.setdefault(tokens, []).append(source)

        # A decoded empty is evidence only when *both* independent endpoint
        # models returned it. Distinguish offline="" (decoded empty) from
        # offline=None (unavailable/error), and keep the exception to the exact
        # measured shape: the baseline is the streaming recognizer's single
        # normalized word. Multi-word turns and controller-owned controls retain
        # the established fallback.
        streaming_tokens = normalized.get(AsrConsensusSource.STREAMING, ())
        offline_tokens = normalized.get(AsrConsensusSource.OFFLINE)
        verifier_tokens = normalized.get(AsrConsensusSource.VERIFIER, ())
        baseline_tokens = _exact_tokens(baseline)
        if (
            offline is not None
            and offline_tokens == ()
            and verifier_tokens == ()
            and len(streaming_tokens) == 1
            and baseline_tokens == streaming_tokens
        ):
            if exact_control_class("") != exact_control_class(baseline):
                return _fallback(
                    baseline,
                    AsrConsensusOutcome.CONTROL_GUARD,
                    support=2,
                )
            return AsrConsensusDecision(
                chosen="",
                outcome=AsrConsensusOutcome.EMPTY_VETO,
                source=AsrConsensusSource.OFFLINE,
                support=2,
                changed=bool(baseline),
            )

        max_support = max((len(sources) for sources in groups.values()), default=0)
        if max_support == 0:
            return _fallback(baseline, AsrConsensusOutcome.NO_QUORUM)
        leaders = tuple(
            tokens for tokens, sources in groups.items() if len(sources) == max_support
        )
        if len(leaders) != 1:
            return _fallback(
                baseline,
                AsrConsensusOutcome.TIE,
                support=max_support,
            )
        if max_support < 2:
            return _fallback(
                baseline,
                AsrConsensusOutcome.NO_QUORUM,
                support=max_support,
            )
        winner = leaders[0]

        renderers: tuple[tuple[AsrConsensusSource, str | None], ...] = (
            (AsrConsensusSource.BASELINE, baseline),
            (AsrConsensusSource.OFFLINE, offline),
            (AsrConsensusSource.VERIFIER, verifier),
            (AsrConsensusSource.STREAMING, streaming),
        )
        selected_source = AsrConsensusSource.BASELINE
        chosen = baseline
        for source, text in renderers:
            if text is None:
                continue
            tokens = (
                baseline_tokens
                if source is AsrConsensusSource.BASELINE
                else normalized.get(source, ())
            )
            if tokens == winner:
                selected_source = source
                chosen = text
                break

        changed = chosen != baseline
        if changed and exact_control_class(chosen) != exact_control_class(baseline):
            return _fallback(
                baseline,
                AsrConsensusOutcome.CONTROL_GUARD,
                support=max_support,
            )
        return AsrConsensusDecision(
            chosen=chosen,
            outcome=AsrConsensusOutcome.CONSENSUS,
            source=selected_source,
            support=max_support,
            changed=changed,
        )
    except Exception:  # noqa: BLE001 - fail closed without transcript details
        return _fallback(baseline, AsrConsensusOutcome.ERROR)


__all__ = [
    "AsrConsensusDecision",
    "AsrConsensusOutcome",
    "AsrConsensusSource",
    "verify_asr_consensus",
]
