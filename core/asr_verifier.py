"""Pure, evidence-constrained final-ASR verification.

The production baseline is a fallback and rendering candidate, never an
additional vote.  Only the independent streaming, offline, and verifier
recognizers contribute support. The established consensus API requires an
exact normalized quorum of at least two recognizers and may not change controller-owned STOP,
confirm/deny, or mode-switch semantics. Two decoded-empty independent models
may suppress only a one-word, non-control streaming-only hypothesis.

A separate corroborated-primary dialogue fallback requires an independent
exact vote for every changed span, without changing control or action trust.
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



# These limits bound the optional quadratic dialogue comparison; they do not
# crop ASR audio/text or alter the established exact-consensus path.
_PRIMARY_MAX_CHARS = 8192
_PRIMARY_MAX_TOKENS = 128


def _unique_edits(
    source: tuple[str, ...], target: tuple[str, ...], *, allow_deletions: bool = False
) -> tuple[tuple[int, int, tuple[str, ...]], ...] | None:
    """Exact edits on the sole optimal alignment, or abstain on any ambiguity.

    Count every minimum-cost Levenshtein path (saturating at two). A chosen
    backtrace alone cannot establish uniqueness, especially with repeated
    words. A deleted source token cannot support a primary correction; reject
    it even inside a mixed span unless mapping a separate verifier error.
    """
    rows, columns = len(source), len(target)
    costs = [[0] * (columns + 1) for _ in range(rows + 1)]
    paths = [[1] * (columns + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        costs[i][0] = i
    for j in range(columns + 1):
        costs[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            diagonal = costs[i - 1][j - 1] + (source[i - 1] != target[j - 1])
            deletion = costs[i - 1][j] + 1
            insertion = costs[i][j - 1] + 1
            best = costs[i][j] = min(diagonal, deletion, insertion)
            paths[i][j] = min(2, (
                (paths[i - 1][j - 1] if diagonal == best else 0)
                + (paths[i - 1][j] if deletion == best else 0)
                + (paths[i][j - 1] if insertion == best else 0)
            ))
    if paths[rows][columns] != 1:
        return None

    i, j = rows, columns
    steps: list[tuple[int, int, tuple[str, ...] | None]] = []
    while i or j:
        if i and j and costs[i][j] == (
            costs[i - 1][j - 1] + (source[i - 1] != target[j - 1])
        ):
            steps.append((i - 1, i, None if source[i - 1] == target[j - 1]
                          else (target[j - 1],)))
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            if not allow_deletions:
                return None
            steps.append((i - 1, i, ()))
            i -= 1
        else:
            steps.append((i, i, (target[j - 1],)))
            j -= 1

    edits: list[tuple[int, int, tuple[str, ...]]] = []
    start: int | None = None
    end = 0
    replacement: list[str] = []

    def finish() -> bool:
        nonlocal start
        if start is None:
            return True
        words = tuple(replacement)
        # An order-only change is not a lexical correction. Do not manufacture
        # a new sequence by lending words from elsewhere in the same span.
        if end - start > 1 and sorted(source[start:end]) == sorted(words):
            return False
        edits.append((start, end, words))
        start = None
        replacement.clear()
        return True

    for before, after, words in reversed(steps):
        if words is None:
            if not finish():
                return None
        else:
            if start is None:
                start = before
            end = after
            replacement.extend(words)
    return tuple(edits) if finish() else None


def corroborated_primary(
    *, streaming: str, primary: str, verifier: str
) -> str | None:
    """Return the existing primary rendering after exact per-edit support.

    This is not whole-transcript consensus or a control-repair mechanism. Each
    primary change must be reproduced by the independent verifier at identical
    streaming-token indices. The verifier may differ only in separate spans
    where the streaming recognizer already supports the primary. Never splice
    renderings, use fuzzy/word-set similarity, accept deletions or resolve an
    ambiguous alignment. Unsupported/bounded/invalid inputs silently abstain.
    Callers additionally restrict this fallback to the CPU Qwen family and to
    the absence of an established exact quorum.
    """
    try:
        renderings = (streaming, primary, verifier)
        if any(not isinstance(text, str) or len(text) > _PRIMARY_MAX_CHARS
               for text in renderings):
            return None
        # All controller-owned meaning stays outside this dialogue fallback.
        # Evaluate original full text, never a token reconstruction.
        if any(exact_control_class(text) is not None for text in renderings):
            return None
        token_sets = tuple(_exact_tokens(text) for text in renderings)
        if any(not tokens or len(tokens) > _PRIMARY_MAX_TOKENS for tokens in token_sets):
            return None
        source, chosen, witness = token_sets
        if len(source) < 3 or len(chosen) < 3 or source == chosen:
            # Short/open-speaker fragments cannot use a partial-phrase vote to
            # grow into a new turn. The original exact consensus stays intact.
            return None
        primary_edits = _unique_edits(source, chosen)
        verifier_edits = _unique_edits(source, witness, allow_deletions=True)
        if not primary_edits or verifier_edits is None:
            return None
        if not all(edit in verifier_edits for edit in primary_edits):
            return None
        for start, end, words in primary_edits:
            # Reject nonadjacent swaps/migration too. Separate substitutions
            # around an unchanged anchor can otherwise form a unique path.
            outside = source[:start] + source[end:]
            if any(word in outside for word in words):
                return None
        return primary
    except Exception:  # noqa: BLE001 - no text/exception detail escapes
        return None


__all__ = [
    "AsrConsensusDecision",
    "AsrConsensusOutcome",
    "AsrConsensusSource",
    "verify_asr_consensus",
    "corroborated_primary",
]
