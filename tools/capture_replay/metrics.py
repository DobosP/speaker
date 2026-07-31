"""Aggregate-only metrics for timestamped production capture replay.

Transcript text exists only in the short-lived in-process observations below.
The reducer deliberately returns counters, rates, and timing distributions only;
neither per-case identifiers nor reference/hypothesis rows leave this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Mapping, Sequence

from core.engine import TranscriptAbortReason
from core.wer import normalize, word_error_rate

from .corpus import LoadedReplayCorpus, ReplayAssertion


_MAX_PARTIALS_PER_RUN = 4_096
_MAX_FINALS_PER_RUN = 128
_MAX_TERMINALS_PER_RUN = 128
_MAX_TEXT_CHARS = 4_096
_MAX_OPAQUE_ID_CHARS = 128
_SAFE_OPAQUE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_SAFE_CONDITION_TAGS = {
    "single-speaker": "single_speaker",
    "overlapping-speakers": "human_overlap",
    "turn-transition": "turn_transition",
}


class ReplayAcousticEvent(str, Enum):
    """Closed, aggregate-safe acoustic events emitted by the replay collector."""

    BARGE_IN = "barge_in"


@dataclass(frozen=True)
class TimedHypothesis:
    """One provisional hypothesis on an engine-owned monotonic timeline."""

    text: str = field(repr=False)
    emitted_at: float
    utterance_id: str
    speech_start_at: float | None = None


@dataclass(frozen=True)
class TimedFinal:
    """One typed final with the endpoint lineage emitted by the live engine."""

    text: str = field(repr=False)
    emitted_at: float
    speech_start_at: float | None
    speech_end_at: float | None
    endpoint_committed_at: float | None
    utterance_id: str


@dataclass(frozen=True)
class ReplayRunRecord:
    """Private observations for one case/repeat execution."""

    case_index: int
    repeat: int
    partials: tuple[TimedHypothesis, ...] = field(default=(), repr=False)
    finals: tuple[TimedFinal, ...] = field(default=(), repr=False)
    commands: tuple[str, ...] = field(default=(), repr=False)
    acoustic_events: tuple[ReplayAcousticEvent, ...] = ()
    abort_reasons: tuple[TranscriptAbortReason, ...] = ()
    wall_seconds: float = 0.0
    peak_rss_mb: float | None = None
    peak_vram_mb: float | None = None


@dataclass
class _AccuracyTotals:
    """Mutable internal counters; no source text survives ``as_report``."""

    runs: int = 0
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    reference_words: int = 0
    hypothesis_words: int = 0
    character_errors: int = 0
    reference_characters: int = 0
    hypothesis_characters: int = 0

    def add(self, expected_text: str, hypothesis: str) -> None:
        measured = word_error_rate(expected_text, hypothesis)
        reference_chars = _normalized_characters(expected_text)
        hypothesis_chars = _normalized_characters(hypothesis)
        self.runs += 1
        self.substitutions += measured.substitutions
        self.insertions += measured.insertions
        self.deletions += measured.deletions
        self.reference_words += measured.ref_words
        self.hypothesis_words += measured.hyp_words
        self.character_errors += _edit_distance(reference_chars, hypothesis_chars)
        self.reference_characters += len(reference_chars)
        self.hypothesis_characters += len(hypothesis_chars)

    def as_report(self) -> dict[str, int | float | None]:
        word_errors = self.substitutions + self.insertions + self.deletions
        return {
            "transcript_runs": self.runs,
            "word_errors": word_errors,
            "reference_words": self.reference_words,
            "hypothesis_words": self.hypothesis_words,
            "wer": (
                round(word_errors / self.reference_words, 4)
                if self.runs and self.reference_words
                else (
                    1.0
                    if self.runs and self.hypothesis_words
                    else None
                )
            ),
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "character_errors": self.character_errors,
            "reference_characters": self.reference_characters,
            "hypothesis_characters": self.hypothesis_characters,
            "cer": (
                round(self.character_errors / self.reference_characters, 4)
                if self.runs and self.reference_characters
                else (
                    1.0
                    if self.runs and self.hypothesis_characters
                    else None
                )
            ),
        }


def _condition_reports(
    totals_by_condition: Mapping[str, _AccuracyTotals],
) -> dict[str, dict[str, int | float | bool | None]]:
    result: dict[str, dict[str, int | float | bool | None]] = {}
    for condition, totals in totals_by_condition.items():
        report: dict[str, int | float | bool | None] = totals.as_report()
        comparable = condition != "human_overlap"
        report["reference_order_comparable"] = comparable
        if not comparable:
            report["linearized_wer"] = report["wer"]
            report["linearized_cer"] = report["cer"]
            report["wer"] = None
            report["cer"] = None
        result[condition] = report
    return result


def _finite_nonnegative(value: object, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError("invalid replay timing")
    return float(value)


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    finite = sorted(
        float(value)
        for value in values
        if not isinstance(value, bool) and math.isfinite(float(value))
    )
    if not finite:
        return None
    index = max(0, math.ceil(quantile * len(finite)) - 1)
    return round(finite[index], 3)


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    """Memory-bounded Levenshtein distance for short normalized sequences."""

    if len(left) > len(right):
        left, right = right, left
    prior = list(range(len(left) + 1))
    for row, right_item in enumerate(right, start=1):
        current = [row]
        for column, left_item in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    prior[column] + 1,
                    prior[column - 1] + (left_item != right_item),
                )
            )
        prior = current
    return prior[-1]


def _normalized_characters(text: str) -> tuple[str, ...]:
    return tuple(" ".join(normalize(text)))


def _contains_tokens(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        tuple(haystack[index : index + len(needle)]) == tuple(needle)
        for index in range(len(haystack) - len(needle) + 1)
    )


def _longest_common_prefix(left: Sequence[str], right: Sequence[str]) -> int:
    length = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        length += 1
    return length


def _assertion_value(value: object) -> str:
    if isinstance(value, ReplayAssertion):
        return value.value
    if isinstance(value, str):
        return value
    raise ValueError("invalid replay assertion")


def _bounded_text(value: object) -> str:
    if not isinstance(value, str) or len(value) > _MAX_TEXT_CHARS:
        raise ValueError("invalid replay text")
    return value


def _opaque_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) > _MAX_OPAQUE_ID_CHARS
        or _SAFE_OPAQUE_ID_RE.fullmatch(value) is None
    ):
        raise ValueError("invalid replay lineage")
    return value


def _validate_record_shape(record: ReplayRunRecord) -> None:
    if (
        len(record.partials) > _MAX_PARTIALS_PER_RUN
        or len(record.finals) > _MAX_FINALS_PER_RUN
        or len(record.commands) > _MAX_TERMINALS_PER_RUN
        or len(record.acoustic_events) > _MAX_TERMINALS_PER_RUN
        or len(record.abort_reasons) > _MAX_TERMINALS_PER_RUN
    ):
        raise ValueError("replay observation exceeds its bound")
    final_ids: set[str] = set()
    for partial in record.partials:
        if not isinstance(partial, TimedHypothesis):
            raise ValueError("invalid replay partial")
        _bounded_text(partial.text)
        _opaque_id(partial.utterance_id)
    for final in record.finals:
        if not isinstance(final, TimedFinal):
            raise ValueError("invalid replay final")
        _bounded_text(final.text)
        utterance_id = _opaque_id(final.utterance_id)
        if utterance_id in final_ids:
            raise ValueError("duplicate replay final lineage")
        final_ids.add(utterance_id)
    for command in record.commands:
        if not isinstance(command, str) or not command or len(command) > 128:
            raise ValueError("invalid replay command")
    if any(
        not isinstance(event, ReplayAcousticEvent)
        for event in record.acoustic_events
    ):
        raise ValueError("invalid acoustic event")
    if any(
        not isinstance(reason, TranscriptAbortReason)
        for reason in record.abort_reasons
    ):
        raise ValueError("invalid abort reason")


def _utterance_stability(
    partials: Sequence[TimedHypothesis],
    finals: Sequence[TimedFinal],
) -> tuple[int, int, list[float], int]:
    """Return churn, retractions, stable-partial latencies, and misses."""

    churn = 0
    retractions = 0
    stable_latency_ms: list[float] = []
    missing_stable = 0
    final_groups = {final.utterance_id: final for final in finals}
    partial_groups: dict[str, list[TimedHypothesis]] = {}
    for partial in partials:
        partial_groups.setdefault(partial.utterance_id, []).append(partial)

    for utterance_id, final in final_groups.items():
        events = sorted(
            partial_groups.get(utterance_id, ()),
            key=lambda event: event.emitted_at,
        )
        normalized = [tuple(normalize(event.text)) for event in events]
        target = tuple(normalize(final.text))
        previous: tuple[str, ...] | None = None
        for current in (*normalized, target):
            if previous is not None:
                churn += _edit_distance(previous, current)
                prefix = _longest_common_prefix(previous, current)
                retractions += max(0, len(previous) - prefix)
            previous = current

        stable_index: int | None = None
        for index, current in enumerate(normalized):
            if current == target and all(tail == target for tail in normalized[index:]):
                stable_index = index
                break
        if stable_index is None:
            missing_stable += 1
            continue
        speech_start = events[stable_index].speech_start_at
        if speech_start is None:
            speech_start = final.speech_start_at
        if speech_start is None:
            missing_stable += 1
            continue
        latency = events[stable_index].emitted_at - speech_start
        if latency < 0.0:
            raise ValueError("invalid replay timing")
        stable_latency_ms.append(latency * 1000.0)
    return churn, retractions, stable_latency_ms, missing_stable


def _require_finite_aggregates(value: object) -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("non-finite replay aggregate")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite_aggregates(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_finite_aggregates(item)
        return
    raise ValueError("unsupported replay aggregate")


def aggregate_metrics(
    corpus: LoadedReplayCorpus,
    records: Sequence[ReplayRunRecord],
    *,
    repeats: int,
) -> dict[str, object]:
    """Reduce timestamped engine observations without exposing transcript rows."""

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("invalid repeats")

    expected_runs = len(corpus.cases) * repeats
    seen: set[tuple[int, int]] = set()
    assertion_counts: dict[str, int] = {
        assertion.value: 0 for assertion in ReplayAssertion
    }
    word_substitutions = 0
    word_insertions = 0
    word_deletions = 0
    reference_words = 0
    hypothesis_words = 0
    character_errors = 0
    reference_characters = 0
    hypothesis_characters = 0
    transcript_runs = 0
    silence_runs = 0
    silence_nonempty_partials = 0
    silence_nonempty_finals = 0
    command_attempts = 0
    command_hits = 0
    lexical_command_hits = 0
    double_talk_runs = 0
    double_talk_with_event = 0
    event_only_runs = 0
    event_only_hits = 0
    partial_events = 0
    final_events = 0
    abort_counts: dict[str, int] = {}
    acoustic_event_counts: dict[str, int] = {}
    first_partial_ms: list[float] = []
    stable_partial_ms: list[float] = []
    speech_end_to_endpoint_ms: list[float] = []
    endpoint_to_final_ms: list[float] = []
    speech_end_to_final_ms: list[float] = []
    churn_edits = 0
    retracted_words = 0
    missing_stable = 0
    wall_seconds = 0.0
    audio_seconds = 0.0
    peak_rss_mb: float | None = None
    peak_vram_mb: float | None = None
    text_runs_without_final = 0
    finals_with_complete_lineage = 0
    finals_missing_speech_start = 0
    finals_missing_speech_end = 0
    finals_missing_endpoint = 0
    partials_with_speech_start = 0
    partials_missing_speech_start = 0
    condition_accuracy = {
        condition: _AccuracyTotals()
        for condition in sorted(_SAFE_CONDITION_TAGS.values())
    }

    for record in records:
        if (
            isinstance(record.case_index, bool)
            or not isinstance(record.case_index, int)
            or record.case_index < 0
            or record.case_index >= len(corpus.cases)
            or isinstance(record.repeat, bool)
            or not isinstance(record.repeat, int)
            or record.repeat < 0
            or record.repeat >= repeats
            or (record.case_index, record.repeat) in seen
        ):
            raise ValueError("invalid replay run record")
        _validate_record_shape(record)
        seen.add((record.case_index, record.repeat))
        case = corpus.cases[record.case_index]
        assertion = _assertion_value(case.assertion)
        if assertion not in assertion_counts:
            raise ValueError("invalid replay assertion")
        assertion_counts[assertion] += 1

        run_wall = _finite_nonnegative(record.wall_seconds)
        assert run_wall is not None
        wall_seconds += run_wall
        audio_seconds += float(case.samples) / float(case.sample_rate_hz)
        for resource_name, resource in (
            ("rss", record.peak_rss_mb),
            ("vram", record.peak_vram_mb),
        ):
            measured = _finite_nonnegative(resource, optional=True)
            if measured is None:
                continue
            if resource_name == "rss":
                peak_rss_mb = (
                    measured if peak_rss_mb is None else max(peak_rss_mb, measured)
                )
            else:
                peak_vram_mb = (
                    measured if peak_vram_mb is None else max(peak_vram_mb, measured)
                )

        ordered_partials = sorted(
            record.partials,
            key=lambda partial: partial.emitted_at,
        )
        first_partial_by_utterance: dict[str, float] = {}
        speech_start_by_utterance: dict[str, float] = {}
        for partial in ordered_partials:
            emitted_at = _finite_nonnegative(partial.emitted_at)
            speech_start = _finite_nonnegative(
                partial.speech_start_at,
                optional=True,
            )
            if speech_start is None:
                partials_missing_speech_start += 1
            else:
                partials_with_speech_start += 1
            if speech_start is not None:
                assert emitted_at is not None
                latency = emitted_at - speech_start
                if latency < 0.0:
                    raise ValueError("invalid replay timing")
                prior_start = speech_start_by_utterance.setdefault(
                    partial.utterance_id,
                    speech_start,
                )
                if abs(prior_start - speech_start) > 1e-6:
                    raise ValueError("inconsistent replay speech start")
                first_partial_by_utterance.setdefault(
                    partial.utterance_id,
                    latency * 1000.0,
                )
        first_partial_ms.extend(first_partial_by_utterance.values())
        for final in record.finals:
            emitted_at = _finite_nonnegative(final.emitted_at)
            speech_start = _finite_nonnegative(final.speech_start_at, optional=True)
            speech_end = _finite_nonnegative(final.speech_end_at, optional=True)
            endpoint = _finite_nonnegative(
                final.endpoint_committed_at,
                optional=True,
            )
            if (
                speech_start is not None
                and speech_end is not None
                and speech_end < speech_start
            ):
                raise ValueError("invalid replay timing")
            finals_missing_speech_start += speech_start is None
            finals_missing_speech_end += speech_end is None
            finals_missing_endpoint += endpoint is None
            finals_with_complete_lineage += (
                speech_start is not None
                and speech_end is not None
                and endpoint is not None
            )
            if endpoint is not None and speech_end is not None:
                if endpoint < speech_end:
                    raise ValueError("invalid replay timing")
                speech_end_to_endpoint_ms.append((endpoint - speech_end) * 1000.0)
            if endpoint is not None:
                assert emitted_at is not None
                if emitted_at < endpoint:
                    raise ValueError("invalid replay timing")
                endpoint_to_final_ms.append((emitted_at - endpoint) * 1000.0)
            if speech_end is not None:
                assert emitted_at is not None
                if emitted_at < speech_end:
                    raise ValueError("invalid replay timing")
                speech_end_to_final_ms.append((emitted_at - speech_end) * 1000.0)

        partial_events += len(record.partials)
        final_events += len(record.finals)
        churn, retractions, stable, missing = _utterance_stability(
            record.partials,
            record.finals,
        )
        churn_edits += churn
        retracted_words += retractions
        stable_partial_ms.extend(stable)
        missing_stable += missing

        for reason in record.abort_reasons:
            key = reason.value
            abort_counts[key] = abort_counts.get(key, 0) + 1
        for event in record.acoustic_events:
            key = event.value
            acoustic_event_counts[key] = acoustic_event_counts.get(key, 0) + 1

        hypothesis = " ".join(
            final.text.strip() for final in record.finals if final.text.strip()
        )
        hypothesis_tokens = tuple(normalize(hypothesis))
        expected_text = case.expected_text
        text_assertion = assertion in {
            ReplayAssertion.TRANSCRIPT.value,
            ReplayAssertion.NEAR_END_ONLY.value,
            ReplayAssertion.DOUBLE_TALK.value,
        }
        silence_assertion = assertion in {
            ReplayAssertion.SILENCE.value,
            ReplayAssertion.FAR_END_ONLY.value,
            ReplayAssertion.EVENT_ONLY.value,
        }
        if text_assertion:
            transcript_runs += 1
            if not record.finals:
                text_runs_without_final += 1
                missing_stable += 1
            measured = word_error_rate(expected_text, hypothesis)
            word_substitutions += measured.substitutions
            word_insertions += measured.insertions
            word_deletions += measured.deletions
            reference_words += measured.ref_words
            hypothesis_words += measured.hyp_words
            reference_chars = _normalized_characters(expected_text)
            hypothesis_chars = _normalized_characters(hypothesis)
            character_errors += _edit_distance(reference_chars, hypothesis_chars)
            reference_characters += len(reference_chars)
            hypothesis_characters += len(hypothesis_chars)
            matched_conditions = {
                _SAFE_CONDITION_TAGS[tag]
                for tag in case.tags
                if tag in _SAFE_CONDITION_TAGS
            }
            if len(matched_conditions) > 1:
                raise ValueError("ambiguous replay condition")
            for condition in matched_conditions:
                condition_accuracy[condition].add(expected_text, hypothesis)
        elif silence_assertion:
            silence_runs += 1
            silence_nonempty_partials += sum(
                bool(partial.text.strip()) for partial in record.partials
            )
            silence_nonempty_finals += sum(
                bool(final.text.strip()) for final in record.finals
            )

        routed_commands = {
            tuple(normalize(command))
            for command in record.commands
            if normalize(command)
        }
        for command in case.commands:
            command_attempts += 1
            command_tokens = tuple(normalize(command))
            if command_tokens in routed_commands:
                command_hits += 1
            if _contains_tokens(hypothesis_tokens, command_tokens):
                lexical_command_hits += 1

        if assertion == ReplayAssertion.DOUBLE_TALK.value:
            double_talk_runs += 1
            double_talk_with_event += (
                ReplayAcousticEvent.BARGE_IN in record.acoustic_events
            )
        elif assertion == ReplayAssertion.EVENT_ONLY.value:
            event_only_runs += 1
            event_only_hits += (
                ReplayAcousticEvent.BARGE_IN in record.acoustic_events
            )

    word_errors = word_substitutions + word_insertions + word_deletions
    result: dict[str, object] = {
        "schema_version": 1,
        "evaluations": len(records),
        "coverage": {
            "expected": expected_runs,
            "complete": len(records) == expected_runs and len(seen) == expected_runs,
            "by_assertion": assertion_counts,
        },
        "accuracy": {
            "transcript_runs": transcript_runs,
            "word_errors": word_errors,
            "reference_words": reference_words,
            "hypothesis_words": hypothesis_words,
            "wer": (
                round(word_errors / reference_words, 4)
                if transcript_runs and reference_words
                else (
                    1.0
                    if transcript_runs and hypothesis_words
                    else None
                )
            ),
            "substitutions": word_substitutions,
            "insertions": word_insertions,
            "deletions": word_deletions,
            "character_errors": character_errors,
            "reference_characters": reference_characters,
            "hypothesis_characters": hypothesis_characters,
            "cer": (
                round(character_errors / reference_characters, 4)
                if transcript_runs and reference_characters
                else (
                    1.0
                    if transcript_runs and hypothesis_characters
                    else None
                )
            ),
            "command_attempts": command_attempts,
            "command_hits": command_hits,
            "lexical_command_hits": lexical_command_hits,
            "command_recall": (
                round(command_hits / command_attempts, 4)
                if command_attempts
                else None
            ),
            "contains_order_ambiguous_reference": (
                condition_accuracy["human_overlap"].runs > 0
            ),
            "by_condition": _condition_reports(condition_accuracy),
            "silence_runs": silence_runs,
            "silence_nonempty_partials": silence_nonempty_partials,
            "silence_nonempty_finals": silence_nonempty_finals,
        },
        "latency": {
            "first_partial_observations": len(first_partial_ms),
            "first_partial_from_speech_start_p50_ms": _percentile(
                first_partial_ms,
                0.50,
            ),
            "first_partial_from_speech_start_p95_ms": _percentile(
                first_partial_ms,
                0.95,
            ),
            "stable_partial_from_speech_start_p50_ms": _percentile(
                stable_partial_ms,
                0.50,
            ),
            "stable_partial_from_speech_start_p95_ms": _percentile(
                stable_partial_ms,
                0.95,
            ),
            "stable_partial_observations": len(stable_partial_ms),
            "speech_end_to_endpoint_observations": len(
                speech_end_to_endpoint_ms
            ),
            "speech_end_to_endpoint_p50_ms": _percentile(
                speech_end_to_endpoint_ms,
                0.50,
            ),
            "speech_end_to_endpoint_p95_ms": _percentile(
                speech_end_to_endpoint_ms,
                0.95,
            ),
            "endpoint_to_final_p50_ms": _percentile(
                endpoint_to_final_ms,
                0.50,
            ),
            "endpoint_to_final_p95_ms": _percentile(
                endpoint_to_final_ms,
                0.95,
            ),
            "endpoint_to_final_observations": len(endpoint_to_final_ms),
            "speech_end_to_final_p50_ms": _percentile(
                speech_end_to_final_ms,
                0.50,
            ),
            "speech_end_to_final_p95_ms": _percentile(
                speech_end_to_final_ms,
                0.95,
            ),
            "speech_end_to_final_observations": len(speech_end_to_final_ms),
        },
        "streaming": {
            "partial_events": partial_events,
            "final_events": final_events,
            "churn_token_edits": churn_edits,
            "retracted_words": retracted_words,
            "stable_partial_missing": missing_stable,
            "text_runs_without_final": text_runs_without_final,
            "abort_reasons": dict(sorted(abort_counts.items())),
        },
        "acoustic": {
            "events": dict(sorted(acoustic_event_counts.items())),
            "double_talk_runs": double_talk_runs,
            "double_talk_with_event": double_talk_with_event,
            "event_only_runs": event_only_runs,
            "event_only_hits": event_only_hits,
            "lineage_coverage": {
                "finals": final_events,
                "finals_complete": finals_with_complete_lineage,
                "finals_missing_speech_start": finals_missing_speech_start,
                "finals_missing_speech_end": finals_missing_speech_end,
                "finals_missing_endpoint": finals_missing_endpoint,
                "partials_with_speech_start": partials_with_speech_start,
                "partials_missing_speech_start": partials_missing_speech_start,
                "speech_end_to_endpoint_observations": len(
                    speech_end_to_endpoint_ms
                ),
                "endpoint_to_final_observations": len(endpoint_to_final_ms),
                "speech_end_to_final_observations": len(speech_end_to_final_ms),
            },
        },
        "throughput": {
            "audio_total_sec": round(audio_seconds, 3),
            "wall_total_sec": round(wall_seconds, 3),
            "wall_to_audio_ratio": (
                round(wall_seconds / audio_seconds, 4) if audio_seconds else 0.0
            ),
        },
        "resources": {
            "peak_rss_mb": (
                None if peak_rss_mb is None else round(peak_rss_mb, 3)
            ),
            "peak_vram_mb": (
                None if peak_vram_mb is None else round(peak_vram_mb, 3)
            ),
        },
    }
    _require_finite_aggregates(result)
    return result
