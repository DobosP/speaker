"""Aggregate-only lexical metrics for production final-selection replay.

Unlike the isolated streaming evaluator, production final selection may emit
zero or multiple terminal decisions for one corpus case. Word-error metrics
grade the normalized decisions in order as one case hypothesis, while command
targets are matched within each decision boundary. A phrase split across two
terminal decisions therefore cannot become a command hit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Sequence

from core.wer import normalize
from tools.recorded_stt_eval import AccuracyTotals, _measure, _sum_accuracy

from .corpus import LoadedCorpus
from .metrics import (
    _contains_tokens,
    _require_finite_aggregates,
    _silence_hallucination,
    validate_stratum_tags,
)


_SAMPLE_RATE_HZ = 16_000
_MAX_REPEATS = 8
_MAX_DECISIONS_PER_EVALUATION = 256
_MAX_DECISION_CHARS = 4096


@dataclass(frozen=True)
class FinalDecisionRecord:
    """One case/repeat's private, ordered production final decisions."""

    case_index: int
    repeat: int
    decisions: tuple[str, ...] = field(repr=False)


@dataclass
class _AggregateTotals:
    """Mergeable raw counters; never stores a reference or hypothesis."""

    accuracy: AccuracyTotals = field(default_factory=AccuracyTotals)
    evaluations: int = 0
    added_cases: int = 0
    command_attempts: int = 0
    command_hits: int = 0
    positive_command_true_positives: int = 0
    positive_command_false_negatives: int = 0
    positive_command_confusion_hits: int = 0
    forbidden_final_false_positives: int = 0
    forbidden_final_true_negatives: int = 0
    negative_pair_false_positives: int = 0
    negative_pair_true_negatives: int = 0
    negative_final_case_false_positives: int = 0
    negative_final_case_true_negatives: int = 0
    monitored_negative_evaluations_including_silence: int = 0
    monitored_negative_audio_seconds_including_silence: float = 0.0
    silence_nonempty_finals: int = 0
    speech_negative_nonempty_finals: int = 0
    decision_count: int = 0
    zero_decision_evaluations: int = 0
    multi_decision_evaluations: int = 0
    cases_with_final_disagreement: int = 0

    def merge(self, other: _AggregateTotals) -> None:
        """Merge only raw additive values; derived rates are rendered later."""

        if type(other) is not _AggregateTotals:
            raise ValueError("invalid final metric totals")
        self.accuracy = _sum_accuracy(self.accuracy, other.accuracy)
        for descriptor in dataclass_fields(self):
            name = descriptor.name
            if name == "accuracy":
                continue
            setattr(self, name, getattr(self, name) + getattr(other, name))


def _validate_record_shape(record: FinalDecisionRecord) -> None:
    if (
        type(record) is not FinalDecisionRecord
        or type(record.case_index) is not int
        or type(record.repeat) is not int
        or type(record.decisions) is not tuple
        or len(record.decisions) > _MAX_DECISIONS_PER_EVALUATION
        or any(
            type(decision) is not str or len(decision) > _MAX_DECISION_CHARS
            for decision in record.decisions
        )
    ):
        raise ValueError("invalid final decision record")


def _normalized_decisions(record: FinalDecisionRecord) -> tuple[tuple[str, ...], ...]:
    _validate_record_shape(record)
    return tuple(tuple(normalize(decision)) for decision in record.decisions)


def _decision_set_digest(decisions: tuple[tuple[str, ...], ...]) -> bytes:
    """Bind normalized text and decision boundaries without retaining either."""

    encoded = json.dumps(
        decisions,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def _target_hit(
    decisions: tuple[tuple[str, ...], ...],
    target: tuple[str, ...],
) -> bool:
    return any(_contains_tokens(decision, target) for decision in decisions)


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _reduce_case(
    corpus: LoadedCorpus,
    case_index: int,
    records: Sequence[FinalDecisionRecord],
    *,
    repeats: int,
) -> _AggregateTotals:
    """Reduce one complete repeat set and return transcript-free raw totals."""

    if (
        type(case_index) is not int
        or not 0 <= case_index < len(corpus.cases)
    ):
        raise ValueError("invalid metric case selection")
    try:
        selected_records = tuple(records)
    except (TypeError, ValueError):
        raise ValueError("invalid final decision records") from None
    if len(selected_records) != repeats:
        raise ValueError("incomplete final decision repeats")

    case = corpus.cases[case_index]
    command_targets = tuple(tuple(normalize(value)) for value in case.commands)
    forbidden_targets = tuple(
        tuple(normalize(value)) for value in case.forbidden_commands
    )
    totals = _AggregateTotals(added_cases=1)
    seen_repeats: set[int] = set()
    final_digests: set[bytes] = set()

    for record in selected_records:
        decisions = _normalized_decisions(record)
        if (
            record.case_index != case_index
            or record.repeat < 0
            or record.repeat >= repeats
            or record.repeat in seen_repeats
        ):
            raise ValueError("invalid final decision record")
        seen_repeats.add(record.repeat)
        totals.evaluations += 1

        # Only WER/CER joins terminal decisions. Command matching below always
        # retains the terminal boundary and cannot synthesize a cross-turn hit.
        joined_for_wer = " ".join(
            token for decision in decisions for token in decision
        )
        if case.assertion == "transcript":
            totals.accuracy = _sum_accuracy(
                totals.accuracy,
                _measure(((case.expected_text, joined_for_wer),)),
            )

        totals.decision_count += len(decisions)
        totals.zero_decision_evaluations += not decisions
        totals.multi_decision_evaluations += len(decisions) > 1
        nonempty_finals = sum(
            _silence_hallucination(raw) for raw in record.decisions
        )
        if case.assertion == "silence":
            totals.silence_nonempty_finals += nonempty_finals
        elif case.assertion == "speech_negative":
            totals.speech_negative_nonempty_finals += nonempty_finals

        for target in command_targets:
            totals.command_attempts += 1
            hit = _target_hit(decisions, target)
            totals.command_hits += hit
            if hit:
                totals.positive_command_true_positives += 1
            else:
                totals.positive_command_false_negatives += 1

        monitored_negative = not command_targets and bool(forbidden_targets)
        negative_case_hit = False
        for target in forbidden_targets:
            hit = _target_hit(decisions, target)
            if hit:
                totals.forbidden_final_false_positives += 1
                if command_targets:
                    totals.positive_command_confusion_hits += 1
            else:
                totals.forbidden_final_true_negatives += 1
            if monitored_negative:
                negative_case_hit |= hit
                if hit:
                    totals.negative_pair_false_positives += 1
                else:
                    totals.negative_pair_true_negatives += 1

        if monitored_negative:
            totals.monitored_negative_evaluations_including_silence += 1
            totals.monitored_negative_audio_seconds_including_silence += (
                case.samples / _SAMPLE_RATE_HZ
            )
            if negative_case_hit:
                totals.negative_final_case_false_positives += 1
            else:
                totals.negative_final_case_true_negatives += 1

        final_digests.add(_decision_set_digest(decisions))

    if seen_repeats != set(range(repeats)):
        raise ValueError("incomplete final decision repeats")
    totals.cases_with_final_disagreement = int(len(final_digests) > 1)
    return totals


def _render_metrics(
    totals: _AggregateTotals,
    *,
    repeats: int,
    expected_cases: int,
) -> dict[str, object]:
    positive_total = (
        totals.positive_command_true_positives
        + totals.positive_command_false_negatives
    )
    final_positive_predictions = (
        totals.positive_command_true_positives
        + totals.forbidden_final_false_positives
    )
    negative_pair_total = (
        totals.negative_pair_false_positives
        + totals.negative_pair_true_negatives
    )
    negative_case_total = (
        totals.negative_final_case_false_positives
        + totals.negative_final_case_true_negatives
    )
    result: dict[str, object] = {
        "evaluations": totals.evaluations,
        "coverage_complete": (
            totals.added_cases == expected_cases
            and totals.evaluations == expected_cases * repeats
        ),
        "accuracy": {
            "transcript": totals.accuracy.as_dict(),
            "command_attempts": totals.command_attempts,
            "command_hits": totals.command_hits,
            "command_recall": _ratio(
                totals.command_hits,
                totals.command_attempts,
            ),
            "positive_command_true_positives": (
                totals.positive_command_true_positives
            ),
            "positive_command_false_negatives": (
                totals.positive_command_false_negatives
            ),
            "positive_command_recall": _ratio(
                totals.positive_command_true_positives,
                positive_total,
            ),
            "positive_command_confusion_hits": (
                totals.positive_command_confusion_hits
            ),
            "forbidden_final_false_positives": (
                totals.forbidden_final_false_positives
            ),
            "forbidden_final_true_negatives": (
                totals.forbidden_final_true_negatives
            ),
            "final_command_target_precision": _ratio(
                totals.positive_command_true_positives,
                final_positive_predictions,
            ),
            "negative_pair_false_positives": (
                totals.negative_pair_false_positives
            ),
            "negative_pair_true_negatives": totals.negative_pair_true_negatives,
            "negative_pair_false_positive_rate": _ratio(
                totals.negative_pair_false_positives,
                negative_pair_total,
            ),
            "negative_final_case_false_positives": (
                totals.negative_final_case_false_positives
            ),
            "negative_final_case_true_negatives": (
                totals.negative_final_case_true_negatives
            ),
            "negative_final_case_false_positive_rate": _ratio(
                totals.negative_final_case_false_positives,
                negative_case_total,
            ),
            "monitored_negative_evaluations_including_silence": (
                totals.monitored_negative_evaluations_including_silence
            ),
            "monitored_negative_audio_seconds_including_silence": round(
                totals.monitored_negative_audio_seconds_including_silence,
                6,
            ),
            "isolated_clip_lexical_false_positives_per_negative_audio_hour": (
                round(
                    totals.negative_final_case_false_positives
                    / (
                        totals.monitored_negative_audio_seconds_including_silence
                        / 3600.0
                    ),
                    4,
                )
                if totals.monitored_negative_audio_seconds_including_silence
                else None
            ),
            "silence_nonempty_finals": totals.silence_nonempty_finals,
            "speech_negative_nonempty_finals": (
                totals.speech_negative_nonempty_finals
            ),
        },
        "decisions": {
            "total": totals.decision_count,
            "zero_decision_evaluations": totals.zero_decision_evaluations,
            "multi_decision_evaluations": totals.multi_decision_evaluations,
        },
        "determinism": {
            "repeats": repeats,
            "cases_with_final_disagreement": (
                totals.cases_with_final_disagreement
            ),
        },
    }
    _require_finite_aggregates(result)
    return result


class FinalMetricsAccumulator:
    """Incrementally reduce complete cases without retaining decision text."""

    __slots__ = (
        "_added_case_indices",
        "_corpus",
        "_repeats",
        "_selected_tags",
        "_stratum_case_indices",
        "_stratum_totals",
        "_totals",
    )

    def __init__(
        self,
        corpus: LoadedCorpus,
        repeats: int,
        stratum_tags: Sequence[str] = (),
    ) -> None:
        if corpus.schema_version != 4:
            raise ValueError("schema-v4 corpus required")
        if not corpus.cases:
            raise ValueError("empty corpus")
        if (
            isinstance(repeats, bool)
            or not isinstance(repeats, int)
            or not 1 <= repeats <= _MAX_REPEATS
        ):
            raise ValueError("invalid repeats")
        selected_tags = validate_stratum_tags(corpus, stratum_tags)
        self._corpus = corpus
        self._repeats = repeats
        self._selected_tags = selected_tags
        self._stratum_case_indices = {
            tag: frozenset(
                index
                for index, case in enumerate(corpus.cases)
                if tag in case.tags
            )
            for tag in selected_tags
        }
        self._totals = _AggregateTotals()
        self._stratum_totals = {
            tag: _AggregateTotals() for tag in selected_tags
        }
        self._added_case_indices: set[int] = set()

    def __repr__(self) -> str:
        return (
            "FinalMetricsAccumulator("
            f"cases={len(self._corpus.cases)},"
            f"added_cases={len(self._added_case_indices)},"
            f"repeats={self._repeats},"
            f"strata={len(self._selected_tags)})"
        )

    def add_case(
        self,
        case_index: int,
        records: Sequence[FinalDecisionRecord],
    ) -> None:
        """Atomically reduce one case containing exactly one row per repeat."""

        if (
            type(case_index) is not int
            or not 0 <= case_index < len(self._corpus.cases)
        ):
            raise ValueError("invalid metric case selection")
        if case_index in self._added_case_indices:
            raise ValueError("duplicate metric case")
        reduced = _reduce_case(
            self._corpus,
            case_index,
            records,
            repeats=self._repeats,
        )
        self._totals.merge(reduced)
        for tag in self._selected_tags:
            if case_index in self._stratum_case_indices[tag]:
                self._stratum_totals[tag].merge(reduced)
        self._added_case_indices.add(case_index)

    def result(self) -> dict[str, object]:
        """Return a fresh aggregate snapshot; incomplete cases stay coverage-red."""

        result = _render_metrics(
            self._totals,
            repeats=self._repeats,
            expected_cases=len(self._corpus.cases),
        )
        if self._selected_tags:
            result["strata"] = [
                {
                    "tag": tag,
                    "cases": len(self._stratum_case_indices[tag]),
                    "metrics": _render_metrics(
                        self._stratum_totals[tag],
                        repeats=self._repeats,
                        expected_cases=len(self._stratum_case_indices[tag]),
                    ),
                }
                for tag in self._selected_tags
            ]
        _require_finite_aggregates(result)
        return result


def aggregate_final_metrics(
    corpus: LoadedCorpus,
    records: Sequence[FinalDecisionRecord],
    *,
    repeats: int,
    stratum_tags: Sequence[str] = (),
) -> dict[str, object]:
    """Batch compatibility wrapper over :class:`FinalMetricsAccumulator`."""

    accumulator = FinalMetricsAccumulator(
        corpus,
        repeats,
        stratum_tags=stratum_tags,
    )
    grouped: dict[int, list[FinalDecisionRecord]] = {}
    try:
        iterator = iter(records)
    except TypeError:
        raise ValueError("invalid final decision records") from None
    for record in iterator:
        _validate_record_shape(record)
        if not 0 <= record.case_index < len(corpus.cases):
            raise ValueError("invalid final decision record")
        grouped.setdefault(record.case_index, []).append(record)
    for case_index, selected_records in grouped.items():
        accumulator.add_case(case_index, selected_records)
    return accumulator.result()
