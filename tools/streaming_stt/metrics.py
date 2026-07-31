"""Aggregate-only streaming accuracy, stability, latency, and resource metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from core.wer import normalize
from tools.recorded_stt_eval import (
    AccuracyTotals,
    _edit_distance,
    _measure,
    _sum_accuracy,
)

from .corpus import LoadedCorpus
from .protocol import CaseTrace, ReadyEvent


@dataclass(frozen=True)
class RunRecord:
    case_index: int
    repeat: int
    trace: CaseTrace = field(repr=False)


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    index = max(0, math.ceil(quantile * len(finite)) - 1)
    return round(finite[index], 3)


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


def _silence_hallucination(text: str) -> bool:
    """Treat any visible hypothesis, including punctuation, as non-silence."""

    return bool(text.strip())


def _require_finite_aggregates(value: object) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        try:
            if not math.isfinite(value):
                raise ValueError("non-finite aggregate")
        except OverflowError:
            raise ValueError("non-finite aggregate") from None
        return
    if isinstance(value, dict):
        for item in value.values():
            _require_finite_aggregates(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_finite_aggregates(item)
        return
    raise ValueError("unsupported aggregate")


def _stability(trace: CaseTrace) -> tuple[float | None, float | None, int, int]:
    normalized = [tuple(normalize(event.text)) for event in trace.partials]
    final = tuple(normalize(trace.final.text))
    first_nonempty: float | None = None
    for event, tokens in zip(trace.partials, normalized):
        if tokens:
            first_nonempty = event.elapsed_ms
            break
    stable: float | None = None
    for index, (event, tokens) in enumerate(zip(trace.partials, normalized)):
        if tokens == final and all(tail == final for tail in normalized[index:]):
            stable = event.elapsed_ms
            break
    churn = 0
    retractions = 0
    previous: tuple[str, ...] | None = None
    # The committed final is also a hypothesis transition. Omitting it makes a
    # last-second rewrite look perfectly stable even when every partial was
    # wrong.
    for current in (*normalized, final):
        if previous is None:
            previous = current
            continue
        churn += _edit_distance(previous, current)
        prefix = _longest_common_prefix(previous, current)
        retractions += max(0, len(previous) - prefix)
        previous = current
    return first_nonempty, stable, churn, retractions


def aggregate_metrics(
    corpus: LoadedCorpus,
    records: Sequence[RunRecord],
    ready: ReadyEvent,
    *,
    repeats: int,
) -> dict[str, object]:
    """Reduce all private text to aggregate counters and timing summaries."""

    accuracy = AccuracyTotals()
    first_partial_ms: list[float] = []
    stable_partial_ms: list[float] = []
    final_lag_ms: list[float] = []
    compute_ms: list[float] = []
    max_backlog_ms: list[float] = []
    audio_seconds = 0.0
    deadline_misses = 0
    model_padding_samples: list[int] = []
    churn_edits = 0
    retracted_words = 0
    partial_events = 0
    missing_stable = 0
    silence_nonempty_partials = 0
    silence_nonempty_finals = 0
    command_attempts = 0
    command_hits = 0
    final_by_case: dict[int, set[tuple[str, ...]]] = {}
    peak_rss = ready.resources.rss_mb
    peak_threads = ready.resources.threads
    peak_vram = ready.resources.vram_mb
    seen_runs: set[tuple[int, int]] = set()

    for record in records:
        if (
            record.case_index < 0
            or record.case_index >= len(corpus.cases)
            or record.repeat < 0
            or record.repeat >= repeats
            or (record.case_index, record.repeat) in seen_runs
        ):
            raise ValueError("invalid run record")
        seen_runs.add((record.case_index, record.repeat))
        case = corpus.cases[record.case_index]
        trace = record.trace
        hypothesis = trace.final.text
        if case.assertion == "transcript":
            accuracy = _sum_accuracy(
                accuracy,
                _measure(((case.expected_text, hypothesis),)),
            )
            hyp_tokens = tuple(normalize(hypothesis))
            for command in case.commands:
                command_attempts += 1
                if _contains_tokens(hyp_tokens, tuple(normalize(command))):
                    command_hits += 1
        else:
            silence_nonempty_partials += sum(
                _silence_hallucination(event.text) for event in trace.partials
            )
            silence_nonempty_finals += _silence_hallucination(hypothesis)

        first, stable, churn, retractions = _stability(trace)
        if case.assertion == "transcript":
            if first is not None:
                first_partial_ms.append(first)
            if stable is None:
                missing_stable += 1
            else:
                stable_partial_ms.append(stable)
        churn_edits += churn
        retracted_words += retractions
        partial_events += len(trace.partials)
        final_lag_ms.append(trace.final.finalization_ms)
        compute_ms.append(trace.final.compute_ms)
        max_backlog_ms.append(trace.final.max_backlog_ms)
        audio_seconds += trace.final.audio_seconds
        deadline_misses += trace.final.deadline_misses
        model_padding_samples.append(trace.final.model_padding_samples)
        final_by_case.setdefault(record.case_index, set()).add(
            tuple(normalize(hypothesis))
        )
        peak_rss = max(peak_rss, trace.final.resources.rss_mb)
        peak_threads = max(peak_threads, trace.final.resources.threads)
        if trace.final.resources.vram_mb is not None:
            peak_vram = (
                trace.final.resources.vram_mb
                if peak_vram is None
                else max(peak_vram, trace.final.resources.vram_mb)
            )

    expected_records = len(corpus.cases) * repeats
    repeat_disagreements = sum(len(outputs) > 1 for outputs in final_by_case.values())
    total_compute_ms = sum(compute_ms)
    result = {
        "evaluations": len(records),
        "coverage_complete": (
            len(records) == expected_records
            and len(final_by_case) == len(corpus.cases)
            and len(seen_runs) == expected_records
        ),
        "accuracy": {
            "transcript": accuracy.as_dict(),
            "command_attempts": command_attempts,
            "command_hits": command_hits,
            "command_recall": (
                round(command_hits / command_attempts, 4) if command_attempts else 1.0
            ),
            "silence_nonempty_partials": silence_nonempty_partials,
            "silence_nonempty_finals": silence_nonempty_finals,
        },
        "latency": {
            "model_load_ms": round(ready.model_load_ms, 3),
            "first_nonempty_partial_p50_ms": _percentile(first_partial_ms, 0.50),
            "first_nonempty_partial_p95_ms": _percentile(first_partial_ms, 0.95),
            "stable_partial_p50_ms": _percentile(stable_partial_ms, 0.50),
            "stable_partial_p95_ms": _percentile(stable_partial_ms, 0.95),
            "stable_partial_missing": missing_stable,
            "finalization_p50_ms": _percentile(final_lag_ms, 0.50),
            "finalization_p95_ms": _percentile(final_lag_ms, 0.95),
            "finalization_max_ms": _percentile(final_lag_ms, 1.0),
        },
        "streaming": {
            "partial_events": partial_events,
            "churn_token_edits": churn_edits,
            "retracted_words": retracted_words,
            "deadline_misses": deadline_misses,
            "max_backlog_p95_ms": _percentile(max_backlog_ms, 0.95),
            "max_backlog_ms": _percentile(max_backlog_ms, 1.0),
            "model_padding_total_samples": sum(model_padding_samples),
            "model_padding_max_samples": max(model_padding_samples, default=0),
        },
        "throughput": {
            "audio_total_sec": round(audio_seconds, 3),
            "compute_total_ms": round(total_compute_ms, 3),
            "aggregate_rtf": (
                round((total_compute_ms / 1000.0) / audio_seconds, 4)
                if audio_seconds
                else 0.0
            ),
        },
        "determinism": {
            "repeats": repeats,
            "cases_with_final_disagreement": repeat_disagreements,
        },
        "resources": {
            "peak_rss_mb": round(peak_rss, 3),
            "peak_threads": peak_threads,
            "peak_vram_mb": None if peak_vram is None else round(peak_vram, 3),
            "source": "worker_reported",
        },
    }
    _require_finite_aggregates(result)
    return result
