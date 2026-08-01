"""Aggregate-only streaming accuracy, stability, latency, and resource metrics."""

from __future__ import annotations

import math
import re
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
from .manifest import PARAKEET_REALTIME_EOU_ADAPTER
from .protocol import (
    NATIVE_ENDPOINT_PROTOCOL_VERSION,
    CaseTrace,
    NativeFinalEvent,
    ReadyEvent,
)


_SAMPLE_RATE_HZ = 16_000
_MAX_STRATUM_TAGS = 32
_SAFE_STRATUM_TAG_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")


@dataclass(frozen=True)
class RunRecord:
    case_index: int
    repeat: int
    trace: CaseTrace = field(repr=False)


def normalize_stratum_tags(value: Sequence[str]) -> tuple[str, ...]:
    """Return one bounded canonical set of caller-selected corpus tags."""

    if isinstance(value, (str, bytes)):
        raise ValueError("invalid stratum tags")
    selected: list[str] = []
    try:
        iterator = iter(value)
        for index in range(_MAX_STRATUM_TAGS + 1):
            try:
                tag = next(iterator)
            except StopIteration:
                break
            if index >= _MAX_STRATUM_TAGS:
                raise ValueError("invalid stratum tags")
            if (
                type(tag) is not str
                or _SAFE_STRATUM_TAG_RE.fullmatch(tag) is None
            ):
                raise ValueError("invalid stratum tags")
            selected.append(tag)
    except Exception:
        raise ValueError("invalid stratum tags") from None
    return tuple(sorted(set(selected)))


def validate_stratum_tags(
    corpus: LoadedCorpus,
    value: Sequence[str],
) -> tuple[str, ...]:
    """Bind a canonical explicit selection to tags present in the corpus."""

    selected = normalize_stratum_tags(value)
    if selected:
        available = {tag for case in corpus.cases for tag in case.tags}
        if any(tag not in available for tag in selected):
            raise ValueError("invalid stratum tags")
    return selected


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


def _is_finite_number(value: object, *, minimum: float = 0.0) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except OverflowError:
        return False
    return math.isfinite(number) and number >= minimum


def _is_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


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


def _aggregate_metrics_for_cases(
    corpus: LoadedCorpus,
    records: Sequence[RunRecord],
    ready: ReadyEvent,
    *,
    repeats: int,
    expected_case_indices: frozenset[int] | None = None,
) -> dict[str, object]:
    """Reduce selected cases without exposing their text or identifiers."""

    selected_case_indices = (
        frozenset(range(len(corpus.cases)))
        if expected_case_indices is None
        else expected_case_indices
    )
    if (
        not selected_case_indices
        or any(
            type(index) is not int or not 0 <= index < len(corpus.cases)
            for index in selected_case_indices
        )
    ):
        raise ValueError("invalid metric case selection")

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
    endpoint_records: list[NativeFinalEvent] = []
    endpoint_source_offered = 0
    endpoint_source_consumed = 0
    endpoint_tail_declared = 0
    endpoint_tail_consumed = 0
    endpoint_model_padding = 0
    endpoint_reasons = {"eou": 0, "eob": 0, "tail_exhausted": 0}
    endpoint_samples: list[float] = []
    endpoint_latencies: list[float] = []
    endpoint_probabilities: list[float] = []
    authoritative_endpoints = 0
    source_early_native_events = 0
    source_early_eou_events = 0
    source_early_eob_backchannels = 0

    if (
        ready.adapter == PARAKEET_REALTIME_EOU_ADAPTER
        and ready.protocol_version != NATIVE_ENDPOINT_PROTOCOL_VERSION
    ):
        raise ValueError("invalid native endpoint ready event")

    for record in records:
        if (
            record.case_index < 0
            or record.case_index >= len(corpus.cases)
            or record.case_index not in selected_case_indices
            or record.repeat < 0
            or record.repeat >= repeats
            or (record.case_index, record.repeat) in seen_runs
        ):
            raise ValueError("invalid run record")
        seen_runs.add((record.case_index, record.repeat))
        case = corpus.cases[record.case_index]
        trace = record.trace
        if type(trace.final) is NativeFinalEvent:
            final = trace.final
            source_complete = final.source_samples_consumed == final.source_samples
            expected_authoritative = (
                final.native_endpoint
                and final.endpoint_reason == "eou"
                and source_complete
            )
            expected_audio_seconds = case.samples / _SAMPLE_RATE_HZ
            if (
                ready.adapter != PARAKEET_REALTIME_EOU_ADAPTER
                or final.protocol_version != NATIVE_ENDPOINT_PROTOCOL_VERSION
                or type(final.native_endpoint) is not bool
                or type(final.authoritative) is not bool
                or any(
                    event.protocol_version != NATIVE_ENDPOINT_PROTOCOL_VERSION
                    for event in trace.partials
                )
                or not _is_nonnegative_int(final.samples_seen)
                or final.samples_seen == 0
                or not _is_nonnegative_int(final.model_padding_samples)
                or not _is_nonnegative_int(final.source_samples)
                or final.source_samples == 0
                or not _is_nonnegative_int(final.source_samples_consumed)
                or not _is_nonnegative_int(final.declared_tail_samples)
                or not _is_nonnegative_int(final.tail_samples_consumed)
                or not _is_finite_number(final.audio_seconds)
                or not math.isclose(
                    float(final.audio_seconds),
                    expected_audio_seconds,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or final.source_samples != case.samples
                or final.source_samples_consumed > final.source_samples
                or final.tail_samples_consumed > final.declared_tail_samples
                or (
                    final.source_samples_consumed < final.source_samples
                    and final.tail_samples_consumed != 0
                )
                or final.samples_seen
                != final.source_samples_consumed + final.tail_samples_consumed
                or (
                    final.native_endpoint
                    and (
                        final.endpoint_reason not in {"eou", "eob"}
                        or not _is_nonnegative_int(final.endpoint_sample)
                        or final.endpoint_sample
                        != final.samples_seen + final.model_padding_samples
                        or not _is_finite_number(final.endpoint_probability)
                        or float(final.endpoint_probability) > 1.0
                        or final.authoritative != expected_authoritative
                        or (
                            final.authoritative
                            and (
                                not _is_finite_number(final.endpoint_latency_ms)
                                or abs(
                                    float(final.endpoint_latency_ms)
                                    - (
                                        final.tail_samples_consumed
                                        + final.model_padding_samples
                                    )
                                    * 1000.0
                                    / _SAMPLE_RATE_HZ
                                )
                                > 1e-6
                            )
                        )
                        or (
                            not final.authoritative
                            and final.endpoint_latency_ms is not None
                        )
                    )
                )
                or (
                    not final.native_endpoint
                    and (
                        final.endpoint_reason != "tail_exhausted"
                        or final.endpoint_sample is not None
                        or final.endpoint_probability is not None
                        or final.endpoint_latency_ms is not None
                        or final.authoritative
                        or final.source_samples_consumed != final.source_samples
                        or final.tail_samples_consumed != final.declared_tail_samples
                    )
                )
            ):
                raise ValueError("invalid native endpoint record")
            endpoint_records.append(final)
            endpoint_source_offered += final.source_samples
            endpoint_source_consumed += final.source_samples_consumed
            endpoint_tail_declared += final.declared_tail_samples
            endpoint_tail_consumed += final.tail_samples_consumed
            endpoint_model_padding += final.model_padding_samples
            endpoint_reasons[final.endpoint_reason] += 1
            if final.native_endpoint:
                endpoint_samples.append(float(final.endpoint_sample))
                endpoint_probabilities.append(float(final.endpoint_probability))
                if final.authoritative:
                    authoritative_endpoints += 1
                    endpoint_latencies.append(float(final.endpoint_latency_ms))
                if not source_complete:
                    source_early_native_events += 1
                    if final.endpoint_reason == "eou":
                        source_early_eou_events += 1
                    else:
                        source_early_eob_backchannels += 1
        elif ready.adapter == PARAKEET_REALTIME_EOU_ADAPTER:
            raise ValueError("missing native endpoint record")
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

    expected_records = len(selected_case_indices) * repeats
    repeat_disagreements = sum(len(outputs) > 1 for outputs in final_by_case.values())
    total_compute_ms = sum(compute_ms)
    result = {
        "evaluations": len(records),
        "coverage_complete": (
            len(records) == expected_records
            and len(final_by_case) == len(selected_case_indices)
            and len(seen_runs) == expected_records
        ),
        "accuracy": {
            "transcript": accuracy.as_dict(),
            "command_attempts": command_attempts,
            "command_hits": command_hits,
            "command_recall": (
                round(command_hits / command_attempts, 4)
                if command_attempts
                else None
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
            "max_reported_rss_mb": round(peak_rss, 3),
            "peak_threads": peak_threads,
            "peak_vram_mb": None if peak_vram is None else round(peak_vram, 3),
            "source": "worker_ready_and_final_resource_reports",
            "rss_scope": "maximum_of_ready_and_final_samples_not_process_peak",
        },
    }
    if endpoint_records:
        source_dropped = endpoint_source_offered - endpoint_source_consumed
        tail_unconsumed = endpoint_tail_declared - endpoint_tail_consumed
        model_input_samples = (
            endpoint_source_consumed
            + endpoint_tail_consumed
            + endpoint_model_padding
        )
        model_input_seconds = model_input_samples / _SAMPLE_RATE_HZ
        result["throughput"].update(
            {
                "aggregate_rtf_scope": "declared_source_audio",
                "model_input_total_sec": round(model_input_seconds, 3),
                "model_input_aggregate_rtf": (
                    round((total_compute_ms / 1000.0) / model_input_seconds, 4)
                    if model_input_seconds
                    else 0.0
                ),
            }
        )
        result["endpointing"] = {
            "evaluations": len(endpoint_records),
            "native_endpoints": (
                endpoint_reasons["eou"] + endpoint_reasons["eob"]
            ),
            "native_eou_events": endpoint_reasons["eou"],
            "native_eob_backchannels": endpoint_reasons["eob"],
            "authoritative_endpoints": authoritative_endpoints,
            "early_source_endpoints": source_early_native_events,
            "source_early_native_events": source_early_native_events,
            "source_early_eou_events": source_early_eou_events,
            "source_early_eob_backchannels": source_early_eob_backchannels,
            "tail_exhaustions": endpoint_reasons["tail_exhausted"],
            "reasons": endpoint_reasons,
            "source_samples": {
                "offered_total": endpoint_source_offered,
                "consumed_total": endpoint_source_consumed,
                "dropped_total": source_dropped,
            },
            "declared_tail_samples": {
                "offered_total": endpoint_tail_declared,
                "consumed_total": endpoint_tail_consumed,
                "unconsumed_total": tail_unconsumed,
            },
            "terminal_model_padding_samples_total": endpoint_model_padding,
            "model_input_samples_consumed_total": model_input_samples,
            "endpoint_sample_p50": _percentile(endpoint_samples, 0.50),
            "endpoint_sample_p95": _percentile(endpoint_samples, 0.95),
            "authoritative_endpoint_latency_p50_ms": _percentile(
                endpoint_latencies, 0.50
            ),
            "authoritative_endpoint_latency_p95_ms": _percentile(
                endpoint_latencies, 0.95
            ),
            "authoritative_endpoint_latency_max_ms": _percentile(
                endpoint_latencies, 1.0
            ),
            "endpoint_probability_p50": _percentile(
                endpoint_probabilities, 0.50
            ),
            "endpoint_probability_p95": _percentile(
                endpoint_probabilities, 0.95
            ),
        }
    _require_finite_aggregates(result)
    return result


def aggregate_metrics(
    corpus: LoadedCorpus,
    records: Sequence[RunRecord],
    ready: ReadyEvent,
    *,
    repeats: int,
    stratum_tags: Sequence[str] = (),
) -> dict[str, object]:
    """Reduce private text globally and, when requested, by explicit tags."""

    selected_tags = validate_stratum_tags(corpus, stratum_tags)
    result = _aggregate_metrics_for_cases(
        corpus,
        records,
        ready,
        repeats=repeats,
    )
    if selected_tags:
        strata: list[dict[str, object]] = []
        for tag in selected_tags:
            case_indices = frozenset(
                index
                for index, case in enumerate(corpus.cases)
                if tag in case.tags
            )
            if not case_indices:
                raise ValueError("invalid stratum tags")
            selected_records = tuple(
                record for record in records if record.case_index in case_indices
            )
            strata.append(
                {
                    "tag": tag,
                    "cases": len(case_indices),
                    "metrics": _aggregate_metrics_for_cases(
                        corpus,
                        selected_records,
                        ready,
                        repeats=repeats,
                        expected_case_indices=case_indices,
                    ),
                }
            )
        result["strata"] = strata
    _require_finite_aggregates(result)
    return result
