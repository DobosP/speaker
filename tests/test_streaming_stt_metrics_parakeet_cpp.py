from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tools.streaming_stt.corpus import CorpusCase, LoadedCorpus
from tools.streaming_stt.manifest import PARAKEET_CPP_ADAPTER
from tools.streaming_stt.metrics import RunRecord, aggregate_metrics
from tools.streaming_stt.protocol import (
    NATIVE_ENDPOINT_PROTOCOL_VERSION,
    PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION,
    PROTOCOL_VERSION,
    CaseTrace,
    FinalEvent,
    ParakeetCppFinalEvent,
    ParakeetCppObservedEvent,
    PartialEvent,
    ReadyEvent,
    ResourceUsage,
)


_USAGE = ResourceUsage(rss_mb=10.0, threads=1, vram_mb=None)
_SOURCE_SAMPLES = 16_000
_TAIL_SAMPLES = 1_280


def _corpus(tmp_path: Path) -> LoadedCorpus:
    case = CorpusCase(
        case_id="aggregate-only",
        source_path=tmp_path / "aggregate-only.f32le",
        sha256="a" * 64,
        samples=_SOURCE_SAMPLES,
        expected_text="hello",
        assertion="transcript",
        commands=(),
        forbidden_commands=(),
        tags=(),
        audio_bytes=b"",
    )
    return LoadedCorpus(
        path=tmp_path / "corpus.json",
        digest="b" * 64,
        schema_version=1,
        purpose="aggregate-only parakeet.cpp metric contract",
        provenance=None,
        cases=(case,),
        audio_bytes=0,
    )


def _ready(
    *, protocol_version: int = PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION
) -> ReadyEvent:
    return ReadyEvent(
        model_id="parakeet-cpp-test",
        manifest_sha256="c" * 64,
        source_bundle_sha256="d" * 64,
        adapter=PARAKEET_CPP_ADAPTER,
        model_load_ms=5.0,
        resources=_USAGE,
        runtime={"python": "3.12", "platform": "linux"},
        protocol_version=protocol_version,
    )


def _observed(
    event_type: str,
    event_origin: str,
    observed_sample: int,
    encoder_frame: int,
) -> ParakeetCppObservedEvent:
    return ParakeetCppObservedEvent(
        event_type=event_type,
        event_origin=event_origin,
        observed_sample=observed_sample,
        encoder_frame=encoder_frame,
        encoder_time_seconds=encoder_frame * 0.08,
    )


def _final(
    *,
    observed_events: tuple[ParakeetCppObservedEvent, ...],
    tail_consumed: int,
) -> ParakeetCppFinalEvent:
    samples_seen = _SOURCE_SAMPLES + tail_consumed
    first_eou = next(
        (event for event in observed_events if event.event_type == "eou"),
        None,
    )
    accepted = (
        first_eou
        if first_eou is not None
        and first_eou.event_origin == "feed"
        and first_eou.observed_sample >= _SOURCE_SAMPLES
        else None
    )
    return ParakeetCppFinalEvent(
        request_id="request",
        seq=0,
        text="hello",
        samples_seen=samples_seen,
        elapsed_ms=100.0,
        finalization_ms=20.0,
        compute_ms=4.0,
        audio_seconds=1.0,
        chunks=max(1, (samples_seen + 1_279) // 1_280),
        deadline_misses=0,
        max_backlog_ms=0.0,
        resources=_USAGE,
        model_padding_samples=0,
        source_samples_offered=_SOURCE_SAMPLES,
        source_samples_consumed=_SOURCE_SAMPLES,
        tail_samples_offered=_TAIL_SAMPLES,
        tail_samples_consumed=tail_consumed,
        observed_events=observed_events,
        endpoint_reason="eou" if accepted is not None else "tail_exhausted",
        native_endpoint=accepted is not None,
        encoder_frame=None if accepted is None else accepted.encoder_frame,
        encoder_time_seconds=(
            None if accepted is None else accepted.encoder_time_seconds
        ),
        event_origin="none" if accepted is None else "feed",
        endpoint_sample=None if accepted is None else accepted.observed_sample,
        endpoint_latency_ms=None if accepted is None else tail_consumed / 16.0,
        authoritative=accepted is not None,
        protocol_version=PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION,
    )


def _accepted_final() -> ParakeetCppFinalEvent:
    return _final(
        observed_events=(
            _observed("eob", "feed", 3_840, 2),
            _observed("eob", "feed", 7_680, 5),
            _observed("eob", "feed", 16_640, 10),
            _observed("eou", "feed", 16_640, 13),
            _observed("eob", "finalize", 16_640, 14),
        ),
        tail_consumed=640,
    )


def _tail_exhausted_final() -> ParakeetCppFinalEvent:
    return _final(
        observed_events=(
            _observed("eou", "feed", 7_680, 5),
            _observed("eob", "feed", 11_520, 7),
            _observed("eob", "feed", 17_280, 11),
            _observed("eou", "feed", 17_280, 12),
            _observed("eou", "finalize", 17_280, 13),
        ),
        tail_consumed=_TAIL_SAMPLES,
    )


def _record(final: ParakeetCppFinalEvent) -> RunRecord:
    return RunRecord(0, 0, CaseTrace(partials=(), final=final))


def _assert_rejected(tmp_path: Path, final: ParakeetCppFinalEvent) -> None:
    with pytest.raises(ValueError, match="invalid parakeet.cpp endpoint record"):
        aggregate_metrics(
            _corpus(tmp_path),
            (_record(final),),
            _ready(),
            repeats=1,
        )


def test_parakeet_cpp_metrics_separate_terminals_from_all_observations(tmp_path):
    partial = PartialEvent(
        "request",
        0,
        "hello",
        1_280,
        10.0,
        1.0,
        protocol_version=PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION,
    )
    records = (
        RunRecord(
            0,
            0,
            CaseTrace(partials=(partial,), final=_accepted_final()),
        ),
        RunRecord(
            0,
            1,
            CaseTrace(partials=(), final=_tail_exhausted_final()),
        ),
    )

    metrics = aggregate_metrics(_corpus(tmp_path), records, _ready(), repeats=2)

    assert metrics["coverage_complete"] is True
    assert metrics["throughput"] == {
        "audio_total_sec": 2.0,
        "compute_total_ms": 8.0,
        "aggregate_rtf": 0.004,
        "aggregate_rtf_scope": "declared_source_audio",
        "model_input_total_sec": 2.12,
        "model_input_aggregate_rtf": 0.0038,
    }
    assert metrics["endpointing"] == {
        "evaluations": 2,
        "accepted_endpoints": 1,
        "tail_exhaustions": 1,
        "terminal_reasons": {"eou": 1, "tail_exhausted": 1},
        "terminal_origins": {"feed": 1, "none": 1},
        "observations": {
            "total": 10,
            "eou": 4,
            "eob": 6,
            "origins": {"feed": 8, "finalize": 2},
            "source_early": {"total": 4, "eou": 1, "eob": 3},
            "post_source_feed": {"total": 4, "eou": 2, "eob": 2},
        },
        "source_samples": {
            "offered_total": 32_000,
            "consumed_total": 32_000,
            "dropped_total": 0,
        },
        "tail_samples": {
            "offered_total": 2_560,
            "consumed_total": 1_920,
            "unconsumed_total": 640,
        },
        "terminal_model_padding_samples_total": 0,
        "model_input_samples_consumed_total": 33_920,
        "accepted_endpoint_sample_p50": 16_640.0,
        "accepted_endpoint_sample_p95": 16_640.0,
        "observed_encoder_frame_p50": 10.0,
        "observed_encoder_frame_p95": 14.0,
        "observed_encoder_time_p50_seconds": 0.8,
        "observed_encoder_time_p95_seconds": 1.12,
        "accepted_endpoint_latency_p50_ms": 40.0,
        "accepted_endpoint_latency_p95_ms": 40.0,
        "accepted_endpoint_latency_max_ms": 40.0,
    }

    def keys(value: object) -> tuple[str, ...]:
        if not isinstance(value, dict):
            return ()
        return tuple(value) + tuple(
            nested for item in value.values() for nested in keys(item)
        )

    assert not any("probability" in key for key in keys(metrics["endpointing"]))


@pytest.mark.parametrize(
    "mutation",
    [
        {"protocol_version": NATIVE_ENDPOINT_PROTOCOL_VERSION},
        {"source_samples_offered": _SOURCE_SAMPLES - 1},
        {"source_samples_consumed": _SOURCE_SAMPLES - 1},
        {"tail_samples_consumed": _TAIL_SAMPLES + 1},
        {"samples_seen": 16_639},
        {"model_padding_samples": 1},
        {"model_padding_samples": False},
        {"audio_seconds": 1.001},
        {"endpoint_reason": "eob"},
        {"event_origin": "finalize"},
        {"encoder_frame": 12},
        {"encoder_time_seconds": 0.96},
        {"endpoint_sample": 16_639},
        {"endpoint_latency_ms": 39.0},
        {"authoritative": False},
        {"native_endpoint": False},
    ],
)
def test_parakeet_cpp_metrics_reject_terminal_policy_mutations(
    tmp_path,
    mutation,
):
    _assert_rejected(tmp_path, replace(_accepted_final(), **mutation))


def test_parakeet_cpp_metrics_independently_reject_observation_mutations(
    tmp_path,
):
    final = _accepted_final()
    events = final.observed_events
    invalid_event_sets: tuple[object, ...] = (
        list(events),
        (object(),),
        (replace(events[0], observed_sample=0), *events[1:]),
        (*events[:-1], replace(events[-1], observed_sample=16_641)),
        (events[1], events[0], *events[2:]),
        (*events[:2], replace(events[2], encoder_frame=4), *events[3:]),
        (*events[:2], replace(events[2], encoder_time_seconds=0.81), *events[3:]),
        (
            events[0],
            replace(
                events[1],
                event_type="eob",
                encoder_frame=2,
                encoder_time_seconds=0.16,
            ),
            *events[2:],
        ),
        (*events[:-1], replace(events[-1], event_origin="feed")),
        (replace(events[0], observed_sample=1_000, encoder_frame=20), *events[1:]),
        (*events[:-1], replace(events[-1], observed_sample=16_639)),
    )
    for invalid_events in invalid_event_sets:
        _assert_rejected(
            tmp_path,
            replace(final, observed_events=invalid_events),  # type: ignore[arg-type]
        )

    tail = _tail_exhausted_final()
    later_complete_eou = tail.observed_events[3]
    _assert_rejected(
        tmp_path,
        replace(
            tail,
            endpoint_reason="eou",
            native_endpoint=True,
            encoder_frame=later_complete_eou.encoder_frame,
            encoder_time_seconds=later_complete_eou.encoder_time_seconds,
            event_origin="feed",
            endpoint_sample=later_complete_eou.observed_sample,
            endpoint_latency_ms=80.0,
            authoritative=True,
        ),
    )


def test_parakeet_cpp_metrics_first_early_eou_forces_full_tail_exhaustion(
    tmp_path,
):
    early = _observed("eou", "feed", 7_680, 5)
    partial_tail = _final(observed_events=(early,), tail_consumed=640)
    _assert_rejected(tmp_path, partial_tail)


def test_parakeet_cpp_metrics_first_finalize_eou_is_never_accepted(tmp_path):
    final = _final(
        observed_events=(
            _observed("eob", "feed", 17_280, 11),
            _observed("eou", "finalize", 17_280, 13),
        ),
        tail_consumed=_TAIL_SAMPLES,
    )
    metrics = aggregate_metrics(
        _corpus(tmp_path),
        (_record(final),),
        _ready(),
        repeats=1,
    )
    assert metrics["endpointing"]["accepted_endpoints"] == 0
    assert metrics["endpointing"]["tail_exhaustions"] == 1


@pytest.mark.parametrize(
    ("event_index", "impossible_sample"),
    [(0, 4_001), (2, 16_001)],
)
def test_parakeet_cpp_metrics_reject_feed_observations_off_native_boundaries(
    tmp_path,
    event_index,
    impossible_sample,
):
    final = _accepted_final()
    events = list(final.observed_events)
    events[event_index] = replace(
        events[event_index], observed_sample=impossible_sample
    )
    _assert_rejected(
        tmp_path,
        replace(final, observed_events=tuple(events)),
    )


def test_parakeet_cpp_metrics_reject_near_time_duplicate_event_identity(
    tmp_path,
):
    final = _accepted_final()
    events = final.observed_events
    near_time_duplicate = replace(
        events[1],
        event_type=events[0].event_type,
        encoder_frame=events[0].encoder_frame,
        encoder_time_seconds=events[0].encoder_time_seconds + 0.0000005,
    )
    _assert_rejected(
        tmp_path,
        replace(
            final,
            observed_events=(events[0], near_time_duplicate, *events[2:]),
        ),
    )


def test_parakeet_cpp_metrics_reject_wrong_ready_partial_and_terminal_protocols(
    tmp_path,
):
    corpus = _corpus(tmp_path)
    final = _accepted_final()
    bad_partial = PartialEvent(
        "request",
        0,
        "hello",
        1_280,
        10.0,
        1.0,
        protocol_version=PROTOCOL_VERSION,
    )
    ordinary_final = FinalEvent(
        request_id="request",
        seq=0,
        text="hello",
        samples_seen=_SOURCE_SAMPLES,
        elapsed_ms=100.0,
        finalization_ms=20.0,
        compute_ms=4.0,
        audio_seconds=1.0,
        chunks=13,
        deadline_misses=0,
        max_backlog_ms=0.0,
        resources=_USAGE,
        protocol_version=PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION,
    )

    with pytest.raises(ValueError, match="ready event"):
        aggregate_metrics(
            corpus,
            (_record(final),),
            _ready(protocol_version=PROTOCOL_VERSION),
            repeats=1,
        )
    with pytest.raises(ValueError, match="endpoint record"):
        aggregate_metrics(
            corpus,
            (RunRecord(0, 0, CaseTrace((bad_partial,), final)),),
            _ready(),
            repeats=1,
        )
    with pytest.raises(ValueError, match="missing native endpoint record"):
        aggregate_metrics(
            corpus,
            (RunRecord(0, 0, CaseTrace((), ordinary_final)),),
            _ready(),
            repeats=1,
        )
