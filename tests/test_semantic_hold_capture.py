"""Deterministic capture-loop tests for semantic HOLD stream resets."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pytest

from always_on_agent.logical_turn import LogicalTurnAbortReason
from always_on_agent.logical_turn_shadow import (
    LogicalTurnCompositionShadow,
    aggregate_logical_turn_terminal_lineage,
)
from core import diagnostic_bundle
from core.contract import is_stop_command
from core.diagnostic_bundle import (
    BundleAudioSpan,
    DiagnosticStage,
    EndpointReplayConfig,
)
from core.endpointing import ScriptedTurnCompletionDetector
from core.engine import EngineCallbacks, TranscriptAbortReason
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from core.media_session import (
    CaptureBlockContext,
    CaptureGap,
    CaptureGapReason,
    CapturedBlock,
)
from core.realtime_media_stage import RealtimeMediaStage


class _IdentityInput:
    actual_samplerate = 16_000
    actual_device = "test-mic"
    generation = 1


class _ClosedMailbox:
    closed = True


class _ScriptedMedia:
    error = None
    mailbox = _ClosedMailbox()

    def __init__(self, engine, entries, *, generation: int = 1) -> None:
        self.engine = engine
        self.entries = list(entries)
        self.generation = generation

    def take(self, timeout=None):
        del timeout
        if not self.entries:
            self.engine._running.clear()
            return None
        block, action = self.entries.pop(0)
        if action is not None:
            action()
        return block

    def request_stop(self) -> None:
        pass


class _LevelVad:
    def __init__(self) -> None:
        self.speech = False
        self.resets = 0

    def accept_waveform(self, samples) -> None:
        self.speech = bool(float(np.mean(np.abs(samples))) > 0.01)

    def is_speech_detected(self) -> bool:
        return self.speech

    def reset(self) -> None:
        self.resets += 1
        self.speech = False


class _GenerationStream:
    def __init__(self) -> None:
        self.generation = 0
        self.accepted: dict[int, list[np.ndarray]] = {0: []}

    def accept_waveform(self, _sample_rate, samples) -> None:
        self.accepted.setdefault(self.generation, []).append(
            np.asarray(samples, dtype="float32").copy()
        )


class _GenerationRecognizer:
    """Native results/endpoints depend only on PCM since the latest reset."""

    def __init__(self) -> None:
        self.stream = _GenerationStream()
        self.resets = 0

    def create_stream(self):
        return self.stream

    def is_ready(self, _stream) -> bool:
        return False

    def decode_stream(self, _stream) -> None:
        pass

    @staticmethod
    def _means(stream: _GenerationStream) -> tuple[float, ...]:
        return tuple(
            round(float(np.mean(block)), 2)
            for block in stream.accepted.get(stream.generation, ())
        )

    def get_result(self, stream) -> str:
        means = self._means(stream)
        if 0.33 in means:
            return "FRESH TURN"
        if 0.25 in means:
            return "ONE MORE"
        if 0.22 in means or 0.23 in means:
            return "ASK ABOUT"
        if 0.11 in means:
            return "I WANT TO"
        if 0.12 in means:
            return "AND THEN"
        return ""

    def is_endpoint(self, stream) -> bool:
        means = self._means(stream)
        return bool(means and means[-1] == 0.0 and any(means[:-1]))

    def reset(self, stream) -> None:
        self.resets += 1
        stream.generation += 1
        stream.accepted.setdefault(stream.generation, [])


class _CollectingDiagnosticBundle:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []
        self.invalidations: list[str] = []
        self._cursor = 0

    def write_frame(self, tracks, coordinate):
        count = len(next(iter(tracks.values())))
        span = BundleAudioSpan(
            frame_index=len(
                [record for record in self.records if record.get("kind") == "frame"]
            ),
            bundle_sample_start=self._cursor,
            bundle_sample_end=self._cursor + count,
            coordinate=coordinate,
        )
        self._cursor += count
        self.records.append({"kind": "frame"})
        return span

    def observe(self, observation) -> bool:
        self.records.append(observation.to_payload())
        return True

    def write_final_model_input(self, *_args, **_kwargs) -> None:
        return None

    def invalidate(self, code: str) -> None:
        self.invalidations.append(code)


def _captured(
    value: float,
    *,
    at: float,
    sequence: int,
    capture_generation: int = 1,
    gap_before: CaptureGap | None = None,
) -> CapturedBlock:
    samples = np.full(1600, value, dtype="float32")
    return CapturedBlock(
        pcm=samples,
        sample_rate_hz=16_000,
        sequence=sequence,
        captured_started_at=at - 0.1,
        captured_at=at,
        capture_epoch=0,
        source_generation=1,
        capture_generation=capture_generation,
        source_device="test-mic",
        source_sample_start=(sequence - 1) * 1600,
        source_sample_end=sequence * 1600,
        context=CaptureBlockContext(),
        gap_before=gap_before,
    )


@dataclass
class _Run:
    engine: SherpaOnnxEngine
    recognizer: _GenerationRecognizer
    vad: _LevelVad
    detector: ScriptedTurnCompletionDetector
    stage: RealtimeMediaStage
    partials: list
    aborts: list
    commands: list
    logical_turn_shadow: LogicalTurnCompositionShadow | None


def _run(
    blocks: list[CapturedBlock],
    *,
    rule3: float = 20.0,
    actions=None,
    diagnostic=None,
    reset_on_hold: bool = True,
    command_value: float | None = None,
    detector_scores: dict[str, float] | None = None,
    logical_turn_shadow: LogicalTurnCompositionShadow | None = None,
    close_logical_turn_shadow: bool = True,
) -> _Run:
    detector = ScriptedTurnCompletionDetector(
        detector_scores
        or {
            "I WANT TO": 0.05,
            "I WANT TO ASK ABOUT": 0.9,
            "AND THEN": 0.05,
            "FRESH TURN": 0.9,
        }
    )
    engine = SherpaOnnxEngine(
        SherpaConfig(
            endpoint_enabled=True,
            endpoint_reset_on_semantic_hold=reset_on_hold,
            endpoint_min_silence_sec=0.7,
            endpoint_max_silence_sec=1.6,
            asr_rule3_min_utterance_length=rule3,
            barge_in_enabled=False,
        ),
        turn_detector=detector,
    )
    recognizer = _GenerationRecognizer()
    vad = _LevelVad()
    stage = RealtimeMediaStage(max_queued=4)
    partials: list = []
    aborts: list = []
    commands: list = []
    engine._recognizer = recognizer
    engine._vad = vad
    engine._stream_in = _IdentityInput()
    engine._capture_sr = 16_000
    engine._capture_media_session = _ScriptedMedia(
        engine,
        list(zip(blocks, actions or [None] * len(blocks))),
    )
    engine._final_stage = stage
    engine._diagnostic_bundle = diagnostic

    def on_abort(result) -> None:
        aborts.append(result)
        if logical_turn_shadow is not None:
            logical_turn_shadow.abort(
                LogicalTurnAbortReason.SHUTDOWN
                if result.reason is TranscriptAbortReason.SHUTDOWN
                else LogicalTurnAbortReason.TRANSCRIPT_ABORT
            )

    def on_command(result) -> None:
        commands.append(result)
        if logical_turn_shadow is not None:
            logical_turn_shadow.abort(
                LogicalTurnAbortReason.STOP
                if is_stop_command(result.text)
                else LogicalTurnAbortReason.CONTROL
            )

    engine._cb = EngineCallbacks(
        on_partial_result=partials.append,
        on_transcript_abort=on_abort,
        on_command_result=on_command,
    )
    if command_value is not None:

        def poll_keywords(samples, **kwargs) -> bool:
            if not np.isclose(float(np.mean(samples)), command_value):
                return False
            return engine._emit_command_callback(
                "stop",
                detected_at=kwargs.get("detected_at"),
            )

        engine._poll_keywords = poll_keywords
    engine._running.set()

    capture_options = {}
    if logical_turn_shadow is not None:
        capture_options["logical_turn_shadow"] = logical_turn_shadow
    if not close_logical_turn_shadow:
        capture_options["close_logical_turn_shadow"] = False
    engine._capture_loop(**capture_options)

    return _Run(
        engine,
        recognizer,
        vad,
        detector,
        stage,
        partials,
        aborts,
        commands,
        logical_turn_shadow,
    )


def _take_one(run: _Run):
    admitted = run.stage.take(timeout=0.0)
    assert admitted is not None
    assert run.stage.take(timeout=0.0) is None
    return admitted


def test_resumed_speech_composes_one_async_turn_with_whole_pcm() -> None:
    blocks = [
        _captured(0.11, at=10.0, sequence=1),
        _captured(0.0, at=10.8, sequence=2),
        _captured(0.22, at=11.0, sequence=3),
        _captured(0.0, at=11.8, sequence=4),
    ]

    shadow = LogicalTurnCompositionShadow()
    run = _run(blocks, logical_turn_shadow=shadow)
    admitted = _take_one(run)
    item = admitted.payload

    assert [partial.text for partial in run.partials] == [
        "I want to",
        "I want to ask about",
    ]
    assert run.recognizer.resets == 3  # VAD onset, held endpoint, final
    assert run.vad.resets == 1
    assert item.raw_final == "I WANT TO ASK ABOUT"
    np.testing.assert_array_equal(
        item.seg,
        np.concatenate([block.pcm for block in blocks]),
    )
    assert admitted.scope.capture_epoch == 0
    assert admitted.scope.capture_generation == 1
    partial_keys = [partial.acoustic.spans[0].key for partial in run.partials]
    assert partial_keys[0] == partial_keys[1] == item.acoustic.spans[0].key
    assert [partial.revision for partial in run.partials] == [0, 1]
    assert item.revision == 2
    snapshot = shadow.snapshot()
    assert snapshot.closed
    assert snapshot.native_epochs == snapshot.native_finals == 2
    assert snapshot.held_native_finals == 1
    assert snapshot.multi_epoch_commits == 1
    assert snapshot.composition_matches == 1
    assert snapshot.boundary_native_final == 1
    assert run.stage.finish(admitted)


def test_capture_binds_raw_commit_before_async_handoff_without_claiming_delivery() -> None:
    blocks = [
        _captured(0.11, at=12.0, sequence=1),
        _captured(0.0, at=12.8, sequence=2),
        _captured(0.22, at=13.0, sequence=3),
        _captured(0.0, at=13.8, sequence=4),
    ]
    shadow = LogicalTurnCompositionShadow(terminal_lineage=True)

    run = _run(blocks, logical_turn_shadow=shadow)
    admitted = _take_one(run)
    report = aggregate_logical_turn_terminal_lineage(
        (shadow.terminal_lineage_snapshot(),)
    )

    assert report["lineage"]["raw_commits"] == 1
    assert report["lineage"]["claimed_raw_commits"] == 1
    assert report["lineage"]["bound_raw_commits"] == 1
    assert report["lineage"]["missing_downstream_terminals"] == 1
    assert report["lineage"]["unbound_raw_commits"] == 0
    assert report["lineage"]["exact"] is False
    assert report["capabilities"]["async_final_selection_parity"] is False
    assert admitted.payload.acoustic is not None
    assert run.stage.finish(admitted)


def test_terminal_lineage_bind_failure_cannot_change_async_handoff(
    monkeypatch,
) -> None:
    blocks = [
        _captured(0.11, at=12.0, sequence=1),
        _captured(0.0, at=12.8, sequence=2),
        _captured(0.22, at=13.0, sequence=3),
        _captured(0.0, at=13.8, sequence=4),
    ]
    shadow = LogicalTurnCompositionShadow(terminal_lineage=True)

    def fail(*_args, **_kwargs):
        raise RuntimeError("PRIVATE BIND FAILURE")

    monkeypatch.setattr(shadow, "bind_terminal_lineage", fail)
    run = _run(blocks, logical_turn_shadow=shadow)
    admitted = _take_one(run)
    report = aggregate_logical_turn_terminal_lineage(
        (shadow.terminal_lineage_snapshot(),)
    )

    assert admitted.payload.raw_final == "I WANT TO ASK ABOUT"
    assert admitted.payload.acoustic is not None
    assert report["lineage"]["raw_commits"] == 1
    assert report["lineage"]["claimed_raw_commits"] == 1
    assert report["lineage"]["bound_raw_commits"] == 0
    assert report["lineage"]["missing_downstream_terminals"] == 1
    assert report["lineage"]["unbound_raw_commits"] == 1
    assert report["health"]["internal_errors"] == 1
    assert report["lineage"]["exact"] is False
    assert "PRIVATE BIND FAILURE" not in json.dumps(report, sort_keys=True)
    assert run.stage.finish(admitted)


def test_held_prefix_commits_once_at_total_silence_when_reset_stream_is_empty() -> None:
    blocks = [
        _captured(0.12, at=20.0, sequence=1),
        _captured(0.0, at=20.8, sequence=2),
        _captured(0.0, at=21.6, sequence=3),
    ]

    shadow = LogicalTurnCompositionShadow()
    run = _run(blocks, logical_turn_shadow=shadow)
    admitted = _take_one(run)

    assert [partial.text for partial in run.partials] == ["And then"]
    assert admitted.payload.raw_final == "AND THEN"
    assert admitted.payload.acoustic.spans[0].endpoint_reason.value == "asr"
    assert run.recognizer.resets == 3
    snapshot = shadow.snapshot()
    assert snapshot.native_epochs == 2
    assert snapshot.native_finals == 1
    assert snapshot.empty_native_epochs == 1
    assert snapshot.composition_matches == 1
    assert snapshot.boundary_silence_limit == 1
    assert run.stage.finish(admitted)


def test_logical_rule3_survives_reset_while_resumed_vad_is_active() -> None:
    blocks = [
        _captured(0.11, at=30.0, sequence=1),
        _captured(0.0, at=30.8, sequence=2),
        _captured(0.22, at=31.0, sequence=3),
        _captured(0.23, at=31.1, sequence=4),
    ]

    shadow = LogicalTurnCompositionShadow()
    run = _run(
        blocks,
        rule3=1.15,
        logical_turn_shadow=shadow,
    )
    admitted = _take_one(run)

    assert admitted.payload.raw_final == "I WANT TO ASK ABOUT"
    assert admitted.payload.acoustic.spans[0].endpoint_reason.value == "max_wait"
    assert run.detector.calls == ["I WANT TO"]
    assert run.recognizer.resets == 3
    snapshot = shadow.snapshot()
    assert snapshot.composition_matches == 1
    assert snapshot.multi_epoch_commits == 1
    assert snapshot.boundary_hard_limit == 1
    assert run.stage.finish(admitted)


def test_runtime_emits_replayable_pii_free_hold_reset_evidence() -> None:
    blocks = [
        _captured(0.11, at=35.0, sequence=1),
        _captured(0.0, at=35.8, sequence=2),
        _captured(0.22, at=36.0, sequence=3),
        _captured(0.0, at=36.8, sequence=4),
    ]
    bundle = _CollectingDiagnosticBundle()
    run = _run(blocks, diagnostic=bundle)
    admitted = _take_one(run)

    observations = [
        record for record in bundle.records if record.get("kind") == "observation"
    ]
    stages = [record["stage"] for record in observations]
    assert stages.count(DiagnosticStage.ENDPOINT_HELD_RESET.value) == 1
    reset = next(
        record
        for record in observations
        if record["stage"] == DiagnosticStage.ENDPOINT_HELD_RESET.value
    )
    assert reset["ordinal"] == 1
    assert reset["outcome"] == "reset"
    assert "text" not in str(observations).lower()
    assert "I WANT TO" not in str(observations)
    assert bundle.invalidations == []
    assert diagnostic_bundle._validate_endpoint_replay(
        observations,
        EndpointReplayConfig(
            algorithm="adaptive_endpoint_v2",
            enabled=True,
            min_silence_sec=0.7,
            max_silence_sec=1.6,
            complete_threshold=0.6,
            incomplete_threshold=0.3,
            semantic_hold_reset_enabled=True,
            rule3_min_utterance_length_sec=20.0,
        ),
    )
    assert run.stage.finish(admitted)


def test_two_consecutive_holds_reset_in_order_and_commit_once() -> None:
    blocks = [
        _captured(0.11, at=37.0, sequence=1),
        _captured(0.0, at=37.8, sequence=2),
        _captured(0.22, at=38.0, sequence=3),
        _captured(0.0, at=38.8, sequence=4),
        _captured(0.25, at=39.0, sequence=5),
        _captured(0.0, at=39.8, sequence=6),
    ]
    bundle = _CollectingDiagnosticBundle()
    shadow = LogicalTurnCompositionShadow()
    run = _run(
        blocks,
        diagnostic=bundle,
        logical_turn_shadow=shadow,
        detector_scores={
            "I WANT TO": 0.05,
            "I WANT TO ASK ABOUT": 0.05,
            "I WANT TO ASK ABOUT ONE MORE": 0.9,
        },
    )
    admitted = _take_one(run)
    observations = [
        record for record in bundle.records if record.get("kind") == "observation"
    ]
    resets = [
        record
        for record in observations
        if record["stage"] == DiagnosticStage.ENDPOINT_HELD_RESET.value
    ]

    assert [record["ordinal"] for record in resets] == [1, 2]
    assert admitted.payload.raw_final == "I WANT TO ASK ABOUT ONE MORE"
    assert run.recognizer.resets == 4
    snapshot = shadow.snapshot()
    assert snapshot.native_epochs == snapshot.native_finals == 3
    assert snapshot.held_native_finals == 2
    assert snapshot.committed == snapshot.multi_epoch_commits == 1
    assert snapshot.composition_matches == 1
    assert diagnostic_bundle._validate_endpoint_replay(
        observations,
        EndpointReplayConfig(
            algorithm="adaptive_endpoint_v2",
            enabled=True,
            min_silence_sec=0.7,
            max_silence_sec=1.6,
            complete_threshold=0.6,
            incomplete_threshold=0.3,
            semantic_hold_reset_enabled=True,
            rule3_min_utterance_length_sec=20.0,
        ),
    )
    assert run.stage.finish(admitted)


def test_capture_gap_aborts_held_turn_and_cannot_leak_prefix() -> None:
    gap = CaptureGap(
        reason=CaptureGapReason.MEDIA_BACKPRESSURE,
        prior_generation=1,
        generation=2,
        first_dropped_sequence=3,
        last_dropped_sequence=3,
        dropped_frames=1,
        dropped_samples=1600,
    )
    blocks = [
        _captured(0.11, at=40.0, sequence=1),
        _captured(0.0, at=40.8, sequence=2),
        _captured(
            0.33,
            at=41.0,
            sequence=3,
            capture_generation=2,
            gap_before=gap,
        ),
        _captured(0.0, at=41.8, sequence=4, capture_generation=2),
    ]

    shadow = LogicalTurnCompositionShadow()
    run = _run(blocks, logical_turn_shadow=shadow)
    admitted = _take_one(run)

    assert [partial.text for partial in run.partials] == [
        "I want to",
        "Fresh turn",
    ]
    assert len(run.aborts) == 1
    assert run.aborts[0].reason is TranscriptAbortReason.BACKPRESSURE
    assert admitted.payload.raw_final == "FRESH TURN"
    assert "I WANT TO" not in admitted.payload.raw_final
    assert run.partials[0].acoustic.spans[0].key != (
        run.partials[1].acoustic.spans[0].key
    )
    snapshot = shadow.snapshot()
    assert snapshot.started == 2
    assert snapshot.aborted == snapshot.abort_transcript == 1
    assert snapshot.committed == snapshot.composition_matches == 1
    assert snapshot.missing_legacy_terminals == 0
    assert run.stage.finish(admitted)


def test_kws_command_terminal_closes_held_endpoint_replay() -> None:
    blocks = [
        _captured(0.11, at=45.0, sequence=1),
        _captured(0.0, at=45.8, sequence=2),
        _captured(0.44, at=46.0, sequence=3),
    ]
    bundle = _CollectingDiagnosticBundle()

    shadow = LogicalTurnCompositionShadow()
    run = _run(
        blocks,
        diagnostic=bundle,
        command_value=0.44,
        logical_turn_shadow=shadow,
    )

    observations = [
        record for record in bundle.records if record.get("kind") == "observation"
    ]
    assert len(run.commands) == 1
    assert run.commands[0].text == "stop"
    assert run.aborts == []
    assert run.stage.take(timeout=0.0) is None
    snapshot = shadow.snapshot()
    assert snapshot.started == snapshot.aborted == 1
    assert snapshot.abort_stop == 1
    assert snapshot.committed == 0
    assert snapshot.missing_legacy_terminals == 0
    terminal = next(
        record
        for record in observations
        if record["stage"] == DiagnosticStage.FINAL_ABORTED.value
    )
    assert terminal["reason"] == "command"
    assert terminal["utterance_id"] == next(
        record["utterance_id"]
        for record in observations
        if record["stage"] == DiagnosticStage.ENDPOINT_HELD_RESET.value
    )
    assert diagnostic_bundle._validate_endpoint_replay(
        observations,
        EndpointReplayConfig(
            algorithm="adaptive_endpoint_v2",
            enabled=True,
            min_silence_sec=0.7,
            max_silence_sec=1.6,
            complete_threshold=0.6,
            incomplete_threshold=0.3,
            semantic_hold_reset_enabled=True,
            rule3_min_utterance_length_sec=20.0,
        ),
    )
    assert run.engine._diagnostic_spans == {}


def test_reset_feature_config_is_strict_opt_in_and_requires_live_vad() -> None:
    assert SherpaConfig().endpoint_reset_on_semantic_hold is False
    assert SherpaConfig.from_dict(
        {"endpoint_reset_on_semantic_hold": True}
    ).endpoint_reset_on_semantic_hold is True
    with pytest.raises(ValueError, match="requires endpoint_enabled"):
        SherpaOnnxEngine(
            SherpaConfig(endpoint_reset_on_semantic_hold=True)
        )
    with pytest.raises(TypeError, match="must be a boolean"):
        SherpaOnnxEngine(
            SherpaConfig(
                endpoint_enabled=True,
                endpoint_reset_on_semantic_hold=1,  # type: ignore[arg-type]
            )
        )

    engine = SherpaOnnxEngine(
        SherpaConfig(
            endpoint_enabled=True,
            endpoint_reset_on_semantic_hold=True,
        )
    )
    engine._build = lambda: setattr(engine, "_vad", None)
    with pytest.raises(RuntimeError, match="requires a live VAD"):
        engine.start(EngineCallbacks())
    assert not engine._running.is_set()


def test_flag_off_preserves_non_reset_semantic_hold_path() -> None:
    blocks = [
        _captured(0.11, at=50.0, sequence=1),
        _captured(0.0, at=50.8, sequence=2),
    ]

    run = _run(blocks, reset_on_hold=False)

    assert [partial.text for partial in run.partials] == ["I want to"]
    assert run.detector.calls == ["I WANT TO"]
    assert run.recognizer.resets == 1  # first-VAD rebase only
    assert run.logical_turn_shadow is None
    assert run.stage.take(timeout=0.0) is None
    assert len(run.aborts) == 1
    assert run.aborts[0].reason is TranscriptAbortReason.SHUTDOWN


def test_shadow_failure_cannot_change_capture_callbacks(monkeypatch) -> None:
    blocks = [
        _captured(0.11, at=60.0, sequence=1),
        _captured(0.0, at=60.8, sequence=2),
        _captured(0.22, at=61.0, sequence=3),
        _captured(0.0, at=61.8, sequence=4),
    ]
    baseline = _run(blocks)
    failing_shadow = LogicalTurnCompositionShadow()

    def fail_observation(**_kwargs):
        raise RuntimeError("fixed shadow failure")

    monkeypatch.setattr(
        failing_shadow,
        "observe_native_final",
        fail_observation,
    )
    observed = _run(blocks, logical_turn_shadow=failing_shadow)
    baseline_final = _take_one(baseline)
    observed_final = _take_one(observed)

    assert observed_final.payload.raw_final == baseline_final.payload.raw_final
    assert [item.text for item in observed.partials] == [
        item.text for item in baseline.partials
    ]
    assert len(observed.aborts) == len(baseline.aborts)
    snapshot = failing_shadow.snapshot()
    assert snapshot.closed
    assert snapshot.internal_errors == 2
    assert snapshot.extra_legacy_terminals == 1
    assert baseline.stage.finish(baseline_final)
    assert observed.stage.finish(observed_final)


def test_capture_can_defer_both_shadow_closes_until_async_delivery_drains() -> None:
    blocks = [
        _captured(0.11, at=65.0, sequence=1),
        _captured(0.0, at=65.8, sequence=2),
        _captured(0.22, at=66.0, sequence=3),
        _captured(0.0, at=66.8, sequence=4),
    ]
    shadow = LogicalTurnCompositionShadow(terminal_lineage=True)

    run = _run(
        blocks,
        logical_turn_shadow=shadow,
        close_logical_turn_shadow=False,
    )

    open_snapshot = shadow.snapshot()
    assert not open_snapshot.closed
    assert open_snapshot.committed == 1
    assert not shadow.terminal_lineage_snapshot().closed
    assert run.engine._logical_turn_shadow_observer() is None
    admitted = _take_one(run)
    assert run.stage.finish(admitted)

    shadow.close()
    assert shadow.snapshot().closed
    assert shadow.terminal_lineage_snapshot().closed


def test_default_capture_never_calls_shadow_evidence_helpers(monkeypatch) -> None:
    blocks = [
        _captured(0.11, at=70.0, sequence=1),
        _captured(0.0, at=70.8, sequence=2),
        _captured(0.22, at=71.0, sequence=3),
        _captured(0.0, at=71.8, sequence=4),
    ]

    def forbidden(*_args, **_kwargs) -> None:
        raise AssertionError("default capture called a shadow evidence helper")

    for name in (
        "_logical_turn_shadow_failure",
        "_logical_turn_shadow_observe",
        "_logical_turn_shadow_observe_empty",
        "_logical_turn_shadow_compare",
        "_logical_turn_shadow_bind_terminal",
    ):
        monkeypatch.setattr(SherpaOnnxEngine, name, forbidden)

    run = _run(blocks)
    admitted = _take_one(run)
    assert admitted.payload.raw_final == "I WANT TO ASK ABOUT"
    assert run.logical_turn_shadow is None
    assert run.stage.finish(admitted)


def test_shadow_binding_cleans_up_after_initialization_failure(monkeypatch) -> None:
    engine = SherpaOnnxEngine(SherpaConfig())
    shadow = LogicalTurnCompositionShadow()
    assert shadow.observe_native_final(
        native_epoch=1,
        native_sequence=0,
        text="PRIVATE",
        held=True,
    )
    monkeypatch.setattr(engine, "_claim_streaming_decode_session", lambda: None)

    def fail_body(
        _decode_session,
        *,
        close_logical_turn_shadow: bool = True,
    ) -> None:
        assert close_logical_turn_shadow
        raise RuntimeError("fixed initialization failure")

    monkeypatch.setattr(engine, "_capture_loop_body", fail_body)
    engine._running.set()
    engine._capture_loop(logical_turn_shadow=shadow)

    snapshot = shadow.snapshot()
    assert snapshot.closed
    assert snapshot.started == snapshot.aborted == snapshot.abort_shutdown == 1
    assert engine._logical_turn_shadow_observer() is None

    observed = []

    def clean_body(
        _decode_session,
        *,
        close_logical_turn_shadow: bool = True,
    ) -> None:
        assert close_logical_turn_shadow
        observed.append(engine._logical_turn_shadow_observer())

    monkeypatch.setattr(engine, "_capture_loop_body", clean_body)
    engine._running.set()
    engine._capture_loop()
    assert observed == [None]
    assert engine._logical_turn_shadow_observer() is None


def test_deferred_shadow_remains_caller_owned_after_capture_error(monkeypatch) -> None:
    engine = SherpaOnnxEngine(SherpaConfig())
    shadow = LogicalTurnCompositionShadow()
    assert shadow.observe_native_final(
        native_epoch=1,
        native_sequence=0,
        text="PRIVATE",
        held=True,
    )
    monkeypatch.setattr(engine, "_claim_streaming_decode_session", lambda: None)

    def fail_body(
        _decode_session,
        *,
        close_logical_turn_shadow: bool = True,
    ) -> None:
        assert close_logical_turn_shadow is False
        raise RuntimeError("fixed deferred initialization failure")

    monkeypatch.setattr(engine, "_capture_loop_body", fail_body)
    engine._running.set()
    engine._capture_loop(
        logical_turn_shadow=shadow,
        close_logical_turn_shadow=False,
    )

    snapshot = shadow.snapshot()
    assert not snapshot.closed
    assert snapshot.active
    assert engine._logical_turn_shadow_observer() is None

    shadow.close()
    closed = shadow.snapshot()
    assert closed.closed
    assert closed.started == closed.aborted == closed.abort_shutdown == 1


def test_shadow_close_ownership_flags_are_strict_booleans() -> None:
    engine = SherpaOnnxEngine(SherpaConfig())

    with pytest.raises(TypeError, match="close_logical_turn_shadow"):
        engine._capture_loop(close_logical_turn_shadow=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="close_logical_turn_shadow"):
        engine._capture_loop_body(
            None,
            close_logical_turn_shadow=0,  # type: ignore[arg-type]
        )
