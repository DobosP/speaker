from __future__ import annotations

import time

import pytest

from always_on_agent.acoustic import AcousticLineage, AcousticSource, AcousticSpan
from core.engine import (
    AudioEngine,
    EngineCallbacks,
    FinalTranscript,
    OwnerVerification,
    TranscriptAbort,
    TranscriptAbortReason,
)
from core.engines.sherpa import SherpaExecutionPurpose
from core.guided_stt_plan import built_in_guided_stt_plan
from core.stt_capture import CaptureOnlyError, CaptureOnlySession


_PRIVATE_TEXT_CANARY = "recognized private text must not be rendered"


def _final(
    ordinal: int,
    *,
    speech_start_at: float | None = None,
    source: AcousticSource = AcousticSource.LIVE_CAPTURE,
    owner: OwnerVerification = OwnerVerification.UNKNOWN,
    acoustic: bool = True,
    stream_id: str = "capture-stream",
    capture_generation: int = 1,
) -> FinalTranscript:
    now = time.perf_counter()
    start = now if speech_start_at is None else speech_start_at
    lineage = (
        AcousticLineage.single(
            AcousticSpan(
                stream_id=stream_id,
                utterance_id=f"utterance-{ordinal}",
                capture_epoch=1,
                capture_generation=capture_generation,
                source=source,
                speech_start_at=start,
                speech_end_at=max(start, now),
                endpoint_committed_at=max(start, now),
                emitted_at=max(start, now),
                sample_rate_hz=16000,
                owned_sample_count=1600,
            )
        )
        if acoustic
        else None
    )
    return FinalTranscript(
        _PRIVATE_TEXT_CANARY,
        owner_verification=owner,
        origin="live_audio",
        acoustic=lineage,
        revision=1 if lineage is not None else 0,
    )


class _CaptureEngine(AudioEngine):
    execution_purpose = SherpaExecutionPurpose.STT_CAPTURE_ONLY

    def __init__(self, *, initial_state: str | None = "open") -> None:
        self.initial_state = initial_state
        self.callbacks: EngineCallbacks | None = None
        self.start_calls = 0
        self.warm_stt_calls = 0
        self.stop_calls = 0

    def start(self, callbacks: EngineCallbacks) -> None:
        self.start_calls += 1
        self.callbacks = callbacks
        if self.initial_state is not None:
            callbacks.on_capture_state(self.initial_state, "private device detail")

    def stop(self) -> None:
        self.stop_calls += 1

    def warm_stt(self) -> None:
        self.warm_stt_calls += 1

    def speak(self, text: str, on_done=None) -> None:  # noqa: ARG002
        raise AssertionError("capture-only session attempted playback")

    def stop_speaking(self) -> None:
        raise AssertionError("capture-only session attempted a barge effect")


class _AutomaticGuide:
    def __init__(self, engine: _CaptureEngine) -> None:
        self.engine = engine
        self.geometries: list[str] = []
        self.prepared: list[str] = []

    def acknowledge_geometry(self, geometry: str) -> None:
        assert self.engine.warm_stt_calls == 1
        self.geometries.append(geometry)

    def prepare_case(self, case, *, ordinal: int, total: int) -> None:  # noqa: ARG002
        self.prepared.append(case.case_id)

    def case_armed(self, *, ordinal: int, total: int) -> None:  # noqa: ARG002
        assert self.engine.callbacks is not None
        self.engine.callbacks.on_final_result(_final(ordinal))


def _session(engine: _CaptureEngine, guide, **kwargs) -> CaptureOnlySession:
    return CaptureOnlySession(
        engine,
        built_in_guided_stt_plan(),
        guide=guide,
        post_terminal_quiet_sec=0.0,
        **kwargs,
    )


def test_exact_plan_completes_with_two_geometry_acknowledgements_and_one_stop(
    caplog,
) -> None:
    engine = _CaptureEngine()
    guide = _AutomaticGuide(engine)

    result = _session(engine, guide).run()

    assert result.completed
    assert result.planned_cases == result.accepted_finals == 16
    assert result.capture_state == "open"
    assert guide.geometries == ["close", "far"]
    assert len(guide.prepared) == 16
    assert engine.start_calls == engine.warm_stt_calls == engine.stop_calls == 1
    assert _PRIVATE_TEXT_CANARY not in caplog.text
    assert _PRIVATE_TEXT_CANARY not in repr(result)


class _PreArmGuide(_AutomaticGuide):
    def acknowledge_geometry(self, geometry: str) -> None:
        super().acknowledge_geometry(geometry)
        assert self.engine.callbacks is not None
        self.engine.callbacks.on_final_result(_final(900))


def test_final_before_case_arm_fails_and_stops_once() -> None:
    engine = _CaptureEngine()

    with pytest.raises(CaptureOnlyError, match="unarmed"):
        _session(engine, _PreArmGuide(engine)).run()

    assert engine.stop_calls == 1


class _QueuedPreArmGuide(_AutomaticGuide):
    def __init__(self, engine: _CaptureEngine) -> None:
        super().__init__(engine)
        self.pre_arm_at = 0.0

    def prepare_case(self, case, *, ordinal: int, total: int) -> None:
        super().prepare_case(case, ordinal=ordinal, total=total)
        self.pre_arm_at = time.perf_counter()

    def case_armed(self, *, ordinal: int, total: int) -> None:  # noqa: ARG002
        assert self.engine.callbacks is not None
        self.engine.callbacks.on_final_result(
            _final(ordinal, speech_start_at=self.pre_arm_at)
        )


def test_queued_pre_arm_final_delivered_after_arm_is_rejected_by_lineage_time() -> None:
    engine = _CaptureEngine()

    with pytest.raises(CaptureOnlyError, match="lineage"):
        _session(engine, _QueuedPreArmGuide(engine)).run()

    assert engine.stop_calls == 1


class _SelectedFinalGuide(_AutomaticGuide):
    def __init__(self, engine: _CaptureEngine, final: FinalTranscript) -> None:
        super().__init__(engine)
        self.final = final

    def case_armed(self, *, ordinal: int, total: int) -> None:  # noqa: ARG002
        assert self.engine.callbacks is not None
        self.engine.callbacks.on_final_result(self.final)


@pytest.mark.parametrize(
    ("final", "message"),
    [
        (_final(1, acoustic=False), "lineage"),
        (_final(1, source=AcousticSource.FILE_REPLAY), "lineage"),
        (
            _final(1, owner=OwnerVerification.VERIFIED),
            "speaker authority",
        ),
    ],
)
def test_missing_foreign_or_verified_selected_final_fails_closed(
    final: FinalTranscript,
    message: str,
) -> None:
    engine = _CaptureEngine()

    with pytest.raises(CaptureOnlyError, match=message):
        _session(engine, _SelectedFinalGuide(engine, final)).run()


def test_duplicate_span_keys_inside_one_final_fail_closed() -> None:
    engine = _CaptureEngine()
    final = _final(1, speech_start_at=time.perf_counter() + 1.0)
    assert final.acoustic is not None
    span = final.acoustic.spans[0]
    object.__setattr__(final.acoustic, "spans", (span, span))

    with pytest.raises(CaptureOnlyError, match="duplicate acoustic"):
        _session(engine, _SelectedFinalGuide(engine, final)).run()


@pytest.mark.parametrize(
    ("initial_state", "message"),
    [(None, "not open"), ("recovering", "recovery invalidated")],
)
def test_case_cannot_arm_until_capture_state_is_open(
    initial_state: str | None,
    message: str,
) -> None:
    engine = _CaptureEngine(initial_state=initial_state)

    with pytest.raises(CaptureOnlyError, match=message):
        _session(engine, _AutomaticGuide(engine)).run()

    assert engine.stop_calls == 1


class _RecoveringOnFinalGuide(_AutomaticGuide):
    def case_armed(self, *, ordinal: int, total: int) -> None:  # noqa: ARG002
        assert self.engine.callbacks is not None
        self.engine.callbacks.on_capture_state("recovering", "private detail")
        self.engine.callbacks.on_final_result(_final(ordinal))


def test_recovering_capture_cannot_accept_a_case_terminal() -> None:
    engine = _CaptureEngine()

    with pytest.raises(CaptureOnlyError, match="recovery invalidated"):
        _session(engine, _RecoveringOnFinalGuide(engine)).run()


class _RecoverThenOpenGuide(_AutomaticGuide):
    def prepare_case(self, case, *, ordinal: int, total: int) -> None:
        super().prepare_case(case, ordinal=ordinal, total=total)
        if ordinal == 2:
            assert self.engine.callbacks is not None
            self.engine.callbacks.on_capture_state("recovering", "private detail")
            self.engine.callbacks.on_capture_state("open", "private detail")


def test_recovering_then_open_still_invalidates_the_fixed_plan() -> None:
    engine = _CaptureEngine()

    with pytest.raises(CaptureOnlyError, match="recovery invalidated"):
        _session(engine, _RecoverThenOpenGuide(engine)).run()

    assert engine.stop_calls == 1


class _DuplicateGuide(_AutomaticGuide):
    def case_armed(self, *, ordinal: int, total: int) -> None:  # noqa: ARG002
        assert self.engine.callbacks is not None
        self.engine.callbacks.on_final_result(_final(ordinal))
        self.engine.callbacks.on_final_result(_final(800 + ordinal))


def test_second_split_final_fails_while_case_is_unarmed() -> None:
    engine = _CaptureEngine()

    with pytest.raises(CaptureOnlyError, match="extra final"):
        _session(engine, _DuplicateGuide(engine)).run()


class _NoFinalGuide(_AutomaticGuide):
    def case_armed(self, *, ordinal: int, total: int) -> None:
        pass


def test_case_timeout_is_bounded_and_stops_once() -> None:
    engine = _CaptureEngine()

    with pytest.raises(CaptureOnlyError, match="timed out"):
        _session(engine, _NoFinalGuide(engine), case_timeout_sec=0.01).run()

    assert engine.stop_calls == 1


class _StopFinalEngine(_CaptureEngine):
    def stop(self) -> None:
        super().stop()
        assert self.callbacks is not None
        self.callbacks.on_final_result(_final(999))


def test_stop_time_extra_final_defeats_provisional_complete_count() -> None:
    engine = _StopFinalEngine()

    with pytest.raises(CaptureOnlyError, match="extra final"):
        _session(engine, _AutomaticGuide(engine)).run()

    assert engine.stop_calls == 1


class _StopAbortEngine(_CaptureEngine):
    def stop(self) -> None:
        super().stop()
        assert self.callbacks is not None
        final = _final(999)
        assert final.acoustic is not None
        self.callbacks.on_transcript_abort(
            TranscriptAbort(
                acoustic=final.acoustic,
                revision=final.revision,
                reason=TranscriptAbortReason.SHUTDOWN,
            )
        )


def test_stop_time_abort_defeats_provisional_complete_count() -> None:
    engine = _StopAbortEngine()

    with pytest.raises(CaptureOnlyError, match="transcript abort"):
        _session(engine, _AutomaticGuide(engine)).run()

    assert engine.stop_calls == 1


def test_capture_session_rejects_normal_assistant_engine() -> None:
    engine = _CaptureEngine()
    engine.execution_purpose = SherpaExecutionPurpose.FULL_ASSISTANT

    with pytest.raises(TypeError, match="capture-purpose"):
        CaptureOnlySession(engine, built_in_guided_stt_plan())
