"""PII-free Sherpa lifecycle evidence, without models or audio devices."""

from __future__ import annotations

import json
import time

import numpy as np

from always_on_agent.acoustic import (
    AcousticLineage,
    AcousticSource,
    AcousticSpan,
    EndpointReason,
)
from core.diagnostic_bundle import DiagnosticStage, validate_manifest
from core.engine import EngineCallbacks
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from core.media_session import CapturedBlock


def _captured(count: int) -> CapturedBlock:
    return CapturedBlock(
        pcm=np.zeros(count, dtype="float32"),
        sample_rate_hz=16000,
        sequence=1,
        captured_started_at=10.0,
        captured_at=10.0 + count / 16000,
        capture_epoch=2,
        source_generation=3,
        capture_generation=4,
        source_sample_start=0,
        source_sample_end=count,
    )


def _acoustic(count: int) -> AcousticLineage:
    return AcousticLineage.single(
        AcousticSpan(
            stream_id="sherpa-11111111111111111111111111111111",
            utterance_id="u1",
            capture_epoch=2,
            capture_generation=4,
            source=AcousticSource.LIVE_CAPTURE,
            speech_start_at=10.0,
            speech_end_at=10.1,
            endpoint_committed_at=10.2,
            sample_rate_hz=16000,
            owned_sample_count=count,
            endpoint_reason=EndpointReason.ASR,
        )
    )


def _record_endpoint(engine, acoustic: AcousticLineage, frame_span) -> None:
    engine._diagnostic_observe(
        DiagnosticStage.ENDPOINT_EVALUATED,
        at=12.0,
        bundle_span=frame_span,
        acoustic=acoustic,
        reason="asr",
        basis="acoustic",
        completion_state="unavailable",
        acoustic_endpoint=True,
        vad_active=False,
        early_endpoint_allowed=True,
        trailing_silence_sec=0.8,
        boolean_value=True,
    )
    engine._diagnostic_observe(
        DiagnosticStage.ENDPOINT_COMMITTED,
        at=12.0,
        bundle_span=frame_span,
        acoustic=acoustic,
        revision=1,
        reason="asr",
        basis="acoustic",
        completion_state="unavailable",
    )
    engine._diagnostic_observe(
        DiagnosticStage.ASR_STREAMING_FINAL,
        at=12.0,
        bundle_span=frame_span,
        acoustic=acoustic,
        revision=1,
        outcome="nonempty",
    )


def test_final_selection_and_dispatch_share_audio_coordinate_without_text(
    tmp_path,
) -> None:
    engine = SherpaOnnxEngine(
        SherpaConfig(
            record_pre_dsp_reference=True,
            record_playback_reference=True,
        )
    )
    record_path = tmp_path / "run.wav"
    engine.set_record_path(str(record_path))
    engine._open_recorders()
    count = 160
    first_span = engine._write_recording_frame(
        np.full(count, -0.1, dtype="float32"),
        np.full(count, -0.2, dtype="float32"),
        asr_input_samples=np.full(count, -0.3, dtype="float32"),
        captured_reference=np.zeros(count, dtype="float32"),
        captured=_captured(count),
    )
    frame_span = engine._write_recording_frame(
        np.full(count, 0.1, dtype="float32"),
        np.full(count, 0.2, dtype="float32"),
        asr_input_samples=np.full(count, 0.3, dtype="float32"),
        captured_reference=np.zeros(count, dtype="float32"),
        captured=CapturedBlock(
            pcm=np.zeros(count, dtype="float32"),
            sample_rate_hz=16000,
            sequence=2,
            captured_started_at=10.01,
            captured_at=10.02,
            capture_epoch=2,
            source_generation=3,
            capture_generation=4,
            source_sample_start=count,
            source_sample_end=2 * count,
        ),
    )
    assert first_span is not None and frame_span is not None

    acoustic = _acoustic(2 * count)
    engine._bind_diagnostic_span(acoustic, frame_span)
    _record_endpoint(engine, acoustic, frame_span)
    engine._final_above_floor = lambda _samples: True
    finals: list[str] = []
    engine._cb = EngineCallbacks(on_final=finals.append)
    canary = "PRIVATE CANARY FINAL"

    engine._finalize_and_dispatch(
        np.ones(2 * count, dtype="float32"),
        canary,
        time.perf_counter(),
        acoustic=acoustic,
        revision=1,
    )
    engine._close_recorders(log_completed=True)

    assert finals == ["Private canary final"]
    manifest_path = tmp_path / "run.diagnostic.json"
    timeline_path = tmp_path / "run.timeline.jsonl"
    assert validate_manifest(manifest_path)
    records = [
        json.loads(line)
        for line in timeline_path.read_text(encoding="utf-8").splitlines()
    ]
    observations = [row for row in records if row["kind"] == "observation"]
    assert [row["stage"] for row in observations] == [
        DiagnosticStage.ENDPOINT_EVALUATED.value,
        DiagnosticStage.ENDPOINT_COMMITTED.value,
        DiagnosticStage.ASR_STREAMING_FINAL.value,
        DiagnosticStage.FINALIZER_STARTED.value,
        DiagnosticStage.FINAL_SELECTION.value,
        DiagnosticStage.FINAL_DISPATCHED.value,
    ]
    assert all(row["utterance_id"] == "u1" for row in observations)
    # Only the endpoint frame is claimed. A word-cut or confirmed-barge replay
    # can make selected turn PCM non-contiguous, so sample count is not a range.
    assert all(row["bundle_sample_start"] == count for row in observations)
    assert all(row["bundle_sample_end"] == 2 * count for row in observations)
    assert observations[4]["selected_source"] == "streaming"
    assert observations[5]["outcome"] == "callback_returned"
    persisted = timeline_path.read_text(encoding="utf-8") + manifest_path.read_text(
        encoding="utf-8"
    )
    assert canary not in persisted
    assert "Private canary final" not in persisted


def test_recording_mode_preserves_established_final_transcribe_override(tmp_path) -> None:
    engine = SherpaOnnxEngine(
        SherpaConfig(
            record_pre_dsp_reference=True,
            record_playback_reference=True,
        )
    )
    engine.set_record_path(str(tmp_path / "override.wav"))
    engine._open_recorders()
    count = 160
    frame_span = engine._write_recording_frame(
        np.zeros(count, dtype="float32"),
        np.zeros(count, dtype="float32"),
        asr_input_samples=np.zeros(count, dtype="float32"),
        captured_reference=np.zeros(count, dtype="float32"),
        captured=_captured(count),
    )
    acoustic = _acoustic(count)
    engine._bind_diagnostic_span(acoustic, frame_span)
    _record_endpoint(engine, acoustic, frame_span)
    calls: list[str] = []

    def override(_seg, raw_final: str, **_kwargs) -> str:
        calls.append(raw_final)
        return "Override result"

    engine._final_transcribe = override
    engine._final_above_floor = lambda _samples: True
    finals: list[str] = []
    engine._cb = EngineCallbacks(on_final=finals.append)
    engine._finalize_and_dispatch(
        np.ones(count, dtype="float32"),
        "private raw input",
        time.perf_counter(),
        acoustic=acoustic,
        revision=1,
    )
    engine._close_recorders(log_completed=True)

    assert calls == ["private raw input"]
    assert finals == ["Override result"]
    records = [
        json.loads(line)
        for line in (tmp_path / "override.timeline.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    selection = next(
        row
        for row in records
        if row.get("stage") == DiagnosticStage.FINAL_SELECTION.value
    )
    assert selection["selected_source"] == "established_override"
    assert validate_manifest(tmp_path / "override.diagnostic.json")
