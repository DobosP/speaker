from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import stat
from threading import Event
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from always_on_agent.acoustic import (
    AcousticLineage,
    AcousticSource,
    AcousticSpan,
    EndpointReason,
)
from always_on_agent.logical_turn import LogicalTurnBoundary
from always_on_agent.logical_turn_shadow import (
    LogicalTurnCompositionShadow,
    aggregate_logical_turn_terminal_lineage,
    aggregate_logical_turn_shadows,
    validate_logical_turn_terminal_lineage_report,
    validate_logical_turn_shadow_report,
)
from core.engine import (
    AcousticSignal,
    CommandDetection,
    EngineCallbacks,
    FinalTranscript,
    PartialTranscript,
    TranscriptAbort,
    TranscriptAbortReason,
)
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from core.realtime_media_stage import CaptureScope, RealtimeMediaStage
from tools.capture_replay import (
    FRAME_SAMPLES,
    SAMPLE_RATE_HZ,
    LoadedReplayCorpus,
    ReplayAssertion,
    ReplayCase,
    ReplayCorpusError,
    ReplayTrack,
    SampleInterval,
    SpeakerInterval,
    SpeakerRole,
    WordInterval,
)
from tools.capture_replay.metrics import (
    ReplayAcousticEvent,
    ReplayRunRecord,
    TimedFinal,
)
from tools import capture_replay_eval as evaluate


def test_evaluator_provenance_includes_streaming_decode_sources() -> None:
    assert "core/engines/_sherpa_streaming_decode.py" in (
        evaluate._EVALUATOR_SOURCE_FILES
    )
    assert "core/engines/_sherpa_streaming_decode_owner.py" in (
        evaluate._EVALUATOR_SOURCE_FILES
    )
    assert "core/endpointing.py" in evaluate._EVALUATOR_SOURCE_FILES
    assert "core/engines/_semantic_hold.py" in (evaluate._EVALUATOR_SOURCE_FILES)
    default_files = evaluate._evaluator_source_files()
    shadow_files = evaluate._evaluator_source_files(logical_turn_shadow=True)
    async_files = evaluate._evaluator_source_files(
        logical_turn_shadow=True,
        logical_turn_async_terminal_delivery=True,
    )
    assert "always_on_agent/logical_turn.py" not in default_files
    assert "always_on_agent/logical_turn.py" in shadow_files
    assert "always_on_agent/logical_turn_shadow.py" in shadow_files
    assert "core/contract.py" in shadow_files
    assert "core/contract.py" not in default_files
    assert len(shadow_files) == len(default_files) + 3
    assert "core/realtime_media_stage.py" not in shadow_files
    assert "core/realtime_media_stage.py" in async_files
    assert "tools/capture_replay/async_delivery.py" in async_files
    assert len(async_files) == len(shadow_files) + 2
    digest = evaluate._evaluator_source_digest()
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")
    assert evaluate._evaluator_source_digest(logical_turn_shadow=True) != digest
    with pytest.raises(evaluate.CaptureReplayEvaluationError):
        evaluate._evaluator_source_files(
            logical_turn_async_terminal_delivery=True,
        )


def test_artifact_metadata_binds_only_active_bpe_hotword_vocab(tmp_path) -> None:
    model = tmp_path / "encoder.onnx"
    vocab = tmp_path / "bpe.vocab"
    model.write_bytes(b"model")
    vocab.write_bytes(b"first")
    active = SherpaConfig(
        asr_encoder=str(model),
        asr_hotwords="VAULT",
        asr_modeling_unit="bpe",
        asr_bpe_vocab=str(vocab),
    )

    first = evaluate._artifact_metadata_digest(active)
    original_stat = vocab.stat()
    vocab.write_bytes(b"other")  # same byte length as ``first``
    os.utime(
        vocab,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    second = evaluate._artifact_metadata_digest(active)

    assert first != second
    stale = SherpaConfig(
        asr_encoder=str(model),
        asr_bpe_vocab=str(tmp_path / "missing.vocab"),
    )
    clean = SherpaConfig(asr_encoder=str(model))
    assert evaluate._artifact_metadata_digest(stale) == (
        evaluate._artifact_metadata_digest(clean)
    )
    for method, unit in (
        ("greedy_search", "bpe"),
        ("modified_beam_search", "cjkchar"),
    ):
        inert = SherpaConfig(
            asr_encoder=str(model),
            asr_hotwords="VAULT",
            asr_decoding_method=method,
            asr_modeling_unit=unit,
            asr_bpe_vocab=str(tmp_path / "missing.vocab"),
        )
        without_vocab = SherpaConfig(
            asr_encoder=str(model),
            asr_hotwords="VAULT",
            asr_decoding_method=method,
            asr_modeling_unit=unit,
        )
        assert evaluate._artifact_metadata_digest(inert) == (
            evaluate._artifact_metadata_digest(without_vocab)
        )


def test_artifact_metadata_content_binds_only_active_prosody_model(tmp_path) -> None:
    encoder = tmp_path / "encoder.onnx"
    model = tmp_path / "smart-turn.onnx"
    encoder.write_bytes(b"encoder")
    model.write_bytes(b"first")
    active = SherpaConfig(
        asr_encoder=str(encoder),
        endpoint_enabled=True,
        endpoint_detector="prosody",
        endpoint_prosody_model=str(model),
    )

    first = evaluate._artifact_metadata_digest(active)
    original = model.stat()
    model.write_bytes(b"other")
    os.utime(model, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert evaluate._artifact_metadata_digest(active) != first

    for inert in (
        replace(active, endpoint_enabled=False),
        replace(active, endpoint_detector="lexical"),
    ):
        assert evaluate._artifact_metadata_digest(inert) == (
            evaluate._artifact_metadata_digest(
                replace(inert, endpoint_prosody_model="")
            )
        )


def test_multi_case_replay_reuses_one_synchronous_session_then_closes_it(
    tmp_path,
) -> None:
    class _Stream:
        def accept_waveform(self, _sample_rate, _samples) -> None:
            pass

    class _Recognizer:
        def __init__(self) -> None:
            self.streams_created = 0

        def create_stream(self):
            self.streams_created += 1
            return _Stream()

        def is_ready(self, _stream) -> bool:
            return False

        def decode_stream(self, _stream) -> None:  # pragma: no cover - never ready
            pass

        def get_result(self, _stream) -> str:
            return ""

        def is_endpoint(self, _stream) -> bool:
            return False

        def reset(self, _stream) -> None:
            pass

    values = np.zeros(FRAME_SAMPLES, dtype="float32")
    replay_case = ReplayCase(
        case_id="owner-lifecycle",
        sample_rate_hz=SAMPLE_RATE_HZ,
        frame_samples=FRAME_SAMPLES,
        tracks=MappingProxyType({"mic": _track(tmp_path, values)}),
        speech_intervals=(),
        speaker_intervals=(),
        word_intervals=(),
        assertion=ReplayAssertion.SILENCE,
        expected_text="",
        commands=(),
        tags=("owner-lifecycle",),
        aec_delay_samples=0,
    )
    engine = SherpaOnnxEngine(
        SherpaConfig(
            input_calibrate=False,
            endpoint_enabled=False,
            final_speech_evidence_enabled=False,
        )
    )
    recognizer = _Recognizer()
    engine._recognizer = recognizer

    first = evaluate._execute_case(engine, replay_case, case_index=0, repeat=0)
    session = engine._streaming_decode_session
    assert session is not None and not session.closed
    second = evaluate._execute_case(engine, replay_case, case_index=1, repeat=0)
    assert engine._streaming_decode_session is session
    assert recognizer.streams_created == 2

    session.close()

    assert first.case_index == 0
    assert second.case_index == 1
    assert evaluate._attest_streaming_decode_execution(
        engine,
        expected_capture_runs=2,
    ) == "single-session-synchronous-replay"

    engine._streaming_decode_owner = object()
    with pytest.raises(evaluate.CaptureReplayEvaluationError):
        evaluate._attest_streaming_decode_execution(
            engine,
            expected_capture_runs=2,
        )


def _track(tmp_path: Path, values: np.ndarray) -> ReplayTrack:
    raw = np.asarray(values, dtype="<f4").tobytes()
    return ReplayTrack(
        role="mic",
        path=tmp_path / "private-mic.f32le",
        sha256=hashlib.sha256(raw).hexdigest(),
        samples=len(raw) // 4,
        pcm_bytes=raw,
    )


def _case(tmp_path: Path, *, calibration_frames: int = 15) -> ReplayCase:
    rng = np.random.default_rng(7)
    calibration = rng.normal(
        0.0,
        0.002,
        calibration_frames * FRAME_SAMPLES,
    ).astype("float32")
    quiet = np.zeros(5 * FRAME_SAMPLES, dtype="float32")
    speech = (
        0.1
        * np.sin(
            np.arange(5 * FRAME_SAMPLES, dtype="float32")
            * np.float32(2.0 * np.pi * 180.0 / SAMPLE_RATE_HZ)
        )
    ).astype("float32")
    tail = np.zeros(20 * FRAME_SAMPLES, dtype="float32")
    values = np.concatenate((calibration, quiet, speech, tail))
    track = _track(tmp_path, values)
    speech_start = (calibration_frames + 5) * FRAME_SAMPLES
    speech_end = speech_start + 5 * FRAME_SAMPLES
    return ReplayCase(
        case_id="private-case",
        sample_rate_hz=SAMPLE_RATE_HZ,
        frame_samples=FRAME_SAMPLES,
        tracks=MappingProxyType({"mic": track}),
        speech_intervals=(SampleInterval(speech_start, speech_end),),
        speaker_intervals=(
            SpeakerInterval(
                speaker_id="speaker-a",
                role=SpeakerRole.UNKNOWN,
                start_sample=speech_start,
                end_sample=speech_end,
            ),
        ),
        word_intervals=(
            WordInterval(
                speaker_id="speaker-a",
                text="private phrase",
                start_sample=speech_start,
                end_sample=speech_end,
            ),
        ),
        assertion=ReplayAssertion.TRANSCRIPT,
        expected_text="private phrase",
        commands=(),
        tags=("private",),
        aec_delay_samples=0,
    )


def _lineage(*, utterance_id: str = "u1") -> AcousticLineage:
    return AcousticLineage.single(
        AcousticSpan(
            stream_id="stream-1",
            utterance_id=utterance_id,
            capture_epoch=1,
            capture_generation=1,
            source=AcousticSource.LIVE_CAPTURE,
            speech_start_at=1.0,
            speech_end_at=1.8,
            endpoint_committed_at=2.0,
            emitted_at=2.2,
            sample_rate_hz=SAMPLE_RATE_HZ,
            owned_sample_count=FRAME_SAMPLES,
            endpoint_reason=EndpointReason.ASR,
        )
    )


def test_replay_config_preserves_capture_models_but_omits_output_and_identity(
    monkeypatch,
):
    configured = SherpaConfig(
        asr_encoder="/private/encoder.onnx",
        asr_decoder="/private/decoder.onnx",
        asr_joiner="/private/joiner.onnx",
        asr_tokens="/private/tokens.txt",
        vad_model="/private/vad.onnx",
        denoise_enabled=True,
        denoise_model="/private/denoise.onnx",
        asr_final_backend="nemo_transducer",
        asr_final_model="/private/final.onnx",
        asr_final_tokens="/private/final-tokens.txt",
        tts_model="/private/tts.onnx",
        tts_tokens="/private/tts-tokens.txt",
        kws_encoder="/private/kws-encoder.onnx",
        speaker_embedding_model="/private/speaker.onnx",
        speaker_gate_input=True,
        asr_final_async=True,
        provider="cpu",
        asr_num_threads=4,
    )

    monkeypatch.setattr(evaluate, "_require_provider_available", lambda _provider: None)
    replay = evaluate._replay_config(
        configured,
        provider="cuda",
        asr_threads=1,
    )

    assert replay.asr_encoder == configured.asr_encoder
    assert replay.vad_model == configured.vad_model
    assert replay.denoise_model == configured.denoise_model
    assert replay.asr_final_model == configured.asr_final_model
    assert replay.provider == "cuda"
    assert replay.asr_num_threads == 1
    assert replay.asr_final_async is False
    assert replay.tts_model == ""
    assert replay.kws_encoder == ""
    assert replay.speaker_embedding_model == ""
    assert replay.speaker_gate_input is False
    assert configured.tts_model == "/private/tts.onnx"
    assert configured.asr_final_async is True

    async_replay = evaluate._replay_config(
        configured,
        provider="cpu",
        asr_threads=1,
        async_final_delivery=True,
    )
    assert async_replay.asr_final_async is True
    with pytest.raises(evaluate.CaptureReplayEvaluationError):
        evaluate._replay_config(
            configured,
            provider="cpu",
            asr_threads=1,
            async_final_delivery=1,  # type: ignore[arg-type]
        )


def test_cuda_override_fails_instead_of_silently_falling_back_to_cpu(
    monkeypatch,
):
    monkeypatch.setitem(
        __import__("sys").modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
        ),
    )

    with pytest.raises(evaluate.CaptureReplayEvaluationError):
        evaluate._replay_config(
            SherpaConfig(provider="cpu"),
            provider="cuda",
            asr_threads=1,
        )


def test_quiet_model_output_suppresses_python_and_native_fds(capfd):
    private = "private native transcript"

    with evaluate._quiet_model_output():
        print(private)
        os.write(1, private.encode("utf-8"))
        os.write(2, private.encode("utf-8"))

    captured = capfd.readouterr()
    assert private not in captured.out
    assert private not in captured.err


def test_verifier_probe_attests_completed_decode_without_exposing_text():
    private = "private verifier transcript"
    delegate = SimpleNamespace(
        transcribe=lambda *_args, **_kwargs: SimpleNamespace(text=private)
    )
    engine = SimpleNamespace(_final_verifier=delegate)
    probe = evaluate._install_verifier_probe(
        engine,
        SherpaConfig(asr_final_verifier_backend="faster_whisper"),
    )

    assert probe is not None
    assert engine._final_verifier.transcribe([], 16_000).text == private
    report = evaluate._attest_verifier_execution(
        engine,
        probe,
        text_expected=True,
    )

    assert report == {
        "requested": True,
        "available_after_run": True,
        "attempted_decodes": 1,
        "completed_decodes": 1,
        "decode_errors": 0,
        "nonempty_decodes": 1,
    }
    assert private not in json.dumps(report)


def test_requested_verifier_must_survive_warm_and_complete_a_decode():
    with pytest.raises(evaluate.CaptureReplayEvaluationError):
        evaluate._install_verifier_probe(
            SimpleNamespace(_final_verifier=None),
            SherpaConfig(asr_final_verifier_backend="faster_whisper"),
        )

    delegate = SimpleNamespace(
        transcribe=lambda *_args, **_kwargs: SimpleNamespace(text="private")
    )
    engine = SimpleNamespace(_final_verifier=delegate)
    probe = evaluate._install_verifier_probe(
        engine,
        SherpaConfig(asr_final_verifier_backend="faster_whisper"),
    )
    with pytest.raises(evaluate.CaptureReplayEvaluationError):
        evaluate._attest_verifier_execution(
            engine,
            probe,
            text_expected=True,
        )


def test_calibration_prefix_is_consumed_before_production_capture_replay(
    tmp_path,
    monkeypatch,
):
    case = _case(tmp_path)
    engine = SherpaOnnxEngine(
        SherpaConfig(
            input_calibrate=True,
            input_calibrate_sec=1.5,
            input_agc=True,
            final_speech_evidence_enabled=True,
        )
    )
    slept: list[float] = []
    monkeypatch.setattr(evaluate.time, "sleep", lambda value: slept.append(value))

    trimmed = evaluate._calibrate_case(engine, case)

    removed = 15 * FRAME_SAMPLES
    assert slept == [1.5]
    assert trimmed.samples == case.samples - removed
    assert trimmed.speech_intervals == (
        SampleInterval(
            case.speech_intervals[0].start_sample - removed,
            case.speech_intervals[0].end_sample - removed,
        ),
    )
    assert trimmed.word_intervals[0].text == "private phrase"
    assert engine._last_calibration is not None
    assert engine._input_agc is not None
    assert engine._input_agc.noise_floor_rms > 0.0
    assert hashlib.sha256(trimmed.mic.pcm_bytes).hexdigest() == trimmed.mic.sha256


def test_case_failure_cleans_private_callbacks_and_capture_state(
    tmp_path,
    monkeypatch,
):
    engine = SherpaOnnxEngine(SherpaConfig(input_calibrate=False))

    def fail_capture(*, close_decode_session: bool = True) -> None:
        assert close_decode_session is False
        raise RuntimeError("private capture failure")

    monkeypatch.setattr(engine, "_capture_loop", fail_capture)

    with pytest.raises(RuntimeError):
        evaluate._execute_case(
            engine,
            _case(tmp_path),
            case_index=0,
            repeat=0,
        )

    assert not engine._running.is_set()
    assert not engine._speaking.is_set()
    assert engine._capture_control_lane is None
    assert engine._capture_media_session is None
    assert engine._stream_in is None
    assert engine._cb == EngineCallbacks()


def test_calibration_rejects_ground_truth_inside_startup_window(tmp_path):
    case = _case(tmp_path)
    contaminated = replace_case = ReplayCase(
        case_id=case.case_id,
        sample_rate_hz=case.sample_rate_hz,
        frame_samples=case.frame_samples,
        tracks=case.tracks,
        speech_intervals=(SampleInterval(0, FRAME_SAMPLES),),
        speaker_intervals=(
            SpeakerInterval(
                speaker_id="speaker-a",
                role=SpeakerRole.UNKNOWN,
                start_sample=0,
                end_sample=FRAME_SAMPLES,
            ),
        ),
        word_intervals=(
            WordInterval(
                speaker_id="speaker-a",
                text="private",
                start_sample=0,
                end_sample=FRAME_SAMPLES,
            ),
        ),
        assertion=case.assertion,
        expected_text="private",
        commands=(),
        tags=case.tags,
        aec_delay_samples=0,
    )
    assert contaminated is replace_case
    engine = SherpaOnnxEngine(
        SherpaConfig(input_calibrate=True, input_calibrate_sec=1.5)
    )

    with pytest.raises(evaluate.CaptureReplayEvaluationError):
        evaluate._calibrate_case(engine, contaminated)


def test_typed_collector_retains_only_closed_events_and_lineage():
    stopped: list[bool] = []
    collector = evaluate._CaseCollector(lambda: stopped.append(True))
    lineage = _lineage()

    collector.on_partial(
        PartialTranscript("private partial", acoustic=lineage, revision=0)
    )
    collector.on_final(
        FinalTranscript("private final", acoustic=lineage, revision=1)
    )
    collector.on_command(
        CommandDetection("stop speaking", acoustic=lineage, revision=2)
    )
    collector.on_barge(
        AcousticSignal(acoustic=lineage, revision=3, detected_at=2.1)
    )
    collector.on_abort(
        TranscriptAbort(
            acoustic=lineage,
            revision=4,
            reason=TranscriptAbortReason.INPUT_REJECTED,
        )
    )
    record = collector.record(
        case_index=0,
        repeat=0,
        wall_seconds=2.5,
        peak_rss_mb=100.0,
    )

    assert stopped == []
    assert record.partials[0].utterance_id == "u1"
    assert record.finals[0].endpoint_committed_at == 2.0
    assert record.commands == ("stop speaking",)
    assert record.acoustic_events == (ReplayAcousticEvent.BARGE_IN,)
    assert record.abort_reasons == (TranscriptAbortReason.INPUT_REJECTED,)


def test_typed_collector_pairs_replay_abort_and_stop_without_text_rows() -> None:
    lineage = _lineage()
    transcript_shadow = LogicalTurnCompositionShadow()
    assert transcript_shadow.observe_native_final(
        native_epoch=1,
        native_sequence=0,
        text="PRIVATE PREFIX",
        held=True,
    )
    transcript_collector = evaluate._CaseCollector(
        lambda: None,
        transcript_shadow,
    )
    transcript_collector.on_abort(
        TranscriptAbort(
            acoustic=lineage,
            revision=1,
            reason=TranscriptAbortReason.INPUT_REJECTED,
        )
    )
    transcript_shadow.close()
    transcript_snapshot = transcript_shadow.snapshot()
    assert transcript_snapshot.abort_transcript == 1
    assert transcript_snapshot.missing_legacy_terminals == 0

    stop_shadow = LogicalTurnCompositionShadow()
    assert stop_shadow.observe_native_final(
        native_epoch=1,
        native_sequence=0,
        text="PRIVATE PREFIX",
        held=True,
    )
    stop_collector = evaluate._CaseCollector(lambda: None, stop_shadow)
    stop_collector.on_command(
        CommandDetection(
            "stop speaking",
            acoustic=lineage,
            revision=1,
        )
    )
    stop_shadow.close()
    stop_snapshot = stop_shadow.snapshot()
    assert stop_snapshot.abort_stop == 1
    assert stop_snapshot.abort_control == 0
    assert stop_snapshot.missing_legacy_terminals == 0
    assert "PRIVATE PREFIX" not in repr(stop_snapshot)


def test_post_composition_rejection_cannot_retroactively_abort_commit() -> None:
    shadow = LogicalTurnCompositionShadow()
    assert shadow.observe_native_final(
        native_epoch=1,
        native_sequence=0,
        text="PRIVATE PREFIX",
        held=True,
    )
    assert shadow.observe_native_final(
        native_epoch=2,
        native_sequence=0,
        text="PRIVATE FINAL",
        held=False,
    )
    assert shadow.compare_legacy_composition(
        boundary=LogicalTurnBoundary.NATIVE_FINAL,
        legacy_text="PRIVATE PREFIX PRIVATE FINAL",
    )
    collector = evaluate._CaseCollector(lambda: None, shadow)

    collector.on_abort(
        TranscriptAbort(
            acoustic=_lineage(),
            revision=1,
            reason=TranscriptAbortReason.INPUT_REJECTED,
        )
    )
    shadow.close()

    snapshot = shadow.snapshot()
    assert snapshot.committed == snapshot.composition_matches == 1
    assert snapshot.aborted == snapshot.abort_transcript == 0
    assert snapshot.idle_aborts == 1


def test_typed_collector_correlates_selected_and_abort_by_bound_lineage() -> None:
    shadow = LogicalTurnCompositionShadow(terminal_lineage=True)
    collector = evaluate._CaseCollector(lambda: None, shadow)
    selected_lineage = _lineage()
    abort_lineage = AcousticLineage.single(
        replace(
            selected_lineage.spans[0],
            utterance_id="u2",
        )
    )

    for text, lineage, revision in (
        ("PRIVATE RAW SELECTED", selected_lineage, 1),
        ("PRIVATE RAW ABORT", abort_lineage, 2),
    ):
        assert shadow.observe_native_final(
            native_epoch=1,
            native_sequence=0,
            text=text,
            held=False,
        )
        assert shadow.compare_legacy_composition(
            boundary=LogicalTurnBoundary.NATIVE_FINAL,
            legacy_text=text,
        )
        token = shadow.claim_terminal_token()
        assert token is not None
        assert shadow.bind_terminal_lineage(token, lineage, revision)

    selected_delivery = AcousticLineage.single(
        replace(selected_lineage.spans[0], emitted_at=2.1)
    )
    collector.on_final(
        FinalTranscript(
            "DIFFERENT SELECTED WORDS",
            acoustic=selected_delivery,
            revision=1,
        )
    )
    collector.on_abort(
        TranscriptAbort(
            acoustic=abort_lineage,
            revision=2,
            reason=TranscriptAbortReason.INPUT_REJECTED,
        )
    )
    shadow.close()
    report = aggregate_logical_turn_terminal_lineage(
        (shadow.terminal_lineage_snapshot(),)
    )

    assert report["outcomes"]["selected_finals"] == 1
    assert report["outcomes"]["typed_aborts"] == 1
    assert report["lineage"]["matched_selected_finals"] == 1
    assert report["lineage"]["matched_typed_aborts"] == 1
    assert report["lineage"]["exact"] is True
    assert shadow.snapshot().idle_aborts == 1
    serialized = json.dumps(report, sort_keys=True)
    assert "PRIVATE RAW" not in serialized
    assert "DIFFERENT SELECTED WORDS" not in serialized


@pytest.mark.parametrize(
    "mutation",
    (
        "raw",
        "raw_bool",
        "selected",
        "selected_bool",
        "abort_total",
        "abort_reason",
        "abort_reason_bool",
        "abort_reason_key",
    ),
)
def test_capture_terminal_lineage_cross_section_mismatch_fails_closed(
    mutation: str,
) -> None:
    shadow = LogicalTurnCompositionShadow(terminal_lineage=True)
    lineage = _lineage()
    assert shadow.observe_native_final(
        native_epoch=1,
        native_sequence=0,
        text="PRIVATE RAW",
        held=False,
    )
    assert shadow.compare_legacy_composition(
        boundary=LogicalTurnBoundary.NATIVE_FINAL,
        legacy_text="PRIVATE RAW",
    )
    token = shadow.claim_terminal_token()
    assert token is not None
    assert shadow.bind_terminal_lineage(token, lineage, 1)
    assert shadow.observe_selected_final(lineage, 1)
    shadow.close()
    raw = aggregate_logical_turn_shadows((shadow.snapshot(),))
    terminal = aggregate_logical_turn_terminal_lineage(
        (shadow.terminal_lineage_snapshot(),)
    )
    metrics = {
        "streaming": {
            "final_events": 1,
            "abort_reasons": {},
        }
    }
    evaluate._validate_terminal_lineage_against_metrics(
        raw_shadow=raw,
        terminal_lineage=terminal,
        metrics=metrics,
    )

    raw = json.loads(json.dumps(raw))
    terminal = json.loads(json.dumps(terminal))
    metrics = json.loads(json.dumps(metrics))
    if mutation == "raw":
        raw["parity"]["paired_terminals"] = 0
    elif mutation == "raw_bool":
        raw["parity"]["paired_terminals"] = True
    elif mutation == "selected":
        metrics["streaming"]["final_events"] = 0
    elif mutation == "selected_bool":
        metrics["streaming"]["final_events"] = True
    elif mutation == "abort_total":
        metrics["streaming"]["abort_reasons"] = {"input_rejected": 1}
    elif mutation == "abort_reason":
        terminal["outcomes"]["abort_reason_counts"]["abandoned"] = 1
    elif mutation == "abort_reason_bool":
        metrics["streaming"]["abort_reasons"] = {"input_rejected": True}
    else:
        metrics["streaming"]["abort_reasons"] = {"not_a_reason": 0}
    with pytest.raises(evaluate.CaptureReplayEvaluationError):
        evaluate._validate_terminal_lineage_against_metrics(
            raw_shadow=raw,
            terminal_lineage=terminal,
            metrics=metrics,
        )


@pytest.mark.parametrize("terminal_kind", ("selected", "abort"))
def test_terminal_observer_exception_cannot_change_collector_record(
    terminal_kind: str,
    monkeypatch,
) -> None:
    shadow = LogicalTurnCompositionShadow(terminal_lineage=True)
    lineage = _lineage()
    assert shadow.observe_native_final(
        native_epoch=1,
        native_sequence=0,
        text="PRIVATE RAW",
        held=False,
    )
    assert shadow.compare_legacy_composition(
        boundary=LogicalTurnBoundary.NATIVE_FINAL,
        legacy_text="PRIVATE RAW",
    )
    token = shadow.claim_terminal_token()
    assert token is not None
    assert shadow.bind_terminal_lineage(token, lineage, 1)

    def fail(*_args, **_kwargs):
        raise RuntimeError("PRIVATE TERMINAL FAILURE")

    collector = evaluate._CaseCollector(lambda: None, shadow)
    if terminal_kind == "selected":
        monkeypatch.setattr(shadow, "observe_selected_final", fail)
        collector.on_final(
            FinalTranscript(
                "PRIVATE SELECTED",
                acoustic=lineage,
                revision=1,
            )
        )
    else:
        monkeypatch.setattr(shadow, "observe_typed_abort", fail)
        collector.on_abort(
            TranscriptAbort(
                acoustic=lineage,
                revision=1,
                reason=TranscriptAbortReason.INPUT_REJECTED,
            )
        )
    record = collector.record(
        case_index=0,
        repeat=0,
        wall_seconds=0.1,
        peak_rss_mb=None,
    )
    shadow.close()
    report = aggregate_logical_turn_terminal_lineage(
        (shadow.terminal_lineage_snapshot(),)
    )

    assert len(record.finals) == int(terminal_kind == "selected")
    assert len(record.abort_reasons) == int(terminal_kind == "abort")
    assert report["health"]["internal_errors"] == 1
    assert report["lineage"]["missing_downstream_terminals"] == 1
    assert report["lineage"]["exact"] is False
    assert "PRIVATE TERMINAL FAILURE" not in json.dumps(report, sort_keys=True)


def test_parser_keeps_shadow_strictly_opt_in() -> None:
    base = [
        "--corpus",
        "private-corpus.json",
        "--report",
        "private-report.json",
    ]
    parsed = evaluate._parser().parse_args(base)
    assert parsed.logical_turn_shadow is False
    assert parsed.logical_turn_terminal_lineage is False
    assert parsed.logical_turn_async_terminal_delivery is False
    assert (
        evaluate._parser()
        .parse_args([*base, "--logical-turn-shadow"])
        .logical_turn_shadow
        is True
    )
    terminal = evaluate._parser().parse_args(
        [
            *base,
            "--logical-turn-shadow",
            "--logical-turn-terminal-lineage",
        ]
    )
    assert terminal.logical_turn_shadow is True
    assert terminal.logical_turn_terminal_lineage is True
    assert terminal.logical_turn_async_terminal_delivery is False
    async_delivery = evaluate._parser().parse_args(
        [
            *base,
            "--logical-turn-shadow",
            "--logical-turn-terminal-lineage",
            "--logical-turn-async-terminal-delivery",
        ]
    )
    assert async_delivery.logical_turn_shadow is True
    assert async_delivery.logical_turn_terminal_lineage is True
    assert async_delivery.logical_turn_async_terminal_delivery is True


def test_evaluator_default_shape_and_opt_in_shadow_schema(
    tmp_path,
    monkeypatch,
) -> None:
    replay_case = _case(tmp_path)
    corpus = LoadedReplayCorpus(
        path=tmp_path / "private-corpus.json",
        digest="c" * 64,
        schema_version=1,
        purpose="private purpose",
        sample_rate_hz=SAMPLE_RATE_HZ,
        frame_samples=FRAME_SAMPLES,
        cases=(replay_case,),
        audio_bytes=len(replay_case.mic.pcm_bytes),
    )

    class FakeEngine:
        def __init__(self, config):
            self.config = config
            self._recognizer = None
            self._final_recognizer = None
            self._final_verifier = None
            self._streaming_decode_session = None

        def _build(self) -> None:
            self._recognizer = object()
            if self.config.asr_final_backend:
                self._final_recognizer = object()

        def warm(self) -> None:
            return None

    observed_shadow_flags: list[tuple[bool, bool]] = []

    def execute_case(_engine, _case_value, **kwargs):
        shadow = kwargs.get("logical_turn_shadow")
        observed_shadow_flags.append(
            (
                shadow is not None,
                bool(shadow and shadow.terminal_lineage_enabled),
            )
        )
        if shadow is not None:
            assert shadow.observe_native_final(
                native_epoch=1,
                native_sequence=0,
                text="PRIVATE",
                held=True,
            )
            assert shadow.observe_native_final(
                native_epoch=2,
                native_sequence=0,
                text="NATIVE",
                held=False,
            )
            assert shadow.compare_legacy_composition(
                boundary=LogicalTurnBoundary.NATIVE_FINAL,
                legacy_text="PRIVATE NATIVE",
            )
            if shadow.terminal_lineage_enabled:
                token = shadow.claim_terminal_token()
                assert token is not None
                lineage = _lineage()
                assert shadow.bind_terminal_lineage(token, lineage, 1)
                assert shadow.observe_selected_final(lineage, 1)
        return ReplayRunRecord(
            case_index=kwargs["case_index"],
            repeat=kwargs["repeat"],
            finals=(
                TimedFinal(
                    "DIFFERENT SELECTED WORDS",
                    emitted_at=2.1,
                    speech_start_at=0.5,
                    speech_end_at=1.5,
                    endpoint_committed_at=2.0,
                    utterance_id="u1",
                ),
            ),
            wall_seconds=0.1,
        )

    monkeypatch.setattr(evaluate, "verify_corpus_snapshot", lambda _value: None)
    monkeypatch.setattr(
        evaluate,
        "_replay_config",
        lambda configured, **_kwargs: configured,
    )
    monkeypatch.setattr(
        evaluate,
        "_artifact_metadata_digest",
        lambda _configured: "a" * 64,
    )
    monkeypatch.setattr(evaluate, "SherpaOnnxEngine", FakeEngine)
    monkeypatch.setattr(evaluate, "_quiet_model_output", nullcontext)
    monkeypatch.setattr(
        evaluate,
        "_install_verifier_probe",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(evaluate, "_execute_case", execute_case)
    monkeypatch.setattr(
        evaluate,
        "aggregate_metrics",
        lambda *_args, **_kwargs: {
            "evaluations": 1,
            "coverage": {"complete": True},
            "streaming": {
                "final_events": 1,
                "abort_reasons": {},
            },
        },
    )
    monkeypatch.setattr(
        evaluate,
        "_attest_verifier_execution",
        lambda *_args, **_kwargs: "not-configured",
    )
    monkeypatch.setattr(
        evaluate,
        "_attest_streaming_decode_execution",
        lambda *_args, **_kwargs: "synchronous-replay-test",
    )
    configured = SherpaConfig()

    default = evaluate.evaluate_capture_replay(corpus, configured)
    shadowed = evaluate.evaluate_capture_replay(
        corpus,
        configured,
        logical_turn_shadow=True,
    )
    terminal_lineage = evaluate.evaluate_capture_replay(
        corpus,
        configured,
        logical_turn_shadow=True,
        logical_turn_terminal_lineage=True,
    )
    with pytest.raises(evaluate.CaptureReplayEvaluationError):
        evaluate.evaluate_capture_replay(
            corpus,
            configured,
            logical_turn_terminal_lineage=True,
        )

    assert set(default) == {
        "execution_complete",
        "quality_verdict",
        "schema_version",
        "corpus",
        "engine",
        "metrics",
    }
    assert default["schema_version"] == 1
    assert "logical_turn_shadow" not in default
    assert shadowed["schema_version"] == 2
    assert set(shadowed) == set(default) | {"logical_turn_shadow"}
    validate_logical_turn_shadow_report(shadowed["logical_turn_shadow"])
    assert shadowed["logical_turn_shadow"]["parity"]["exact"] is True
    assert terminal_lineage["schema_version"] == 3
    assert set(terminal_lineage) == set(shadowed) | {
        "logical_turn_terminal_lineage"
    }
    validate_logical_turn_terminal_lineage_report(
        terminal_lineage["logical_turn_terminal_lineage"]
    )
    assert terminal_lineage["logical_turn_terminal_lineage"]["lineage"][
        "exact"
    ] is True
    assert observed_shadow_flags == [
        (False, False),
        (True, False),
        (True, True),
    ]
    serialized = json.dumps(shadowed, sort_keys=True)
    assert "PRIVATE NATIVE" not in serialized
    assert shadowed["engine"]["evaluator_source_files"] == (
        default["engine"]["evaluator_source_files"] + 3
    )


def test_evaluator_async_mode_uses_schema_four_and_real_worker_harness(
    tmp_path,
    monkeypatch,
) -> None:
    replay_case = _case(tmp_path)
    corpus = LoadedReplayCorpus(
        path=tmp_path / "private-corpus.json",
        digest="c" * 64,
        schema_version=1,
        purpose="private purpose",
        sample_rate_hz=SAMPLE_RATE_HZ,
        frame_samples=FRAME_SAMPLES,
        cases=(replay_case,),
        audio_bytes=len(replay_case.mic.pcm_bytes),
    )
    lineage = _lineage()

    class FakeEngine:
        def __init__(self, config):
            self.config = config
            self._recognizer = None
            self._final_recognizer = None
            self._final_verifier = None
            self._final_stage = None
            self._streaming_decode_session = None
            self._running = Event()

        def _build(self) -> None:
            self._recognizer = object()
            self._final_recognizer = object()
            self._maybe_setup_async_final()

        def _maybe_setup_async_final(self) -> None:
            if self._final_stage is not None:
                self._final_stage.close()
            self._final_stage = RealtimeMediaStage(8)

        def _final_worker(self, stage, *, final_delivery_observer=None) -> None:
            assert final_delivery_observer is not None
            while self._running.is_set():
                item = stage.take(timeout=0.01)
                if item is None:
                    if stage.snapshot().closed:
                        return
                    continue
                work = item.payload
                final_delivery_observer.begin_worker_item(
                    item.sequence,
                    work.acoustic,
                    work.revision,
                )
                assert work.shadow.observe_selected_final(
                    work.acoustic,
                    work.revision,
                )
                final_delivery_observer.observe_selected(
                    FinalTranscript(
                        "PRIVATE SELECTED WORDS",
                        acoustic=work.acoustic,
                        revision=work.revision,
                    )
                )
                succeeded = stage.finish(item)
                final_delivery_observer.finish_worker_item(
                    item.sequence,
                    succeeded,
                )

        def warm(self) -> None:
            return None

    def execute_case(engine, _case_value, **kwargs):
        shadow = kwargs["logical_turn_shadow"]
        observer = kwargs["async_final_delivery"]
        worker = kwargs["async_final_worker"]
        assert shadow.observe_native_final(
            native_epoch=1,
            native_sequence=0,
            text="PRIVATE RAW WORDS",
            held=False,
        )
        assert shadow.compare_legacy_composition(
            boundary=LogicalTurnBoundary.NATIVE_FINAL,
            legacy_text="PRIVATE RAW WORDS",
        )
        token = shadow.claim_terminal_token()
        assert token is not None
        assert shadow.bind_terminal_lineage(token, lineage, 1)
        engine._maybe_setup_async_final()
        stage = engine._final_stage
        assert stage is not None
        work = SimpleNamespace(acoustic=lineage, revision=1, shadow=shadow)
        assert stage.offer_nowait(CaptureScope(1, 1), work).accepted
        worker.drain(stage, observer, timeout=2.0)
        observer.close()
        return ReplayRunRecord(
            case_index=kwargs["case_index"],
            repeat=kwargs["repeat"],
            finals=(
                TimedFinal(
                    "PRIVATE SELECTED WORDS",
                    emitted_at=2.1,
                    speech_start_at=0.5,
                    speech_end_at=1.5,
                    endpoint_committed_at=2.0,
                    utterance_id="u1",
                ),
            ),
            wall_seconds=0.1,
        )

    monkeypatch.setattr(evaluate, "verify_corpus_snapshot", lambda _value: None)
    monkeypatch.setattr(
        evaluate,
        "_replay_config",
        lambda configured, **_kwargs: configured,
    )
    monkeypatch.setattr(
        evaluate,
        "_artifact_metadata_digest",
        lambda _configured: "a" * 64,
    )
    monkeypatch.setattr(evaluate, "SherpaOnnxEngine", FakeEngine)
    monkeypatch.setattr(evaluate, "_quiet_model_output", nullcontext)
    monkeypatch.setattr(
        evaluate,
        "_install_verifier_probe",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(evaluate, "_execute_case", execute_case)
    monkeypatch.setattr(
        evaluate,
        "aggregate_metrics",
        lambda *_args, **_kwargs: {
            "evaluations": 1,
            "coverage": {"complete": True},
            "streaming": {"final_events": 1, "abort_reasons": {}},
        },
    )
    monkeypatch.setattr(
        evaluate,
        "_attest_verifier_execution",
        lambda *_args, **_kwargs: "not-configured",
    )
    monkeypatch.setattr(
        evaluate,
        "_attest_streaming_decode_execution",
        lambda *_args, **_kwargs: "synchronous-replay-test",
    )

    report = evaluate.evaluate_capture_replay(
        corpus,
        SherpaConfig(
            asr_final_backend="sense_voice",
            asr_final_async=True,
        ),
        logical_turn_shadow=True,
        logical_turn_terminal_lineage=True,
        logical_turn_async_terminal_delivery=True,
    )

    assert report["schema_version"] == 4
    evidence = report["logical_turn_async_terminal_delivery"]
    assert evidence["exact"] is True
    assert evidence["callbacks"]["worker_selected_finals"] == 1
    assert report["engine"]["final_execution"] == (
        "post-capture-dedicated-worker-drain"
    )
    assert report["engine"]["evaluator_source_files"] == 32
    assert "PRIVATE" not in json.dumps(evidence, sort_keys=True)
    payload = (
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )
    retained = tmp_path / "async-report.json"
    evaluate._write_new_private(retained, payload)
    receipt = {
        "report_sha256": hashlib.sha256(payload).hexdigest(),
        "corpus_sha256": corpus.digest,
        "evaluations": 1,
        "coverage_complete": True,
    }
    evaluate._reopen_async_report(retained, receipt)
    forged_receipt = dict(receipt)
    forged_receipt["report_sha256"] = "0" * 64
    with pytest.raises(evaluate.CaptureReplayEvaluationError):
        evaluate._reopen_async_report(retained, forged_receipt)
    with pytest.raises(evaluate.CaptureReplayEvaluationError):
        evaluate.evaluate_capture_replay(
            corpus,
            SherpaConfig(),
            logical_turn_shadow=True,
            logical_turn_async_terminal_delivery=True,
        )


def test_report_writer_is_private_no_overwrite(tmp_path):
    report = tmp_path / "report.json"
    evaluate._write_new_private(report, b'{"execution_complete":true}\n')

    assert stat.S_IMODE(report.stat().st_mode) == 0o600
    assert report.read_bytes() == b'{"execution_complete":true}\n'
    with pytest.raises(evaluate.CaptureReplayEvaluationError):
        evaluate._write_new_private(report, b"different")
    assert report.read_bytes() == b'{"execution_complete":true}\n'


@pytest.mark.parametrize(
    ("mode_args", "schema_version", "expected_flags"),
    (
        ((), 1, (False, False, False)),
        (("--logical-turn-shadow",), 2, (True, False, False)),
        (
            (
                "--logical-turn-shadow",
                "--logical-turn-terminal-lineage",
            ),
            3,
            (True, True, False),
        ),
    ),
)
def test_cli_success_and_failure_emit_only_aggregate_receipts(
    tmp_path,
    monkeypatch,
    capsys,
    mode_args: tuple[str, ...],
    schema_version: int,
    expected_flags: tuple[bool, bool, bool],
):
    private_phrase = "private transcript must not escape"
    private_path = str(tmp_path / "private-corpus.json")
    corpus = LoadedReplayCorpus(
        path=Path(private_path),
        digest="a" * 64,
        schema_version=1,
        purpose=private_phrase,
        sample_rate_hz=SAMPLE_RATE_HZ,
        frame_samples=FRAME_SAMPLES,
        cases=(_case(tmp_path),),
        audio_bytes=100,
    )
    mode_sections: dict[str, object] = {}
    metrics: dict[str, object] = {
        "evaluations": 1,
        "coverage": {"complete": True},
        "streaming": {
            "final_events": 0,
            "abort_reasons": {},
        },
    }
    if schema_version >= 2:
        shadow = LogicalTurnCompositionShadow(
            terminal_lineage=schema_version == 3
        )
        lineage = _lineage()
        assert shadow.observe_native_final(
            native_epoch=1,
            native_sequence=0,
            text="PRIVATE RAW",
            held=False,
        )
        assert shadow.compare_legacy_composition(
            boundary=LogicalTurnBoundary.NATIVE_FINAL,
            legacy_text="PRIVATE RAW",
        )
        if schema_version == 3:
            token = shadow.claim_terminal_token()
            assert token is not None
            assert shadow.bind_terminal_lineage(token, lineage, 1)
            assert shadow.observe_selected_final(lineage, 1)
            metrics["streaming"]["final_events"] = 1
        shadow.close()
        mode_sections["logical_turn_shadow"] = aggregate_logical_turn_shadows(
            (shadow.snapshot(),)
        )
        if schema_version == 3:
            mode_sections["logical_turn_terminal_lineage"] = (
                aggregate_logical_turn_terminal_lineage(
                    (shadow.terminal_lineage_snapshot(),)
                )
            )
    report = {
        "execution_complete": True,
        "quality_verdict": "diagnostic_only",
        "schema_version": schema_version,
        "corpus": {"manifest_sha256": corpus.digest},
        "engine": {"provider": "cpu"},
        "metrics": metrics,
        **mode_sections,
    }
    monkeypatch.setattr(evaluate, "load_corpus", lambda _path: corpus)
    monkeypatch.setattr(evaluate, "load_config", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        evaluate,
        "apply_device_profile",
        lambda config, _device, **_kwargs: config,
    )
    shadow_flags: list[tuple[bool, bool, bool]] = []

    def evaluate_replay(*_args, **kwargs):
        shadow_flags.append(
            (
                kwargs["logical_turn_shadow"],
                kwargs["logical_turn_terminal_lineage"],
                kwargs["logical_turn_async_terminal_delivery"],
            )
        )
        return report

    monkeypatch.setattr(evaluate, "evaluate_capture_replay", evaluate_replay)
    output = tmp_path / "aggregate-report.json"

    code = evaluate.main(
        [
            "--corpus",
            private_path,
            *mode_args,
            "--report",
            str(output),
        ]
    )

    assert code == 0
    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    assert receipt == {
        "execution_complete": True,
        "quality_verdict": "diagnostic_only",
        "report_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "corpus_sha256": "a" * 64,
        "evaluations": 1,
        "coverage_complete": True,
        "schema_version": schema_version,
        "logical_turn_shadow": expected_flags[0],
        "logical_turn_terminal_lineage": expected_flags[1],
        "logical_turn_async_terminal_delivery": expected_flags[2],
    }
    assert captured.err == ""
    assert private_phrase not in captured.out
    assert private_path not in captured.out
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert shadow_flags == [expected_flags]

    monkeypatch.setattr(
        evaluate,
        "load_corpus",
        lambda _path: (_ for _ in ()).throw(ReplayCorpusError(private_phrase)),
    )
    code = evaluate.main(
        [
            "--corpus",
            private_path,
            "--report",
            str(tmp_path / "must-not-exist.json"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert json.loads(captured.out) == {
        "execution_complete": False,
        "error": "capture_replay_evaluation_unavailable",
    }
    assert captured.err == ""
    assert private_phrase not in captured.out
    assert private_path not in captured.out


def test_cli_rejects_report_schema_mismatch_before_publication(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    corpus = LoadedReplayCorpus(
        path=tmp_path / "private-corpus.json",
        digest="a" * 64,
        schema_version=1,
        purpose="PRIVATE PURPOSE",
        sample_rate_hz=SAMPLE_RATE_HZ,
        frame_samples=FRAME_SAMPLES,
        cases=(_case(tmp_path),),
        audio_bytes=100,
    )
    report = {
        "execution_complete": True,
        "quality_verdict": "diagnostic_only",
        "schema_version": 1,
        "corpus": {"manifest_sha256": corpus.digest},
        "engine": {"provider": "cpu"},
        "metrics": {
            "evaluations": 1,
            "coverage": {"complete": True},
        },
    }
    monkeypatch.setattr(evaluate, "load_corpus", lambda _path: corpus)
    monkeypatch.setattr(evaluate, "load_config", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        evaluate,
        "apply_device_profile",
        lambda config, _device, **_kwargs: config,
    )
    monkeypatch.setattr(
        evaluate,
        "evaluate_capture_replay",
        lambda *_args, **_kwargs: report,
    )
    output = tmp_path / "must-not-exist.json"

    code = evaluate.main(
        [
            "--corpus",
            str(corpus.path),
            "--logical-turn-shadow",
            "--report",
            str(output),
        ]
    )

    assert code == 2
    assert not output.exists()
    assert json.loads(capsys.readouterr().out) == {
        "execution_complete": False,
        "error": "capture_replay_evaluation_unavailable",
    }


@pytest.mark.parametrize(
    ("mode_args", "schema_version", "mode_sections"),
    (
        ((), 1, {"logical_turn_shadow": {}}),
        (("--logical-turn-shadow",), 2, {}),
        (
            ("--logical-turn-shadow",),
            2,
            {
                "logical_turn_shadow": {},
                "logical_turn_terminal_lineage": {},
            },
        ),
        (
            (
                "--logical-turn-shadow",
                "--logical-turn-terminal-lineage",
            ),
            3,
            {"logical_turn_shadow": {}},
        ),
    ),
)
def test_cli_rejects_report_mode_section_mismatch_before_publication(
    tmp_path,
    monkeypatch,
    capsys,
    mode_args: tuple[str, ...],
    schema_version: int,
    mode_sections: dict[str, object],
) -> None:
    corpus = LoadedReplayCorpus(
        path=tmp_path / "private-corpus.json",
        digest="a" * 64,
        schema_version=1,
        purpose="PRIVATE PURPOSE",
        sample_rate_hz=SAMPLE_RATE_HZ,
        frame_samples=FRAME_SAMPLES,
        cases=(_case(tmp_path),),
        audio_bytes=100,
    )
    report = {
        "execution_complete": True,
        "quality_verdict": "diagnostic_only",
        "schema_version": schema_version,
        "corpus": {"manifest_sha256": corpus.digest},
        "engine": {"provider": "cpu"},
        "metrics": {
            "evaluations": 1,
            "coverage": {"complete": True},
        },
        **mode_sections,
    }
    monkeypatch.setattr(evaluate, "load_corpus", lambda _path: corpus)
    monkeypatch.setattr(evaluate, "load_config", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        evaluate,
        "apply_device_profile",
        lambda config, _device, **_kwargs: config,
    )
    monkeypatch.setattr(
        evaluate,
        "evaluate_capture_replay",
        lambda *_args, **_kwargs: report,
    )
    output = tmp_path / "must-not-exist.json"

    code = evaluate.main(
        [
            "--corpus",
            str(corpus.path),
            *mode_args,
            "--report",
            str(output),
        ]
    )

    assert code == 2
    assert not output.exists()
    assert json.loads(capsys.readouterr().out) == {
        "execution_complete": False,
        "error": "capture_replay_evaluation_unavailable",
    }


def test_guarded_cli_uses_killable_bounded_worker(
    tmp_path,
    monkeypatch,
    capsys,
):
    receipt = {
        "execution_complete": True,
        "quality_verdict": "diagnostic_only",
        "report_sha256": "a" * 64,
        "corpus_sha256": "b" * 64,
        "evaluations": 4,
        "coverage_complete": True,
        "schema_version": 2,
        "logical_turn_shadow": True,
        "logical_turn_terminal_lineage": False,
        "logical_turn_async_terminal_delivery": False,
    }
    observed: dict[str, object] = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed["timeout"] = kwargs["timeout"]
        observed["worker"] = kwargs["env"][evaluate._WORKER_ENV]
        kwargs["stdout"].write(
            json.dumps(receipt, separators=(",", ":")).encode("utf-8")
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(evaluate.subprocess, "run", run)

    code = evaluate.guarded_main(
        [
            "--corpus",
            str(tmp_path / "private-corpus.json"),
            "--logical-turn-shadow",
            "--report",
            str(tmp_path / "private-report.json"),
            "--watchdog-seconds",
            "123",
        ]
    )

    assert code == 0
    assert observed["timeout"] == 123
    assert observed["worker"] == "1"
    assert observed["command"][:4] == [
        __import__("sys").executable,
        "-B",
        "-m",
        "tools.capture_replay_eval",
    ]
    assert "--logical-turn-shadow" in observed["command"]
    assert json.loads(capsys.readouterr().out) == receipt


@pytest.mark.parametrize(
    ("mode_args", "worker_mode"),
    (
        ((), (2, True, False, False)),
        (("--logical-turn-shadow",), (1, False, False, False)),
        (
            (
                "--logical-turn-shadow",
                "--logical-turn-terminal-lineage",
            ),
            (2, True, False, False),
        ),
    ),
)
def test_guarded_cli_rejects_worker_mode_mismatch(
    tmp_path,
    monkeypatch,
    capsys,
    mode_args: tuple[str, ...],
    worker_mode: tuple[int, bool, bool, bool],
) -> None:
    receipt = {
        "execution_complete": True,
        "quality_verdict": "diagnostic_only",
        "report_sha256": "a" * 64,
        "corpus_sha256": "b" * 64,
        "evaluations": 1,
        "coverage_complete": True,
        "schema_version": worker_mode[0],
        "logical_turn_shadow": worker_mode[1],
        "logical_turn_terminal_lineage": worker_mode[2],
        "logical_turn_async_terminal_delivery": worker_mode[3],
    }

    def run(_command, **kwargs):
        kwargs["stdout"].write(
            json.dumps(receipt, separators=(",", ":")).encode("utf-8")
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(evaluate.subprocess, "run", run)

    code = evaluate.guarded_main(
        [
            "--corpus",
            str(tmp_path / "private-corpus.json"),
            *mode_args,
            "--report",
            str(tmp_path / "private-report.json"),
        ]
    )

    assert code == 2
    assert json.loads(capsys.readouterr().out) == {
        "execution_complete": False,
        "error": "capture_replay_evaluation_unavailable",
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "schema_bool",
        "raw_not_bool",
        "terminal_not_bool",
        "async_not_bool",
        "terminal_without_raw",
        "async_without_terminal",
        "missing",
        "extra",
    ),
)
def test_worker_receipt_rejects_invalid_mode_binding(mutation: str) -> None:
    receipt = {
        "execution_complete": True,
        "quality_verdict": "diagnostic_only",
        "report_sha256": "a" * 64,
        "corpus_sha256": "b" * 64,
        "evaluations": 1,
        "coverage_complete": True,
        "schema_version": 2,
        "logical_turn_shadow": True,
        "logical_turn_terminal_lineage": False,
        "logical_turn_async_terminal_delivery": False,
    }
    if mutation == "schema_bool":
        receipt["schema_version"] = True
    elif mutation == "raw_not_bool":
        receipt["logical_turn_shadow"] = 1
    elif mutation == "terminal_not_bool":
        receipt["logical_turn_terminal_lineage"] = 0
    elif mutation == "async_not_bool":
        receipt["logical_turn_async_terminal_delivery"] = 0
    elif mutation == "terminal_without_raw":
        receipt["schema_version"] = 3
        receipt["logical_turn_shadow"] = False
        receipt["logical_turn_terminal_lineage"] = True
    elif mutation == "async_without_terminal":
        receipt["schema_version"] = 4
        receipt["logical_turn_async_terminal_delivery"] = True
    elif mutation == "missing":
        del receipt["logical_turn_shadow"]
    else:
        receipt["unexpected"] = False

    with pytest.raises(evaluate.CaptureReplayEvaluationError):
        evaluate._validated_worker_receipt(
            json.dumps(receipt, separators=(",", ":")).encode("utf-8"),
            returncode=0,
            logical_turn_shadow=True,
            logical_turn_terminal_lineage=False,
            logical_turn_async_terminal_delivery=False,
        )


def test_guarded_cli_rejects_terminal_lineage_without_raw_before_spawning(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    def must_not_run(*_args, **_kwargs):
        raise AssertionError("worker must not start")

    monkeypatch.setattr(evaluate.subprocess, "run", must_not_run)

    code = evaluate.guarded_main(
        [
            "--corpus",
            str(tmp_path / "private-corpus.json"),
            "--logical-turn-terminal-lineage",
            "--report",
            str(tmp_path / "private-report.json"),
        ]
    )

    assert code == 2
    assert json.loads(capsys.readouterr().out) == {
        "execution_complete": False,
        "error": "capture_replay_evaluation_unavailable",
    }


def test_guarded_cli_timeout_is_detail_free(tmp_path, monkeypatch, capsys):
    private = str(tmp_path / "private-corpus.json")

    def timeout(*_args, **_kwargs):
        raise evaluate.subprocess.TimeoutExpired(
            cmd=["private", private],
            timeout=1,
        )

    monkeypatch.setattr(evaluate.subprocess, "run", timeout)

    code = evaluate.guarded_main(
        [
            "--corpus",
            private,
            "--report",
            str(tmp_path / "private-report.json"),
            "--watchdog-seconds",
            "1",
        ]
    )

    assert code == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "execution_complete": False,
        "error": "capture_replay_evaluation_unavailable",
    }
    assert private not in captured.out
    assert captured.err == ""
