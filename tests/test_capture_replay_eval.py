from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from always_on_agent.acoustic import (
    AcousticLineage,
    AcousticSource,
    AcousticSpan,
    EndpointReason,
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
from tools.capture_replay.metrics import ReplayAcousticEvent
from tools import capture_replay_eval as evaluate


def test_evaluator_provenance_includes_streaming_decode_owner() -> None:
    assert (
        "core/engines/_sherpa_streaming_decode.py"
        in evaluate._EVALUATOR_SOURCE_FILES
    )
    digest = evaluate._evaluator_source_digest()
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_multi_case_replay_reuses_one_owner_then_closes_it(tmp_path) -> None:
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
    ) == "single-owner-synchronous"


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


def test_report_writer_is_private_no_overwrite(tmp_path):
    report = tmp_path / "report.json"
    evaluate._write_new_private(report, b'{"execution_complete":true}\n')

    assert stat.S_IMODE(report.stat().st_mode) == 0o600
    assert report.read_bytes() == b'{"execution_complete":true}\n'
    with pytest.raises(evaluate.CaptureReplayEvaluationError):
        evaluate._write_new_private(report, b"different")
    assert report.read_bytes() == b'{"execution_complete":true}\n'


def test_cli_success_and_failure_emit_only_aggregate_receipts(
    tmp_path,
    monkeypatch,
    capsys,
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
    output = tmp_path / "aggregate-report.json"

    code = evaluate.main(
        [
            "--corpus",
            private_path,
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
    }
    assert captured.err == ""
    assert private_phrase not in captured.out
    assert private_path not in captured.out
    assert stat.S_IMODE(output.stat().st_mode) == 0o600

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
    assert json.loads(capsys.readouterr().out) == receipt


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
