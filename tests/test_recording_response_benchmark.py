from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import recording_response_benchmark as b


def _server():
    return {
        "daemon_version": "0.30.6",
        "roles": {
            role: {
                "resident": False,
                "size_bytes": None,
                "size_vram_bytes": None,
                "context_length": None,
            }
            for role in ("main", "fast")
        },
    }


def _binding():
    package = {"version": "1.2.3", "record_sha256": "e" * 64}
    return {
        **{
            key: "a" * 64
            for key in (
                "resolved_config_sha256",
                "execution_config_sha256",
                "selection_sha256",
                "asr_settings_sha256",
                "llm_settings_sha256",
                "safety_policy_sha256",
                "corpus_sha256",
                "manifest_sha256",
                "scoring_sha256",
                "audio_artifacts_sha256",
                "ollama_sha256",
                "ollama_model_root_sha256",
            )
        },
        "source": {
            "sha256": "b" * 64,
            "files": len(b._SOURCE_FILES),
            "bytes": 1000,
            "revision": "a" * 40,
            "clean": True,
        },
        "corpus_kind": "recorded-legacy",
        "corpus_clips": 1,
        "corpus_seconds": 1.0,
        "audio_artifact_files": 5,
        "audio_artifact_bytes": 5000,
        "runtime": {
            "python": {"version": "3.12.3", "executable_sha256": "c" * 64},
            "distributions": {
                key: dict(package)
                for key in ("numpy", "sherpa_onnx", "onnxruntime", "ollama")
            },
        },
        "server_before": _server(),
        "ollama_roles": {
            role: {
                "manifest_sha256": "d" * 64,
                "layer_tuple_sha256": "d" * 64,
                "layer_files": 3,
                "layer_bytes": 5000,
                "effective_config_sha256": "e" * 64,
                "effective_show_sha256": "e" * 64,
                "completion": True,
                "vision": False,
                "canonical_minicpm_adoption_check": False,
            }
            for role in ("main", "fast")
        },
    }


def _row(**overrides):
    return {
        "repeat": 0,
        "ordinal": 0,
        "phase": "first",
        "responded": True,
        "idle": True,
        "handled_local": False,
        "asr_accuracy": {
            "legacy": b._accuracy_pairs([("public", "public")]),
            "english": b._accuracy_pairs([("public", "public")]),
        },
        **{
            key: 1
            for key in (
                "final_count",
                "reply_fragments",
                "auxiliary_fragments",
                "invalid_chunks",
                "failed_fragments",
                "dropped_fragments",
                "nonzero_samples",
                "answer_pcm_samples",
                "reply_text_chars",
                "provider_text_chunks",
                "provider_text_chars",
                "provider_requests",
                "control_events",
                "task_failures",
                "task_cancellations",
            )
        },
        "answer_audio_seconds": 0.1,
        **{key: 0.1 for key in b._METRICS},
        **overrides,
    }


def _report():
    return {
        "status": "ok",
        "planned": 1,
        "completed": 1,
        "rows": [_row()],
        "startup_seconds": 0.2,
        "warm_plan_finished": True,
        "failure_stage": None,
        "resources": {"process_peak_rss_bytes": 1000, "process_cpu_seconds": 0.1},
        "server_after": _server(),
        "binding": _binding(),
        "inputs_verified_after": True,
    }


def test_source_manifest_is_bounded_explicit_seam():
    receipt = b.source_binding()
    assert receipt["files"] == len(b._SOURCE_FILES)
    assert 1000 < receipt["bytes"] <= 8 * 1024**2
    assert b._HASH.fullmatch(receipt["sha256"])


@pytest.mark.parametrize(
    "settings",
    [
        {"asr_final_backend": "qwen_cpu"},
        {"asr_final_backend": " QWEN_CPU "},
        {"asr_final_verifier_backend": "sense_voice"},
        {"asr_final_verifier_backend": "parakeet_tdt_cpu"},
        {"asr_final_backend": "future_leased_adapter"},
        {"asr_final_verifier_backend": None},
    ],
)
def test_leased_or_unknown_endpoints_refuse_before_assets(monkeypatch, tmp_path, settings):
    import core.config as config_module
    import core.performance as performance
    import core.speech_chunking as speech
    import core.voice_model_selection as voice
    import tools.bench.production as production

    original = {"sherpa": settings}
    monkeypatch.setattr(config_module, "load_config", lambda *a, **k: deepcopy(original))
    monkeypatch.setattr(config_module, "apply_device_profile", lambda cfg, *a, **k: cfg)
    monkeypatch.setattr(performance, "apply_performance_mode", lambda cfg, **k: (cfg, None))
    monkeypatch.setattr(voice, "select_voice_model", lambda cfg: (cfg, None))
    monkeypatch.setattr(speech, "apply_speech_latency", lambda cfg: (cfg, None))
    monkeypatch.setattr(
        production,
        "_bind_execution_artifact_paths",
        lambda *a, **k: pytest.fail("unsupported endpoints reached asset binding"),
    )
    with pytest.raises(b.BenchmarkError, match="^invalid_inputs$"):
        b.resolve_config(tmp_path, "cpu_laptop", "current")


def test_endpoint_retirement_source_changes_measurement_binding(monkeypatch):
    from tools.conversation_eval import provenance

    lease_bytes = [b"first endpoint retirement implementation"]
    monkeypatch.setattr(
        b,
        "_read",
        lambda path, limit: lease_bytes[0]
        if path.name == "_endpoint_lease.py"
        else b"unchanged measured source",
    )
    monkeypatch.setattr(
        provenance, "repository_metadata", lambda root: {"revision": "a" * 40, "dirty": False}
    )
    before = b.source_binding()
    lease_bytes[0] = b"changed endpoint retirement implementation"
    after = b.source_binding()
    assert before["sha256"] != after["sha256"]


@pytest.mark.parametrize(
    "settings",
    [
        {"asr_final_backend": "qwen_cpu"},
        {"asr_final_verifier_backend": "sense_voice"},
        {"asr_final_verifier_backend": "parakeet_tdt_cpu"},
    ],
)
def test_direct_pipeline_and_engine_refuse_leased_endpoints(monkeypatch, settings):
    import core.llm_factory as factory

    monkeypatch.setattr(
        factory, "build_llms", lambda *a, **k: pytest.fail("model startup was reached")
    )
    with pytest.raises(b.BenchmarkError, match="^invalid_inputs$"):
        b.run_pipeline(
            None, {"sherpa": settings}, repeats=1, startup="configured",
            case_timeout=1, publish=lambda row: pytest.fail("a receipt was published"),
        )
    with pytest.raises(b.BenchmarkError, match="^invalid_inputs$"):
        b.measured_engine(_cfg(**settings))


def test_observer_ignores_aux_silence_and_stamps_actual_nonzero():
    observed = b.Observation()
    observed.diagnostic(
        SimpleNamespace(
            stage="playback_admitted", playback_kind="latency_ack", correlation_id="ack"
        )
    )
    observed.diagnostic(
        SimpleNamespace(
            stage="playback_admitted", playback_kind="reply", correlation_id="reply"
        )
    )
    observed.pcm("ack", np.ones(4, dtype=np.float32), clock=lambda: 1.0)
    observed.pcm("reply", np.zeros(4, dtype=np.float32), clock=lambda: 2.0)
    assert observed.first_pcm is None
    observed.pcm(
        "reply",
        np.array([0.0, 0.5], dtype=np.float32),
        clock=lambda: 3.0,
        sample_rate=2,
    )
    assert observed.first_pcm == 3.0
    assert observed.nonzero_samples == 1
    assert observed.answer_pcm_samples == 6
    assert observed.auxiliary_fragments == 1
    assert observed.answer_audio_seconds > 1.0


@pytest.mark.parametrize(
    "pcm",
    [
        np.array([float("nan")], dtype=np.float32),
        np.array([float("inf")], dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        np.array([1], dtype=np.int16),
    ],
)
def test_invalid_pcm_cannot_claim_first_audio(pcm):
    observed = b.Observation()
    observed.reply_ids.add("reply")
    with pytest.raises(b.BenchmarkError):
        observed.pcm("reply", pcm)
    assert observed.first_pcm is None and observed.invalid_chunks == 1


class FakeTts:
    sample_rate = 16000
    num_speakers = 1

    def __init__(self):
        self.calls = []

    def generate(self, text, sid=0, speed=1.0, callback=None):
        self.calls.append((text, sid, speed))
        samples = np.array([0.0, 0.25, 0.25, 0.25], dtype=np.float32)
        if callback is not None:
            callback(np.zeros(4, dtype=np.float32), 0.1)
            callback(samples, 1.0)
        return SimpleNamespace(samples=samples, sample_rate=16000)


def _cfg(**options):
    from core.engines.sherpa import SherpaConfig

    return SherpaConfig(
        tts_target_rms=0.0,
        tts_output_leveler=False,
        tts_declick=False,
        tts_dc_block=False,
        tts_output_lowpass_hz=0.0,
        **options,
    )


class FakeStream:
    def accept_waveform(self, *args):
        pass


class FakeRecognizer:
    def __init__(self):
        self.text = "What is the capital of France?"

    def create_stream(self):
        return FakeStream()

    def is_ready(self, stream):
        return False

    def get_result(self, stream):
        return self.text

    def is_endpoint(self, stream):
        return bool(self.text)

    def reset(self, stream):
        self.text = ""


def test_real_runtime_and_production_synth_use_null_sink_no_audio(monkeypatch):
    from core.runtime import VoiceRuntime
    from core.llm import EchoLLM
    from core.engines.sherpa import SherpaOnnxEngine
    from core.metrics import TTS_RENDER_START_OBSERVED

    monkeypatch.setattr(
        SherpaOnnxEngine,
        "start",
        lambda *_a, **_k: pytest.fail("audio start forbidden"),
    )
    tts = FakeTts()
    observed = b.Observation()
    engine = b.measured_engine(
        _cfg(), (FakeRecognizer(), None, None, None, tts), observation=observed
    )
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        stream_tts=True,
        warm_on_start=False,
    )
    engine.reply_classifier = lambda fragment: (
        fragment in runtime._playback_metric_turns
    )
    runtime.bus.subscribe(observed.event)
    runtime.start(run_bus=False)
    try:
        engine.replay_samples(np.ones(16, dtype=np.float32), 16000)
        assert runtime.wait_idle(timeout=3.0)
        assert observed.final_count == 1
        assert observed.first_pcm is not None
        assert observed.reply_fragments >= 1
        assert observed.nonzero_samples >= 3
        runtime.metrics.close_turn()
        assert any(
            TTS_RENDER_START_OBSERVED in record.stamps
            for record in runtime.metrics.records()
        )
        assert tts.calls
    finally:
        runtime.stop()


def test_tracked_sink_preserves_expression_and_completed_text(monkeypatch):
    from core.engine import EngineCallbacks, TrackedSpeech, SpeechStyle, PlaybackOutcome
    from core.engines.sherpa import SherpaOnnxEngine

    monkeypatch.setattr(
        SherpaOnnxEngine,
        "start",
        lambda *_a, **_k: pytest.fail("audio start forbidden"),
    )
    tts = FakeTts()
    observed = b.Observation()
    observed.reply_ids.add("reply")
    engine = b.measured_engine(
        _cfg(tts_markup=True, tts_lock_speaker_id=True),
        (FakeRecognizer(), None, None, None, tts),
        observation=observed,
    )
    engine.start(EngineCallbacks())
    receipts = []
    starts = []
    engine.speak_tracked(
        TrackedSpeech("reply", "A public test.", SpeechStyle(rate=1.1)),
        on_terminal=receipts.append,
        on_started=starts.append,
    )
    assert starts == ["reply"] and len(receipts) == 1
    assert receipts[0].outcome is PlaybackOutcome.COMPLETED
    assert receipts[0].safe_text_prefix == "A public test."
    assert tts.calls[0] == ("A public test.", 0, 1.1)
    engine.stop()


def test_entered_tts_error_is_not_retried_or_claimed_as_pcm():
    from core.engine import EngineCallbacks, TrackedSpeech, PlaybackOutcome

    class FailedTts(FakeTts):
        def generate(self, *args, **kwargs):
            self.calls.append(1)
            raise TypeError("PRIVATE_NATIVE_ERROR")

    tts = FailedTts()
    observed = b.Observation()
    observed.reply_ids.add("reply")
    engine = b.measured_engine(
        _cfg(), (FakeRecognizer(), None, None, None, tts), observation=observed
    )
    engine.start(EngineCallbacks())
    receipts = []
    engine.speak_tracked(TrackedSpeech("reply", "Public."), on_terminal=receipts.append)
    assert tts.calls == [1] and observed.first_pcm is None
    assert observed.failed_fragments == 1
    assert receipts[0].outcome is PlaybackOutcome.FAILED
    engine.stop()


def test_valid_protocol_and_no_private_fields():
    report = _report()
    b.validate_report(report, {"repeats": 1})
    assert "transcript" not in json.dumps(report)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["rows"][0].update(provider_requests={"PRIVATE": 1}),
        lambda r: r["rows"][0].update(pcm_to_final_seconds=float("nan")),
        lambda r: r["rows"][0].update(transcript="PRIVATE"),
        lambda r: r["rows"][0].update(ordinal=9),
        lambda r: r["rows"][0].update(phase="PRIVATE"),
        lambda r: r["rows"][0].update(responded=False),
        lambda r: r["binding"].update(corpus_kind="PRIVATE"),
        lambda r: r["binding"]["runtime"]["distributions"].update(PRIVATE={}),
        lambda r: r["binding"]["ollama_roles"]["main"].update(blob_sha256="PRIVATE"),
        lambda r: r["binding"]["source"].update(files=True),
    ],
)
def test_rejects_worker_protocol_text_and_counter_exfiltration(mutation):
    report = _report()
    mutation(report)
    with pytest.raises(b.BenchmarkError):
        b.validate_report(report, {"repeats": 1})


def test_first_and_subsequent_cohorts_and_silence_are_not_pooled():
    report = _report()
    report["rows"].append(
        _row(
            phase="subsequent",
            repeat=1,
            responded=False,
            pcm_to_first_answer_pcm_seconds=None,
        )
    )
    report.update(planned=2, completed=2)
    b.validate_report(report, {"repeats": 2})
    result = b.aggregate(report)
    assert result["summary"]["first"]["responses"] == 1
    assert result["summary"]["subsequent"]["responses"] == 0
    assert result["summary"]["subsequent"]["no_answer_pcm"] == 1
    assert (
        result["summary"]["subsequent"]["metrics"]["pcm_to_first_answer_pcm_seconds"]
        is None
    )


def test_paired_comparison_uses_joint_responding_cohort():
    left = _report()
    right = deepcopy(left)
    right["rows"][0]["pcm_to_first_answer_pcm_seconds"] = 0.05
    assert b.compare_paired(left, right)["candidate_minus_baseline_first_pcm_seconds"][
        "p50"
    ] == pytest.approx(-0.05)
    right["rows"][0]["responded"] = False
    right["rows"][0]["pcm_to_first_answer_pcm_seconds"] = None
    result = b.compare_paired(left, right)
    assert result["responding_pairs"] == 0 and result["response_losses"] == 1
    assert result["candidate_minus_baseline_first_pcm_seconds"] is None


@pytest.mark.parametrize(
    "key",
    ["corpus_sha256", "manifest_sha256", "asr_settings_sha256", "llm_settings_sha256"],
)
def test_paired_comparison_refuses_changed_input_or_model_policy(key):
    left = _report()
    right = deepcopy(left)
    right["binding"][key] = "f" * 64
    with pytest.raises(b.BenchmarkError):
        b.compare_paired(left, right)


def test_environment_scrubs_proxy_credentials_and_cloud(monkeypatch):
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "OLLAMA_API_KEY",
        "OPENAI_API_KEY",
        "HF_TOKEN",
        "SPEAKER_CONFIG",
        "SPEAKER_NO_LOCAL_CONFIG",
    ):
        monkeypatch.setenv(key, "PRIVATE")
    environment = b.child_environment()
    assert not any("PRIVATE" in value for value in environment.values())
    assert environment["HF_HUB_OFFLINE"] == "1"


def test_transport_guard_denies_nonloopback_and_wrong_port(monkeypatch):
    import socket

    called = []
    monkeypatch.setattr(
        socket.socket, "connect", lambda self, address: called.append(address)
    )
    monkeypatch.setattr(
        socket.socket, "connect_ex", lambda self, address: called.append(address)
    )
    original = socket.socket.connect
    original_ex = socket.socket.connect_ex
    original_send = socket.socket.sendto
    try:
        b._local_guard()
        for address in [
            ("example.com", 11434),
            ("127.0.0.1", 443),
            "/private/unix/socket",
        ]:
            with pytest.raises(b.BenchmarkError):
                socket.socket.connect(object(), address)
        socket.socket.connect(object(), ("127.0.0.1", 11434))
        assert called == [("127.0.0.1", 11434)]
    finally:
        socket.socket.connect = original
        socket.socket.connect_ex = original_ex
        socket.socket.sendto = original_send


def test_private_report_never_overwrites(tmp_path):
    report = tmp_path / "report.json"
    b._write_private(report, {"count": 1}, exclusive=True)
    assert report.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        b._write_private(report, {"count": 2}, exclusive=True)
    assert json.loads(report.read_text()) == {"count": 1}


def test_multifinal_failure_retains_observed_pcm_and_completed_denominator():
    report = _report()
    report.update(
        status="worker_failed", inputs_verified_after=False, failure_stage="pipeline"
    )
    report["rows"][0].update(
        final_count=2,
        pcm_to_final_seconds=None,
        final_to_first_answer_pcm_seconds=None,
        asr_accuracy=None,
    )
    b.validate_report(report, {"repeats": 1})
    assert report["completed"] == 1 and report["rows"][0]["responded"] is True


def test_request_rejects_unbounded_or_implicit_configuration():
    valid = {
        "config_root": "/public",
        "manifest": "/public/manifest.json",
        "source_root": "/public",
        "device": "cpu_laptop",
        "variant": "current",
        "startup": "configured",
        "repeats": 1,
        "case_timeout": 2.0,
        "ollama_model_root": None,
    }
    b.validate_request(valid)
    for changes in (
        {"repeats": True},
        {"case_timeout": float("nan")},
        {"case_timeout": 601.0},
        {"device": "auto"},
        {"manifest": "relative"},
        {"variant": "other"},
        {"extra": "PRIVATE"},
    ):
        with pytest.raises(b.BenchmarkError):
            b.validate_request({**valid, **changes})


def test_tts_variants_preserve_asr_llm_authority_and_dsp(monkeypatch, tmp_path):
    import core.config as config_module
    import core.performance as performance
    from core.config import deep_merge

    original = {
        "llm": {
            "backend": "ollama",
            "main_model": "main",
            "fast_model": "fast",
            "options": {"num_ctx": 2048, "num_predict": 256},
            "cloud": {"enabled": False},
        },
        "sherpa": {
            "asr_encoder": "stream.onnx",
            "asr_final_backend": "nemo_transducer",
            "asr_final_verifier_backend": "faster_whisper",
            "asr_num_threads": 4,
            "tts_model": "kokoro.onnx",
            "tts_tokens": "tokens",
            "tts_output_leveler": True,
            "tts_markup": True,
        },
        "tts": {"streaming": True, "speech_latency": "normal"},
        "reminders": {"enabled": True},
        "trusted_apps": {"enabled": True},
        "input_gate": {"enabled": False},
        "performance_mode": "current",
        "voice_model_profile": "current",
    }
    monkeypatch.setattr(
        config_module, "load_config", lambda *a, **k: deepcopy(original)
    )
    monkeypatch.setattr(config_module, "apply_device_profile", lambda cfg, *a, **k: cfg)

    def select(cfg, requested=None, **kwargs):
        if requested is None:
            return cfg, None
        assert requested == "responsive"
        return deep_merge(
            cfg,
            {
                "sherpa": {
                    "asr_encoder": "int8.onnx",
                    "asr_num_threads": 2,
                    "tts_model": "vits.onnx",
                    "tts_num_threads": 2,
                },
                "llm": {"main_keep_alive": "60s", "fast_keep_alive": -1},
            },
        ), SimpleNamespace(sha256="c" * 64)

    monkeypatch.setattr(performance, "apply_performance_mode", select)
    current, current_id = b.resolve_config(tmp_path, "cpu_laptop", "current")
    vits, vits_id = b.resolve_config(tmp_path, "cpu_laptop", "vits")
    fast, fast_id = b.resolve_config(tmp_path, "cpu_laptop", "vits-fast")
    for cfg in (vits, fast):
        assert cfg["sherpa"]["asr_encoder"] == current["sherpa"]["asr_encoder"]
        assert cfg["sherpa"]["asr_num_threads"] == 4
        assert cfg["sherpa"]["asr_final_verifier_backend"] == "faster_whisper"
        assert cfg["llm"] == current["llm"]
        assert cfg["sherpa"]["tts_output_leveler"] is True
        assert not cfg["reminders"]["enabled"] and not cfg["trusted_apps"]["enabled"]
        assert cfg["input_gate"] == current["input_gate"]
    assert vits["tts"]["speech_latency"] == "normal"
    assert fast["tts"]["speech_latency"] == "fast"
    assert (
        current_id["asr_settings_sha256"]
        == vits_id["asr_settings_sha256"]
        == fast_id["asr_settings_sha256"]
    )
    assert (
        current_id["llm_settings_sha256"]
        == vits_id["llm_settings_sha256"]
        == fast_id["llm_settings_sha256"]
    )
    assert original["reminders"]["enabled"] is True


def test_isolated_timeout_retires_exact_worker_and_keeps_prefix(monkeypatch, tmp_path):
    import subprocess

    retired = []

    class Process:
        returncode = None

        def communicate(self, raw, timeout):
            request = json.loads(raw)
            result = _report()
            result.update(
                status="case_timeout",
                inputs_verified_after=False,
                failure_stage="pipeline",
            )
            b._write_private(request["report"], result)
            raise subprocess.TimeoutExpired("public-worker", timeout)

    process = Process()
    monkeypatch.setattr(b.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(b, "_stop", lambda target: retired.append(target))
    result = b.run_isolated({"repeats": 1}, scratch=tmp_path, timeout=0.1)
    assert retired == [process] and result["status"] == "cell_timeout"
    assert result["completed"] == 1


def test_parent_interrupt_always_retires_exact_worker(monkeypatch, tmp_path):
    retired = []

    class Process:
        def communicate(self, *a, **k):
            raise KeyboardInterrupt

    process = Process()
    monkeypatch.setattr(b.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(b, "_stop", lambda target: retired.append(target))
    with pytest.raises(KeyboardInterrupt):
        b.run_isolated({"repeats": 1}, scratch=tmp_path, timeout=0.1)
    assert retired == [process]


def test_stopped_engine_cannot_enter_native_generation():
    from core.engine import EngineCallbacks, TrackedSpeech, PlaybackOutcome

    tts = FakeTts()
    engine = b.measured_engine(_cfg(), (FakeRecognizer(), None, None, None, tts))
    engine.start(EngineCallbacks())
    engine.stop()
    receipts = []
    engine.speak_tracked(TrackedSpeech("reply", "Public."), on_terminal=receipts.append)
    assert not tts.calls and receipts[0].outcome is PlaybackOutcome.DROPPED


def test_pipeline_reuses_models_fresh_sessions_and_warms_only_once(
    monkeypatch, tmp_path
):
    import threading
    import wave
    import ollama
    import core.app as app
    import core.llm_factory as factory
    from core.llm import OllamaLLM

    waveform = tmp_path / "synthetic.wav"
    with wave.open(str(waveform), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(bytes(32))
    corpus = SimpleNamespace(
        clips=(SimpleNamespace(path=waveform, reference="Public."),)
    )
    transports = []
    models_built = []
    engines = []
    runtimes = []
    configs = []
    timers = []
    closed = []

    class SDK:
        def __init__(self, **kwargs):
            self._client = SimpleNamespace(close=lambda: closed.append(self))
            transports.append(kwargs)

    monkeypatch.setattr(ollama, "Client", SDK)
    main = OllamaLLM("public-main", client=object())
    fast = OllamaLLM("public-fast", client=object())
    monkeypatch.setattr(factory, "build_llms", lambda *a, **k: (main, fast))

    class Engine:
        def __init__(self, cfg, models, renderer, observed):
            self.observed = observed
            self.last_final = "Public."
            if models is None:
                models_built.append(1)
            self._recognizer = object()
            self._final_recognizer = None
            self._final_verifier = None
            self._punct = None
            self._tts = FakeTts()
            self.renderer = renderer or SimpleNamespace(
                _stop_speaking=threading.Event()
            )
            engines.append(self)

        def replay_samples(self, *args):
            import time

            self.observed.final_count = 1
            self.observed.final_at = time.perf_counter()
            self.observed.first_pcm = time.perf_counter()

        def stop(self):
            pass

    monkeypatch.setattr(b, "measured_engine", Engine)

    class Timer:
        def __init__(self, seconds, callback):
            self.callback = callback
            self.seconds = seconds
            self.cancelled = False
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(b.threading, "Timer", Timer)

    class Runtime:
        def __init__(self, config, engine):
            self.engine = engine
            self._playback_metric_turns = {}
            self.warm_ready = threading.Event()
            self.warm_ready.set()
            self.bus = SimpleNamespace(subscribe=lambda handler: None)
            self.metrics = SimpleNamespace(close_turn=lambda: None, records=lambda: [])
            self.stopped = False
            configs.append(deepcopy(config))
            runtimes.append(self)

        def start(self, *, run_bus):
            assert run_bus is True

        def wait_idle(self, timeout):
            return True

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(
        app, "build_runtime", lambda config, **kwargs: Runtime(config, kwargs["engine"])
    )
    receipts = []
    publish_fails = [False]

    def publish(value):
        if publish_fails[0]:
            raise OSError("public-receipt-failure")
        receipts.append(value)

    config = {"llm": {"backend": "ollama"}, "warm_on_start": True, "sherpa": {}}
    result = b.run_pipeline(
        corpus,
        config,
        repeats=2,
        startup="configured",
        case_timeout=3.0,
        publish=publish,
    )
    assert len(models_built) == 1 and len(runtimes) == 2
    assert configs[0]["warm_on_start"] is True and configs[1]["warm_on_start"] is False
    assert engines[0].renderer is engines[1].renderer
    assert all(runtime.stopped for runtime in runtimes)
    assert all(timer.cancelled for timer in timers) and len(closed) == 2
    assert result["completed"] == result["planned"] == 2
    assert (
        result["rows"][0]["phase"] == "first"
        and result["rows"][1]["phase"] == "subsequent"
    )
    assert all(
        options["trust_env"] is False and options["follow_redirects"] is False
        for options in transports
    )

    exits = []

    def exit_attempt(code):
        exits.append(code)
        raise RuntimeError("public-exit")

    monkeypatch.setattr(b.os, "_exit", exit_attempt)
    publish_fails[0] = True
    with pytest.raises(RuntimeError, match="public-exit"):
        timers[0].callback()
    assert exits == [71]


def test_null_sink_publication_is_atomic_with_stop_and_callback_is_unlocked():
    import threading
    from core.engine import EngineCallbacks, TrackedSpeech

    entered = threading.Event()
    release = threading.Event()
    stopped = threading.Event()
    callback_unlocked = []
    observed = b.Observation()
    observed.reply_ids.add("reply")
    original = observed.pcm

    def blocked(*a, **k):
        entered.set()
        assert release.wait(2.0)
        original(*a, **k)

    observed.pcm = blocked

    class OneChunkTts(FakeTts):
        def generate(self, text, sid=0, speed=1.0, callback=None):
            values = np.ones(4, dtype=np.float32)
            if callback is not None:
                callback(values, 1.0)
            return SimpleNamespace(samples=values, sample_rate=16000)

    engine = b.measured_engine(
        _cfg(),
        (FakeRecognizer(), None, None, None, OneChunkTts()),
        observation=observed,
    )
    engine.start(EngineCallbacks())

    def started(_fragment):
        # Another thread can take receipt lock while external callback runs.
        acquired = threading.Event()

        def probe():
            with engine._receipt_lock:
                acquired.set()

        thread = threading.Thread(target=probe)
        thread.start()
        callback_unlocked.append(acquired.wait(1.0))
        thread.join(1.0)

    worker = threading.Thread(
        target=lambda: engine.speak_tracked(
            TrackedSpeech("reply", "Public."),
            on_terminal=lambda _: None,
            on_started=started,
        )
    )
    worker.start()
    assert entered.wait(2.0)
    stopper = threading.Thread(target=lambda: (engine.stop_speaking(), stopped.set()))
    stopper.start()
    assert not stopped.wait(0.05)
    release.set()
    worker.join(2.0)
    stopper.join(2.0)
    assert not worker.is_alive() and not stopper.is_alive()
    assert stopped.is_set()
    assert callback_unlocked == [True]
    assert observed.first_pcm is not None
    engine.stop()


def test_model_binding_hashes_multimodal_layers_and_all_show_settings(
    monkeypatch, tmp_path
):
    import hashlib
    import ollama

    blobs = tmp_path / "models" / "blobs"
    blobs.mkdir(parents=True)
    layers = []
    for data in (b"public-config", b"public-text-weights", b"public-vision-weights"):
        sha = hashlib.sha256(data).hexdigest()
        (blobs / ("sha256-" + sha)).write_bytes(data)
        layers.append(
            {
                "mediaType": "application/test",
                "from": "public-source",
                "digest": "sha256:" + sha,
                "size": len(data),
            }
        )
    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/test",
        "format": "gguf",
        "runner": "llama",
        "config": layers[0],
        "layers": layers[1:],
    }
    raw = b.canonical(manifest)
    manifest_sha = hashlib.sha256(raw).hexdigest()
    aliases = ("public-main:latest", "public-fast:latest")
    for alias in aliases:
        name, tag = alias.split(":")
        path = (
            blobs.parent / "manifests" / "registry.ollama.ai" / "library" / name / tag
        )
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)
    shown = {
        "modelfile": "\n".join(
            "FROM " + str(blobs / ("sha256-" + layer["digest"].split(":")[1]))
            for layer in layers[1:]
        ),
        "template": "public-template",
        "parameters": "num_ctx 2048",
        "capabilities": ["completion", "vision"],
        "renderer": "public-new-renderer",
        "system": "public system boundary",
        "modified_at": "public-timestamp",
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_bytes(self):
            yield b.canonical(shown)

    class Client:
        def __init__(self, **kwargs):
            self._client = SimpleNamespace(
                stream=lambda *a, **k: Response(), close=lambda: None
            )

        def list(self):
            return SimpleNamespace(
                models=[
                    SimpleNamespace(model=alias, name=alias, digest=manifest_sha)
                    for alias in aliases
                ]
            )

    monkeypatch.setattr(ollama, "Client", Client)
    config = {"llm": {"main_model": aliases[0], "fast_model": aliases[1]}}
    first = b._ollama_binding(config)
    assert first.role_contracts["main"]["layer_files"] == 3
    assert first.role_contracts["main"]["vision"] is True
    assert first.role_contracts["main"]["canonical_minicpm_adoption_check"] is None
    shown["renderer"] = "public-changed-renderer"
    second = b._ollama_binding(config)
    assert first.sha256 != second.sha256
    assert (
        first.role_contracts["main"]["layer_tuple_sha256"]
        == second.role_contracts["main"]["layer_tuple_sha256"]
    )
    # A legitimate machine cache link requires an explicit canonical grant.
    link = tmp_path / "logical-model-cache"
    link.symlink_to(blobs.parent, target_is_directory=True)
    shown["modelfile"] = shown["modelfile"].replace(str(blobs), str(link / "blobs"))
    with pytest.raises(Exception):
        b._ollama_binding(config)
    explicit = b._ollama_binding(config, model_root=blobs.parent)
    assert (
        explicit.role_contracts["main"]["layer_tuple_sha256"]
        == second.role_contracts["main"]["layer_tuple_sha256"]
    )
    outside = tmp_path / "outside-blobs"
    outside.mkdir()
    escaped = shown["modelfile"].replace(str(link / "blobs"), str(outside))
    original = shown["modelfile"]
    shown["modelfile"] = escaped
    with pytest.raises(Exception):
        b._ollama_binding(config, model_root=blobs.parent)
    shown["modelfile"] = original
    # Even with an explicit ancestor grant, individual blob symlinks refuse.
    config_blob = blobs / ("sha256-" + layers[0]["digest"].split(":")[1])
    outside_config = tmp_path / "outside-config"
    outside_config.write_bytes(config_blob.read_bytes())
    config_blob.unlink()
    config_blob.symlink_to(outside_config)
    with pytest.raises(Exception):
        b._ollama_binding(config, model_root=blobs.parent)
    config_blob.unlink()
    config_blob.write_bytes(outside_config.read_bytes())
    # Corruption cannot hide behind a content-addressed filename.
    (blobs / ("sha256-" + layers[1]["digest"].split(":")[1])).write_bytes(
        b"corrupt-public"
    )
    with pytest.raises(b.BenchmarkError):
        b._ollama_binding(config, model_root=blobs.parent)


@pytest.mark.parametrize(
    "change", [{"clips": 2}, {"exact": 2}, {"empty": 2}, {"wer": 99.0}, {"cer": 99.0}]
)
def test_quality_receipt_rejects_inconsistent_counts_and_ratios(change):
    receipt = _report()["rows"][0]["asr_accuracy"]
    receipt["english"].update(change)
    with pytest.raises(b.BenchmarkError):
        b.validate_accuracy(receipt)


def test_effective_show_hash_is_a_typed_digest_not_an_exfiltration_surface():
    report = _report()
    report["binding"]["ollama_roles"]["main"]["effective_show_sha256"] = {"PRIVATE": 1}
    with pytest.raises(b.BenchmarkError):
        b.validate_report(report, {"repeats": 1})


def test_tracked_speech_lifecycle_pairs_end_even_on_error():
    from core.engine import EngineCallbacks, TrackedSpeech

    lifecycle = []

    class FailedTts(FakeTts):
        def generate(self, *a, **k):
            raise RuntimeError("PRIVATE")

    engine = b.measured_engine(
        _cfg(), (FakeRecognizer(), None, None, None, FailedTts())
    )
    engine.start(
        EngineCallbacks(
            on_speech_start=lambda: lifecycle.append("start"),
            on_speech_end=lambda: lifecycle.append("end"),
        )
    )
    engine.speak_tracked(
        TrackedSpeech("reply", "Public."),
        on_terminal=lambda _: lifecycle.append("terminal"),
    )
    assert lifecycle == ["start", "end", "terminal"]
    engine.stop()


def test_show_digest_ignores_only_proven_parameter_and_capability_order():
    first = {
        "modelfile": "FROM /public/sha256-"
        + "a" * 64
        + '\nTEMPLATE """public\nPARAMETER literal inside template\n"""\nPARAMETER num_ctx 2048\nPARAMETER stop A\nPARAMETER stop B\n',
        "parameters": "num_ctx 2048\nstop A\nstop B\n",
        "template": "exact native template",
        "capabilities": ["completion", "vision"],
        "renderer": "public",
        "other_ordered": ["A", "B"],
    }
    second = deepcopy(first)
    second["modelfile"] = first["modelfile"].replace(
        "PARAMETER num_ctx 2048\nPARAMETER stop A\nPARAMETER stop B\n",
        "PARAMETER stop B\nPARAMETER num_ctx 2048\nPARAMETER stop A\n",
    )
    second["parameters"] = "stop B\nnum_ctx 2048\nstop A\n"
    second["capabilities"].reverse()
    assert b._show_setting_digests(first) == b._show_setting_digests(second)
    second["other_ordered"].reverse()
    assert b._show_setting_digests(first)[1] != b._show_setting_digests(second)[1]
    second = deepcopy(first)
    second["modelfile"] = second["modelfile"].replace(
        "literal inside template", "changed inside template"
    )
    assert b._show_setting_digests(first) != b._show_setting_digests(second)
    second = deepcopy(first)
    second["parameters"] += "num_ctx 4096\n"
    with pytest.raises(b.BenchmarkError):
        b._show_setting_digests(second)


def test_group_gone_after_kill_race_still_reaps_worker(monkeypatch):
    import subprocess

    calls = []

    class Process:
        pid = 1234567

        def poll(self):
            return None

        def wait(self, timeout):
            calls.append(("wait", timeout))
            if len([call for call in calls if call[0] == "wait"]) == 1:
                raise subprocess.TimeoutExpired("public-worker", timeout)
            return 0

    def killpg(pid, signal):
        calls.append(("kill", pid, signal))
        if signal == b.signal.SIGKILL:
            raise ProcessLookupError

    monkeypatch.setattr(b.os, "killpg", killpg)
    b._stop(Process())
    assert len([call for call in calls if call[0] == "wait"]) == 2


def test_residency_works_with_old_sdk_without_version_method(monkeypatch):
    import ollama

    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def raise_for_status(self):
            pass

        def iter_bytes(self):
            yield b'{"version":"0.30.6"}'

    class Client:
        def __init__(self, **kwargs):
            self._client = SimpleNamespace(stream=self.stream, close=lambda: None)
            assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False

        def stream(self, method, path, **kwargs):
            calls.append((method, path))
            return Response()

        def ps(self):
            return SimpleNamespace(models=[])

    monkeypatch.setattr(ollama, "Client", Client)
    result = b.ollama_residency(
        {
            "llm": {
                "main_model": "public-main:latest",
                "fast_model": "public-fast:latest",
            }
        }
    )
    assert result["daemon_version"] == "0.30.6"
    assert calls == [("GET", "/api/version")]
    assert not any(role["resident"] for role in result["roles"].values())


def _changed_leaves(left, right, prefix=()):
    if type(left) is dict and type(right) is dict:
        return set().union(
            *(
                _changed_leaves(left.get(key), right.get(key), (*prefix, key))
                for key in set(left) | set(right)
            )
        )
    return {prefix} if left != right or type(left) is not type(right) else set()


@pytest.mark.parametrize(
    "variant,field,previous,requested",
    [
        ("vits-asr2", "asr_num_threads", 4, 2),
        ("vits-asr8", "asr_num_threads", 4, 8),
        ("vits-tts4", "tts_num_threads", 2, 4),
        ("vits-cuda", "provider", "cpu", "cuda"),
    ],
)
def test_named_resource_variant_changes_exactly_one_deep_field(
    variant, field, previous, requested
):
    control = {
        "sherpa": {
            "provider": "cpu",
            "asr_num_threads": 4,
            "tts_num_threads": 2,
            "asr_encoder": "public-stream.onnx",
            "asr_decoding_method": "modified_beam_search",
            "asr_max_active_paths": 4,
            "asr_final_backend": "nemo_transducer",
            "asr_final_verifier_backend": "faster_whisper",
            "tts_output_leveler": True,
            "tts_markup": True,
        },
        "llm": {
            "main_model": "public-main",
            "fast_model": "public-fast",
            "options": {"num_ctx": 2048, "num_predict": 256},
        },
        "input_gate": {"enabled": False},
        "turn_merge": {"enabled": True, "hold_sec": 1.2},
        "commands": {"public-command": "public-action"},
        "tts": {"streaming": True, "speech_latency": "normal"},
    }
    before = deepcopy(control)
    candidate = b._apply_resource_variant(control, variant)
    assert _changed_leaves(control, candidate) == {("sherpa", field)}
    assert candidate["sherpa"][field] == requested
    assert control == before and control["sherpa"][field] == previous
    assert (
        candidate["llm"] == control["llm"]
        and candidate["turn_merge"] == control["turn_merge"]
    )


@pytest.mark.parametrize("previous", [None, True, "4", 0, 3, 8])
def test_asr_thread_variant_refuses_an_unmatched_control(previous):
    with pytest.raises(b.BenchmarkError):
        b._apply_resource_variant(
            {"sherpa": {"asr_num_threads": previous}}, "vits-asr2"
        )


def test_thread_variant_requires_a_literal_known_name():
    assert set(b._RESOURCE_VARIANTS) == {
        "vits-asr2",
        "vits-asr8",
        "vits-tts4",
        "vits-cuda",
    }
    with pytest.raises(KeyError):
        b._apply_resource_variant({"sherpa": {"asr_num_threads": 4}}, "vits-asr16")


@pytest.mark.parametrize(
    "variant,field,requested",
    [
        ("vits-asr2", "asr_num_threads", 2),
        ("vits-asr8", "asr_num_threads", 8),
        ("vits-tts4", "tts_num_threads", 4),
        ("vits-cuda", "provider", "cuda"),
    ],
)
def test_resolved_thread_variant_retains_full_binding_policy(
    monkeypatch, tmp_path, variant, field, requested
):
    import core.config as config_module
    import core.performance as performance
    from core.config import deep_merge

    original = {
        "llm": {
            "backend": "ollama",
            "main_model": "main",
            "fast_model": "fast",
            "options": {"num_ctx": 2048, "num_predict": 256},
            "cloud": {"enabled": False},
        },
        "sherpa": {
            "provider": "cpu",
            "asr_encoder": "stream.onnx",
            "asr_final_backend": "nemo_transducer",
            "asr_final_verifier_backend": "faster_whisper",
            "asr_num_threads": 4,
            "tts_num_threads": 2,
            "tts_model": "kokoro.onnx",
            "tts_tokens": "tokens",
            "tts_output_leveler": True,
            "tts_markup": True,
        },
        "tts": {"streaming": True, "speech_latency": "normal"},
        "turn_merge": {"enabled": True, "hold_sec": 1.2},
        "performance_mode": "current",
        "voice_model_profile": "current",
    }
    monkeypatch.setattr(
        config_module, "load_config", lambda *a, **k: deepcopy(original)
    )
    monkeypatch.setattr(config_module, "apply_device_profile", lambda cfg, *a, **k: cfg)

    def select(cfg, requested=None, **kwargs):
        if requested is None:
            return cfg, None
        assert requested == "responsive"
        return deep_merge(
            cfg,
            {
                "sherpa": {
                    "asr_encoder": "int8.onnx",
                    "asr_num_threads": 2,
                    "tts_model": "vits.onnx",
                    "tts_num_threads": 2,
                },
                "llm": {"main_keep_alive": "60s", "fast_keep_alive": -1},
            },
        ), SimpleNamespace(sha256="c" * 64)

    monkeypatch.setattr(performance, "apply_performance_mode", select)
    control, control_id = b.resolve_config(tmp_path, "cpu_laptop", "vits")
    candidate, candidate_id = b.resolve_config(tmp_path, "cpu_laptop", variant)
    assert _changed_leaves(control, candidate) == {("sherpa", field)}
    assert candidate["sherpa"][field] == requested
    assert candidate_id["selection_sha256"] != control_id["selection_sha256"]
    assert candidate_id["llm_settings_sha256"] == control_id["llm_settings_sha256"]
    assert candidate_id["safety_policy_sha256"] == control_id["safety_policy_sha256"]
    assert (
        candidate_id["asr_settings_sha256"] == control_id["asr_settings_sha256"]
    ) == (field == "tts_num_threads")
    # Existing strict pairing intentionally rejects an ASR settings change.
    before = _report()
    after = deepcopy(before)
    before["binding"]["asr_settings_sha256"] = control_id["asr_settings_sha256"]
    after["binding"]["asr_settings_sha256"] = candidate_id["asr_settings_sha256"]
    if field != "tts_num_threads":
        with pytest.raises(b.BenchmarkError):
            b.compare_paired(before, after)
    else:
        assert b.compare_paired(before, after)["attempted_pairs"] == 1


@pytest.mark.parametrize(
    "previous", [None, False, 0, 1, "CUDA", "cuda", "cpu:settings.conf", " cpu"]
)
def test_cuda_variant_refuses_nonexact_cpu_control(previous):
    with pytest.raises(b.BenchmarkError):
        b._apply_resource_variant({"sherpa": {"provider": previous}}, "vits-cuda")


def test_cuda_selection_does_not_construct_native_or_change_environment(monkeypatch):
    import os
    from core.engines.sherpa import SherpaOnnxEngine

    monkeypatch.setattr(
        SherpaOnnxEngine, "start", lambda *a, **k: pytest.fail("no native constructor")
    )
    before = dict(os.environ)
    control = {
        "sherpa": {"provider": "cpu", "asr_num_threads": 4, "tts_num_threads": 2},
        "llm": {"options": {"num_ctx": 2048, "num_predict": 256}},
    }
    result = b._apply_resource_variant(control, "vits-cuda")
    assert result["sherpa"]["provider"] == "cuda"
    assert _changed_leaves(control, result) == {("sherpa", "provider")}
    assert dict(os.environ) == before
