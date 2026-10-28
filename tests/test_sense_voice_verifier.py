"""CPU voter, full PCM and exact engine retirement composition, no models/devices."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from core.engine import EngineCallbacks
from core.engines import _endpoint_lease, _sense_voice
from core.engines._sense_voice import SenseVoiceEndpointRecognizer
from core.engines._sherpa_models import build_final_verifier
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine


@pytest.fixture(autouse=True)
def isolate_quarantine(monkeypatch):
    monkeypatch.setattr(_endpoint_lease, "_UNCERTAIN_LEASES", [])
    monkeypatch.setattr(_endpoint_lease, "_RETIRED_LEASES", {})


@pytest.fixture
def directory(tmp_path):
    for name in ("model.int8.onnx", "tokens.txt"):
        (tmp_path / name).write_bytes(b"public fake")
    return tmp_path


class Native:
    def __init__(self, before_decode=None):
        self.inputs = []
        self.before_decode = before_decode
    def create_stream(self):
        parent = self
        class Stream:
            result = SimpleNamespace(text="  Public result.  ")
            def accept_waveform(self, rate, samples):
                parent.inputs.append((rate, samples.copy()))
        return Stream()
    def decode_stream(self, stream):
        if self.before_decode is not None:
            self.before_decode()


def test_cpu_constructor_once_full_pcm_no_silent_thirty_second_crop(directory):
    calls, native = [], Native()
    def factory(**kwargs):
        calls.append(kwargs)
        return native
    voter = SenseVoiceEndpointRecognizer(directory, model_validator=lambda path: path, cpu_threads=2, model_factory=factory)
    voter.warm()
    voter.warm()
    samples = np.ones(31 * 16000, dtype=np.float32)
    result = voter.transcribe(samples, 16000)
    assert len(calls) == 1
    assert calls[0] == {"model": str(directory / "model.int8.onnx"), "tokens": str(directory / "tokens.txt"),
                        "provider": "cpu", "num_threads": 2, "language": "en", "use_itn": True}
    assert native.inputs[0][1].shape == samples.shape
    assert result.text == "Public result." and result.audio_seconds == 31.
    assert result.text not in repr(result)
    voter.close()
    assert voter._endpoint_lease.wait_closed(1)


def test_empty_or_invalid_input_never_constructs_model(directory):
    calls = []
    voter = SenseVoiceEndpointRecognizer(directory, model_validator=lambda path: path, model_factory=lambda **kw: calls.append(kw))
    assert voter.transcribe(np.array([], dtype=np.float32), 16000).text == ""
    for value in (np.array([np.nan], dtype=np.float32), np.ones((2, 2), dtype=np.float32), np.ones(10, dtype=np.int16)):
        with pytest.raises(ValueError):
            voter.transcribe(value, 16000)
    assert calls == []
    assert voter.close()
    with pytest.raises(_endpoint_lease.EndpointModelRetired):
        voter.transcribe(np.array([], dtype=np.float32), 16000)


def test_resampler_flushes_all_samples_without_context_hints(directory):
    native = Native()
    events = []
    class Resampler:
        def __init__(self, source, target, quality):
            events.append((source, target, quality))
        def process(self, samples, last):
            assert last is True
            return np.repeat(samples, 2)
    voter = SenseVoiceEndpointRecognizer(directory, model_validator=lambda path: path, model_factory=lambda **kw: native, resampler_factory=Resampler)
    result = voter.transcribe(np.ones(8000, dtype=np.float32), 8000)
    assert events == [(8000, 16000, "VHQ")]
    assert native.inputs[0][0] == 16000 and native.inputs[0][1].size == 16000
    assert result.audio_seconds == 1.
    voter.close()
    assert voter._endpoint_lease.wait_closed(1)


def test_factory_selects_local_cpu_voter_and_rejects_duplicate_primary(directory, monkeypatch):
    calls = []
    class Voter:
        def __init__(self, path, **options):
            calls.append((path, options))
        def warm(self):
            calls.append("warm")
    monkeypatch.setattr(_sense_voice, "SenseVoiceEndpointRecognizer", Voter)
    config = SherpaConfig(asr_final_backend="nemo_transducer", asr_final_verifier_backend="sense_voice",
                          asr_final_verifier_model=str(directory), asr_final_verifier_cpu_threads=2,
                          asr_final_required=True, provider="cuda")
    assert isinstance(build_final_verifier(config), Voter)
    assert calls == [(str(directory), {"cpu_threads": 2}), "warm"]
    config.asr_final_backend = "sense_voice"
    with pytest.raises(RuntimeError, match="required sense_voice"):
        build_final_verifier(config)
    assert len(calls) == 2


def test_runtime_stop_retires_blocked_background_warm_and_old_error_cannot_disable_new(directory):
    entered, finish = threading.Event(), threading.Event()
    def factory(**kwargs):
        entered.set()
        assert finish.wait(2)
        return Native()
    voter = SenseVoiceEndpointRecognizer(directory, model_validator=lambda path: path, model_factory=factory)
    engine = SherpaOnnxEngine(SherpaConfig())
    engine._final_verifier = voter
    thread = threading.Thread(target=engine.warm)
    thread.start()
    assert entered.wait(1)
    engine.stop()
    assert not voter._endpoint_lease.closed
    with pytest.raises(RuntimeError, match="pending"):
        _endpoint_lease.require_endpoint_models_retired(engine)
    successor = object()
    engine._final_verifier = successor
    finish.set()
    thread.join(2)
    assert not thread.is_alive()
    assert voter._endpoint_lease.closed
    assert engine._final_verifier is successor


def test_runtime_stop_covers_inline_decode_even_without_final_thread(directory):
    entered, finish = threading.Event(), threading.Event()
    def decode():
        entered.set()
        assert finish.wait(2)
    voter = SenseVoiceEndpointRecognizer(directory, model_validator=lambda path: path, model_factory=lambda **kwargs: Native(decode))
    engine = SherpaOnnxEngine(SherpaConfig())
    engine._final_verifier = voter
    thread = threading.Thread(target=lambda: voter.transcribe(np.ones(16000, dtype=np.float32), 16000))
    thread.start()
    assert entered.wait(1)
    engine.stop()
    assert not voter._endpoint_lease.closed
    finish.set()
    thread.join(2)
    assert not thread.is_alive() and voter._endpoint_lease.closed
    _endpoint_lease.require_endpoint_models_retired(engine)


def test_replay_restart_waits_exact_old_cleanup_then_builds_fresh(directory, monkeypatch):
    from core.engines import file_replay as replay
    started, finish = threading.Event(), threading.Event()
    def release(native):
        started.set()
        assert finish.wait(2)
    lease = _endpoint_lease.EndpointModelLease(object, release)
    lease.run(lambda native: None)
    old = SimpleNamespace(_endpoint_lease=lease)
    engine = replay.FileReplayEngine(SherpaConfig(), asr_only=True)
    engine._final_verifier = old
    engine.stop()
    assert started.wait(1)
    builds = []
    monkeypatch.setattr(replay, "build_recognizer", lambda cfg: builds.append("build") or object())
    monkeypatch.setattr(replay, "build_final_recognizer", lambda cfg: None)
    monkeypatch.setattr(replay, "build_final_verifier", lambda cfg: None)
    monkeypatch.setattr(replay, "build_punctuation", lambda cfg: None)
    monkeypatch.setattr(replay, "create_recognizer_stream", lambda *args: object())
    with pytest.raises(RuntimeError, match="pending"):
        engine.start(EngineCallbacks())
    assert builds == []
    finish.set()
    assert lease.wait_closed(1)
    engine.start(EngineCallbacks())
    assert builds == ["build"] and engine._final_verifier is None
    engine.stop()


def test_cpu_readiness_never_loads_cuda(directory):
    from core.readiness import check_asr_final_verifier_runtime
    events = []
    config = {"sherpa": {"asr_final_backend": "nemo_transducer", "asr_final_verifier_backend": "sense_voice",
                         "asr_final_verifier_model": str(directory), "asr_final_verifier_cpu_threads": 2}}
    def forbid_cuda():
        raise AssertionError("CUDA bootstrap must not be called")
    check = check_asr_final_verifier_runtime(config, import_fn=lambda name: events.append(name),
                                           preload_fn=forbid_cuda, model_probe_fn=lambda path: events.append(path))
    assert check.ok and events == ["sherpa_onnx", str(directory)]
    config["sherpa"]["asr_final_backend"] = "sense_voice"
    assert not check_asr_final_verifier_runtime(config, import_fn=lambda name: None, model_probe_fn=lambda path: None).ok



@pytest.mark.parametrize("failure_stage", ["later_builder", "capture_warm", "pre_device_validation"])
def test_public_start_failure_retires_only_exact_new_models_without_masking_error(failure_stage):
    engine = SherpaOnnxEngine(SherpaConfig())
    freed = threading.Event()
    lease = _endpoint_lease.EndpointModelLease(object, lambda native: freed.set())
    lease.run(lambda native: None)
    model = SimpleNamespace(_endpoint_lease=lease)
    failure = ValueError(failure_stage)
    def start(callbacks):
        engine._final_recognizer = model
        raise failure
    engine._start_impl = start
    with pytest.raises(ValueError) as caught:
        engine.start(EngineCallbacks())
    assert caught.value is failure
    assert freed.wait(1) and lease.wait_closed(1)
    _endpoint_lease.require_endpoint_models_retired(engine)


def test_required_voter_warm_failure_retires_partial_leased_native(directory, monkeypatch):
    freed = threading.Event()
    lease = _endpoint_lease.EndpointModelLease(object, lambda native: freed.set())
    class Voter:
        _endpoint_lease = lease
        def __init__(self, *args, **kwargs):
            pass
        def warm(self):
            lease.run(lambda native: (_ for _ in ()).throw(ValueError("public fixture")))
    monkeypatch.setattr(_sense_voice, "SenseVoiceEndpointRecognizer", Voter)
    config = SherpaConfig(asr_final_backend="qwen_cpu", asr_final_verifier_backend="sense_voice",
                          asr_final_verifier_model=str(directory), asr_final_verifier_cpu_threads=2,
                          asr_final_required=True)
    with pytest.raises(RuntimeError, match="required sense_voice"):
        build_final_verifier(config)
    assert freed.wait(1) and lease.wait_closed(1)



def test_duplicate_start_rejection_cannot_retire_existing_healthy_model():
    engine = SherpaOnnxEngine(SherpaConfig())
    calls = []
    lease = _endpoint_lease.EndpointModelLease(object, lambda native: calls.append("closed"))
    lease.run(lambda native: calls.append("loaded"))
    model = SimpleNamespace(_endpoint_lease=lease)
    engine._final_recognizer = model
    def early_rejection(callbacks):
        raise RuntimeError("already started")
    engine._start_impl = early_rejection
    with pytest.raises(RuntimeError, match="already started"):
        engine.start(EngineCallbacks())
    lease.run(lambda native: calls.append("still usable"))
    assert calls == ["loaded", "still usable"] and not lease.closed
    lease.retire()
    assert lease.wait_closed(1)
