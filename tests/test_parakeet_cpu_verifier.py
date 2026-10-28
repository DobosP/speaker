"""Fixed CPU TDT voter and retained shared lease/full-PCM contracts, no models."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from core.engines import _endpoint_lease, _parakeet_verifier
from core.engines._parakeet_verifier import ParakeetEndpointRecognizer
from core.engines._sherpa_models import build_final_verifier
from core.engines.sherpa import SherpaConfig


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(_endpoint_lease, "_UNCERTAIN_LEASES", [])
    monkeypatch.setattr(_endpoint_lease, "_RETIRED_LEASES", {})


class Native:
    def __init__(self, block=None):
        self.block = block
        self.inputs = []
    def create_stream(self):
        outer = self
        class Stream:
            result = SimpleNamespace(text=" A public result. ")
            def accept_waveform(self, rate, samples):
                outer.inputs.append((rate, samples.copy()))
        return Stream()
    def decode_stream(self, stream):
        if self.block:
            self.block()


def test_exact_pinned_constructor_cpu_two_full_pcm_and_repr_safe_result(tmp_path):
    calls, native = [], Native()
    def factory(**kw):
        calls.append(kw)
        return native
    voter = ParakeetEndpointRecognizer(tmp_path, model_factory=factory, model_validator=lambda p: p)
    voter.warm()
    voter.warm()
    pcm = np.ones(32 * 16000, dtype=np.float32)
    result = voter.transcribe(pcm, 16000)
    assert calls == [{"encoder": str(tmp_path / "encoder.int8.onnx"), "decoder": str(tmp_path / "decoder.int8.onnx"),
                      "joiner": str(tmp_path / "joiner.int8.onnx"), "tokens": str(tmp_path / "tokens.txt"),
                      "provider": "cpu", "num_threads": 2, "sample_rate": 16000, "feature_dim": 80,
                      "decoding_method": "greedy_search", "max_active_paths": 4, "model_type": "nemo_transducer"}]
    assert native.inputs[0][0] == 16000
    np.testing.assert_array_equal(native.inputs[0][1], pcm)
    assert result.text == "A public result." and result.text not in repr(result)
    assert result.audio_seconds == 32.
    voter.close()
    assert voter._endpoint_lease.wait_closed(1)


def test_fixed_asset_validation_happens_before_native_constructor(tmp_path):
    entered = []
    def invalid(path):
        raise ValueError("public pin mismatch")
    with pytest.raises(ValueError):
        ParakeetEndpointRecognizer(tmp_path, model_factory=lambda **kw: entered.append(kw), model_validator=invalid)
    assert entered == []


def test_retirement_rejects_successor_while_entered_decode_finishes(tmp_path):
    entered, finish = threading.Event(), threading.Event()
    def decode():
        entered.set()
        assert finish.wait(2)
    voter = ParakeetEndpointRecognizer(tmp_path, model_factory=lambda **kw: Native(decode), model_validator=lambda p: p)
    thread = threading.Thread(target=lambda: voter.transcribe(np.ones(16000, dtype=np.float32), 16000))
    thread.start()
    assert entered.wait(1)
    assert voter.close() is False
    with pytest.raises(_endpoint_lease.EndpointModelRetired):
        voter.warm()
    assert not voter._endpoint_lease.closed
    finish.set()
    thread.join(2)
    assert not thread.is_alive() and voter._endpoint_lease.closed


def test_factory_requires_distinct_primary_and_warms_fixed_cpu_voter(monkeypatch, tmp_path):
    calls = []
    class Voter:
        def __init__(self, path, **kwargs):
            calls.append((path, kwargs))
        def warm(self):
            calls.append("warm")
    monkeypatch.setattr(_parakeet_verifier, "ParakeetEndpointRecognizer", Voter)
    cfg = SherpaConfig(asr_final_backend="qwen_cpu", asr_final_verifier_backend="parakeet_tdt_cpu",
                       asr_final_verifier_model=str(tmp_path), asr_final_verifier_cpu_threads=2,
                       asr_final_required=True, provider="cuda")
    assert isinstance(build_final_verifier(cfg), Voter)
    assert calls == [(str(tmp_path), {"cpu_threads": 2}), "warm"]
    cfg.asr_final_backend = "nemo_transducer"
    with pytest.raises(RuntimeError, match="required parakeet_tdt_cpu"):
        build_final_verifier(cfg)
    assert len(calls) == 2


def test_readiness_cpu_voter_never_touches_cuda_and_rejects_duplicate(monkeypatch, tmp_path):
    from core.readiness import check_asr_final_verifier_runtime
    events = []
    cfg = {"sherpa": {"asr_final_backend": "qwen_cpu", "asr_final_verifier_backend": "parakeet_tdt_cpu",
                      "asr_final_verifier_model": str(tmp_path), "asr_final_verifier_cpu_threads": 2}}
    def no_cuda():
        raise AssertionError("CPU voter cannot bootstrap CUDA")
    checked = check_asr_final_verifier_runtime(cfg, import_fn=lambda name: events.append(name), preload_fn=no_cuda,
                                             model_probe_fn=lambda path: events.append(path))
    assert checked.ok and events == ["sherpa_onnx", str(tmp_path)]
    cfg["sherpa"]["asr_final_backend"] = "nemo_transducer"
    assert not check_asr_final_verifier_runtime(cfg, import_fn=lambda name: None,
                                              model_probe_fn=lambda path: None).ok
