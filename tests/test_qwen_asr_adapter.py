"""Pinned C-API public fakes and offline full-PCM/lease contracts, no models."""
from types import SimpleNamespace
import ctypes as C
import threading

import numpy as np
import pytest

from core.engines import _endpoint_lease as leases
from core.engines import _qwen_asr as adapter
from core.engines._qwen_asr import QwenNative, QwenOfflineRecognizer, QwenAsrError


@pytest.fixture(autouse=True)
def isolate_registry(monkeypatch):
    monkeypatch.setattr(leases, "_RETIRED_LEASES", {})
    monkeypatch.setattr(leases, "_UNCERTAIN_LEASES", [])


class Function:
    def __init__(self, fn):
        self.fn = fn
    def __call__(self, *args):
        return self.fn(*args)


class Library:
    def __init__(self, *, fail=None, free_error=None, result_stop=0):
        self.events = []
        self.fail = fail
        self.free_error = free_error
        self.stop = result_stop
        self.count = None
        self.audio = None
        def handler(name):
            def call(*args):
                self.events.append((name, args))
                if name == free_error:
                    raise TypeError("private native exception")
                if name in {"speech_load_params_new", "speech_model_get_info", "speech_request_new"}:
                    C.cast(args[-1], C.POINTER(C.c_void_p))[0] = {"speech_load_params_new": 1, "speech_model_get_info": 3, "speech_request_new": 4}[name]
                if name == "speech_model_load":
                    C.cast(args[-1], C.POINTER(C.c_void_p))[0] = 2
                if name == fail:
                    return 9
                if name == "speech_model_info_threads":
                    return 2
                if name == "speech_model_info_device":
                    return b"CPU"
                if name == "speech_request_set_audio":
                    self.count = int(args[2])
                    self.audio = np.ctypeslib.as_array(args[1], shape=(self.count,)).copy()
                if name == "speech_request_result":
                    return 5
                if name == "speech_result_stop":
                    return self.stop
                if name == "speech_result_text":
                    return b"  Public response.  "
                if name.endswith("_free") or name == "speech_log_set":
                    return None
                return 0
            return call
        for name in ("speech_log_set", "speech_load_params_new", "speech_load_params_free",
                     "speech_load_params_set_device", "speech_load_params_set_threads", "speech_load_params_set_warmup",
                     "speech_model_load", "speech_model_free", "speech_model_get_info", "speech_model_info_free",
                     "speech_model_info_threads", "speech_model_info_device", "speech_request_new", "speech_request_free",
                     "speech_request_set_audio", "speech_request_set_string", "speech_transcribe", "speech_request_result",
                     "speech_result_stop", "speech_result_text"):
            setattr(self, name, Function(handler(name)))


def native(lib):
    loaded = []
    def loader(path, **kwargs):
        loaded.append((path, kwargs))
        return lib
    obj = QwenNative("/public/model.gguf", "/public/libspeech.so", library_loader=loader)
    assert loaded == [("/public/libspeech.so", {"mode": getattr(C, "RTLD_LOCAL", 0)})]
    return obj


def test_native_exact_cpu_controls_full_audio_and_owned_request_result():
    lib = Library()
    obj = native(lib)
    assert not any(name == "speech_model_load" for name, _ in lib.events)
    obj.warm()
    obj.warm()
    samples = np.linspace(-.5, .5, 31 * 16000, dtype=np.float32)
    assert obj.transcribe(samples) == "Public response."
    assert lib.count == len(samples) > 480000
    np.testing.assert_array_equal(lib.audio, samples)
    by_name = {name: args for name, args in lib.events}
    assert by_name["speech_load_params_set_device"][1] == b"cpu"
    assert by_name["speech_load_params_set_threads"][1] == 2
    assert by_name["speech_load_params_set_warmup"][1] == 0
    assert by_name["speech_request_set_string"][1:] == (1, b"en")
    names = [name for name, _ in lib.events]
    assert names.count("speech_model_load") == 1 and names.count("speech_transcribe") == 1
    assert names.index("speech_result_text") < names.index("speech_request_free")
    obj.close()
    obj.close()
    assert [name for name, _ in lib.events].count("speech_model_free") == 1


def test_model_load_error_keeps_partial_model_for_one_exact_cleanup():
    lib = Library(fail="speech_model_load")
    obj = native(lib)
    with pytest.raises(QwenAsrError):
        obj.warm()
    assert obj.model.value == 2
    with pytest.raises(QwenAsrError):
        obj.warm()
    assert [name for name, _ in lib.events].count("speech_model_load") == 1
    obj.close()
    names = [name for name, _ in lib.events]
    assert names.count("speech_load_params_free") == names.count("speech_model_free") == 1


@pytest.mark.parametrize("method", ["speech_request_new", "speech_request_set_audio", "speech_request_set_string", "speech_transcribe"])
def test_entered_request_error_never_retries_and_frees_exact_request(method):
    lib = Library(fail=method)
    obj = native(lib)
    with pytest.raises(QwenAsrError):
        obj.transcribe(np.ones(16000, dtype=np.float32))
    names = [name for name, _ in lib.events]
    assert names.count(method) == 1
    assert names.count("speech_request_free") == 1
    obj.close()


def test_noncomplete_result_is_not_admitted():
    lib = Library(result_stop=1)
    obj = native(lib)
    with pytest.raises(QwenAsrError, match="complete"):
        obj.transcribe(np.ones(16000, dtype=np.float32))
    assert [name for name, _ in lib.events].count("speech_request_free") == 1
    obj.close()


def test_uncertain_request_free_retains_model_and_quarantines_lease_without_retry():
    lib = Library(free_error="speech_request_free")
    obj = native(lib)
    lease = leases.EndpointModelLease(lambda: obj, lambda value: value.close())
    with pytest.raises(QwenAsrError, match="uncertain"):
        lease.run(lambda value: value.transcribe(np.ones(16000, dtype=np.float32)))
    lease.retire()
    lease._cleanup_thread.join(1)
    assert not lease.closed
    assert [name for name, _ in lib.events].count("speech_request_free") == 1
    assert not any(name == "speech_model_free" for name, _ in lib.events)
    with pytest.raises(RuntimeError, match="uncertain"):
        leases.EndpointModelLease(object, lambda value: None)


class FakeNative:
    def __init__(self, **kwargs):
        self.calls, self.closed = [], 0
    def warm(self):
        self.calls.append("warm")
    def transcribe(self, samples):
        self.calls.append(samples.copy())
        return " Public text. "
    def close(self):
        self.closed += 1


def facade():
    calls, obj = [], FakeNative()
    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return obj
    voter = QwenOfflineRecognizer("/public/bundle", bundle_validator=lambda p: ("/public/model.gguf", "/public/libspeech.so"), native_factory=factory)
    return voter, obj, calls


def test_offline_facade_owns_all_chunks_and_has_no_thirty_second_limit():
    voter, obj, calls = facade()
    a = np.ones(16 * 16000, dtype=np.float32)
    b = np.full(16 * 16000, .25, dtype=np.float32)
    stream = voter.create_stream()
    stream.accept_waveform(16000, a)
    stream.accept_waveform(16000, b)
    a[:] = 0
    voter.decode_stream(stream)
    assert stream.result.text == "Public text." and "Public text." not in repr(stream.result)
    assert len(obj.calls[0]) == 32 * 16000
    assert obj.calls[0][0] == 1 and obj.calls[0][-1] == .25
    assert stream.chunks == []
    assert calls == [(("/public/model.gguf", "/public/libspeech.so"), {"cpu_threads": 2})]
    with pytest.raises(ValueError, match="already submitted"):
        voter.decode_stream(stream)
    with pytest.raises(ValueError):
        stream.accept_waveform(16000, b)
    voter.close()
    assert voter._endpoint_lease.wait_closed(1) and obj.closed == 1


def test_offline_stream_refuses_foreign_owner_and_invalid_audio():
    voter, _, _ = facade()
    foreign, _, _ = facade()
    stream = voter.create_stream()
    with pytest.raises(ValueError):
        foreign.decode_stream(stream)
    with pytest.raises(ValueError):
        stream.accept_waveform(16000, np.array([np.nan], dtype=np.float32))
    voter.close()
    assert voter._endpoint_lease.closed
    foreign.close()


def test_queued_stream_cannot_enter_native_after_stop():
    voter, obj, _ = facade()
    stream = voter.create_stream()
    stream.accept_waveform(16000, np.ones(16000, dtype=np.float32))
    assert voter.close()
    with pytest.raises(leases.EndpointModelRetired):
        voter.decode_stream(stream)
    assert obj.calls == [] and stream.chunks == []


def test_qwen_factory_and_readiness_share_real_facade_and_no_cuda(tmp_path, monkeypatch):
    from core.engines._sherpa_models import build_final_recognizer
    from core.engines.sherpa import SherpaConfig
    from core.readiness import check_asr_final_runtime
    calls = []
    class Model:
        def __init__(self, path, **options):
            calls.append((path, options))
        def warm(self):
            calls.append("warm")
    monkeypatch.setattr(adapter, "QwenOfflineRecognizer", Model)
    config = SherpaConfig(asr_final_backend="qwen_cpu", asr_final_model=str(tmp_path),
                          asr_final_required=True, asr_num_threads=2, provider="cuda")
    assert isinstance(build_final_recognizer(config), Model)
    assert calls == [(str(tmp_path), {"cpu_threads": 2}), "warm"]
    seen = []
    checked = check_asr_final_runtime({"sherpa": config.__dict__},
                                     import_fn=lambda name: seen.append(name),
                                     model_probe_fn=lambda selected: seen.append(selected["asr_final_backend"]))
    assert checked.ok and seen == ["qwen_cpu"]
