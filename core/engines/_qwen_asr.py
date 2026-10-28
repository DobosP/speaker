"""Local CPU Qwen3-ASR offline facade; streaming/control authority stays external."""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass, field
from pathlib import Path
import threading

from core.audio_frontend import AudioResampler
from ._endpoint_lease import EndpointModelLease
from ._faster_whisper import _mono_float32, _positive_integer_sample_rate

P = C.c_void_p
FP = C.POINTER(C.c_float)


class QwenAsrError(RuntimeError):
    """Fixed, transcript-free native adapter failure."""


def _ok(value):
    if value != 0:
        raise QwenAsrError("local Qwen ASR call failed")


def _text(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    if type(value) is not str or len(value) > 1024**2:
        raise QwenAsrError("local Qwen ASR result is invalid")
    return value.strip()


class QwenNative:
    """Pinned speech.cpp C API subset, identical CPU/math controls to the benchmark.

    Model allocation happens in warm(), after the endpoint lease owns this
    holder, so partial model/info/parameter handles always have a cleanup owner.
    """
    def __init__(self, model, library, *, cpu_threads=2, library_loader=C.CDLL):
        self.model_path = Path(model)
        self.threads = cpu_threads
        self.lib = library_loader(str(library), mode=getattr(C, "RTLD_LOCAL", 0))
        self.model = P()
        self.params = P()
        self.info = P()
        self.request = P()
        self._loaded = False
        self._failed = False
        self._uncertain = False
        self._uncertain_handles = []
        for name, result, args in [
            ("speech_log_set", None, (P, P)),
            ("speech_load_params_new", C.c_int, (C.POINTER(P),)),
            ("speech_load_params_free", None, (P,)),
            ("speech_load_params_set_device", C.c_int, (P, C.c_char_p)),
            ("speech_load_params_set_threads", C.c_int, (P, C.c_int)),
            ("speech_load_params_set_warmup", C.c_int, (P, C.c_int)),
            ("speech_model_load", C.c_int, (C.c_char_p, P, C.POINTER(P))),
            ("speech_model_free", None, (P,)),
            ("speech_model_get_info", C.c_int, (P, C.POINTER(P))),
            ("speech_model_info_free", None, (P,)),
            ("speech_model_info_threads", C.c_int, (P,)),
            ("speech_model_info_device", C.c_char_p, (P,)),
            ("speech_request_new", C.c_int, (P, C.POINTER(P))),
            ("speech_request_free", None, (P,)),
            ("speech_request_set_audio", C.c_int, (P, FP, C.c_size_t, C.c_int)),
            ("speech_request_set_string", C.c_int, (P, C.c_int, C.c_char_p)),
            ("speech_transcribe", C.c_int, (P,)),
            ("speech_request_result", P, (P,)),
            ("speech_result_stop", C.c_int, (P,)),
            ("speech_result_text", C.c_char_p, (P,)),
        ]:
            native = getattr(self.lib, name)
            native.restype, native.argtypes = result, list(args)
        self.lib.speech_log_set(None, None)

    def _release(self, attribute, function):
        handle = getattr(self, attribute)
        if not handle:
            return
        setattr(self, attribute, P())  # Detach before an entered free; never retry it.
        try:
            function(handle)
        except BaseException:
            self._uncertain = True
            self._uncertain_handles.append(handle)
            raise QwenAsrError("local Qwen ASR cleanup is uncertain") from None

    def warm(self):
        if self._uncertain or self._failed:
            raise QwenAsrError("local Qwen ASR model is unavailable")
        if self._loaded:
            return
        lib = self.lib
        try:
            _ok(lib.speech_load_params_new(C.byref(self.params)))
            if not self.params:
                raise QwenAsrError("local Qwen ASR parameters are invalid")
            _ok(lib.speech_load_params_set_device(self.params, b"cpu"))
            _ok(lib.speech_load_params_set_threads(self.params, self.threads))
            _ok(lib.speech_load_params_set_warmup(self.params, 0))
            _ok(lib.speech_model_load(str(self.model_path).encode(), self.params, C.byref(self.model)))
            if not self.model:
                raise QwenAsrError("local Qwen ASR model is invalid")
            _ok(lib.speech_model_get_info(self.model, C.byref(self.info)))
            if not self.info or lib.speech_model_info_threads(self.info) != self.threads or lib.speech_model_info_device(self.info) not in (b"CPU", b"cpu"):
                raise QwenAsrError("local Qwen ASR CPU settings were not applied")
            self._loaded = True
        except BaseException:
            self._failed = True
            raise
        finally:
            if not self._uncertain:
                self._release("info", lib.speech_model_info_free)
                self._release("params", lib.speech_load_params_free)

    def transcribe(self, samples):
        self.warm()
        lib = self.lib
        try:
            _ok(lib.speech_request_new(self.model, C.byref(self.request)))
            if not self.request:
                raise QwenAsrError("local Qwen ASR request is invalid")
            # The API copies the entire mono PCM. Its long-input chunking stays
            # native; the research-only 480000-sample bound is not a product cap.
            _ok(lib.speech_request_set_audio(self.request, samples.ctypes.data_as(FP), samples.size, 16000))
            _ok(lib.speech_request_set_string(self.request, 1, b"en"))
            _ok(lib.speech_transcribe(self.request))
            result = lib.speech_request_result(self.request)
            if not result or lib.speech_result_stop(result) != 0:
                raise QwenAsrError("local Qwen ASR did not return a complete result")
            return _text(lib.speech_result_text(result))
        finally:
            if not self._uncertain:
                self._release("request", lib.speech_request_free)

    def close(self):
        if self._uncertain:
            raise QwenAsrError("local Qwen ASR cleanup is uncertain")
        self._release("request", self.lib.speech_request_free)
        self._release("info", self.lib.speech_model_info_free)
        self._release("params", self.lib.speech_load_params_free)
        self._release("model", self.lib.speech_model_free)
        self._loaded = False


@dataclass(frozen=True)
class QwenResult:
    text: str = field(repr=False)


class _QwenStream:
    def __init__(self, owner):
        self.owner = owner
        self.chunks = []
        self.rate = None
        self.submitted = False
        self.lock = threading.Lock()
        self._result = None

    def accept_waveform(self, sample_rate, pcm):
        rate = _positive_integer_sample_rate(sample_rate)
        samples = _mono_float32(pcm)
        with self.lock:
            if self.submitted or self.rate not in (None, rate):
                raise ValueError("Qwen offline stream no longer accepts this waveform")
            self.rate = rate
            self.chunks.append(samples.copy())

    @property
    def result(self):
        if self._result is None:
            raise QwenAsrError("local Qwen ASR result is not ready")
        return self._result


class QwenOfflineRecognizer:
    def __init__(self, bundle_dir, *, cpu_threads=2, bundle_validator=None, native_factory=None, resampler_factory=None):
        if type(cpu_threads) is not int or not 1 <= cpu_threads <= 8:
            raise ValueError("Qwen CPU threads must be between 1 and 8")
        if bundle_validator is None:
            from ._qwen_assets import validate_qwen_bundle
            bundle_validator = validate_qwen_bundle
        model, library = bundle_validator(bundle_dir)
        factory = native_factory or QwenNative
        self._endpoint_lease = EndpointModelLease(
            lambda: factory(model, library, cpu_threads=cpu_threads), lambda native: native.close())
        self._resampler = resampler_factory or AudioResampler

    def warm(self):
        self._endpoint_lease.run(lambda native: native.warm())

    def create_stream(self):
        self._endpoint_lease.ensure_open()
        return _QwenStream(self)

    def decode_stream(self, stream):
        if not isinstance(stream, _QwenStream) or stream.owner is not self:
            raise ValueError("Qwen offline stream belongs to another recognizer")
        with stream.lock:
            if stream.submitted:
                raise ValueError("Qwen offline stream was already submitted")
            stream.submitted = True
            try:
                import numpy as np
                samples = np.concatenate(stream.chunks) if stream.chunks else np.zeros(0, dtype=np.float32)
                if not samples.size:
                    self._endpoint_lease.ensure_open()
                    stream._result = QwenResult("")
                    return
                if stream.rate != 16000:
                    converter = self._resampler(stream.rate, 16000, quality="VHQ")
                    samples = _mono_float32(converter.process(samples, last=True))
                value = self._endpoint_lease.run(lambda native: native.transcribe(samples))
                stream._result = QwenResult(_text(value))
            finally:
                stream.chunks.clear()

    def close(self):
        return self._endpoint_lease.retire()
