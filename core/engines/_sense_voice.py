"""Local CPU SenseVoice endpoint voter; no streaming/control authority."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.audio_frontend import AudioResampler
from ._endpoint_lease import EndpointModelLease
from ._faster_whisper import _mono_float32, _positive_integer_sample_rate


@dataclass(frozen=True)
class SenseVoiceResult:
    text: str = field(repr=False)
    audio_seconds: float


class SenseVoiceEndpointRecognizer:
    def __init__(self, model_path, *, cpu_threads=2, model_factory=None, resampler_factory=None, model_validator=None):
        if type(cpu_threads) is not int or not 1 <= cpu_threads <= 8:
            raise ValueError("SenseVoice CPU threads must be between 1 and 8")
        if model_validator is None:
            from ._qwen_assets import validate_sense_voice
            model_validator = validate_sense_voice
        root = Path(model_validator(Path(model_path)))
        self._root = root
        self._threads = cpu_threads
        self._factory = model_factory
        self._resampler = resampler_factory or AudioResampler
        self._endpoint_lease = EndpointModelLease(self._load_model, lambda native: None)

    def _load_model(self):
        factory = self._factory
        if factory is None:
            import sherpa_onnx
            factory = sherpa_onnx.OfflineRecognizer.from_sense_voice
        return factory(model=str(self._root / "model.int8.onnx"), tokens=str(self._root / "tokens.txt"),
                       num_threads=self._threads, provider="cpu", language="en", use_itn=True)

    def warm(self):
        self._endpoint_lease.run(lambda native: None)

    def transcribe(self, pcm, sample_rate):
        rate = _positive_integer_sample_rate(sample_rate)
        samples = _mono_float32(pcm)
        duration = samples.size / float(rate)
        if not samples.size:
            # Even empty requests cannot revive a retired model incarnation.
            self._endpoint_lease.ensure_open()
            return SenseVoiceResult("", duration)
        if rate != 16000:
            resampler = self._resampler(rate, 16000, quality="VHQ")
            samples = _mono_float32(resampler.process(samples, last=True))

        def decode(native):
            stream = native.create_stream()
            stream.accept_waveform(16000, samples)
            native.decode_stream(stream)
            text = stream.result.text
            if type(text) is not str:
                raise RuntimeError("SenseVoice returned an invalid transcript")
            return SenseVoiceResult(text.strip(), duration)

        return self._endpoint_lease.run(decode)

    def close(self):
        """Request retirement; false means exact cleanup is still pending."""
        return self._endpoint_lease.retire()
