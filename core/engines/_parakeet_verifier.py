"""Pinned Parakeet TDT v3 CPU voter; same bounded endpoint lease as SenseVoice."""
from __future__ import annotations

from ._sense_voice import SenseVoiceEndpointRecognizer


class ParakeetEndpointRecognizer(SenseVoiceEndpointRecognizer):
    """A fixed different model family, not an extra SenseVoice acoustic vote."""
    def __init__(self, model_path, *, cpu_threads=2, model_factory=None,
                 resampler_factory=None, model_validator=None):
        if model_validator is None:
            from ._qwen_assets import validate_parakeet_tdt
            model_validator = validate_parakeet_tdt
        super().__init__(model_path, cpu_threads=cpu_threads, model_factory=model_factory,
                         resampler_factory=resampler_factory, model_validator=model_validator)

    def _load_model(self):
        factory = self._factory
        if factory is None:
            import sherpa_onnx
            factory = sherpa_onnx.OfflineRecognizer.from_transducer
        return factory(encoder=str(self._root / "encoder.int8.onnx"),
                       decoder=str(self._root / "decoder.int8.onnx"),
                       joiner=str(self._root / "joiner.int8.onnx"),
                       tokens=str(self._root / "tokens.txt"), provider="cpu",
                       num_threads=self._threads, sample_rate=16000, feature_dim=80,
                       decoding_method="greedy_search", max_active_paths=4,
                       model_type="nemo_transducer")
