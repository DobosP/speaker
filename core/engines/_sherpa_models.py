from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sherpa import SherpaConfig

# sherpa-onnx model builders shared by the local engine (``SherpaOnnxEngine``,
# local mic/speaker) and the remote engine (``LiveKitEngine``, a WebRTC room).
# Both turn the same :class:`SherpaConfig` into the same on-device recognizer /
# VAD / TTS objects and differ only in audio transport, so the model wiring
# lives here once. ``sherpa_onnx`` is imported lazily inside each builder so the
# runtime and test suite import without the native package installed.


def build_recognizer(c: "SherpaConfig"):
    """Streaming transducer ASR recognizer, or ``None`` if no model configured."""
    if not c.asr_encoder:
        return None
    import sherpa_onnx

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=c.asr_tokens,
        encoder=c.asr_encoder,
        decoder=c.asr_decoder,
        joiner=c.asr_joiner,
        num_threads=c.resolved_asr_threads,
        provider=c.provider,
        sample_rate=c.sample_rate,
        feature_dim=80,
        enable_endpoint_detection=True,
    )


def build_vad(c: "SherpaConfig"):
    """Silero VAD detector for endpointing / barge-in, or ``None``."""
    if not c.vad_model:
        return None
    import sherpa_onnx

    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = c.vad_model
    vad_config.sample_rate = c.sample_rate
    vad_config.num_threads = c.resolved_asr_threads
    vad_config.provider = c.provider
    return sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)


def build_keyword_spotter(c: "SherpaConfig"):
    """Streaming keyword spotter for the command fast-path, or ``None``.

    A separate, small streaming transducer (sherpa-onnx ships pretrained KWS
    models) that runs alongside the ASR recognizer and fires the moment a
    configured control phrase is heard -- the lowest-latency path to an action,
    since it never touches the LLM. Disabled (``None``) when no model is set.
    """
    if not c.kws_encoder:
        return None
    import sherpa_onnx

    return sherpa_onnx.KeywordSpotter(
        tokens=c.kws_tokens,
        encoder=c.kws_encoder,
        decoder=c.kws_decoder,
        joiner=c.kws_joiner,
        keywords_file=c.kws_keywords_file,
        num_threads=c.resolved_asr_threads,
        provider=c.provider,
        keywords_threshold=c.kws_threshold,
        keywords_score=c.kws_score,
    )


def build_tts(c: "SherpaConfig"):
    """Offline VITS TTS, or ``None`` if no model configured."""
    if not c.tts_model:
        return None
    import sherpa_onnx

    tts_config = sherpa_onnx.OfflineTtsConfig()
    tts_config.model.vits.model = c.tts_model
    tts_config.model.vits.tokens = c.tts_tokens
    if c.tts_data_dir:
        tts_config.model.vits.data_dir = c.tts_data_dir
    tts_config.model.num_threads = c.resolved_tts_threads
    tts_config.model.provider = c.provider
    return sherpa_onnx.OfflineTts(tts_config)
