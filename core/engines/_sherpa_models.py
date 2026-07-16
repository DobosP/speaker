from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sherpa import SherpaConfig


log = logging.getLogger("speaker.sherpa_models")


def create_recognizer_stream(
    recognizer,
    config: "SherpaConfig",
    *,
    hotwords: list[str] | tuple[str, ...] | None = None,
):
    """Create one streaming-ASR stream with production contextual biasing.

    Live capture and recorded replay must use the same per-stream hotword seam;
    otherwise a recording A/B silently measures an un-biased recognizer even
    though the live engine is biased. Older sherpa-onnx builds may reject the
    keyword, so retain the existing plain-stream fallback.
    """
    phrases = list(hotwords) if hotwords is not None else [
        line.strip()
        for line in (getattr(config, "asr_hotwords", "") or "").splitlines()
        if line.strip()
    ]
    if (
        phrases
        and getattr(config, "asr_decoding_method", "") == "modified_beam_search"
    ):
        try:
            return recognizer.create_stream(hotwords="\n".join(phrases))
        except TypeError:
            log.warning(
                "this sherpa-onnx build ignores per-stream hotwords; biasing disabled"
            )
    return recognizer.create_stream()

# sherpa-onnx model builders shared by the local engine (``SherpaOnnxEngine``,
# local mic/speaker) and the remote engine (``LiveKitEngine``, a WebRTC room).
# Both turn the same :class:`SherpaConfig` into the same on-device recognizer /
# VAD / TTS objects and differ only in audio transport, so the model wiring
# lives here once. ``sherpa_onnx`` is imported lazily inside each builder so the
# runtime and test suite import without the native package installed.


def build_recognizer(c: "SherpaConfig"):
    """Streaming transducer ASR recognizer, or ``None`` if no model configured.

    Beyond the model paths we pass three quality/latency levers:
    ``decoding_method`` (``modified_beam_search`` is more accurate than greedy
    and is what enables hotword biasing), the endpoint rules (``rule2`` is the
    turn-commit latency knob), and the hotword score for contextual biasing.
    Extra kwargs are filtered against ``from_transducer``'s real signature so an
    older sherpa-onnx that lacks one of them still builds instead of crashing."""
    if not c.asr_encoder:
        return None
    import sherpa_onnx

    kwargs = dict(
        tokens=c.asr_tokens,
        encoder=c.asr_encoder,
        decoder=c.asr_decoder,
        joiner=c.asr_joiner,
        num_threads=c.resolved_asr_threads,
        provider=c.provider,
        sample_rate=c.sample_rate,
        feature_dim=80,
        enable_endpoint_detection=True,
        decoding_method=c.asr_decoding_method,
        max_active_paths=c.asr_max_active_paths,
        rule1_min_trailing_silence=c.asr_rule1_min_trailing_silence,
        rule2_min_trailing_silence=c.asr_rule2_min_trailing_silence,
        rule3_min_utterance_length=c.asr_rule3_min_utterance_length,
    )
    # Contextual biasing is only honored by beam search; pass the hotword score
    # so a phrase list supplied per-stream (see SherpaOnnxEngine) is boosted.
    if c.asr_hotwords and c.asr_decoding_method == "modified_beam_search":
        kwargs["hotwords_score"] = c.asr_hotwords_score
    return sherpa_onnx.OnlineRecognizer.from_transducer(**_supported(
        sherpa_onnx.OnlineRecognizer.from_transducer, kwargs
    ))


def build_final_recognizer(c: "SherpaConfig"):
    """Optional OFFLINE second-pass recognizer for the FINAL transcript (the text
    that reaches the LLM). The streaming transducer gives low-latency partials +
    the endpoint; this re-transcribes the endpointed UTTERANCE with a stronger
    offline model that sees the WHOLE utterance at once -- far more robust on
    run-on / casual speech, with punctuation + casing + ITN built in. Measured
    2026-06-01: SenseVoice fixed the streaming garble ("HEY IRIC LISTENING TO ME"
    -> "Hey, are you listening to me.") at ~150ms/utterance.

    None unless ``asr_final_backend`` ('sense_voice' | 'whisper') is set and the
    model exists. Fail-OPEN: any build error returns None so the engine simply
    keeps the streaming final -- a bad second-pass config never breaks capture."""
    backend = (getattr(c, "asr_final_backend", "") or "").strip().lower()
    if not backend:
        return None
    import os

    model = getattr(c, "asr_final_model", "") or ""
    tokens = getattr(c, "asr_final_tokens", "") or ""
    if not model or not os.path.exists(model):
        # The backend IS configured (we passed the `if not backend` guard) but its
        # model artifact is absent -> we silently fall back to the STREAMING-only
        # final, which is much lower accuracy (the garbled-transcript symptom). Make
        # that LOUD so a missing/relative-path download isn't invisible in the run
        # bundle, instead of returning None with no trace.
        import logging

        logging.getLogger("speaker.sherpa").warning(
            "asr_final_backend=%r is set but its model is missing (asr_final_model=%r) "
            "-- using STREAMING-ONLY finals (lower accuracy). Fetch it with "
            "`python -m tools.setup_models --sense-voice`, or set sherpa.asr_final_model "
            "to an existing path.",
            backend, model or "(unset)",
        )
        return None
    try:
        import sherpa_onnx

        if backend == "sense_voice":
            kwargs = dict(
                model=model, tokens=tokens, num_threads=c.resolved_asr_threads,
                provider=c.provider, use_itn=bool(getattr(c, "asr_final_use_itn", True)),
                language=getattr(c, "asr_final_language", "") or "",
            )
            # Contextual biasing for the FINAL transcript (homophone replacement +
            # rule FSTs). Added ONLY when set so the call is byte-identical when
            # unconfigured; _supported drops them on an older sherpa build.
            for cfg_key, kw in (
                ("asr_final_hr_dict_dir", "hr_dict_dir"),
                ("asr_final_hr_lexicon", "hr_lexicon"),
                ("asr_final_hr_rule_fsts", "hr_rule_fsts"),
                ("asr_final_rule_fsts", "rule_fsts"),
            ):
                val = getattr(c, cfg_key, "") or ""
                if val:
                    kwargs[kw] = val
            return sherpa_onnx.OfflineRecognizer.from_sense_voice(**_supported(
                sherpa_onnx.OfflineRecognizer.from_sense_voice, kwargs))
        if backend == "whisper":
            kwargs = dict(
                encoder=model, decoder=getattr(c, "asr_final_decoder", "") or "",
                tokens=tokens, num_threads=c.resolved_asr_threads, provider=c.provider,
            )
            return sherpa_onnx.OfflineRecognizer.from_whisper(**_supported(
                sherpa_onnx.OfflineRecognizer.from_whisper, kwargs))
    except Exception:  # noqa: BLE001 - fail open to the streaming final
        import logging

        logging.getLogger("speaker.sherpa").warning(
            "second-pass recognizer (%s) failed to build; using the streaming final",
            backend, exc_info=True)
    return None


def _supported(fn, kwargs: dict) -> dict:
    """Drop kwargs the target callable doesn't accept.

    sherpa-onnx's ``from_transducer`` has grown parameters over releases; rather
    than pin a version we keep the ones the installed build actually declares
    (and always keep everything when it takes ``**kwargs``, e.g. the test fake)."""
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # builtin without a signature -> send all
        return kwargs
    params = sig.parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def build_punctuation(c: "SherpaConfig"):
    """Offline punctuation restorer applied to ASR finals, or ``None``.

    sherpa-onnx ships a CT-Transformer punctuation model that adds ``.,?`` to
    raw recognizer text. Applied to finals only (cheap, off the partial hot
    path). Empty ``punct_model`` -> ``None`` and the engine falls back to pure
    casing restoration."""
    if not getattr(c, "punct_model", ""):
        return None
    import sherpa_onnx

    config = sherpa_onnx.OfflinePunctuationConfig(
        model=sherpa_onnx.OfflinePunctuationModelConfig(
            ct_transformer=c.punct_model,
            num_threads=c.resolved_asr_threads,
            provider=c.provider,
        )
    )
    return sherpa_onnx.OfflinePunctuation(config)


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


def build_tts(c: "SherpaConfig", *, deterministic_vits: bool = False):
    """Offline TTS (VITS/Piper by default, Kokoro when ``tts_voices`` is set), or
    ``None`` if no model configured.

    The Kokoro family (StyleTTS2-based, many built-in voices, more natural than the
    libritts VITS) is a sibling of ``vits`` on ``OfflineTtsConfig.model``: it needs a
    ``voices.bin`` (hence keying on ``tts_voices``) plus the same tokens + espeak-ng
    ``data_dir``, and the multi-lang packages also a ``lexicon``. Everything
    downstream is family-agnostic -- ``generate(text, sid=, speed=, callback=)`` and
    ``.sample_rate`` are identical -- so voice selection stays ``tts_speaker_id`` and
    the sample rate auto-adapts (Kokoro is 24 kHz). The VITS path is byte-identical
    when ``tts_voices`` is empty (default), so this is a drop-in, opt-in addition.

    ``deterministic_vits`` is a harness-only construction mode: it zeros VITS's
    acoustic and duration noise scales so repeated renderings of one validation
    script are byte-stable. Runtime callers keep the native stochastic defaults.
    Kokoro does not expose those VITS controls, so the flag is inert there.

    Fails OPEN like ``build_final_recognizer``: a Kokoro config whose model files
    are missing (e.g. ``tts_voices`` set but the package was never fetched) is
    caught BEFORE the native constructor -- which otherwise aborts cryptically --
    and returns ``None`` with a clear, actionable warning. The engine already
    treats ``_tts is None`` as "no speech" (a mute assistant + a loud log beats a
    hard crash on the capture thread), and the doctor preflight names the fix."""
    if not c.tts_model:
        return None
    import os

    kokoro = bool(getattr(c, "tts_voices", ""))
    if kokoro:
        # Kokoro's native loader hard-aborts (not a catchable Python error) on a
        # missing model/voices/tokens file, so guard the required paths up front.
        missing = [
            p for p in (c.tts_model, c.tts_voices, c.tts_tokens)
            if p and not os.path.exists(p)
        ]
        if missing:
            import logging

            logging.getLogger("speaker.sherpa").warning(
                "Kokoro TTS is selected (tts_voices set) but required file(s) are "
                "missing: %s -- speech is DISABLED until they exist. Fetch the "
                "package (model.onnx + voices.bin + tokens.txt + espeak-ng-data) "
                "with `python -m tools.setup_models --kokoro` and point "
                "tts_model/tts_voices/tts_tokens at it, or clear tts_voices to use "
                "the Piper/VITS voice.",
                ", ".join(missing),
            )
            return None
    import sherpa_onnx

    tts_config = sherpa_onnx.OfflineTtsConfig()
    if kokoro:  # Kokoro (voices.bin present)
        k = tts_config.model.kokoro
        k.model = c.tts_model
        k.voices = c.tts_voices
        k.tokens = c.tts_tokens
        if c.tts_data_dir:
            k.data_dir = c.tts_data_dir
        if getattr(c, "tts_lexicon", ""):  # multi-lang packages ship a lexicon
            k.lexicon = c.tts_lexicon
    else:  # VITS / Piper (unchanged)
        vits = tts_config.model.vits
        vits.model = c.tts_model
        vits.tokens = c.tts_tokens
        if c.tts_data_dir:
            vits.data_dir = c.tts_data_dir
        if deterministic_vits:
            vits.noise_scale = 0.0
            vits.noise_scale_w = 0.0
    tts_config.model.num_threads = c.resolved_tts_threads
    tts_config.model.provider = c.provider
    try:
        return sherpa_onnx.OfflineTts(tts_config)
    except Exception:  # noqa: BLE001 - fail open to no-TTS with a loud, actionable log
        import logging

        logging.getLogger("speaker.sherpa").warning(
            "TTS model failed to build (%s backend) -- speech disabled; verify the "
            "model paths in the sherpa config.", "kokoro" if kokoro else "vits",
            exc_info=True,
        )
        return None
