from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import stat
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sherpa import SherpaConfig


log = logging.getLogger("speaker.sherpa_models")
_MAX_BPE_VOCAB_BYTES = 4 * 1024 * 1024
_UPPER_ASCII_WORDS = re.compile(r"[A-Z]+(?: [A-Z]+)*")


def validate_bpe_vocab_file(path: str, *, expected_sha256: str = "") -> str:
    """Validate and hash the small two-column vocabulary consumed by Sherpa."""
    expected = str(expected_sha256 or "").strip()
    if expected and re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("BPE vocabulary SHA-256 must be 64 lowercase hex characters")
    try:
        with open(path, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("BPE streaming hotword vocabulary is not a regular file")
            if before.st_size <= 0:
                raise ValueError("BPE streaming hotword vocabulary is empty")
            if before.st_size > _MAX_BPE_VOCAB_BYTES:
                raise ValueError("BPE streaming hotword vocabulary is unexpectedly large")
            payload = handle.read(_MAX_BPE_VOCAB_BYTES + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ValueError("BPE streaming hotword vocabulary is unreadable") from exc
    if (
        len(payload) != before.st_size
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ValueError("BPE streaming hotword vocabulary changed while reading")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("BPE streaming hotword vocabulary is not UTF-8") from exc
    pieces: set[str] = set()
    for line in text.splitlines():
        columns = line.split("\t")
        if len(columns) != 2 or not columns[0]:
            raise ValueError("BPE streaming hotword vocabulary is malformed")
        if columns[0] in pieces:
            raise ValueError("BPE streaming hotword vocabulary has duplicate pieces")
        pieces.add(columns[0])
        try:
            score = float(columns[1])
        except ValueError as exc:
            raise ValueError("BPE streaming hotword vocabulary score is invalid") from exc
        if not math.isfinite(score):
            raise ValueError("BPE streaming hotword vocabulary score is non-finite")
    if not pieces:
        raise ValueError("BPE streaming hotword vocabulary has no pieces")
    digest = hashlib.sha256(payload).hexdigest()
    if expected and digest != expected:
        raise ValueError("BPE streaming hotword vocabulary checksum mismatch")
    return digest


def _streaming_hotword_context_kwargs(config: "SherpaConfig") -> dict[str, str]:
    """Return the model-specific context needed to encode active hotwords.

    sherpa-onnx defaults ``modeling_unit`` to ``cjkchar``.  That default cannot
    encode phrases for an English BPE Zipformer, so accepting an empty unit here
    would make an explicitly selected hotword candidate look enabled while doing
    nothing.  Keep inactive/default configurations byte-identical and fail an
    active, incomplete candidate before the native recognizer is constructed.
    """
    phrases = [
        line.strip()
        for line in str(getattr(config, "asr_hotwords", "") or "").splitlines()
        if line.strip()
    ]
    if not phrases:
        return {}
    _validate_streaming_hotword_phrases(config, phrases)
    if getattr(config, "asr_decoding_method", "") != "modified_beam_search":
        raise ValueError(
            "active streaming hotwords require modified_beam_search"
        )

    unit = str(getattr(config, "asr_modeling_unit", "") or "").strip().lower()
    if unit not in {"bpe", "cjkchar", "cjkchar+bpe"}:
        raise ValueError(
            "active streaming hotwords require asr_modeling_unit to be one of "
            "bpe, cjkchar, or cjkchar+bpe"
        )
    kwargs = {"modeling_unit": unit}
    if unit in {"bpe", "cjkchar+bpe"}:
        vocab = str(getattr(config, "asr_bpe_vocab", "") or "").strip()
        if not vocab:
            raise ValueError(
                "BPE streaming hotwords require an asr_bpe_vocab file"
            )
        validate_bpe_vocab_file(
            vocab,
            expected_sha256=str(
                getattr(config, "asr_bpe_vocab_sha256", "") or ""
            ),
        )
        kwargs["bpe_vocab"] = vocab
    return kwargs


def _validate_streaming_hotword_phrases(
    config: "SherpaConfig", phrases: list[str] | tuple[str, ...]
) -> None:
    """Validate an explicit phrase contract without constraining custom BPEs.

    The pinned English Zipformer setup selects ``upper`` because its token table
    cannot encode lowercase domain words without the unknown token. Custom BPE
    models keep the default empty policy and may define their own casing.
    """
    policy = str(
        getattr(config, "asr_hotwords_case_policy", "") or ""
    ).strip().lower()
    if policy not in {"", "upper_ascii_words"}:
        raise ValueError(
            "active streaming hotwords have an unsupported "
            "asr_hotwords_case_policy"
        )
    if policy == "upper_ascii_words":
        invalid = [
            phrase for phrase in phrases
            if _UPPER_ASCII_WORDS.fullmatch(phrase) is None
        ]
        if invalid:
            raise ValueError(
                "the selected English BPE context requires UPPERCASE ASCII "
                "word phrases separated by single spaces"
            )


def create_recognizer_stream(
    recognizer,
    config: "SherpaConfig",
    *,
    hotwords: list[str] | tuple[str, ...] | None = None,
    stream_role=None,
):
    """Create one streaming-ASR stream with production contextual biasing.

    Live capture and recorded replay must use the same per-stream hotword seam;
    otherwise a recording A/B silently measures an un-biased recognizer even
    though the live engine is biased. An active phrase list fails closed when
    the runtime rejects the per-stream argument; silently retrying a plain
    stream would invalidate both live behavior and an A/B result.
    """
    phrases = list(hotwords) if hotwords is not None else [
        line.strip()
        for line in (getattr(config, "asr_hotwords", "") or "").splitlines()
        if line.strip()
    ]
    if phrases:
        _validate_streaming_hotword_phrases(config, phrases)
        if getattr(config, "asr_decoding_method", "") != "modified_beam_search":
            raise ValueError(
                "active streaming hotwords require modified_beam_search"
            )
        kwargs = {"hotwords": "\n".join(phrases)}
        if stream_role is not None:
            kwargs["role"] = stream_role
        try:
            return recognizer.create_stream(**kwargs)
        except TypeError as exc:
            raise RuntimeError(
                "installed sherpa-onnx lacks active per-stream hotword support"
            ) from exc
    if stream_role is not None:
        return recognizer.create_stream(role=stream_role)
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
    Extra non-context kwargs are filtered against ``from_transducer``'s real
    signature. An explicitly selected model-specific hotword context instead
    fails closed when the installed runtime cannot accept it; otherwise an A/B
    could silently measure an un-biased recognizer."""
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
    context_kwargs = _streaming_hotword_context_kwargs(c)
    if context_kwargs:
        kwargs["hotwords_score"] = c.asr_hotwords_score
        kwargs.update(context_kwargs)
    constructor = sherpa_onnx.OnlineRecognizer.from_transducer
    supported = _supported(constructor, kwargs)
    required_hotword_args = set(context_kwargs)
    if context_kwargs:
        required_hotword_args.update({"decoding_method", "hotwords_score"})
    missing_hotword_args = required_hotword_args.difference(supported)
    if missing_hotword_args:
        raise RuntimeError(
            "installed sherpa-onnx lacks selected streaming hotword "
            f"arguments: {', '.join(sorted(missing_hotword_args))}"
        )
    return constructor(**supported)


def build_final_recognizer(c: "SherpaConfig"):
    """Optional OFFLINE second-pass recognizer for the FINAL transcript (the text
    that reaches the LLM). The streaming transducer gives low-latency partials +
    the endpoint; this re-transcribes the endpointed UTTERANCE with a stronger
    offline model that sees the WHOLE utterance at once -- far more robust on
    run-on / casual speech, with punctuation + casing + ITN built in. Measured
    2026-06-01: SenseVoice fixed the streaming garble ("HEY IRIC LISTENING TO ME"
    -> "Hey, are you listening to me.") at ~150ms/utterance.

    None unless ``asr_final_backend`` (``sense_voice``, ``whisper``, or
    ``nemo_transducer``) is set and the model exists. Ordinary configuration
    stays fail-open: a build error returns None and keeps the streaming final.
    An explicit named profile sets ``asr_final_required`` and fails closed
    before capture instead of silently degrading to streaming-only finals.

    ``nemo_transducer`` is the exact sherpa-onnx export contract used by the
    measured Parakeet candidates. Keep sherpa-onnx's documented
    ``feature_dim=80`` constructor setting and the explicit model type. The
    export's 128-wide encoder input is model-internal metadata handled by the
    NeMo path, not a reason to silently rewrite the frontend contract.
    """
    backend = (getattr(c, "asr_final_backend", "") or "").strip().lower()
    required = bool(getattr(c, "asr_final_required", False))
    if not backend:
        if required:
            raise RuntimeError("required final recognizer backend is unset")
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
        if required:
            raise RuntimeError(
                f"required {backend} final recognizer model is unavailable"
            )
        import logging

        logging.getLogger("speaker.sherpa").warning(
            "asr_final_backend=%r is set but its model is missing (asr_final_model=%r) "
            "-- using STREAMING-ONLY finals (lower accuracy). Run the selected "
            "offline-ASR setup, or set sherpa.asr_final_model to an existing path.",
            backend, model or "(unset)",
        )
        return None
    if required:
        required_paths = {"tokens": tokens}
        if backend in {"whisper", "nemo_transducer"}:
            required_paths["decoder"] = (
                getattr(c, "asr_final_decoder", "") or ""
            )
        if backend == "nemo_transducer":
            required_paths["joiner"] = (
                getattr(c, "asr_final_joiner", "") or ""
            )
        missing = [
            label
            for label, path in required_paths.items()
            if not path or not os.path.exists(path)
        ]
        if missing:
            raise RuntimeError(
                f"required {backend} final recognizer artifacts unavailable: "
                + ", ".join(missing)
            )
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
            recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                **_supported(
                    sherpa_onnx.OfflineRecognizer.from_sense_voice,
                    kwargs,
                )
            )
            if required and recognizer is None:
                raise RuntimeError(
                    "required sense_voice constructor returned no recognizer"
                )
            return recognizer
        if backend == "whisper":
            kwargs = dict(
                encoder=model, decoder=getattr(c, "asr_final_decoder", "") or "",
                tokens=tokens, num_threads=c.resolved_asr_threads, provider=c.provider,
            )
            recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                **_supported(
                    sherpa_onnx.OfflineRecognizer.from_whisper,
                    kwargs,
                )
            )
            if required and recognizer is None:
                raise RuntimeError(
                    "required whisper constructor returned no recognizer"
                )
            return recognizer
        if backend == "nemo_transducer":
            recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=model,
                decoder=getattr(c, "asr_final_decoder", "") or "",
                joiner=getattr(c, "asr_final_joiner", "") or "",
                tokens=tokens,
                num_threads=c.resolved_asr_threads,
                sample_rate=c.sample_rate,
                feature_dim=80,
                decoding_method="greedy_search",
                max_active_paths=4,
                provider=c.provider,
                model_type="nemo_transducer",
            )
            if required and recognizer is None:
                raise RuntimeError(
                    "required nemo_transducer constructor returned no recognizer"
                )
            return recognizer
    except Exception as exc:  # noqa: BLE001 - optional path fails open
        if required:
            raise RuntimeError(
                f"required {backend} final recognizer failed to build"
            ) from exc
        import logging

        logging.getLogger("speaker.sherpa").warning(
            "second-pass recognizer (%s) failed to build; using the streaming final",
            backend, exc_info=True)
    if required:
        raise RuntimeError(
            f"required final recognizer backend {backend!r} is unsupported"
        )
    return None


def build_final_verifier(c: "SherpaConfig"):
    """Optional local-only Faster-Whisper verifier for endpointed finals.

    The verifier is independent of the existing streaming and offline
    recognizers.  It is disabled unless an explicit backend and existing local
    model directory are configured. Construction failures fail open to the
    established final-selection baseline unless an explicit named profile set
    ``asr_final_required``. Required verifiers also warm eagerly so corrupt
    lazy-loaded model artifacts cannot reach capture.
    """
    backend = (
        getattr(c, "asr_final_verifier_backend", "") or ""
    ).strip().lower()
    required = bool(getattr(c, "asr_final_required", False))
    if not backend:
        return None
    if backend != "faster_whisper":
        if required:
            raise RuntimeError(
                f"required final verifier backend {backend!r} is unsupported"
            )
        log.warning(
            "unsupported final verifier backend %r; verifier disabled",
            backend,
        )
        return None

    model = getattr(c, "asr_final_verifier_model", "") or ""
    if not model:
        if required:
            raise RuntimeError(
                "required faster_whisper verifier model is unavailable"
            )
        log.warning(
            "faster_whisper final verifier has no local model path; "
            "verifier disabled"
        )
        return None
    try:
        from ._faster_whisper import FasterWhisperEndpointRecognizer

        cpu_threads = getattr(c, "asr_final_verifier_cpu_threads", 0)
        options = {"cpu_threads": cpu_threads} if cpu_threads else {}
        verifier = FasterWhisperEndpointRecognizer(model, **options)
        if required:
            verifier.warm()
        return verifier
    except Exception as exc:  # noqa: BLE001 - optional path preserves baseline
        if required:
            raise RuntimeError(
                "required faster_whisper final verifier failed to build"
            ) from exc
        log.warning(
            "faster_whisper final verifier failed to build; verifier disabled",
            exc_info=True,
        )
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


def _read_onnx_custom_metadata(path: str) -> dict[str, str] | None:
    """Read an ONNX file's ``metadata_props`` (custom key/value metadata) in
    O(metadata) time, or ``None`` if the file can't be parsed as a ModelProto.

    Deliberately NOT ``onnx.load`` / ``onnxruntime.InferenceSession``: both
    deserialize (or fully initialize) the whole 80-115 MB model, and the caller
    is about to hand the file to sherpa-onnx anyway -- the model must never be
    loaded twice just for a preflight. A ModelProto is a flat protobuf whose
    huge ``graph`` lives in field 7 and whose ``metadata_props``
    (StringStringEntryProto: key=1, value=2) live in field 14, so a top-level
    scan can ``seek`` past the graph and read only the metadata bytes."""

    def _varint(fh) -> int | None:
        result, shift = 0, 0
        while True:
            byte = fh.read(1)
            if not byte:
                return None
            result |= (byte[0] & 0x7F) << shift
            if not byte[0] & 0x80:
                return result
            shift += 7
            if shift > 63:  # malformed / not a protobuf
                return None

    def _entry(data: bytes) -> tuple[str, str]:
        import io

        key = value = ""
        fh = io.BytesIO(data)
        while True:
            tag = _varint(fh)
            if tag is None:
                break
            length = _varint(fh)
            if length is None:
                break
            field = fh.read(length).decode("utf-8", errors="replace")
            if tag >> 3 == 1:
                key = field
            elif tag >> 3 == 2:
                value = field
        return key, value

    meta: dict[str, str] = {}
    try:
        with open(path, "rb") as fh:
            while True:
                tag = _varint(fh)
                if tag is None:
                    break  # clean EOF
                field, wire = tag >> 3, tag & 0x07
                if wire == 0:  # varint scalar
                    if _varint(fh) is None:
                        return None
                elif wire == 1:  # fixed64
                    fh.seek(8, 1)
                elif wire == 5:  # fixed32
                    fh.seek(4, 1)
                elif wire == 2:  # length-delimited
                    length = _varint(fh)
                    if length is None:
                        return None
                    if field == 14:  # metadata_props -- the only bytes we read
                        key, value = _entry(fh.read(length))
                        if key:
                            meta[key] = value
                    else:  # graph / opset / producer... -- skip without reading
                        fh.seek(length, 1)
                else:  # unknown wire type -> not a protobuf we understand
                    return None
    except OSError:
        return None
    return meta


def _tts_family_preflight(c: "SherpaConfig", kokoro: bool) -> None:
    """Refuse a config whose TTS family selection (``tts_voices``) contradicts
    what ``tts_model`` actually is, BEFORE sherpa-onnx sees it.

    Root cause this exists for (2026-07 incident): a half-finished Kokoro
    switch left ``tts_voices`` pointing at Kokoro's voices.bin while
    ``tts_model`` still named the VITS/Piper file. sherpa-onnx's Kokoro loader
    calls ``exit(-1)`` from C++ on the metadata mismatch -- the interpreter
    dies with rc 255 and ZERO Python-visible output, which blinded the test
    harness for 10 days. A RuntimeError here is a readable test failure and a
    fixable startup error instead.

    Classification uses the export's own custom metadata (``model_type``:
    'kokoro'/'vits' on the k2-fsa + piper exports; ``style_dim`` is a
    Kokoro-only fingerprint as backup). Fail-open by design: a missing file,
    unreadable protobuf, or inconclusive metadata only warns/skips -- this
    preflight must never become its own blocker."""
    import logging
    import os

    if not os.path.isfile(c.tts_model):
        return  # missing files are handled (loudly) by the existing paths
    meta = _read_onnx_custom_metadata(c.tts_model)
    if meta is None:
        logging.getLogger("speaker.sherpa").warning(
            "Could not read ONNX metadata from tts_model (%s) -- skipping the "
            "TTS family preflight and trusting the config.", c.tts_model,
        )
        return
    model_type = meta.get("model_type", "").strip().lower()
    if model_type:
        model_is_kokoro = model_type == "kokoro"
    elif "style_dim" in meta:  # older Kokoro exports without model_type
        model_is_kokoro = True
    else:
        return  # no recognizable family fingerprint -> inconclusive, proceed
    if kokoro and not model_is_kokoro:
        raise RuntimeError(
            f"TTS config mismatch: tts_voices is set ({c.tts_voices}), selecting "
            f"the Kokoro family, but tts_model ({c.tts_model}) is a "
            f"'{model_type or 'non-Kokoro'}' export. sherpa-onnx would abort the "
            "whole process (C++ exit(-1), no traceback) on this. Fix: point "
            "tts_model at the Kokoro package's model file (the sibling of "
            "voices.bin), or clear tts_voices to use the VITS/Piper voice."
        )
    if model_is_kokoro and not kokoro:
        raise RuntimeError(
            f"TTS config mismatch: tts_model ({c.tts_model}) is a Kokoro export "
            "but tts_voices is empty, selecting the VITS/Piper family. "
            "sherpa-onnx would abort the whole process (C++ exit(-1), no "
            "traceback) on this. Fix: set tts_voices to the Kokoro package's "
            "voices.bin (and tts_tokens/tts_data_dir to its siblings), or point "
            "tts_model at a VITS/Piper voice."
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
    hard crash on the capture thread), and the doctor preflight names the fix.

    One deliberate exception to fail-open: a family/model MISMATCH (Kokoro
    selected via ``tts_voices`` but ``tts_model`` is a VITS export, or vice
    versa) raises ``RuntimeError`` via ``_tts_family_preflight`` -- sherpa's
    native loader would ``exit(-1)`` the whole interpreter on that config, so a
    readable Python error naming the fix is strictly better than either dying
    silently or muting speech on a config the owner believes is Kokoro."""
    if not c.tts_model:
        return None
    import os

    kokoro = bool(getattr(c, "tts_voices", ""))
    _tts_family_preflight(c, kokoro)
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
