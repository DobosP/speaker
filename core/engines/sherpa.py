from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

log = logging.getLogger("speaker.sherpa")


def _auto_threads() -> int:
    """Sensible CPU thread count for one ONNX model on a laptop.

    STT/TTS run on the CPU (the GPU is reserved for the LLM), so we use about
    half the logical cores, clamped to 2..4 -- sherpa-onnx rarely benefits from
    more, and we leave headroom for the capture loop and the rest of the system.
    """
    cores = os.cpu_count() or 4
    return max(2, min(4, cores // 2))

from ..asr_text import restore_casing
from ..audio_frontend import CLEAN_CAPTURE_RATES, AudioResampler, apply_gain_soft_limit
from ..engine import AudioEngine, EngineCallbacks
from ..metrics import BARGE_IN_STOP, SPEECH_END, TTS_FIRST_AUDIO
from ._denoiser import build_denoiser
from ._sherpa_models import (
    build_keyword_spotter,
    build_final_recognizer,
    build_punctuation,
    build_recognizer,
    build_tts,
    build_vad,
)
from .speaker_gate import (
    SpeakerGate,
    loudness_admits,
    passes_output_margin,
    rms,
    sherpa_speaker_gate,
)

# Production audio engine built on sherpa-onnx (k2-fsa) + sounddevice.
#
# sherpa-onnx is the cross-platform, on-device replacement for the hand-rolled
# utils/audio.py: it provides VAD, streaming ASR, endpointing, TTS, and speaker
# ID from ONNX models on Linux/Windows/macOS/Android/iOS. We import it (and
# sounddevice) lazily so the rest of the runtime and the test suite work in
# environments without the native deps or a sound card.
#
# NOTE on echo / self-barge-in: sherpa-onnx does not do acoustic echo
# cancellation. For robust full-duplex barge-in either (a) use a headset, or
# (b) gate barge-in on speaker identity (sherpa-onnx ships speaker-ID models)
# so the assistant's own TTS isn't mistaken for the user. v1 ships VAD-gated
# barge-in; the gate hook is left as `_looks_like_user`. Tune on hardware.


def _capture_attempts(
    in_dev,
    *,
    preferred_sr: int,
    dev_sr_in: int,
    pinned_sr: int = 0,
    clean_rates=(48000, 32000, 96000),
    supports=lambda device, rate: True,
):
    """Ordered list of mic-open attempts for the recovering input stream.

    ``supports(device, rate) -> bool`` probes whether a rate is accepted (wraps
    ``sd.check_input_settings``; injectable for tests).

    When ``pinned_sr > 0`` the mic is opened at EXACTLY that rate on the preferred
    device and no other rate is probed first -- every extra open/negotiate is a
    chance to reconfigure (and, on some USB mics like the AT2020USB-X, re-mute) the
    device. Otherwise the auto ladder is: preferred (16 kHz) -> a clean
    integer-ratio rate the device supports (48000 -> 16000 is an exact /3
    decimation) -> the device native rate. Both modes end with system-default
    backstops so a mid-session reopen can still recover.
    """
    from ._recovering_input import OpenAttempt

    attempts: list = []
    if pinned_sr > 0:
        attempts.append(OpenAttempt(device=in_dev, samplerate=pinned_sr))
        if dev_sr_in and dev_sr_in != pinned_sr:
            attempts.append(OpenAttempt(device=in_dev, samplerate=dev_sr_in))
    else:
        # preferred device at preferred rate (16 kHz; many laptop mics reject it)
        attempts.append(OpenAttempt(device=in_dev, samplerate=preferred_sr))
        # a CLEAN integer-ratio capture rate it actually supports, ahead of its
        # (often non-integer-ratio) native rate.
        if preferred_sr == 16000:
            for rate in clean_rates:
                if rate in (preferred_sr, dev_sr_in):
                    continue
                if supports(in_dev, rate):
                    attempts.append(OpenAttempt(device=in_dev, samplerate=rate))
        # preferred device at its native rate (deduped)
        if dev_sr_in != preferred_sr:
            attempts.append(OpenAttempt(device=in_dev, samplerate=dev_sr_in))
    # system-default backstops (both modes)
    if in_dev is not None:
        attempts.append(OpenAttempt(device=None, samplerate=preferred_sr))
    if preferred_sr != 16000:
        attempts.append(OpenAttempt(device=None, samplerate=16000))
    return attempts


@dataclass
class SherpaConfig:
    sample_rate: int = 16000
    # Streaming transducer model dir (zipformer). Required for ASR.
    asr_tokens: str = ""
    asr_encoder: str = ""
    asr_decoder: str = ""
    asr_joiner: str = ""
    # OPTIONAL second-pass (offline) recognizer for the FINAL transcript -- the
    # text that reaches the LLM. The streaming transducer above stays for the
    # low-latency partials + endpoint; when this is set, the endpointed utterance
    # is RE-transcribed by a stronger offline model (sees the whole utterance ->
    # robust on run-on/casual speech; punctuation+casing+ITN built in). Empty
    # backend (default) = streaming final only (byte-identical). ~150ms/utterance.
    asr_final_backend: str = ""          # "" | "sense_voice" | "whisper"
    asr_final_model: str = ""            # SenseVoice model.onnx, or Whisper encoder
    asr_final_tokens: str = ""
    asr_final_decoder: str = ""          # Whisper only
    asr_final_use_itn: bool = True       # SenseVoice inverse-text-normalization
    asr_final_language: str = ""         # SenseVoice language hint ("en", "" = auto)
    # Skip the second pass on utterances shorter than this (a tiny "yes"/"stop"
    # the streaming model already nails) to save the offline-decode latency.
    asr_final_min_sec: float = 0.0
    # ASR decoding method: "modified_beam_search" (more accurate, the default)
    # or "greedy_search" (slightly faster, lower accuracy). Beam search also
    # enables contextual biasing via ``asr_hotwords``.
    asr_decoding_method: str = "modified_beam_search"
    asr_max_active_paths: int = 4
    # Contextual biasing: a newline-separated list of phrases the recognizer
    # should be primed to hear (names, jargon, command words), so they aren't
    # mis-transcribed. ``asr_hotwords_score`` is the boost strength. Needs
    # ``modified_beam_search``. Empty -> no biasing.
    asr_hotwords: str = ""
    asr_hotwords_score: float = 1.5
    # Endpoint rules (the turn-commit latency knobs). rule1 fires on trailing
    # silence with no decoded text; rule2 fires on trailing silence AFTER speech
    # (this is the one that decides how long we wait before committing a final --
    # lower = snappier, higher = fewer mid-thought cut-offs); rule3 is a hard
    # utterance-length ceiling. Tuned below the sherpa defaults (2.4/1.2/20s) for
    # a more responsive feel while keeping a safe floor.
    asr_rule1_min_trailing_silence: float = 2.4
    asr_rule2_min_trailing_silence: float = 0.8
    asr_rule3_min_utterance_length: float = 20.0
    # Semantic (turn-completion) endpointing layered on the acoustic rule2 timer.
    # When enabled, a turn-completion detector + adaptive policy (core.endpointing)
    # commit a final EARLY when the partial reads as a complete turn (down to
    # endpoint_min_silence_sec) and HOLD past rule2 (up to endpoint_max_silence_sec)
    # when it ends mid-phrase. Disabled (default) -> pure acoustic, unchanged.
    endpoint_enabled: bool = False
    # MUST exceed the decoder's lookahead or an early commit clips the last word
    # (see core.endpointing). 0.5s is a safe default; validate on device.
    endpoint_min_silence_sec: float = 0.5
    endpoint_max_silence_sec: float = 1.6
    endpoint_complete_threshold: float = 0.6
    endpoint_incomplete_threshold: float = 0.3
    # Adaptive confidence-tiered SHORTEN floor (default 0.0 = OFF = uniform
    # endpoint_min_silence_sec). When > 0, a high-confidence completion (the
    # lexical 0.75 bin -- a normal ending word, never a conjunction/article) may
    # commit at this LOWER trailing silence, reclaiming ~150ms on the common case.
    # MUST exceed the decoder lookahead (~0.3-0.6s) AND a typical comma pause
    # (~0.2-0.3s); validate on device. See core.endpointing.EndpointConfig.
    endpoint_high_confidence_floor: float = 0.0
    endpoint_high_confidence_score: float = 0.75
    # Turn-completion detector for the semantic endpoint: "lexical" (text-only,
    # cheap, default) or "prosody" (the Smart Turn v3 audio model -- reads the
    # rising/sustained intonation of a mid-thought trailing-off that lexical can't,
    # so the floor can drop further). Prosody needs endpoint_prosody_model set.
    endpoint_detector: str = "lexical"
    endpoint_prosody_model: str = ""        # path to the Smart Turn ONNX
    # The prosody model (~10-25ms/call) is consulted ONLY once trailing silence has
    # reached this floor (the decision window) -- during active speech the acoustic
    # endpoint is False anyway, so this bounds the cost to a few calls per turn.
    endpoint_prosody_min_silence: float = 0.15
    endpoint_prosody_threads: int = 1       # onnxruntime intra-op threads (capture thread)
    # Restore conventional casing on partials/finals (the streaming model emits
    # ALL-CAPS unpunctuated text). Pure-Python, cheap; on by default.
    asr_restore_casing: bool = True
    # Optional sherpa-onnx punctuation model (.onnx) applied to FINALS only --
    # adds real ".,?" so the transcript reads naturally and the LLM gets clean
    # sentences. Empty -> skipped (casing restoration still applies).
    punct_model: str = ""
    # Optional speech denoiser (sherpa-onnx GTCRN, .onnx) applied to the 16 kHz
    # capture block right after resampling -- BEFORE the recognizer, the speaker
    # embedder, and the VAD all read it. Broadband noise garbles STT *and* drops
    # the enrolled-user speaker embedding below the gate (lockout); one denoise
    # cleans the single block for every downstream model. OFF by default: when
    # ``denoise_enabled`` is False OR ``denoise_model`` is empty, NOTHING is
    # built and the capture path is byte-identical to no-denoise (the gate mirrors
    # ``endpoint_enabled``). One tiny ONNX inference per 0.1 s block on the capture
    # thread (GTCRN RTF<<1, but measure before enabling on weak tiers). Recording
    # writes the DENOISED block, so a recorded run replays already-denoised --
    # keep ``denoise_enabled`` consistent across record/replay or you double-denoise.
    denoise_enabled: bool = False
    denoise_model: str = ""
    # Silero VAD model (.onnx). Required for endpointing / barge-in.
    vad_model: str = ""
    # Command fast-path: a streaming keyword spotter (separate small transducer)
    # that runs continuously -- even during playback -- so control phrases like
    # "stop" trigger an action instantly without waiting for ASR + LLM. All
    # empty -> the spotter is disabled and only the normal ASR path runs.
    kws_tokens: str = ""
    kws_encoder: str = ""
    kws_decoder: str = ""
    kws_joiner: str = ""
    kws_keywords_file: str = ""
    kws_threshold: float = 0.25
    kws_score: float = 1.0
    # Offline TTS (e.g. vits / kokoro export). Required for speech output.
    tts_model: str = ""
    tts_tokens: str = ""
    tts_data_dir: str = ""
    tts_speaker_id: int = 0
    tts_speed: float = 1.0
    # Barge-in: seconds of detected voice during playback before we interrupt.
    barge_in_min_speech_sec: float = 0.2
    # Master switch for talk-over barge-in during playback. On OPEN SPEAKERS with
    # no AEC, the assistant's own (loud) TTS leaks into the mic and a level-only
    # gate self-interrupts longer responses (observed: a story reply cut itself
    # off after 0.13s). Until AEC lands, set this False to let responses finish;
    # ASR is still never fed while speaking, so it won't transcribe itself either.
    # Re-enable once AEC (or a headset / reliable speaker-ID) removes the echo.
    barge_in_enabled: bool = True
    # Self-interruption suppression (realtime-concurrency-5). Without AEC the
    # assistant's own TTS bleeds into the mic and can look like a barge-in.
    # When the speaker gate is *unenrolled* (fail-open), require detected speech
    # to sit this many dB above the current playback level before it counts as a
    # barge-in -- a real user talking over the assistant clears the margin,
    # residual echo does not.
    # DEFAULT 6.0 dB. On-device calibration (docs/audio_calibration.md; Realtek
    # laptop, unenrolled) showed 0.0 self-interrupts 15-21x at EVERY volume, while
    # 6 dB -> 0 self-interruptions across 30-100% volume. The comparison is
    # mic-capture RMS vs playback-buffer RMS (different scales), so this margin is
    # DEVICE-SPECIFIC -- re-run `python -m tools.echo_probe` on other hardware, and
    # prefer speaker-ID enrollment (level-independent) where possible. Set 0.0 to
    # restore the legacy fail-open behaviour (no self-interruption suppression).
    barge_in_output_margin_db: float = 6.0
    # After a barge-in fires (or a watchdog storm is reported) ignore further
    # barge-in triggers for this long. Debounces a flapping VAD gate / TTS-echo
    # storm into a single interrupt instead of a rapid-fire string of them.
    barge_in_suppress_sec: float = 0.5
    # Speaker-ID gate: only treat playback-time voice as barge-in if it matches
    # the enrolled user (keeps the assistant's own TTS from self-interrupting).
    speaker_embedding_model: str = ""
    speaker_enroll_wav: str = ""
    # Persisted enrollment embedding (JSON written by ``python -m core --enroll``).
    # Preferred over re-extracting from ``speaker_enroll_wav`` every boot: cheaper
    # startup and it can average several passes. Empty -> fall back to the WAV.
    speaker_enroll_embedding: str = ""
    speaker_threshold: float = 0.5
    # Optional LOUDNESS gate (a near-field 'is this the user' signal layered on the
    # voice-identity gate). The user is CLOSE to the mic -> loud; a TV / far speaker
    # sits near the ambient floor. When > 0, a final whose voice-identity DIPPED can
    # still be admitted if its level is >= this many dB above the running ambient
    # floor (so the user is never wrongly dropped when the embedding wavers). 0
    # (default) = identity-only, unchanged. Only narrows toward MORE admits (a
    # rescue), never rejects what identity already accepted.
    input_loudness_margin_db: float = 0.0
    # When enrolled, also gate the *normal* ASR finals on speaker identity (not
    # just barge-in), so ambient voices / a TV / a read-aloud quotation aren't
    # answered as if addressed to the assistant. Fail-open when unenrolled.
    speaker_gate_input: bool = True
    # Audio device selection (sounddevice index or name; None/"" = system
    # default). List them with `python -m core --list-devices`. Use these when
    # the default output is e.g. an HDMI monitor with no speakers.
    input_device: object = None
    output_device: object = None
    # Close the TTS output stream when the play queue drains (default: keep it
    # open for low-latency next-utterance start). Enable this when ANOTHER process
    # must share the one output device -- e.g. the live_session ACOUSTIC test,
    # where the synthetic user plays its voice over the same speaker and the
    # exclusive-ALSA backend allows only one open output stream at a time. Costs a
    # stream (re)open per assistant utterance; off in the real always-on app.
    release_output_when_idle: bool = False
    # Software gain applied to captured audio before ASR -- a quick boost for a
    # quiet mic (1.0 = off). Prefer raising the OS mic level; this is a stopgap.
    input_gain: float = 1.0
    # PIN the mic capture sample rate (0 = auto: probe 16k, then a clean
    # integer-ratio rate, then the device native rate). Set this to the device's
    # NATIVE rate for a USB mic that self-mutes when ALSA reconfigures it to a
    # non-native rate (e.g. the AT2020USB-X touch-mute self-engages on the USB
    # altsetting change a 48 kHz open triggers). When pinned, the engine opens at
    # exactly this rate and never probes others; the anti-aliased soxr resampler
    # converts to ``sample_rate`` (16 kHz) regardless of ratio.
    capture_samplerate: int = 0
    # ONNX execution provider for STT/TTS/speaker-ID. Keep "cpu" so the GPU
    # stays free for the LLM; sherpa-onnx also supports "cuda"/"coreml".
    provider: str = "cpu"
    # CPU threads. ``num_threads`` is the base; ``asr_num_threads`` /
    # ``tts_num_threads`` override per-model. 0 means auto-detect from cores.
    num_threads: int = 0
    asr_num_threads: int = 0
    tts_num_threads: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "SherpaConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def base_threads(self) -> int:
        return self.num_threads if self.num_threads > 0 else _auto_threads()

    @property
    def resolved_asr_threads(self) -> int:
        return self.asr_num_threads if self.asr_num_threads > 0 else self.base_threads

    @property
    def resolved_tts_threads(self) -> int:
        return self.tts_num_threads if self.tts_num_threads > 0 else self.base_threads


def _norm_device(dev):
    """Normalize a device setting: '', None -> None (system default); a digit
    string -> int index; anything else (a device name) is returned as-is."""
    if dev in (None, ""):
        return None
    if isinstance(dev, str) and dev.lstrip("-").isdigit():
        return int(dev)
    return dev


def _resample_linear(samples, src_sr: int, dst_sr: int):
    """Linear-resample mono float32 audio. Bridges a sound card that only opens
    at its native rate to sherpa's fixed 16 kHz (and TTS output back the other
    way), so capture/playback work even when the device rejects the target rate."""
    import numpy as np

    x = np.asarray(samples, dtype="float32").reshape(-1)
    if src_sr == dst_sr or x.size == 0:
        return x
    n_out = int(round(x.shape[0] * float(dst_sr) / float(src_sr)))
    if n_out <= 0:
        return np.zeros(0, dtype="float32")
    idx = np.linspace(0.0, x.shape[0] - 1, num=n_out)
    return np.interp(idx, np.arange(x.shape[0]), x).astype("float32")


class SherpaOnnxEngine(AudioEngine):
    """Real-time on-device engine. Requires model files in :class:`SherpaConfig`.

    Lifecycle: ``start`` opens the mic and runs a capture loop that streams
    audio into the recognizer; partial hypotheses fire ``on_partial`` and an
    endpoint fires ``on_final``. ``speak`` synthesizes via sherpa-onnx TTS and
    plays it while watching for barge-in.
    """

    def __init__(self, config: SherpaConfig, *, turn_detector=None):
        self.config = config
        self._cb = EngineCallbacks()
        # Semantic endpointing (smart-endpoint): a turn-completion detector +
        # adaptive policy that the capture loop consults alongside the acoustic
        # endpoint. Built only when enabled, so the disabled default allocates
        # nothing and the decision is byte-identical to the pure acoustic path.
        # ``turn_detector`` may be injected (tests / a future Smart Turn audio
        # model); otherwise the cheap lexical detector is used.
        self._turn_detector = turn_detector
        self._endpoint_policy = None
        if config.endpoint_enabled:
            from ..endpointing import (
                AdaptiveEndpointPolicy,
                EndpointConfig,
                LexicalTurnCompletionDetector,
            )

            self._endpoint_policy = AdaptiveEndpointPolicy(EndpointConfig.from_sherpa(config))
            if self._turn_detector is None:
                self._turn_detector = self._build_turn_detector(config)
        # Only consult an audio (prosody) detector once trailing silence reaches
        # this floor -- the decision window -- so the per-block capture loop never
        # pays for it during active speech.
        self._endpoint_prosody_min_silence = float(
            getattr(config, "endpoint_prosody_min_silence", 0.15) or 0.0
        )
        # Optional loudness gate: a running ambient-noise floor (asymmetric EWMA --
        # falls fast to the floor, rises slowly so speech barely lifts it). Only
        # tracked + consulted when input_loudness_margin_db > 0.
        self._input_loudness_margin_db = float(
            getattr(config, "input_loudness_margin_db", 0.0) or 0.0
        )
        self._ambient_rms: float = 0.0
        # Only assemble the utterance audio buffer for the endpoint check when a
        # detector actually consumes it (a prosodic model); the lexical default
        # is text-only, so the capture loop pays nothing.
        self._endpoint_wants_audio = bool(
            self._turn_detector is not None and getattr(self._turn_detector, "needs_audio", False)
        )
        self._recognizer = None
        self._final_recognizer = None
        self._vad = None
        self._tts = None
        self._kws = None
        self._kws_stream = None
        self._punct = None
        self._hotwords: list[str] = []
        self._stream_in = None
        # Optional session recording (the 16 kHz audio fed to the recognizer,
        # written to WAV so the run can be replayed and frozen into a test).
        self._record_path: Optional[str] = None
        self._recorder = None
        # Actual mic capture rate (may differ from config.sample_rate when the
        # device won't open at 16 kHz); captured audio is resampled to 16 kHz.
        self._capture_sr = config.sample_rate
        # Stateful anti-aliased resampler (built in start() when the mic opens at
        # a rate other than 16 kHz); None means no resampling needed.
        self._resampler: Optional[AudioResampler] = None
        # Optional speech denoiser applied to the 16 kHz block after resampling,
        # before the recognizer/embedder/VAD (built in _build when enabled). None
        # -> the capture path is byte-identical to no-denoise.
        self._denoiser = None
        # TTS native rate vs. the rate the speaker actually opened at.
        self._tts_sr = 0
        self._play_sr = 0
        self._capture_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._speaking = threading.Event()
        self._stop_speaking = threading.Event()
        self._speaker_gate: Optional[SpeakerGate] = None
        # Single playback sink: every utterance is queued onto one worker thread
        # so sentences play in order and never overlap (streaming TTS emits many
        # short sentences in quick succession). Bounded so a runaway producer
        # can't grow memory; oldest is dropped under backpressure to stay fresh.
        self._play_q: "queue.Queue[tuple[Optional[str], Optional[Callable[[], None]]]]" = (
            queue.Queue(maxsize=64)
        )
        self._play_thread: Optional[threading.Thread] = None
        # The live PortAudio output stream, shared with stop_speaking() so a
        # barge-in can abort() it -- dropping audio already buffered in the
        # device -- instead of only ceasing to feed it (which leaves the tail
        # playing for hundreds of ms). Guarded by a lock because the playback
        # thread owns its lifecycle while stop_speaking() runs on another thread.
        self._out_stream = None
        self._out_lock = threading.Lock()
        # Serializes access to the single TTS model so the startup warm pass and a
        # live synthesis can never call ``tts.generate`` concurrently (sherpa's
        # OfflineTts is not safe for concurrent generation). Contended only at
        # startup (warm vs the first reply); the playback thread is otherwise the
        # sole synthesizer, so this is uncontended on the hot path.
        self._tts_lock = threading.Lock()
        # Whether this sherpa-onnx build supports the streaming TTS callback
        # (play audio as it is synthesized). Flipped off on the first build that
        # rejects the ``callback`` kwarg, after which we chunk the finished wave.
        self._tts_can_stream = True
        # Self-interruption suppression (realtime-concurrency-5). An EWMA of the
        # RMS level of the audio currently being played, written by the playback
        # thread and read by the capture thread as the reference level for the
        # unenrolled-conservative barge-in gate. A bare float -- the GIL makes
        # the read/write atomic and a stale sample only nudges a threshold, so
        # no lock is taken on the hot path. Decays to 0 once playback stops.
        self._playback_level: float = 0.0
        # monotonic deadline until which barge-in triggers are debounced (set
        # after a barge-in fires or when a watchdog storm is reported).
        self._barge_in_suppressed_until: float = 0.0
        # Latch: once a barge-in has fired during the CURRENT speaking run, do
        # not emit another on_barge_in for the same run -- open speakers with no
        # AEC make the VAD re-fire ~12x/utterance on self-echo, each one
        # cancelling the (already-cancelled) turn. The 0.5s suppress window above
        # only debounces; this latch hard-caps it to one interrupt per run. Reset
        # in _playback_loop on the silent->speaking transition (a genuinely new
        # reply), so a fresh interruption after the assistant goes idle still
        # fires. Dormant while barge_in_enabled is False (the watch is skipped).
        self._barge_in_fired_this_run: bool = False

    def set_record_path(self, path: Optional[str]) -> None:
        """Record this session's recognizer-rate audio to ``path`` (WAV)."""
        self._record_path = path

    # --- lazy model construction ---
    def _build(self) -> None:
        c = self.config
        self._recognizer = build_recognizer(c)
        # Optional offline second-pass recognizer for the final transcript.
        self._final_recognizer = build_final_recognizer(c)
        if self._final_recognizer is not None:
            log.info("second-pass final ASR: %s (%s)", c.asr_final_backend, c.asr_final_model)
        self._vad = build_vad(c)
        self._tts = build_tts(c)
        # Speech denoiser (None unless denoise_enabled AND a model path is set).
        # build_denoiser fails open (returns None) on a bad path so start() never
        # crashes; the capture-loop branch is skipped when this is None.
        self._denoiser = build_denoiser(c)
        if self._denoiser is not None:
            log.info("speech denoiser ACTIVE on the capture path (16 kHz, GTCRN)")
        self._kws = build_keyword_spotter(c)
        if self._kws is not None:
            self._kws_stream = self._kws.create_stream()
        # Optional punctuation restorer for finals (None -> casing only).
        self._punct = build_punctuation(c)
        # Pre-split the hotword phrase list once (newline-separated in config).
        self._hotwords = [
            line.strip() for line in (c.asr_hotwords or "").splitlines() if line.strip()
        ]
        if self._punct is not None:
            log.info("punctuation model loaded for ASR finals")
        if self._hotwords:
            log.info("ASR contextual biasing: %d hotword phrase(s)", len(self._hotwords))
        if c.speaker_embedding_model:
            self._speaker_gate = sherpa_speaker_gate(
                c.speaker_embedding_model,
                threshold=c.speaker_threshold,
                num_threads=c.resolved_asr_threads,
                provider=c.provider,
            )
            self._enroll_speaker_gate()
        if self._speaker_gate is not None and self._speaker_gate.is_enrolled:
            log.info(
                "speaker-ID gate enrolled (threshold=%.2f, input gating=%s)",
                c.speaker_threshold, c.speaker_gate_input,
            )
        elif c.speaker_embedding_model:
            log.warning(
                "speaker-ID model loaded but no enrollment found -- gate is fail-open "
                "(everything passes). Run `python -m core --enroll` to enroll your voice."
            )

    def _enroll_speaker_gate(self) -> None:
        """Load the enrolled reference into the gate.

        Prefer the persisted embedding JSON (cheap, multi-pass averaged); fall
        back to extracting from ``speaker_enroll_wav``. A reference produced by a
        different embedding model is skipped (incomparable vector space) rather
        than trusted. Any failure leaves the gate unenrolled (fail-open)."""
        c = self.config
        gate = self._speaker_gate
        if gate is None:
            return
        if c.speaker_enroll_embedding and os.path.exists(c.speaker_enroll_embedding):
            from ..enroll import enrollment_matches_model, load_enrollment

            try:
                enrollment = load_enrollment(c.speaker_enroll_embedding)
            except Exception as exc:  # noqa: BLE001 - corrupt/old file shouldn't crash boot
                log.warning("could not load enrollment %s: %s", c.speaker_enroll_embedding, exc)
            else:
                if enrollment_matches_model(enrollment, c.speaker_embedding_model):
                    gate.enroll_embedding(enrollment.embedding)
                    return
                log.warning(
                    "enrollment %s was made with a different model (%s); ignoring it -- "
                    "re-run `python -m core --enroll`.",
                    c.speaker_enroll_embedding, enrollment.model or "?",
                )
        if c.speaker_enroll_wav:
            import sherpa_onnx  # lazy

            samples, sr = sherpa_onnx.read_wave(c.speaker_enroll_wav)
            gate.enroll(samples, sr)

    # --- AudioEngine ---
    def start(self, callbacks: EngineCallbacks) -> None:
        import sounddevice as sd  # lazy

        self._cb = callbacks
        self._build()
        self._running.set()
        in_dev = _norm_device(self.config.input_device)
        out_dev = _norm_device(self.config.output_device)
        try:
            log.info("input device: %s", sd.query_devices(in_dev, kind="input").get("name", "?"))
            log.info("output device: %s", sd.query_devices(out_dev, kind="output").get("name", "?"))
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            log.warning("could not query audio devices: %s", exc)
        log.info(
            "models: recognizer=%s vad=%s tts=%s kws=%s",
            self._recognizer is not None,
            self._vad is not None,
            self._tts is not None,
            self._kws is not None,
        )
        # Build the fallback chain: (preferred_device, preferred_sr) ->
        # (preferred_device, native_sr) -> (system_default, preferred_sr)
        # -> (system_default, 16000). The recovering wrapper walks this
        # list on every (re)open, so the same logic that covers initial
        # bring-up also covers reopen after a mid-session PortAudio error.
        from ._recovering_input import _RecoveringInputStream

        preferred_sr = self.config.sample_rate
        try:
            dev_sr_in = int(
                sd.query_devices(in_dev, kind="input")["default_samplerate"]
            )
        except Exception:
            dev_sr_in = preferred_sr

        def _supports(device, rate):
            try:
                sd.check_input_settings(
                    device=device, samplerate=rate, channels=1, dtype="float32"
                )
                return True
            except Exception:  # noqa: BLE001 - rate unsupported; skip it
                return False

        attempts = _capture_attempts(
            in_dev,
            preferred_sr=preferred_sr,
            dev_sr_in=dev_sr_in,
            pinned_sr=int(self.config.capture_samplerate or 0),
            clean_rates=CLEAN_CAPTURE_RATES,
            supports=_supports,
        )

        def _open(device, samplerate):
            return sd.InputStream(
                channels=1,
                samplerate=samplerate,
                dtype="float32",
                blocksize=int(samplerate * 0.1),
                device=device,
            )

        self._stream_in = _RecoveringInputStream(
            attempts,
            opener=_open,
            on_state=self._on_capture_state,
            channels=1,
            block_seconds=0.1,
        )
        self._stream_in.open()
        self._capture_sr = self._stream_in.actual_samplerate
        # Stateful anti-aliased resampler for the capture hot path (soxr ->
        # scipy polyphase -> linear). Replaces the old per-block np.interp, which
        # aliased content >8 kHz into the speech band and corrupted ASR features.
        self._resampler = (
            AudioResampler(self._capture_sr, preferred_sr)
            if self._capture_sr != preferred_sr
            else None
        )
        if self._capture_sr != preferred_sr:
            kind = self._resampler.kind if self._resampler else "linear"
            print(
                f"[sherpa] mic rejected {preferred_sr} Hz; capturing at "
                f"{self._capture_sr} Hz and resampling to {preferred_sr} Hz ({kind})"
            )
        if self._record_path:
            from ..recorder import WavRecorder

            self._recorder = WavRecorder(self._record_path, self.config.sample_rate)
            log.info("recording session audio -> %s", self._record_path)
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self._play_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._play_thread.start()

    def stop(self) -> None:
        self._running.clear()
        self._stop_speaking.set()
        self._drain_play_q()
        # Abort the live output stream BEFORE pushing the sentinel / joining the
        # play thread. On a dead/stalled ALSA device the play thread is blocked
        # in the C-level out.write() and will never reach _play_q.get() to see
        # the sentinel, so the join below would hang (the classic "second
        # Ctrl-C needed" shutdown). abort() drops the buffered audio and makes
        # the blocked write() return; _stop_speaking is already set (above) so
        # the write() closure's except swallows the resulting error instead of
        # re-raising. On a healthy/idle stream this is a cheap no-op, so the
        # normal shutdown path is unchanged. Mirrors stop_speaking() but emits
        # no barge_in_stop metric (this is teardown, not an interruption).
        with self._out_lock:
            if self._out_stream is not None:
                try:
                    self._out_stream.abort()
                except Exception:  # noqa: BLE001 - device may be mid-teardown
                    pass
        self._play_q.put((None, None))  # sentinel: wake the worker so it exits
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
            if self._capture_thread.is_alive():
                # Wedged somewhere abort() can't unblock; never hang stop().
                # The thread is a daemon and is left to interpreter teardown.
                log.warning("capture thread did not exit within 1.0s; proceeding")
        if self._play_thread is not None:
            self._play_thread.join(timeout=1.0)
            if self._play_thread.is_alive():
                log.warning("playback thread did not exit within 1.0s; proceeding")
        if self._stream_in is not None:
            # The recovering wrapper handles its own internal close-best-effort;
            # we just need to release its handle so the engine can be GC'd.
            try:
                self._stream_in.close()
            except Exception:  # noqa: BLE001 - device may already be gone
                pass
            self._stream_in = None
        if self._recorder is not None:
            log.info("recorded %.1fs of session audio -> %s",
                     self._recorder.seconds, self._recorder.path)
            self._recorder.close()
            self._recorder = None

    def speak(self, text: str, on_done: Optional[Callable[[], None]] = None) -> None:
        # Non-blocking: hand the utterance to the single playback worker. Keeping
        # one sink (instead of a thread per call) is what makes sentence-level
        # streaming play in order rather than on top of itself.
        if self._tts is None:
            if on_done:
                on_done()
            return
        self._enqueue_play(text, on_done)

    def stop_speaking(self) -> None:
        # Cut the current utterance and discard whatever is queued behind it, so
        # a barge-in flushes pending speech instead of letting it play out.
        self._stop_speaking.set()
        self._drain_play_q()
        # Abort the live output stream so audio already handed to the device is
        # dropped immediately (PortAudio abort() discards the buffer; stop()
        # would drain it). This is the difference between a ~700ms and a ~100ms
        # barge-in stop. The playback loop re-start()s the stream for the next
        # utterance. abort() leaves it stopped-but-open, so this is cheap.
        aborted = False
        with self._out_lock:
            if self._out_stream is not None:
                try:
                    self._out_stream.abort()
                    aborted = True
                except Exception:  # noqa: BLE001 - device may be mid-teardown
                    pass
        # Playback is being cut: drop the echo reference so the capture loop
        # doesn't keep gating barge-in against a level that is no longer audible.
        self._playback_level = 0.0
        # Stamp the *true* audible-stop instant (when the buffer was dropped),
        # not when synthesis later notices the flag. Only when we actually cut
        # playing audio, so a no-op stop (nothing speaking) records nothing.
        if aborted:
            self._cb.on_metric(BARGE_IN_STOP)

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def warm(self) -> None:
        """Pay the TTS model's cold-start cost before the first reply, off the
        hot path. Best-effort: a failure just means the first synthesis is cold,
        as before. Called from the runtime's background warm thread once
        ``start()`` has built the models.

        The recognizer and keyword spotter are fed continuously by the capture
        loop from the moment capture begins, so they JIT on the first blocks of
        ambient audio; the VAD is tiny. What stays cold until the first *reply* /
        first *final* are: the TTS (touched only by the idle playback thread),
        the punctuation restorer, and the speaker-ID embedder (both run only on a
        final) -- so we exercise those three here. Everything is best-effort and
        synthesized output is discarded (never enqueued for playback), so nothing
        is heard; it just makes turn 1 land sooner."""
        import numpy as np

        # TTS (vits) -- the biggest cold cost. Under the lock so it never races a
        # live first-reply synthesis; skipped if something is already speaking.
        tts = self._tts
        if tts is not None and not self._speaking.is_set():
            try:
                with self._tts_lock:
                    tts.generate("ok", sid=self.config.tts_speaker_id, speed=self.config.tts_speed)
                log.info("sherpa TTS warm-up complete")
            except Exception:  # noqa: BLE001 - warm-up is best-effort, never fatal
                log.debug("sherpa TTS warm-up failed", exc_info=True)
        # Punctuation restorer -- runs only on a final, so it's cold on turn 1.
        if self._punct is not None:
            try:
                self._punct.add_punctuation("ok")
            except Exception:  # noqa: BLE001 - best-effort
                log.debug("punctuation warm-up failed", exc_info=True)
        # Speaker-ID embedder -- runs only on a final (input gating). similarity()
        # computes the embedding (the cold ONNX) before any cosine, so this JITs
        # it even when unenrolled; the cosine may then no-op/raise harmlessly.
        gate = self._speaker_gate
        if gate is not None:
            try:
                silence = np.zeros(int(self.config.sample_rate * 0.3), dtype="float32")
                gate.similarity(silence, self.config.sample_rate)
            except Exception:  # noqa: BLE001 - best-effort; the JIT already ran
                log.debug("speaker-ID warm-up failed", exc_info=True)

    # --- ASR text helpers ---
    def _new_asr_stream(self):
        """Create a recognizer stream, with hotword biasing when available.

        Contextual biasing is per-stream in sherpa-onnx and only honored by
        ``modified_beam_search``. We try the hotword-aware factory and fall back
        to the plain one, so an older build (or greedy decoding) still works."""
        rec = self._recognizer
        if self._hotwords and self.config.asr_decoding_method == "modified_beam_search":
            try:
                return rec.create_stream(hotwords="\n".join(self._hotwords))
            except TypeError:
                log.warning("this sherpa-onnx build ignores per-stream hotwords; "
                            "biasing disabled")
        return rec.create_stream()

    def _postprocess_final(self, text: str) -> str:
        """Turn raw recognizer text into a readable final.

        Punctuation model first (when configured) so it sees the model's own
        spacing, then casing restoration. Punctuation failure is non-fatal --
        we degrade to casing-only rather than drop the turn."""
        result = text
        if self._punct is not None:
            try:
                result = self._punct.add_punctuation(result)
            except Exception:  # noqa: BLE001 - never let post-proc lose a turn
                log.exception("punctuation model failed; using raw text")
                result = text
        if self.config.asr_restore_casing:
            # Force lowercasing-first when a punctuation model already added
            # terminators (its output may stay all-caps from the source model).
            result = restore_casing(result, force=self._punct is not None)
        return result

    def _final_transcribe(self, seg, raw_final: str) -> str:
        """The FINAL transcript for ``seg``. When a second-pass offline recognizer
        is configured, RE-transcribe the endpointed utterance with it (robust on
        run-on speech; punctuation+casing+ITN already applied -> no _postprocess).
        Otherwise (or on any failure / too-short utterance / empty result) fall
        back to the streaming final + the usual post-processing."""
        rec = self._final_recognizer
        if rec is not None and seg is not None:
            import numpy as np

            n = int(np.asarray(seg).size)
            if n >= int(self.config.asr_final_min_sec * self.config.sample_rate):
                try:
                    st = rec.create_stream()
                    st.accept_waveform(self.config.sample_rate, np.asarray(seg, dtype="float32"))
                    rec.decode_stream(st)
                    text = (st.result.text or "").strip()
                    if text:
                        return text
                except Exception:  # noqa: BLE001 - fall back to the streaming final
                    log.debug("second-pass recognizer failed; using streaming final", exc_info=True)
        return self._postprocess_final(raw_final)

    @staticmethod
    def _build_turn_detector(config):
        """Build the configured turn-completion detector. ``prosody`` loads the
        Smart Turn ONNX (when the model path exists + onnxruntime imports);
        anything else, or any failure, falls back to the cheap lexical detector so
        a misconfiguration never breaks capture."""
        import os

        from ..endpointing import LexicalTurnCompletionDetector

        which = str(getattr(config, "endpoint_detector", "lexical") or "lexical").lower()
        if which == "prosody":
            model = getattr(config, "endpoint_prosody_model", "") or ""
            if model and os.path.exists(model):
                try:
                    from ..endpointing import ProsodyTurnCompletionDetector

                    log.info("endpoint detector: prosody (Smart Turn) %s", model)
                    return ProsodyTurnCompletionDetector(
                        model, num_threads=int(getattr(config, "endpoint_prosody_threads", 1) or 1)
                    )
                except Exception:  # noqa: BLE001
                    log.warning("prosody turn-detector failed to load (%s); using lexical",
                                model, exc_info=True)
            else:
                log.warning("endpoint_detector=prosody but endpoint_prosody_model missing/"
                            "not found (%r); using lexical", model)
        return LexicalTurnCompletionDetector()

    def _decide_endpoint(
        self, *, acoustic_endpoint: bool, partial: str, silence_sec: float, samples=None
    ) -> bool:
        """Combine the acoustic endpoint with a semantic turn-completion decision.

        When semantic endpointing is disabled (the default) or there's no partial
        to judge, this is exactly ``acoustic_endpoint`` -- byte-identical to the
        pure acoustic path. Otherwise the policy may commit a final EARLY on a
        confident-complete partial, or HOLD (bounded) past the acoustic timer on a
        mid-phrase one. Pure + side-effect-free, so it is unit-testable without
        models or audio."""
        if self._endpoint_policy is None or self._turn_detector is None:
            return acoustic_endpoint
        text = (partial or "").strip()
        if not text:
            return acoustic_endpoint
        # An audio (prosody) detector is expensive (~10-25ms); consult it only once
        # trailing silence has reached the decision window. During active speech the
        # acoustic endpoint is False, so skipping it is behaviour-identical but free.
        if self._endpoint_wants_audio and silence_sec < self._endpoint_prosody_min_silence:
            return acoustic_endpoint
        try:
            score = self._turn_detector.completion_score(
                text, samples=samples, sample_rate=self.config.sample_rate
            )
        except Exception:  # noqa: BLE001 - a detector error must never break capture
            log.debug("turn-completion detector failed; using acoustic endpoint", exc_info=True)
            return acoustic_endpoint
        return self._endpoint_policy.decide(
            acoustic_endpoint=acoustic_endpoint, completion_score=score, silence_sec=silence_sec
        )

    # --- internals ---
    def _capture_loop(self) -> None:
        import numpy as np

        last_partial = ""
        # perf_counter of the most recent block that advanced the ASR result --
        # i.e. the recognizer's own notion of "speech is still arriving". When
        # the endpoint later fires (after rule2's trailing silence), this is the
        # true speech-end instant, so SPEECH_END is stamped here rather than at
        # endpoint time -- otherwise endpoint_latency reads ~0 and the fixed
        # ~0.8s trailing-silence wait is invisible to every metric (lat-1).
        # ``None`` -> no speech seen this segment; SPEECH_END falls back to now.
        last_voiced_ts: float | None = None
        recognizer = self._recognizer
        if recognizer is None:
            log.error("no recognizer built (ASR model paths missing in config?); "
                      "capture loop idle -- the assistant will never hear you")
            return
        stream = self._new_asr_stream()
        voiced_run = 0.0
        block_sec = 0.1
        # Rolling buffer of the current (non-speaking) ASR segment, used to embed
        # the speaker when an endpoint fires so input can be gated on identity.
        # Capped so a long monologue can't grow memory; the tail is what the
        # speaker model needs. Reset on every endpoint and on decode-recovery.
        utterance: list = []
        utterance_len = 0
        max_utterance = int(self.config.sample_rate * 10)
        # Diagnostics: cumulative + per-interval counters for the 2 s heartbeat.
        total_blocks = partials = finals = 0
        beat_blocks = 0
        beat_level = 0.0
        asr_errors = 0
        last_beat = time.monotonic()
        log.info("capture loop started (capture_sr=%d -> asr_sr=%d)",
                 self._capture_sr, self.config.sample_rate)
        try:
            while self._running.is_set():
                audio, _ = self._stream_in.read(int(self._capture_sr * block_sec))
                samples = np.asarray(audio, dtype="float32").reshape(-1)
                # Gain BEFORE resample (soft-knee limiter, not a hard clip) so the
                # anti-alias FIR filters any saturation harmonics above 8 kHz out
                # before the recognizer sees them.
                if self.config.input_gain != 1.0:
                    samples = apply_gain_soft_limit(samples, self.config.input_gain)
                if self._resampler is not None:
                    samples = self._resampler.process(samples)
                # Speech denoise on the 16 kHz block, AFTER resampling and BEFORE
                # every consumer (recorder, accept_waveform, the speaker embedder,
                # the VAD). When no denoiser is built this is a zero-cost identity,
                # so the path stays byte-identical to no-denoise. Passthrough-on-
                # error inside, so it can never crash this daemon thread.
                if self._denoiser is not None:
                    samples = self._denoiser.process_16k(samples)

                if self._recorder is not None:
                    self._recorder.write(samples)

                total_blocks += 1
                beat_blocks += 1
                if samples.size:
                    beat_level += float(np.sqrt(np.mean(samples * samples)))
                now = time.monotonic()
                if now - last_beat >= 2.0:
                    avg = beat_level / max(beat_blocks, 1)
                    log.debug(
                        "capture heartbeat: blocks=%d avg_rms=%.4f partials=%d finals=%d speaking=%s",
                        total_blocks, avg, partials, finals, self._speaking.is_set(),
                    )
                    if avg < 1e-4:
                        log.warning(
                            "input is ~silent (avg_rms=%.6f) -- wrong mic, muted, or no "
                            "permission? run `python -m sounddevice` to list devices", avg,
                        )
                    self._cb.on_heartbeat()
                    last_beat, beat_blocks, beat_level = now, 0, 0.0

                # Command fast-path: keyword spotting runs every block, including
                # during playback, so phrases like "stop" act with the lowest
                # possible latency and never wait on ASR endpointing or the LLM.
                self._poll_keywords(samples)

                # Track the ambient noise floor for the optional loudness gate
                # (asymmetric EWMA: fall fast to the floor, rise slowly so speech
                # barely lifts it). Cheap; only when the loudness gate is enabled.
                if self._input_loudness_margin_db > 0.0:
                    _r = rms(samples)
                    _a = self._ambient_rms
                    if _a <= 0.0:
                        self._ambient_rms = _r
                    elif _r < _a:
                        self._ambient_rms = 0.9 * _a + 0.1 * _r
                    else:
                        self._ambient_rms = 0.995 * _a + 0.005 * _r

                # Barge-in watch while the assistant is speaking.
                # While the assistant is speaking we NEVER feed ASR (so it can't
                # transcribe its own TTS). Barge-in watch is gated by
                # ``barge_in_enabled`` -- off until AEC, so the loud TTS leaking
                # into an open-speaker mic can't self-interrupt.
                if self._speaking.is_set():
                    if self.config.barge_in_enabled and self._vad is not None:
                        self._vad.accept_waveform(samples)
                        # Debounce: ignore triggers for a short window after a
                        # barge-in / reported storm so a flapping VAD gate (TTS
                        # echo with no AEC) collapses into a single interrupt.
                        if now < self._barge_in_suppressed_until:
                            voiced_run = 0.0
                        elif self._barge_in_fire_eligible(samples):
                            voiced_run += block_sec
                            if voiced_run >= self.config.barge_in_min_speech_sec:
                                voiced_run = 0.0
                                # Latch: one barge-in per speaking run. The 0.5s
                                # suppress window still debounces; the latch
                                # caps the whole run so self-echo can't re-fire.
                                self._barge_in_fired_this_run = True
                                self._barge_in_suppressed_until = (
                                    now + max(0.0, self.config.barge_in_suppress_sec)
                                )
                                log.info("barge-in detected")
                                self._cb.on_barge_in()
                        else:
                            voiced_run = 0.0
                    continue

                try:
                    utterance.append(samples)
                    utterance_len += samples.size
                    while utterance_len > max_utterance and len(utterance) > 1:
                        utterance_len -= utterance[0].size
                        utterance.pop(0)
                    stream.accept_waveform(self.config.sample_rate, samples)
                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)
                    text = recognizer.get_result(stream)
                    if text and text != last_partial:
                        last_partial = text
                        # The result advanced -> speech is actively arriving.
                        # Record the instant (perf_counter, matching the metrics
                        # clock) so a later endpoint can backdate SPEECH_END to
                        # here, exposing the trailing-silence wait (lat-1).
                        last_voiced_ts = time.perf_counter()
                        partials += 1
                        # Casing only on partials (cheap, every block); the
                        # heavier punctuation model is reserved for the final.
                        shown = restore_casing(text) if self.config.asr_restore_casing else text
                        log.debug("asr partial: %r", shown)
                        self._cb.on_partial(shown)
                    acoustic_endpoint = recognizer.is_endpoint(stream)
                    endpoint_silence = (
                        time.perf_counter() - last_voiced_ts
                        if last_voiced_ts is not None else 0.0
                    )
                    endpoint_samples = (
                        np.concatenate(utterance)
                        if (self._endpoint_wants_audio and utterance) else None
                    )
                    if self._decide_endpoint(
                        acoustic_endpoint=acoustic_endpoint,
                        partial=last_partial,
                        silence_sec=endpoint_silence,
                        samples=endpoint_samples,
                    ):
                        raw_final = recognizer.get_result(stream)
                        recognizer.reset(stream)
                        last_partial = ""
                        # Capture this segment's true speech-end instant (the last
                        # block that advanced the result) and reset for the next
                        # segment. Stamping SPEECH_END here -- not at endpoint time
                        # -- makes endpoint_latency reflect the real trailing
                        # silence instead of reading ~0 (lat-1). ``None`` (no
                        # partial ever fired) falls back to now inside mark().
                        speech_end_ts, last_voiced_ts = last_voiced_ts, None
                        seg = np.concatenate(utterance) if utterance else samples
                        utterance, utterance_len = [], 0
                        if raw_final.strip():
                            finals += 1
                            # Second-pass offline re-transcription (when configured)
                            # turns the streaming final into a robust, punctuated one;
                            # else this is the streaming final + post-processing.
                            final_text = self._final_transcribe(seg, raw_final)
                            log.info("asr final: %r (raw %r)", final_text, raw_final)
                            if self._should_act_on_final(seg):
                                self._cb.on_metric(SPEECH_END, at=speech_end_ts)
                                self._cb.on_final(final_text)
                            else:
                                log.info(
                                    "dropping final %r -- speaker is not the enrolled user",
                                    final_text,
                                )
                                self._cb.on_metric("speaker_rejected_final")
                    asr_errors = 0
                except Exception:
                    # Recover from a transient decode error by resetting the
                    # stream; bail with a clear message if it keeps failing
                    # (almost always an ASR model/tokens mismatch -- re-run
                    # `python -m tools.setup_models`).
                    asr_errors += 1
                    log.warning("ASR decode error #%d (resetting stream)", asr_errors,
                                exc_info=(asr_errors == 1))
                    last_partial = ""
                    last_voiced_ts = None
                    utterance, utterance_len = [], 0
                    # Clear the denoiser's streaming state too, so a recovered
                    # stream starts the front-end fresh (best-effort; no-op when
                    # there's no denoiser).
                    if self._denoiser is not None:
                        self._denoiser.reset()
                    try:
                        recognizer.reset(stream)
                    except Exception:
                        stream = self._new_asr_stream()
                    if asr_errors >= 10:
                        log.error(
                            "ASR failed %d times in a row -- likely an ASR model/tokens "
                            "mismatch. Re-run `python -m tools.setup_models` to refetch.",
                            asr_errors,
                        )
                        self._running.clear()
                        break
        except Exception:
            # A daemon thread that dies silently is exactly what makes the app
            # look "stuck": surface the traceback instead of vanishing.
            log.exception("capture loop crashed -- the assistant has stopped listening")
            self._running.clear()

    def _on_capture_state(self, state, message: str) -> None:
        """Forward the recovering wrapper's state changes up to the runtime.

        ``state`` is a :class:`core.engines._recovering_input.StreamState`
        but we marshal it to its string value so the engine callback
        contract stays plain-Python -- no cross-module enum imports
        required of shells. The runtime publishes it as an AgentEvent;
        the watchdog uses it to suppress its 'audio thread stalled'
        warning during legitimate reopens."""
        state_str = getattr(state, "value", str(state))
        try:
            self._cb.on_capture_state(state_str, message)
        except Exception:  # noqa: BLE001
            log.exception("on_capture_state callback raised")
        # Metric for the run summary: every reopen leaves a trace.
        if state_str == "recovering":
            self._cb.on_metric("capture_recovery_start")
        elif state_str == "open":
            self._cb.on_metric("capture_recovery_end")

    def _poll_keywords(self, samples) -> None:
        kws = self._kws
        ks = self._kws_stream
        if kws is None or ks is None:
            return
        ks.accept_waveform(self.config.sample_rate, samples)
        while kws.is_ready(ks):
            kws.decode_stream(ks)
        keyword = kws.get_result(ks)
        if keyword:
            kws.reset_stream(ks)
            self._cb.on_command(keyword)

    def _enqueue_play(self, text: str, on_done: Optional[Callable[[], None]]) -> None:
        try:
            self._play_q.put_nowait((text, on_done))
        except queue.Full:
            # Drop the oldest queued sentence rather than block or lag playback.
            try:
                self._play_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._play_q.put_nowait((text, on_done))
            except queue.Full:
                if on_done:
                    on_done()

    def _drain_play_q(self) -> None:
        try:
            while True:
                self._play_q.get_nowait()
        except queue.Empty:
            pass

    def _playback_loop(self) -> None:
        import sounddevice as sd

        out = None
        try:
            while self._running.is_set():
                try:
                    text, on_done = self._play_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if text is None:  # shutdown sentinel from stop()
                    break
                self._stop_speaking.clear()
                # Reset the one-barge-in-per-run latch only on the silent->
                # speaking transition (a genuinely NEW reply), NOT per dequeued
                # sentence: _speaking is set per sentence but clears only when the
                # queue drains (below), so an unconditional reset would reopen
                # firing mid-reply and reintroduce the self-storm. A fresh
                # interruption after the assistant goes idle still re-enables.
                was_speaking = self._speaking.is_set()
                self._speaking.set()
                if not was_speaking:
                    self._barge_in_fired_this_run = False
                self._cb.on_speech_start()
                log.debug("speaking: %r (queue depth=%d)", text, self._play_q.qsize())
                played = {"any": False}

                def write(samples) -> None:
                    if self._stop_speaking.is_set():
                        return
                    if not played["any"]:
                        played["any"] = True
                        self._cb.on_metric(TTS_FIRST_AUDIO)
                    if self._play_sr != self._tts_sr:
                        samples = _resample_linear(samples, self._tts_sr, self._play_sr)
                    # Track the level of what we're actually playing so the
                    # capture loop can require user voice to stand *above* it
                    # before counting as barge-in (self-interruption guard).
                    self._note_playback_level(samples)
                    try:
                        out.write(samples)
                    except Exception:  # noqa: BLE001
                        # A barge-in abort() on another thread can stop the
                        # stream mid-write; that's expected on interruption, not
                        # a crash. Swallow only while stopping -- a genuine write
                        # failure (not a barge-in) still surfaces and is logged.
                        if not self._stop_speaking.is_set():
                            raise

                try:
                    if out is None:
                        out_dev = _norm_device(self.config.output_device)
                        self._tts_sr = int(getattr(self._tts, "sample_rate", 0)) or 22050
                        try:
                            out = sd.OutputStream(
                                channels=1, samplerate=self._tts_sr, dtype="float32",
                                device=out_dev, latency="low",
                            )
                            out.start()
                            self._play_sr = self._tts_sr
                            log.info("playback opened at %d Hz on device %s",
                                     self._play_sr, out_dev if out_dev is not None else "default")
                        except sd.PortAudioError:
                            dev_sr = int(sd.query_devices(out_dev, kind="output")["default_samplerate"])
                            out = sd.OutputStream(
                                channels=1, samplerate=dev_sr, dtype="float32",
                                device=out_dev, latency="low",
                            )
                            out.start()
                            self._play_sr = dev_sr
                            log.warning(
                                "speaker rejected %d Hz; playing at %d Hz and resampling",
                                self._tts_sr, dev_sr,
                            )
                        with self._out_lock:
                            self._out_stream = out
                    elif not out.active:
                        # A previous barge-in abort()ed the stream; bring it back
                        # up before this utterance writes to it.
                        out.start()
                    self._synthesize(text, write)
                    # (true barge-in-stop is stamped in stop_speaking() at the
                    # abort() instant -- the moment audio actually goes silent.)
                finally:
                    if on_done:
                        on_done()
                    # Only fall idle once the queue drains, so the capture loop
                    # doesn't flap ASR/barge-in on/off between adjacent sentences.
                    if self._play_q.empty():
                        self._speaking.clear()
                        self._playback_level = 0.0  # nothing playing -> no echo ref
                        # Hand the output device back when asked (so a co-located
                        # process -- e.g. the acoustic test's synthetic user -- can
                        # open it for its turn). Drop the shared handle under the
                        # lock FIRST so a concurrent stop_speaking() can't abort a
                        # stream we're closing; reopened lazily on the next utterance.
                        if self.config.release_output_when_idle and out is not None:
                            with self._out_lock:
                                self._out_stream = None
                            try:
                                out.stop()
                                out.close()
                            except Exception:  # noqa: BLE001 - already aborted/closed
                                pass
                            out = None
                        self._cb.on_speech_end()
        except Exception:
            log.exception("playback loop crashed -- the assistant has gone mute")
            self._running.clear()
        finally:
            # Drop the shared handle before teardown so a concurrent
            # stop_speaking() can't abort() a stream we're closing.
            with self._out_lock:
                self._out_stream = None
            if out is not None:
                try:
                    out.stop()
                    out.close()
                except Exception:  # noqa: BLE001 - may already be aborted/closed
                    pass

    def _synthesize(self, text: str, write: Callable[[object], None]) -> None:
        """Synthesize ``text``, handing each audio chunk to ``write`` as it is
        produced. sherpa-onnx ``OfflineTts.generate`` streams via a ``callback``,
        so the first samples play before the whole sentence is synthesized; a
        build without that param falls back to chunking the finished waveform."""
        import numpy as np

        tts = self._tts
        sid = self.config.tts_speaker_id
        speed = self.config.tts_speed
        # Hold the TTS lock for the whole synthesis so a concurrent startup warm
        # pass can't drive the same model at the same time.
        with self._tts_lock:
            if self._tts_can_stream:

                def on_chunk(samples, *_progress) -> int:
                    write(np.asarray(samples, dtype="float32").reshape(-1))
                    return 0 if self._stop_speaking.is_set() else 1

                try:
                    tts.generate(text, sid=sid, speed=speed, callback=on_chunk)
                    return
                except TypeError:
                    self._tts_can_stream = False  # this build has no streaming callback

            audio = tts.generate(text, sid=sid, speed=speed)
        samples = np.asarray(audio.samples, dtype="float32").reshape(-1)
        sr = int(getattr(audio, "sample_rate", 0)) or 22050
        chunk = max(1, int(sr * 0.1))
        for i in range(0, len(samples), chunk):
            if self._stop_speaking.is_set():
                break
            write(samples[i : i + chunk])

    def _looks_like_user(self, samples) -> bool:
        # Decide whether playback-time voice is a genuine barge-in vs the
        # assistant's own TTS echo. Without AEC the discriminator depends on the
        # setup:
        #
        # - Enrolled AND input gating on: gate on speaker IDENTITY. On OPEN
        #   SPEAKERS the assistant's TTS is loud enough to clear any level margin
        #   (a level-only gate self-interrupted 134x in one live session), so
        #   identity is the only reliable rejector -- a genuine user matches the
        #   enrolled voice, the TTS does not. If your OWN voice is rejected,
        #   re-enroll (the soft-limiter+soxr capture fixes corrupted enrollments);
        #   the escape hatch is speaker_gate_input=false (level gate below) which
        #   is only safe on a headset or with a high margin.
        # - Unenrolled OR gating off: best-effort output-margin level gate. A real
        #   user stands a margin above playback; residual echo does not. With
        #   nothing playing it fails open; margin_db <= 0 disables it.
        # IDENTITY OR LOUDNESS. Identity is a strong POSITIVE (the user's enrolled
        # voice matches; the TTS does not), but the embedder can be UNRELIABLE on
        # some mics/voices (measured 2026-06-01: it scored the user's own voice
        # ~0.15 across a run). So when identity does not confirm, fall back to the
        # LOUDNESS signal: a barge LOUDER than the assistant's own playback by
        # ``barge_in_output_margin_db`` is the user talking OVER it (the TTS leaking
        # into the mic sits AT the playback level, not above). Raise the margin if
        # an open speaker's TTS leak self-interrupts; the one-per-run latch caps it.
        gate = self._speaker_gate
        identity_on = (
            gate is not None and gate.is_enrolled and self.config.speaker_gate_input
        )
        if identity_on and gate.accept(samples, self.config.sample_rate):
            return True
        margin_db = self.config.barge_in_output_margin_db
        if margin_db > 0.0:
            return passes_output_margin(
                rms(samples), self._playback_level, margin_db=margin_db
            )
        # No level margin configured: identity-only when on; else fail open.
        return not identity_on

    def _note_playback_level(self, samples) -> None:
        """Update the EWMA of the level we're currently playing.

        Called from the playback thread per chunk; read lock-free by the
        capture thread as the echo reference for the unenrolled barge-in
        gate. The EWMA smooths over silent gaps between phonemes so a brief
        trough doesn't momentarily open the gate to the assistant's own tail."""
        level = rms(samples)
        prev = self._playback_level
        # Fast attack (track loud output quickly), slow-ish release.
        alpha = 0.5 if level >= prev else 0.2
        self._playback_level = prev + alpha * (level - prev)

    def _barge_in_fire_eligible(self, samples) -> bool:
        """Whether this block may *start/continue* arming a barge-in.

        Combines the one-per-run latch with the VAD + identity/level gate.
        Factored out of the capture loop so the latch behaviour is unit-
        testable without an audio device. The latch (``_barge_in_fired_this_run``)
        hard-caps a speaking run to a single interrupt: open speakers with no
        AEC make the VAD re-fire many times per utterance on the assistant's own
        echo, and without the latch each one re-cancels the (already-cancelled)
        turn. The latch is reset on the silent->speaking transition in
        ``_playback_loop`` so a genuinely new interruption still fires."""
        if self._barge_in_fired_this_run:
            return False
        if self._vad is None or not self._vad.is_speech_detected():
            return False
        return self._looks_like_user(samples)

    def note_barge_in_storm(self) -> None:
        """Hook for the watchdog: a barge-in storm was detected (gate flapping,
        likely TTS leaking into the mic). Arm the debounce window so the rapid
        repeats collapse into one interrupt instead of a rattling string of
        them. Safe to call from the watchdog thread (single float write)."""
        self._barge_in_suppressed_until = (
            time.monotonic() + max(0.0, self.config.barge_in_suppress_sec)
        )

    def _should_act_on_final(self, samples) -> bool:
        """Whether a completed ASR utterance should be delivered to the runtime.

        With the gate enrolled and ``speaker_gate_input`` on, only the enrolled
        user's speech is answered -- a TV, another person, or a read-aloud
        quotation is dropped instead of being treated as a request. Fail-open
        when gating is off, there's no gate, or no enrollment, so an
        unconfigured setup behaves exactly as before."""
        if not self.config.speaker_gate_input:
            return True
        gate = self._speaker_gate
        if gate is None or not gate.is_enrolled:
            return True
        if gate.accept(samples, self.config.sample_rate):
            return True
        # Identity DIPPED. The optional loudness gate can still admit a LOUD
        # near-field speaker (the user close to the mic) -- so a wavering embedding
        # never wrongly drops the user. Only when configured (margin > 0); otherwise
        # identity-only (unchanged).
        if self._input_loudness_margin_db > 0.0:
            return loudness_admits(
                rms(samples), self._ambient_rms, margin_db=self._input_loudness_margin_db
            )
        return False
