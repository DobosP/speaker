from __future__ import annotations

import json
import logging
import math
import os
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("speaker.sherpa")

# audio-bargein-8: max played-reference blocks the audio callback may queue for
# the capture thread to ingest into the coherence detector. Generous (the
# capture thread drains it every ~0.1 s block); a bound only matters if the
# capture thread stalls, where dropping the oldest played blocks is the safe
# degradation (the coherence ring is a rolling window anyway).
_COH_REF_Q_MAX = 256
# How long a just-played block may keep the barge watch armed while the next
# queued sentence is still waiting for its own first audio. Past this, a stale
# playback-level EWMA is not evidence of an audible tail.
_BARGE_TAIL_FRESH_SEC = 0.30
# Fix 5b self-sizing playback FIFO: growth factor on a starved reply, decay factor
# on a clean one, and the UX latency CEILING the buffer may grow to. These are
# bounds (a control-loop rate + a max-added-latency ceiling), NOT per-machine
# operating values -- the operating depth is measured from this box's underruns.
_FIFO_GROW = 1.5
_FIFO_DECAY = 0.9
_FIFO_SEC_MAX = 4.0


def _auto_threads() -> int:
    """Sensible CPU thread count for one ONNX model on a laptop.

    STT/TTS run on the CPU (the GPU is reserved for the LLM), so we use about
    half the logical cores, clamped to 2..4 -- sherpa-onnx rarely benefits from
    more, and we leave headroom for the capture loop and the rest of the system.
    """
    cores = os.cpu_count() or 4
    return max(2, min(4, cores // 2))

from ..asr_text import agreement_guard, restore_casing
from ..contract import is_stop_command
from ..audio_frontend import (
    CLEAN_CAPTURE_RATES,
    AudioResampler,
    DCBlocker,
    InputAGC,
    StreamingLowpass,
    apply_gain_soft_limit,
    audio_quality_metrics,
    compute_input_calibration,
    declick,
    lowpass_soft,
    normalize_rms,
    output_leveler,
    rms_of,
)
from ..engine import AudioEngine, EngineCallbacks
from ..metrics import BARGE_IN_STOP, SPEECH_END, TTS_FIRST_AUDIO
from ..tts_markup import parse_tts_markup, resolve_tts_params
from ._denoiser import build_denoiser
from ._aec import AecDelayCalibrator, FarEndRing, PlaybackFIFO, build_aec
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
from ._dtd import AdaptiveDTD, BargeSustain
from .echo_coherence import EchoCoherenceDetector

# Production audio engine built on sherpa-onnx (k2-fsa) + sounddevice.
#
# sherpa-onnx is the cross-platform, on-device replacement for the hand-rolled
# utils/audio.py: it provides VAD, streaming ASR, endpointing, TTS, and speaker
# ID from ONNX models on Linux/Windows/macOS/Android/iOS. We import it (and
# sounddevice) lazily so the rest of the runtime and the test suite work in
# environments without the native deps or a sound card.
#
# NOTE on echo / self-barge-in: sherpa-onnx itself does not do acoustic echo
# cancellation, so this engine adds its own optional AEC front-end (core/engines/
# _aec.py, OFF by default via sherpa.aec_enabled) -- a NumPy adaptive filter that
# subtracts the played TTS (far-end, teed from the playback thread into a ring and
# read at sherpa.aec_ref_delay_ms) from the mic block before any consumer. Barge-in
# must work on the OPEN laptop speaker (no headphones -- HARD REQUIREMENT). The
# PRIMARY trigger is scale-invariant reference COHERENCE on the RAW pre-AEC mic
# (sees the user, volume-independent); the level gates (residual-floor / output-
# margin) are a fallback only for when coherence can't decide. AEC still cleans the
# block for ASR. The gate hook is `_looks_like_user`.


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
    # audio-bargein-1: capture block size (seconds). The single real-time
    # granularity for the device read, the barge windowed-sustain, and the
    # coherence/FFT work. Per-profile-overridable (device_profiles[*].sherpa) so a
    # weak phone tier can shrink per-block work inside this budget; the 0.1s
    # default keeps current behaviour byte-identical until a profile overrides it.
    block_sec: float = 0.1
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
    # Contextual biasing for the SECOND-PASS final (the only biasing that reaches
    # the text the LLM sees -- ``asr_hotwords`` biases ONLY the streaming pass,
    # which the SenseVoice second pass overrides for normal-length turns). These
    # wire sherpa-onnx's homophone-replacement + rule-FST machinery: hr_dict_dir +
    # hr_lexicon + hr_rule_fsts replace homophones in the final (e.g. a name the
    # model consistently mishears), and rule_fsts applies general replacement FSTs.
    # All empty (default) = byte-identical; an older sherpa build that lacks these
    # params drops them via ``_supported`` (no crash). Author an hr dict per the
    # sherpa-onnx homophone-replacer docs. See docs/asr_biasing.md.
    asr_final_hr_dict_dir: str = ""
    asr_final_hr_lexicon: str = ""
    asr_final_hr_rule_fsts: str = ""
    asr_final_rule_fsts: str = ""
    # Skip the second pass on utterances shorter than this (a tiny "yes"/"stop"
    # the streaming model already nails) to save the offline-decode latency.
    asr_final_min_sec: float = 0.0
    # Run the offline second pass (+ the L1 echo-floor and speaker-ID gates) on a
    # DEDICATED worker thread instead of inline on the capture loop. Default ON:
    # the synchronous decode (~150ms, longer on weak CPU) stalls the one thread
    # that also reads the mic and services barge-in -- measured in
    # run-20260617-103630 to starve TTS playback (white-noise output) and
    # time-misalign the mic/echo-reference rings on resume (false-barge
    # self-interrupts). The worker dispatches the SINGLE upgraded final in capture
    # order (the runtime is one-final-per-utterance: newest-input-wins is a
    # cancel, not an upgrade, so streaming-first-then-correct would speak garbage
    # then supersede it). Only engages when a second-pass recognizer is actually
    # built; off = the legacy inline path (kept for A/B + as the queue-full
    # fallback). asr-tts-2.
    asr_final_async: bool = True
    # ASR decoding method: "modified_beam_search" (more accurate, the default)
    # or "greedy_search" (slightly faster, lower accuracy). Beam search also
    # enables contextual biasing via ``asr_hotwords``.
    asr_decoding_method: str = "modified_beam_search"
    asr_max_active_paths: int = 4
    # Contextual biasing: a newline-separated list of phrases the recognizer
    # should be primed to hear (names, jargon, command words), so they aren't
    # mis-transcribed. ``asr_hotwords_score`` is the boost strength. Needs
    # ``modified_beam_search``. Empty -> no biasing.
    # IMPORTANT: this biases ONLY the STREAMING transducer (the partials + the
    # short-clip / no-second-pass final). When ``asr_final_backend='sense_voice'``
    # is on (the default when the model is present), the SenseVoice second pass
    # RE-transcribes the utterance and OVERRIDES the streaming text for any
    # normal-length turn -- so a name fixed here will NOT appear in the LLM-facing
    # final. To bias the FINAL transcript, use the ``asr_final_hr_*`` / rule-FST
    # fields above (homophone replacement) instead. See docs/asr_biasing.md.
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
    # Adaptive endpoint floor (SessionPauseModel, core/endpointing.py). When on,
    # the trailing-silence COMMIT floor is LEARNED per session from this speaker's
    # own mid-utterance pause distribution (kills mid-sentence over-fragmentation
    # like "THE MERE STORY"/"ABOUT"/"MY CAT BIKI") instead of a fixed silence
    # number. The three floors above are REINTERPRETED as bounds, not operating
    # points: high_confidence_floor = floor_lo (decoder-lookahead safe minimum),
    # max_silence_sec = floor_hi (hard hold cap), min_silence_sec = cold-start
    # until pause_min_samples gather. window/quantile/margin are learner knobs, not
    # per-machine numbers. Dataclass default OFF (byte-identical); on in config.json.
    endpoint_adaptive_floor: bool = False
    endpoint_pause_window: int = 64
    endpoint_pause_quantile: float = 0.85
    endpoint_pause_margin: float = 0.15
    endpoint_pause_min_samples: int = 8
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
    # On-device Acoustic Echo Cancellation (AEC). Subtracts the assistant's own
    # played TTS (the far-end reference) from the mic block so the loop does not
    # self-interrupt on open speakers (the no-AEC failure today; only the 6 dB
    # output-margin guard + speaker-ID gate hold it back). Inserted after resample,
    # BEFORE denoise / VAD / recognizer / speaker-gate. OFF by default
    # (aec_enabled=False -> build_aec returns None, capture path byte-identical).
    # Backends: "nlms" (dependency-free NumPy adaptive filter, ships now) or "dtln"
    # (deep ONNX tier, deferred -> currently no-op). aec_ref_delay_ms is the
    # speaker->mic delay used to time-align the far-end reference (CALIBRATE per
    # device with tools/echo_probe.py). aec_filter_taps is the modeled echo tail
    # (rounded to a power of two; also the added capture latency in samples).
    # aec_doubletalk_freeze stops the filter diverging onto the user's voice during
    # talk-over. aec_relaxed_margin_db is the smaller barge-in output-margin used
    # WHEN AEC is active (echo already cancelled -> soft interrupts audible again).
    aec_enabled: bool = False
    aec_backend: str = "nlms"
    aec_model: str = ""
    aec_ref_delay_ms: int = 80
    aec_filter_taps: int = 512
    aec_doubletalk_freeze: bool = True
    # FDAF adaptive-filter stability knobs (tunable live for open-speaker rooms):
    # aec_mu = NLMS step size (lower = slower but more stable; 0.5 diverged on an
    # open laptop speaker); aec_leak = coefficient leakage (< 1.0 bounds divergence
    # via exponential forgetting -- the leaky-LMS safeguard; 1.0 = no bound).
    aec_mu: float = 0.3
    aec_leak: float = 0.9999
    # Intra-op threads for the DTLN ONNX canceller. DTLN is tiny and runs on the
    # realtime capture path; onnxruntime's default (all-core, spin-waiting) pool
    # pegs the CPU and starves the capture + LLM threads (deaf mic / stalled turns).
    # 1 keeps it to a slice of one core. Raise only on a fast many-core box if AEC
    # can't keep real-time.
    aec_num_threads: int = 1
    aec_relaxed_margin_db: float = 3.0
    # Auto-calibrate the AEC far-end read delay during a run (AecDelayCalibrator,
    # below). When on, the capture loop measures the true speaker->mic lag on-device
    # by normalized cross-correlation and drives _aec_ref_delay from it, so a mis-set
    # seed self-corrects instead of degrading cancellation for the whole session (a
    # wrong delay = poor/zero ERLE; the true delay is device-specific: ~40ms a
    # laptop speaker vs ~260ms Bluetooth). No dependency on the coherence detector.
    # No-op only when AEC or aec_auto_delay is off.
    aec_auto_delay: bool = True
    # AecDelayCalibrator bounds (core/engines/_aec.py). The OPERATING far->near
    # delay is MEASURED on-device by normalized cross-correlation of the mic vs the
    # true-playback-aligned far reference (the same robust estimator diagnose_run
    # uses), so ``aec_ref_delay_ms`` above is only a SEED used until the first
    # high-correlation estimate is accepted. These three are PHYSICS bounds, NOT
    # per-machine tuning: min_corr rejects noise-floor lags, window is the
    # correlation integration length, max clamps the search (laptop speaker..BT).
    aec_delay_window_ms: float = 1500.0
    aec_delay_min_corr: float = 0.15
    aec_delay_max_ms: float = 400.0
    # --- WebRTC Audio Processing Module (aec_backend="apm") ----------------------
    # The production audio front-end Chrome/Meet/Teams ship: AEC3 (multi-delay,
    # nonlinear-tolerant echo cancel) + a residual-echo suppressor + ML noise
    # suppression + AGC2 + a high-pass filter, in one stage. Selected via
    # aec_backend="apm" (needs the `livekit` package, which exposes
    # rtc.AudioProcessingModule; build_aec fails OPEN to no-AEC if it is absent).
    # Unlike the hand-rolled NLMS/DTLN backends, AEC3 handles a nonlinear open
    # laptop speaker (where linear NLMS measures ~0 dB ERLE and diverges). These
    # toggle the four APM sub-stages; the echo canceller is always on for the
    # "apm" backend (that is the point). apm_always_on runs the whole APM on EVERY
    # capture block (not just during playback) so its NS/AGC/HPF clean the user's
    # own idle-path utterance too -- the desktop analog of the OS voice-comm path
    # the mobile app already uses; OFF keeps the idle path byte-identical and only
    # engages the APM (for echo cancel) while the assistant speaks.
    apm_noise_suppression: bool = True
    apm_high_pass_filter: bool = True
    apm_gain_control: bool = False
    apm_always_on: bool = False
    # When the always-on APM owns noise suppression (_apm_owns_ns), feed the
    # RECOGNIZER a parallel AEC+RES+HPF tap with the aggressive ML noise-suppressor
    # OFF, so the near-end user's words survive for recognition (real usage: NS was
    # erasing them; the raw-mic replay proved the words are present). Echo
    # cancellation still runs -- only the ML NS is dropped, and ONLY on the ASR
    # text path (the barge/floor/VAD/speaker gates keep the NS-on signal). Intent
    # flag, auto-activated at runtime from _apm_owns_ns; no per-machine value.
    apm_asr_relax_ns: bool = True
    # Hint the AEC3 render->capture delay (ms). 0 is correct here because the
    # far-end reference is already time-aligned to the mic by the FarEndRing read
    # at aec_ref_delay_ms; AEC3 refines it internally from there.
    apm_stream_delay_ms: int = 0
    # Post-AEC barge-in: a real barge must stand this many dB above the AUTO-
    # CALIBRATED residual echo+noise floor (``_playback_floor_rms``, learned online
    # during playback with a freeze-on-burst EWMA). Because the floor tracks the
    # actual post-cancellation echo at the CURRENT speaker volume + room noise, the
    # assistant's own echo sits AT the floor and cannot self-interrupt regardless
    # of how loud it plays, while a genuine talk-over (well above the floor) still
    # fires. This is the PRIMARY barge gate whenever AEC is on -- it makes the
    # coherence detector veto-only, so nonlinear open-speaker echo (which over-fires
    # the linear coherence model) can no longer self-interrupt. 0 disables it (fall
    # back to the coherence-primary / level-margin path).
    barge_in_residual_margin_db: float = 10.0
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
    # Kokoro TTS (StyleTTS2-based, many natural built-in voices): set tts_voices to
    # the package's voices.bin to select the Kokoro family in build_tts (a sibling
    # of vits). tts_lexicon is the (comma-separated) lexicon the multi-lang packages
    # ship. Both empty (default) -> the VITS/Piper path is byte-identical. Voice is
    # still picked by tts_speaker_id; sample rate auto-adapts (Kokoro is 24 kHz).
    tts_voices: str = ""
    tts_lexicon: str = ""
    tts_speaker_id: int = 0
    tts_speed: float = 1.0
    # EXPRESSIVE MARKUP (opt-in, default OFF -> byte-identical). When True the LLM
    # may prefix a sentence with a leading directive tag -- [emotion:.. voice:..
    # rate:..] (see core/tts_markup.py) -- which SherpaOnnxEngine.speak() strips
    # from the spoken text and maps to a per-utterance (speaker_id, speed). This is
    # the cheap on-device "emotion + voice diversity" capability: no extra model,
    # just a regex + two dict lookups per sentence. Requires a system prompt that
    # teaches the grammar (also opt-in); with the default tag-unaware prompt the
    # LLM never emits a tag, so enabling this alone changes nothing. The two maps
    # below are inert while this is False.
    tts_markup: bool = False
    # Named-voice map (diversity): voice name -> Kokoro sid. Lets the markup (or a
    # deployment) pick a voice by role ("warm", "narrator", ...) instead of a raw
    # integer. Empty (default) -> only tts_speaker_id is ever used. Out-of-range
    # sids are validated against the model's speaker count at synth time.
    tts_speaker_voices: dict = field(default_factory=dict)
    # Emotion -> speed-multiplier map (rate-as-affect, the realistic cheap emotion
    # lever sherpa-onnx Kokoro exposes -- there is no latent style vector). e.g.
    # {"calm": 0.9, "excited": 1.1}. Multiplies tts_speed for the tagged sentence;
    # the result is clamped to [tts_speed_min, tts_speed_max]. Empty -> no effect.
    tts_emotion_speed_map: dict = field(default_factory=dict)
    # Safe synthesis-speed band a markup rate/emotion can never exceed (a runaway
    # "[rate:50]" must not produce unusable audio). Only relevant when tts_markup.
    tts_speed_min: float = 0.5
    tts_speed_max: float = 2.0
    # Per-sentence TTS loudness normalization (core/audio_frontend.normalize_rms).
    # The offline VITS model emits a DIFFERENT amplitude per sentence, so on an open
    # speaker the played-back echo level swings and the barge-in echo floor can never
    # settle (the 2026-06-10 self-interrupt + missed-talk-over root: resid_floor swung
    # ~20-90x in one reply). >0 normalizes every sentence to this RMS before playback
    # -> a STABLE echo level for the barge floor + an even output volume; this forgoes
    # the streaming-synth callback (synth is ~0.1s) for the whole-clip level measure.
    # 0.0 disables -> the streaming path is byte-identical to before. ~0.1-0.15 is a
    # normal speech RMS; the soft-knee limiter keeps loud sentences from clipping.
    tts_target_rms: float = 0.0
    # Post-synthesis de-click: the on-device VITS voice emits deterministic
    # sample-level impulse spikes on some text (dozens per sentence) -> audible
    # clicks/crackle on an open speaker + a spike teed into the echo reference can
    # nudge a false self-interrupt. ``declick`` (core.audio_frontend) repairs the
    # isolated impulses on the synthesis (producer) thread; a no-op on clean
    # speech. True by default (the artifact is real on the shipped voice); set
    # false for byte-identical legacy output.
    tts_declick: bool = True
    # DC-blocking high-pass on the TTS output (core/audio_frontend.DCBlocker),
    # applied FIRST in the synth chain so every downstream level/peak stage sees a
    # DC-free signal. Real usage showed every synthesized sentence carried a ~0.05
    # DC offset (wasted headroom + a steady bias into the AEC reference). tts_dc_
    # block_hz is a UNIVERSAL corner (below the ~85 Hz speech fundamental), not a
    # per-machine number: only the one-pole coefficient scales with the output rate.
    tts_dc_block: bool = True
    tts_dc_block_hz: float = 20.0
    # Impulse-detection threshold for the de-clicker (|sample - local median| that
    # counts as a spike to repair). The real VITS spikes jump to ~0.5-0.95 while
    # legitimate dense-consonant/fricative energy peaks around ~0.14, so a bar at
    # ~0.22 catches every real click yet leaves consonants untouched -- the prior
    # default (0.18, the function default) sat close enough to speech to smear the
    # consonant-densest words ("robotic" timbre). Lower it only if real clicks are
    # getting through; raise it if speech detail is being softened.
    tts_declick_threshold: float = 0.22
    # OUTPUT LEVELER (opt-in, default OFF). A pure-numpy port of WebRTC AGC2's
    # adaptive-digital loudness leveler + look-ahead true-peak soft-knee limiter
    # (core/audio_frontend.output_leveler), as the TTS output stage. When True it
    # REPLACES normalize_rms (which targets a LINEAR RMS and would fight the
    # perceptual target): the synth -> declick -> output_leveler -> FIFO chain
    # owns loudness + peak. Brings Teams/Zoom-grade consistent loudness that never
    # clips on weak/old speakers, and slews the gain smoothly across a multi-
    # sentence reply (carried in self._tts_level_gain_db). When False the legacy
    # normalize_rms path is BYTE-IDENTICAL (the leveler is never reached). The two
    # numeric keys below are inert until this bool is True.
    tts_output_leveler: bool = False
    # Speech-level (loudness) target in dBFS (full scale = 1.0; broadband RMS over
    # the voiced portion is brought toward this). -18.0 matches the shipped VITS
    # voice's natural level (~-18.4 dBFS) so steady-state action is near-neutral and
    # the limiter is the only thing that touches loud peaks. Inert unless
    # tts_output_leveler=True.
    tts_loudness_target_dbfs: float = -18.0
    # True-peak ceiling in dBTP (relative to 1.0 full scale). -1.0 dBTP is the
    # standard true-peak ceiling: it leaves ~1 dB of inter-sample headroom so the
    # DAC reconstruction of an old/weak speaker never clips. Inert unless
    # tts_output_leveler=True.
    tts_true_peak_dbtp: float = -1.0
    # TIME-AWARE loudness slew rate (dB/second, like WebRTC AGC2 ~6 dB/s). The
    # per-call gain change is this rate * the sentence's duration, so loudness
    # converges proportionally to audio time (a long sentence corrects fully, a
    # short one barely moves) -- NOT a fixed per-sentence step (which would make
    # loudness depend on reply length). The FIRST utterance of a session seeds
    # straight to target (no ramp). Inert unless tts_output_leveler=True.
    tts_loudness_slew_db_per_s: float = 6.0
    # OUTPUT HIGH-FREQUENCY ROLL-OFF for small/cheap OPEN speakers (opt-in, default
    # OFF). A bright TTS voice (Kokoro spectral centroid ~2.8 kHz) can overdrive a
    # bare laptop speaker into a buzzy / "vibrating" rasp the dark legacy VITS
    # (~0.8 kHz) never triggered (owner live A/B 2026-06-22: raw Kokoro "vibrated",
    # a ~7 kHz roll-off removed it while keeping clarity). When >0, whole-clip
    # fallback uses the zero-phase FFT low-pass (core.audio_frontend.lowpass_soft);
    # the streaming TTS callback uses a fresh per-utterance IIR low-pass so this
    # knob no longer serializes every sentence. 0 = OFF (byte-identical).
    # tts_output_lowpass_width_hz is the FFT path's raised-cosine transition width.
    tts_output_lowpass_hz: float = 0.0
    tts_output_lowpass_width_hz: float = 1500.0
    # Barge-in temporal confirmation (core/engines/_dtd.BargeSustain): a cut fires
    # when at least barge_in_min_speech_sec of detected talk-over lands within the
    # trailing barge_in_sustain_window_sec. The per-frame DTD verdict FLICKERS on a
    # real talk-over (breath/pauses + AEC suppressing the user mid-double-talk), so
    # the bar is "enough eligible blocks in a short window", NOT "N consecutive":
    # 0.2s (2 blocks) within a 0.5s window cuts a normal-volume talk-over without a
    # shout, while the BOUNDED window keeps a sporadic echo leak from ever
    # accumulating to a self-interrupt. This is the fix for the run-20260609-203236
    # "needs a shout" failure: the DTD fired on 3 of 5 turn-2 blocks (2 on the
    # turn-3 pre-shout) but the old voiced_run *= 0.5 decay never reached the 0.3s
    # threshold on intermittent fires. Raise the window to tolerate more flicker;
    # raise min_speech to demand a denser/longer talk-over before cutting.
    barge_in_min_speech_sec: float = 0.2
    barge_in_sustain_window_sec: float = 0.5
    # Raised-cosine fade-out (ms) applied to the playback FIFO tail on a barge-in
    # cut, so the audible stop is a smooth ~few-ms glide to silence instead of an
    # instant step discontinuity (a click/pop on every interrupt, also teed as a
    # transient into the echo reference). 0 = legacy hard cut.
    barge_fade_ms: float = 4.0
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
    # PRIMARY barge-in discriminator (preferred over the level margin above): a
    # scale-invariant reference-coherence detector. The assistant knows exactly
    # what it is playing, so it asks "does the mic contain sound the playback does
    # NOT explain?" -- measured by magnitude-squared coherence between the
    # time-aligned TTS reference and the mic over the voiced band. This is volume-
    # INDEPENDENT by construction (the playback/capture gains cancel in the ratio),
    # needs NO enrollment, and rejects the assistant's own TTS structurally (it is
    # fully reference-explained -> incoherent fraction ~0). When it can decide it
    # OVERRIDES the identity/level gates; when it abstains (no reference yet, or
    # silence) the legacy level gate below still runs, so behaviour is never worse
    # than today. See core/engines/echo_coherence.py and
    # docs/barge_in_coherence_2026-06-02.md.
    coherence_barge_in_enabled: bool = True
    # Voiced band (Hz) the coherence is measured over (speech energy lives here).
    coherence_voiced_band_hz: tuple = (300.0, 3400.0)
    # FLOOR for how far the per-frame incoherent fraction must sit ABOVE the
    # runtime-learned echo baseline to count as user voice. The detector ALSO
    # learns the room's own echo-incoherence spread (sigma) and triggers on the
    # larger of this floor and k*sigma, so the live margin self-calibrates to the
    # room -- you rarely need to touch this. Lower it only if a real barge is
    # missed in a clean room; raise it if the assistant self-interrupts faster
    # than sigma adapts.
    coherence_margin_delta: float = 0.08
    # A barge must clear the coherence threshold for this many CONSECUTIVE capture
    # blocks (~0.1 s each) before it fires -- "a bit slower, higher confidence".
    # Trades a small amount of interrupt latency for robustness against one-off
    # over-threshold spikes (cheap-speaker nonlinearity, transient noise): a real
    # talk-over is sustained and clears it; a single bad frame does not. This is
    # the enrollment-free, identity-free interrupt -- it does NOT use speaker-ID
    # (that is a separate feature). 1 = the original fire-on-first-frame behaviour;
    # 2 is a conservative default. Raise it for more confidence at more latency.
    coherence_confirm_frames: int = 2
    # Warm-up: the first N echo-bearing blocks (~0.1 s each) after a reply starts
    # seed the detector's echo-incoherence baseline UNCONDITIONALLY (they are
    # echo-only by construction), so the self-calibrating control chart learns this
    # room+speaker's TRUE echo floor instead of starving at the provisional value
    # when a nonlinear speaker's echo is persistently over-threshold (which would
    # self-interrupt). Verdict is echo-only during warm-up. Raise it if a barge in
    # the first ~0.5 s is being missed; lower it for a faster cold-start.
    coherence_warmup_frames: int = 5
    # Self-calibration of the control chart, exposed so a nonlinear room is field-
    # tunable (calibrate with tools.echo_probe / tools.interrupt_suite). The live
    # barge margin is max(coherence_margin_delta, sigma_k * sqrt(running echo-incoh
    # variance)); baseline_alpha/var_alpha are the EWMA rates for that mean/variance;
    # provisional_baseline is the pre-warm-up starting floor.
    coherence_sigma_k: float = 3.0
    coherence_baseline_alpha: float = 0.2
    coherence_var_alpha: float = 0.15
    coherence_provisional_baseline: float = 0.5
    # --- Device-adaptive fused double-talk detector (core/engines/_dtd.py) -------
    # The open-speaker barge trigger. ON when coherence + AEC are both on. It fuses
    # three self-calibrated z-scores (raw-mic energy, post-AEC residual energy,
    # coherence incoherent-fraction) and fires when their weighted SUM exceeds
    # dtd_k for dtd_confirm_frames consecutive blocks. NO fixed dB/coherence margin:
    # each feature's bar is mean + k*sigma learned from THIS device's echo, so the
    # threshold auto-adapts to the speaker/volume/room. dtd_k is dimensionless and
    # device-INDEPENDENT (z-scores are unit-variance) -- calibrate it once with
    # `tools.echo_probe` (it prints per-frame z_raw/z_resid/z_coh/D + the echo-only
    # p95(D) vs talk-over min(D) headroom). Lower dtd_k if a real talk-over is
    # missed; raise it if the assistant self-interrupts. Coherence is weighted
    # lower by default (it overlaps voice on a nonlinear speaker).
    dtd_enabled: bool = True
    dtd_k: float = 5.0
    # Weights tuned from live fire data (2026-06-08): z_resid is the true
    # discriminator -- the user's voice is NOT in the played reference, so AEC
    # cannot cancel it and it lands in the residual (real talk-overs measured
    # resid z 85-121); z_raw fires on loud ECHO transients (kept tiny as a shout
    # nudge only); z_coh is erratic/backwards on a nonlinear speaker (dropped).
    dtd_weight_raw: float = 0.2
    dtd_weight_resid: float = 1.0
    dtd_weight_coh: float = 0.0
    # Coherence echo veto on the DTD path. The DTD still owns the positive
    # trigger, but an explicit coherence ``False`` means "the playback reference
    # explains this frame"; letting a residual z-spike override that is exactly
    # how a nonlinear TTS echo can self-interrupt while ``w_coh`` is 0.0. ``True``
    # or ``None`` keep the previous DTD behavior (user / no reference).
    dtd_coherence_echo_veto: bool = True
    # confirm_frames=1: the DTD reports per-frame; the capture loop's LEAKY
    # integrator (barge_in_min_speech_sec) does the temporal confirmation. A real
    # talk-over's D flickers frame-to-frame (breath/pauses), so requiring N
    # CONSECUTIVE in-detector frames rejected huge-D real barges (measured D=103
    # fired=False) -- the leaky outer integrator tolerates the flicker instead.
    dtd_confirm_frames: int = 1
    dtd_warmup_frames: int = 5
    # Sigma floor as a fraction of the chart mean: a higher floor stops a small
    # DTLN echo LEAK (residual 0.04 over a 0.02 echo floor) from printing a huge
    # z and tripping K, without needing a shout (precision against leaks).
    dtd_chart_rel_floor: float = 0.4
    # Chart anti-contamination (2026-06-10 plan step 3; run-20260610-003800).
    # The live misses: (a) the per-reply chart restart re-seeded the baseline on
    # whatever played at reply onset -- often the USER, already objecting -- so
    # z_resid pinned at 0 for ~4s of screaming; (b) every missed talk-over block
    # was then ABSORBED into the baseline (mean+variance), making the next frame
    # even less likely to fire (the miss-feedback loop). Three levers, all
    # off-switchable:
    # * dtd_chart_persist: charts survive speaking-run boundaries (per-sentence
    #   TTS normalization keeps the echo stable run-to-run); false = legacy
    #   per-reply re-warm-up.
    # * dtd_chart_z_freeze: an echo-only update whose own z exceeds this is never
    #   absorbed (plausibly the user); 0 disables.
    # * dtd_chart_robust_seed: warm-up seeds from the LOWER HALF of the warm
    #   samples (user speech only ADDS energy, so the low end is the best echo
    #   estimate under double-talk); false = legacy running mean.
    # * dtd_chart_freeze_limit: regime-change backstop -- after this many
    #   CONSECUTIVE frozen updates the chart absorbs anyway (a volume step-up
    #   must not deadlock a persistent chart below the new echo floor); <= 0
    #   freezes indefinitely.
    dtd_chart_z_freeze: float = 3.0
    dtd_chart_persist: bool = True
    dtd_chart_robust_seed: bool = True
    dtd_chart_freeze_limit: int = 30
    # Residual-energy floor gate on the DTD barge path (the 2026-06-10 self-
    # interrupt fix). A DTD fire only counts as eligible if the post-AEC residual
    # stands at least this many dB above the LEARNED residual echo floor
    # (_playback_floor_rms, re-bootstrapped per reply at speaking-START). This is
    # NOT a fixed-energy magic number: the bar is RELATIVE to the device/room echo
    # floor the engine just measured, so it auto-adapts. It exists because on a
    # starved laptop mic the per-feature z-scores BLOAT (a tiny absolute residual
    # transient prints a huge z against a near-silent warmup baseline), so a 2-fire
    # reply-onset echo burst could trip BargeSustain(need=2) and self-interrupt
    # (run-20260609-234435). The echo fires sit at resid 0.0008-0.0018 (~at the
    # floor); a real talk-over (run-20260609-203236 turn-2 resid 0.0024-0.0041,
    # turn-3 pre-shout resid 0.0105-0.0128) stands well above it, so 12 dB rejects
    # the echo while the normal-volume talk-over STILL cuts (no shout). The raw
    # mic does NOT separate the two (echo raw 0.0026-0.0058 overlaps talk-over raw
    # 0.0024-0.0068) -- only the residual does. 0 disables the gate (pre-fix
    # behaviour); raise it if the assistant still self-interrupts, lower it if a
    # real talk-over is missed. See core/engines/_dtd.py + tests/test_barge_*.py.
    dtd_residual_floor_margin_db: float = 12.0
    # Echo reference ring length / max mic<->playback delay searched (ms).
    coherence_ring_ms: float = 600.0
    coherence_max_delay_ms: float = 400.0
    # audio-bargein-1: FFT segment for the coherence/Welch estimate. Smaller =
    # less per-block CPU (a phone tier can drop to 128) at coarser frequency
    # resolution; 256 keeps the detector's current default unchanged.
    coherence_nperseg: int = 256
    # After a barge-in fires (or a watchdog storm is reported) ignore further
    # barge-in triggers for this long. Debounces a flapping VAD gate / TTS-echo
    # storm into a single interrupt instead of a rapid-fire string of them.
    barge_in_suppress_sec: float = 0.5
    # L2 post-speaking refractory: for this long AFTER the assistant stops speaking
    # (a turn ends, or a barge-in cancels it), suppress a RE-fired barge-in so the
    # just-cancelled utterance's own echo TAIL cannot immediately self-interrupt the
    # next reply -- the cross-response runaway seen on the open speaker. A short
    # fixed CLEARANCE window (a time, not an energy threshold -- mirrors
    # barge_in_suppress_sec); 0 disables (byte-identical parity).
    barge_in_refractory_sec: float = 0.5
    # L3 playback-ONSET grace: for this long after the assistant COMMITS to a new
    # reply (silent->speaking, i.e. synthesis start), suppress barge-in. The live
    # self-interrupts (run-20260617-231807 / -234000) fire 0.04-0.24s after
    # "speaking:" -- DURING the synth lead-in, before any audio plays -- because
    # the echo-coherence reference ring is reset+empty at onset, so its Welch
    # estimate is unstable and reads the assistant's OWN echo (or the just-spoken
    # final's tail) as a barge. Anchored at synth start (not first-audio, which
    # the barge beats) so the window covers synth + the audio onset. A real
    # talk-over PAST the window still cuts (worst case <= grace later, never
    # lost). 0 disables (byte-identical parity).
    barge_in_playback_onset_grace_sec: float = 0.40
    # Word-gated barge confirmation (duck-then-confirm, 2026-07-02). The live
    # dichotomy on an open nonlinear speaker (run-20260702-220207): with the
    # coherence veto ON the DTD's correct fires are all vetoed (no barge); with it
    # OFF the DTD also fires on the assistant's own imperfectly-cancelled echo
    # (self-interrupt). No acoustic detector alone can be perfect there, so --
    # following the LiveKit/Pipecat word-gate pattern -- an acoustic trigger no
    # longer hard-cuts: playback DUCKS to barge_confirm_duck_gain (audible but
    # quiet), the streaming recognizer is fed for up to barge_confirm_window_sec,
    # and the barge hard-fires ONLY when >= barge_confirm_min_words NEW words are
    # transcribed that do not read as the assistant's own speech (or a stop
    # command, which confirms alone). No words -> restore volume, keep speaking:
    # a false trigger costs a brief volume dip instead of a self-interrupt, and a
    # real talk-over still cuts -- with the user's words already in the stream
    # (free pre-roll for the final). Ducking also physically shrinks the echo
    # during the window, so the recognizer mostly hears the USER. After an
    # unconfirmed window, re-triggers are suppressed for
    # barge_confirm_retry_suppress_sec so an echo-heavy reply can't audibly pump
    # the volume. False (default) -> the legacy immediate hard-fire, byte-identical.
    barge_confirm_enabled: bool = False
    barge_confirm_window_sec: float = 1.5
    barge_confirm_min_words: int = 2
    barge_confirm_duck_gain: float = 0.15
    barge_confirm_retry_suppress_sec: float = 2.0
    # OS-capture duck-open level gate (2026-07-05). Only consulted when the word
    # gate opens the duck from sustained voiced-speech-during-playback (the OS
    # echo-cancel path, aec_enabled=false, where the acoustic eligibility gate has
    # no user/echo discriminant). The triggering block must sit within this many
    # dB of the running playback reference (_playback_level) to open the duck --
    # i.e. LOUD enough to be the near user, not the far residual-echo floor the OS
    # canceller leaves. Measured run-20260705-230222: OS-cancelled echo ~-26 dB
    # below the TTS reference, a real talk-over ~-14 dB, so -18 dB splits them. A
    # dB margin vs playback (device-relative, NOT a hardcoded RMS); more-negative
    # = looser (admits quieter speech -> more pumping); a very negative value
    # (e.g. -60) effectively disables it. Unused on the legacy path.
    barge_confirm_duck_margin_db: float = -18.0
    # CONTINUOUS no-duck WORD-CUT for the OS echo-cancel path (2026-07-05). When
    # the in-app AEC/APM are OFF (aec_enabled=false -> self._aec is None) the OS
    # voice-comm canceller keeps the near-end USER clean but leaves residual-echo
    # BURSTS as loud as the user, so no LEVEL gate has a discriminant (the duck-
    # then-confirm path then PUMPS: echo opens the duck). With this ON, the
    # streaming recognizer is fed EVERY playback block on the OS-cancelled mic and
    # the barge hard-cuts the instant >= barge_word_cut_min_words NEW non-own-speech
    # words appear since the reply started (or a stop command, which cuts alone) --
    # WORD content is the discriminant, not level. No ducking, so playback level
    # never changes until the cut (no pumping). Live-scoped to self._aec is None so
    # legacy (flag off) AND every in-app AEC/APM path stay byte-identical. False
    # (default) -> inert.
    barge_word_cut_enabled: bool = False
    # Word-cut CUT floor (2026-07-05, distinct from barge_confirm_min_words so the
    # shared duck-confirm path is untouched). On a nonlinear speaker the residual
    # echo transcribes as GARBLED short fragments that don't match the clean played
    # text, so a 2-word fragment ("YOU'RE ANY") would slip past _reads_like_own_speech
    # and false-cut. Requiring >= this many NEW non-own words (a real talk-over
    # sentence clears it; a garbled 2-3 word echo hallucination does not) is the
    # primary no-false-cut gate; a bare "stop" still cuts alone via is_stop_command.
    barge_word_cut_min_words: int = 4
    # Word-cut burst-reset debounce (2026-07-06, live run-20260706-231226). The
    # per-burst stream reset guards the word floor against echo accumulating
    # across bursts, but a SINGLE VAD-quiet block is a hair trigger: the OS
    # canceller gates the near-end in and out during double-talk, so one quiet
    # block mid-sentence is VAD flicker, not a burst boundary -- and the reset
    # wipes the very words a real talk-over accumulated toward the floor.
    # Require this many CONSECUTIVE quiet blocks (~100 ms each) before the
    # reset; 1 restores the original hair-trigger semantics. Real inter-burst
    # gaps are far longer than 3 blocks (Silero's own release hysteresis alone
    # exceeds it), so echo still cannot accumulate across bursts.
    barge_word_cut_reset_quiet_blocks: int = 3
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
    # L1 echo gate on the FINAL-dispatch path: a completed ASR final reaches the
    # brain only when its level stands at least this many dB above the device's
    # LEARNED echo/quiet floor -- max(_ambient_rms, _playback_floor_rms), both
    # learned online. The assistant's own residual echo / ambient noise that the
    # recognizer turned into words sits AT that floor and is dropped; real speech
    # is many dB above it and passes. A dB-above-LEARNED-floor margin (relative,
    # device-adaptive -- never an absolute RMS), exactly like input_loudness_margin_db.
    # 0 (the dataclass default) disables it so every existing unit test is
    # byte-identical; config.json turns it on (config.json wins for real runs).
    final_floor_margin_db: float = 0.0
    # When enrolled, also gate the *normal* ASR finals on speaker identity (not
    # just barge-in), so ambient voices / a TV / a read-aloud quotation aren't
    # answered as if addressed to the assistant. Fail-open when unenrolled.
    speaker_gate_input: bool = True
    # Audio device selection (sounddevice index or name; None/"" = system
    # default). List them with `python -m core --list-devices`. Use these when
    # the default output is e.g. an HDMI monitor with no speakers.
    input_device: object = None
    output_device: object = None
    # Capture from the OS "voice communication" path so the DRIVER/OS applies its
    # own AEC + noise-suppression + AGC before the app reads a sample -- the same
    # path Microsoft Teams / browser getUserMedia (and this project's own Android
    # app) use, and the single biggest reason Teams' capture sounds clean on the
    # bare laptop while a raw-mic app does not. On WINDOWS this passes
    # sounddevice's WasapiSettings(communications=True) (the AEC/NS Communications
    # category). On LINUX the OS voice-comm path is PipeWire's webrtc
    # module-echo-cancel virtual source: load it once and point ``input_device`` at
    # the "Echo Cancellation Source" node (no code flag -- see docs/audio_pipeline.md);
    # ``python -m tools.doctor`` checks whether it is loaded. macOS uses the
    # VoiceProcessingIO unit (not yet wired). Fails open: if the platform hint
    # can't be applied the raw stream is opened as before.
    capture_voice_comm: bool = False
    # Close the TTS output stream when the play queue drains (default: keep it
    # open for low-latency next-utterance start). Enable this when ANOTHER process
    # must share the one output device -- e.g. the live_session ACOUSTIC test,
    # where the synthetic user plays its voice over the same speaker and the
    # exclusive-ALSA backend allows only one open output stream at a time. Costs a
    # stream (re)open per assistant utterance; off in the real always-on app.
    release_output_when_idle: bool = False
    # Depth (seconds of audio) of the playback FIFO between the synthesizer and
    # the PortAudio output callback. The producer (synthesis) BLOCKS on this FIFO
    # when it fills, which is the backpressure that paces synthesis to real-time
    # playback -- the precondition for the far-end AEC reference being aligned to
    # actual acoustic output (the callback tees the exact frames it plays into the
    # far ring). Too small -> audible mid-sentence underruns on a slow synth; too
    # large -> the producer can run ahead and the ring write head drifts past real
    # playback again. 1.0 s is a safe default with latency="low".
    playback_fifo_sec: float = 1.0
    # Software gain applied to captured audio before ASR -- a quick boost for a
    # quiet mic (1.0 = off). Prefer raising the OS mic level; this is a stopgap.
    input_gain: float = 1.0
    # When recording (--record), ALSO write the assistant's played-back reference
    # (the far-end TTS the open mic hears) to ``run-<id>.ref.wav`` at 16 kHz,
    # FRAME-ALIGNED with the mic WAV. This is what makes an open-speaker barge /
    # self-interrupt run faithfully REPLAYABLE headlessly: a replay can feed the
    # mic to decide() and the reference to note_playback() and reproduce the exact
    # coherence comparison (the mic-only recording can't, since the reference is
    # the engine's own playback). Off by default (no extra file).
    record_playback_reference: bool = False
    # Input AGC (automatic gain control): normalize the captured level toward
    # ``input_agc_target_rms`` so a deliberately-LOW (never-clipping) OS mic gain
    # still reaches the recognizer at a healthy level -- no manual ``input_gain``
    # tuning. BOOST-ONLY (can't un-clip a hardware-saturated mic, so set the OS
    # gain below the ADC clip point and let the AGC carry the rest). Off by
    # default (then ``input_gain`` is the static path). Takes precedence over
    # ``input_gain`` when on. See :class:`core.audio_frontend.InputAGC`.
    input_agc: bool = False
    input_agc_target_rms: float = 0.12
    input_agc_max_gain: float = 12.0
    input_agc_noise_floor_rms: float = 0.004
    input_agc_rise: float = 0.08
    input_agc_fall: float = 0.4
    # Startup ambient calibration (core/audio_frontend.compute_input_calibration):
    # before the capture loop starts, listen for ``input_calibrate_sec`` of room
    # tone and set the AGC's noise-floor gate just above THIS device's measured
    # quiet level -- the device-generic "establish an operating point" step so the
    # AGC doesn't cold-start on a hardcoded floor that's wrong for the mic. Also
    # measures the ADC clip fraction and surfaces an `input_clipping` metric (the
    # boost-only AGC can't fix a hot ADC -- the OS level must come down). OFF by
    # default (adds the calibration window to startup); only adjusts the AGC when
    # ``input_agc`` is also on, otherwise it just logs the measured floor.
    input_calibrate: bool = False
    input_calibrate_sec: float = 1.5
    # PIN the mic capture sample rate (0 = auto: probe 16k, then a clean
    # integer-ratio rate, then the device native rate). Set this to the device's
    # NATIVE rate for a USB mic that self-mutes when ALSA reconfigures it to a
    # non-native rate (e.g. the AT2020USB-X touch-mute self-engages on the USB
    # altsetting change a 48 kHz open triggers). When pinned, the engine opens at
    # exactly this rate and never probes others; the anti-aliased soxr resampler
    # converts to ``sample_rate`` (16 kHz) regardless of ratio.
    capture_samplerate: int = 0
    # soxr resampler kernel quality for the capture downsample (only runs when the
    # mic can't open at 16 kHz). "HQ" (default) is transparent for speech; "VHQ"
    # spends a little more CPU for a steeper anti-alias on a capable box; "LQ"/"MQ"
    # trade fidelity for CPU on a weak SoC. Default "HQ" = byte-identical.
    resampler_quality: str = "HQ"
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
        # Optional second recorder: the played-back reference (far-end TTS),
        # frame-aligned with the mic recording via a small 16 kHz accumulator so a
        # replay can reproduce the open-speaker barge comparison (see
        # record_playback_reference). None unless recording + the flag is on.
        self._ref_recorder = None
        self._ref_accum = None  # np.float32 ring of pending reference samples
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
        self._aec = None
        self._far_ref = None
        # audio-bargein-8: lock-free SPSC hand-off of played reference blocks from
        # the real-time audio callback to the capture thread, which feeds them to
        # the coherence detector's note_playback (keeping that lock off the audio
        # thread). None until the coherence detector is built.
        self._coh_ref_q: "Optional[deque]" = None
        self._aec_ref_delay = 0
        # WebRTC-APM flags (set in _build from the canceller the backend returns):
        # always_on => run the APM on every capture block (idle path too) for its
        # NS/AGC/HPF; suppresses_noise => it owns NS, so skip the GTCRN denoiser.
        self._apm_always_on = False
        self._apm_owns_ns = False
        # The active canceller MASKS/suppresses the near-end user during double-talk
        # (the always-on APM's NS, or DTLN's spectral mask) -> the post-AEC residual
        # goes blind to a real talk-over, so the DTD reads the RAW pre-AEC mic + its
        # floor instead. False for a genuinely LINEAR canceller (nlms/headphones)
        # that leaves the user in the residual. Set in _build.
        self._resid_blind = False
        # Parallel NS-OFF APM tap for the recognizer under _apm_owns_ns (fix 2):
        # AEC+RES+HPF on, ML NS off, so near-end words survive for ASR. None on
        # every other backend/profile -> the ASR path aliases the NS-on samples
        # (byte-identical). Built in _build; touched only on the capture thread.
        self._aec_asr = None
        # aec_auto_delay: the operating far->near delay is MEASURED on-device by
        # this calibrator (normalized cross-correlation, correlation-gated) and
        # written into _aec_ref_delay every energetic block; None until built (or
        # when auto-delay is off), in which case _aec_ref_delay holds the seed.
        self._aec_delay_cal: "Optional[AecDelayCalibrator]" = None
        # TTS native rate vs. the rate the speaker actually opened at.
        self._tts_sr = 0
        self._play_sr = 0
        # A SINGLE long-lived DC blocker for the TTS output: its one-pole state
        # carries across chunks AND sentences (a per-sentence reset would re-settle
        # the pole from zero -> an audible low-frequency thump). Lazily built at the
        # output rate; rebuilt only if that rate changes. Touched only by the synth/
        # playback worker thread (serialized), so no lock.
        self._tts_dc_blocker: "Optional[DCBlocker]" = None
        self._tts_dc_sr = 0
        self._capture_thread: Optional[threading.Thread] = None
        # asr-tts-2: dedicated second-pass worker. The queue (work items
        # ``(seg, raw_final, speech_end_ts)``) + thread are created in ``_build``
        # ONLY when a second-pass recognizer is built and ``asr_final_async`` is
        # on; left None otherwise so the finalize path runs inline (byte-identical
        # to the legacy behaviour). A small bound is plenty -- utterances arrive
        # seconds apart and a decode is ~150ms -- and on the rare overflow the
        # capture loop finalizes inline rather than blocking real-time.
        self._final_q: "Optional[queue.Queue[Optional[tuple]]]" = None
        self._final_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._speaking = threading.Event()
        self._stop_speaking = threading.Event()
        # Per-utterance generation counter (rc-3). Every queued sentence carries
        # the generation current when it was enqueued; a barge-in / stop bumps it
        # (under _gen_lock). A dequeued sentence whose generation is stale (a
        # barge happened after it was enqueued but it slipped past the queue
        # drain) is SKIPPED instead of played -- closing the dequeue->clear race
        # where the shared _stop_speaking flag could be wiped before it bit. The
        # in-flight synthesis guards also check the generation, so a barge that
        # lands while a current sentence is mid-flight aborts it even if the
        # clear-vs-set ordering momentarily wiped _stop_speaking.
        self._speak_gen = 0
        self._gen_lock = threading.Lock()
        self._speaker_gate: Optional[SpeakerGate] = None
        # Single playback sink: every utterance is queued onto one worker thread
        # so sentences play in order and never overlap (streaming TTS emits many
        # short sentences in quick succession). Bounded so a runaway producer
        # can't grow memory; oldest is dropped under backpressure to stay fresh.
        self._play_q: "queue.Queue[tuple[Optional[str], Optional[Callable[[], None]], int]]" = (
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
        # Playback FIFO between the synthesizer (producer) and the PortAudio
        # output callback (consumer). Allocated at stream open (sized in
        # play_sr samples); None for the scripted/console engines and before the
        # first utterance. The callback DRAINS it and tees the exact played block
        # into the far-end ring -> the ring write head tracks TRUE acoustic
        # playback, which is what brings the far->near lag inside DTLN tolerance.
        self._fifo: Optional[PlaybackFIFO] = None
        # Per-utterance flag: the worker sets it True when it dequeues a reply;
        # the audio callback checks-and-clears it on the first block with real
        # audio to stamp TTS_FIRST_AUDIO at the TRUE first-played instant (a
        # flushed/stopped utterance that never plays audio thus never stamps it).
        self._first_audio_pending: bool = False
        # monotonic stamp of THIS reply's synth-start (silent->speaking); 0 = none.
        # The playback-onset barge grace is measured from here (covers the synth
        # lead-in where the live self-interrupts fire, plus the audio onset).
        self._playback_onset_at: float = 0.0
        # Output-underrun diagnostics (acoustic-artifact hunt): the hard-real-time
        # _audio_cb cannot log, so it just COUNTS blocks where the FIFO ran dry
        # mid-block while speaking -- a gap PortAudio zero-fills, i.e. the audible
        # buzzy/glitchy artifact. The playback loop reports the per-reply delta
        # off-thread. Plain ints -> GIL-atomic, alloc-free, real-time safe.
        self._underrun_blocks: int = 0
        self._underrun_at_reply_start: int = 0
        # Fix 5b: self-sizing playback FIFO lead. Seeded from playback_fifo_sec and
        # then DERIVED at runtime from the measured per-reply underrun count -- grow
        # the buffer when a reply starved (the leveler forces whole-clip synth, so a
        # long next sentence can out-run a shallow FIFO -> inter-sentence gaps),
        # slow-decay on clean replies. No per-machine constant: it converges to just
        # above THIS box's worst inter-sentence synth spike, bounded by a UX ceiling.
        self._playback_fifo_sec_cur: float = float(config.playback_fifo_sec)
        # Input-clipping diagnostic (one-shot WARNING): a mic gain so hot that the
        # ADC rails shreds the waveform -> garbled STT. Detected in the capture
        # heartbeat from the per-block clipped-sample fraction.
        self._clip_warned: bool = False
        # Result of the optional startup ambient calibration (None until run).
        self._last_calibration: Optional[dict] = None
        # Optional input AGC: lets the user run a low (non-clipping) OS mic and
        # have the captured level normalized to the recognizer's sweet spot.
        c = self.config
        self._input_agc = (
            InputAGC(
                target_rms=c.input_agc_target_rms,
                max_gain=c.input_agc_max_gain,
                noise_floor_rms=c.input_agc_noise_floor_rms,
                rise=c.input_agc_rise,
                fall=c.input_agc_fall,
            )
            if c.input_agc else None
        )
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
        # Output-leveler inter-sentence loudness slew state (output_leveler).
        # Carries the applied loudness gain (dB) across sentences so loudness
        # converges smoothly (time-aware, AGC2-style) across a multi-sentence
        # reply instead of jumping per sentence. ``None`` = no utterance leveled
        # yet -> the first sentence SEEDS straight to target (no audible ramp-up);
        # thereafter it holds a float and slews. Only read/written on the synthesis
        # (producer) thread; inert unless tts_output_leveler=True.
        self._tts_level_gain_db = None
        # normalize_rms feed-forward gain (linear) carried from the previous
        # sentence. None until the first whole-clip sentence establishes it.
        # Once set, _synthesize() takes the STREAMING callback path for subsequent
        # sentences and applies this gain per-chunk instead of buffering the whole
        # clip -- reclaiming the low first-audio latency the whole-clip normalize
        # path loses without sacrificing a stable echo floor (the barge gate learns
        # against the played level; adjacent sentences differ by ≤2 dB, which the
        # EWMA absorbs). Only read/written on the synthesis (producer) thread;
        # inert unless tts_target_rms > 0 and tts_output_leveler is False.
        self._tts_normalize_gain: Optional[float] = None
        # Self-interruption suppression (realtime-concurrency-5). An EWMA of the
        # RMS level of the audio currently being played, written by the playback
        # thread and read by the capture thread as the reference level for the
        # unenrolled-conservative barge-in gate. A bare float -- the GIL makes
        # the read/write atomic and a stale sample only nudges a threshold, so
        # no lock is taken on the hot path. Decays to 0 once playback stops.
        self._playback_level: float = 0.0
        # monotonic timestamp of the last callback block that contained real
        # played audio. Used to distinguish a currently audible queued-sentence
        # tail from a stale nonzero _playback_level during a long synth lead-in.
        self._last_playback_at: float = 0.0
        # Duck-then-confirm barge state (barge_confirm_enabled). _duck_gain is
        # applied in-place by the audio callback (1.0 = no duck; bare float, GIL-
        # atomic, written by the capture thread / stop paths). _confirm_until > 0
        # marks an ACTIVE confirm window (monotonic deadline); _confirm_base_text
        # is the recognizer partial at window start so only NEW words count.
        self._duck_gain: float = 1.0
        self._confirm_until: float = 0.0
        self._confirm_base_text: str = ""
        # Continuous no-duck word-cut state (barge_word_cut_enabled). _word_cut_base
        # is the recognizer result snapshotted at a speech-burst boundary so only
        # NEW words can cut; it advances to fold pure-echo text so a real talk-over
        # is never swamped. _word_cut_fed_stream records the recognizer was fed
        # during playback (mostly echo): if the burst/reply ends with NO cut the
        # stream is reset so echo can't prefix the next final; after a cut it's
        # cleared so the user's words stay as free pre-roll. Capture-thread only.
        self._word_cut_base: str = ""
        self._word_cut_fed_stream: bool = False
        # Consecutive VAD-quiet playback blocks (word-cut burst-reset debounce).
        self._word_cut_quiet_run: int = 0
        # ADR-0013 word-cut funnel telemetry, per reply (capture-thread only).
        # Counters accumulate in _barge_word_cut_step and are emitted as ONE
        # "word-cut funnel:" INFO line at reply end -- the live failure
        # run-20260706-231226 was undiagnosable because this path was silent:
        # zero words could mean canceller-suppressed voice, a starved VAD gate,
        # or a decode failure, and the bundle couldn't tell them apart.
        self._wc_stats: dict = {}
        self._wc_win: dict = {}
        self._wc_reply_active: bool = False
        # Recently ENQUEUED sentences (post-markup, what will be synthesized) +
        # the one audibly playing right now: the echo filter for confirm-window
        # partials -- transcribed text that reads like these is the assistant's
        # own (ducked) echo, not the user. _now_playing is authoritative (a long
        # reply enqueues more sentences than any ring holds); the ring is wide so
        # the just-finished sentences' echo tails are also covered. Appended by
        # speak() (bus thread) / the playback worker, read by the capture thread;
        # GIL-safe for this diagnostic use.
        self._recent_spoken: deque = deque(maxlen=32)
        self._now_playing: str = ""
        # Echo observations collected DURING a confirm window (raw_rms,
        # resid_rms, incoherent_fraction per block). An UNCONFIRMED window is
        # verified echo-only evidence -- exactly the diet the DTD charts starve
        # for on a box whose TTS echo always reads as VAD-speech (the
        # observe_echo tap requires VAD-quiet). Fed to the charts on expiry so
        # the echo level is LEARNED and the trigger flood decays instead of
        # re-ducking every few seconds (live: 14 triggers/45s, run-223217).
        self._confirm_echo_obs: list = []
        # Auto-calibrated post-AEC residual echo+noise floor (capture thread only),
        # learned online DURING playback by _update_playback_floor. The barge gate
        # (when AEC is on) requires a real interrupt to stand
        # barge_in_residual_margin_db ABOVE this, so the assistant's own cancelled
        # echo -- which sits at the floor -- cannot self-interrupt at any speaker
        # volume. Persists across turns (the echo level tracks the ~constant speaker
        # volume) so it is ready immediately, with no per-turn re-climb that early
        # echo could exploit. Unlike _ambient_rms (the QUIET floor, which freezes
        # the instant playback echo arrives and so stays at the quiet level), this
        # one deliberately tracks the echo level present during playback.
        self._playback_floor_rms: float = 0.0
        # The same echo-only floor measured on the RAW pre-AEC mic. The barge
        # energy-confirmation uses THIS instead of the post-AEC residual: DTLN
        # suppresses the near-end (user) voice during double-talk, so the residual
        # does not rise on a real talk-over, but the raw mic does (the user's
        # energy adds on top of the echo). See _looks_like_user / the capture loop.
        self._raw_playback_floor_rms: float = 0.0
        # Scale-invariant reference-coherence barge-in detector (the PRIMARY
        # discriminator; the level margin above is only a fallback). Built in
        # _build() when coherence_barge_in_enabled and scipy is present; fed the
        # played TTS from the playback thread and queried from the capture thread.
        self._echo_coherence: Optional[EchoCoherenceDetector] = None
        # Device-adaptive fused double-talk detector (raw + residual + coherence as
        # self-calibrated z-scores; no fixed margin). Built in _build() when
        # coherence + AEC are on (the open-speaker case); else the legacy coherence/
        # level path runs. See core/engines/_dtd.py.
        self._dtd: Optional[AdaptiveDTD] = None
        # monotonic deadline until which barge-in triggers are debounced (set
        # after a barge-in fires or when a watchdog storm is reported).
        self._barge_in_suppressed_until: float = 0.0
        # monotonic stamp of the most recent _speaking->clear (a turn ended or a
        # barge cancelled it). The L2 refractory (barge_in_refractory_sec) reads it
        # to suppress an immediate re-fired barge on the cancelled tail's echo.
        # Plain float, GIL-atomic, no lock -- same idiom as _barge_in_suppressed_until.
        self._last_speaking_end: float = 0.0
        # Latch: once a barge-in has fired during the CURRENT speaking run, do
        # not emit another on_barge_in for the same run -- open speakers with no
        # AEC make the VAD re-fire ~12x/utterance on self-echo, each one
        # cancelling the (already-cancelled) turn. The 0.5s suppress window above
        # only debounces; this latch hard-caps it to one interrupt per run. Reset
        # in _playback_loop on the silent->speaking transition (a genuinely new
        # reply), so a fresh interruption after the assistant goes idle still
        # fires. Dormant while barge_in_enabled is False (the watch is skipped).
        self._barge_in_fired_this_run: bool = False
        # Cross-thread one-shot: the playback loop sets this True on the silent->
        # speaking transition; the capture loop consumes it to clear the
        # BargeSustain window so the prior reply's tail/onset echo can't prime a
        # 2-fire burst into the new reply (the run-20260609-234435 self-interrupt).
        # BargeSustain lives in the capture-loop scope, so it can't be reset from
        # the playback thread directly -- this flag bridges the two. GIL-atomic
        # bool, same lock-free idiom as _barge_in_fired_this_run.
        self._barge_sustain_reset_pending: bool = False

    def set_record_path(self, path: Optional[str]) -> None:
        """Record this session's recognizer-rate audio to ``path`` (WAV)."""
        self._record_path = path

    def _accumulate_reference(self, blk, sr: int) -> None:
        """Append one played reference block (resampled to 16 kHz) to the
        frame-aligned reference accumulator. Cheap; only while speaking + the
        reference recorder is on (record_playback_reference). Unused when the
        FarEndRing is available (AEC on) -- _write_reference_frame reads that
        directly, so skip to keep the accumulator from growing."""
        import numpy as np

        if self._ref_accum is None or self._far_ref is not None:
            return
        ref16 = (
            _resample_linear(blk, sr, self.config.sample_rate)
            if sr != self.config.sample_rate
            else np.asarray(blk, dtype="float32").reshape(-1)
        )
        self._ref_accum = np.concatenate(
            [self._ref_accum, np.asarray(ref16, dtype="float32").reshape(-1)]
        )
        # Defensive: push~=pop in balance, but never grow unbounded on a stall.
        cap = self.config.sample_rate * 5
        if self._ref_accum.shape[0] > cap:
            self._ref_accum = self._ref_accum[-cap:]

    def _write_reference_frame(self, n: int) -> None:
        """Write exactly ``n`` reference samples (silence when the assistant isn't
        playing) to the reference recorder, FRAME-ALIGNED with the mic recording.

        When AEC is on the FarEndRing exists: read it at delay 0 -- the EXACT
        true-playback-aligned far-end the canceller reads -- so a replay can sweep
        the delay and recover the LIVE ``aec_ref_delay_ms`` (calibratable headless;
        the coherence-queue accumulator's timeline can't). Without AEC, fall back
        to the accumulator (the coherence reference, for barge replay)."""
        import numpy as np

        if self._ref_recorder is None:
            return
        if self._far_ref is not None:
            out = np.asarray(self._far_ref.read(n, 0), dtype="float32").reshape(-1)
            if out.shape[0] < n:
                out = np.concatenate([out, np.zeros(n - out.shape[0], dtype="float32")])
            self._ref_recorder.write(out[:n])
            return
        if self._ref_accum is None:
            return
        if self._ref_accum.shape[0] >= n:
            out, self._ref_accum = self._ref_accum[:n], self._ref_accum[n:]
        else:
            pad = np.zeros(n - self._ref_accum.shape[0], dtype="float32")
            out, self._ref_accum = np.concatenate([self._ref_accum, pad]), np.zeros(0, dtype="float32")
        self._ref_recorder.write(out)

    # --- lazy model construction ---
    def _build(self) -> None:
        c = self.config
        self._recognizer = build_recognizer(c)
        # Optional offline second-pass recognizer for the final transcript.
        self._final_recognizer = build_final_recognizer(c)
        if self._final_recognizer is not None:
            log.info("second-pass final ASR: %s (%s)", c.asr_final_backend, c.asr_final_model)
        self._maybe_setup_async_final()
        self._vad = build_vad(c)
        # Scale-invariant reference-coherence barge-in detector (volume-
        # independent, zero-enrollment). Fails open to the level gate if scipy
        # is somehow unavailable, so start() never crashes on a partial install.
        if c.coherence_barge_in_enabled:
            det = EchoCoherenceDetector(
                c.sample_rate,
                voiced_band=tuple(c.coherence_voiced_band_hz),
                ring_ms=c.coherence_ring_ms,
                max_delay_ms=c.coherence_max_delay_ms,
                nperseg=c.coherence_nperseg,
                margin_delta=c.coherence_margin_delta,
                confirm_frames=c.coherence_confirm_frames,
                warmup_frames=c.coherence_warmup_frames,
                sigma_k=c.coherence_sigma_k,
                baseline_alpha=c.coherence_baseline_alpha,
                var_alpha=c.coherence_var_alpha,
                provisional_baseline=c.coherence_provisional_baseline,
            )
            if det.available:
                self._echo_coherence = det
                # audio-bargein-8: the played-reference hand-off queue lives as
                # long as the detector.
                self._coh_ref_q = deque(maxlen=_COH_REF_Q_MAX)
                log.info(
                    "coherence barge-in ACTIVE (scale-invariant, no enrollment; "
                    "band %s Hz, margin delta %.2f, confirm %d frames)",
                    c.coherence_voiced_band_hz, c.coherence_margin_delta,
                    c.coherence_confirm_frames,
                )
            else:
                log.warning(
                    "coherence barge-in requested but scipy unavailable; "
                    "falling back to the level-margin gate"
                )
        self._tts = build_tts(c)
        # Speech denoiser (None unless denoise_enabled AND a model path is set).
        # build_denoiser fails open (returns None) on a bad path so start() never
        # crashes; the capture-loop branch is skipped when this is None.
        self._denoiser = build_denoiser(c)
        if self._denoiser is not None:
            log.info("speech denoiser ACTIVE on the capture path (16 kHz, GTCRN)")
        # AEC (None unless aec_enabled). Built like the denoiser, fails open. The
        # far-end ring is fed by the playback thread and read at the configured
        # speaker->mic delay; only allocated when AEC is on.
        self._aec = build_aec(c)
        self._far_ref = FarEndRing() if self._aec is not None else None
        if self._aec is not None:
            self._aec_ref_delay = int(c.sample_rate * c.aec_ref_delay_ms / 1000)
            # Runtime delay auto-calibration: measure the true speaker->mic delay
            # on-device (aec_ref_delay_ms is only the seed until the first accepted
            # estimate). Bounds are physics, not per-machine tuning.
            if c.aec_auto_delay:
                self._aec_delay_cal = AecDelayCalibrator(
                    c.sample_rate,
                    seed_delay_samples=self._aec_ref_delay,
                    window_ms=c.aec_delay_window_ms,
                    min_corr=c.aec_delay_min_corr,
                    max_delay_ms=c.aec_delay_max_ms,
                )
            # The WebRTC-APM backend tags itself with these (absent on nlms/dtln).
            self._apm_always_on = bool(getattr(self._aec, "always_on", False))
            self._apm_owns_ns = self._apm_always_on and bool(
                getattr(self._aec, "suppresses_noise", False)
            )
            # A masking canceller (APM-NS or DTLN) suppresses the near-end user in
            # the residual, so the DTD residual feature + floor read the raw mic.
            self._resid_blind = self._apm_owns_ns or bool(
                getattr(self._aec, "suppresses_nearend", False)
            )
            # Fix 2: when the always-on APM owns NS, build a second AEC tap with the
            # ML NS OFF, for the recognizer only. Its echo model converges to the
            # same solution (identical near/far inputs, same delay), so it just
            # keeps the near-end words the primary's NS erases.
            if self._apm_owns_ns and c.apm_asr_relax_ns:
                self._aec_asr = build_aec(c, ns_override=False)
                if self._aec_asr is not None:
                    log.info(
                        "ASR relaxed-NS tap ACTIVE (recognizer reads AEC+RES+HPF "
                        "with ML noise-suppression OFF; barge/floor/speaker gates "
                        "keep the NS-on signal)"
                    )
            log.info(
                "AEC ACTIVE on the capture path (16 kHz, backend=%s, ref_delay=%dms"
                "%s, apm_always_on=%s)",
                c.aec_backend, c.aec_ref_delay_ms,
                " [seed; auto-calibrated at runtime]" if c.aec_auto_delay else "",
                self._apm_always_on,
            )
        # Device-adaptive fused double-talk detector: the open-speaker barge trigger.
        # Built only when coherence + AEC are both on (it fuses the coherence
        # incoherent-fraction with the raw-mic + post-AEC residual energy). With no
        # AEC (headphones / echo-free) the legacy coherence-alone path handles it.
        if c.dtd_enabled and self._echo_coherence is not None and self._aec is not None:
            self._dtd = AdaptiveDTD(
                k=c.dtd_k,
                weights=(c.dtd_weight_raw, c.dtd_weight_resid, c.dtd_weight_coh),
                confirm_frames=c.dtd_confirm_frames,
                warmup_frames=c.dtd_warmup_frames,
                chart_rel_floor=c.dtd_chart_rel_floor,
                chart_z_freeze=c.dtd_chart_z_freeze,
                chart_robust_seed=c.dtd_chart_robust_seed,
                chart_freeze_limit=c.dtd_chart_freeze_limit,
                persistent_charts=c.dtd_chart_persist,
            )
            log.info(
                "adaptive barge-in (fused z-score DTD) ACTIVE: K=%.1f weights=(%.1f,%.1f,%.1f) "
                "confirm=%d -- no fixed margin, self-calibrating per device",
                c.dtd_k, c.dtd_weight_raw, c.dtd_weight_resid, c.dtd_weight_coh,
                c.dtd_confirm_frames,
            )
        # L1 echo-floor gate needs a LEARNED floor to compare against, and the only
        # floor sources are the post-AEC residual-echo floor (_playback_floor_rms,
        # maintained only when AEC is on) and the quiet ambient floor (_ambient_rms,
        # maintained only when input_loudness_margin_db > 0). With NEITHER, the floor
        # stays 0.0 and _final_above_floor fails OPEN -- L1 is inert. That is fine for
        # a no-AEC (echo-free) setup, but if the gate is configured ON with no source
        # the no-op should be LOUD, not silent.
        if (
            c.final_floor_margin_db > 0.0
            and c.input_loudness_margin_db <= 0.0
            and self._aec is None
        ):
            log.warning(
                "final_floor_margin_db=%.1f is set but the L1 echo-floor gate is INERT: "
                "no learned floor (enable aec_enabled=true OR input_loudness_margin_db>0). "
                "Harmless on an echo-free setup; required for open-speaker echo rejection.",
                c.final_floor_margin_db,
            )
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

        # Windows-only: the WASAPI "Communications" category turns on the OS
        # AEC/NS/AGC the same way Teams gets it. Best-effort -- absent on
        # non-WASAPI hosts, so build it once and fail open if unavailable.
        extra_settings = None
        if self.config.capture_voice_comm:
            try:
                extra_settings = sd.WasapiSettings(communications=True)
                log.info("capture: requesting the WASAPI Communications (voice-comm) path")
            except Exception as exc:  # noqa: BLE001 - not WASAPI / old sounddevice
                log.info(
                    "capture_voice_comm set but the WASAPI path is unavailable (%s); on Linux "
                    "load PipeWire module-echo-cancel and point input_device at the EC source",
                    exc,
                )
                extra_settings = None

        def _open(device, samplerate):
            return sd.InputStream(
                channels=1,
                samplerate=samplerate,
                dtype="float32",
                blocksize=int(samplerate * 0.1),
                device=device,
                extra_settings=extra_settings,
            )

        self._stream_in = _RecoveringInputStream(
            attempts,
            opener=_open,
            on_state=self._on_capture_state,
            channels=1,
            block_seconds=self.config.block_sec,
        )
        self._stream_in.open()
        self._capture_sr = self._stream_in.actual_samplerate
        # Stateful anti-aliased resampler for the capture hot path (soxr ->
        # scipy polyphase -> linear). Replaces the old per-block np.interp, which
        # aliased content >8 kHz into the speech band and corrupted ASR features.
        self._resampler = (
            AudioResampler(self._capture_sr, preferred_sr, quality=self.config.resampler_quality)
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
            if self.config.record_playback_reference:
                import numpy as np

                ref_path = (
                    self._record_path[:-4] + ".ref.wav"
                    if self._record_path.endswith(".wav")
                    else self._record_path + ".ref.wav"
                )
                self._ref_recorder = WavRecorder(ref_path, self.config.sample_rate)
                self._ref_accum = np.zeros(0, dtype="float32")
                log.info("recording playback reference (replay) -> %s", ref_path)
        if self.config.input_calibrate:
            self._calibrate_input()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self._play_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._play_thread.start()
        # asr-tts-2: the second-pass worker (only when async + a recognizer were
        # wired in _build). Keeps the slow offline decode off the capture loop.
        if self._final_q is not None:
            self._final_thread = threading.Thread(target=self._final_worker, daemon=True)
            self._final_thread.start()

    def stop(self) -> None:
        self._running.clear()
        self._stop_speaking.set()
        self._drain_play_q()
        # Tear the live output stream down BEFORE pushing the sentinel / joining
        # the play thread. On a dead/stalled device the play thread can be blocked
        # in FIFO.write() and would never reach _play_q.get() to see the sentinel,
        # so the join below would hang (the classic "second Ctrl-C needed"
        # shutdown). _stop_speaking is already set (above), which is the
        # should_abort predicate that releases a blocked producer; flush()'s
        # notify_all also wakes it. We use stop()/close() (NOT abort) for a clean
        # device handback -- PortAudio stop() JOINS the callback so none fires
        # post-close. On a healthy/idle stream this is cheap. Emits no
        # barge_in_stop metric (teardown, not an interruption).
        with self._out_lock:
            if self._fifo is not None:
                self._fifo.flush()
            if self._out_stream is not None:
                try:
                    self._out_stream.stop()
                    self._out_stream.close()
                except Exception:  # noqa: BLE001 - device may be mid-teardown
                    pass
        self._play_q.put((None, None, 0, None))  # sentinel: wake the worker so it exits
        if self._final_q is not None:
            # Any finals still queued here are intentionally dropped on shutdown
            # (like _play_q) -- dispatching a final into a tearing-down runtime is
            # wrong, and the window is sub-second.
            try:
                self._final_q.put_nowait(None)  # sentinel: wake the second-pass worker
            except queue.Full:
                pass  # _running is already clear; it exits on its own next loop
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
        if self._final_thread is not None:
            self._final_thread.join(timeout=1.0)
            if self._final_thread.is_alive():
                log.warning("second-pass thread did not exit within 1.0s; proceeding")
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
        if self._ref_recorder is not None:
            log.info("recorded playback reference -> %s", self._ref_recorder.path)
            self._ref_recorder.close()
            self._ref_recorder = None
            self._ref_accum = None

    def speak(self, text: str, on_done: Optional[Callable[[], None]] = None) -> None:
        # Non-blocking: hand the utterance to the single playback worker. Keeping
        # one sink (instead of a thread per call) is what makes sentence-level
        # streaming play in order rather than on top of itself.
        if self._tts is None:
            if on_done:
                on_done()
            return
        # Expressive markup (opt-in): strip a leading [emotion:.. voice:.. rate:..]
        # tag off the text and carry the parsed directives to _synthesize, which
        # maps them to this utterance's (sid, speed). Default OFF -> no parsing,
        # byte-identical. A tag-only emission (nothing left to say) is dropped.
        directives = None
        if self.config.tts_markup:
            raw_text = text
            text, directives = parse_tts_markup(
                text,
                voices=self.config.tts_speaker_voices.keys(),
                emotions=self.config.tts_emotion_speed_map.keys(),
            )
            if raw_text != text or raw_text.lstrip().startswith("["):
                log.info(
                    "tts sanitize: %s",
                    json.dumps(
                        {
                            "raw": raw_text,
                            "spoken": text,
                            "directives": directives or {},
                            "status": "parsed" if directives else "unrecognized_tag",
                        },
                        sort_keys=True,
                        ensure_ascii=True,
                        default=str,
                    ),
                )
            if not text:
                if on_done:
                    on_done()
                return
        # Echo filter feed for the duck-then-confirm barge gate: what is about to
        # be synthesized is what a confirm-window partial may transcribe back as
        # the assistant's own (ducked) echo. Cheap ring; inert unless
        # barge_confirm_enabled consults it. getattr: markup fixtures build the
        # engine bypassing __init__, where the ring doesn't exist.
        spoken_ring = getattr(self, "_recent_spoken", None)
        if spoken_ring is not None:
            spoken_ring.append(text)
        self._enqueue_play(text, on_done, directives)

    def stop_speaking(self) -> None:
        # Cut the current utterance and discard whatever is queued behind it, so
        # a barge-in flushes pending speech instead of letting it play out.
        self._stop_speaking.set()
        # Bump the generation atomically with the stop (rc-3): every sentence
        # enqueued before now is invalidated, so one that slips past the drain
        # below (already dequeued at this instant) is skipped by the worker, and
        # an in-flight current sentence aborts on the generation mismatch even if
        # the worker's clear momentarily wipes _stop_speaking.
        with self._gen_lock:
            self._speak_gen += 1
        self._drain_play_q()
        # Cut the live audio by FLUSHING the playback FIFO rather than aborting
        # the stream. flush() drops every queued sample in one short lock, so the
        # very next PortAudio callback emits a silent zero-fill -- equivalent to
        # abort() discarding the device buffer, but the stream STAYS OPEN playing
        # silence (cheaper; avoids the abort->restart race). _stop_speaking (set
        # above) also early-returns the producer write() and releases a producer
        # blocked in FIFO.write via its should_abort predicate. Residual latency
        # is now one-two low-latency callback periods, comparable to abort()'s.
        cut = False
        with self._out_lock:
            if (
                self._out_stream is not None
                and self._fifo is not None
                and self._speaking.is_set()
            ):
                # Fade the tail to silence instead of a hard cut (de-click the
                # barge-in). The FIFO is sized in play_sr samples.
                fade = (
                    int(self._play_sr * self.config.barge_fade_ms / 1000.0)
                    if self._play_sr else 0
                )
                self._fifo.flush(fade)
                cut = True
        # Playback is being cut: drop the echo reference so the capture loop
        # doesn't keep gating barge-in against a level that is no longer audible.
        self._playback_level = 0.0
        self._last_playback_at = 0.0
        # A stop/barge ends any open confirm window and restores full volume for
        # the next reply (idempotent; the confirm path also restores on its own).
        self._end_barge_confirm()
        # AUTHORITATIVELY end the speaking state HERE (RC-2). A barge-in/stop is
        # "stop now"; ownership of the _speaking transition must NOT be left to the
        # playback worker's epilogue (sherpa.py finally, ~_speaking.clear()), which
        # only runs after _synthesize()/tts.generate() returns. If that native TTS
        # call wedges, the worker never clears _speaking, the capture loop keeps
        # `continue`-ing past ASR, and the assistant goes DEAF for the rest of the
        # session (observed: speaking=True for ~15 s after a 2 s reply, ending in
        # "playback thread did not exit within 1.0s"). So clear it here so ASR
        # resumes at once, re-arm the one-per-run barge latch (otherwise it stays
        # latched True and disarms further barge-ins), and reset the echo refs the
        # worker would otherwise reset. All idempotent -- the worker clearing them
        # again on its way out is harmless.
        self._speaking.clear()
        self._barge_in_fired_this_run = False
        self._last_speaking_end = time.monotonic()  # arm the L2 post-speaking refractory
        if self._echo_coherence is not None:
            self._echo_coherence.reset()
            if self._coh_ref_q is not None:
                self._coh_ref_q.clear()  # audio-bargein-8: drop un-ingested played blocks
        if self._dtd is not None:
            self._dtd.new_run()  # charts persist (2026-06-10); only the candidate run clears
        if self._far_ref is not None:
            self._far_ref.clear()
        if self._aec is not None:
            self._aec.reset()
        if self._aec_asr is not None:
            self._aec_asr.reset()
        # Stamp the *true* audible-stop instant (when the FIFO was flushed), not
        # when synthesis later notices the flag. Only when we actually flushed a
        # speaking stream, so a no-op stop (nothing playing) records nothing --
        # preserving the old aborted-guarded behavior.
        if cut:
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
                        # L3: the offline 2nd pass HALLUCINATES a plausible sentence
                        # from a short open-speaker echo clip ("BEING" -> "I."). Trust
                        # it only when it agrees with / clearly improves the streaming
                        # final; the discriminator keys on clip LENGTH, not energy. A
                        # rejected hallucination falls back to the (post-processed)
                        # streaming final, which the L1 echo-floor gate then drops.
                        return agreement_guard(
                            self._postprocess_final(raw_final),
                            text,
                            segment_sec=n / self.config.sample_rate,
                        )
                except Exception:  # noqa: BLE001 - fall back to the streaming final
                    log.debug("second-pass recognizer failed; using streaming final", exc_info=True)
        return self._postprocess_final(raw_final)

    def _finalize_and_dispatch(self, seg, raw_final: str, speech_end_ts, asr_seg=None) -> None:
        """Produce the FINAL transcript and deliver it (or its drop metric).

        Pulled out of the capture loop so the SAME logic can run either inline
        (no second pass, or ``asr_final_async`` off) or on the dedicated
        second-pass worker (:meth:`_final_worker`). It does the three heavy,
        capture-thread-hostile steps -- the offline second-pass decode
        (``_final_transcribe``), the L1 echo-floor gate, and the speaker-ID gate
        (CAM++ embedding) -- then dispatches at most one final per utterance.
        Every input is owned by the work item (``seg`` is an already-concatenated
        copy; ``speech_end_ts`` a captured ``perf_counter`` so SPEECH_END stays
        correctly backdated however late this runs), so it is safe off-thread.

        ``asr_seg`` (fix 2): the NS-RELAXED copy of the utterance for the offline
        2nd-pass decode under ``_apm_owns_ns`` (so the LLM-facing final also keeps
        the near-end words NS erased). The L1 echo-floor gate + the speaker-ID gate
        deliberately keep the NS-ON ``seg`` (their learned floor + enrolled
        embedding live in the NS-on domain; the louder NS-off signal would clear
        the floor too easily and re-open the echo-final cascade). ``None`` (every
        non-apm-owns-ns path) -> decode ``seg``, byte-identical."""
        final_text = self._final_transcribe(asr_seg if asr_seg is not None else seg, raw_final)
        log.info("asr final: %r (raw %r)", final_text, raw_final)
        if not self._final_above_floor(seg):
            # L1: at/near the device's learned echo/quiet floor -- the assistant's
            # own residual echo or ambient noise transcribed into words. Drop it
            # (this is what breaks the open-speaker echo-final self-interrupt
            # cascade).
            log.info(
                "dropping final %r -- at/near the learned echo/quiet floor "
                "(echo/ambient, not speech)", final_text,
            )
            self._cb.on_metric("echo_floor_rejected_final")
        elif self._should_act_on_final(seg):
            self._cb.on_metric(SPEECH_END, at=speech_end_ts)
            self._cb.on_final(final_text)
        else:
            log.info("dropping final %r -- speaker is not the enrolled user", final_text)
            self._cb.on_metric("speaker_rejected_final")

    def _maybe_setup_async_final(self) -> None:
        """Create the second-pass worker queue iff a 2nd-pass recognizer was built
        AND ``asr_final_async`` is on. Split out of ``_build`` so the gate -- the
        thing that decides "async worker vs byte-identical inline" -- is unit-
        testable without standing up every model. ``_final_q`` left None otherwise
        (the capture loop then finalizes inline). asr-tts-2."""
        if self._final_recognizer is not None and self.config.asr_final_async:
            self._final_q = queue.Queue(maxsize=8)
            log.info("second-pass final ASR runs ASYNC (off the capture thread)")

    def _enqueue_final(self, seg, raw_final: str, speech_end_ts, asr_seg=None) -> None:
        """Hand an endpointed utterance to the second-pass worker WITHOUT ever
        blocking the capture loop. On overflow (the worker is wedged/very slow --
        normally the queue sits near-empty), drop the OLDEST queued utterance to
        make room for this newer one, mirroring ``_play_q``'s drop-oldest
        backpressure. This preserves capture-order dispatch, which matters: the
        runtime's supersede is newest-ARRIVAL-wins, so a stale final arriving
        after a newer one would wrongly cancel the newer turn. Single producer
        (this capture thread), so after one ``get_nowait`` a slot is free."""
        try:
            self._final_q.put_nowait((seg, raw_final, speech_end_ts, asr_seg))
            return
        except queue.Full:
            pass
        log.warning("second-pass queue full; dropping the oldest pending final")
        try:
            self._final_q.get_nowait()
            # Make the drop visible in the run bundle, like the floor/speaker
            # drop paths -- otherwise a wedged worker silently eats turns.
            self._cb.on_metric("second_pass_queue_overflow_dropped_final")
        except queue.Empty:
            pass  # the worker just drained one; a slot is free now
        try:
            self._final_q.put_nowait((seg, raw_final, speech_end_ts, asr_seg))
        except queue.Full:
            # Only reachable if a sentinel raced in; never block capture.
            log.warning("second-pass queue still full; finalizing inline")
            self._finalize_and_dispatch(seg, raw_final, speech_end_ts, asr_seg)

    def _final_worker(self) -> None:
        """Drain the second-pass work queue, finalizing one utterance at a time.

        Single consumer -> finals dispatch in capture order even though the
        offline decode is slow; running off the capture thread is the whole point
        (asr-tts-2): the real-time loop keeps reading the mic, updating the echo
        reference, and servicing barge-in while this decodes. A broad guard keeps
        the worker alive across a bad turn -- a finalize failure drops that one
        turn, it never wedges the queue."""
        assert self._final_q is not None
        while self._running.is_set():
            try:
                item = self._final_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:  # shutdown sentinel
                break
            seg, raw_final, speech_end_ts, asr_seg = item
            try:
                self._finalize_and_dispatch(seg, raw_final, speech_end_ts, asr_seg)
            except Exception:  # noqa: BLE001 - never let the worker die on one turn
                log.exception("second-pass finalize failed; dropping this turn")

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
    def _calibrate_input(self) -> None:
        """Startup ambient calibration: read ``input_calibrate_sec`` of room tone
        off the open stream (BEFORE the capture loop starts) and set the AGC's
        noise floor to this device's measured quiet level -- the device-generic
        operating point. Best-effort: any read error just ends calibration early
        and the engine proceeds with the configured defaults. Also surfaces an
        ``input_clipping`` metric into the run bundle if the ADC is already hot."""
        import numpy as np

        sec = float(self.config.input_calibrate_sec)
        if sec <= 0.0 or self._stream_in is None:
            return
        block_sec = self.config.block_sec
        n_blocks = max(1, int(round(sec / block_sec)))
        blocks = []
        for _ in range(n_blocks):
            try:
                audio, _ = self._stream_in.read(int(self._capture_sr * block_sec))
            except Exception:  # noqa: BLE001 - end calibration early, proceed
                break
            s = np.asarray(audio, dtype="float32").reshape(-1)
            if self._resampler is not None:
                s = self._resampler.process(s)
            if s.size:
                blocks.append(s)
        if not blocks:
            return
        cal = compute_input_calibration(blocks)
        self._last_calibration = cal
        log.info(
            "input calibration: ambient_rms=%.4f -> noise_floor=%.4f peak=%.3f clip=%.1f%% (%d blocks)",
            cal["ambient_rms"], cal["noise_floor_rms"], cal["peak"],
            cal["clipping_fraction"] * 100.0, cal["n_blocks"],
        )
        if self._input_agc is not None:
            self._input_agc.noise_floor_rms = cal["noise_floor_rms"]
        if cal["clipping_fraction"] > 0.02:
            self._clip_warned = True  # don't double-warn in the heartbeat
            log.warning(
                "input is CLIPPING during calibration (%.1f%% railed) -- the boost-only AGC "
                "cannot fix a hot ADC; LOWER the OS mic level / disable 'mic boost'.",
                cal["clipping_fraction"] * 100.0,
            )
            try:
                self._cb.on_metric("input_clipping")
            except Exception:  # noqa: BLE001 - metric is best-effort
                pass

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
        # RC-5 observability: track voiced speech DURING playback that the barge
        # gate rejected, so a "user kept talking but nothing fired" failure is
        # visible in the run bundle instead of being silently dropped.
        rejected_run = 0.0
        rejected_flagged = False
        block_sec = self.config.block_sec
        # Temporal confirmation for barge-in: a windowed sustain over the per-frame
        # DTD/gate eligibility (replaces the old leaky `voiced_run` accumulator that
        # intermittent fires starved -- see core/engines/_dtd.BargeSustain).
        barge_sustain = BargeSustain(
            window_sec=self.config.barge_in_sustain_window_sec,
            block_sec=block_sec,
            min_voiced_sec=self.config.barge_in_min_speech_sec,
        )
        # Rolling buffer of the current (non-speaking) ASR segment, used to embed
        # the speaker when an endpoint fires so input can be gated on identity.
        # Capped so a long monologue can't grow memory; the tail is what the
        # speaker model needs. Reset on every endpoint and on decode-recovery.
        utterance: list = []
        # fix 2: parallel NS-off utterance (populated only under the relaxed-NS
        # tap) so the offline 2nd pass decodes the NS-relaxed audio too.
        asr_utterance: list = []
        utterance_len = 0
        max_utterance = int(self.config.sample_rate * 10)
        # Diagnostics: cumulative + per-interval counters for the 2 s heartbeat.
        total_blocks = partials = finals = 0
        beat_blocks = 0
        beat_level = 0.0
        beat_clip = 0.0
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
                # before the recognizer sees them. AGC (when on) normalizes the
                # level dynamically and takes precedence over the static gain.
                if self._input_agc is not None:
                    samples = self._input_agc.process(samples)
                elif self.config.input_gain != 1.0:
                    samples = apply_gain_soft_limit(samples, self.config.input_gain)
                if self._resampler is not None:
                    samples = self._resampler.process(samples)
                # AEC on the 16 kHz block, AFTER resampling and BEFORE the denoiser
                # and every consumer: subtract the assistant's own played TTS (the
                # far-end, read from the ring at the speaker->mic delay) so the loop
                # doesn't self-interrupt. No-op when AEC is off; passthrough-on-error
                # inside, so it can never crash this daemon thread. Output length may
                # differ (the adaptive filter frames internally) -- consumers accept
                # any length, like the resampler/denoiser.
                # Tee the RAW pre-AEC mic block for the coherence barge detector.
                # Coherence measures correlation between the mic and the played
                # reference; AEC REMOVES that correlation, so the detector must see
                # the echo-bearing raw mic, not the suppressed residual (feeding it
                # the residual made cancelled echo read as "user" and self-interrupt).
                # ASR/VAD/the level-floor fallback keep using the post-AEC samples.
                # No copy needed: AEC.process_16k / denoiser.process_16k RETURN new
                # arrays (never mutate the input), and ``samples`` below is reassigned
                # to those new arrays -- so this reference keeps pointing at the
                # untouched pre-AEC block for the rest of the iteration.
                mic_raw = samples
                # fix 2: NS-off recognizer tap. Stays None (the recognizer then
                # reads the NS-on ``samples``, byte-identical) unless the relaxed
                # APM tap is built (only under _apm_owns_ns).
                asr_samples = None
                if self._aec is not None:
                    # Auto-calibrate the far->near delay from the true-playback-
                    # aligned far (read at delay 0) vs the raw mic: the calibrator
                    # self-gates on far energy + correlation, so idle/uncorrelated
                    # blocks leave the operating delay unchanged. Drives the read
                    # below, so a mis-set seed self-corrects within ~1 window.
                    if self._aec_delay_cal is not None:
                        far0 = self._far_ref.read(samples.shape[0], 0)
                        if far0 is not None and far0.size:
                            self._aec_delay_cal.observe(mic_raw, far0)
                            self._aec_ref_delay = self._aec_delay_cal.current_delay_samples()
                    far = self._far_ref.read(samples.shape[0], self._aec_ref_delay)
                    # ONLY cancel when the assistant is actually playing (the
                    # far-end reference has energy). With no recent playback the
                    # far ring is ~zeros and the deep DTLN canceller would just
                    # process CLEAN near-end speech -- a neural net can distort it,
                    # which garbles ASR on the user's own (echo-free) voice. So
                    # skip it then: nothing to cancel, and the input reaches ASR
                    # untouched. (Also saves the DTLN cost on every idle block.)
                    far_energetic = (
                        far is not None and far.size > 0
                        and float(np.sqrt(np.mean(
                            np.asarray(far, dtype="float64") ** 2
                        ))) > 1e-4
                    )
                    if self._apm_always_on:
                        # WebRTC APM runs on EVERY block so its NS/AGC/HPF also
                        # clean the user's own (echo-free) idle utterance -- the
                        # desktop analog of the OS voice-comm path. When nothing is
                        # playing the far ref is ~zeros, so the echo canceller
                        # self-cancels to a no-op (measured idle passthrough ~93%).
                        if far is None:
                            far = np.zeros(samples.shape[0], dtype="float32")
                        samples = self._aec.process_16k(samples, far)
                        if self._aec_asr is not None:
                            # Same near (pre-AEC mic_raw) + far + delay as the
                            # primary APM, but ML NS OFF -> the user's near-end words
                            # survive for the recognizer. process_16k returns a new
                            # array and never mutates mic_raw. The GTCRN denoiser is
                            # skipped under _apm_owns_ns, so this stays parallel to
                            # ``samples`` down to the recognizer feed.
                            asr_samples = self._aec_asr.process_16k(mic_raw, far)
                    elif far_energetic:
                        samples = self._aec.process_16k(samples, far)
                # Speech denoise on the 16 kHz block, AFTER resampling and BEFORE
                # every consumer (recorder, accept_waveform, the speaker embedder,
                # the VAD). When no denoiser is built this is a zero-cost identity,
                # so the path stays byte-identical to no-denoise. Passthrough-on-
                # error inside, so it can never crash this daemon thread.
                # Skip the GTCRN denoiser when the always-on APM already owns noise
                # suppression (it cleaned every block above) -- double-NS over-
                # suppresses. A playback-gated APM does NOT clean idle blocks, so
                # the denoiser still runs then.
                if self._denoiser is not None and not self._apm_owns_ns:
                    samples = self._denoiser.process_16k(samples)

                if self._recorder is not None:
                    self._recorder.write(samples)
                    # Frame-aligned reference: one ref frame per mic frame (silence
                    # when not playing), so the .ref.wav indexes 1:1 with the mic.
                    if self._ref_recorder is not None:
                        self._write_reference_frame(samples.shape[0])

                total_blocks += 1
                beat_blocks += 1
                if samples.size:
                    beat_level += float(np.sqrt(np.mean(samples * samples)))
                    # Clipped-sample fraction this block (rail-pinned input).
                    beat_clip += float(np.mean(np.abs(samples) >= 0.98))
                now = time.monotonic()
                if now - last_beat >= 2.0:
                    avg = beat_level / max(beat_blocks, 1)
                    avg_clip = beat_clip / max(beat_blocks, 1)
                    log.debug(
                        "capture heartbeat: blocks=%d avg_rms=%.4f clip=%.1f%% underruns=%d "
                        "partials=%d finals=%d speaking=%s",
                        total_blocks, avg, avg_clip * 100.0, self._underrun_blocks,
                        partials, finals, self._speaking.is_set(),
                    )
                    if avg < 1e-4:
                        log.warning(
                            "input is ~silent (avg_rms=%.6f) -- wrong mic, muted, or no "
                            "permission? run `python -m sounddevice` to list devices", avg,
                        )
                    # Input-clipping diagnostic: sustained rail-pinning (>2% of
                    # samples on average) shreds the waveform -> garbled STT. Warn
                    # once so a hot OS mic gain is diagnosed up front, not silently
                    # transcribed as nonsense.
                    if avg_clip > 0.02 and not self._clip_warned:
                        self._clip_warned = True
                        log.warning(
                            "input is CLIPPING (%.1f%% of samples railed, avg_rms=%.3f) -- mic "
                            "gain too HOT; the recognizer is fed a shredded waveform. Lower the "
                            "OS mic input level / disable 'mic boost' (target rms ~0.1-0.2).",
                            avg_clip * 100.0, avg,
                        )
                        # Surface it into the run bundle (summary.json), not just the
                        # log -- a hot ADC is the #1 silent STT-garbler.
                        try:
                            self._cb.on_metric("input_clipping")
                        except Exception:  # noqa: BLE001 - metric is best-effort
                            pass
                    self._cb.on_heartbeat()
                    last_beat, beat_blocks, beat_level, beat_clip = now, 0, 0.0, 0.0

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
                        self._ambient_rms = 0.9 * _a + 0.1 * _r       # fall fast to a new floor
                    elif _r < _a * 2.0:
                        self._ambient_rms = 0.995 * _a + 0.005 * _r   # genuine drift (<+6 dB): track slowly
                    # else: _r >> floor -- a barge / echo burst, NOT the ambient
                    # floor. FREEZE the rise (don't update) so a SUSTAINED talk-over
                    # can't drag the floor up to itself and thereby raise the barge
                    # bar above the user -- the observed missed-barge during a long
                    # answer. Mirrors the coherence chart's "ignore upward excursions"
                    # rule, so the floor self-calibrates to the true quiet level on
                    # any device/room with no manual tuning.

                # Barge-in watch while the assistant is speaking.
                # While the assistant is speaking we NEVER feed ASR (so it can't
                # transcribe its own TTS). Barge-in watch is gated by
                # ``barge_in_enabled`` -- off until AEC, so the loud TTS leaking
                # into an open-speaker mic can't self-interrupt.
                if self._speaking.is_set():
                    # audio-bargein-8: ingest the played reference blocks the audio
                    # callback queued (off its real-time thread) into the coherence
                    # detector HERE, on the capture thread -- so note_playback's lock
                    # (held by decide() while it concatenates the whole ring) never
                    # contends with the real-time callback. Drains FIFO-ordered;
                    # note_playback resamples play_sr->16k exactly as the old inline
                    # call did, so the reference content is unchanged.
                    if self._coh_ref_q is not None and self._echo_coherence is not None:
                        _q = self._coh_ref_q
                        _play_sr = self._play_sr
                        while _play_sr:
                            try:
                                _blk = _q.popleft()
                            except IndexError:
                                break
                            self._echo_coherence.note_playback(_blk, _play_sr)
                            # Tee the same played reference into the replay recorder.
                            if self._ref_recorder is not None:
                                self._accumulate_reference(_blk, _play_sr)
                    # Consume the silent->speaking re-arm signalled by the playback
                    # loop: clear the BargeSustain window so the prior reply's fires
                    # can't carry into this reply's onset (2026-06-10 self-interrupt
                    # fix; the floors were reset on the playback thread).
                    if self._barge_sustain_reset_pending:
                        self._barge_sustain_reset_pending = False
                        barge_sustain.reset()
                        rejected_run = 0.0
                        # Word-cut: snapshot the recognizer base at this reply's
                        # first speaking block so only words transcribed AFTER the
                        # reply started can cut (the prior final was reset out of the
                        # stream at reply-end, so usually empty; robust to residue).
                        if self._barge_word_cut_active():
                            try:
                                self._word_cut_base = (
                                    recognizer.get_result(stream) or ""
                                ).strip()
                            except Exception:  # noqa: BLE001 - stream hiccup != crash
                                self._word_cut_base = ""
                            self._word_cut_fed_stream = False
                            # Fresh reply -> fresh funnel window (emitted at
                            # reply end as one "word-cut funnel:" line).
                            self._word_cut_quiet_run = 0
                            self._wc_stats = {}
                            self._wc_win = {}
                            self._wc_reply_active = True
                    # The playback worker marks _speaking at synth start, before the
                    # first audible block necessarily reaches PortAudio. Until a real
                    # playback reference exists, DTD/coherence/ref-floor decisions are
                    # ref-empty and can only produce misleading self-interrupts or
                    # "barge-in REJECTED during playback" diagnostics. Keep ASR gated
                    # for this committed reply, but do not arm playback-time barge-in
                    # until audio is actually audible. If a previous queued sentence's
                    # tail is still audible, a fresh playback stamp keeps the watch armed.
                    if not self._barge_watch_active():
                        barge_sustain.reset()
                        rejected_run = 0.0
                        rejected_flagged = False
                        continue
                    # CONTINUOUS no-duck word-cut (OS echo-cancel path). Feed THIS
                    # playback block to the streaming recognizer on the OS-cancelled
                    # mic and hard-cut the moment enough NEW non-own-speech words (or
                    # a stop command) appear since the reply started. No duck is ever
                    # opened, so playback can't pump; word content -- not level --
                    # decides. continue BEFORE the acoustic duck/sustain machinery so
                    # that path (incl. its RC-5 rejected-branch duck-open) is fully
                    # bypassed here; _duck_gain stays 1.0. Legacy/APM paths never
                    # enter (flag off or self._aec is not None) -> byte-identical.
                    if (
                        self.config.barge_in_enabled
                        and self._barge_word_cut_active()
                        and recognizer is not None
                        and stream is not None
                    ):
                        self._barge_word_cut_step(
                            recognizer, stream,
                            asr_samples if asr_samples is not None else samples,
                            now,
                        )
                        barge_sustain.reset()
                        rejected_run = 0.0
                        continue
                    # Duck-then-confirm: a prior acoustic trigger ducked playback
                    # and opened a confirm window. Feed THIS block to the streaming
                    # recognizer and look for real transcribed words (the user) vs
                    # nothing/own-echo (a false trigger). Handled BEFORE the floor
                    # updates below so the ducked echo can't be learned as the
                    # floor, and instead of the sustain machinery -- word evidence
                    # decides now, not more acoustic trips.
                    if self._barge_confirm_active():
                        # Decode the confirm window from the NS-off tap (words
                        # survive); the step's internal DTD echo obs still uses mic_raw.
                        self._barge_confirm_step(
                            recognizer, stream,
                            asr_samples if asr_samples is not None else samples,
                            now, mic_raw=mic_raw,
                        )
                        barge_sustain.reset()
                        rejected_run = 0.0
                        continue
                    # Auto-calibrate the post-AEC residual echo+noise floor on EVERY
                    # playback block (the assistant's own cancelled echo). The barge
                    # gate keys off this floor, so the interrupt threshold tracks the
                    # current speaker volume + room noise online -- no manual
                    # calibration, and the steady echo can't self-interrupt.
                    if self._aec is not None:
                        # (The far->near read delay is auto-calibrated up front by
                        # AecDelayCalibrator via normalized cross-correlation -- see
                        # the observe() call before the far read -- which replaced
                        # the coherence-median feedback that never converged on the
                        # open speaker: un-normalized, no accept-gate, reset every
                        # reply. See docs/session_2026-07-04.)
                        self._update_playback_floor(rms(samples))
                        # ALSO track the echo-only floor on the RAW pre-AEC mic. The
                        # barge energy-confirmation keys off THIS (not the post-AEC
                        # residual): DTLN suppresses the user's voice during double-
                        # talk, so the residual barely rises on a real talk-over and
                        # the gate rejected every barge (live: 0 fired / 7 rejected).
                        # The raw mic is untouched by AEC, so a talk-over genuinely
                        # adds the user's energy on top of the steady echo floor.
                        self._update_raw_playback_floor(rms(mic_raw))
                    if self.config.barge_in_enabled and self._vad is not None:
                        self._vad.accept_waveform(samples)
                        # Debounce: ignore triggers for a short window after a
                        # barge-in / reported storm so a flapping VAD gate (TTS
                        # echo with no AEC) collapses into a single interrupt.
                        if (
                            now < self._barge_in_suppressed_until
                            or self._in_post_speaking_refractory(now)
                        ):
                            barge_sustain.reset()
                            rejected_run = 0.0
                        else:
                            # Echo-learning tap (2026-06-10): on a VAD-QUIET
                            # playback block, feed the DTD charts an echo-only
                            # observation. decide() runs only on VAD-speech
                            # blocks (the eligibility gate below), so without
                            # this the charts' learning diet is biased toward
                            # exactly the blocks most likely to contain the
                            # USER; the quiet blocks are the most reliably
                            # echo-only samples there are. Skipped during the
                            # suppress/refractory window above (echo tail mixed
                            # with the user's own barge speech).
                            if self._dtd is not None and not self._vad.is_speech_detected():
                                det = self._echo_coherence
                                incoh = 0.0
                                if det is not None:
                                    det.decide(mic_raw)
                                    incoh = float(det.last_incoherent_fraction)
                                # Residual feature source MUST match the gate's
                                # (_dtd_residual_level): under _apm_owns_ns the
                                # chart learns the raw-mic echo floor, so the gate's
                                # raw-derived resid scores against the right
                                # baseline (a post-NS-learned chart vs a raw-read
                                # value would self-interrupt on echo-only).
                                self._dtd.observe_echo(
                                    rms(mic_raw),
                                    self._dtd_residual_level(samples, mic_raw),
                                    incoh,
                                )
                            # Windowed temporal confirmation (BargeSustain): the
                            # per-frame eligibility FLICKERS on a real open-speaker
                            # talk-over (breath/pauses + AEC suppressing the user
                            # mid-double-talk), so a cut needs enough eligible blocks
                            # within a short trailing window -- NOT N consecutive, and
                            # NOT a leaky accumulator the flicker starves (the
                            # run-20260609-203236 "needs a shout" bug: the DTD fired on
                            # 3 of 5 turn-2 blocks but voiced_run *= 0.5 never reached
                            # the threshold). The bounded window keeps a sporadic echo
                            # leak from ever summing to a self-interrupt.
                            eligible = self._barge_in_fire_eligible(samples, mic_raw)
                            if barge_sustain.update(eligible):
                                barge_sustain.reset()
                                rejected_run = 0.0
                                if (
                                    self.config.barge_confirm_enabled
                                    and recognizer is not None
                                    and stream is not None
                                ):
                                    # Word gate: do NOT cut on acoustics alone --
                                    # duck playback and require transcribed words
                                    # within the confirm window. No latch burned:
                                    # an unconfirmed (echo) trigger must not
                                    # disarm a later REAL talk-over this run.
                                    self._begin_barge_confirm(recognizer, stream, now)
                                else:
                                    # Legacy immediate hard-fire (byte-identical
                                    # when barge_confirm_enabled is False).
                                    # Latch: one barge-in per speaking run. The
                                    # 0.5s suppress window still debounces; the
                                    # latch caps the whole run so self-echo can't
                                    # re-fire.
                                    self._barge_in_fired_this_run = True
                                    self._barge_in_suppressed_until = (
                                        now + max(0.0, self.config.barge_in_suppress_sec)
                                    )
                                    log.info("barge-in detected")
                                    self._cb.on_barge_in()
                            elif eligible:
                                rejected_run = 0.0
                            else:
                                # RC-5: the gate rejected this block. If the VAD still
                                # hears SUSTAINED speech during playback and no barge
                                # has fired this run, the user is likely talking over
                                # the assistant but the echo/level gate is rejecting
                                # them -- surface it (once per episode) instead of
                                # dropping it silently, so the failure shows up in the
                                # run bundle (metric) + watchdog rather than a clean log.
                                if (
                                    not self._barge_in_fired_this_run
                                    and self._vad.is_speech_detected()
                                ):
                                    # (Reverted 2026-07-04: the loose duck here opened
                                    # on a fraction of K and PUMPED the volume on echo
                                    # without cutting -- see the permanent-plan doc. The
                                    # real fix is the capture path, not a looser trigger.)
                                    rejected_run += block_sec
                                    # OS-capture word-gate duck-open (2026-07-05): when
                                    # the in-app AEC/DTD are OFF (OS echo-cancel owns
                                    # cancellation) the acoustic eligibility gate has no
                                    # user/echo discriminant, so a real talk-over lands
                                    # HERE as "rejected" voiced speech. If the word gate
                                    # is on, treat SUSTAINED, loud-enough voiced speech
                                    # during playback as a loose DUCK trigger and let
                                    # transcribed words confirm-or-restore. The
                                    # passes_output_margin check is the ANTI-PUMP key:
                                    # the OS canceller leaves a residual echo ~-26 dB
                                    # below the TTS reference (measured run-230222) which
                                    # the VAD reads as speech; requiring the block within
                                    # barge_confirm_duck_margin_db of the playback level
                                    # admits the near user (~-14 dB) and rejects that echo
                                    # floor, so echo no longer opens a (pumping) duck.
                                    # Retry-suppress + the confirm word gate bound the
                                    # rest. Decoupled from rejected_flagged so an early
                                    # echo can't latch out a later real talk-over. Legacy
                                    # path (word gate off) is byte-identical.
                                    if (
                                        rejected_run >= self.config.barge_in_min_speech_sec
                                        and self.config.barge_confirm_enabled
                                        and recognizer is not None
                                        and stream is not None
                                        and not self._barge_confirm_active()
                                        and passes_output_margin(
                                            rms(samples), self._playback_level,
                                            margin_db=self.config.barge_confirm_duck_margin_db,
                                        )
                                    ):
                                        self._begin_barge_confirm(recognizer, stream, now)
                                        rejected_run = 0.0
                                    elif (
                                        rejected_run >= self.config.barge_in_min_speech_sec
                                        and not rejected_flagged
                                    ):
                                        rejected_flagged = True
                                        log.info(
                                            "barge-in REJECTED: %.1fs of voiced speech during "
                                            "playback did not trip the gate (talk-over ignored?)",
                                            rejected_run,
                                        )
                                        self._cb.on_metric("barge_in_rejected")
                                else:
                                    rejected_run = 0.0
                    continue
                # Not speaking -> reset the rejected-talk-over episode tracking.
                rejected_run = 0.0
                rejected_flagged = False
                # Playback ended (naturally or via stop) while a confirm window
                # was still open: close it so the duck gain never leaks into the
                # next reply. Idempotent; stop_speaking() also restores.
                if self._barge_confirm_active():
                    self._end_barge_confirm()
                # One-shot ADR-0013 funnel summary for the reply that just ended:
                # everything the word-cut path saw (or didn't) in a single line,
                # so a live run scores from the bundle alone -- run-20260706-231226
                # produced ZERO word-cut signal and was undiagnosable post-hoc.
                if getattr(self, "_wc_reply_active", False):
                    self._wc_reply_active = False
                    self._emit_word_cut_funnel()
                # Word-cut fed the recognizer during playback (mostly the assistant's
                # own echo). If the reply ended with NO cut, drop that accumulated
                # text so it can't prefix the next real final. After a cut the flag
                # was cleared in _barge_word_cut_step (user's words stay as pre-roll),
                # so this is a no-op then.
                if getattr(self, "_word_cut_fed_stream", False):
                    self._word_cut_fed_stream = False
                    self._word_cut_base = ""
                    try:
                        recognizer.reset(stream)
                    except Exception:  # noqa: BLE001 - reset is best-effort
                        pass
                    last_partial = ""
                    last_voiced_ts = None

                try:
                    utterance.append(samples)
                    utterance_len += samples.size
                    if self._aec_asr is not None:
                        asr_utterance.append(
                            asr_samples if asr_samples is not None else samples
                        )
                    while utterance_len > max_utterance and len(utterance) > 1:
                        utterance_len -= utterance[0].size
                        utterance.pop(0)
                        if self._aec_asr is not None and asr_utterance:
                            asr_utterance.pop(0)  # keep the NS-off buffer aligned
                    # fix 2: the streaming recognizer reads the NS-off tap so
                    # near-end words survive; the utterance buffer (seg) + the
                    # floor/speaker gates keep the NS-on ``samples``.
                    stream.accept_waveform(
                        self.config.sample_rate,
                        asr_samples if asr_samples is not None else samples,
                    )
                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)
                    text = recognizer.get_result(stream)
                    if text and text != last_partial:
                        last_partial = text
                        # The result advanced -> speech is actively arriving.
                        # Record the instant (perf_counter, matching the metrics
                        # clock) so a later endpoint can backdate SPEECH_END to
                        # here, exposing the trailing-silence wait (lat-1).
                        _now = time.perf_counter()
                        # Feed the inter-partial gap (a MID-utterance pause, not the
                        # final end-of-turn silence) to the adaptive endpoint
                        # learner: it learns THIS speaker's pause distribution so the
                        # commit floor adapts per session instead of a fixed number.
                        if last_voiced_ts is not None and self._endpoint_policy is not None:
                            self._endpoint_policy.observe_pause(_now - last_voiced_ts)
                        last_voiced_ts = _now
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
                        # fix 2: NS-off segment for the 2nd pass. None off the
                        # relaxed-NS path -> _finalize decodes `seg` (byte-identical).
                        asr_seg = (
                            np.concatenate(asr_utterance)
                            if (self._aec_asr is not None and asr_utterance) else None
                        )
                        utterance, utterance_len = [], 0
                        asr_utterance = []
                        if raw_final.strip():
                            finals += 1
                            # Finalize (second-pass re-transcription, when
                            # configured, + the echo-floor/speaker gates) and
                            # dispatch the single final. asr-tts-2: when the async
                            # worker is live, hand it the segment so the heavy
                            # offline decode never stalls this real-time loop;
                            # otherwise finalize inline (legacy, byte-identical).
                            if self._final_q is not None:
                                self._enqueue_final(seg, raw_final, speech_end_ts, asr_seg)
                            else:
                                self._finalize_and_dispatch(seg, raw_final, speech_end_ts, asr_seg)
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
                    asr_utterance = []
                    # Clear the denoiser's streaming state too, so a recovered
                    # stream starts the front-end fresh (best-effort; no-op when
                    # there's no denoiser).
                    if self._denoiser is not None:
                        self._denoiser.reset()
                    # Same for the AEC adaptive state + its far-end ring, so a
                    # recovered stream starts the canceller fresh (no stale echo
                    # tail subtracted from the first recovered block).
                    if self._aec is not None:
                        self._aec.reset()
                        self._far_ref.clear()
                    if self._aec_asr is not None:
                        self._aec_asr.reset()
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

    def _enqueue_play(
        self,
        text: str,
        on_done: Optional[Callable[[], None]],
        directives: Optional[dict] = None,
    ) -> None:
        # Stamp the sentence with the current generation (rc-3): if a barge bumps
        # the generation before this sentence is dequeued, the worker drops it.
        # ``directives`` (opt-in expressive markup) rides alongside to _synthesize.
        gen = self._speak_gen
        item = (text, on_done, gen, directives)
        try:
            self._play_q.put_nowait(item)
        except queue.Full:
            # Drop the oldest queued sentence rather than block or lag playback.
            try:
                self._play_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._play_q.put_nowait(item)
            except queue.Full:
                if on_done:
                    on_done()

    def _drain_play_q(self) -> None:
        try:
            while True:
                self._play_q.get_nowait()
        except queue.Empty:
            pass

    def _claim_utterance(self, item_gen: int) -> Optional[int]:
        """Claim a dequeued sentence for playback, or reject it as stale (rc-3).

        Returns the utterance's generation -- and CLEARS ``_stop_speaking`` -- when
        the sentence is current (``item_gen == self._speak_gen``). Returns ``None``
        WITHOUT touching ``_stop_speaking`` when a barge/stop has bumped the
        generation since the sentence was enqueued, so the pending stop is not
        wiped and the playback worker skips the stale sentence. Factored out of
        ``_playback_loop`` (which needs a real device) so this load-bearing
        wipe-race guard is unit-testable.
        """
        if item_gen != self._speak_gen:
            return None
        self._stop_speaking.clear()
        return item_gen

    def _audio_cb(self, outdata, frames, time_info, status) -> None:
        """PortAudio output callback -- runs on a HIGH-PRIORITY audio thread.

        This is the single place the far-end AEC reference is pushed, and that is
        the whole point of the callback rewrite: the block we tee here is the EXACT
        block PortAudio is about to play, so :class:`FarEndRing`'s write head tracks
        TRUE acoustic playback (not the producer's run-ahead into a blocking
        ``out.write()``). That collapses the far->near lag from a 50..1200 ms
        output-buffer race down to the small, STABLE physical speaker->mic +
        fixed-output latency -- inside DTLN's +/-60 ms tolerance -- so the deep
        canceller works live.

        HARD REAL-TIME RULES (kept as tight as the design allows): no Python
        logging, no f-strings, no exception may escape into PortAudio (bare
        try/except around the whole body), no ``_out_lock``/``_tts_lock``, no
        ``_play_q``, no ``FIFO.write``, no blocking wait. ``status`` (underflow) is
        ignored -- an underrun is a silent zero-fill from ``read_into``, never a
        stall.

        Locks taken here are all short copy locks: the FIFO's own (+ a
        non-blocking notify) and ``FarEndRing``'s (a microsecond slice copy,
        shared with the capture thread's cheap ring read). The coherence
        reference is NO LONGER ingested here (audio-bargein-8): instead of calling
        ``EchoCoherenceDetector.note_playback`` -- whose lock the capture thread
        holds while ``decide`` concatenates the whole reference ring -- the played
        block is handed to the capture thread via a lock-free SPSC deque
        (``_coh_ref_q``) and ingested there. That removes the only contended lock
        from this real-time thread, so ``coherence_ring_ms`` can be raised (e.g.
        per-profile audio tuning) without risking an audio-callback stall.
        Allocation-light but not alloc-free: one small play_sr->16k resample for
        the far ring + one block copy for the coherence hand-off, sub-100us."""
        view = outdata[:, 0]
        try:
            fifo = self._fifo
            if fifo is None:
                view[:] = 0.0
                return
            # Drain the FIFO into the device buffer (zero-fills any underrun tail)
            # and learn how many of those samples are REAL audio (vs zero-fill) so
            # we tee only the played audio into the echo refs.
            n = fifo.read_into(view)
            # Duck-then-confirm barge gate: while a confirm window is open the
            # capture thread sets _duck_gain < 1.0 and playback is attenuated
            # IN PLACE, BEFORE the echo-reference tees below -- so the far-end
            # AEC reference and the coherence reference describe the DUCKED
            # signal that actually leaves the speaker. Real-time safe: one float
            # read + one in-place multiply, no allocation. 1.0 = no-op.
            duck = self._duck_gain
            if duck != 1.0 and n > 0:
                view[:n] *= duck
            # Underrun diagnostic: the FIFO ran dry PARTWAY through this block while
            # still speaking -> PortAudio plays the zero-filled tail as a gap (the
            # buzzy/glitchy artifact). Count only -- no logging on this thread. A
            # full empty read (n == 0) is the normal end-of-utterance, not a glitch.
            if 0 < n < frames and self._speaking.is_set():
                self._underrun_blocks += 1
            if n > 0:
                played = view[:n]
                # Mirror the producer's old per-chunk bookkeeping, but now driven
                # by ACTUAL playback: the level EWMA (barge-in reference), the
                # coherence echo reference, and the AEC far-end ring.
                self._note_playback_level(played)
                self._last_playback_at = time.monotonic()
                # audio-bargein-8: hand the played block to the capture thread via
                # a lock-free deque instead of calling note_playback HERE -- that
                # takes the coherence lock (and the capture thread holds it while
                # concatenating the whole reference ring in decide()), the one
                # contended lock on this real-time audio thread. COPY: `played` is
                # a view into the device buffer PortAudio reuses, so it must be
                # snapshotted before it leaves this callback.
                if self._coh_ref_q is not None and self._play_sr:
                    self._coh_ref_q.append(np.array(played, dtype=np.float32, copy=True))
                if self._far_ref is not None:
                    # Tee the just-played block into the AEC far ring at 16 kHz.
                    # The ring write head now == the true playback position, which
                    # is what makes the far-end reference align for DTLN.
                    ref16 = (
                        _resample_linear(played, self._play_sr, self.config.sample_rate)
                        if self._play_sr != self.config.sample_rate
                        else played
                    )
                    self._far_ref.push(ref16)
                # Stamp TTS_FIRST_AUDIO at the TRUE first-played instant (moved
                # here from the producer so the metric means "first audible", and
                # a flushed/stopped utterance that never plays never stamps it).
                if self._first_audio_pending:
                    self._first_audio_pending = False
                    self._cb.on_metric(TTS_FIRST_AUDIO)
        except Exception:  # noqa: BLE001 - a transient must never kill the audio thread
            pass

    def _barge_watch_active(self) -> bool:
        """Whether playback-time barge-in has a real acoustic reference.

        ``_speaking`` flips at synthesis start so ASR is gated as soon as the
        assistant has committed to a reply, but the first audible block can arrive
        later. Before that block, coherence/DTD/ref-floor gates are ref-empty and
        a VAD blip can only produce misleading "during playback" barge diagnostics
        or a false cut. Once the audio callback has played anything,
        ``_first_audio_pending`` clears. Between queued sentences it may be armed
        for the next utterance while the prior tail is still audible; a recent
        playback block plus playback level keeps the watch enabled for that gap.
        """
        if not self._speaking.is_set():
            return False
        if not self._first_audio_pending:
            return True
        level = self._playback_level
        last_playback_at = getattr(self, "_last_playback_at", 0.0)
        if not (math.isfinite(level) and level > 1e-5 and last_playback_at > 0.0):
            return False
        return time.monotonic() - last_playback_at <= _BARGE_TAIL_FRESH_SEC

    def _playback_loop(self) -> None:
        import sounddevice as sd

        out = None
        try:
            while self._running.is_set():
                try:
                    text, on_done, item_gen, directives = self._play_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if text is None:  # shutdown sentinel from stop()
                    break
                # rc-3: claim the sentence for playback, or skip it as stale. A
                # barge/stop after it was enqueued bumped _speak_gen (and drained
                # the queue); a sentence that slipped past the drain is stale and
                # _claim_utterance returns None WITHOUT clearing _stop_speaking
                # (so the pending barge isn't wiped). A current sentence is
                # claimed (clearing the flag for this fresh utterance).
                my_gen = self._claim_utterance(item_gen)
                if my_gen is None:
                    if on_done:
                        on_done()
                    continue
                # What is audibly playing RIGHT NOW -- the primary reference for
                # the barge-confirm echo filter. The _recent_spoken ring holds
                # ENQUEUED sentences, and a long reply queues more than the ring
                # holds, evicting the currently-playing (older) one -- live bug
                # run-20260702-223217: the current sentence's garbled echo passed
                # the filter and false-confirmed a barge. Bare str write.
                self._now_playing = text
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
                    self._underrun_at_reply_start = self._underrun_blocks
                    # Arm the playback-onset grace from the moment we COMMIT to
                    # speaking (synthesis start), not first-audio: the live
                    # self-interrupts fire during the synth lead-in (before any
                    # audio plays), where a first-audio anchor would still be unset.
                    # A grace measured from here covers synth + the audio onset.
                    self._playback_onset_at = time.monotonic()
                    # Re-arm the per-reply barge state on the silent->speaking
                    # transition (2026-06-10 self-interrupt fix). Clearing the
                    # BargeSustain window (via the cross-thread flag) means the
                    # prior reply's residual fires can't prime a 2-fire burst into
                    # this reply's onset echo. Re-bootstrapping the learned echo
                    # floors means the residual-floor gate in _looks_like_user
                    # measures THIS reply's echo level, not a stale one -- so a
                    # reply-onset echo transient (which sits at the floor) is
                    # rejected, while a real talk-over (well above it) still cuts.
                    self._barge_sustain_reset_pending = True
                    self._playback_floor_rms = 0.0
                    self._raw_playback_floor_rms = 0.0
                # Arm the per-utterance first-audio stamp; the audio callback
                # clears it and stamps TTS_FIRST_AUDIO when real audio first plays.
                self._first_audio_pending = True
                self._cb.on_speech_start()
                log.debug("speaking: %r (queue depth=%d)", text, self._play_q.qsize())

                def write(samples) -> None:
                    # Producer side of the playback FIFO (runs on this worker
                    # thread, NOT the audio callback). All the echo-reference tees
                    # (level EWMA, coherence ref, far-end ring) and the
                    # TTS_FIRST_AUDIO stamp have MOVED into _audio_cb so they fire
                    # off ACTUAL playback, not synthesis -- that alignment is the
                    # whole point of the rewrite. Here we just resample to play_sr
                    # and hand the samples to the FIFO, which BLOCKS when full
                    # (backpressure that paces synthesis to real-time playback).
                    if self._stop_speaking.is_set() or self._speak_gen != my_gen:
                        return  # barged / superseded chunk: never enqueue it
                    if self._play_sr != self._tts_sr:
                        samples = _resample_linear(samples, self._tts_sr, self._play_sr)
                    fifo = self._fifo
                    if fifo is not None:
                        # should_abort releases a producer blocked on a full FIFO
                        # the instant a barge-in (_stop_speaking / a generation
                        # bump) or shutdown (not _running) is requested -- no
                        # deadlock on teardown.
                        fifo.write(
                            samples,
                            should_abort=lambda: (
                                self._stop_speaking.is_set()
                                or self._speak_gen != my_gen
                                or not self._running.is_set()
                            ),
                        )

                try:
                    if out is None:
                        out_dev = _norm_device(self.config.output_device)
                        self._tts_sr = int(getattr(self._tts, "sample_rate", 0)) or 22050
                        # Callback-driven stream: PortAudio pulls audio from
                        # _audio_cb (which drains the FIFO + tees the far ring from
                        # the true playback position), instead of us pushing via a
                        # blocking out.write(). latency="low" keeps the callback
                        # period << the FIFO depth so barge-in residual stays small.
                        # Choose a rate the device ACTUALLY supports BEFORE opening.
                        # `check_output_settings` reliably RAISES on an unsupported
                        # rate; `OutputStream(...).start()` does NOT on some ALSA
                        # backends -- it prints "paInvalidSampleRate" to stderr but
                        # opens anyway, then the hardware runs at its own rate while
                        # we feed tts_sr-rate frames => wrong speed / pitch (the
                        # ALC285 rejects 22050 but only sometimes raised). Prefer the
                        # TTS rate (no resample), else the device default, else a
                        # common rate -- portable across devices, deterministic.
                        try:
                            default_sr = int(
                                sd.query_devices(out_dev, kind="output")["default_samplerate"]
                            )
                        except Exception:  # noqa: BLE001 - fall back to a safe common rate
                            default_sr = 44100
                        play_sr = default_sr
                        for cand in (self._tts_sr, default_sr, 48000, 44100):
                            try:
                                sd.check_output_settings(
                                    device=out_dev, samplerate=cand, channels=1, dtype="float32"
                                )
                                play_sr = cand
                                break
                            except Exception:  # noqa: BLE001 - try the next candidate
                                continue
                        out = sd.OutputStream(
                            channels=1, samplerate=play_sr, dtype="float32",
                            device=out_dev, latency="low", callback=self._audio_cb,
                        )
                        out.start()
                        self._play_sr = play_sr
                        if play_sr == self._tts_sr:
                            log.info("playback opened at %d Hz on device %s (callback)",
                                     play_sr, out_dev if out_dev is not None else "default")
                        else:
                            log.info(
                                "playback opened at %d Hz on device %s (callback; "
                                "resampling from %d Hz TTS -- device rejects %d Hz)",
                                play_sr, out_dev if out_dev is not None else "default",
                                self._tts_sr, self._tts_sr,
                            )
                        # Allocate the FIFO once the FINAL play_sr is known (so it
                        # survives the 22050->44100 reopen). The producer write()
                        # resamples tts_sr->play_sr before pushing, so the FIFO and
                        # outdata are both play_sr. The audio callback may already
                        # be firing (zero-filling) before this assignment -- that's
                        # fine: it checks `self._fifo is None` and emits silence.
                        self._fifo = PlaybackFIFO(int(self._play_sr * self._playback_fifo_sec_cur))
                        with self._out_lock:
                            self._out_stream = out
                    elif not out.active:
                        # Rare safety net: with the callback path a barge-in only
                        # flushes the FIFO (the stream stays open playing silence),
                        # so the stream is normally already active here. Restart it
                        # only if something stopped it.
                        out.start()
                    self._synthesize(text, write, my_gen, directives)
                    # (true barge-in-stop is stamped in stop_speaking() at the
                    # abort() instant -- the moment audio actually goes silent.)
                finally:
                    if on_done:
                        on_done()
                    # Only fall idle once the queue drains, so the capture loop
                    # doesn't flap ASR/barge-in on/off between adjacent sentences.
                    if self._play_q.empty():
                        # Wait for the FIFO to actually PLAY OUT before tearing
                        # down the echo refs: with the callback path the producer
                        # has handed the last samples to the FIFO but the audio
                        # thread is still draining them. Bounded by a deadline +
                        # broken by not-_running / _stop_speaking so a wedged or
                        # unplugged device can never hang the worker (mirrors the
                        # 1.0s join guards in stop()). A barge-in flush makes
                        # count()==0 immediately, so this returns at once then.
                        self._wait_for_playback_drain()
                        self._speaking.clear()
                        # Acoustic-artifact diagnostic (off the real-time thread):
                        # if the FIFO underran during this reply the listener heard
                        # gaps/static. Surface it so a glitchy reply is visible in
                        # the bundle instead of silently zero-filled.
                        _ur = self._underrun_blocks - self._underrun_at_reply_start
                        if _ur > 2:
                            log.warning(
                                "playback underran %d blocks this reply -- TTS synth not "
                                "keeping the output buffer full (CPU contention from the "
                                "ASR/second-pass worker?); raises buzzy/glitchy artifacts", _ur,
                            )
                            self._cb.on_metric("playback_underrun")
                        # Fix 5b: self-size the FIFO lead from THIS reply's underrun
                        # count -- grow when it starved, slow-decay toward the seed
                        # when clean. Recreated HERE only: the queue is empty and the
                        # drain wait above completed, so the old FIFO is spent and the
                        # audio callback (which snapshots self._fifo then zero-fills an
                        # empty one) is safe across the swap. Takes effect next reply.
                        seed = float(self.config.playback_fifo_sec)
                        prev = self._playback_fifo_sec_cur
                        self._playback_fifo_sec_cur = self._next_fifo_sec(prev, _ur, seed)
                        if (abs(self._playback_fifo_sec_cur - prev) > 1e-3
                                and self._play_sr > 0):
                            self._fifo = PlaybackFIFO(
                                int(self._play_sr * self._playback_fifo_sec_cur))
                            log.info(
                                "playback FIFO lead -> %.2fs (seed %.2fs, underran %d)",
                                self._playback_fifo_sec_cur, seed, _ur,
                            )
                        self._playback_level = 0.0  # nothing playing -> no echo ref
                        self._last_playback_at = 0.0
                        self._last_speaking_end = time.monotonic()  # arm the L2 refractory
                        if self._echo_coherence is not None:
                            self._echo_coherence.reset()  # drop the stale reference
                            if self._coh_ref_q is not None:
                                self._coh_ref_q.clear()  # audio-bargein-8: + un-ingested blocks
                        if self._dtd is not None:
                            # Run boundary, NOT a full reset: the learned echo charts
                            # persist across replies (2026-06-10 contamination fix --
                            # the per-reply re-warm-up seeded on the user talking at
                            # reply onset). Only the candidate run clears.
                            self._dtd.new_run()
                        # Drop the far-end reference + reset the canceller so a
                        # cut-off sentence's stale tail isn't subtracted from the
                        # next near-end block once playback resumes.
                        if self._far_ref is not None:
                            self._far_ref.clear()
                        if self._aec is not None:
                            self._aec.reset()
                        if self._aec_asr is not None:
                            self._aec_asr.reset()
                        # Hand the output device back when asked (so a co-located
                        # process -- e.g. the acoustic test's synthetic user -- can
                        # open it for its turn). Flush the FIFO + drop the shared
                        # handle under the lock FIRST so a concurrent stop_speaking()
                        # can't flush a stream we're closing; stop()/close() (NOT
                        # abort) joins the callback so none fires post-close. Clear
                        # _fifo so the next utterance re-allocates a clean one.
                        if self.config.release_output_when_idle and out is not None:
                            if self._fifo is not None:
                                self._fifo.flush()
                            with self._out_lock:
                                self._out_stream = None
                            try:
                                out.stop()
                                out.close()
                            except Exception:  # noqa: BLE001 - already aborted/closed
                                pass
                            out = None
                            self._fifo = None
                        self._cb.on_speech_end()
        except Exception:
            log.exception("playback loop crashed -- the assistant has gone mute")
            self._running.clear()
        finally:
            # Drop the shared handle before teardown so a concurrent
            # stop_speaking() can't flush a stream we're closing. Clear the FIFO
            # too so a never-restarted loop holds no playback buffer. stop()/
            # close() (NOT abort) joins the callback before close.
            with self._out_lock:
                self._out_stream = None
            self._fifo = None
            if out is not None:
                try:
                    out.stop()
                    out.close()
                except Exception:  # noqa: BLE001 - may already be aborted/closed
                    pass

    def _wait_for_playback_drain(self, timeout_sec: Optional[float] = None) -> bool:
        """Wait for queued playback to drain before reopening ASR.

        Returns True when no queued FIFO audio remains. If the output callback stalls
        past the bounded deadline while the session is otherwise healthy, flush the
        unplayed tail before ``_speaking`` clears; otherwise that tail can play after
        ASR reopens and be transcribed as a user final.
        """
        fifo = self._fifo
        if fifo is None:
            return True
        if timeout_sec is None:
            timeout_sec = self.config.playback_fifo_sec + 0.5
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while (
            fifo is not None
            and fifo.count() > 0
            and self._running.is_set()
            and not self._stop_speaking.is_set()
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
            fifo = self._fifo
        if fifo is None or fifo.count() <= 0:
            return True
        if self._running.is_set() and not self._stop_speaking.is_set():
            remaining = fifo.count()
            fifo.flush()
            log.warning(
                "playback FIFO did not drain before idle deadline; flushed %d queued "
                "samples before reopening ASR",
                remaining,
            )
            try:
                self._cb.on_metric("playback_drain_timeout")
            except Exception:  # noqa: BLE001 - metric is best-effort
                pass
        return False

    @staticmethod
    def _next_fifo_sec(prev: float, ur: int, seed: float) -> float:
        """Self-size the playback FIFO lead from a reply's underrun count (fix 5b).

        Grow (multiplicatively, fast converge) when the reply STARVED (ur>2 -- 1-2
        is the benign end-of-utterance straddle), slow-decay toward the ``seed``
        floor when clean, bounded by the ``_FIFO_SEC_MAX`` UX-latency ceiling. Pure
        + static so the control law is unit-tested without driving the audio loop.
        No per-machine operating value: the ceiling/rates are bounds, the operating
        depth is derived from THIS box's measured underruns."""
        if ur > 2:
            return min(_FIFO_SEC_MAX, prev * _FIFO_GROW)
        if prev > seed:
            return max(seed, prev * _FIFO_DECAY)
        return prev

    def _dc_block(self, samples, sr: int):
        """DC-block a synth chunk/clip with the long-lived one-pole high-pass.

        Applied FIRST in both synth paths so declick/leveler/normalize and every
        level/peak metric see the DC-free signal. The single instance is built
        lazily at the output rate and reused across chunks + sentences (state
        continuity), rebuilt only if the rate changes. No-op when disabled or the
        rate is unusable (DCBlocker returns the input unchanged)."""
        if not self.config.tts_dc_block:
            return samples
        # getattr default: markup/barge fixtures build the engine bypassing
        # __init__, so the handle may be unset on the first call.
        blk = getattr(self, "_tts_dc_blocker", None)
        if blk is None or getattr(self, "_tts_dc_sr", 0) != int(sr):
            blk = DCBlocker(int(sr), self.config.tts_dc_block_hz)
            self._tts_dc_blocker = blk
            self._tts_dc_sr = int(sr)
        return blk.process(samples)

    def _synthesize(
        self,
        text: str,
        write: Callable[[object], None],
        gen: Optional[int] = None,
        directives: Optional[dict] = None,
    ) -> None:
        """Synthesize ``text``, handing each audio chunk to ``write`` as it is
        produced. sherpa-onnx ``OfflineTts.generate`` streams via a ``callback``,
        so the first samples play before the whole sentence is synthesized; a
        build without that param falls back to chunking the finished waveform.

        ``gen`` is the per-utterance generation (rc-3): when set, synthesis stops
        if a barge bumps :attr:`_speak_gen` past it, even if the worker's
        clear-vs-set ordering momentarily wiped ``_stop_speaking``. ``None``
        (direct callers/tests) keeps the legacy ``_stop_speaking``-only check."""
        import numpy as np

        tts = self._tts
        if self.config.tts_markup:
            parsed_text, parsed_directives = parse_tts_markup(
                text,
                voices=self.config.tts_speaker_voices.keys(),
                emotions=self.config.tts_emotion_speed_map.keys(),
            )
            if parsed_directives:
                text = parsed_text
                merged_directives = dict(parsed_directives)
                if directives:
                    merged_directives.update(directives)
                directives = merged_directives
                if not text:
                    return
        sid = self.config.tts_speaker_id
        speed = self.config.tts_speed
        # Opt-in expressive markup: map this utterance's parsed directives to a
        # per-sentence (sid, speed). Fail-soft + clamped (resolve_tts_params);
        # ``directives`` is None on the default path -> defaults unchanged.
        if directives:
            sid, speed = resolve_tts_params(
                directives,
                default_sid=self.config.tts_speaker_id,
                default_speed=self.config.tts_speed,
                voice_map=self.config.tts_speaker_voices,
                emotion_speed_map=self.config.tts_emotion_speed_map,
                num_speakers=int(getattr(tts, "num_speakers", 0) or 0),
                speed_min=self.config.tts_speed_min,
                speed_max=self.config.tts_speed_max,
            )
        target_rms = self.config.tts_target_rms
        leveler_on = self.config.tts_output_leveler
        lowpass_hz = self.config.tts_output_lowpass_hz
        try:
            stream_sr = int(getattr(tts, "sample_rate", 0) or self._tts_sr or 22050)
        except (TypeError, ValueError):
            stream_sr = 22050
        carried_gain = getattr(self, "_tts_normalize_gain", None)
        streaming_candidate = bool(
            self._tts_can_stream
            and not leveler_on
            and (
                target_rms <= 0.0
                or (target_rms > 0.0 and carried_gain is not None)
            )
        )
        log.info(
            "tts resolved: %s",
            json.dumps(
                {
                    "text": text,
                    "sid": sid,
                    "speed": round(float(speed), 4),
                    "directives": directives or {},
                    "sample_rate": stream_sr,
                    "streaming_candidate": streaming_candidate,
                    "lowpass_hz": round(float(lowpass_hz), 1),
                    "target_rms": round(float(target_rms), 4),
                    "leveler": bool(leveler_on),
                    "declick": bool(self.config.tts_declick),
                },
                sort_keys=True,
                ensure_ascii=True,
                default=str,
            ),
        )
        # Hold the TTS lock for the whole synthesis so a concurrent startup warm
        # pass can't drive the same model at the same time.
        with self._tts_lock:
            # Streaming path (first samples play before the whole sentence is
            # synthesized). The output_leveler still owns a whole-clip AGC2-style
            # stage and stays non-streaming. normalize_rms can stream after the
            # first target_rms>0 sentence seeds _tts_normalize_gain; lowpass_hz can
            # stream through a fresh per-utterance IIR filter. The filter is not
            # carried across sentence boundaries so one sentence's tail cannot
            # color the onset of the next sentence.
            _norm_gain = (
                float(carried_gain)
                if target_rms > 0.0 and carried_gain is not None
                else None
            )
            _stream_with_rms = _norm_gain is not None
            if self._tts_can_stream and not leveler_on and (
                target_rms <= 0.0 or _stream_with_rms
            ):
                _lowpass = StreamingLowpass(stream_sr, lowpass_hz)
                _raw_sumsq = 0.0
                _raw_count = 0
                # Cheap running accumulators for the post-DSP "tts audio quality"
                # summary below -- scalar-only (no spectral metrics): a per-chunk
                # FFT would need Welch-style aggregation across non-uniform chunk
                # sizes, and buffering the whole clip just to measure it would
                # defeat the point of this streaming path (first audio before the
                # whole sentence is ready). The whole-clip path below logs the
                # full metric set (incl. hf_ratio/spectral_flatness) instead.
                _q_sum = 0.0
                _q_sumsq = 0.0
                _q_peak = 0.0
                _q_clip = 0
                _q_n = 0

                def on_chunk(samples, *_progress) -> int:
                    nonlocal _raw_sumsq, _raw_count, _q_sum, _q_sumsq, _q_peak, _q_clip, _q_n
                    raw = np.asarray(samples, dtype="float32").reshape(-1)
                    # DC-block FIRST so the raw-RMS/peak/dc metrics + normalize all
                    # see a centred signal (state carries across chunks + sentences).
                    raw = self._dc_block(raw, stream_sr)
                    blk = raw
                    if _norm_gain is not None:
                        if raw.size:
                            raw64 = raw.astype("float64")
                            _raw_sumsq += float(np.dot(raw64, raw64))
                            _raw_count += int(raw.size)
                        blk = np.asarray(
                            apply_gain_soft_limit(raw, _norm_gain),
                            dtype="float32",
                        ).reshape(-1)
                    if self.config.tts_declick:        # repair VITS impulse spikes
                        blk = np.asarray(
                            declick(blk, threshold=self.config.tts_declick_threshold),
                            dtype="float32",
                        ).reshape(-1)
                    if _lowpass.enabled:
                        blk = np.asarray(_lowpass.process(blk), dtype="float32").reshape(-1)
                        # A causal filter can overshoot after upstream limiting; keep
                        # the actual PortAudio feed finite and inside float full-scale.
                        blk = np.nan_to_num(blk, nan=0.0, posinf=0.0, neginf=0.0)
                        np.clip(blk, -1.0, 1.0, out=blk)
                    if blk.size:
                        blk64 = blk.astype("float64")
                        _q_sum += float(np.sum(blk64))
                        _q_sumsq += float(np.dot(blk64, blk64))
                        _q_peak = max(_q_peak, float(np.max(np.abs(blk64))))
                        _q_clip += int(np.count_nonzero(np.abs(blk64) >= 0.99))
                        _q_n += int(blk64.size)
                    write(blk)
                    return 0 if (
                        self._stop_speaking.is_set()
                        or (gen is not None and self._speak_gen != gen)
                    ) else 1

                try:
                    tts.generate(text, sid=sid, speed=speed, callback=on_chunk)
                    if _norm_gain is not None and _raw_count > 0:
                        r = math.sqrt(_raw_sumsq / float(_raw_count))
                        if r > 1e-6:
                            self._tts_normalize_gain = min(float(target_rms) / r, 20.0)
                    if _q_n > 0:
                        log.info(
                            "tts audio quality: %s",
                            json.dumps(
                                {
                                    "mode": "streaming",
                                    "rms": round(math.sqrt(_q_sumsq / _q_n), 5),
                                    "peak": round(_q_peak, 5),
                                    "clip_pct": round(100.0 * _q_clip / _q_n, 3),
                                    "dc_offset": round(_q_sum / _q_n, 6),
                                    "hf_ratio": None,
                                    "spectral_flatness": None,
                                    "n_samples": _q_n,
                                },
                                sort_keys=True,
                                ensure_ascii=True,
                                default=str,
                            ),
                        )
                    return
                except TypeError:
                    self._tts_can_stream = False  # this build has no streaming callback

            audio = tts.generate(text, sid=sid, speed=speed)
        samples = np.asarray(audio.samples, dtype="float32").reshape(-1)
        sr = int(getattr(audio, "sample_rate", 0)) or 22050
        # DC-block FIRST (before declick/leveler/normalize + their level metrics).
        samples = self._dc_block(samples, sr)
        if leveler_on:
            # OUTPUT LEVELER path: declick FIRST (so the limiter's true-peak
            # estimate is not driven by a VITS impulse spike declick would have
            # removed), THEN the fused perceptual-loudness + true-peak limiter
            # OWNS loudness + peak (normalize_rms is SKIPPED -- its linear-RMS
            # target would fight the perceptual target). The applied gain is
            # carried across sentences (self._tts_level_gain_db) so loudness slews
            # smoothly across a multi-sentence reply.
            if self.config.tts_declick:                # repair VITS impulse spikes first
                samples = np.asarray(
                    declick(samples, threshold=self.config.tts_declick_threshold),
                    dtype="float32",
                ).reshape(-1)
            samples, self._tts_level_gain_db = output_leveler(
                samples,
                target_dbfs=self.config.tts_loudness_target_dbfs,
                true_peak_dbtp=self.config.tts_true_peak_dbtp,
                sr=sr,
                prev_gain_db=self._tts_level_gain_db,
                max_slew_db_per_s=self.config.tts_loudness_slew_db_per_s,
            )
            samples = np.asarray(samples, dtype="float32").reshape(-1)
        else:
            # Legacy whole-clip path: normalize_rms needs the full clip's RMS.
            # Capture the raw (pre-gain) RMS so we can carry the applied gain to
            # subsequent sentences -- enabling the streaming path above from
            # sentence 2 onward (feed-forward with this session's established level).
            raw_rms = rms_of(samples) if target_rms > 0.0 else 0.0
            samples = np.asarray(
                normalize_rms(samples, target_rms), dtype="float32"
            ).reshape(-1)
            if target_rms > 0.0 and raw_rms > 1e-6:
                # Store the actual linear gain applied; subsequent sentences can
                # use the streaming path with this as their feed-forward seed.
                self._tts_normalize_gain = min(float(target_rms) / raw_rms, 20.0)
            if self.config.tts_declick:                # repair VITS impulse spikes
                samples = np.asarray(
                    declick(samples, threshold=self.config.tts_declick_threshold),
                    dtype="float32",
                ).reshape(-1)
        # HF roll-off (final stage, after loudness): tame a bright voice's highs so
        # they don't overdrive a small/cheap open speaker into a buzzy rasp. Whole-
        # clip fallback uses the zero-phase FFT filter; the streaming callback path
        # above uses StreamingLowpass chunk-by-chunk.
        if lowpass_hz > 0.0:
            samples = np.asarray(
                lowpass_soft(samples, sr, lowpass_hz,
                             width_hz=self.config.tts_output_lowpass_width_hz),
                dtype="float32",
            ).reshape(-1)
            # The FFT low-pass is linear and can overshoot the leveler's ceiling.
            # Keep the final playback buffer finite and inside float full-scale.
            samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(samples, -1.0, 1.0, out=samples)
        # Final-signal quality snapshot -- the EXACT samples about to reach the
        # FIFO/speaker, at the TTS's native rate (see audio_quality_metrics'
        # docstring for why this is more trustworthy than the .ref.wav AEC tap).
        log.info(
            "tts audio quality: %s",
            json.dumps(
                {"mode": "whole_clip", **audio_quality_metrics(samples, sr)},
                sort_keys=True,
                ensure_ascii=True,
                default=str,
            ),
        )
        chunk = max(1, int(sr * 0.1))
        for i in range(0, len(samples), chunk):
            if self._stop_speaking.is_set() or (gen is not None and self._speak_gen != gen):
                break
            write(samples[i : i + chunk])

    def _dtd_residual_level(self, samples, mic_raw) -> float:
        """RMS of the signal the DTD's *residual* feature (and its floor gate)
        should read for this block.

        A LINEAR echo canceller (nlms/dtln, or headphones with no echo) removes
        only the far-end reference it has -- it CANNOT cancel the near-end USER
        (the user is not in the reference), so the user's voice lands in the
        post-AEC residual, which is therefore the strongest barge feature (DTD
        weight 1.0). The always-on WebRTC APM, however, ALSO runs ML noise
        suppression on every block (``_apm_owns_ns``); during double-talk that NS
        attenuates the near-end user in the residual, so the residual goes BLIND
        to a real talk-over -- the documented open-speaker "user had to scream /
        0 fired" miss (``_apm_owns_ns`` profiles: e.g. ``open_speaker``). The RAW
        pre-AEC/pre-NS mic still carries the user, so under ``_apm_owns_ns`` the
        residual feature is read from ``mic_raw`` instead of the suppressed
        ``samples`` (and the caller pairs it with the raw-mic floor,
        ``_raw_playback_floor_rms``). Non-APM paths are byte-identical
        (``_apm_owns_ns`` is False). MUST be the single source of truth shared by
        the gate (``_looks_like_user``) and the echo-learning tap
        (``observe_echo`` in the capture loop), so the chart the DTD calibrates
        and the value it scores against it always come from the SAME signal --
        otherwise a chart learned on the near-silent post-NS floor would make the
        raw-mic level a z-outlier and self-interrupt on echo-only.

        NOTE: DTLN is NOT linear -- it is a spectral-MASKING canceller that, like
        the APM's NS, attenuates the near-end user during double-talk (live proof:
        run-20260704-143112 z_resid pinned at 0 on a loud talk-over -> D capped
        ~1.5 < K, barge never fired). So ``_resid_blind`` covers DTLN + APM-NS, not
        just ``_apm_owns_ns``; only a genuinely linear canceller (nlms/headphones)
        leaves the user in the residual and reads ``samples``."""
        return rms(mic_raw) if getattr(self, "_resid_blind", False) else rms(samples)

    def _looks_like_user(self, samples, mic_raw=None) -> bool:
        # Is playback-time mic voice a genuine barge-in (vs the assistant's own TTS
        # echo)? IDENTITY-FREE: speaker/user detection is a SEPARATE feature used
        # only to gate FINALS (``_should_act_on_final``); the interrupt must never
        # depend on it (the enrolled embedder is unreliable -- measured 2026-06-01
        # it scored the user's OWN voice ~0.15 -- and the brief is "no user
        # detection in the interrupt").
        #
        # ARCHITECTURE (2026-06-07 redesign -- see the barge-in audit). The PRIMARY
        # trigger is scale-invariant reference COHERENCE on the RAW pre-AEC mic
        # (``mic_raw``): "does the mic contain sound the played reference can't
        # explain?" The user adds such sound; the assistant's own echo does not.
        # Coherence is volume-INDEPENDENT and structurally rejects the echo, so it
        # has the clean operating point a residual LEVEL gate can never have on an
        # open NONLINEAR speaker (where post-AEC residual echo spikes overlap real-
        # voice levels -- no threshold both ignores echo and catches a talk-over).
        # It MUST see the RAW mic: AEC removes the very correlation coherence keys
        # on, so feeding it the post-AEC residual made it misread cancelled echo as
        # "user" (the old self-interrupt; that is why coherence was wrongly thought
        # to "conflict with AEC"). The in-detector ``confirm_frames`` hysteresis +
        # the self-calibrating control chart reject one-off nonlinear-echo spikes.
        #   decide() -> True  : user barge        -> fire
        #            -> False : echo-only         -> reject
        #            -> None  : no reference yet / TTS silence -> fall through to the
        #                       level-gate FALLBACK below (never worse than before).
        # The level gates (residual-floor / ambient / output-margin) are retained
        # ONLY as that fallback, for the moments coherence cannot decide. The barge
        # gate does NOT re-align the AEC reference from coherence's delay estimate:
        # on a nonlinear speaker that peak is noisy and re-aligning every block
        # destabilises the canceller. The reference stays pinned at aec_ref_delay_ms.
        if mic_raw is None:
            mic_raw = samples  # single-arg callers (legacy/tests): no separate raw tee
        # DEVICE-ADAPTIVE PRIMARY (open speaker, AEC on): the fused z-score DTD.
        # No fixed margin -- each feature (raw-mic energy, post-AEC residual energy,
        # coherence incoherent-fraction) is a self-calibrated outlier from THIS
        # device's echo-only chart, and a real talk-over is the JOINT outlier whose
        # summed z-scores cross the dimensionless threshold K. This escapes BOTH the
        # self-interrupt (coherence alone overlaps voice on a nonlinear speaker) AND
        # the scream-to-stop (a fixed dB over a loud echo) failure modes that the
        # prior single-feature gates hit live. The coherence detector still runs --
        # but only to PRODUCE the incoherent-fraction feature, not to decide. See
        # core/engines/_dtd.py.
        if self._dtd is not None:
            det = self._echo_coherence
            coh_verdict = None
            incoh = 0.0
            if det is not None:
                coh_verdict = det.decide(mic_raw)
                incoh = float(det.last_incoherent_fraction)
            # Which signal carries the user for the DTD's "residual" feature + its
            # floor gate. A linear AEC (nlms/dtln, or headphones with no echo)
            # leaves the near-end user IN the post-AEC residual; the always-on APM
            # additionally runs ML noise suppression on EVERY block
            # (``_apm_owns_ns``) and attenuates the user there, so under APM the
            # residual goes BLIND to a real talk-over -- read the raw pre-NS mic +
            # its floor instead. See ``_dtd_residual_level``. (Non-APM paths are
            # byte-identical: ``_apm_owns_ns`` is False.)
            resid_rms = self._dtd_residual_level(samples, mic_raw)
            resid_floor = (
                self._raw_playback_floor_rms if getattr(self, "_resid_blind", False)
                else self._playback_floor_rms
            )
            fired = self._dtd.decide(
                raw_rms=rms(mic_raw), resid_rms=resid_rms, incoherent_fraction=incoh,
            )
            # Coherence echo veto (2026-06-27 self-interrupt fix). DTD's shipped
            # weights currently ignore coherence (w_coh=0.0), so a DTLN residual
            # spike from nonlinear laptop-speaker echo can cross K even when the
            # reference-coherence detector explicitly classified the frame as
            # echo-only. Honor that explicit False only as a veto: True and None
            # keep the prior DTD behavior, so a real coherence-confirmed talk-over
            # still cuts and no-reference moments still fall back to the DTD.
            # Coherence echo veto: honor an explicit echo-only verdict as a veto.
            # (Reverted 2026-07-04: disabling it under _resid_blind removed the only
            # echo guard on the raw-mic path and caused the APM self-interrupt --
            # the raw-mic DTD collapses to a loudness gate that fires on the
            # assistant's own loud narration syllables. See the permanent-plan doc.)
            coh_veto = bool(
                fired
                and getattr(self.config, "dtd_coherence_echo_veto", True)
                and coh_verdict is False
            )

            # Residual-floor gate (2026-06-10 self-interrupt fix). On a starved mic
            # the z-scores BLOAT against a near-silent warmup baseline, so a 2-fire
            # reply-onset echo burst can fire the DTD even though the absolute
            # residual barely lifts off the echo floor (run-20260609-234435: echo
            # fires at resid 0.0008-0.0018). A genuine talk-over stands clearly
            # above the LEARNED residual echo floor (run-20260609-203236: resid
            # 0.0024-0.0128). Require the post-AEC residual to clear
            # dtd_residual_floor_margin_db above the floor before a DTD fire counts
            # -- RELATIVE to the per-reply-learned floor, never a fixed energy. The
            # floor is re-bootstrapped each reply (speaking-START reset) so it is the
            # CURRENT reply's echo level, not a stale prior one. Fail-open until the
            # floor is learned (cold start) and when the margin is 0 (disabled), so
            # an echo-free / non-AEC setup is unchanged.
            floored = fired and not coh_veto
            if (
                floored
                and self.config.dtd_residual_floor_margin_db > 0.0
                and resid_floor > 0.0
                and not loudness_admits(
                    resid_rms, resid_floor,
                    margin_db=self.config.dtd_residual_floor_margin_db,
                )
            ):
                floored = False  # DTD said barge, but the residual is at the echo floor
            # Log EVERY evaluation (not just fires) so a run bundle shows the full D
            # distribution -- echo-only D vs talk-over D -- to calibrate K per device.
            # ``gated`` reflects the post-floor verdict the caller acts on.
            ref_delay_ms = (
                1000.0 * float(getattr(self, "_aec_ref_delay", 0))
                / max(1, int(self.config.sample_rate))
            )
            log.debug(
                "dtd: D=%.2f K=%.1f fired=%s gated=%s (z_raw=%.2f z_resid=%.2f z_coh=%.2f) "
                "raw=%.4f resid=%.4f incoh=%.2f resid_floor=%.4f consec=%d "
                "coh=%s coh_veto=%s ref_delay=%.0fms",
                self._dtd.last_D, self._dtd.k, fired, floored, self._dtd.last_z_raw,
                self._dtd.last_z_resid, self._dtd.last_z_coh,
                rms(mic_raw), resid_rms, incoh, resid_floor, self._dtd.last_consec,
                coh_verdict, coh_veto, ref_delay_ms,
            )
            return floored
        det = self._echo_coherence
        if det is not None:
            verdict = det.decide(mic_raw)
            if verdict is False:
                return False  # coherence is confident this is echo-only
            if verdict is True:
                # Coherence says "user" -- but on an OPEN speaker the nonlinear echo
                # can FOOL coherence: its raw-mic incoherent fraction overlaps a real
                # voice (measured on the ALC285: echo p50~0.88, p95~0.98, NEGATIVE
                # safety headroom -> intermittent self-interrupt). Confirm with an
                # ORTHOGONAL ENERGY signal -- but on the RAW pre-AEC mic, NOT the
                # post-AEC residual: DTLN suppresses the near-end (user) voice during
                # double-talk, so the residual barely rises on a real talk-over and
                # the gate rejected EVERY barge (live: 0 fired / 7 rejected, user had
                # to scream). On the raw mic the echo-only level is a steady floor and
                # a talk-over genuinely adds the user's energy on top, so the +Nd B
                # elevation the gate looks for actually exists there. Echo = incoherent
                # but no raw elevation -> rejected; user = incoherent AND raw-loud ->
                # fires. (AEC off -- headphones, no echo -- coherence alone fires;
                # raw floor not learned yet -> coherence alone.)
                if (
                    self._aec is not None
                    and self.config.barge_in_residual_margin_db > 0.0
                    and self._raw_playback_floor_rms > 0.0
                    and not loudness_admits(
                        rms(mic_raw), self._raw_playback_floor_rms,
                        margin_db=self.config.barge_in_residual_margin_db,
                    )
                ):
                    return False  # coherence said user, but the RAW mic is at the echo floor
                log.debug(
                    "coherence barge: incoherent=%.2f baseline=%.2f eff_margin=%.2f "
                    "delay=%.0fms consec=%d raw=%.4f raw_floor=%.4f residual=%.4f",
                    det.last_incoherent_fraction, det.last_baseline,
                    det.last_effective_margin, det.last_delay_ms, det.last_consec,
                    rms(mic_raw), self._raw_playback_floor_rms, rms(samples),
                )
                return True
            # verdict is None -> coherence abstains (no reference yet / TTS silence);
            # fall through to the level-gate fallback so behaviour is never worse.

        # FALLBACK 1 -- AEC on, auto-calibrated residual FLOOR. With the echo
        # cancelled the residual during echo-only sits at _playback_floor_rms
        # (learned online during playback; freeze-on-burst -> adapts to speaker
        # volume + room noise); a real barge stands barge_in_residual_margin_db
        # ABOVE it. Fails closed until the floor is learned (no warmup self-fire).
        if self._aec is not None and self.config.barge_in_residual_margin_db > 0.0:
            if self._playback_floor_rms <= 0.0:
                return False
            return loudness_admits(
                rms(samples), self._playback_floor_rms,
                margin_db=self.config.barge_in_residual_margin_db,
            )
        # FALLBACK 2 -- AEC on, ambient-floor loudness gate (residual vs the quiet
        # floor the capture loop maintains when input_loudness_margin_db > 0).
        if self._aec is not None and self._input_loudness_margin_db > 0.0:
            return loudness_admits(
                rms(samples), self._ambient_rms, margin_db=self._input_loudness_margin_db
            )
        # FALLBACK 3 -- no AEC (the echo is still in `samples`): the playback-
        # relative output-margin gate. With AEC the smaller aec_relaxed_margin_db.
        margin_db = (
            self.config.aec_relaxed_margin_db
            if self._aec is not None
            else self.config.barge_in_output_margin_db
        )
        if margin_db > 0.0:
            return passes_output_margin(
                rms(samples), self._playback_level, margin_db=margin_db
            )
        # No discriminator configured -> fail open (any playback-time voice is a
        # barge). Identity is intentionally NOT consulted here: it gates FINALS.
        return True

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

    # Online residual-floor tracking. The floor EWMA-tracks the post-AEC echo+noise
    # level within +6 dB (BURST_RATIO) and FREEZES above it -- a real barge is a
    # large outlier that must NOT raise its own bar (mirrors the coherence chart and
    # _ambient_rms "ignore upward excursions" rule).
    _PLAYBACK_FLOOR_BURST_RATIO = 2.0   # +6 dB
    _PLAYBACK_FLOOR_ALPHA = 0.05

    def _update_playback_floor(self, level: float) -> None:
        """Online estimate of the post-AEC residual echo+noise floor during
        playback (capture thread only). Bootstrap on the first block; then EWMA-
        track the floor within +6 dB so it follows the current speaker volume + room
        noise up and down; FREEZE on a larger burst so a genuine talk-over (the
        thing the barge gate fires on) can never drag the floor up to itself. Pure
        and side-effect-free apart from the single float write -> unit-testable."""
        if level < 0.0 or not math.isfinite(level):
            return
        f = self._playback_floor_rms
        if f <= 0.0:
            self._playback_floor_rms = level          # bootstrap to the first echo level
            return
        if level <= f * self._PLAYBACK_FLOOR_BURST_RATIO:
            a = self._PLAYBACK_FLOOR_ALPHA
            self._playback_floor_rms = (1.0 - a) * f + a * level
        # else: a >+6 dB burst -- likely a real barge -> freeze the floor.

    def _update_raw_playback_floor(self, level: float) -> None:
        """Same online floor estimate as :meth:`_update_playback_floor`, but on the
        RAW pre-AEC mic. This is the echo-only level the assistant's own TTS couples
        into the mic before cancellation; the barge energy-confirmation requires a
        real talk-over to stand barge_in_residual_margin_db above it. Bootstrap on
        the first block, EWMA within +6 dB, FREEZE on a larger burst (the talk-over
        itself must not raise its own bar). Capture thread only; single float write."""
        if level < 0.0 or not math.isfinite(level):
            return
        f = self._raw_playback_floor_rms
        if f <= 0.0:
            self._raw_playback_floor_rms = level
            return
        if level <= f * self._PLAYBACK_FLOOR_BURST_RATIO:
            a = self._PLAYBACK_FLOOR_ALPHA
            self._raw_playback_floor_rms = (1.0 - a) * f + a * level
        # else: a >+6 dB burst -- likely a real talk-over -> freeze the floor.

    # --- duck-then-confirm barge gate (word-gated interrupt) -----------------
    # The LiveKit/Pipecat word-gate pattern, sliced for this engine: an acoustic
    # trigger ducks playback (reversible, one callback period) and opens a short
    # confirm window during which the capture loop feeds the streaming recognizer;
    # only real transcribed words -- that don't read as the assistant's own ducked
    # echo -- hard-fire the barge. See the barge_confirm_* config block.

    def _barge_confirm_active(self) -> bool:
        # getattr default: barge fixtures may build the engine bypassing
        # __init__; a missing stamp reads as "no window open".
        return getattr(self, "_confirm_until", 0.0) > 0.0

    def _begin_barge_confirm(self, recognizer, stream, now: float) -> None:
        """Open the confirm window: duck playback + snapshot the partial base.

        The base snapshot means only words transcribed AFTER the trigger count
        as confirmation -- residue already in the stream (a pre-reply partial
        tail) can't confirm by itself."""
        self._confirm_until = now + max(0.1, self.config.barge_confirm_window_sec)
        base = ""
        try:
            base = (recognizer.get_result(stream) or "").strip()
        except Exception:  # noqa: BLE001 - a stream hiccup must not break capture
            base = ""
        self._confirm_base_text = base
        self._duck_gain = min(1.0, max(0.0, self.config.barge_confirm_duck_gain))
        log.info(
            "barge-in: acoustic trigger -- ducking playback, awaiting speech "
            "confirmation (%.1fs window)", self.config.barge_confirm_window_sec,
        )
        self._cb.on_metric("barge_in_duck")

    def _barge_confirm_step(
        self, recognizer, stream, samples, now: float, mic_raw=None
    ) -> bool:
        """One capture block inside an active confirm window.

        Feeds the recognizer (playback is ducked, so the block is mostly the
        USER if anyone is talking) and decides: enough NEW words that aren't the
        assistant's own echo -> hard-fire the barge (True). A stop command
        confirms alone. Window expired with no evidence -> restore volume, reset
        the stream (drop any echo it was fed), arm the retry suppress, teach the
        DTD charts the window's (verified-echo) levels, and keep speaking (False)."""
        if mic_raw is None:
            mic_raw = samples
        # Bank this block's levels as a POTENTIAL echo observation -- committed
        # to the DTD charts only if the window expires unconfirmed (then it was
        # echo by definition; a confirmed window is user-contaminated -> discard).
        if self._dtd is not None:
            det = self._echo_coherence
            incoh = float(det.last_incoherent_fraction) if det is not None else 0.0
            self._confirm_echo_obs.append(
                (rms(mic_raw), self._dtd_residual_level(samples, mic_raw), incoh)
            )
        text = ""
        try:
            # C (masking-canceller path): decode the RAW mic, not the DTLN/NS-masked
            # residual. Playback is ducked to barge_confirm_duck_gain during the
            # window, so the raw mic is user-dominated and the words SURVIVE -- the
            # masked residual would erase them and the window could never confirm.
            decode_src = mic_raw if getattr(self, "_resid_blind", False) else samples
            stream.accept_waveform(self.config.sample_rate, decode_src)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            text = (recognizer.get_result(stream) or "").strip()
        except Exception:  # noqa: BLE001 - decode errors must not break capture
            pass
        base = self._confirm_base_text
        new_text = text[len(base):].strip() if base and text.startswith(base) else text
        words = [w for w in new_text.split() if any(ch.isalpha() for ch in w)]
        confirmed = bool(new_text) and (
            is_stop_command(new_text)
            or (
                len(words) >= max(1, self.config.barge_confirm_min_words)
                and not self._reads_like_own_speech(new_text)
            )
        )
        if confirmed:
            self._end_barge_confirm()
            self._barge_in_fired_this_run = True
            self._barge_in_suppressed_until = (
                now + max(0.0, self.config.barge_in_suppress_sec)
            )
            log.info("barge-in confirmed by speech: %r", new_text[:80])
            self._cb.on_metric("barge_in_confirmed")
            # Keep the exact legacy log line so run-bundle tooling
            # (grep "barge-in detected") sees confirmed barges too.
            log.info("barge-in detected")
            self._cb.on_barge_in()
            # The confirm-window audio is already IN the stream: the user's
            # first words become the head of the next final (free pre-roll)
            # once _speaking clears and the normal ASR path resumes.
            return True
        if now >= self._confirm_until:
            # Echo (or a transient) tripped the acoustics but nobody is talking:
            # TEACH the DTD charts this window's levels before restoring. An
            # unconfirmed window is the one reliably-labeled echo sample this box
            # produces (its TTS echo always reads as VAD-speech, starving the
            # normal VAD-quiet observe_echo tap) -- so the chart baseline rises
            # to the true echo level and the trigger flood decays instead of
            # re-ducking every few seconds (run-223217: 14 triggers/45s).
            if self._dtd is not None:
                for obs in self._confirm_echo_obs:
                    self._dtd.observe_echo(*obs)
            self._end_barge_confirm()
            # Restore volume and keep speaking. Reset the stream so any ducked
            # echo it was fed can't pollute the next real final, and suppress
            # re-triggers so an echo-heavy reply can't pump the volume.
            try:
                recognizer.reset(stream)
            except Exception:  # noqa: BLE001 - reset is best-effort
                pass
            self._barge_in_suppressed_until = (
                now + max(0.0, self.config.barge_confirm_retry_suppress_sec)
            )
            log.info(
                "barge-in NOT confirmed (no talk-over speech in %.1fs) -- "
                "restoring volume", self.config.barge_confirm_window_sec,
            )
            self._cb.on_metric("barge_in_unconfirmed")
        return False

    def _end_barge_confirm(self) -> None:
        """Close the window + restore full volume (idempotent, any thread)."""
        self._confirm_until = 0.0
        self._confirm_base_text = ""
        self._duck_gain = 1.0
        obs = getattr(self, "_confirm_echo_obs", None)
        if obs:
            obs.clear()

    def _barge_word_cut_active(self) -> bool:
        """Whether the continuous no-duck word-cut path is live. Scoped to the OS
        echo-cancel path: the flag AND self._aec is None (no in-app AEC/APM). With
        an in-app canceller the acoustic DTD path owns the decision, so this stays
        inert and that path is byte-identical. getattr keeps fixtures inert."""
        return (
            bool(getattr(self.config, "barge_word_cut_enabled", False))
            and self._aec is None
        )

    def _barge_word_cut_step(self, recognizer, stream, samples, now: float) -> bool:
        """One playback block on the continuous no-duck word-cut path. Feeds the
        recognizer THIS block (OS-cancelled mic, clean of the assistant's echo) and
        hard-cuts the instant enough NEW non-own-speech words appear since the burst
        started -- or a stop command. No ducking / no volume change: word CONTENT
        decides, not level. Returns True when it fired the cut."""
        st = self._wc_stats
        # Feed the VAD THIS block BEFORE consulting it (live fix, run-20260706-
        # 231226): the word-cut branch `continue`s before the acoustic path's
        # accept_waveform, so nothing else updates the VAD during playback.
        # is_speech_detected() stayed frozen at its pre-reply (quiet) state and
        # the burst gate below starved the recognizer for the WHOLE reply --
        # zero words ever fed, so the live talk-over batch could not cut.
        # Mirrors the acoustic path's per-block accept.
        if self._vad is not None:
            self._vad.accept_waveform(samples)
        # Near-end evidence window (post-EC mic RMS + VAD fraction while the
        # assistant speaks): the only signal that can separate "voice never
        # survived the canceller" from "voice arrived but was never transcribed"
        # after the fact.
        self._wc_window_update(samples, now)
        # Bound accumulation to ONE speech burst: after ENOUGH CONSECUTIVE
        # VAD-quiet blocks (barge_word_cut_reset_quiet_blocks) reset the stream +
        # base so a prior burst's echo/user text can't pile up toward the word
        # floor and streaming prefix-revision can't flip text.startswith(base)
        # into feeding the whole reply's echo blob (the streaming recognizer never
        # self-resets during a reply). Debounced because the OS canceller gates
        # the near-end in and out during double-talk: a single quiet block is VAD
        # flicker mid-sentence, not a burst boundary, and the old hair-trigger
        # wiped the very words a talk-over had accumulated. Quiet blocks are
        # never fed.
        if self._vad is not None and not self._vad.is_speech_detected():
            st["skipped_quiet"] = st.get("skipped_quiet", 0) + 1
            self._word_cut_quiet_run += 1
            debounce = max(
                1, int(getattr(self.config, "barge_word_cut_reset_quiet_blocks", 3))
            )
            if self._word_cut_fed_stream and self._word_cut_quiet_run >= debounce:
                lost = ""
                try:
                    txt = (recognizer.get_result(stream) or "").strip()
                    base = self._word_cut_base
                    lost = (
                        txt[len(base):].strip()
                        if base and txt.startswith(base)
                        else txt
                    )
                except Exception:  # noqa: BLE001 - telemetry is best-effort
                    pass
                try:
                    recognizer.reset(stream)
                except Exception:  # noqa: BLE001 - reset is best-effort
                    pass
                self._word_cut_fed_stream = False
                self._word_cut_base = ""
                st["resets"] = st.get("resets", 0) + 1
                lost_words = [
                    w for w in lost.split() if any(ch.isalpha() for ch in w)
                ]
                if lost_words:
                    # A burst boundary wiped words a cut could have used -- the
                    # smoking gun for "user words swallowed by the reset".
                    st["dropped_words"] = (
                        st.get("dropped_words", 0) + len(lost_words)
                    )
                    log.info(
                        "word-cut burst reset: dropped %d word(s) %r",
                        len(lost_words), lost[:60],
                    )
            return False
        self._word_cut_quiet_run = 0
        self._word_cut_fed_stream = True
        st["fed"] = st.get("fed", 0) + 1
        text = ""
        try:
            stream.accept_waveform(self.config.sample_rate, samples)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            text = (recognizer.get_result(stream) or "").strip()
        except Exception:  # noqa: BLE001 - decode errors must not break capture
            # Without this trace a recognizer failure is indistinguishable from
            # "no user words" (the run-20260706-231226 lesson). Warn once per
            # reply; keep counting silently after that.
            st["decode_errors"] = st.get("decode_errors", 0) + 1
            if not st.get("_decode_warned"):
                st["_decode_warned"] = True
                log.warning("word-cut: recognizer decode failed during playback")
            return False
        base = self._word_cut_base
        new_text = text[len(base):].strip() if base and text.startswith(base) else text
        words = [w for w in new_text.split() if any(ch.isalpha() for ch in w)]
        if len(words) > st.get("max_words", 0):
            # Novel words appeared on the cancelled mic during playback -- the
            # positive half of the funnel. One line per per-reply high-water
            # mark, so a long reply cannot spam the bundle.
            st["max_words"] = len(words)
            log.info("word-cut trace: %d word(s) %r", len(words), new_text[:80])
        # SAME guards the acoustic path honours -- suppress the CUT only: one-cut-
        # per-run latch, debounce suppress window, post-speaking refractory (just-
        # cancelled tail), 0.4s playback-onset grace (reply-onset echo transient).
        grace = self.config.barge_in_playback_onset_grace_sec
        onset = getattr(self, "_playback_onset_at", 0.0)
        if (
            self._barge_in_fired_this_run
            or now < self._barge_in_suppressed_until
            or self._in_post_speaking_refractory(now)
            or (grace > 0.0 and onset > 0.0 and now < onset + grace)
        ):
            st["guard_suppressed"] = st.get("guard_suppressed", 0) + 1
            return False
        # WORD-CUT floor (barge_word_cut_min_words, default 4) -- higher than the
        # shared min_words=2 because nonlinear-speaker echo transcribes as garbled
        # 2-3 word fragments that don't match the played text (so _reads_like_own_
        # speech can't reject them). A real talk-over sentence clears 4; a bare
        # "stop" cuts alone via is_stop_command.
        floor = max(1, int(getattr(self.config, "barge_word_cut_min_words", 4)))
        confirmed = bool(new_text) and (
            is_stop_command(new_text)
            or (len(words) >= floor and not self._reads_like_own_speech(new_text))
        )
        if not confirmed:
            # Fold pure-echo text (reads as own speech) into the base so it can't
            # pile up and swamp the next block's novelty test; a real user (novel)
            # is NOT folded -> their words accumulate toward the floor.
            if new_text and self._reads_like_own_speech(new_text):
                st["own_folds"] = st.get("own_folds", 0) + 1
                self._word_cut_base = text
            return False
        st["cuts"] = st.get("cuts", 0) + 1
        # Fire the SAME cut the duck-confirm path uses. Keep the user's words in the
        # stream as free pre-roll -> clear the fed-stream flag so the reply-end
        # transition won't reset them out.
        self._word_cut_fed_stream = False
        self._barge_in_fired_this_run = True
        self._barge_in_suppressed_until = (
            now + max(0.0, self.config.barge_in_suppress_sec)
        )
        log.info("barge-in confirmed by speech (word-cut): %r", new_text[:80])
        self._cb.on_metric("barge_in_confirmed")
        # Keep the exact legacy log line so run-bundle tooling (grep
        # "barge-in detected") sees word-cut barges too.
        log.info("barge-in detected")
        self._cb.on_barge_in()
        return True

    def _wc_window_update(self, samples, now: float) -> None:
        """Accumulate the ~2s near-end evidence window while word-cut is live and
        emit one "word-cut near-end:" INFO line per window: post-EC mic RMS
        (avg + peak), the VAD-active fraction, and the calibrated noise floor.
        This is the capture-side ground truth the failed live batch lacked --
        rms above floor with vad_frac ~0 says the canceller passed energy the
        VAD never accepted as speech; rms at the floor says the near-end never
        survived capture at all. Capture-thread only; scalar math per block."""
        w = self._wc_win
        if not w:
            w.update(start=now, rms_sum=0.0, rms_peak=0.0, blocks=0, vad=0)
        r = float(rms(samples))
        w["rms_sum"] += r
        w["rms_peak"] = max(w["rms_peak"], r)
        w["blocks"] += 1
        if self._vad is not None and self._vad.is_speech_detected():
            w["vad"] += 1
        if now - w["start"] < 2.0:
            return
        blocks = max(1, int(w["blocks"]))
        avg = w["rms_sum"] / blocks
        vad_frac = w["vad"] / blocks
        cal = self._last_calibration or {}
        floor = float(cal.get("noise_floor_rms", 0.0) or 0.0)
        log.info(
            "word-cut near-end: rms_avg=%.4f rms_peak=%.4f vad_frac=%.2f "
            "floor=%.4f blocks=%d",
            avg, w["rms_peak"], vad_frac, floor, blocks,
        )
        st = self._wc_stats
        st.setdefault("win_rms", []).append(round(avg, 5))
        if len(st["win_rms"]) > 300:  # a very long reply can't grow unbounded
            del st["win_rms"][:100]
        self._wc_win = {}

    def _emit_word_cut_funnel(self) -> None:
        """One INFO line per reply summarizing the ADR-0013 word-cut funnel:
        blocks fed vs VAD-skipped, burst resets (+ words they dropped), the
        per-reply word high-water mark, own-speech folds, guard suppressions,
        decode errors, cuts, and near-end RMS percentiles. Every count of zero
        is meaningful: fed=0 means the VAD gate starved the recognizer (the
        run-20260706-231226 failure mode); max_words=0 with rms above floor
        means the recognizer transcribed nothing from real energy."""
        st = self._wc_stats
        wins = st.get("win_rms") or []
        p50 = p95 = 0.0
        if wins:
            ordered = sorted(wins)
            p50 = ordered[len(ordered) // 2]
            p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        log.info(
            "word-cut funnel: fed=%d skipped_quiet=%d resets=%d dropped_words=%d "
            "max_words=%d own_folds=%d guard_suppressed=%d decode_errors=%d "
            "cuts=%d nearend_rms_p50=%.4f nearend_rms_p95=%.4f",
            st.get("fed", 0), st.get("skipped_quiet", 0), st.get("resets", 0),
            st.get("dropped_words", 0), st.get("max_words", 0),
            st.get("own_folds", 0), st.get("guard_suppressed", 0),
            st.get("decode_errors", 0), st.get("cuts", 0), p50, p95,
        )
        self._wc_stats = {}
        self._wc_win = {}

    def _reads_like_own_speech(self, text: str) -> bool:
        """Does confirm-window text read as the assistant's own (ducked) echo?

        Token-overlap against the recently synthesized sentences: when most of
        the transcribed words appear in something the assistant just said, the
        recognizer is hearing the speaker, not the user. Conservative on ties --
        an empty/inaudible transcription is treated as echo (not confirmation)."""
        words = set(re.findall(r"[a-z']+", text.lower()))
        if not words:
            return True
        candidates = (getattr(self, "_now_playing", ""),) + tuple(
            getattr(self, "_recent_spoken", ())
        )
        for spoken in candidates:
            spoken_words = set(re.findall(r"[a-z']+", str(spoken).lower()))
            if spoken_words and len(words & spoken_words) / len(words) >= 0.6:
                return True
        return False

    def _barge_in_fire_eligible(self, samples, mic_raw=None) -> bool:
        """Whether this block may *start/continue* arming a barge-in.

        Combines the one-per-run latch with the VAD + coherence/level gate.
        ``samples`` is the post-AEC block (VAD + level-floor fallback); ``mic_raw``
        is the RAW pre-AEC block the coherence detector needs (it defaults to
        ``samples`` for single-arg callers/tests). Factored out of the capture loop
        so the latch behaviour is unit-testable without an audio device. The latch
        (``_barge_in_fired_this_run``) hard-caps a speaking run to a single
        interrupt: open speakers with no AEC make the VAD re-fire many times per
        utterance on the assistant's own echo, and without the latch each one
        re-cancels the (already-cancelled) turn. The latch is reset on the
        silent->speaking transition in ``_playback_loop`` so a genuinely new
        interruption still fires."""
        if self._barge_in_fired_this_run:
            return False
        # L3 playback-onset grace: while within the grace window of this reply's
        # first audible sample, the echo-coherence reference ring is still filling
        # and reads the assistant's OWN echo as a barge. Suppress here -- the one
        # chokepoint every fire path crosses -- WITHOUT stopping reference ingest /
        # chart learning (those run earlier), so the detector is calibrated the
        # instant the window lifts. A real talk-over past the window still fires.
        grace = self.config.barge_in_playback_onset_grace_sec
        # getattr default: barge fixtures build the engine bypassing __init__, so a
        # missing stamp reads as 0.0 -> grace inert (the correct partial-build state).
        onset = getattr(self, "_playback_onset_at", 0.0)
        if grace > 0.0 and onset > 0.0 and time.monotonic() < onset + grace:
            return False
        if self._vad is None or not self._vad.is_speech_detected():
            return False
        return self._looks_like_user(samples, mic_raw)

    def note_barge_in_storm(self) -> None:
        """Hook for the watchdog: a barge-in storm was detected (gate flapping,
        likely TTS leaking into the mic). Arm the debounce window so the rapid
        repeats collapse into one interrupt instead of a rattling string of
        them. Safe to call from the watchdog thread (single float write)."""
        self._barge_in_suppressed_until = (
            time.monotonic() + max(0.0, self.config.barge_in_suppress_sec)
        )

    def _final_above_floor(self, samples) -> bool:
        """L1 echo gate: True iff this completed final stands clearly above the
        device's LEARNED echo/quiet floor -- i.e. it is real speech, not the
        assistant's own residual echo or ambient noise the recognizer turned into
        words. The reference is ``max(_ambient_rms, _playback_floor_rms)``: an
        echo-borne final sits AT the playback residual-echo floor and an ambient-
        noise final at the quiet floor, while real speech is many dB above BOTH.
        Both floors are learned online per device/room, so the bar is RELATIVE (a
        dB margin), never an absolute RMS. Fail OPEN until a floor is learned (cold
        start) so the first real turn is never dropped; disabled when
        ``final_floor_margin_db <= 0`` (the dataclass default).

        NB a learned floor exists only when AEC is on (``_playback_floor_rms``) or
        ``input_loudness_margin_db > 0`` (``_ambient_rms``); with NEITHER the gate
        is inert (fail-open) -- fine for an echo-free setup, and ``_build`` logs a
        warning if the gate is configured ON with no source. The open-speaker
        configs that actually cascade run AEC, so the gate is live exactly there."""
        margin = self.config.final_floor_margin_db
        if margin <= 0.0:
            return True
        floor = max(self._ambient_rms, self._playback_floor_rms)
        if floor <= 0.0:
            return True
        return loudness_admits(rms(samples), floor, margin_db=margin)

    def _in_post_speaking_refractory(self, now: float) -> bool:
        """L2: True while within ``barge_in_refractory_sec`` of the last
        ``_speaking``->clear (a turn ended or a barge cancelled it). Used to
        suppress a re-fired barge-in on the just-cancelled utterance's echo TAIL.
        ``barge_in_refractory_sec <= 0`` disables it (the deadline is then never in
        the future, since ``_last_speaking_end`` is a past stamp)."""
        return now < self._last_speaking_end + self.config.barge_in_refractory_sec

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
