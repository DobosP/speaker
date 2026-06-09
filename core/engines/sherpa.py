from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("speaker.sherpa")


def _auto_threads() -> int:
    """Sensible CPU thread count for one ONNX model on a laptop.

    STT/TTS run on the CPU (the GPU is reserved for the LLM), so we use about
    half the logical cores, clamped to 2..4 -- sherpa-onnx rarely benefits from
    more, and we leave headroom for the capture loop and the rest of the system.
    """
    cores = os.cpu_count() or 4
    return max(2, min(4, cores // 2))

from ..asr_text import agreement_guard, restore_casing
from ..audio_frontend import CLEAN_CAPTURE_RATES, AudioResampler, apply_gain_soft_limit
from ..engine import AudioEngine, EngineCallbacks
from ..metrics import BARGE_IN_STOP, SPEECH_END, TTS_FIRST_AUDIO
from ._denoiser import build_denoiser
from ._aec import FarEndRing, PlaybackFIFO, build_aec
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
    tts_speaker_id: int = 0
    tts_speed: float = 1.0
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
        self._aec = None
        self._far_ref = None
        self._aec_ref_delay = 0
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

    # --- lazy model construction ---
    def _build(self) -> None:
        c = self.config
        self._recognizer = build_recognizer(c)
        # Optional offline second-pass recognizer for the final transcript.
        self._final_recognizer = build_final_recognizer(c)
        if self._final_recognizer is not None:
            log.info("second-pass final ASR: %s (%s)", c.asr_final_backend, c.asr_final_model)
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
            log.info(
                "AEC ACTIVE on the capture path (16 kHz, backend=%s, ref_delay=%dms)",
                c.aec_backend, c.aec_ref_delay_ms,
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
                self._fifo.flush()
                cut = True
        # Playback is being cut: drop the echo reference so the capture loop
        # doesn't keep gating barge-in against a level that is no longer audible.
        self._playback_level = 0.0
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
        if self._dtd is not None:
            self._dtd.reset()
        if self._far_ref is not None:
            self._far_ref.clear()
        if self._aec is not None:
            self._aec.reset()
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
        # RC-5 observability: track voiced speech DURING playback that the barge
        # gate rejected, so a "user kept talking but nothing fired" failure is
        # visible in the run bundle instead of being silently dropped.
        rejected_run = 0.0
        rejected_flagged = False
        block_sec = 0.1
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
                if self._aec is not None:
                    far = self._far_ref.read(samples.shape[0], self._aec_ref_delay)
                    # ONLY cancel when the assistant is actually playing (the
                    # far-end reference has energy). With no recent playback the
                    # far ring is ~zeros and the deep DTLN canceller would just
                    # process CLEAN near-end speech -- a neural net can distort it,
                    # which garbles ASR on the user's own (echo-free) voice. So
                    # skip it then: nothing to cancel, and the input reaches ASR
                    # untouched. (Also saves the DTLN cost on every idle block.)
                    if far is not None and float(np.sqrt(np.mean(
                        np.asarray(far, dtype="float64") ** 2
                    ))) > 1e-4:
                        samples = self._aec.process_16k(samples, far)
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
                    # Consume the silent->speaking re-arm signalled by the playback
                    # loop: clear the BargeSustain window so the prior reply's fires
                    # can't carry into this reply's onset (2026-06-10 self-interrupt
                    # fix; the floors were reset on the playback thread).
                    if self._barge_sustain_reset_pending:
                        self._barge_sustain_reset_pending = False
                        barge_sustain.reset()
                        rejected_run = 0.0
                    # Auto-calibrate the post-AEC residual echo+noise floor on EVERY
                    # playback block (the assistant's own cancelled echo). The barge
                    # gate keys off this floor, so the interrupt threshold tracks the
                    # current speaker volume + room noise online -- no manual
                    # calibration, and the steady echo can't self-interrupt.
                    if self._aec is not None:
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
                                # Latch: one barge-in per speaking run. The 0.5s
                                # suppress window still debounces; the latch
                                # caps the whole run so self-echo can't re-fire.
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
                                    rejected_run += block_sec
                                    if (
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
                            if not self._final_above_floor(seg):
                                # L1: at/near the device's learned echo/quiet floor --
                                # the assistant's own residual echo or ambient noise
                                # transcribed into words. Drop it (this is what breaks
                                # the open-speaker echo-final self-interrupt cascade).
                                log.info(
                                    "dropping final %r -- at/near the learned echo/quiet "
                                    "floor (echo/ambient, not speech)", final_text,
                                )
                                self._cb.on_metric("echo_floor_rejected_final")
                            elif self._should_act_on_final(seg):
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
                    # Same for the AEC adaptive state + its far-end ring, so a
                    # recovered stream starts the canceller fresh (no stale echo
                    # tail subtracted from the first recovered block).
                    if self._aec is not None:
                        self._aec.reset()
                        self._far_ref.clear()
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
        shared with the capture thread's cheap ring read). The ONE real-time
        caveat is ``EchoCoherenceDetector``'s lock, taken via
        :meth:`note_playback`: the capture thread also briefly holds it while
        ``decide`` concatenates the reference ring. That concat is bounded by
        ``coherence_ring_ms`` (~38 KB / sub-100us at the default), so the
        worst-case audio-thread stall is well inside one low-latency callback
        period today -- but it is the only contended lock on this thread, and it
        would need to move OFF it (feed coherence from a lock-free SPSC stage,
        like the far ring) before ``coherence_ring_ms`` is raised materially.
        Allocation-light but not alloc-free: two small play_sr->16k resamples per
        block (one inside ``note_playback``, one for the far ring), sub-100us."""
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
            if n > 0:
                played = view[:n]
                # Mirror the producer's old per-chunk bookkeeping, but now driven
                # by ACTUAL playback: the level EWMA (barge-in reference), the
                # coherence echo reference, and the AEC far-end ring.
                self._note_playback_level(played)
                if self._echo_coherence is not None and self._play_sr:
                    self._echo_coherence.note_playback(played, self._play_sr)
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
                    if self._stop_speaking.is_set():
                        return  # barged chunk: never enqueue it
                    if self._play_sr != self._tts_sr:
                        samples = _resample_linear(samples, self._tts_sr, self._play_sr)
                    fifo = self._fifo
                    if fifo is not None:
                        # should_abort releases a producer blocked on a full FIFO
                        # the instant a barge-in (_stop_speaking) or shutdown
                        # (not _running) is requested -- no deadlock on teardown.
                        fifo.write(
                            samples,
                            should_abort=lambda: (
                                self._stop_speaking.is_set() or not self._running.is_set()
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
                        self._fifo = PlaybackFIFO(int(self._play_sr * self.config.playback_fifo_sec))
                        with self._out_lock:
                            self._out_stream = out
                    elif not out.active:
                        # Rare safety net: with the callback path a barge-in only
                        # flushes the FIFO (the stream stays open playing silence),
                        # so the stream is normally already active here. Restart it
                        # only if something stopped it.
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
                        # Wait for the FIFO to actually PLAY OUT before tearing
                        # down the echo refs: with the callback path the producer
                        # has handed the last samples to the FIFO but the audio
                        # thread is still draining them. Bounded by a deadline +
                        # broken by not-_running / _stop_speaking so a wedged or
                        # unplugged device can never hang the worker (mirrors the
                        # 1.0s join guards in stop()). A barge-in flush makes
                        # count()==0 immediately, so this returns at once then.
                        deadline = time.monotonic() + self.config.playback_fifo_sec + 0.5
                        while (
                            self._fifo is not None
                            and self._fifo.count() > 0
                            and self._running.is_set()
                            and not self._stop_speaking.is_set()
                            and time.monotonic() < deadline
                        ):
                            time.sleep(0.01)
                        self._speaking.clear()
                        self._playback_level = 0.0  # nothing playing -> no echo ref
                        self._last_speaking_end = time.monotonic()  # arm the L2 refractory
                        if self._echo_coherence is not None:
                            self._echo_coherence.reset()  # drop the stale reference
                        if self._dtd is not None:
                            self._dtd.reset()  # re-arm per-device chart warm-up next run
                        # Drop the far-end reference + reset the canceller so a
                        # cut-off sentence's stale tail isn't subtracted from the
                        # next near-end block once playback resumes.
                        if self._far_ref is not None:
                            self._far_ref.clear()
                        if self._aec is not None:
                            self._aec.reset()
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
            incoh = 0.0
            if det is not None:
                det.decide(mic_raw)
                incoh = float(det.last_incoherent_fraction)
            resid_rms = rms(samples)
            fired = self._dtd.decide(
                raw_rms=rms(mic_raw), resid_rms=resid_rms, incoherent_fraction=incoh,
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
            floored = fired
            if (
                fired
                and self.config.dtd_residual_floor_margin_db > 0.0
                and self._playback_floor_rms > 0.0
                and not loudness_admits(
                    resid_rms, self._playback_floor_rms,
                    margin_db=self.config.dtd_residual_floor_margin_db,
                )
            ):
                floored = False  # DTD said barge, but the residual is at the echo floor
            # Log EVERY evaluation (not just fires) so a run bundle shows the full D
            # distribution -- echo-only D vs talk-over D -- to calibrate K per device.
            # ``gated`` reflects the post-floor verdict the caller acts on.
            log.debug(
                "dtd: D=%.2f K=%.1f fired=%s gated=%s (z_raw=%.2f z_resid=%.2f z_coh=%.2f) "
                "raw=%.4f resid=%.4f incoh=%.2f resid_floor=%.4f consec=%d",
                self._dtd.last_D, self._dtd.k, fired, floored, self._dtd.last_z_raw,
                self._dtd.last_z_resid, self._dtd.last_z_coh,
                rms(mic_raw), resid_rms, incoh, self._playback_floor_rms,
                self._dtd.last_consec,
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
