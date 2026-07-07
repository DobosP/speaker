#!/usr/bin/env python3
"""Microphone capture-calibration recording suite -- find the mic settings that
give the cleanest signal into the STT, ON ANY DEVICE, by ear + by ASR.

The hypothesis under test (owner, 2026-07-07): garbled STT is a *capture* problem
-- the mic level / DSP path is wrong for this machine, so poor audio reaches the
zipformer. This tool records the SAME spoken phrase several times, each through a
DIFFERENT capture calibration, into one folder. Every file is the exact 16 kHz
mono signal the recognizer would receive (same front-end as the live capture loop
in ``core/engines/sherpa.py``: input-gain / AGC / device-floor calibration ->
anti-alias resample). You then play them back, grade each by ear, and the tool
also transcribes each with an INDEPENDENT recognizer (faster-whisper) and scores
word-error-rate against the known phrase -- so the "best calibration" is picked
from your ear AND the machine, not a guess.

    # record the default preset sweep, ~5 s each, into ./calib_runs/<timestamp>/
    python -m tools.calibration_suite

    python -m tools.calibration_suite --talk-over       # DOUBLE-TALK: assistant
                                                        # speech plays while you talk
    python -m tools.calibration_suite --listen          # ear-grade the last run
                                                        # (clips loudness-matched)
    python -m tools.calibration_suite --seconds 6 --out calib_runs/mytest
    python -m tools.calibration_suite --presets raw,apm,denoise
    python -m tools.calibration_suite --list-presets
    python -m tools.calibration_suite --list-devices
    python -m tools.calibration_suite --no-asr          # ear-grade only, no whisper
    python -m tools.calibration_suite --selftest        # no mic; synthetic self-check

Each preset varies only capture-side knobs (never the ASR model), so any quality
difference is the calibration, not the recognizer:

  * ``capture_voice_comm``  -- open the OS "voice communication" path (WASAPI
    Communications on Windows / PipeWire echo-cancel source on Linux) so the
    DRIVER runs AEC + noise-suppression + AGC before the app reads a sample. This
    is why Teams sounds clean on the same laptop (docs/audio_pipeline.md).
  * ``input_agc`` + ``input_calibrate`` -- the DEVICE-AGNOSTIC dynamic path:
    measure THIS mic's own quiet floor at startup and boost a clean-but-quiet
    signal toward a healthy level, no per-machine tuning
    (core/audio_frontend.py::InputAGC / compute_input_calibration).
  * ``input_gain`` -- a static soft-limited digital boost (the quiet-mic stopgap).
  * ``apm`` -- the in-app WebRTC AudioProcessingModule always-on (AEC3 + deep
    noise suppression + AGC2 leveler + high-pass), the Teams-style DSP chain
    (core/engines/_apm.py; needs ``pip install livekit``).
  * ``denoise`` -- the GTCRN deep denoiser on the 16 kHz capture, the ML-NS half
    of the Teams recipe (fetch once: ``python -m tools.setup_models
    --denoise-model``).

``--talk-over`` is the DOUBLE-TALK test: a reference clip (the configured TTS
voice) plays on loop through the speakers while you record, so each preset shows
how well YOUR voice survives assistant playback -- the condition that actually
matters for barge-in. APM presets receive the played frames as a time-aligned
far-end (FarEndRing + AecDelayCalibrator, ADR-0012) and cancel them.

Outputs in the folder: ``NN_<preset>.wav`` per preset, ``summary.json`` (knobs +
audio metrics + transcript + WER per preset), and ``GRADES.md`` -- a grading
sheet to fill in by ear. At the end it prints a ranked table and the config block
for the winning preset to paste into ``config.local.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Presets: each is a set of capture-path knobs applied on top of the config.
# Data-driven so adding a 6th, 7th, ... is a one-line append.
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationPreset:
    """One capture calibration to A/B. Only the fields it overrides are set; the
    rest fall back to the live front-end defaults (SherpaConfig)."""

    name: str
    blurb: str                      # one line: what this changes and why
    input_gain: float = 1.0
    input_agc: bool = False
    input_calibrate: bool = False
    input_calibrate_sec: float = 1.5
    capture_voice_comm: bool = False
    capture_samplerate: int = 0     # 0 = auto (prefer device-native, then resample)
    resampler_quality: str = "HQ"
    # In-app DSP after the resampler, mirroring the engine's capture order
    # (AEC/APM -> denoiser). ``apm`` runs the WebRTC AudioProcessingModule
    # always-on (AEC3+NS+AGC2+HPF -- the Teams-style chain, core/engines/_apm.py);
    # ``denoise`` runs the GTCRN deep denoiser (core/engines/_denoiser.py).
    apm: bool = False
    apm_gain_control: bool = True   # AGC2: a proper leveler (not boost-only InputAGC)
    denoise: bool = False
    # InputAGC tuning (only used when input_agc); defaults mirror SherpaConfig.
    agc_target_rms: float = 0.12
    agc_max_gain: float = 12.0
    agc_noise_floor_rms: float = 0.004
    agc_rise: float = 0.08
    agc_fall: float = 0.4

    def config_overrides(self) -> dict:
        """The minimal ``sherpa`` config block that reproduces this preset live."""
        out: dict = {}
        if self.input_gain != 1.0:
            out["input_gain"] = self.input_gain
        if self.input_agc:
            out["input_agc"] = True
        if self.input_calibrate:
            out["input_calibrate"] = True
            if self.input_calibrate_sec != 1.5:
                out["input_calibrate_sec"] = self.input_calibrate_sec
        if self.capture_voice_comm:
            out["capture_voice_comm"] = True
        if self.capture_samplerate:
            out["capture_samplerate"] = self.capture_samplerate
        if self.apm:
            out["aec_enabled"] = True
            out["aec_backend"] = "apm"
            out["apm_always_on"] = True
            out["apm_gain_control"] = self.apm_gain_control
        if self.denoise:
            out["denoise_enabled"] = True  # denoise_model path set by setup_models
        return out


# Every known preset. Round 1 (2026-07-07 ear grades: raw=4 > voice_comm=3.5 =
# gain_boost=3.5 > agc=2 = voice_comm_agc=2) established that the boost-only
# InputAGC audibly pumps the noise floor and voice-comm alone doesn't beat raw;
# round 2 (the new default sweep) tests the Teams-style DSP the first round
# lacked: deep noise suppression (GTCRN) and the full WebRTC APM chain.
ALL_PRESETS: list[CalibrationPreset] = [
    CalibrationPreset(
        name="raw",
        blurb="Baseline: raw mic, no gain, no AGC, no OS DSP (what a naive app does).",
    ),
    CalibrationPreset(
        name="voice_comm",
        blurb="OS voice-comm capture (WASAPI Communications / PipeWire EC) -- the "
              "Teams path: driver AEC+NS+AGC before the app reads a sample.",
        capture_voice_comm=True,
    ),
    CalibrationPreset(
        name="denoise",
        blurb="GTCRN deep noise suppression on the 16 kHz capture -- the ML-NS half "
              "of the Teams recipe (fetch once: python -m tools.setup_models "
              "--denoise-model).",
        denoise=True,
    ),
    CalibrationPreset(
        name="apm",
        blurb="WebRTC APM always-on (AEC3+NS+AGC2+HPF) -- the full Teams-style DSP "
              "chain in-app; AGC2 is a real leveler, unlike the boost-only InputAGC.",
        apm=True,
    ),
    CalibrationPreset(
        name="voice_comm_denoise",
        blurb="Teams-parity candidate: OS voice-comm capture + GTCRN deep NS on top.",
        capture_voice_comm=True,
        denoise=True,
    ),
    CalibrationPreset(
        name="agc_calibrated",
        blurb="Device-agnostic dynamic AGC: measure THIS mic's quiet floor at "
              "startup, then boost a clean-but-quiet signal toward a healthy level.",
        input_agc=True,
        input_calibrate=True,
    ),
    CalibrationPreset(
        name="gain_boost",
        blurb="Static x4 soft-limited digital boost (the quiet-mic stopgap; tests "
              "whether a flat boost helps or just amplifies noise/clipping).",
        input_gain=4.0,
    ),
    CalibrationPreset(
        name="voice_comm_agc",
        blurb="OS voice-comm + in-app dynamic AGC together (belt-and-suspenders; "
              "also exposes the double-AGC 'pumping' risk if the OS already levels).",
        capture_voice_comm=True,
        input_agc=True,
        input_calibrate=True,
    ),
]

# The default sweep: raw as the control + the Teams-style DSP candidates.
_DEFAULT_NAMES = ("raw", "voice_comm", "denoise", "apm", "voice_comm_denoise")
DEFAULT_PRESETS: list[CalibrationPreset] = [
    p for p in ALL_PRESETS if p.name in _DEFAULT_NAMES
]

# A phonetically broad default phrase: plosives, sibilants, fricatives, nasals,
# and a couple of numbers -- the sounds a mis-levelled mic garbles first.
DEFAULT_PHRASE = (
    "The quick brown fox jumps over the lazy dog while she sells six bright "
    "seashells by the shore at half past three."
)


# --------------------------------------------------------------------------- #
# Capture front-end -- mirrors core/engines/sherpa.py::_capture_loop per block.
# Kept as a small stateful object so it is unit-testable over a synthetic block
# source with no microphone.
# --------------------------------------------------------------------------- #
class CaptureFrontEnd:
    """Per-block capture processing identical to the live engine: AGC (or static
    gain) BEFORE an anti-alias resample to 16 kHz, then the in-app DSP in engine
    order (APM/AEC -> GTCRN denoiser). Stateful (AGC ramp + soxr FIR + APM/GTCRN
    streaming state carry across blocks).

    ``sherpa_cfg`` is the merged ``sherpa`` config dict; needed only for the APM /
    denoiser presets (model path, threads). Both builders FAIL OPEN exactly like
    the engine: missing livekit / missing GTCRN model -> that stage is skipped and
    ``warnings`` records why, so a preset can never crash the sweep -- but it also
    won't silently pretend to be different from raw."""

    def __init__(self, preset: CalibrationPreset, capture_sr: int, asr_sr: int = 16000,
                 sherpa_cfg: Optional[dict] = None):
        from core.audio_frontend import AudioResampler, InputAGC

        self.preset = preset
        self.asr_sr = asr_sr
        self.warnings: list[str] = []
        self.agc = (
            InputAGC(
                target_rms=preset.agc_target_rms,
                max_gain=preset.agc_max_gain,
                noise_floor_rms=preset.agc_noise_floor_rms,
                rise=preset.agc_rise,
                fall=preset.agc_fall,
            )
            if preset.input_agc
            else None
        )
        self.resampler = (
            AudioResampler(capture_sr, asr_sr, quality=preset.resampler_quality)
            if capture_sr != asr_sr
            else None
        )
        # --- in-app DSP (engine order: APM/AEC first, then the denoiser) ---
        base = dict(sherpa_cfg or {})
        self._aec = None
        if preset.apm:
            try:
                from core.engines._aec import build_aec
                from core.engines.sherpa import SherpaConfig

                c = SherpaConfig.from_dict({**base, **preset.config_overrides()})
                self._aec = build_aec(c)
                if self._aec is None:
                    self.warnings.append(
                        "APM unavailable (pip install livekit) -- preset ran as raw")
            except Exception as exc:  # noqa: BLE001 - fail open like the engine
                self.warnings.append(f"APM build failed ({exc}) -- preset ran as raw")
        self._denoiser = None
        if preset.denoise:
            try:
                from core.engines._denoiser import build_denoiser
                from core.engines.sherpa import SherpaConfig

                c = SherpaConfig.from_dict({**base, **preset.config_overrides()})
                self._denoiser = build_denoiser(c)
                if self._denoiser is None:
                    self.warnings.append(
                        "GTCRN model missing (python -m tools.setup_models "
                        "--denoise-model) -- denoise stage skipped")
            except Exception as exc:  # noqa: BLE001
                self.warnings.append(f"denoiser build failed ({exc}) -- stage skipped")
        # Far-end source for the APM during --talk-over (None -> zeros far, the
        # engine's apm_always_on idle behaviour).
        self._far_ring = None
        self._delay_cal = None
        self._seed_delay = 0

    def set_far_source(self, ring, delay_cal, seed_delay_samples: int) -> None:
        """Wire the played-reference ring + delay calibrator (talk-over mode), so
        the APM cancels the speaker audio instead of treating it as near-end."""
        self._far_ring = ring
        self._delay_cal = delay_cal
        self._seed_delay = int(seed_delay_samples)

    def set_noise_floor(self, floor: float) -> None:
        if self.agc is not None:
            self.agc.noise_floor_rms = float(floor)

    def resample_only(self, block) -> "np.ndarray":
        """Downsample a raw block to the ASR rate WITHOUT gain/AGC -- used to build
        the calibration room-tone window (compute_input_calibration wants the
        pre-AGC signal at the ASR rate)."""
        s = np.asarray(block, dtype="float32").reshape(-1)
        if self.resampler is not None:
            s = self.resampler.process(s)
        return s

    def process(self, block) -> "np.ndarray":
        from core.audio_frontend import apply_gain_soft_limit

        s = np.asarray(block, dtype="float32").reshape(-1)
        # AGC (dynamic) takes precedence over the static gain, exactly like the
        # engine; gain/AGC run BEFORE the resampler so saturation harmonics above
        # 8 kHz are filtered out before the recognizer sees them.
        if self.agc is not None:
            s = self.agc.process(s)
        elif self.preset.input_gain != 1.0:
            s = apply_gain_soft_limit(s, self.preset.input_gain)
        if self.resampler is not None:
            s = self.resampler.process(s)
        # APM (always-on): far = the played reference at the measured speaker->mic
        # delay during --talk-over, zeros otherwise (the engine's idle behaviour --
        # the echo canceller self-cancels to a no-op on a zero far).
        if self._aec is not None and s.size:
            if self._far_ring is not None:
                far0 = self._far_ring.read(s.size, 0)
                delay = self._seed_delay
                if self._delay_cal is not None:
                    self._delay_cal.observe(s, far0)
                    delay = self._delay_cal.current_delay_samples()
                far = self._far_ring.read(s.size, delay)
            else:
                far = np.zeros(s.size, dtype="float32")
            s = self._aec.process_16k(s, far)
            s = np.asarray(s, dtype="float32").reshape(-1)
        # GTCRN deep NS last, exactly where the engine applies it (after AEC,
        # before every consumer). Passthrough-on-error inside.
        if self._denoiser is not None and s.size:
            s = np.asarray(self._denoiser.process_16k(s), dtype="float32").reshape(-1)
        return s


# --------------------------------------------------------------------------- #
# Microphone capture
# --------------------------------------------------------------------------- #
def _open_input_stream(device, *, pinned_sr: int, voice_comm: bool, block_sec: float):
    """Open a blocking input stream, mirroring the engine's device/rate choices.

    Prefers the device's NATIVE rate (no driver reconfiguration -> no USB
    self-mute; the anti-alias soxr resampler converts to 16 kHz), unless a rate is
    pinned. ``voice_comm`` requests the WASAPI Communications category on Windows;
    it fails open (raw stream) on non-WASAPI hosts. Returns ``(stream, capture_sr)``.
    """
    import sounddevice as sd

    from core.engines.sherpa import _norm_device

    dev = _norm_device(device)

    extra = None
    if voice_comm:
        try:
            extra = sd.WasapiSettings(communications=True)
        except Exception:  # noqa: BLE001 - not WASAPI (Linux/macOS/old sounddevice)
            extra = None  # fail open; on Linux use a PipeWire echo-cancel source

    try:
        dev_default = int(round(sd.query_devices(dev, kind="input")["default_samplerate"]))
    except Exception:  # noqa: BLE001
        dev_default = 48000

    if pinned_sr:
        candidates = [int(pinned_sr), dev_default]
    else:
        candidates = [dev_default, 48000, 44100, 16000]
    seen: set[int] = set()
    candidates = [c for c in candidates if c and not (c in seen or seen.add(c))]

    last_err: Optional[Exception] = None
    for sr in candidates:
        try:
            stream = sd.InputStream(
                channels=1,
                samplerate=sr,
                dtype="float32",
                blocksize=int(sr * block_sec),
                device=dev,
                extra_settings=extra,
            )
            stream.start()
            return stream, sr
        except Exception as exc:  # noqa: BLE001 - try the next rate
            last_err = exc
            continue
    raise RuntimeError(f"could not open input device {device!r} at any rate: {last_err}")


def _looped_chunk(data: "np.ndarray", pos: int, frames: int) -> tuple["np.ndarray", int]:
    """Next ``frames`` samples of ``data`` starting at ``pos``, wrapping around at
    the end (the talk-over reference loops for as long as the recording runs).
    Pure + separate so the wraparound is unit-testable without an audio device."""
    n = data.shape[0]
    if n == 0:
        return np.zeros(frames, dtype="float32"), 0
    out = np.empty(frames, dtype="float32")
    filled = 0
    while filled < frames:
        take = min(frames - filled, n - pos)
        out[filled:filled + take] = data[pos:pos + take]
        filled += take
        pos = (pos + take) % n
    return out, pos


class TalkOverPlayer:
    """Plays a reference WAV on loop through the speakers while the mic records,
    teeing the EXACT played frames (resampled to 16 kHz) into a ``FarEndRing`` --
    from the output callback, not producer-side, so the ring tracks acoustic
    playback instead of running ahead of it (core/engines/sherpa.py:3002 pattern).
    The ring is what lets the APM preset cancel the speaker audio as far-end."""

    def __init__(self, wav_path: str, *, ring=None, device=None, gain: float = 1.0):
        from core.audio_frontend import AudioResampler

        data, sr = _read_wav(wav_path)
        self._data = (np.asarray(data, dtype="float32").reshape(-1) * float(gain)).clip(-1.0, 1.0)
        self._sr = sr
        self._pos = 0
        self._ring = ring
        self._device = device
        self._stream = None
        # Stateful resampler: played chunk (device rate) -> 16 kHz for the ring.
        self._tee_resampler = AudioResampler(sr, 16000) if sr != 16000 else None

    def start(self) -> None:
        import sounddevice as sd

        def _cb(outdata, frames, time_info, status):  # noqa: ARG001 - sd signature
            chunk, self._pos = _looped_chunk(self._data, self._pos, frames)
            outdata[:, 0] = chunk
            if self._ring is not None:
                try:
                    tee = chunk if self._tee_resampler is None else self._tee_resampler.process(chunk)
                    if tee.size:
                        self._ring.push(tee)
                except Exception:  # noqa: BLE001 - never crash the audio callback
                    pass

        self._stream = sd.OutputStream(
            channels=1, samplerate=self._sr, dtype="float32",
            device=self._device, latency="low", callback=_cb,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None


def _resolve_talkover_wav(spec: Optional[str], folder: str, sherpa_cfg: dict) -> Optional[str]:
    """The reference clip --talk-over plays: an explicit WAV path, else a sentence
    synthesized by the CONFIGURED TTS voice (so the echo matches what the real
    assistant sounds like), else the newest committed assistant clip from
    logs/live. Returns None (with a message) only if all three fail."""
    if spec and spec != "auto":
        return spec if os.path.exists(spec) else None
    out = os.path.join(folder, "talkover_ref.wav")
    try:
        from tools.autotest.audio import synth_to_wav

        synth_to_wav(
            "I am the assistant and I am speaking right now. Please talk over me "
            "and keep talking while my voice keeps playing from the speaker.",
            out, sherpa_cfg=sherpa_cfg,
        )
        if os.path.exists(out):
            return out
    except Exception as exc:  # noqa: BLE001 - fall through to a canned clip
        print(f"  (TTS synth for the talk-over reference failed: {exc})")
    import glob

    clips = sorted(glob.glob(os.path.join("logs", "live", "*", "*", "assistant", "*.wav")),
                   key=os.path.getmtime, reverse=True)
    return clips[0] if clips else None


@dataclass
class RecordResult:
    preset: CalibrationPreset
    samples: "np.ndarray"          # 16 kHz mono float32, what the ASR would hear
    capture_sr: int
    calibration: Optional[dict]    # compute_input_calibration output, or None
    metrics: dict = field(default_factory=dict)
    transcript: str = ""
    wer: Optional[float] = None
    asr_confidence: Optional[float] = None   # mean whisper avg_logprob (closer to 0 = cleaner)
    wav_path: str = ""
    talk_over: bool = False                  # recorded while reference audio played
    warnings: list = field(default_factory=list)  # skipped DSP stages etc.


def record_preset(
    preset: CalibrationPreset,
    *,
    device,
    seconds: float,
    block_sec: float = 0.1,
    asr_sr: int = 16000,
    sherpa_cfg: Optional[dict] = None,
    talkover_wav: Optional[str] = None,
    talkover_gain: float = 1.0,
    output_device=None,
    say: Callable[[str], None] = print,
    countdown: Callable[[CalibrationPreset], None] = lambda p: None,
    quiet: Callable[[CalibrationPreset], None] = lambda p: None,
) -> RecordResult:
    """Open the mic with this preset's calibration, (optionally) measure the room
    floor, then record ``seconds`` of speech through the live front-end.

    ``talkover_wav``: play this clip on loop through the speakers DURING the
    recording (the double-talk test). For an APM preset the played frames are fed
    to the canceller as far-end at the auto-measured speaker->mic delay
    (FarEndRing + AecDelayCalibrator, ADR-0012); for every other preset the echo
    simply lands in the capture -- which is the comparison the mode exists for."""
    from core.audio_frontend import compute_input_calibration

    stream, capture_sr = _open_input_stream(
        device, pinned_sr=preset.capture_samplerate,
        voice_comm=preset.capture_voice_comm, block_sec=block_sec,
    )
    fe = CaptureFrontEnd(preset, capture_sr, asr_sr=asr_sr, sherpa_cfg=sherpa_cfg)
    for w in fe.warnings:
        say(f"    !! {w}")

    player: Optional[TalkOverPlayer] = None
    if talkover_wav:
        ring = None
        if preset.apm and fe._aec is not None:
            from core.engines._aec import AecDelayCalibrator, FarEndRing

            ring = FarEndRing()
            ref_delay_ms = float((sherpa_cfg or {}).get("aec_ref_delay_ms", 80) or 80)
            seed = int(asr_sr * ref_delay_ms / 1000.0)
            fe.set_far_source(ring, AecDelayCalibrator(asr_sr, seed_delay_samples=seed), seed)
        player = TalkOverPlayer(talkover_wav, ring=ring, device=output_device,
                                gain=talkover_gain)

    frames = int(capture_sr * block_sec)
    cal: Optional[dict] = None
    out_blocks: list[np.ndarray] = []
    try:
        if preset.input_calibrate:
            quiet(preset)  # ask the user to stay quiet for the room-tone window
            cal_blocks: list[np.ndarray] = []
            n_cal = max(1, int(round(preset.input_calibrate_sec / block_sec)))
            for _ in range(n_cal):
                data, _ = stream.read(frames)
                s = fe.resample_only(data)
                if s.size:
                    cal_blocks.append(s)
            if cal_blocks:
                cal = compute_input_calibration(cal_blocks)
                if preset.input_agc:
                    fe.set_noise_floor(cal["noise_floor_rms"])

        countdown(preset)  # "3.. 2.. 1.. speak"
        if player is not None:
            player.start()  # speaker starts talking; user talks over it
        n_rec = max(1, int(round(seconds / block_sec)))
        for _ in range(n_rec):
            data, _ = stream.read(frames)
            out_blocks.append(fe.process(data))
    finally:
        if player is not None:
            player.stop()
        try:
            stream.stop()
            stream.close()
        except Exception:  # noqa: BLE001
            pass

    samples = (
        np.concatenate(out_blocks).astype("float32")
        if out_blocks else np.zeros(0, dtype="float32")
    )
    return RecordResult(
        preset=preset, samples=samples, capture_sr=capture_sr, calibration=cal,
        metrics=_audio_metrics(samples, asr_sr, cal),
        talk_over=bool(talkover_wav), warnings=list(fe.warnings),
    )


# --------------------------------------------------------------------------- #
# Metrics + scoring
# --------------------------------------------------------------------------- #
def _audio_metrics(samples, sr: int, cal: Optional[dict]) -> dict:
    from core.audio_frontend import audio_quality_metrics

    m = dict(audio_quality_metrics(samples, sr))
    m["duration_s"] = round(samples.size / float(sr), 2) if sr else 0.0
    # A crude but useful voiced-vs-floor SNR proxy: level of the loud (voiced)
    # samples over the quiet (floor) samples, in dB.
    m["est_snr_db"] = _est_snr_db(samples)
    if cal is not None:
        m["calib_ambient_rms"] = round(float(cal.get("ambient_rms", 0.0)), 5)
        m["calib_noise_floor_rms"] = round(float(cal.get("noise_floor_rms", 0.0)), 5)
        m["calib_clip_pct"] = round(float(cal.get("clipping_fraction", 0.0)) * 100.0, 2)
    return m


def _est_snr_db(samples) -> Optional[float]:
    """Rough SNR: 90th-pct short-window energy (speech) over 10th-pct (floor)."""
    x = np.asarray(samples, dtype="float32").reshape(-1)
    if x.size < 1600:
        return None
    w = 320  # 20 ms at 16 kHz
    n = (x.size // w) * w
    e = np.sqrt((x[:n].reshape(-1, w).astype("float64") ** 2).mean(axis=1))
    e = e[e > 0]
    if e.size < 4:
        return None
    hi = float(np.percentile(e, 90))
    lo = float(np.percentile(e, 10))
    if lo <= 1e-7 or hi <= 1e-7:
        return None
    return round(20.0 * np.log10(hi / lo), 1)


def _normalize_text(s: str) -> list[str]:
    keep = []
    for ch in s.lower():
        keep.append(ch if (ch.isalnum() or ch.isspace()) else " ")
    return "".join(keep).split()


def word_error_rate(reference: str, hypothesis: str) -> Optional[float]:
    """Word-level edit distance / reference length. ``None`` if the reference is
    empty. 0.0 = perfect; can exceed 1.0 when the hypothesis over-inserts."""
    ref = _normalize_text(reference)
    hyp = _normalize_text(hypothesis)
    if not ref:
        return None
    # Levenshtein over word tokens (classic DP).
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return round(prev[-1] / len(ref), 3)


_WHISPER_CACHE: dict = {}


def _transcribe(path: str, model_name: str) -> tuple[str, Optional[float], Optional[str]]:
    """Independent transcript + reference-free confidence via faster-whisper.

    Returns ``(text, confidence, error)``. ``confidence`` is the duration-weighted
    mean of whisper's per-segment ``avg_logprob`` (a value <= 0; CLOSER TO 0 means
    the recognizer was more certain -> cleaner audio). This is the capture-quality
    signal that works even when you DON'T read the fixed phrase (free speech), for
    which WER is meaningless. Fails OPEN: a missing faster-whisper is reported, not
    raised -- ear-grading still works. The model is cached across clips."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        return "", None, f"faster-whisper not installed ({exc}); pip install faster-whisper"
    except Exception as exc:  # noqa: BLE001
        return "", None, f"{type(exc).__name__}: {exc}"
    try:
        from tools.transcribe_run import _load_wav_16k

        model = _WHISPER_CACHE.get(model_name)
        if model is None:
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
            _WHISPER_CACHE[model_name] = model
        audio = _load_wav_16k(path)
        segments, _info = model.transcribe(audio, language="en", vad_filter=True)
        texts: list[str] = []
        lp_sum = 0.0
        dur = 0.0
        for s in segments:
            texts.append(s.text.strip())
            d = max(0.01, float(s.end) - float(s.start))
            lp_sum += float(s.avg_logprob) * d
            dur += d
        conf = round(lp_sum / dur, 3) if dur > 0 else None
        return " ".join(texts).strip(), conf, None
    except Exception as exc:  # noqa: BLE001
        return "", None, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# WAV I/O
# --------------------------------------------------------------------------- #
def write_wav(path: str, samples, sr: int = 16000) -> None:
    x = np.clip(np.asarray(samples, dtype="float32").reshape(-1), -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


def _play_wav(path: str) -> None:
    import sounddevice as sd

    w = wave.open(path, "rb")
    sr = w.getframerate()
    data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype("float32") / 32768.0
    w.close()
    sd.play(data, sr)
    sd.wait()


def _read_wav(path: str) -> tuple["np.ndarray", int]:
    w = wave.open(path, "rb")
    sr = w.getframerate()
    data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype("float32") / 32768.0
    w.close()
    return data, sr


def loudness_normalize(samples, target_rms: float = 0.12, *, max_gain: float = 20.0):
    """Scale a clip so its VOICED level ~= ``target_rms``, so quiet and loud
    captures can be A/B'd for CLARITY, not loudness (a faint clip otherwise sounds
    'worse' only because it's quiet). Reference level is the 90th-percentile of the
    20 ms short-window RMS -- the level of the actual speech, ignoring the silent
    head/tail/gaps that a whole-clip RMS would let drag the number down."""
    x = np.asarray(samples, dtype="float32").reshape(-1)
    if x.size < 320:
        return x
    w = 320
    n = (x.size // w) * w
    e = np.sqrt((x[:n].reshape(-1, w).astype("float64") ** 2).mean(axis=1))
    e = e[e > 1e-6]
    ref = float(np.percentile(e, 90)) if e.size else 0.0
    if ref <= 1e-6:
        return x
    gain = min(float(target_rms) / ref, float(max_gain))
    from core.audio_frontend import apply_gain_soft_limit

    return apply_gain_soft_limit(x, gain)


def _latest_run_folder() -> Optional[str]:
    import glob

    folders = [f for f in glob.glob(os.path.join("calib_runs", "*")) if os.path.isdir(f)]
    return max(folders, key=os.path.getmtime) if folders else None


def listen_mode(folder: str, *, target_rms: float = 0.12, loops: int = 2,
                gap_sec: float = 0.6) -> int:
    """Loudness-match every clip in ``folder`` and play them back labeled, for a
    fair by-ear grading of CLARITY. Writes the level-matched copies to
    ``<folder>/normalized/`` so they can be replayed in any audio player too."""
    import glob
    import time as _t

    wavs = sorted(
        p for p in glob.glob(os.path.join(folder, "*.wav"))
        if "normalized" not in os.path.basename(os.path.dirname(p))
    )
    if not wavs:
        print(f"No .wav clips found in {folder}")
        return 1
    norm_dir = os.path.join(folder, "normalized")
    os.makedirs(norm_dir, exist_ok=True)

    print(f"\nLoudness-matching {len(wavs)} clips to equal volume "
          f"(target_rms={target_rms}) so you grade CLARITY, not loudness.")
    print(f"Level-matched copies -> {os.path.abspath(norm_dir)}\n")

    entries = []
    for p in wavs:
        name = os.path.splitext(os.path.basename(p))[0]
        data, sr = _read_wav(p)
        normed = loudness_normalize(data, target_rms)
        outp = os.path.join(norm_dir, f"{name}.wav")
        write_wav(outp, normed, sr)
        entries.append((name, outp))

    try:
        for rnd in range(max(1, loops)):
            print(f"--- listening pass {rnd + 1}/{loops} ---")
            for name, outp in entries:
                print(f"  ♪ playing: {name}", flush=True)
                try:
                    _play_wav(outp)
                except Exception as exc:  # noqa: BLE001
                    print(f"    playback failed ({exc}); open {outp} manually")
                _t.sleep(gap_sec)
            print()
    except KeyboardInterrupt:
        print("\n(stopped)")

    print("Grade each by ear (1=worst .. 5=best) for CLARITY / least noise / most")
    print("natural, and tell me the numbers or just the winner. Files:")
    for name, outp in entries:
        print(f"  {name:<22} {outp}")
    return 0


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def phrase_was_read(results: list[RecordResult]) -> bool:
    """True if at least one clip closely matches the reference phrase -- i.e. the
    user actually READ it, so WER is meaningful. When everyone free-speaks, WER is
    ~1.0 everywhere and we rank by reference-free ASR confidence instead."""
    return any(r.wer is not None and r.wer < 0.5 for r in results)


def rank_results(results: list[RecordResult]) -> list[RecordResult]:
    """Best first. When the phrase was read, primary key is WER; otherwise it is
    reference-free ASR confidence (whisper avg_logprob, closer to 0 = cleaner
    capture). Tiebreak toward a healthy level (RMS near the 0.12 target) + low clip."""
    use_wer = phrase_was_read(results)

    def key(r: RecordResult):
        rms = float(r.metrics.get("rms") or 0.0)
        clip = float(r.metrics.get("clip_pct") or 0.0)
        level_penalty = abs(rms - 0.12)   # distance from the healthy operating point
        if use_wer:
            primary = r.wer if r.wer is not None else 9.9
        else:
            # avg_logprob is <= 0; higher (closer to 0) is better -> negate for asc sort.
            primary = -(r.asr_confidence if r.asr_confidence is not None else -99.0)
        return (primary, clip, level_penalty)
    return sorted(results, key=key)


def write_summary(folder: str, results: list[RecordResult], phrase: str) -> str:
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "phrase": phrase,
        "presets": [
            {
                "name": r.preset.name,
                "wav": os.path.basename(r.wav_path),
                "blurb": r.preset.blurb,
                "config_overrides": r.preset.config_overrides(),
                "capture_sr": r.capture_sr,
                "metrics": r.metrics,
                "transcript": r.transcript,
                "wer": r.wer,
                "asr_confidence": r.asr_confidence,
                "talk_over": r.talk_over,
                "warnings": r.warnings,
            }
            for r in results
        ],
    }
    path = os.path.join(folder, "summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def write_grades_sheet(folder: str, results: list[RecordResult], phrase: str) -> str:
    lines = [
        "# Calibration grading sheet",
        "",
        f"Phrase you read: **{phrase}**",
        "",
        "Play each WAV, then write a grade 1 (worst) - 5 (best) for how clean /",
        "intelligible YOUR voice sounds. The ASR transcript + WER is the machine's",
        "opinion; your ear is the tiebreak. Tell Claude the grades (or the winner).",
        "",
    ]
    for i, r in enumerate(results, 1):
        wer = "n/a" if r.wer is None else f"{r.wer:.2f}"
        rms = r.metrics.get("rms")
        clip = r.metrics.get("clip_pct")
        snr = r.metrics.get("est_snr_db")
        lines += [
            f"## {i:02d}. `{os.path.basename(r.wav_path)}`  ({r.preset.name})",
            f"- What it is: {r.preset.blurb}",
            f"- ASR heard: \"{r.transcript or '(nothing / ASR skipped)'}\"",
            f"- WER: {wer}   conf: {r.asr_confidence}   RMS: {rms}   clip%: {clip}   est_SNR_dB: {snr}",
            "- **Your grade (1-5): ____**   notes: ",
            "",
        ]
    path = os.path.join(folder, "GRADES.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def print_ranking(results: list[RecordResult]) -> None:
    ranked = rank_results(results)
    read = phrase_was_read(results)
    driver = "WER (you read the phrase)" if read else "ASR confidence (free speech; WER ignored)"
    print("\n" + "=" * 78)
    print(f"RANKING (best first; ranked by {driver})")
    print("  WER: word-error vs phrase, lower=better.  conf: whisper avg_logprob, closer to 0=cleaner.")
    print("=" * 78)
    print(f"{'#':>2}  {'preset':<20} {'WER':>6} {'conf':>7} {'RMS':>7} {'clip%':>6} {'SNRdB':>6}  file")
    for i, r in enumerate(ranked, 1):
        wer = "  n/a" if r.wer is None else f"{r.wer:6.2f}"
        conf = f"{r.asr_confidence:7.3f}" if isinstance(r.asr_confidence, (int, float)) else f"{'':>7}"
        rms = r.metrics.get("rms")
        clip = r.metrics.get("clip_pct")
        snr = r.metrics.get("est_snr_db")
        rms_s = f"{rms:7.4f}" if isinstance(rms, (int, float)) else f"{'':>7}"
        clip_s = f"{clip:6.2f}" if isinstance(clip, (int, float)) else f"{'':>6}"
        snr_s = f"{snr:6.1f}" if isinstance(snr, (int, float)) else f"{'':>6}"
        print(f"{i:>2}  {r.preset.name:<20} {wer} {conf} {rms_s} {clip_s} {snr_s}  {os.path.basename(r.wav_path)}")

    if not read:
        print("\n(Every WER is high because the reference phrase wasn't read -- that means"
              "\n free speech, NOT garbled audio. Coherent transcripts = clean capture.)")
    best = ranked[0]
    print("\n" + "-" * 72)
    print(f"ASR-best calibration: {best.preset.name}")
    print(f"  {best.preset.blurb}")
    overrides = best.preset.config_overrides()
    print("\nPaste into config.local.json to make it live:")
    print(json.dumps({"sherpa": overrides or {"(raw defaults -- no overrides needed)": True}}, indent=2))
    print(
        "\nNote: `input_agc`+`input_calibrate` and `capture_voice_comm` are the two\n"
        "DEVICE-AGNOSTIC options -- they measure/adapt to the machine, so a winner\n"
        "among them generalizes. A static `input_gain` winner is machine-specific.\n"
        "Grade by ear too (GRADES.md) and tell Claude -- the ear breaks WER ties."
    )


# --------------------------------------------------------------------------- #
# Synthetic self-test (no microphone) -- exercises the whole path offline.
# --------------------------------------------------------------------------- #
def _synthetic_capture(seconds: float, sr: int = 16000):
    """A quiet-then-voiced speech-like signal (formant-ish tones + light noise)
    so --selftest can drive record/process/score with no mic on any box."""
    t = np.arange(int(seconds * sr)) / float(sr)
    voiced = (
        0.20 * np.sin(2 * np.pi * 140 * t)
        + 0.12 * np.sin(2 * np.pi * 700 * t)
        + 0.06 * np.sin(2 * np.pi * 2500 * t)
    ).astype("float32")
    env = (np.sin(2 * np.pi * 3.0 * t) * 0.5 + 0.5).astype("float32")  # syllable-rate
    sig = voiced * env + 0.003 * np.random.default_rng(0).standard_normal(t.size).astype("float32")
    return sig.astype("float32")


def _run_selftest(presets: list[CalibrationPreset], folder: str, seconds: float,
                  asr_model: str, do_asr: bool,
                  sherpa_cfg: Optional[dict] = None) -> list[RecordResult]:
    from core.audio_frontend import compute_input_calibration

    sr = 16000
    block = int(sr * 0.1)
    results: list[RecordResult] = []
    for idx, preset in enumerate(presets, 1):
        sig = _synthetic_capture(seconds + preset.input_calibrate_sec, sr)
        fe = CaptureFrontEnd(preset, sr, asr_sr=sr, sherpa_cfg=sherpa_cfg)
        for w in fe.warnings:
            print(f"  [selftest] {preset.name}: !! {w}")
        pos = 0
        cal = None
        if preset.input_calibrate:
            n_cal = max(1, int(round(preset.input_calibrate_sec / 0.1)))
            cal_blocks = []
            for _ in range(n_cal):
                b = sig[pos:pos + block] * 0.02  # room tone is quiet
                pos += block
                cal_blocks.append(fe.resample_only(b))
            cal = compute_input_calibration(cal_blocks)
            if preset.input_agc:
                fe.set_noise_floor(cal["noise_floor_rms"])
        out = []
        while pos + block <= sig.size:
            out.append(fe.process(sig[pos:pos + block]))
            pos += block
        samples = np.concatenate(out).astype("float32") if out else np.zeros(0, "float32")
        r = RecordResult(preset=preset, samples=samples, capture_sr=sr, calibration=cal,
                          metrics=_audio_metrics(samples, sr, cal))
        r.wav_path = os.path.join(folder, f"{idx:02d}_{preset.name}.wav")
        write_wav(r.wav_path, samples, sr)
        if do_asr:
            r.transcript, r.asr_confidence, err = _transcribe(r.wav_path, asr_model)
            if err:
                print(f"  [asr] {preset.name}: {err}")
        results.append(r)
        print(f"  [selftest] {preset.name}: {samples.size/sr:.1f}s -> {os.path.basename(r.wav_path)}")
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _select_presets(spec: str) -> list[CalibrationPreset]:
    by_name = {p.name: p for p in ALL_PRESETS}
    if not spec:
        return list(DEFAULT_PRESETS)
    chosen = []
    for name in [s.strip() for s in spec.split(",") if s.strip()]:
        if name not in by_name:
            raise SystemExit(f"unknown preset '{name}'. Known: {', '.join(by_name)}")
        chosen.append(by_name[name])
    return chosen


def _countdown(preset: CalibrationPreset) -> None:
    print(f"\n>>> [{preset.name}] {preset.blurb}")
    for n in (3, 2, 1):
        print(f"    speak in {n}...", flush=True)
        time.sleep(1.0)
    print("    >>> SPEAK NOW (read the phrase) <<<", flush=True)


def _quiet(preset: CalibrationPreset) -> None:
    print(f"\n>>> [{preset.name}] calibrating room floor -- STAY QUIET "
          f"({preset.input_calibrate_sec:.1f}s)...", flush=True)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Microphone calibration recording suite.")
    ap.add_argument("--out", default=None, help="output folder (default: calib_runs/<timestamp>)")
    ap.add_argument("--seconds", type=float, default=5.0, help="record seconds per preset")
    ap.add_argument("--phrase", default=DEFAULT_PHRASE, help="the phrase to read (for WER)")
    ap.add_argument("--presets", default="", help="comma list to run a subset (see --list-presets)")
    ap.add_argument("--repeat", type=int, default=1, help="record each preset N times")
    ap.add_argument("--input-device", default=None, help="mic device name/index (default: config/system)")
    ap.add_argument("--output-device", default=None, help="speaker device for --talk-over playback")
    ap.add_argument("--talk-over", nargs="?", const="auto", default=None, metavar="WAV",
                    help="DOUBLE-TALK test: play a reference clip on loop through the "
                         "speakers WHILE recording (default clip: synthesize with the "
                         "configured TTS voice, else the newest logs/live assistant clip). "
                         "APM presets cancel it as far-end; every other preset shows how "
                         "much echo swamps your voice.")
    ap.add_argument("--talkover-gain", type=float, default=1.0,
                    help="scale the talk-over playback level (1.0 = as synthesized)")
    ap.add_argument("--config", default="config.json", help="config file (for the default input device)")
    ap.add_argument("--asr-model", default="small.en", help="faster-whisper model for scoring")
    ap.add_argument("--no-asr", action="store_true", help="skip transcription (ear-grade only)")
    ap.add_argument("--play", action="store_true", help="play each clip back right after recording")
    ap.add_argument("--listen", nargs="?", const="__latest__", default=None,
                    metavar="FOLDER",
                    help="grade an existing run by EAR: loudness-match its clips and "
                         "play them back labeled (default: the latest calib_runs folder)")
    ap.add_argument("--listen-target", type=float, default=0.12,
                    help="target RMS for the loudness-matched playback (--listen)")
    ap.add_argument("--listen-loops", type=int, default=2,
                    help="how many times to play through the clips (--listen)")
    ap.add_argument("--list-presets", action="store_true")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="no mic: synthetic self-check")
    args = ap.parse_args(argv)

    # utf-8 stdout so a non-ASCII transcript never crashes the cp1252 console.
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    if args.list_presets:
        default_names = {p.name for p in DEFAULT_PRESETS}
        for p in ALL_PRESETS:
            mark = "*" if p.name in default_names else " "
            print(f"{mark} {p.name:<20} {p.blurb}")
            print(f"  {'':<20} overrides: {json.dumps(p.config_overrides()) or '{}'}")
        print("\n(* = in the default sweep; pick any subset with --presets)")
        return 0

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    if args.listen is not None:
        folder = args.listen if args.listen != "__latest__" else _latest_run_folder()
        if not folder or not os.path.isdir(folder):
            print("No run folder to listen to. Record one first, or pass --listen <folder>.")
            return 1
        print(f"Listening to: {os.path.abspath(folder)}")
        return listen_mode(folder, target_rms=args.listen_target, loops=args.listen_loops)

    presets = _select_presets(args.presets)

    folder = args.out or os.path.join("calib_runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    do_asr = not args.no_asr

    print(f"Calibration suite -> {os.path.abspath(folder)}")
    print(f"Presets: {', '.join(p.name for p in presets)}")
    print(f"Phrase : {args.phrase}\n")

    # Merged sherpa config: default devices + the model paths the APM/denoise
    # presets and the talk-over TTS reference need. Best-effort (the tool still
    # runs on a box with no config; DSP presets then fail open with a warning).
    sherpa_cfg: dict = {}
    try:
        from core.config import load_config
        sherpa_cfg = dict(load_config(args.config).get("sherpa", {}) or {})
    except Exception:  # noqa: BLE001
        sherpa_cfg = {}

    if args.selftest:
        results = _run_selftest(presets, folder, args.seconds, args.asr_model, do_asr,
                                sherpa_cfg=sherpa_cfg)
    else:
        device = args.input_device if args.input_device is not None else sherpa_cfg.get("input_device")

        talkover_wav = None
        if args.talk_over is not None:
            talkover_wav = _resolve_talkover_wav(args.talk_over, folder, sherpa_cfg)
            if talkover_wav is None:
                print("No talk-over reference clip available (synth failed and no "
                      "logs/live assistant clips); pass --talk-over <path-to.wav>.")
                return 1
            print(f"TALK-OVER mode: '{os.path.basename(talkover_wav)}' will play on "
                  "loop through the speakers while you record. Talk over it the whole "
                  "time -- this measures how well YOUR voice survives playback.\n")

        print("Read the SAME phrase each time, at your normal speaking distance and "
              "volume. A calibration preset first asks for a quiet moment.\n")
        results = []
        counter = 0
        for preset in presets:
            for rep in range(max(1, args.repeat)):
                counter += 1
                try:
                    r = record_preset(
                        preset, device=device, seconds=args.seconds,
                        sherpa_cfg=sherpa_cfg, talkover_wav=talkover_wav,
                        talkover_gain=args.talkover_gain,
                        output_device=args.output_device,
                        countdown=_countdown, quiet=_quiet,
                    )
                except Exception as exc:  # noqa: BLE001 - record the rest
                    print(f"  !! {preset.name} failed to record: {type(exc).__name__}: {exc}")
                    continue
                suffix = "" if args.repeat == 1 else f"_r{rep + 1}"
                r.wav_path = os.path.join(folder, f"{counter:02d}_{preset.name}{suffix}.wav")
                write_wav(r.wav_path, r.samples, 16000)
                cs = r.calibration
                extra = ""
                if cs is not None:
                    extra = (f"  [calib ambient={cs['ambient_rms']:.4f} "
                             f"clip={cs['clipping_fraction']*100:.1f}%]")
                print(f"    captured {r.samples.size/16000:.1f}s "
                      f"rms={r.metrics.get('rms')} clip%={r.metrics.get('clip_pct')}{extra}")
                if args.play:
                    print("    (playing back...)")
                    try:
                        _play_wav(r.wav_path)
                    except Exception as exc:  # noqa: BLE001
                        print(f"    playback failed: {exc}")
                if do_asr:
                    r.transcript, r.asr_confidence, err = _transcribe(r.wav_path, args.asr_model)
                    if err:
                        print(f"    [asr] {err}")
                        do_asr = False  # don't retry a broken whisper for every clip
                    else:
                        r.wer = word_error_rate(args.phrase, r.transcript)
                        print(f"    ASR: \"{r.transcript}\"  "
                              f"(WER {r.wer}, confidence {r.asr_confidence})")
                results.append(r)
                time.sleep(0.4)

    # Fill in WER for the selftest path / any clip transcribed but unscored.
    for r in results:
        if r.wer is None and r.transcript:
            r.wer = word_error_rate(args.phrase, r.transcript)

    if not results:
        print("\nNo clips recorded. Check --list-devices and --input-device.")
        return 1

    summary = write_summary(folder, results, args.phrase)
    grades = write_grades_sheet(folder, results, args.phrase)
    print_ranking(results)
    print(f"\nWrote: {summary}\n       {grades}")
    print(f"Play the WAVs in {os.path.abspath(folder)} and grade them in GRADES.md "
          "(or just tell me which sounds best).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
