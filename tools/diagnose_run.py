"""Post-hoc diagnostic analysis of a run bundle (.txt log + optional .ref.wav).

Extracts from a run bundle:
  - Spoken sentence timeline with playback-open latency
  - Per-sentence DTD frame summary while speaking was active
  - Barge-in events (DETECTED / REJECTED) with DTD context + speaking state
  - Self-interrupt / echo-only suspicion markers
  - Per-sentence playback-reference audio metrics (RMS, peak, clip, HF ratio)
    when --wav is supplied or auto-discovered from the log

Usage::

    python -m tools.diagnose_run logs/runs/run-20260627-121459.txt
    python -m tools.diagnose_run logs/runs/run-20260627-121459.txt --json
    python -m tools.diagnose_run logs/runs/run-20260627-121459.txt \\
        --wav logs/runs/run-20260627-121459.ref.wav
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

_TIME_PAT = re.compile(r"^(\d+):(\d+):(\d+)\.(\d+)")


def _parse_ts(s: str) -> float:
    """Parse HH:MM:SS.mmm → seconds since midnight."""
    m = _TIME_PAT.match(s)
    if not m:
        return 0.0
    h, mi, sec, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return h * 3600 + mi * 60 + sec + ms / 1000.0


def _fmt_offset(t: float, base: float) -> str:
    delta = t - base
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{abs(delta):.3f}s"


# ---------------------------------------------------------------------------
# Regex patterns for log lines
# ---------------------------------------------------------------------------

_LOG_PAT = re.compile(
    r"^(\d+:\d+:\d+\.\d+)\s+(\S+)\s+(\S+)\s+\|\s+(.*)"
)
_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

_SPEAKING_PAT = re.compile(r"speaking:\s+'(.+)'\s+\(queue depth=(\d+)\)")
_PLAYBACK_OPEN_PAT = re.compile(r"playback opened at (\d+) Hz")
# Anchored to end-of-message: the engine logs the bare string "barge-in
# detected" (core/engines/sherpa.py), so `$` matches it but NOT a confirm line
# whose embedded transcription happens to contain the substring, e.g.
# `barge-in confirmed by speech: 'barge-in detected'` (that ends in a quote).
# Without the anchor such a line would be miscounted as a detected event and
# skip the ADR-0011 confirm funnel.
_BARGE_DETECTED_PAT = re.compile(r"barge-in detected\s*$")
_BARGE_REJECTED_PAT = re.compile(r"barge-in REJECTED:\s+(.*)")
# ADR-0011 word-gated duck-then-confirm funnel. These lines carry no
# "detected"/"REJECTED" marker of their own, so they are counted directly to
# expose the acoustic-trigger -> speech-confirm funnel -- including runs where
# every trigger self-healed (ducked then expired) with zero hard-fires, which is
# the echo-pumping signature the word gate defuses.
_BARGE_DUCK_PAT = re.compile(r"barge-in: acoustic trigger")
_BARGE_CONFIRMED_PAT = re.compile(r"barge-in confirmed by speech")
_BARGE_UNCONFIRMED_PAT = re.compile(r"barge-in NOT confirmed")
# ADR-0013 continuous no-duck WORD-CUT funnel (OS echo-cancel path). The engine
# emits these only while barge_word_cut_enabled is live, so legacy bundles carry
# none and the section stays absent. The confirmed line MUST be matched before
# the generic ADR-0011 "barge-in confirmed by speech" counter or every word-cut
# would be miscounted into the duck-confirm funnel (which never ducked).
_WC_CONFIRMED_PAT = re.compile(r"barge-in confirmed by speech \(word-cut\)")
_WC_TRACE_PAT = re.compile(r"word-cut trace:\s+(\d+) word\(s\)\s+(.*)")
_WC_RESET_PAT = re.compile(r"word-cut burst reset: dropped (\d+) word\(s\)\s+(.*)")
_WC_NEAREND_PAT = re.compile(
    rf"word-cut near-end: rms_avg=({_NUM}) rms_peak=({_NUM}) vad_frac=({_NUM})"
    rf" floor=({_NUM}) blocks=(\d+)"
)
_WC_FUNNEL_PAT = re.compile(
    r"word-cut funnel: fed=(\d+) skipped_quiet=(\d+) resets=(\d+)"
    r" dropped_words=(\d+) max_words=(\d+) own_folds=(\d+) guard_suppressed=(\d+)"
    rf" decode_errors=(\d+) cuts=(\d+) nearend_rms_p50=({_NUM})"
    rf" nearend_rms_p95=({_NUM})"
)
_DTD_PAT = re.compile(
    rf"dtd:\s+D=({_NUM})\s+K=({_NUM})\s+fired=(True|False)\s+gated=(True|False)"
    rf"\s+\(z_raw=({_NUM})\s+z_resid=({_NUM})\s+z_coh=({_NUM})\)"
    rf"\s+raw=({_NUM})\s+resid=({_NUM})\s+incoh=({_NUM})"
    rf"\s+resid_floor=({_NUM})\s+consec=(\d+)"
    rf"(?:\s+coh=(\S+)\s+coh_veto=(True|False)\s+ref_delay=({_NUM})ms)?"
)
_HEARTBEAT_PAT = re.compile(
    r"capture heartbeat:\s+blocks=(\d+)\s+avg_rms=([\d.]+)\s+clip=([\d.]+)%"
    r"\s+underruns=(\d+)\s+partials=(\d+)\s+finals=(\d+)\s+speaking=(True|False)"
)
_REF_WAV_PAT = re.compile(r"recording playback reference.*->\s+(\S+\.ref\.wav)")
_RUN_START_PAT = re.compile(r"run (\S+) started")
_AEC_ACTIVE_PAT = re.compile(
    r"AEC ACTIVE .*backend=([^,\s]+),\s+ref_delay=([\d.]+)ms,\s+apm_always_on=(True|False)"
)
_TTS_SANITIZE_PAT = re.compile(r"tts sanitize:\s+(\{.*\})")
_TTS_RESOLVED_PAT = re.compile(r"tts resolved:\s+(\{.*\})")
_TTS_QUALITY_PAT = re.compile(r"tts audio quality:\s+(\{.*\})")


# ---------------------------------------------------------------------------
# Dataclasses for events
# ---------------------------------------------------------------------------

@dataclass
class DTDFrame:
    t: float
    D: float
    K: float
    fired: bool
    gated: bool
    z_raw: float
    z_resid: float
    z_coh: float
    raw: float
    resid: float
    incoh: float
    resid_floor: float
    consec: int
    coh_verdict: Optional[str] = None
    coh_veto: Optional[bool] = None
    ref_delay_ms: Optional[float] = None


@dataclass
class BargeEvent:
    t: float
    kind: str          # "detected" | "rejected"
    reason: str        # rejection reason if kind=="rejected"
    speaking_at_event: Optional[bool]   # heartbeat state nearest to this event
    dtd_context: list  # last ≤3 DTD frames before this event
    sentence_idx: Optional[int]         # which sentence was playing


@dataclass
class Sentence:
    idx: int
    t_speak: float      # timestamp of the speaking: line
    text: str
    queue_depth: int
    tts_sanitize: Optional[dict] = None
    tts_resolved: Optional[dict] = None
    tts_quality: Optional[dict] = None
    t_playback_open: Optional[float] = None
    playback_sample_rate: Optional[int] = None
    playback_open_latency_ms: Optional[float] = None
    t_end: Optional[float] = None       # next sentence start or barge-in or EOF
    dtd_frames: list = field(default_factory=list)
    barge_events: list = field(default_factory=list)


@dataclass
class Heartbeat:
    t: float
    blocks: int
    avg_rms: float
    clip: float
    speaking: bool
    underruns: int = 0    # cumulative total since run start
    partials: int = 0     # partial transcripts in this 2 s interval
    finals: int = 0       # final transcripts in this 2 s interval


@dataclass
class WordCutReply:
    """One "word-cut funnel:" line = one reply's ADR-0013 word-cut summary."""
    t: float
    fed: int
    skipped_quiet: int
    resets: int
    dropped_words: int
    max_words: int
    own_folds: int
    guard_suppressed: int
    decode_errors: int
    cuts: int
    nearend_rms_p50: float
    nearend_rms_p95: float
    sentence_idx: Optional[int] = None


@dataclass
class WordCutWindow:
    """One "word-cut near-end:" ~2s evidence window during playback."""
    t: float
    rms_avg: float
    rms_peak: float
    vad_frac: float
    floor: float
    blocks: int


@dataclass
class ParsedRun:
    run_id: str
    run_start_t: Optional[float]
    ref_wav_path: Optional[str]
    ref_wav_start_t: Optional[float]   # absolute timestamp when ref.wav recording began
    sentences: list
    barge_events: list
    heartbeats: list
    dtd_frames: list
    aec_backend: Optional[str] = None
    aec_config_ref_delay_ms: Optional[float] = None
    apm_always_on: Optional[bool] = None
    # ADR-0011 confirm-window funnel counts (log-derived): acoustic triggers that
    # ducked, of those how many a real talk-over confirmed (hard-cut) vs
    # self-healed (window expired with no words).
    barge_duck: int = 0
    barge_confirmed: int = 0
    barge_unconfirmed: int = 0
    # ADR-0013 word-cut funnel (log-derived; empty on runs without the path).
    word_cut_replies: list = field(default_factory=list)
    word_cut_windows: list = field(default_factory=list)
    word_cut_traces: list = field(default_factory=list)   # (t, words, text)
    word_cut_resets: list = field(default_factory=list)   # (t, words, text)
    word_cut_confirmed: int = 0


# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------

def parse_log(txt_path: str) -> ParsedRun:
    sentences: list[Sentence] = []
    barge_events: list[BargeEvent] = []
    heartbeats: list[Heartbeat] = []
    dtd_frames: list[DTDFrame] = []
    run_id = Path(txt_path).stem.replace("run-", "")
    run_start_t: Optional[float] = None
    ref_wav_path: Optional[str] = None
    ref_wav_start_t: Optional[float] = None
    aec_backend: Optional[str] = None
    aec_config_ref_delay_ms: Optional[float] = None
    apm_always_on: Optional[bool] = None
    barge_duck = barge_confirmed = barge_unconfirmed = 0
    word_cut_replies: list[WordCutReply] = []
    word_cut_windows: list[WordCutWindow] = []
    word_cut_traces: list[tuple] = []
    word_cut_resets: list[tuple] = []
    word_cut_confirmed = 0
    pending_tts_sanitize: Optional[dict] = None
    last_playback_sample_rate: Optional[int] = None

    with open(txt_path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip()
            m = _LOG_PAT.match(line)
            if not m:
                continue
            ts_str, level, logger, msg = m.group(1), m.group(2), m.group(3), m.group(4)
            t = _parse_ts(ts_str)

            # Run start
            if run_start_t is None:
                ms = _RUN_START_PAT.search(msg)
                if ms:
                    run_start_t = t
                    run_id = ms.group(1)

            # Ref WAV path (and its recording start time)
            mw = _REF_WAV_PAT.search(msg)
            if mw:
                ref_wav_path = mw.group(1)
                if ref_wav_start_t is None:
                    ref_wav_start_t = t

            maec = _AEC_ACTIVE_PAT.search(msg)
            if maec:
                aec_backend = maec.group(1)
                aec_config_ref_delay_ms = float(maec.group(2))
                apm_always_on = maec.group(3) == "True"

            mts = _TTS_SANITIZE_PAT.search(msg)
            if mts and "speaker.sherpa" in logger:
                try:
                    pending_tts_sanitize = json.loads(mts.group(1))
                except json.JSONDecodeError:
                    pending_tts_sanitize = None
                continue

            mtr = _TTS_RESOLVED_PAT.search(msg)
            if mtr and "speaker.sherpa" in logger and sentences:
                try:
                    sentences[-1].tts_resolved = json.loads(mtr.group(1))
                except json.JSONDecodeError:
                    sentences[-1].tts_resolved = None
                continue

            mtq = _TTS_QUALITY_PAT.search(msg)
            if mtq and "speaker.sherpa" in logger and sentences:
                try:
                    sentences[-1].tts_quality = json.loads(mtq.group(1))
                except json.JSONDecodeError:
                    sentences[-1].tts_quality = None
                continue

            # Heartbeat
            mhb = _HEARTBEAT_PAT.search(msg)
            if mhb:
                heartbeats.append(Heartbeat(
                    t=t,
                    blocks=int(mhb.group(1)),
                    avg_rms=float(mhb.group(2)),
                    clip=float(mhb.group(3)),
                    speaking=mhb.group(7) == "True",
                    underruns=int(mhb.group(4)),
                    partials=int(mhb.group(5)),
                    finals=int(mhb.group(6)),
                ))
                continue

            # DTD frame
            md = _DTD_PAT.search(msg)
            if md:
                frame = DTDFrame(
                    t=t,
                    D=float(md.group(1)), K=float(md.group(2)),
                    fired=md.group(3) == "True", gated=md.group(4) == "True",
                    z_raw=float(md.group(5)), z_resid=float(md.group(6)), z_coh=float(md.group(7)),
                    raw=float(md.group(8)), resid=float(md.group(9)),
                    incoh=float(md.group(10)), resid_floor=float(md.group(11)),
                    consec=int(md.group(12)),
                    coh_verdict=md.group(13),
                    coh_veto=(md.group(14) == "True") if md.group(14) else None,
                    ref_delay_ms=float(md.group(15)) if md.group(15) else None,
                )
                dtd_frames.append(frame)
                if sentences:
                    sentences[-1].dtd_frames.append(frame)
                continue

            # Speaking line
            ms2 = _SPEAKING_PAT.search(msg)
            if ms2 and "speaker.sherpa" in logger:
                # Close previous sentence at this time
                if sentences:
                    sentences[-1].t_end = t
                sent = Sentence(
                    idx=len(sentences),
                    t_speak=t,
                    text=ms2.group(1),
                    queue_depth=int(ms2.group(2)),
                    tts_sanitize=pending_tts_sanitize,
                    playback_sample_rate=last_playback_sample_rate,
                )
                pending_tts_sanitize = None
                sentences.append(sent)
                continue

            # Playback open
            mp = _PLAYBACK_OPEN_PAT.search(msg)
            if mp and "speaker.sherpa" in logger and sentences:
                last_playback_sample_rate = int(mp.group(1))
                s = sentences[-1]
                if s.t_playback_open is None:
                    s.t_playback_open = t
                    s.playback_sample_rate = last_playback_sample_rate
                    s.playback_open_latency_ms = round((t - s.t_speak) * 1000, 1)
                continue

            # Barge-in detected
            if _BARGE_DETECTED_PAT.search(msg) and "speaker.sherpa" in logger:
                speaking_now = _nearest_speaking_state(heartbeats, t)
                ctx = [f for f in dtd_frames if 0 <= t - f.t <= 0.5][-3:]
                sent_idx = len(sentences) - 1 if sentences else None
                be = BargeEvent(
                    t=t, kind="detected", reason="",
                    speaking_at_event=speaking_now,
                    dtd_context=ctx,
                    sentence_idx=sent_idx,
                )
                barge_events.append(be)
                if sentences:
                    sentences[-1].barge_events.append(be)
                    sentences[-1].t_end = t  # sentence ended here
                continue

            # Barge-in rejected
            mbr = _BARGE_REJECTED_PAT.search(msg)
            if mbr and "speaker.sherpa" in logger:
                speaking_now = _nearest_speaking_state(heartbeats, t)
                ctx = [f for f in dtd_frames if 0 <= t - f.t <= 0.5][-3:]
                sent_idx = len(sentences) - 1 if sentences else None
                be = BargeEvent(
                    t=t, kind="rejected", reason=mbr.group(1),
                    speaking_at_event=speaking_now,
                    dtd_context=ctx,
                    sentence_idx=sent_idx,
                )
                barge_events.append(be)
                if sentences:
                    sentences[-1].barge_events.append(be)
                continue

            # ADR-0013 word-cut funnel lines. Matched BEFORE the ADR-0011
            # counters: the word-cut confirmed line contains the generic
            # "barge-in confirmed by speech" substring and would otherwise be
            # miscounted into the duck-confirm funnel (which never ducked).
            if "speaker.sherpa" in logger:
                if _WC_CONFIRMED_PAT.search(msg):
                    word_cut_confirmed += 1
                    continue
                mwf = _WC_FUNNEL_PAT.search(msg)
                if mwf:
                    word_cut_replies.append(WordCutReply(
                        t=t,
                        fed=int(mwf.group(1)),
                        skipped_quiet=int(mwf.group(2)),
                        resets=int(mwf.group(3)),
                        dropped_words=int(mwf.group(4)),
                        max_words=int(mwf.group(5)),
                        own_folds=int(mwf.group(6)),
                        guard_suppressed=int(mwf.group(7)),
                        decode_errors=int(mwf.group(8)),
                        cuts=int(mwf.group(9)),
                        nearend_rms_p50=float(mwf.group(10)),
                        nearend_rms_p95=float(mwf.group(11)),
                        sentence_idx=len(sentences) - 1 if sentences else None,
                    ))
                    continue
                mwn = _WC_NEAREND_PAT.search(msg)
                if mwn:
                    word_cut_windows.append(WordCutWindow(
                        t=t,
                        rms_avg=float(mwn.group(1)),
                        rms_peak=float(mwn.group(2)),
                        vad_frac=float(mwn.group(3)),
                        floor=float(mwn.group(4)),
                        blocks=int(mwn.group(5)),
                    ))
                    continue
                mwt = _WC_TRACE_PAT.search(msg)
                if mwt:
                    word_cut_traces.append((t, int(mwt.group(1)), mwt.group(2)))
                    continue
                mwr = _WC_RESET_PAT.search(msg)
                if mwr:
                    word_cut_resets.append((t, int(mwr.group(1)), mwr.group(2)))
                    continue

            # Barge-in confirm-window funnel (ADR-0011). These lines carry no
            # detected/REJECTED marker, so they fall through to here; count them
            # directly. A confirmed window ALSO logs "barge-in detected" (handled
            # above as a BargeEvent) so hard-fires stay in both views.
            if "speaker.sherpa" in logger:
                if _BARGE_DUCK_PAT.search(msg):
                    barge_duck += 1
                    continue
                if _BARGE_CONFIRMED_PAT.search(msg):
                    barge_confirmed += 1
                    continue
                if _BARGE_UNCONFIRMED_PAT.search(msg):
                    barge_unconfirmed += 1
                    continue

    return ParsedRun(
        run_id=run_id,
        run_start_t=run_start_t,
        ref_wav_path=ref_wav_path,
        ref_wav_start_t=ref_wav_start_t,
        sentences=sentences,
        barge_events=barge_events,
        heartbeats=heartbeats,
        dtd_frames=dtd_frames,
        aec_backend=aec_backend,
        aec_config_ref_delay_ms=aec_config_ref_delay_ms,
        apm_always_on=apm_always_on,
        barge_duck=barge_duck,
        barge_confirmed=barge_confirmed,
        barge_unconfirmed=barge_unconfirmed,
        word_cut_replies=word_cut_replies,
        word_cut_windows=word_cut_windows,
        word_cut_traces=word_cut_traces,
        word_cut_resets=word_cut_resets,
        word_cut_confirmed=word_cut_confirmed,
    )


def _nearest_speaking_state(heartbeats: list[Heartbeat], t: float) -> Optional[bool]:
    """Return speaking state of the heartbeat closest in time before or at t."""
    best: Optional[Heartbeat] = None
    for hb in heartbeats:
        if hb.t <= t + 0.1:  # allow 100ms slack for async ordering
            best = hb
        else:
            break
    return best.speaking if best else None


# ---------------------------------------------------------------------------
# WAV analysis
# ---------------------------------------------------------------------------

def _analyze_wav_segment(
    samples: "np.ndarray",
    sample_rate: int,
    hf_cutoff_hz: int = 4000,
) -> dict:
    """Compute RMS, peak, clip fraction, HF ratio for a float32 PCM array."""
    import numpy as np

    if samples.size == 0:
        return {
            "rms": None,
            "active_rms": None,
            "peak": None,
            "clip_pct": None,
            "hf_ratio": None,
            "duration_s": 0.0,
            "first_audio_offset_s": None,
        }

    rms = float(np.sqrt(np.mean(samples ** 2)))
    peak = float(np.max(np.abs(samples)))
    clip_pct = float(100.0 * np.mean(np.abs(samples) >= 0.99))
    duration_s = round(len(samples) / sample_rate, 3)
    first_audio_offset = _first_audio_offset(samples, sample_rate)
    active_rms = None
    if first_audio_offset is not None:
        active = samples[int(first_audio_offset * sample_rate):]
        if active.size:
            active_rms = float(np.sqrt(np.mean(active ** 2)))

    # HF ratio via FFT
    n = len(samples)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mag = np.abs(np.fft.rfft(samples))
    total_power = float(np.sum(mag ** 2)) or 1e-12
    hf_mask = freqs >= hf_cutoff_hz
    hf_power = float(np.sum(mag[hf_mask] ** 2))
    hf_ratio = round(hf_power / total_power, 4)

    return {
        "rms": round(rms, 4),
        "active_rms": round(active_rms, 4) if active_rms is not None else None,
        "peak": round(peak, 4),
        "clip_pct": round(clip_pct, 2),
        "hf_ratio": hf_ratio,
        "duration_s": duration_s,
        "first_audio_offset_s": (
            round(first_audio_offset, 3)
            if first_audio_offset is not None else None
        ),
    }


def _first_audio_offset(
    samples: "np.ndarray",
    sample_rate: int,
    *,
    frame_ms: float = 20.0,
    min_rms: float = 0.0015,
) -> Optional[float]:
    """Return the first frame offset whose RMS looks like real reference audio."""
    import numpy as np

    if samples.size == 0:
        return None
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    peak = float(np.max(np.abs(samples)))
    if peak < min_rms:
        return None
    threshold = max(min_rms, 0.03 * peak)
    for i in range(0, max(1, samples.size - frame + 1), frame):
        block = samples[i:i + frame]
        if block.size == 0:
            continue
        block_rms = float(np.sqrt(np.mean(block ** 2)))
        if block_rms >= threshold:
            return i / sample_rate
    return None


def _read_wav_mono(wav_path: str) -> tuple["np.ndarray", int]:
    """Read a PCM WAV as mono float32 in [-1, 1]."""
    import numpy as np

    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        channels = w.getnchannels()
        sample_width = w.getsampwidth()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)

    if sample_width == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        pcm = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")

    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.float32)
    return pcm, sr


def _estimate_ref_delay(
    ref: "np.ndarray",
    mic: "np.ndarray",
    sample_rate: int,
    *,
    max_delay_ms: float = 400.0,
    min_ref_rms: float = 0.002,
) -> Optional[dict]:
    """Estimate mic lag behind the playback reference with normalized correlation."""
    import numpy as np

    n = min(len(ref), len(mic))
    if n <= max(32, int(0.08 * sample_rate)):
        return None
    ref = np.asarray(ref[:n], dtype=np.float32)
    mic = np.asarray(mic[:n], dtype=np.float32)

    onset = _first_audio_offset(ref, sample_rate)
    if onset is not None:
        start = max(0, int((onset - 0.02) * sample_rate))
        ref = ref[start:]
        mic = mic[start:]
        n = min(len(ref), len(mic))
        ref = ref[:n]
        mic = mic[:n]

    max_samples = int(6.0 * sample_rate)
    if len(ref) > max_samples:
        ref = ref[:max_samples]
        mic = mic[:max_samples]
    if len(ref) <= max(32, int(0.08 * sample_rate)):
        return None

    ref = ref - float(np.mean(ref))
    mic = mic - float(np.mean(mic))
    ref_rms = float(np.sqrt(np.mean(ref ** 2)))
    mic_rms = float(np.sqrt(np.mean(mic ** 2)))
    if ref_rms < min_ref_rms or mic_rms <= 1e-6:
        return None

    step = max(1, int(sample_rate / 4000))
    ref_d = ref[::step]
    mic_d = mic[::step]
    sr_d = sample_rate / step
    max_lag = min(int(max_delay_ms * sr_d / 1000.0), len(ref_d) - 32)
    if max_lag <= 0:
        return None

    best_lag = 0
    best_corr = -1.0
    for lag in range(max_lag + 1):
        r = ref_d[: len(ref_d) - lag]
        m = mic_d[lag: lag + len(r)]
        if len(r) < 32:
            break
        denom = float(np.linalg.norm(r) * np.linalg.norm(m))
        if denom <= 1e-12:
            continue
        corr = float(np.dot(r, m) / denom)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    return {
        "estimated_delay_ms": round(1000.0 * best_lag / sr_d, 1),
        "delay_correlation": round(best_corr, 3),
    }


def analyze_ref_wav(
    wav_path: str,
    run: ParsedRun,
    *,
    hf_cutoff_hz: int = 4000,
) -> dict:
    """Load the .ref.wav and compute per-sentence audio metrics.

    Returns a dict keyed by sentence index. Each value is the output of
    _analyze_wav_segment plus a 'window_s' tuple.
    """
    try:
        pcm, sr = _read_wav_mono(wav_path)
    except Exception as exc:
        return {"_error": str(exc)}

    # The ref.wav records PLAYBACK output starting from the moment the engine
    # opened the reference recorder (logged as "recording playback reference").
    # Use ref_wav_start_t as the absolute time base; fall back to the first
    # playback-open event if that wasn't recorded.
    wav_base = run.ref_wav_start_t
    if wav_base is None:
        for s in run.sentences:
            if s.t_playback_open is not None:
                wav_base = s.t_playback_open
                break
    if wav_base is None:
        return {"_error": "cannot determine ref.wav start time"}

    results: dict = {}
    results["_sample_rate"] = sr
    results["_duration_s"] = round(len(pcm) / sr, 3)
    for s in run.sentences:
        pb_start = s.t_playback_open if s.t_playback_open is not None else s.t_speak
        t_start = pb_start - wav_base
        t_end = (s.t_end - wav_base) if s.t_end is not None else (t_start + 10.0)

        i0 = max(0, int(t_start * sr))
        i1 = min(len(pcm), int(t_end * sr))
        segment = pcm[i0:i1]
        metrics = _analyze_wav_segment(segment, sr, hf_cutoff_hz=hf_cutoff_hz)
        metrics["window_s"] = (round(t_start, 3), round(t_end, 3))
        metrics["window_log_start_t"] = pb_start
        if metrics.get("first_audio_offset_s") is not None:
            metrics["first_audio_log_t"] = round(
                pb_start + float(metrics["first_audio_offset_s"]),
                3,
            )
        else:
            metrics["first_audio_log_t"] = None
        results[s.idx] = metrics

    return results


def analyze_mic_ref_wav(
    mic_wav_path: str,
    ref_wav_path: str,
    run: ParsedRun,
    *,
    max_delay_ms: float = 400.0,
) -> dict:
    """Estimate acoustic delay per sentence from frame-aligned mic/ref WAVs."""
    import numpy as np

    try:
        mic, mic_sr = _read_wav_mono(mic_wav_path)
        ref, ref_sr = _read_wav_mono(ref_wav_path)
    except Exception as exc:
        return {"_error": str(exc)}

    if mic_sr != ref_sr:
        # Keep the dependency surface small: linear resampling is good enough for
        # a diagnostic correlation estimate.
        x_old = np.linspace(0.0, len(mic) / mic_sr, num=len(mic), endpoint=False)
        x_new = np.linspace(
            0.0,
            len(mic) / mic_sr,
            num=int(len(mic) * ref_sr / mic_sr),
            endpoint=False,
        )
        mic = np.interp(x_new, x_old, mic).astype(np.float32)
        mic_sr = ref_sr

    wav_base = run.ref_wav_start_t
    if wav_base is None:
        return {"_error": "cannot determine WAV start time"}

    results: dict = {}
    for s in run.sentences:
        pb_start = s.t_playback_open if s.t_playback_open is not None else s.t_speak
        t_start = pb_start - wav_base
        t_end = (s.t_end - wav_base) if s.t_end is not None else (t_start + 10.0)
        i0 = max(0, int(t_start * ref_sr))
        i1 = min(len(ref), len(mic), int(t_end * ref_sr))
        if i1 <= i0:
            results[s.idx] = {"estimated_delay_ms": None, "delay_correlation": None}
            continue
        estimate = _estimate_ref_delay(
            ref[i0:i1],
            mic[i0:i1],
            ref_sr,
            max_delay_ms=max_delay_ms,
        )
        results[s.idx] = estimate or {
            "estimated_delay_ms": None,
            "delay_correlation": None,
        }
    return results


# ---------------------------------------------------------------------------
# Self-interrupt / echo-only suspicion
# ---------------------------------------------------------------------------

def classify_barge_event(be: BargeEvent) -> str:
    """Return a suspicion label for a barge-in event."""
    if be.kind != "detected":
        return "ok"
    if be.speaking_at_event is False:
        return "ok"  # not speaking → real user barge-in, likely legitimate
    if be.speaking_at_event is None:
        return "uncertain"

    # speaking=True at barge-in → self-interrupt candidate
    # Assess DTD context: high incoh = mic signal coherent with reference = echo
    gated_frames = [f for f in be.dtd_context if f.gated]
    if not gated_frames:
        return "suspect:no-dtd"
    avg_incoh = sum(f.incoh for f in gated_frames) / len(gated_frames)
    if avg_incoh >= 0.70:
        return "suspect:echo-high-incoh"
    return "suspect:speaking"


def word_cut_funnel(run: ParsedRun) -> dict:
    """Aggregate the ADR-0013 word-cut funnel across the run and derive the
    live-failure verdict signals. Every zero is diagnostic (run-20260706-231226
    had NO word-cut telemetry and was undiagnosable):

    - ``starved_replies``  (fed=0): the VAD gate never fed the recognizer -- the
      state-machine defect class, nothing acoustic.
    - ``voice_present_zero_words``: near-end windows with energy well above the
      calibrated floor while the run transcribed no words -- energy survived the
      canceller but the recognizer/VAD produced nothing from it.
    - ``max rms_avg ~ floor`` (voiced_windows=0 with windows present): the
      near-end never survived capture -- the canceller suppressed the user.
    - ``dropped_words``: burst resets wiped accumulated talk-over words.
    """
    replies = run.word_cut_replies
    if not (replies or run.word_cut_windows or run.word_cut_confirmed):
        return {"present": False}

    def _tot(key: str) -> int:
        return sum(getattr(r, key) for r in replies)

    voiced = [
        w for w in run.word_cut_windows
        if w.floor > 0 and w.rms_avg >= 2.0 * w.floor
    ]
    max_words = max((r.max_words for r in replies), default=0)
    if run.word_cut_traces:
        max_words = max(max_words, max(t[1] for t in run.word_cut_traces))
    return {
        "present": True,
        "replies": len(replies),
        "fed": _tot("fed"),
        "skipped_quiet": _tot("skipped_quiet"),
        "resets": _tot("resets"),
        "dropped_words": _tot("dropped_words"),
        "max_words": max_words,
        "own_folds": _tot("own_folds"),
        "guard_suppressed": _tot("guard_suppressed"),
        "decode_errors": _tot("decode_errors"),
        "cuts": _tot("cuts"),
        "confirmed_lines": run.word_cut_confirmed,
        "windows": len(run.word_cut_windows),
        "voiced_windows": len(voiced),
        "starved_replies": sum(1 for r in replies if r.fed == 0),
        "voice_present_zero_words": len(voiced) if max_words == 0 else 0,
    }


def self_interrupt_summary(run: ParsedRun) -> dict:
    """High-level self-interrupt verdict for the run."""
    total_detected = sum(1 for be in run.barge_events if be.kind == "detected")
    suspects = []
    for be in run.barge_events:
        label = classify_barge_event(be)
        if label.startswith("suspect"):
            suspects.append({"t": be.t, "label": label, "sentence": be.sentence_idx})

    rejected_while_speaking = [
        be for be in run.barge_events
        if be.kind == "rejected" and be.speaking_at_event is True
    ]

    verdict = "clean"
    if suspects:
        ratio = len(suspects) / max(total_detected, 1)
        if ratio >= 0.5:
            verdict = "self-interrupt-likely"
        else:
            verdict = "self-interrupt-possible"

    return {
        "verdict": verdict,
        "detected_total": total_detected,
        "suspect_count": len(suspects),
        "suspects": suspects,
        "rejected_while_speaking": len(rejected_while_speaking),
    }


def _ref_quality_label(wm: Optional[dict]) -> str:
    if not wm or wm.get("duration_s", 0.0) <= 0.0:
        return "unknown"
    peak = wm.get("peak")
    active_rms = wm.get("active_rms")
    rms = wm.get("rms")
    if peak is None or peak < 0.005 or wm.get("first_audio_offset_s") is None:
        return "no-ref"
    if (
        (active_rms is not None and active_rms < 0.01)
        or (rms is not None and rms < 0.003)
    ):
        return "weak-ref"
    return "ok"


def _ref_delay_summary(frames: list[DTDFrame]) -> str:
    delays = [f.ref_delay_ms for f in frames if f.ref_delay_ms is not None]
    if not delays:
        return ""
    lo = min(delays)
    hi = max(delays)
    if abs(lo - hi) < 0.5:
        return f" ref_delay={lo:.0f}ms"
    avg = sum(delays) / len(delays)
    return f" ref_delay={avg:.0f}ms range={lo:.0f}-{hi:.0f}ms"


def _barge_phase(be: BargeEvent, run: ParsedRun, wav_metrics: Optional[dict]) -> str:
    if be.sentence_idx is None or be.sentence_idx >= len(run.sentences):
        return "unknown"
    sent = run.sentences[be.sentence_idx]
    wm = wav_metrics.get(sent.idx) if wav_metrics else None
    if wm and wm.get("first_audio_log_t") is not None:
        if be.t < float(wm["first_audio_log_t"]) - 0.02:
            return "pre-first-ref-audio"
        return "after-first-ref-audio"
    if sent.t_playback_open is not None and be.t < sent.t_playback_open:
        return "pre-playback-open"
    if wm and _ref_quality_label(wm) in ("no-ref", "weak-ref"):
        return _ref_quality_label(wm)
    return "unknown"


#: Thresholds for the "tts audio quality" findings below. Calibrated against
#: tests/test_output_leveler.py's audio_quality_metrics tests: a pure tone
#: reads spectral_flatness < 0.01, white noise > 0.3 -- 0.35 leaves margin on
#: both sides. clip/DC thresholds are deliberately tighter than the .ref.wav
#: checks elsewhere in this file because this metric is computed on the EXACT
#: final synthesized samples (not a lossy 16 kHz proxy), so a real digital-
#: domain defect is fully visible here.
_TTS_QUALITY_FLATNESS_WARN = 0.35   # >= this reads noise-like, not tonal speech
_TTS_QUALITY_CLIP_WARN_PCT = 1.0    # % of samples at/near full scale
_TTS_QUALITY_DC_WARN = 0.02         # DC offset, full scale = 1.0


def diagnostic_findings(
    run: ParsedRun,
    wav_metrics: Optional[dict] = None,
    delay_metrics: Optional[dict] = None,
) -> list[str]:
    """Human-focused anomalies that should be obvious in a run review."""
    findings: list[str] = []

    for s in run.sentences:
        tq = s.tts_quality
        if not tq:
            continue
        flatness = tq.get("spectral_flatness")
        if flatness is not None and flatness >= _TTS_QUALITY_FLATNESS_WARN:
            findings.append(
                f"sentence[{s.idx}] synthesized audio reads noise-like "
                f"(spectral_flatness={flatness}, mode={tq.get('mode')}) -- "
                "a DIGITAL-domain defect, not just acoustic/speaker"
            )
        clip_pct = tq.get("clip_pct")
        if clip_pct is not None and clip_pct >= _TTS_QUALITY_CLIP_WARN_PCT:
            findings.append(
                f"sentence[{s.idx}] synthesized audio is clipping "
                f"(clip_pct={clip_pct}%, mode={tq.get('mode')})"
            )
        dc = tq.get("dc_offset")
        if dc is not None and abs(dc) >= _TTS_QUALITY_DC_WARN:
            findings.append(
                f"sentence[{s.idx}] synthesized audio has a DC offset "
                f"(dc_offset={dc}, mode={tq.get('mode')})"
            )

    for be in run.barge_events:
        phase = _barge_phase(be, run, wav_metrics)
        if phase in ("pre-first-ref-audio", "pre-playback-open"):
            sent = run.sentences[be.sentence_idx] if be.sentence_idx is not None else None
            if sent is not None:
                rel = be.t - sent.t_speak
                wm = wav_metrics.get(sent.idx) if wav_metrics else None
                onset = wm.get("first_audio_offset_s") if wm else None
                if onset is not None:
                    findings.append(
                        f"sentence[{sent.idx}] {be.kind} barge at +{rel:.3f}s before "
                        f"first playback reference audio (+{float(onset):.3f}s)"
                    )
                else:
                    findings.append(
                        f"sentence[{sent.idx}] {be.kind} barge at +{rel:.3f}s before playback opened"
                    )

    if wav_metrics:
        ref_sr = wav_metrics.get("_sample_rate")
        for s in run.sentences:
            if (
                ref_sr is not None
                and s.playback_sample_rate is not None
                and int(ref_sr) != int(s.playback_sample_rate)
            ):
                findings.append(
                    f"sentence[{s.idx}] playback reference is {int(ref_sr)} Hz "
                    f"frame-aligned diagnostic audio, not bit-exact "
                    f"{int(s.playback_sample_rate)} Hz PortAudio output"
                )
        for s in run.sentences:
            wm = wav_metrics.get(s.idx)
            quality = _ref_quality_label(wm)
            if quality == "no-ref":
                findings.append(f"sentence[{s.idx}] playback reference has no detected audio")
            elif quality == "weak-ref":
                findings.append(
                    f"sentence[{s.idx}] playback reference is weak "
                    f"(rms={wm.get('rms')} active_rms={wm.get('active_rms')} peak={wm.get('peak')})"
                )

    if delay_metrics and run.aec_config_ref_delay_ms is not None:
        for s in run.sentences:
            dm = delay_metrics.get(s.idx)
            if not dm:
                continue
            est = dm.get("estimated_delay_ms")
            corr = dm.get("delay_correlation")
            if est is None or corr is None:
                continue
            if (
                corr >= 0.15
                and abs(float(est) - run.aec_config_ref_delay_ms) >= 50.0
            ):
                findings.append(
                    f"sentence[{s.idx}] AEC delay mismatch: config={run.aec_config_ref_delay_ms:.0f}ms "
                    f"estimated={float(est):.0f}ms corr={float(corr):.2f}"
                )

    if not findings:
        findings.append(
            "no first-audio/ref-energy/AEC-delay anomalies detected from available data"
        )
    return findings


# ---------------------------------------------------------------------------
# Pass/fail verdict for open-speaker A/B
# ---------------------------------------------------------------------------

#: Thresholds for the live A/B pass/fail verdict.
_FIRST_AUDIO_WARN_MS = 300.0   # warn if avg > 300 ms
_FIRST_AUDIO_FAIL_MS = 600.0   # fail if avg > 600 ms
_UNDERRUN_WARN = 3             # warn if total underruns > this
_UNDERRUN_FAIL = 10            # fail if total underruns > this


def pass_fail_verdict(run: ParsedRun) -> dict:
    """Structured PASS/WARN/FAIL verdict for open-speaker A/B validation.

    Criteria (all headless-measurable from a run log):

    * **self_interrupt** — FAIL if any barge-in suspect during speaking.
    * **underruns** — WARN/FAIL based on cumulative playback underruns from the
      last heartbeat (the engine's own metric; reflects TTS synth falling behind
      PortAudio under CPU load).
    * **first_audio** — WARN if avg first-audio latency > 300 ms; FAIL > 600 ms.
    * **missed_barges** — informational count of "barge-in REJECTED while
      speaking" events (not a FAIL criterion alone — the rate depends on whether
      the user actually tried to interrupt).
    * **pre_first_audio_noise** — count of barge events that arrived BEFORE
      ``playback opened`` within a sentence (the window the ``_barge_watch_active``
      gate now suppresses; should be 0 with the gate on).

    Overall:
    * PASS  — all criteria PASS.
    * WARN  — no FAIL criterion but at least one WARN.
    * FAIL  — any FAIL criterion (self-interrupt suspect or hard thresholds).
    """
    si = self_interrupt_summary(run)

    # Self-interrupt
    si_fail = si["suspect_count"] > 0
    si_status = "FAIL" if si_fail else "PASS"

    # Underruns (cumulative from last heartbeat)
    total_underruns = run.heartbeats[-1].underruns if run.heartbeats else 0
    if total_underruns > _UNDERRUN_FAIL:
        underrun_status = "FAIL"
    elif total_underruns > _UNDERRUN_WARN:
        underrun_status = "WARN"
    else:
        underrun_status = "PASS"

    # First-audio latency
    latencies = [
        s.playback_open_latency_ms
        for s in run.sentences
        if s.playback_open_latency_ms is not None
    ]
    avg_latency_ms = round(sum(latencies) / len(latencies), 1) if latencies else None
    if avg_latency_ms is None:
        first_audio_status = "UNKNOWN"
    elif avg_latency_ms > _FIRST_AUDIO_FAIL_MS:
        first_audio_status = "FAIL"
    elif avg_latency_ms > _FIRST_AUDIO_WARN_MS:
        first_audio_status = "WARN"
    else:
        first_audio_status = "PASS"

    # Pre-first-audio barge noise (events before playback opened within a sentence)
    pre_first_audio_noise = 0
    for s in run.sentences:
        if s.t_playback_open is None:
            continue
        for be in s.barge_events:
            if be.t < s.t_playback_open:
                pre_first_audio_noise += 1

    # Overall verdict
    fail_criteria = [c for c in (si_status, underrun_status, first_audio_status) if c == "FAIL"]
    warn_criteria = [c for c in (si_status, underrun_status, first_audio_status) if c == "WARN"]
    if fail_criteria:
        overall = "FAIL"
    elif warn_criteria:
        overall = "WARN"
    else:
        overall = "PASS"

    return {
        "overall": overall,
        "self_interrupt": si_status,
        "self_interrupt_suspects": si["suspect_count"],
        "rejected_while_speaking": si["rejected_while_speaking"],
        "underruns_total": total_underruns,
        "underrun_verdict": underrun_status,
        "first_audio_avg_ms": avg_latency_ms,
        "first_audio_count": len(latencies),
        "first_audio_verdict": first_audio_status,
        "pre_first_audio_noise": pre_first_audio_noise,
    }


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def _dtd_summary(frames: list[DTDFrame]) -> str:
    if not frames:
        return "no DTD frames"
    fired = sum(1 for f in frames if f.fired)
    gated = sum(1 for f in frames if f.gated)
    avg_incoh = sum(f.incoh for f in frames) / len(frames)
    return (
        f"{len(frames)} frames: fired={fired}/{len(frames)} "
        f"gated={gated}/{len(frames)} avg_incoh={avg_incoh:.2f}"
        f"{_ref_delay_summary(frames)}"
    )


def _dtd_context_str(ctx: list[DTDFrame]) -> str:
    if not ctx:
        return "  (no DTD context)"
    lines = []
    for f in ctx:
        extra = ""
        if f.coh_verdict is not None or f.ref_delay_ms is not None:
            extra = f" coh={f.coh_verdict} coh_veto={f.coh_veto} ref_delay={f.ref_delay_ms}ms"
        lines.append(
            f"  dtd D={f.D:.1f} gated={f.gated} incoh={f.incoh:.2f} "
            f"raw={f.raw:.4f} resid={f.resid:.4f}{extra}"
        )
    return "\n".join(lines)


def format_report(run: ParsedRun, wav_metrics: Optional[dict] = None, delay_metrics: Optional[dict] = None) -> str:
    base = run.run_start_t or 0.0
    lines: list[str] = []
    lines.append(f"=== Run Diagnostics: {run.run_id} ===\n")
    if run.aec_backend or run.aec_config_ref_delay_ms is not None:
        lines.append(
            "AEC: "
            f"backend={run.aec_backend or 'unknown'} "
            f"configured_ref_delay={run.aec_config_ref_delay_ms if run.aec_config_ref_delay_ms is not None else 'unknown'}ms "
            f"apm_always_on={run.apm_always_on}"
        )

    lines.append("--- Findings ---")
    for finding in diagnostic_findings(run, wav_metrics, delay_metrics):
        lines.append(f"  - {finding}")
    lines.append("")

    # --- Sentence Timeline ---
    lines.append("--- Sentence Timeline ---")
    for s in run.sentences:
        t_off = _fmt_offset(s.t_speak, base) if base else f"t={s.t_speak:.3f}"
        lines.append(f"\n[{s.idx}] {t_off}  \"{s.text[:80]}\"")
        if s.tts_sanitize:
            lines.append(
                "     tts sanitize: "
                f"status={s.tts_sanitize.get('status')} "
                f"spoken={str(s.tts_sanitize.get('spoken', ''))[:80]!r}"
            )
        if s.tts_resolved:
            lines.append(
                "     tts resolved: "
                f"sid={s.tts_resolved.get('sid')} "
                f"speed={s.tts_resolved.get('speed')} "
                f"lowpass={s.tts_resolved.get('lowpass_hz')} "
                f"streaming_candidate={s.tts_resolved.get('streaming_candidate')}"
            )
        if s.tts_quality:
            tq = s.tts_quality
            noise_flag = (
                "  [NOISE-LIKE]"
                if (tq.get("spectral_flatness") or 0) >= _TTS_QUALITY_FLATNESS_WARN
                else ""
            )
            lines.append(
                "     tts audio quality: "
                f"mode={tq.get('mode')} rms={tq.get('rms')} peak={tq.get('peak')} "
                f"clip={tq.get('clip_pct')}% dc={tq.get('dc_offset')} "
                f"hf_ratio={tq.get('hf_ratio')} flatness={tq.get('spectral_flatness')}"
                f"{noise_flag}"
            )
        if s.t_playback_open is not None:
            sr_note = (
                f" at {s.playback_sample_rate} Hz"
                if s.playback_sample_rate is not None else ""
            )
            lines.append(
                f"     playback-open: +{s.playback_open_latency_ms:.0f}ms "
                f"after speak command{sr_note}"
            )
        else:
            sr_note = (
                f" (stream already open at {s.playback_sample_rate} Hz)"
                if s.playback_sample_rate is not None else ""
            )
            lines.append(f"     playback-open: (no separate open event logged){sr_note}")
        lines.append(f"     DTD while playing: {_dtd_summary(s.dtd_frames)}")
        for be in s.barge_events:
            t_boff = _fmt_offset(be.t, s.t_speak)
            label = classify_barge_event(be)
            flag = "  [SELF-INTERRUPT SUSPECT]" if label.startswith("suspect") else ""
            if be.kind == "detected":
                lines.append(
                    f"     barge-in DETECTED at {t_boff} speaking={be.speaking_at_event}{flag}"
                )
            else:
                lines.append(
                    f"     barge-in REJECTED at {t_boff} speaking={be.speaking_at_event}  ({be.reason[:60]})"
                )
            if be.dtd_context:
                lines.append(_dtd_context_str(be.dtd_context))
        if wav_metrics and s.idx in wav_metrics:
            wm = wav_metrics[s.idx]
            if "rms" in wm and wm["rms"] is not None:
                noise_flag = ""
                if wm.get("hf_ratio", 0) > 0.35:
                    noise_flag = "  [HIGH-HF: possible white-noise/artifact]"
                if wm.get("clip_pct", 0) > 5:
                    noise_flag += "  [CLIPPING]"
                quality = _ref_quality_label(wm)
                if quality in ("no-ref", "weak-ref"):
                    noise_flag += f"  [{quality.upper()}]"
                onset = wm.get("first_audio_offset_s")
                onset_str = (
                    f" first_ref_audio=+{float(onset):.3f}s"
                    if onset is not None else " first_ref_audio=(none)"
                )
                lines.append(
                    f"     ref_wav ({wm['duration_s']:.2f}s): "
                    f"rms={wm['rms']:.4f} active_rms={wm.get('active_rms')} "
                    f"peak={wm['peak']:.4f} clip={wm['clip_pct']:.1f}% "
                    f"hf_ratio={wm['hf_ratio']:.3f}{onset_str}{noise_flag}"
                )
        if delay_metrics and s.idx in delay_metrics:
            dm = delay_metrics[s.idx]
            if dm.get("estimated_delay_ms") is not None:
                mismatch = ""
                if (
                    run.aec_config_ref_delay_ms is not None
                    and dm.get("delay_correlation") is not None
                    and float(dm["delay_correlation"]) >= 0.15
                    and abs(float(dm["estimated_delay_ms"]) - run.aec_config_ref_delay_ms) >= 50.0
                ):
                    mismatch = "  [AEC-DELAY-MISMATCH]"
                lines.append(
                    f"     mic/ref delay estimate: {dm['estimated_delay_ms']:.1f}ms "
                    f"corr={dm['delay_correlation']:.3f}{mismatch}"
                )

    # --- Barge-in Event Log ---
    lines.append("\n--- Barge-in Events ---")
    if not run.barge_events:
        lines.append("  (none)")
    for be in run.barge_events:
        t_off = _fmt_offset(be.t, base) if base else f"t={be.t:.3f}"
        label = classify_barge_event(be)
        flag = f"  [{label.upper()}]" if label not in ("ok", "uncertain") else ""
        tag = "DETECTED" if be.kind == "detected" else "REJECTED"
        lines.append(f"[{tag}] {t_off}  sentence={be.sentence_idx}  speaking={be.speaking_at_event}{flag}")
        if be.dtd_context:
            lines.append(_dtd_context_str(be.dtd_context))

    # --- Word-Cut Funnel (ADR-0013) ---
    # Only rendered when the continuous word-cut path emitted telemetry, so a
    # legacy / APM run stays quiet instead of showing a zero block.
    wcf = word_cut_funnel(run)
    if wcf.get("present"):
        lines.append("\n--- Word-Cut Funnel (ADR-0013) ---")
        lines.append(
            f"  replies watched: {wcf['replies']}  fed={wcf['fed']} "
            f"skipped_quiet={wcf['skipped_quiet']} resets={wcf['resets']}"
        )
        lines.append(
            f"  words: max_seen={wcf['max_words']} dropped_by_reset={wcf['dropped_words']} "
            f"own_folds={wcf['own_folds']}"
        )
        lines.append(
            f"  cuts: {wcf['cuts']} (confirmed lines: {wcf['confirmed_lines']})  "
            f"guard_suppressed={wcf['guard_suppressed']} decode_errors={wcf['decode_errors']}"
        )
        lines.append(
            f"  near-end windows: {wcf['windows']} total, {wcf['voiced_windows']} "
            "with energy >=2x floor"
        )
        if wcf["starved_replies"]:
            lines.append(
                f"  [!] {wcf['starved_replies']} reply(ies) with fed=0 -- the VAD "
                "gate starved the recognizer (state-machine, not acoustics)"
            )
        if wcf["voice_present_zero_words"]:
            lines.append(
                f"  [!] {wcf['voice_present_zero_words']} voiced window(s) but ZERO "
                "words transcribed -- energy survived the canceller, text did not"
            )
        if wcf["windows"] and not wcf["voiced_windows"]:
            lines.append(
                "  [!] near-end never rose >=2x above the floor -- the OS "
                "canceller suppressed the user during playback"
            )
        if wcf["dropped_words"]:
            lines.append(
                f"  [~] burst resets wiped {wcf['dropped_words']} accumulated "
                "word(s) -- check barge_word_cut_reset_quiet_blocks"
            )

    # --- Barge Confirm Funnel (ADR-0011) ---
    # Only rendered when the word-gated duck-then-confirm path was active, so a
    # legacy run without it stays quiet instead of showing a zero block.
    if run.barge_duck or run.barge_confirmed or run.barge_unconfirmed:
        lines.append("\n--- Barge Confirm Funnel (ADR-0011) ---")
        lines.append(f"  acoustic triggers  (barge_in_duck):        {run.barge_duck}")
        lines.append(f"  confirmed by speech (barge_in_confirmed):  {run.barge_confirmed}")
        lines.append(f"  self-healed        (barge_in_unconfirmed): {run.barge_unconfirmed}")
        if run.barge_duck and run.barge_unconfirmed / run.barge_duck >= 0.5:
            lines.append(
                f"  [~] {run.barge_unconfirmed}/{run.barge_duck} triggers self-healed "
                "— acoustic gate firing on echo (word gate absorbing it, no pumping)"
            )

    # --- Self-interrupt Summary ---
    si = self_interrupt_summary(run)
    lines.append("\n--- Self-Interrupt Summary ---")
    lines.append(f"verdict: {si['verdict'].upper()}")
    lines.append(f"  barge-in detected: {si['detected_total']}  suspects: {si['suspect_count']}")
    lines.append(f"  real-barge rejected while speaking: {si['rejected_while_speaking']}")
    if si["suspects"]:
        for sp in si["suspects"]:
            lines.append(f"  → sentence[{sp['sentence']}] t={sp['t']:.3f}  {sp['label']}")

    # --- Capture Heartbeat Stats ---
    if run.heartbeats:
        hbs_while_speaking = [hb for hb in run.heartbeats if hb.speaking]
        hbs_silent = [hb for hb in run.heartbeats if not hb.speaking]
        total_underruns = run.heartbeats[-1].underruns
        lines.append("\n--- Capture Heartbeat Stats ---")
        lines.append(f"  playback underruns (cumulative): {total_underruns}")
        if hbs_while_speaking:
            avg_rms = sum(hb.avg_rms for hb in hbs_while_speaking) / len(hbs_while_speaking)
            avg_clip = sum(hb.clip for hb in hbs_while_speaking) / len(hbs_while_speaking)
            lines.append(
                f"  while speaking ({len(hbs_while_speaking)} hb): "
                f"avg_rms={avg_rms:.4f} avg_clip={avg_clip:.2f}%"
            )
        if hbs_silent:
            avg_rms = sum(hb.avg_rms for hb in hbs_silent) / len(hbs_silent)
            lines.append(
                f"  while silent ({len(hbs_silent)} hb): avg_rms={avg_rms:.4f}"
            )

    # --- Pass/Fail Verdict (for live A/B) ---
    pf = pass_fail_verdict(run)
    lines.append("\n--- Pass/Fail Verdict ---")
    lines.append(f"OVERALL: {pf['overall']}")
    _pf_icons = {"PASS": "✓", "WARN": "~", "FAIL": "✗", "UNKNOWN": "?"}
    lines.append(
        f"  [{_pf_icons.get(pf['self_interrupt'], '?')}] self-interrupt: "
        f"{pf['self_interrupt']}  (suspects={pf['self_interrupt_suspects']})"
    )
    lines.append(
        f"  [{_pf_icons.get(pf['underrun_verdict'], '?')}] underruns: "
        f"{pf['underrun_verdict']}  (total={pf['underruns_total']})"
    )
    fa_ms = f"{pf['first_audio_avg_ms']:.0f} ms" if pf["first_audio_avg_ms"] is not None else "n/a"
    lines.append(
        f"  [{_pf_icons.get(pf['first_audio_verdict'], '?')}] first-audio latency: "
        f"{pf['first_audio_verdict']}  (avg={fa_ms} over {pf['first_audio_count']} sentences)"
    )
    if pf["rejected_while_speaking"] > 0:
        lines.append(
            f"  [~] real-barge rejected while speaking: {pf['rejected_while_speaking']}"
            f"  (potential missed cut-offs — validate live)"
        )
    if pf["pre_first_audio_noise"] > 0:
        lines.append(
            f"  [~] pre-first-audio barge noise: {pf['pre_first_audio_noise']} events"
            f"  (suppressed by _barge_watch_active gate)"
        )

    return "\n".join(lines)


def to_json(run: ParsedRun, wav_metrics: Optional[dict] = None, si: Optional[dict] = None, delay_metrics: Optional[dict] = None) -> dict:
    """Structured JSON-serializable representation."""
    si = si or self_interrupt_summary(run)
    pf = pass_fail_verdict(run)
    return {
        "run_id": run.run_id,
        "aec": {
            "backend": run.aec_backend,
            "configured_ref_delay_ms": run.aec_config_ref_delay_ms,
            "apm_always_on": run.apm_always_on,
        },
        "self_interrupt": si,
        "barge_confirm_funnel": {
            "barge_in_duck": run.barge_duck,
            "barge_in_confirmed": run.barge_confirmed,
            "barge_in_unconfirmed": run.barge_unconfirmed,
        },
        "word_cut_funnel": word_cut_funnel(run),
        "pass_fail": pf,
        "findings": diagnostic_findings(run, wav_metrics, delay_metrics),
        "sentences": [
            {
                "idx": s.idx,
                "t_speak": s.t_speak,
                "text": s.text,
                "tts_sanitize": s.tts_sanitize,
                "tts_resolved": s.tts_resolved,
                "tts_quality": s.tts_quality,
                "queue_depth": s.queue_depth,
                "playback_sample_rate": s.playback_sample_rate,
                "playback_open_latency_ms": s.playback_open_latency_ms,
                "t_end": s.t_end,
                "dtd": {
                    "n_frames": len(s.dtd_frames),
                    "n_fired": sum(1 for f in s.dtd_frames if f.fired),
                    "n_gated": sum(1 for f in s.dtd_frames if f.gated),
                    "avg_incoh": (
                        round(sum(f.incoh for f in s.dtd_frames) / len(s.dtd_frames), 3)
                        if s.dtd_frames else None
                    ),
                },
                "barge_events": [
                    {
                        "kind": be.kind,
                        "t": be.t,
                        "speaking": be.speaking_at_event,
                        "suspicion": classify_barge_event(be),
                        "phase": _barge_phase(be, run, wav_metrics),
                        "dtd_context": [
                            {
                                "D": f.D, "gated": f.gated, "incoh": f.incoh,
                                "raw": f.raw, "resid": f.resid,
                                "coh": f.coh_verdict, "coh_veto": f.coh_veto,
                                "ref_delay_ms": f.ref_delay_ms,
                            }
                            for f in be.dtd_context
                        ],
                    }
                    for be in s.barge_events
                ],
                "wav": wav_metrics.get(s.idx) if wav_metrics else None,
                "mic_ref_delay": delay_metrics.get(s.idx) if delay_metrics else None,
            }
            for s in run.sentences
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Post-hoc diagnostics for a run bundle (.txt log)."
    )
    parser.add_argument("log", help="Path to run-<id>.txt log file")
    parser.add_argument(
        "--wav", default=None,
        help="Path to .ref.wav (playback reference); auto-discovered if omitted",
    )
    parser.add_argument(
        "--mic-wav", default=None,
        help="Path to session mic .wav for mic/ref delay estimate; auto-discovered if omitted",
    )
    parser.add_argument(
        "--hf-cutoff", type=int, default=4000,
        help="Hz above which energy is counted as HF (default: 4000)",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument(
        "--exit-code", action="store_true",
        help="Exit 1 if overall verdict is FAIL, 2 if WARN (useful for CI / scripting)",
    )
    parser.add_argument(
        "--verdict-only", action="store_true",
        help="Print only the pass/fail verdict block and exit (implies --exit-code)",
    )
    args = parser.parse_args(argv)

    txt_path = args.log
    if not Path(txt_path).exists():
        print(f"ERROR: log not found: {txt_path}", file=sys.stderr)
        return 1

    run = parse_log(txt_path)

    # Auto-discover .ref.wav
    wav_path = args.wav
    if wav_path is None and run.ref_wav_path:
        candidate = Path(txt_path).parent / Path(run.ref_wav_path).name
        if candidate.exists():
            wav_path = str(candidate)

    mic_wav_path = args.mic_wav
    if mic_wav_path is None:
        candidate = Path(txt_path).with_suffix(".wav")
        if candidate.exists():
            mic_wav_path = str(candidate)

    wav_metrics: Optional[dict] = None
    delay_metrics: Optional[dict] = None
    if wav_path:
        try:
            import numpy  # noqa: F401 — check available before calling
            wav_metrics = analyze_ref_wav(wav_path, run, hf_cutoff_hz=args.hf_cutoff)
            if mic_wav_path:
                delay_metrics = analyze_mic_ref_wav(mic_wav_path, wav_path, run)
        except ImportError:
            print(
                "WARNING: numpy not available; WAV analysis skipped",
                file=sys.stderr,
            )

    pf = pass_fail_verdict(run)

    if getattr(args, "verdict_only", False):
        print(f"OVERALL: {pf['overall']}")
        for key in ("self_interrupt", "underrun_verdict", "first_audio_verdict"):
            print(f"  {key}: {pf[key]}")
        print(f"  underruns_total: {pf['underruns_total']}")
        print(f"  first_audio_avg_ms: {pf['first_audio_avg_ms']}")
        print(f"  pre_first_audio_noise: {pf['pre_first_audio_noise']}")
    elif args.json:
        print(json.dumps(to_json(run, wav_metrics, delay_metrics=delay_metrics), indent=2))
    else:
        print(format_report(run, wav_metrics, delay_metrics))

    if getattr(args, "exit_code", False) or getattr(args, "verdict_only", False):
        if pf["overall"] == "FAIL":
            return 1
        if pf["overall"] == "WARN":
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
