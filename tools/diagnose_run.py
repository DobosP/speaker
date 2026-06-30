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

_SPEAKING_PAT = re.compile(r"speaking:\s+'(.+)'\s+\(queue depth=(\d+)\)")
_PLAYBACK_OPEN_PAT = re.compile(r"playback opened at (\d+) Hz")
_BARGE_DETECTED_PAT = re.compile(r"barge-in detected")
_BARGE_REJECTED_PAT = re.compile(r"barge-in REJECTED:\s+(.*)")
_DTD_PAT = re.compile(
    r"dtd:\s+D=([\d.]+)\s+K=([\d.]+)\s+fired=(True|False)\s+gated=(True|False)"
    r"\s+\(z_raw=([\d.]+)\s+z_resid=([\d.]+)\s+z_coh=([\d.]+)\)"
    r"\s+raw=([\d.]+)\s+resid=([\d.]+)\s+incoh=([\d.]+)"
    r"\s+resid_floor=([\d.]+)\s+consec=(\d+)"
)
_HEARTBEAT_PAT = re.compile(
    r"capture heartbeat:\s+blocks=(\d+)\s+avg_rms=([\d.]+)\s+clip=([\d.]+)%"
    r"\s+underruns=(\d+)\s+partials=(\d+)\s+finals=(\d+)\s+speaking=(True|False)"
)
_REF_WAV_PAT = re.compile(r"recording playback reference.*->\s+(\S+\.ref\.wav)")
_RUN_START_PAT = re.compile(r"run (\S+) started")


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
    t_playback_open: Optional[float] = None
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
class ParsedRun:
    run_id: str
    run_start_t: Optional[float]
    ref_wav_path: Optional[str]
    ref_wav_start_t: Optional[float]   # absolute timestamp when ref.wav recording began
    sentences: list
    barge_events: list
    heartbeats: list
    dtd_frames: list


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
                )
                sentences.append(sent)
                continue

            # Playback open
            mp = _PLAYBACK_OPEN_PAT.search(msg)
            if mp and "speaker.sherpa" in logger and sentences:
                s = sentences[-1]
                if s.t_playback_open is None:
                    s.t_playback_open = t
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

    return ParsedRun(
        run_id=run_id,
        run_start_t=run_start_t,
        ref_wav_path=ref_wav_path,
        ref_wav_start_t=ref_wav_start_t,
        sentences=sentences,
        barge_events=barge_events,
        heartbeats=heartbeats,
        dtd_frames=dtd_frames,
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
        return {"rms": None, "peak": None, "clip_pct": None, "hf_ratio": None, "duration_s": 0.0}

    rms = float(np.sqrt(np.mean(samples ** 2)))
    peak = float(np.max(np.abs(samples)))
    clip_pct = float(100.0 * np.mean(np.abs(samples) >= 0.99))
    duration_s = round(len(samples) / sample_rate, 3)

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
        "peak": round(peak, 4),
        "clip_pct": round(clip_pct, 2),
        "hf_ratio": hf_ratio,
        "duration_s": duration_s,
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
    import numpy as np

    try:
        w = wave.open(wav_path, "rb")
        sr = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
        w.close()
    except Exception as exc:
        return {"_error": str(exc)}

    # int16 PCM → float32 in [-1, 1]
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

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
    for s in run.sentences:
        pb_start = s.t_playback_open if s.t_playback_open is not None else s.t_speak
        t_start = pb_start - wav_base
        t_end = (s.t_end - wav_base) if s.t_end is not None else (t_start + 10.0)

        i0 = max(0, int(t_start * sr))
        i1 = min(len(pcm), int(t_end * sr))
        segment = pcm[i0:i1]
        metrics = _analyze_wav_segment(segment, sr, hf_cutoff_hz=hf_cutoff_hz)
        metrics["window_s"] = (round(t_start, 3), round(t_end, 3))
        results[s.idx] = metrics

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
    )


def _dtd_context_str(ctx: list[DTDFrame]) -> str:
    if not ctx:
        return "  (no DTD context)"
    lines = []
    for f in ctx:
        lines.append(
            f"  dtd D={f.D:.1f} gated={f.gated} incoh={f.incoh:.2f} "
            f"raw={f.raw:.4f} resid={f.resid:.4f}"
        )
    return "\n".join(lines)


def format_report(run: ParsedRun, wav_metrics: Optional[dict] = None) -> str:
    base = run.run_start_t or 0.0
    lines: list[str] = []
    lines.append(f"=== Run Diagnostics: {run.run_id} ===\n")

    # --- Sentence Timeline ---
    lines.append("--- Sentence Timeline ---")
    for s in run.sentences:
        t_off = _fmt_offset(s.t_speak, base) if base else f"t={s.t_speak:.3f}"
        lines.append(f"\n[{s.idx}] {t_off}  \"{s.text[:80]}\"")
        if s.t_playback_open is not None:
            lines.append(f"     playback-open: +{s.playback_open_latency_ms:.0f}ms after speak command")
        else:
            lines.append("     playback-open: (no separate open event logged)")
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
                lines.append(
                    f"     ref_wav ({wm['duration_s']:.2f}s): "
                    f"rms={wm['rms']:.4f} peak={wm['peak']:.4f} "
                    f"clip={wm['clip_pct']:.1f}% hf_ratio={wm['hf_ratio']:.3f}{noise_flag}"
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


def to_json(run: ParsedRun, wav_metrics: Optional[dict] = None, si: Optional[dict] = None) -> dict:
    """Structured JSON-serializable representation."""
    si = si or self_interrupt_summary(run)
    pf = pass_fail_verdict(run)
    return {
        "run_id": run.run_id,
        "self_interrupt": si,
        "pass_fail": pf,
        "sentences": [
            {
                "idx": s.idx,
                "t_speak": s.t_speak,
                "text": s.text,
                "queue_depth": s.queue_depth,
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
                        "dtd_context": [
                            {"D": f.D, "gated": f.gated, "incoh": f.incoh,
                             "raw": f.raw, "resid": f.resid}
                            for f in be.dtd_context
                        ],
                    }
                    for be in s.barge_events
                ],
                "wav": wav_metrics.get(s.idx) if wav_metrics else None,
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

    wav_metrics: Optional[dict] = None
    if wav_path:
        try:
            import numpy  # noqa: F401 — check available before calling
            wav_metrics = analyze_ref_wav(wav_path, run, hf_cutoff_hz=args.hf_cutoff)
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
        print(json.dumps(to_json(run, wav_metrics), indent=2))
    else:
        print(format_report(run, wav_metrics))

    if getattr(args, "exit_code", False) or getattr(args, "verdict_only", False):
        if pf["overall"] == "FAIL":
            return 1
        if pf["overall"] == "WARN":
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
