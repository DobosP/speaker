"""Replay an open-speaker run's mic + playback-reference through the REAL
EchoCoherenceDetector -- reproduce the self-interrupt headlessly, no mic.

A mic-ONLY recording can't reproduce the open-speaker barge loop, because the
thing the detector compares the mic against is the assistant's OWN playback. With
``record_playback_reference=true`` a run ALSO writes ``run-<id>.ref.wav`` (the
played far-end, frame-aligned with the mic). This tool feeds the mic to
``decide()`` and the reference to ``note_playback()`` and replays the EXACT
coherence comparison -- so barge tuning (margin / confirm / the playback-onset
grace) can be iterated WITHOUT a live mic.

Usage:
    python -m tools.replay_barge logs/runs/run-<id>.wav [--grace 0.40] \
        [--device <profile>] [--margin 0.08] [--confirm 2]

Expects a sibling ``run-<id>.ref.wav`` (re-run with record_playback_reference on).
Reports, per reply: when the reference (assistant) is playing, where decide()
says "user barge" (a SELF-INTERRUPT, since the user isn't talking on the
reference), and how many of those the playback-onset grace would suppress.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _load(path: str):
    from core.engines.file_replay import load_waveform

    samples, sr = load_waveform(path)
    return np.asarray(samples, dtype="float32").reshape(-1), sr


def _build_detector(sc, *, margin=None, confirm=None):
    from core.engines.echo_coherence import EchoCoherenceDetector

    return EchoCoherenceDetector(
        sc.sample_rate,
        voiced_band=tuple(sc.coherence_voiced_band_hz),
        ring_ms=sc.coherence_ring_ms,
        max_delay_ms=sc.coherence_max_delay_ms,
        nperseg=sc.coherence_nperseg,
        margin_delta=sc.coherence_margin_delta if margin is None else float(margin),
        confirm_frames=sc.coherence_confirm_frames if confirm is None else int(confirm),
        warmup_frames=sc.coherence_warmup_frames,
        sigma_k=sc.coherence_sigma_k,
        baseline_alpha=sc.coherence_baseline_alpha,
        var_alpha=sc.coherence_var_alpha,
        provisional_baseline=sc.coherence_provisional_baseline,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tools.replay_barge", description=__doc__)
    p.add_argument("mic_wav", help="run-<id>.wav (mic); expects a sibling run-<id>.ref.wav")
    p.add_argument("--device", default=None, help="device profile for the sherpa config")
    p.add_argument("--grace", type=float, default=None,
                   help="playback-onset grace (s) to model; default = config value")
    p.add_argument("--margin", type=float, default=None, help="override coherence_margin_delta")
    p.add_argument("--confirm", type=int, default=None, help="override coherence_confirm_frames")
    p.add_argument("--ref-active-rms", type=float, default=0.003,
                   help="ref rms above this = 'assistant playing' this frame")
    args = p.parse_args(argv)

    ref_wav = args.mic_wav[:-4] + ".ref.wav" if args.mic_wav.endswith(".wav") else args.mic_wav + ".ref.wav"
    if not os.path.exists(ref_wav):
        print(f"[replay_barge] no reference WAV {ref_wav!r} -- re-run with "
              f"record_playback_reference=true to capture it.", file=sys.stderr)
        return 2

    from core.config import apply_device_profile, load_config
    from core.engines.sherpa import SherpaConfig

    cfg = apply_device_profile(load_config(), args.device or load_config().get("device", "auto"))
    sc = SherpaConfig.from_dict(cfg.get("sherpa", {}))
    det = _build_detector(sc, margin=args.margin, confirm=args.confirm)
    if not det.available:
        print("[replay_barge] scipy not available -- coherence detector inert", file=sys.stderr)
        return 2
    grace = sc.barge_in_playback_onset_grace_sec if args.grace is None else args.grace

    mic, msr = _load(args.mic_wav)
    ref, rsr = _load(ref_wav)
    sr = sc.sample_rate
    block = max(1, int(sr * 0.1))
    n = min(len(mic), len(ref)) // block

    print(f"[replay_barge] {args.mic_wav}")
    print(f"  mic {len(mic)/msr:.1f}s  ref {len(ref)/rsr:.1f}s  | grace={grace:.2f}s "
          f"margin={det.margin_delta:.3f} confirm={det._confirm_frames}")

    playing = False
    onset_t = 0.0
    replies = 0
    self_int_raw = 0          # decide=True while the assistant is playing
    self_int_after_grace = 0  # ... and OUTSIDE the onset grace (would still self-interrupt)
    grace_saved = 0
    for i in range(n):
        t = i * 0.1
        mblk = mic[i * block:(i + 1) * block]
        rblk = ref[i * block:(i + 1) * block]
        ref_active = float(np.sqrt(np.mean(rblk * rblk))) > args.ref_active_rms
        if ref_active:
            if not playing:                 # silent -> playing: a new reply onset
                playing, onset_t = True, t
                replies += 1
                det.reset()                 # mirror the engine's per-reply reset
            det.note_playback(rblk, sr)
            verdict = det.decide(mblk)
            # A self-interrupt on an open speaker (AEC off) is EITHER coherence
            # false-firing (True) OR coherence abstaining (None) -> the loud-mic
            # LEVEL GATE fires. Both cut the assistant's own reply.
            if verdict is True or verdict is None:
                self_int_raw += 1
                why = "coherence=True" if verdict is True else "coherence=None->level-gate"
                if grace > 0.0 and t < onset_t + grace:
                    grace_saved += 1
                else:
                    self_int_after_grace += 1
                    print(f"  t={t:5.1f}s SELF-INTERRUPT ({why}, frac={det.last_incoherent_fraction:.2f} "
                          f"baseline={det.last_baseline:.2f} {t-onset_t:.2f}s into reply)")
        else:
            playing = False

    print(f"  replies={replies}  raw self-interrupts={self_int_raw}  "
          f"grace-suppressed={grace_saved}  REMAINING after grace={self_int_after_grace}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
