"""Measure AEC echo reduction (ERLE) over a recorded run's mic + playback
reference -- HEADLESS, no live audio. Sweeps the reference delay to find the
acoustic speaker->mic latency, and reports ERLE + how often the canceller
DIVERGES (resets to passthrough), per backend. Lets AEC be calibrated against
the owner's real audio without re-running live.

ERLE (Echo Return Loss Enhancement) = 10*log10(echo_power_in / residual_power_out)
over the frames where the assistant is playing (its echo dominates the mic).
Higher = more echo removed; ~0 dB or divergence = the canceller isn't helping.

Usage:
    python -m tools.aec_probe logs/runs/run-<id>.wav [--backend nlms] \
        [--device <profile>] [--taps 512] [--mu 0.3]
Needs a sibling run-<id>.ref.wav (record_playback_reference=true).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _load(path: str):
    from core.engines.file_replay import load_waveform

    s, sr = load_waveform(path)
    return np.asarray(s, dtype="float32").reshape(-1), sr


def _erle_for_delay(mic, ref, sc, *, delay_samples, ref_active_rms):
    """Run the AEC with the far reference pre-delayed by ``delay_samples`` and
    return (ERLE dB, divergence_fraction, playback_frames)."""
    from core.engines._aec import build_aec

    aec = build_aec(sc)
    if aec is None:
        return None
    far = np.concatenate([np.zeros(delay_samples, dtype="float32"), ref])[: len(mic)]
    sr, b = sc.sample_rate, max(1, int(sc.sample_rate * 0.1))
    n = min(len(mic), len(far)) // b
    echo_in = resid_out = 0.0
    frames = diverged = 0
    for i in range(n):
        m = mic[i * b:(i + 1) * b]
        f = far[i * b:(i + 1) * b]
        if float(np.sqrt(np.mean(f * f))) <= ref_active_rms:
            continue                                  # not playing -> not an echo frame
        out = np.asarray(aec.process_16k(m, f), dtype="float64").reshape(-1)
        k = min(len(out), len(m))
        if k == 0:
            continue
        mm = m[:k].astype("float64")
        oo = out[:k]
        echo_in += float(np.sum(mm ** 2))
        resid_out += float(np.sum(oo ** 2))
        frames += 1
        # The guard returns the raw near-end on divergence -> residual ~= input.
        if float(np.sqrt(np.mean((oo - mm) ** 2))) < 1e-6:
            diverged += 1
    if frames == 0 or resid_out <= 0:
        return None
    erle = 10.0 * np.log10(echo_in / resid_out)
    return erle, diverged / frames, frames


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tools.aec_probe", description=__doc__)
    p.add_argument("mic_wav")
    p.add_argument("--device", default=None)
    p.add_argument("--backend", default="nlms", help="nlms | dtln")
    p.add_argument("--taps", type=int, default=None)
    p.add_argument("--mu", type=float, default=None)
    p.add_argument("--ref-active-rms", type=float, default=0.003)
    p.add_argument("--max-delay-ms", type=int, default=400)
    args = p.parse_args(argv)

    ref_wav = args.mic_wav[:-4] + ".ref.wav" if args.mic_wav.endswith(".wav") else args.mic_wav + ".ref.wav"
    if not os.path.exists(ref_wav):
        print(f"[aec_probe] no reference WAV {ref_wav!r} -- re-run with "
              f"record_playback_reference=true.", file=sys.stderr)
        return 2

    from core.config import apply_device_profile, load_config
    from core.engines.sherpa import SherpaConfig

    base = load_config()
    cfg = apply_device_profile(base, args.device or base.get("device", "auto"))
    sherpa = dict(cfg.get("sherpa", {}))
    sherpa.update(aec_enabled=True, aec_backend=args.backend)
    if args.taps is not None:
        sherpa["aec_filter_taps"] = args.taps
    if args.mu is not None:
        sherpa["aec_mu"] = args.mu
    sc = SherpaConfig.from_dict(sherpa)

    mic, _ = _load(args.mic_wav)
    ref, _ = _load(ref_wav)
    sr = sc.sample_rate
    print(f"[aec_probe] {args.mic_wav}  backend={args.backend} taps={sc.aec_filter_taps} mu={sc.aec_mu}")
    print(f"  sweeping reference delay 0..{args.max_delay_ms}ms for max ERLE "
          f"(echo reduction) over playback frames:")
    best = None
    for ms in range(0, args.max_delay_ms + 1, 20):
        r = _erle_for_delay(mic, ref, sc, delay_samples=int(sr * ms / 1000), ref_active_rms=args.ref_active_rms)
        if r is None:
            continue
        erle, divf, frames = r
        flag = "  <-- diverges" if divf > 0.2 else ""
        print(f"   delay={ms:4d}ms  ERLE={erle:+5.1f}dB  diverged={divf*100:3.0f}%  ({frames} frames){flag}")
        if best is None or erle > best[1]:
            best = (ms, erle, divf)
    if best:
        print(f"  BEST: delay={best[0]}ms  ERLE={best[1]:+.1f}dB  diverged={best[2]*100:.0f}%")
        print("  (ERLE > ~6dB = useful echo reduction; ~0dB or high divergence = this "
              "backend can't cancel this speaker -> try --backend dtln)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
