#!/usr/bin/env python3
"""Live audio self-interruption (TTS-echo) probe -- PLAYS AUDIO OUT LOUD.

Drives the REAL ``sherpa`` engine (mic + speakers) to measure whether the
assistant's own TTS, played through the speakers, is re-captured by the mic
strongly enough to trip a barge-in (a *self-interruption*), and how the
unenrolled output-margin gate (``barge_in_output_margin_db``) responds.

This is the on-device calibration the P1 review deferred (we shipped
``barge_in_output_margin_db = 0`` -- suppression OFF -- pending real data). The
gate compares mic RMS (post speaker-volume + room coupling) against
``_playback_level`` (the pre-volume TTS *buffer* RMS), so the OS output volume
is exactly what can break that comparison: crank the speaker and the captured
echo can exceed the buffer reference even though it is "just" the assistant.

    python -m tools.echo_probe --margin-db 0 --sentences 4
    python -m tools.echo_probe --margin-db 6 --gain 1.0

Prints a JSON summary so a run is interpretable even if nothing coupled:
``self_interruptions`` (barge-ins fired during the assistant's own speech),
the (mic_rms, playback_level, gate_passed) samples the gate evaluated, the
mic/playback ratio in dB, and the peak playback level.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import threading
import time

SENTENCES = [
    "This is a live audio calibration test of the speaker assistant.",
    "I am checking whether my own voice is captured back by the microphone.",
    "The barge in gate should not interrupt me while I am still speaking.",
    "If you hear this whole message without it cutting off, suppression works.",
]


def _rms(samples) -> float:
    import numpy as np

    a = np.asarray(samples, dtype="float32").reshape(-1)
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(a * a)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Live TTS-echo / self-interruption probe.")
    ap.add_argument("--margin-db", type=float, default=0.0, help="barge_in_output_margin_db override")
    ap.add_argument("--gain", type=float, default=1.0, help="scale synthesized TTS output (in-app 'volume')")
    ap.add_argument("--sentences", type=int, default=4)
    ap.add_argument("--device", default=None, help="device_profile (default: config.device)")
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    # utf-8 stdout so any non-ASCII never crashes the cp1252 console.
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    from core.config import apply_device_profile, load_config
    from core.engine import EngineCallbacks
    from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine

    cfg = load_config(args.config)
    device = args.device or cfg.get("device") or "desktop"
    cfg = apply_device_profile(cfg, device)
    sherpa_cfg = SherpaConfig.from_dict(cfg.get("sherpa", {}))
    try:
        sherpa_cfg.barge_in_output_margin_db = args.margin_db
    except dataclasses.FrozenInstanceError:
        sherpa_cfg = dataclasses.replace(sherpa_cfg, barge_in_output_margin_db=args.margin_db)

    if not getattr(sherpa_cfg, "tts_model", ""):
        print(json.dumps({"error": "no sherpa.tts_model configured; run `python -m tools.setup_models`"}))
        return 1

    engine = SherpaOnnxEngine(sherpa_cfg)

    barge_ins: list[float] = []
    gate_samples: list[tuple[float, float, bool]] = []
    peak_playback = 0.0

    # Instrument the unenrolled echo gate to record exactly what it compares.
    _orig_llu = engine._looks_like_user

    def _wrapped_llu(samples):
        passed = _orig_llu(samples)
        try:
            gate_samples.append((round(_rms(samples), 5), round(float(engine._playback_level), 5), bool(passed)))
        except Exception:
            pass
        return passed

    engine._looks_like_user = _wrapped_llu  # type: ignore[assignment]

    # In-app gain: scale synthesized output so we can sweep "volume" deterministically
    # (note: this scales BOTH the speaker output and the _playback_level reference,
    # so it does NOT expose the OS-volume scale mismatch -- vary OS volume for that).
    if args.gain != 1.0:
        _orig_synth = engine._synthesize

        def _scaled_synth(text, write):
            _orig_synth(text, lambda s: write(__import__("numpy").asarray(s, dtype="float32") * args.gain))

        engine._synthesize = _scaled_synth  # type: ignore[assignment]

    cb = EngineCallbacks(
        on_barge_in=lambda: barge_ins.append(time.monotonic()),
    )

    try:
        engine.start(cb)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"engine.start failed: {type(exc).__name__}: {exc}"}))
        return 1

    time.sleep(1.0)  # let the capture loop spin up
    spoken = 0
    try:
        for i in range(max(1, args.sentences)):
            done = threading.Event()
            engine.speak(SENTENCES[i % len(SENTENCES)], on_done=done.set)
            deadline = time.monotonic() + 20.0
            while not done.wait(0.05):
                lvl = float(engine._playback_level)
                if lvl > peak_playback:
                    peak_playback = lvl
                if time.monotonic() > deadline:
                    break
            spoken += 1
            time.sleep(0.4)
        time.sleep(0.5)
    finally:
        try:
            engine.stop()
        except Exception:
            pass

    ratios = [
        round(20.0 * math.log10(mic / pb), 1)
        for (mic, pb, _) in gate_samples
        if pb > 1e-6 and mic > 1e-6
    ]
    ratios_sorted = sorted(ratios)
    out = {
        "margin_db": args.margin_db,
        "gain": args.gain,
        "device": device,
        "sentences_spoken": spoken,
        "self_interruptions": len(barge_ins),
        "vad_flagged_during_play": len(gate_samples),
        "gate_passed_count": sum(1 for (_, _, p) in gate_samples if p),
        "peak_playback_level": round(peak_playback, 5),
        "median_mic_over_playback_dB": (ratios_sorted[len(ratios_sorted) // 2] if ratios_sorted else None),
        "max_mic_over_playback_dB": (ratios_sorted[-1] if ratios_sorted else None),
        "mic_over_playback_dB_samples": ratios[:40],
        "note": (
            "self_interruptions>0 means the assistant's own TTS tripped barge-in. "
            "peak_playback_level~0 or vad_flagged_during_play==0 means little/no echo "
            "coupled (volume low/muted or headphones) -- not conclusive; raise the volume."
        ),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
