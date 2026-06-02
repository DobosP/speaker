#!/usr/bin/env python3
"""Interrupt (barge-in) test SUITE -- sweeps the interrupt matrix on real hardware.

PLAYS AUDIO OUT LOUD and captures the mic. For each (microphone x interrupt
strategy) cell it drives :mod:`tools.echo_probe` (the real sherpa engine) which
plays N TTS sentences and -- because the probe never talks over itself -- counts
every barge-in as a SELF-INTERRUPT. The suite tabulates self-interrupts +
coherence headroom across the matrix so you can see *what type of interrupt
works* on *which mic* without self-firing on the assistant's own echo.

    python -m tools.interrupt_suite                 # both mics, default matrix
    python -m tools.interrupt_suite --mics at2020   # one mic
    python -m tools.interrupt_suite --sentences 4

IMPORTANT: stay QUIET while it runs -- any voice during a cell looks like a real
barge and inflates that cell's self-interrupt count. The fires-on-a-REAL-barge
direction is covered by the offline unit tests (tests/test_echo_coherence.py)
and a live `python -m core --engine sherpa` talk-over; this suite measures the
no-self-interrupt half, which is the failure mode being debugged.

Each cell is a fresh echo_probe subprocess (clean device open/close). A mic is
liveness-checked with a short arecord capture first; a dead/muted mic (the AT2020
touch-mute) is reported and skipped.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import wave

import numpy as np

# Microphone specs: label -> (engine device name, arecord device for liveness).
MICS = {
    "alc285": ("ALC285 Analog", "plughw:1,0"),   # laptop built-in mic
    "at2020": ("AT2020USB-X", "plughw:2,0"),     # external USB condenser
}
OUTPUT_DEVICE = "ALC285 Analog"  # the only real speaker on this box

# Interrupt strategies to sweep. Each is a set of echo_probe overrides.
STRATEGIES = [
    {"label": "coherence-confirm1", "coherence": "on", "confirm_frames": 1, "aec": "off", "margin_db": 0.0},
    {"label": "coherence-confirm2", "coherence": "on", "confirm_frames": 2, "aec": "off", "margin_db": 0.0},
    {"label": "coherence-confirm3", "coherence": "on", "confirm_frames": 3, "aec": "off", "margin_db": 0.0},
    {"label": "level-margin6",      "coherence": "off", "confirm_frames": 2, "aec": "off", "margin_db": 6.0},
    {"label": "dtln-aec",           "coherence": "off", "confirm_frames": 2, "aec": "on",  "margin_db": 0.0},
]


def _wav_rms(path: str) -> tuple[float, float, int]:
    try:
        w = wave.open(path, "rb")
        n = w.getnframes()
        a = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
        w.close()
    except Exception:
        return 0.0, 0.0, 0
    if a.size == 0:
        return 0.0, 0.0, 0
    return float(np.sqrt(np.mean(a * a))), float(np.max(np.abs(a))), int(np.count_nonzero(a))


def mic_is_live(arecord_dev: str) -> tuple[bool, float]:
    """Capture ~2 s and report whether the mic produces signal (vs hard-muted)."""
    path = "/tmp/interrupt_suite_liveness.wav"
    try:
        subprocess.run(
            ["arecord", "-D", arecord_dev, "-f", "S16_LE", "-r", "44100", "-c", "1", "-d", "2", path],
            check=True, capture_output=True, timeout=10,
        )
    except Exception as e:
        print(f"    liveness capture FAILED on {arecord_dev}: {e}")
        return False, 0.0
    rms, peak, nz = _wav_rms(path)
    # A muted mic delivers (near-)exact zeros; a live one always has a noise floor.
    return (rms > 1e-5 and nz > 0), rms


def run_cell(mic_name: str, strat: dict, sentences: int, timeout: float) -> dict:
    label = f"{mic_name}/{strat['label']}"
    cmd = [
        sys.executable, "-m", "tools.echo_probe",
        "--label", label,
        "--input-device", mic_name,
        "--output-device", OUTPUT_DEVICE,
        "--coherence", strat["coherence"],
        "--confirm-frames", str(strat["confirm_frames"]),
        "--aec", strat["aec"],
        "--margin-db", str(strat["margin_db"]),
        "--sentences", str(sentences),
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"label": label, "error": "timeout"}
    out = p.stdout.strip()
    # echo_probe prints one JSON object to stdout; logs go to stderr. Be defensive
    # and parse from the first brace in case anything leaks onto stdout.
    i = out.find("{")
    if i < 0:
        return {"label": label, "error": "no-json", "stderr_tail": p.stderr.strip()[-400:]}
    try:
        return json.loads(out[i:])
    except Exception as e:
        return {"label": label, "error": f"json:{e}", "stdout_tail": out[-400:]}


def summarize(cell: dict) -> dict:
    coh = cell.get("coherence") or {}
    return {
        "label": cell.get("label"),
        "self_int": cell.get("self_interruptions"),
        "coh_fired": coh.get("coherence_fired_on_own_tts"),
        "headroom_p95": coh.get("headroom_p95"),
        "self_cal_margin": coh.get("self_calibrated_margin"),
        "vad_flagged": cell.get("vad_flagged_during_play"),
        "peak_play": cell.get("peak_playback_level"),
        "mic_over_play_dB": cell.get("median_mic_over_playback_dB"),
        "error": cell.get("error"),
    }


def coupled(s: dict) -> bool:
    """Did the assistant's echo actually reach the mic this cell? (Else the
    zero-self-interrupt result is vacuous -- volume too low / mic muted.)"""
    return bool((s.get("vad_flagged") or 0) > 0 and (s.get("peak_play") or 0) > 1e-4)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep the barge-in interrupt matrix on real hardware.")
    ap.add_argument("--mics", default="alc285,at2020", help="comma list: alc285,at2020")
    ap.add_argument("--sentences", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=90.0, help="per-cell subprocess timeout (s)")
    ap.add_argument("--out", default="test-reports/interrupt_suite.json")
    args = ap.parse_args()

    mics = [m.strip() for m in args.mics.split(",") if m.strip()]
    rows: list[dict] = []
    raw: list[dict] = []

    for mic in mics:
        if mic not in MICS:
            print(f"unknown mic {mic!r}; known: {list(MICS)}")
            continue
        engine_name, arecord_dev = MICS[mic]
        print(f"\n=== MIC: {mic} ({engine_name}) ===")
        live, rms = mic_is_live(arecord_dev)
        print(f"    liveness: rms={rms:.6f} -> {'LIVE' if live else 'DEAD/MUTED -- skipping'}")
        if not live:
            rows.append({"label": f"{mic}/<all>", "error": "mic-dead", "self_int": None})
            continue
        for strat in STRATEGIES:
            print(f"    cell: {strat['label']} ...", flush=True)
            cell = run_cell(engine_name, strat, args.sentences, args.timeout)
            cell.setdefault("label", f"{engine_name}/{strat['label']}")
            cell["_mic"] = mic
            raw.append(cell)
            s = summarize(cell)
            s["coupled"] = coupled(s)
            rows.append(s)
            print(f"        self_int={s['self_int']} coh_fired={s['coh_fired']} "
                  f"headroom_p95={s['headroom_p95']} coupled={s['coupled']} "
                  f"vad={s['vad_flagged']} peak={s['peak_play']} err={s['error']}")

    # --- table ---
    print("\n================ INTERRUPT MATRIX ================")
    hdr = f"{'cell':28s} {'self_int':>8s} {'coh_fire':>8s} {'headroom':>9s} {'coupled':>7s} {'vad':>5s}"
    print(hdr)
    print("-" * len(hdr))
    for s in rows:
        if s.get("error") and s.get("self_int") is None:
            print(f"{str(s.get('label')):28s} {'--':>8s} {'--':>8s} {'--':>9s} {'--':>7s} {'--':>5s}  ({s['error']})")
            continue
        print(f"{str(s.get('label')):28s} {str(s.get('self_int')):>8s} {str(s.get('coh_fired')):>8s} "
              f"{str(s.get('headroom_p95')):>9s} {str(s.get('coupled')):>7s} {str(s.get('vad_flagged')):>5s}")

    # --- recommendation ---
    print("\n================ RECOMMENDATION ================")
    good = [s for s in rows if s.get("self_int") == 0 and s.get("coupled")]
    inconclusive = [s for s in rows if s.get("self_int") == 0 and not s.get("coupled") and not s.get("error")]
    if good:
        # Prefer coherence with the smallest confirm_frames that reaches 0; then any.
        def rank(s):
            lbl = s["label"]
            is_coh = "coherence" in lbl
            cf = 9
            for n in (1, 2, 3):
                if f"confirm{n}" in lbl:
                    cf = n
            return (0 if is_coh else 1, cf)
        good.sort(key=rank)
        for s in good:
            print(f"  WORKS (0 self-interrupts, echo coupled): {s['label']}  headroom_p95={s['headroom_p95']}")
        print(f"\n  -> recommended: {good[0]['label']}")
    else:
        print("  No strategy reached 0 self-interrupts with echo coupled.")
        if inconclusive:
            print("  (cells with 0 self-interrupts but NO echo coupling -- raise OS volume / unmute, re-run:)")
            for s in inconclusive:
                print(f"     {s['label']}")
        print("  If even the open speaker can't be tamed: use HEADPHONES (echo-free) or rely on DTLN-AEC.")

    try:
        import os
        os.makedirs("test-reports", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"rows": rows, "raw": raw}, f, indent=2)
        print(f"\n[suite] raw results -> {args.out}")
    except Exception as e:
        print(f"[suite] could not write {args.out}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
