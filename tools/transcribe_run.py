#!/usr/bin/env python3
"""Ground-truth transcription of a recorded session WAV, with timestamps.

The live engine's own ASR (streaming zipformer + optional SenseVoice) is the very
thing under investigation when a session misbehaves, so to debug it we need an
INDEPENDENT transcript of what the user actually said and WHEN. This uses
faster-whisper (robust on short/casual speech) over the committed run-bundle
recording (``logs/runs/run-*.wav`` -- the post-AEC 16 kHz capture the engine fed
to ASR) and prints a timeline you can line up against the run log's events
(asr partials/finals, barge-in, assistant speech).

    python -m tools.transcribe_run logs/runs/run-20260602-235431.wav
    python -m tools.transcribe_run logs/runs/*.wav --model small.en --json

Note: this is the SAME audio the engine saw (post-AEC), so an artifact here is an
artifact the engine also had to transcribe -- which is itself a useful signal.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import wave

import numpy as np


def _load_wav_16k(path: str) -> np.ndarray:
    w = wave.open(path, "rb")
    sr = w.getframerate()
    n = w.getnframes()
    ch = w.getnchannels()
    a = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    w.close()
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != 16000:  # whisper wants 16 kHz; linear resample is fine for ASR
        m = int(round(a.size * 16000 / sr))
        a = np.interp(
            np.linspace(0, 1, m, endpoint=False),
            np.linspace(0, 1, a.size, endpoint=False),
            a,
        ).astype("float32")
    return a


def transcribe(path: str, model_name: str) -> list[dict]:
    from faster_whisper import WhisperModel

    audio = _load_wav_16k(path)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        audio, language="en", vad_filter=True, word_timestamps=True
    )
    out = []
    for s in segments:
        out.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Timestamped ground-truth transcript of a session WAV.")
    ap.add_argument("wavs", nargs="+", help="run-bundle .wav file(s) or a glob")
    ap.add_argument("--model", default="small.en", help="faster-whisper model (base.en|small.en|medium.en)")
    ap.add_argument("--json", action="store_true", help="emit JSON (machine-readable)")
    args = ap.parse_args()

    paths = []
    for w in args.wavs:
        paths.extend(sorted(glob.glob(w)) if any(c in w for c in "*?[") else [w])

    all_results = {}
    for path in paths:
        try:
            segs = transcribe(path, args.model)
        except Exception as e:  # noqa: BLE001
            print(f"# {path}: TRANSCRIBE ERROR {type(e).__name__}: {e}", file=sys.stderr)
            all_results[path] = {"error": f"{type(e).__name__}: {e}"}
            continue
        all_results[path] = segs
        if not args.json:
            print(f"\n=== {path} ===")
            if not segs:
                print("  (no speech detected)")
            for s in segs:
                print(f"  [{s['start']:6.2f}-{s['end']:6.2f}]  {s['text']}")
    if args.json:
        print(json.dumps(all_results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
