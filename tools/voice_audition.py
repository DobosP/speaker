"""Headless objective pre-screen for configured Kokoro/VITS named voices.

The 2026-06-22 voice-swap plan (``docs/voice_upgrade_plan.md`` P1) flagged the
shipped named-voice set as PROVISIONAL: the model is Kokoro v1.1-**zh** (a
multi-lang package), "order undocumented -- some sids may have a non-English
timbre", and asked the owner to "finalize the voice set by ear" -- a step that
was never picked up. This tool does not replace that listening pass (no DSP
metric can judge "sounds robotic" or "non-English timbre"), but it removes the
"listen to all N voices blind" tax: it synthesizes ONE fixed sentence through
the REAL production pipeline (``SherpaOnnxEngine._synthesize`` -- the exact
declick/leveler/lowpass chain this config actually applies, not a re-implementation)
once per configured voice, writes a WAV per voice, and reports objective
peak/RMS/clip/DC/HF-ratio/spectral-flatness/centroid so the owner can prioritize
which voices to listen to first.

Usage::

    python -m tools.voice_audition
    python -m tools.voice_audition --voices warm,soft --text "Hello there."
    python -m tools.voice_audition --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from pathlib import Path
from typing import Optional

DEFAULT_TEXT = (
    "The quick brown fox jumps over the lazy dog, while the old church bell "
    "rings twelve times near the harbor."
)

# A voice whose synthesized clip looks unusually noise-like vs normal speech
# (see audio_quality_metrics' spectral_flatness docstring) gets flagged first.
_FLATNESS_FLAG_THRESHOLD = 0.35


def select_voices(tts_speaker_voices, default_sid: int, requested: Optional[list] = None) -> list:
    """Build an ordered ``(name, sid, directives)`` list to audition.

    Always includes ``("default", default_sid, None)`` as a baseline.
    ``requested`` (a list of names), when given, narrows to just those names
    (plus 'default'); ``None`` auditions every configured named voice. Pure /
    no I/O -- unit-testable without the real model. An out-of-range/bad sid in
    config is skipped rather than raising (the table just has fewer rows)."""
    rows = [("default", int(default_sid), None)]
    for name, sid in (tts_speaker_voices or {}).items():
        if requested is not None and name not in requested:
            continue
        try:
            rows.append((str(name), int(sid), {"voice": str(name)}))
        except (TypeError, ValueError):
            continue
    return rows


def spectral_centroid(samples, sr: int) -> Optional[float]:
    """Energy-weighted mean frequency (Hz); ``None`` for a silent/empty clip.

    The brightness metric the voice-swap doc used by ear (Kokoro ~1.8-2.9 kHz
    vs the dark legacy VITS ~0.8 kHz). Lives here rather than in
    ``core.audio_frontend`` because it is an audition/comparison metric, not
    something the production pipeline consumes."""
    import numpy as np

    x = np.asarray(samples, dtype="float32").reshape(-1)
    if x.size == 0:
        return None
    mag = np.abs(np.fft.rfft(x.astype("float64")))
    total = float(np.sum(mag))
    if total <= 1e-9:
        return None
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    return float(np.sum(freqs * mag) / total)


_COLUMNS = (
    "name", "sid", "rms", "peak", "clip_pct", "dc_offset",
    "hf_ratio", "spectral_flatness", "centroid_hz",
)


def format_table(rows: list) -> str:
    """Pure string formatting (unit-testable without the real model)."""
    if not rows:
        return "(no voices auditioned)"
    widths = {c: max(len(c), *(len(str(r.get(c))) for r in rows)) for c in _COLUMNS}
    header = "  ".join(c.rjust(widths[c]) for c in _COLUMNS)
    lines = [header]
    for r in rows:
        lines.append("  ".join(str(r.get(c)).rjust(widths[c]) for c in _COLUMNS))
        lines.append(f"  -> {r.get('wav')}")
    return "\n".join(lines)


def _write_wav(path: Path, samples, sr: int) -> None:
    import numpy as np

    x = np.asarray(samples, dtype="float32").reshape(-1)
    pcm16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype("int16")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm16.tobytes())


def _build_engine_or_die(device: Optional[str]):
    from core.config import apply_device_profile, load_config, resolve_device
    from core.engines._sherpa_models import build_tts
    from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine

    config = load_config()
    requested = device or config.get("device", "auto")
    resolved, rationale = resolve_device(config, requested)
    if rationale:
        print(f"[device] auto-selected: {resolved} ({rationale})", file=sys.stderr)
    config = apply_device_profile(config, resolved)
    scfg = SherpaConfig.from_dict(config.get("sherpa", {}))
    if not scfg.tts_model or not os.path.exists(scfg.tts_model):
        print(
            "ERROR: no TTS model configured/found "
            f"(sherpa.tts_model={scfg.tts_model!r}). Set it in config.local.json "
            "(see docs/voice_upgrade_plan.md) before running the audition.",
            file=sys.stderr,
        )
        return None
    tts = build_tts(scfg)
    if tts is None:
        print("ERROR: build_tts() returned None despite a configured tts_model.", file=sys.stderr)
        return None
    eng = SherpaOnnxEngine(scfg)
    eng._tts = tts
    return eng


def run_audition(eng, text: str, voices: list, out_dir: Path) -> list:
    """Synthesize ``text`` through the REAL engine pipeline once per voice and
    return one metrics row per voice. Writes one WAV per voice into
    ``out_dir`` so the owner can listen; never deletes anything.

    Resets the engine's carried loudness-gain state before each voice so every
    voice is leveled as if it were the FIRST utterance of a session (an
    independent, comparable measurement) instead of slewing from whatever the
    previous voice in the loop happened to measure."""
    import numpy as np

    from core.audio_frontend import audio_quality_metrics

    out_dir.mkdir(parents=True, exist_ok=True)
    sr = int(getattr(eng._tts, "sample_rate", 0) or 24000)
    rows = []
    for name, sid, directives in voices:
        eng._tts_level_gain_db = None
        eng._tts_normalize_gain = None
        written: list = []
        eng._synthesize(text, written.append, directives=directives)
        samples = np.concatenate(written) if written else np.zeros(0, dtype="float32")
        metrics = audio_quality_metrics(samples, sr)
        centroid = spectral_centroid(samples, sr)
        wav_path = out_dir / f"{name}_sid{sid}.wav"
        _write_wav(wav_path, samples, sr)
        rows.append({
            "name": name,
            "sid": sid,
            "centroid_hz": round(centroid, 1) if centroid is not None else None,
            "wav": str(wav_path),
            **metrics,
        })
    return rows


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--device", default=None, help="device profile (default: config.device / auto)")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="sentence to synthesize for every voice")
    parser.add_argument(
        "--voices", default=None,
        help="comma-separated subset of configured voice names (default: every configured voice)",
    )
    parser.add_argument(
        "--out-dir", default="logs/voice_audition",
        help="directory for the per-voice WAVs (default: logs/voice_audition, gitignored)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args(argv)

    eng = _build_engine_or_die(args.device)
    if eng is None:
        return 1
    requested = [v.strip() for v in args.voices.split(",") if v.strip()] if args.voices else None
    voices = select_voices(eng.config.tts_speaker_voices, eng.config.tts_speaker_id, requested)
    if len(voices) <= 1:
        print(
            "WARNING: no named voices configured (sherpa.tts_speaker_voices is "
            "empty) -- only the bare default sid was auditioned.",
            file=sys.stderr,
        )
    rows = run_audition(eng, args.text, voices, Path(args.out_dir))

    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0

    print(format_table(rows))
    flagged = [r for r in rows if (r.get("spectral_flatness") or 0) >= _FLATNESS_FLAG_THRESHOLD]
    if flagged:
        print("\nListen to these first (highest spectral flatness = most noise-like):")
        for r in sorted(flagged, key=lambda r: -(r.get("spectral_flatness") or 0)):
            print(f"  {r['name']} (sid={r['sid']}): flatness={r['spectral_flatness']} -> {r['wav']}")
    print(
        "\nThis is a pre-screen, not a verdict -- 'robotic' / 'non-English "
        "timbre' is judged by ear. Listen to every WAV above before picking a "
        "final voice set (docs/voice_upgrade_plan.md P1)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
