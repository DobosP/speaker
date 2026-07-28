"""Tests for tools.replay_barge -- a silent (effectively all-zero) playback
reference must never read like a clean/evaluated run (see tools/diagnose_run.py
for the twin fix on the post-hoc log analyzer; both tools share the same
"silent .ref.wav" definition via ``tools.diagnose_run._ref_wav_is_silent``).

No live mic/model weights required: scipy + the checked-in config.json (already
a runtime dependency of the tool itself) are the only prerequisites.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.replay_barge import main


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> str:
    import wave

    pcm = np.clip(samples, -1.0, 1.0)
    data = (pcm * 32767.0).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(data)
    return str(path)


def _write_pair(tmp_path: Path, *, ref_samples: np.ndarray, mic_samples=None, sr: int = 16000) -> str:
    mic_samples = ref_samples if mic_samples is None else mic_samples
    mic_path = tmp_path / "run-test.wav"
    _write_wav(mic_path, mic_samples, sr)
    _write_wav(tmp_path / "run-test.ref.wav", ref_samples, sr)
    return str(mic_path)


def _tone(seconds: float, sr: int = 16000, freq: float = 440.0, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Silent (broken-capture) reference
# ---------------------------------------------------------------------------

def test_silent_ref_wav_is_flagged_and_returns_nonzero(tmp_path, capsys):
    ref = np.zeros(16000 * 5, dtype=np.float32)  # all-zero: the SUITE bundle defect
    mic_wav = _write_pair(tmp_path, ref_samples=ref, mic_samples=_tone(5.0))

    rc = main([mic_wav])
    out = capsys.readouterr()

    assert rc == 3
    combined = out.out + out.err
    assert "silent" in combined.lower()
    assert "echo attribution impossible" in combined
    assert "REF-SILENT" in out.out
    # A silent-ref run must not print the normal clean-looking summary line.
    assert "REMAINING after grace=" not in out.out


def test_near_zero_ref_wav_also_flagged(tmp_path, capsys):
    # Not bit-for-bit zero (e.g. dither/quantization noise from a broken
    # capture path) but still far below any real signal -- must still trip
    # the guard, not read as "very quiet but real".
    rng = np.random.default_rng(0)
    ref = (rng.standard_normal(16000 * 3) * 1e-6).astype(np.float32)
    mic_wav = _write_pair(tmp_path, ref_samples=ref, mic_samples=_tone(3.0))

    rc = main([mic_wav])
    out = capsys.readouterr()

    assert rc == 3
    assert "REF-SILENT" in out.out


# ---------------------------------------------------------------------------
# Real reference -- must behave exactly as before this fix
# ---------------------------------------------------------------------------

def test_real_ref_wav_runs_normally(tmp_path, capsys):
    ref = _tone(3.0, amp=0.25)
    mic_wav = _write_pair(tmp_path, ref_samples=ref, mic_samples=ref.copy())

    rc = main([mic_wav])
    out = capsys.readouterr()

    assert rc == 0
    assert "REMAINING after grace=" in out.out
    assert "silent" not in out.out.lower()
    assert "REF-SILENT" not in out.out


# ---------------------------------------------------------------------------
# Pre-existing missing-reference behavior (regression guard)
# ---------------------------------------------------------------------------

def test_missing_ref_wav_returns_two(tmp_path, capsys):
    mic_wav = tmp_path / "run-nosib.wav"
    _write_wav(mic_wav, _tone(1.0))
    # No sibling run-nosib.ref.wav written.

    rc = main([str(mic_wav)])
    out = capsys.readouterr()

    assert rc == 2
    assert "no reference WAV" in out.err
