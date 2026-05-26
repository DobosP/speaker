"""
pytest session configuration: prepares all audio test fixtures.

Two categories of fixtures
--------------------------

1. **TTS-generated speech** (``tests/fixture_audio/``)
   Synthesised on first run using edge-tts or gTTS (requires network once).
   Represents the TTS assistant's voice leaking into the mic (echo reference)
   and the user's voice in a realistic barge-in scenario.

2. **Open-source human voice samples** (``tests/voice_samples/``)
   Short recordings from the Free Spoken Digit Dataset (FSDD):
     https://github.com/Jakobovski/free-spoken-digit-dataset
   MIT License.  Multiple real human speakers at 8 kHz, resampled to 16 kHz.
   Used for:
   - VAD accuracy contracts (Silero MUST classify these as voiced)
   - Barge-in tests that exercise the Silero path at low amplitude
     (amplitude kept low enough that the energy-only path CANNOT fire — only
      Silero can produce the scored frames needed)
   - Adversarial tests: real speech vs. noise at controlled SNR

Both categories are cached locally after first download/generation.  All
tests degrade gracefully when network is unavailable (TTS falls back to
synthetic; FSDD-dependent tests are skipped with a clear message).

Usage
-----
    python -m pytest tests/ -v          # first run: downloads + generates
    python -m pytest tests/ -v          # subsequent: uses cache, no network
"""

from __future__ import annotations

import asyncio
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixture_audio")

_PHRASES = {
    "barge_in": "Hey excuse me, I wanted to ask you something about that.",
    "tts_echo": (
        "Let me tell you about this topic. "
        "There are several important things to consider here."
    ),
    "wakeword_speech": "Yes, I need some help with a question please.",
}

_TARGET_SR = 16_000


# ── TTS backends ──────────────────────────────────────────────────────────────

def _generate_with_edge_tts(text: str, out_path: str) -> bool:
    """Generate speech with edge-tts (requires network). Returns True on success."""
    try:
        import edge_tts  # noqa: PLC0415
        import soundfile as sf  # noqa: PLC0415
        import librosa  # noqa: PLC0415

        tmp_path = out_path + ".tmp.wav"

        async def _run():
            communicate = edge_tts.Communicate(text, "en-US-JennyNeural")
            await communicate.save(tmp_path)

        asyncio.run(_run())

        if not os.path.exists(tmp_path):
            return False

        audio, sr = sf.read(tmp_path)
        audio = np.array(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != _TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=_TARGET_SR)

        sf.write(out_path, audio, _TARGET_SR, subtype="PCM_16")
        os.remove(tmp_path)
        return True
    except Exception as exc:
        warnings.warn(f"edge-tts fixture generation failed: {exc}")
        return False


def _generate_with_gtts(text: str, out_path: str) -> bool:
    """Generate speech with gTTS (requires network, uses ffmpeg for MP3→WAV)."""
    try:
        from gtts import gTTS  # noqa: PLC0415
        import soundfile as sf  # noqa: PLC0415
        import librosa  # noqa: PLC0415

        tmp_mp3 = out_path + ".tmp.mp3"
        gTTS(text, lang="en").save(tmp_mp3)

        if not os.path.exists(tmp_mp3):
            return False

        # soundfile 0.13+ supports MP3 via libmpg123
        try:
            audio, sr = sf.read(tmp_mp3)
        except Exception:
            os.remove(tmp_mp3)
            return False

        audio = np.array(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != _TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=_TARGET_SR)

        sf.write(out_path, audio, _TARGET_SR, subtype="PCM_16")
        os.remove(tmp_mp3)
        return True
    except Exception as exc:
        warnings.warn(f"gTTS fixture generation failed: {exc}")
        return False


# ── Session-level generation ──────────────────────────────────────────────────

def _ensure_fixture(name: str) -> bool:
    """Generate fixture WAV if not already cached.  Returns True if available."""
    out_path = os.path.join(FIXTURE_DIR, f"{name}.wav")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        return True  # cached — no network call needed

    text = _PHRASES[name]
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    if _generate_with_edge_tts(text, out_path):
        print(f"\n[fixtures] Generated {name}.wav via edge-tts")
        return True

    if _generate_with_gtts(text, out_path):
        print(f"\n[fixtures] Generated {name}.wav via gTTS")
        return True

    return False


# ── FSDD open-source voice samples ───────────────────────────────────────────
#
# Free Spoken Digit Dataset (FSDD)
# https://github.com/Jakobovski/free-spoken-digit-dataset
# MIT License — multiple real human speakers, ~8 kHz WAV, <30 KB each
#
# We download a small, curated set: 3 speakers × 3 digits = 9 files (~200 KB).
# Files are stored in tests/voice_samples/ and reused on subsequent runs.

VOICE_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "voice_samples")

_FSDD_BASE = (
    "https://raw.githubusercontent.com/Jakobovski/"
    "free-spoken-digit-dataset/master/recordings/"
)

# Expanded speaker/digit matrix for better voice variability coverage.
# Digits 1–3: consistent voiced energy.
# Additional takes (take _1): second recording of same speaker×digit for
# within-speaker variability tests.
_FSDD_SAMPLES = [
    # Jackson — American English male
    "1_jackson_0.wav", "2_jackson_0.wav", "3_jackson_0.wav",
    "1_jackson_1.wav", "2_jackson_1.wav",
    # Nicolas — different accent male
    "1_nicolas_0.wav", "2_nicolas_0.wav", "3_nicolas_0.wav",
    # George — third male speaker
    "1_george_0.wav",  "2_george_0.wav",  "3_george_0.wav",
    # Theo — fourth speaker
    "1_theo_0.wav",    "2_theo_0.wav",    "3_theo_0.wav",
    # Lucas — fifth speaker (available in FSDD)
    "1_lucas_0.wav",   "2_lucas_0.wav",   "3_lucas_0.wav",
    # Additional digits for phoneme variety (digit "0" has fricative onset)
    "0_jackson_0.wav", "0_nicolas_0.wav", "0_george_0.wav",
]

# How many consecutive samples to concatenate to create a >0.5 s sample
# (FSDD clips are ~0.25–0.55 s each at 8 kHz → 0.5–1.1 s at 16 kHz)
_FSDD_CONCAT_NAME = "human_voice_concat.wav"


def _download_fsdd_samples() -> int:
    """
    Download FSDD WAV samples.  Returns count of successfully cached files.
    Silently skips already-cached files.
    """
    import urllib.request  # noqa: PLC0415

    os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)
    downloaded = 0
    for fname in _FSDD_SAMPLES:
        out = os.path.join(VOICE_SAMPLES_DIR, fname)
        if os.path.exists(out) and os.path.getsize(out) > 200:
            downloaded += 1
            continue
        try:
            urllib.request.urlretrieve(_FSDD_BASE + fname, out)
            downloaded += 1
        except Exception as exc:
            warnings.warn(f"FSDD download failed for {fname}: {exc}")
    return downloaded


def _build_concat_sample() -> bool:
    """
    Concatenate several FSDD clips to produce a sample >1.0 s long.
    This gives barge-in tests enough audio to exceed the min_speech_sec gate.
    """
    try:
        import soundfile as sf  # noqa: PLC0415
        import librosa  # noqa: PLC0415

        out_path = os.path.join(VOICE_SAMPLES_DIR, _FSDD_CONCAT_NAME)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 4096:
            return True

        parts = []
        # Use first 6 from the original 4 speakers (varied speakers, not takes)
        concat_sources = [f for f in _FSDD_SAMPLES if f.endswith("_0.wav")][:6]
        for fname in concat_sources:  # 6 clips
            path = os.path.join(VOICE_SAMPLES_DIR, fname)
            if not os.path.exists(path):
                continue
            audio, sr = sf.read(path)
            audio = np.array(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != _TARGET_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=_TARGET_SR)
            # Add 80 ms pause between clips (realistic inter-word gap)
            parts.append(audio)
            parts.append(np.zeros(int(0.08 * _TARGET_SR), dtype=np.float32))

        if not parts:
            return False

        concat = np.concatenate(parts).astype(np.float32)
        sf.write(out_path, concat, _TARGET_SR, subtype="PCM_16")
        return True
    except Exception as exc:
        warnings.warn(f"Failed to build concat sample: {exc}")
        return False


# ── Session entry point ───────────────────────────────────────────────────────

def pytest_configure(config):
    """
    Called once at the very start of the pytest session.

    1. Generates TTS fixture WAVs (used by test_bargein_scenarios).
    2. Downloads FSDD open-source human voice samples (used by
       test_vad_accuracy and test_bargein_contracts).

    Both steps are best-effort: tests degrade gracefully when offline.
    """
    # ── 1. TTS fixtures ───────────────────────────────────────────────────────
    any_tts_ok = False
    for name in _PHRASES:
        try:
            if _ensure_fixture(name):
                any_tts_ok = True
        except Exception as exc:
            warnings.warn(f"TTS fixture '{name}' error: {exc}")

    if any_tts_ok:
        print("\n[fixtures] TTS speech fixtures ready.")
    else:
        print("\n[fixtures] TTS fixtures unavailable — falling back to synthetic.")

    # ── 2. FSDD human voice samples ───────────────────────────────────────────
    try:
        n = _download_fsdd_samples()
        if n > 0:
            _build_concat_sample()
            print(
                f"[fixtures] {n}/{len(_FSDD_SAMPLES)} FSDD voice samples ready "
                "(real human speech — Silero accuracy tests enabled)."
            )
        else:
            print(
                "[fixtures] FSDD download failed — VAD accuracy tests will be skipped."
            )
    except Exception as exc:
        warnings.warn(f"FSDD setup error: {exc}")
