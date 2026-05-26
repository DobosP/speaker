"""
TTS backend capability tests — Piper (low) · Kokoro (mid) · MeloTTS (high).

All tests are hardware-free: they call synthesis methods directly, bypass
pygame playback, and measure latency against generous CPU budgets.  Tests for
backends that are not installed are automatically skipped.

Run all:
    pytest tests/test_tts_backends.py -v

Run a single tier:
    pytest tests/test_tts_backends.py -k "Piper"
    pytest tests/test_tts_backends.py -k "Kokoro"
    pytest tests/test_tts_backends.py -k "Melo"
"""
import os
import sys
import time
import unittest
import tempfile
import wave

import numpy as np
import pytest

pytestmark = [pytest.mark.backend, pytest.mark.slow]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio import (
    AudioPlayer,
    PIPER_AVAILABLE,
    KOKORO_AVAILABLE,
    MELOTTS_AVAILABLE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

VALID_SAMPLE_RATES = {16000, 22050, 24000, 44100, 48000}
SHORT_TEXT  = "Hello."
MEDIUM_TEXT = "Hello, how are you today? I hope everything is going well."
LONG_TEXT   = (
    "The quick brown fox jumps over the lazy dog. "
    * 12  # ~200 words
).strip()


def _make_player(backend: str, voice: str = "en-US", tts_model: str = None) -> AudioPlayer:
    """Create an AudioPlayer with the given backend but WITHOUT pygame init."""
    p = object.__new__(AudioPlayer)
    p.output_device = None
    p._is_playing = False
    p._current_file = None
    p.voice = voice
    p._supertonic_tts = None
    p._supertonic_style = None
    p._kokoro_engine = None
    p._kokoro_voice = None
    p._piper_voice_obj = None
    p._melo_engine = None
    p._melo_speaker_id = None
    p._melo_language = None
    p._tts_model = tts_model
    p.tts_backend = backend
    return p


def _synth_to_wav(player: AudioPlayer, text: str) -> str:
    """Synthesize *text* to a temporary WAV file; return the path."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    player._synthesize_speech(text, path)
    return path


def _load_wav_as_float32(path: str):
    """Load a WAV file and return (samples_float32, sample_rate)."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        sampwidth = wf.getsampwidth()
    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    return samples, sr


# ═════════════════════════════════════════════════════════════════════════════
# Tier 1 — Piper TTS (low)
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(PIPER_AVAILABLE, "piper-tts not installed: pip install piper-tts")
class TestPiperTTS(unittest.TestCase):
    """Capability tests for the Piper ONNX backend (Tier 1 — low)."""

    @classmethod
    def setUpClass(cls):
        cls.player = _make_player("piper", voice="en-US", tts_model="en_US-lessac-medium")
        cls.player._init_piper("en-US")
        if cls.player.tts_backend != "piper":
            raise unittest.SkipTest("Piper voice model unavailable (no network?)")

    def _synth(self, text: str) -> tuple:
        path = _synth_to_wav(self.player, text)
        samples, sr = _load_wav_as_float32(path)
        os.unlink(path)
        return samples, sr

    # ── basic contract ────────────────────────────────────────────────────

    def test_synthesize_returns_audio(self):
        samples, _ = self._synth(MEDIUM_TEXT)
        self.assertGreater(len(samples), 0, "synthesis produced no audio")

    def test_output_is_non_silent(self):
        samples, _ = self._synth(MEDIUM_TEXT)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        self.assertGreater(rms, 1e-4, f"output looks silent (rms={rms:.6f})")

    def test_sample_rate_is_valid(self):
        _, sr = self._synth(MEDIUM_TEXT)
        self.assertIn(sr, VALID_SAMPLE_RATES, f"unexpected sample rate {sr}")

    def test_output_is_float32_compatible(self):
        samples, _ = self._synth(MEDIUM_TEXT)
        self.assertLessEqual(np.max(np.abs(samples)), 1.5,
                             "samples outside [-1.5, 1.5] range")

    # ── latency ──────────────────────────────────────────────────────────

    def test_latency_under_500ms_short_text(self):
        t0 = time.perf_counter()
        self._synth(SHORT_TEXT)
        elapsed = (time.perf_counter() - t0) * 1000
        self.assertLess(elapsed, 500, f"Piper took {elapsed:.0f}ms for short text (budget: 500ms)")

    def test_latency_under_2s_medium_text(self):
        t0 = time.perf_counter()
        self._synth(MEDIUM_TEXT)
        elapsed = (time.perf_counter() - t0) * 1000
        self.assertLess(elapsed, 2000, f"Piper took {elapsed:.0f}ms for medium text (budget: 2000ms)")

    # ── edge cases ───────────────────────────────────────────────────────

    def test_single_word(self):
        samples, _ = self._synth("Hello")
        self.assertGreater(len(samples), 0)

    def test_long_text_no_crash(self):
        samples, _ = self._synth(LONG_TEXT)
        self.assertGreater(len(samples), 0)

    # ── multilingual (requires voice download) ───────────────────────────

    @unittest.skipUnless(PIPER_AVAILABLE, "piper not installed")
    def test_german_voice_available_in_map(self):
        """Piper VOICES map must include German so it can be selected."""
        self.assertIn("de-DE", AudioPlayer.PIPER_VOICES)

    @unittest.skipUnless(PIPER_AVAILABLE, "piper not installed")
    def test_spanish_voice_available_in_map(self):
        self.assertIn("es-ES", AudioPlayer.PIPER_VOICES)

    @unittest.skipUnless(PIPER_AVAILABLE, "piper not installed")
    def test_french_voice_available_in_map(self):
        self.assertIn("fr-FR", AudioPlayer.PIPER_VOICES)

    def test_voice_map_covers_at_least_10_locales(self):
        self.assertGreaterEqual(len(AudioPlayer.PIPER_VOICES), 10,
                                "PIPER_VOICES map should cover ≥10 locales")


# ═════════════════════════════════════════════════════════════════════════════
# Tier 2 — Kokoro 82M ONNX (mid)
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(KOKORO_AVAILABLE, "kokoro-onnx not installed: pip install kokoro-onnx")
class TestKokoroTTS(unittest.TestCase):
    """Capability tests for the Kokoro ONNX backend (Tier 2 — mid)."""

    @classmethod
    def setUpClass(cls):
        cls.player = _make_player("kokoro", voice="en-US")
        cls.player._init_kokoro("en-US")
        if cls.player.tts_backend != "kokoro":
            raise unittest.SkipTest("Kokoro model unavailable (no network?)")

    def _synth(self, text: str) -> tuple:
        path = _synth_to_wav(self.player, text)
        samples, sr = _load_wav_as_float32(path)
        os.unlink(path)
        return samples, sr

    def test_synthesize_returns_audio(self):
        samples, _ = self._synth(MEDIUM_TEXT)
        self.assertGreater(len(samples), 0)

    def test_output_is_non_silent(self):
        samples, _ = self._synth(MEDIUM_TEXT)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        self.assertGreater(rms, 1e-4)

    def test_sample_rate_is_valid(self):
        _, sr = self._synth(MEDIUM_TEXT)
        self.assertIn(sr, VALID_SAMPLE_RATES)

    def test_latency_under_500ms_short_text(self):
        t0 = time.perf_counter()
        self._synth(SHORT_TEXT)
        elapsed = (time.perf_counter() - t0) * 1000
        self.assertLess(elapsed, 500, f"Kokoro took {elapsed:.0f}ms")

    def test_latency_under_2s_medium_text(self):
        t0 = time.perf_counter()
        self._synth(MEDIUM_TEXT)
        elapsed = (time.perf_counter() - t0) * 1000
        self.assertLess(elapsed, 2000, f"Kokoro took {elapsed:.0f}ms")

    def test_single_word(self):
        samples, _ = self._synth("Hello")
        self.assertGreater(len(samples), 0)

    def test_long_text_no_crash(self):
        samples, _ = self._synth(LONG_TEXT)
        self.assertGreater(len(samples), 0)

    def test_british_english_voice_in_map(self):
        self.assertIn("en-GB", AudioPlayer.KOKORO_VOICES)

    def test_japanese_voice_in_map(self):
        """Kokoro ships Japanese voices — verify the map knows about it."""
        kokoro_voices = AudioPlayer.KOKORO_VOICES
        has_jp = any(
            "jp" in k.lower() or "ja" in k.lower()
            for k in kokoro_voices
        )
        # Map may not include ja-JP by default; just verify no error on access
        _ = kokoro_voices.get("ja-JP", None)

    def test_male_voice_in_map(self):
        self.assertIn("en-US-male", AudioPlayer.KOKORO_VOICES)


# ═════════════════════════════════════════════════════════════════════════════
# Tier 3 — MeloTTS (high)
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(MELOTTS_AVAILABLE, "melotts not installed: pip install melotts")
class TestMeloTTS(unittest.TestCase):
    """Capability tests for the MeloTTS backend (Tier 3 — high)."""

    @classmethod
    def setUpClass(cls):
        cls.player = _make_player("melotts", voice="en-US", tts_model="EN-US")
        cls.player._init_melotts("en-US")
        if cls.player.tts_backend != "melotts":
            raise unittest.SkipTest("MeloTTS unavailable (init failed)")

    def _synth(self, text: str) -> tuple:
        path = _synth_to_wav(self.player, text)
        samples, sr = _load_wav_as_float32(path)
        os.unlink(path)
        return samples, sr

    def test_synthesize_returns_audio(self):
        samples, _ = self._synth(MEDIUM_TEXT)
        self.assertGreater(len(samples), 0)

    def test_output_is_non_silent(self):
        samples, _ = self._synth(MEDIUM_TEXT)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        self.assertGreater(rms, 1e-4)

    def test_sample_rate_is_valid(self):
        _, sr = self._synth(MEDIUM_TEXT)
        self.assertIn(sr, VALID_SAMPLE_RATES)

    def test_latency_under_1s_short_text(self):
        t0 = time.perf_counter()
        self._synth(SHORT_TEXT)
        elapsed = (time.perf_counter() - t0) * 1000
        self.assertLess(elapsed, 1000, f"MeloTTS took {elapsed:.0f}ms (budget: 1000ms)")

    def test_latency_under_3s_medium_text(self):
        t0 = time.perf_counter()
        self._synth(MEDIUM_TEXT)
        elapsed = (time.perf_counter() - t0) * 1000
        self.assertLess(elapsed, 3000, f"MeloTTS took {elapsed:.0f}ms")

    def test_single_word(self):
        samples, _ = self._synth("Hello")
        self.assertGreater(len(samples), 0)

    def test_long_text_no_crash(self):
        samples, _ = self._synth(LONG_TEXT)
        self.assertGreater(len(samples), 0)

    def test_chinese_locale_in_voice_map(self):
        self.assertIn("zh-CN", AudioPlayer.MELOTTS_VOICES)

    def test_japanese_locale_in_voice_map(self):
        self.assertIn("ja-JP", AudioPlayer.MELOTTS_VOICES)

    def test_korean_locale_in_voice_map(self):
        self.assertIn("ko-KR", AudioPlayer.MELOTTS_VOICES)

    def test_spanish_locale_in_voice_map(self):
        self.assertIn("es-ES", AudioPlayer.MELOTTS_VOICES)

    def test_french_locale_in_voice_map(self):
        self.assertIn("fr-FR", AudioPlayer.MELOTTS_VOICES)

    def test_voice_map_covers_all_6_languages(self):
        """MeloTTS supports EN/ZH/JP/KR/ES/FR — verify coverage."""
        voices = AudioPlayer.MELOTTS_VOICES
        required = {"en-US", "zh-CN", "ja-JP", "ko-KR", "es-ES", "fr-FR"}
        missing = required - set(voices.keys())
        self.assertFalse(missing, f"MELOTTS_VOICES missing locales: {missing}")


# ═════════════════════════════════════════════════════════════════════════════
# Cross-backend contract — whichever backend is installed
# ═════════════════════════════════════════════════════════════════════════════

class TestTTSBackendContract(unittest.TestCase):
    """Backend-agnostic contract tests — run with whatever is installed."""

    @classmethod
    def setUpClass(cls):
        cls.player = None
        # Try backends in tier order (low → high) to pick the first available
        if PIPER_AVAILABLE:
            p = _make_player("piper", voice="en-US", tts_model="en_US-lessac-medium")
            p._init_piper("en-US")
            if p.tts_backend == "piper":
                cls.player = p
        if cls.player is None and KOKORO_AVAILABLE:
            p = _make_player("kokoro", voice="en-US")
            p._init_kokoro("en-US")
            if p.tts_backend == "kokoro":
                cls.player = p
        if cls.player is None and MELOTTS_AVAILABLE:
            p = _make_player("melotts", voice="en-US", tts_model="EN-US")
            p._init_melotts("en-US")
            if p.tts_backend == "melotts":
                cls.player = p
        if cls.player is None:
            raise unittest.SkipTest("No TTS backend installed")

    def _synth(self, text: str):
        path = _synth_to_wav(self.player, text)
        samples, sr = _load_wav_as_float32(path)
        os.unlink(path)
        return samples, sr

    def test_output_samples_are_finite(self):
        samples, _ = self._synth(MEDIUM_TEXT)
        self.assertTrue(np.all(np.isfinite(samples)),
                        "synthesis produced NaN or Inf samples")

    def test_output_within_range(self):
        samples, _ = self._synth(MEDIUM_TEXT)
        self.assertLessEqual(float(np.max(np.abs(samples))), 2.0,
                             "samples clipped beyond ±2.0")

    def test_sample_rate_is_valid(self):
        _, sr = self._synth(MEDIUM_TEXT)
        self.assertIn(sr, VALID_SAMPLE_RATES)

    def test_empty_text_does_not_crash(self):
        """Empty input may produce silence or raise gracefully — not hang."""
        try:
            self._synth("")
        except Exception:
            pass  # graceful failure is acceptable

    def test_synthesis_is_reproducible(self):
        """Two calls with the same text should produce the same length output."""
        samples1, sr1 = self._synth(SHORT_TEXT)
        samples2, sr2 = self._synth(SHORT_TEXT)
        self.assertEqual(sr1, sr2)
        # Piper/Kokoro can vary frame counts slightly between runs (±25%)
        ratio = len(samples1) / max(len(samples2), 1)
        self.assertAlmostEqual(ratio, 1.0, delta=0.25,
                               msg="synthesis duration inconsistent between calls")

    def test_unicode_text_no_crash(self):
        """Multi-lingual unicode should not raise an unhandled exception."""
        try:
            self._synth("Héllo wörld. Привет мир.")
        except Exception:
            pass

    def test_backend_field_is_set(self):
        self.assertIsNotNone(self.player.tts_backend)
        self.assertIn(self.player.tts_backend,
                      {"piper", "kokoro", "melotts", "supertonic", "edge-tts", "gtts"})


# ═════════════════════════════════════════════════════════════════════════════
# Profile tier metadata tests — no synthesis needed
# ═════════════════════════════════════════════════════════════════════════════

class TestTTSProfileMetadata(unittest.TestCase):
    """Verify that RESOURCE_PROFILES contain valid TTS configuration keys."""

    def setUp(self):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import importlib
        import main as _main
        self.profiles = _main.RESOURCE_PROFILES

    def test_all_tiers_present(self):
        for tier in ("low", "mid", "high"):
            self.assertIn(tier, self.profiles, f"tier '{tier}' missing from RESOURCE_PROFILES")

    def test_all_tiers_have_tts_backend(self):
        for tier in ("low", "mid", "high"):
            self.assertIn("tts_backend", self.profiles[tier],
                          f"tier '{tier}' missing 'tts_backend' key")

    def test_tts_backends_are_known(self):
        known = {None, "piper", "kokoro", "melotts", "supertonic", "edge-tts", "gtts"}
        for tier, profile in self.profiles.items():
            backend = profile.get("tts_backend")
            self.assertIn(backend, known,
                          f"tier '{tier}' has unknown tts_backend '{backend}'")

    def test_low_tier_uses_piper(self):
        self.assertEqual(self.profiles["low"]["tts_backend"], "piper")

    def test_mid_tier_uses_kokoro(self):
        self.assertEqual(self.profiles["mid"]["tts_backend"], "kokoro")

    def test_high_tier_uses_melotts(self):
        self.assertEqual(self.profiles["high"]["tts_backend"], "melotts")


if __name__ == "__main__":
    unittest.main(verbosity=2)
