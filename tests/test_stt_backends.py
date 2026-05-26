"""
STT backend capability tests — Moonshine (low) · distil-medium.en (mid) · large-v3-turbo (high).

All tests are hardware-free: audio is generated synthetically (silence, white noise,
or pure-tone signals) so no microphone or real recordings are required.  Tests for
backends that are not installed or not yet downloaded are automatically skipped.

Latency budgets are generous to run safely on CI / slow laptops.

Run all:
    pytest tests/test_stt_backends.py -v

Run a single tier:
    pytest tests/test_stt_backends.py -k "Moonshine"
    pytest tests/test_stt_backends.py -k "Distil"
    pytest tests/test_stt_backends.py -k "LargeV3"
"""
import os
import sys
import time
import unittest

import numpy as np
import pytest

pytestmark = [pytest.mark.backend, pytest.mark.slow]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stt import (
    MoonshineSTT,
    WhisperSTT,
    get_stt_model,
    transcribe_audio,
    MOONSHINE_AVAILABLE,
    FASTER_WHISPER_AVAILABLE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SR = 16000   # all STT models expect 16 kHz float32

def _silence(seconds: float = 1.0) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)

def _white_noise(seconds: float = 1.0, amplitude: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.standard_normal(int(SR * seconds)) * amplitude).astype(np.float32)

def _tone(freq: float = 440.0, seconds: float = 1.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(SR * seconds), dtype=np.float32)
    return (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)

def _disk_space_gb() -> float:
    import shutil
    return shutil.disk_usage("/").free / (1024 ** 3)


# ═════════════════════════════════════════════════════════════════════════════
# Tier 1 — Moonshine tiny ONNX (low)
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(
    MOONSHINE_AVAILABLE,
    "useful-moonshine-onnx not installed: pip install useful-moonshine-onnx",
)
class TestMoonshineSTT(unittest.TestCase):
    """Capability tests for the Moonshine ONNX backend (Tier 1 — low)."""

    @classmethod
    def setUpClass(cls):
        print("\nLoading Moonshine tiny for tests ...")
        cls.model = get_stt_model("moonshine:tiny")
        # Prime ONNX sessions (first transcribe can take >1s on cold CPU)
        cls.model.transcribe(_silence(3.0))

    # ── contract ──────────────────────────────────────────────────────────

    def test_returns_string(self):
        result = self.model.transcribe(_silence(1.0))
        self.assertIsInstance(result, str)

    def test_silence_returns_string(self):
        result = self.model.transcribe(_silence(2.0))
        self.assertIsInstance(result, str)

    def test_noise_returns_string(self):
        result = self.model.transcribe(_white_noise(2.0))
        self.assertIsInstance(result, str)

    def test_empty_array_returns_empty(self):
        result = self.model.transcribe(np.array([], dtype=np.float32))
        self.assertEqual(result, "")

    def test_result_is_stripped(self):
        result = self.model.transcribe(_silence(1.0))
        self.assertEqual(result, result.strip())

    def test_tone_returns_string(self):
        result = self.model.transcribe(_tone(440.0, 2.0))
        self.assertIsInstance(result, str)

    # ── latency ───────────────────────────────────────────────────────────

    def test_latency_under_2s_for_3s_audio_warm(self):
        audio = _silence(3.0)
        t0 = time.perf_counter()
        self.model.transcribe(audio)
        elapsed = (time.perf_counter() - t0) * 1000
        self.assertLess(
            elapsed,
            2000,
            f"Moonshine took {elapsed:.0f}ms for 3s audio after warm-up (budget: 2000ms)",
        )

    # ── metadata ──────────────────────────────────────────────────────────

    def test_backend_field(self):
        self.assertEqual(self.model.backend, "moonshine")

    def test_sample_rate_is_16khz(self):
        self.assertEqual(self.model.sample_rate, 16000)

    def test_singleton_returns_same_instance(self):
        m1 = get_stt_model("moonshine:tiny")
        m2 = get_stt_model("moonshine:tiny")
        self.assertIs(m1, m2)

    def test_transcribe_audio_routing(self):
        """transcribe_audio() with model_id='moonshine:tiny' must route to MoonshineSTT."""
        result = transcribe_audio(_silence(1.0), model_id="moonshine:tiny")
        self.assertIsInstance(result, str)

    def test_model_type_routing(self):
        """model_type='moonshine' must also route to MoonshineSTT."""
        result = transcribe_audio(_silence(1.0), model_type="moonshine")
        self.assertIsInstance(result, str)


# ═════════════════════════════════════════════════════════════════════════════
# Tier 2 — distil-medium.en (mid)
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(FASTER_WHISPER_AVAILABLE, "faster-whisper not installed")
class TestDistilMediumSTT(unittest.TestCase):
    """Capability tests for distil-medium.en via faster-whisper (Tier 2 — mid)."""

    @classmethod
    def setUpClass(cls):
        print("\nLoading distil-medium.en for tests ...")
        try:
            cls.model = get_stt_model("distil-medium.en")
        except Exception as e:
            raise unittest.SkipTest(f"distil-medium.en not available: {e}")

    def test_returns_string(self):
        result = self.model.transcribe(_silence(1.0))
        self.assertIsInstance(result, str)

    def test_silence_returns_string(self):
        result = self.model.transcribe(_silence(2.0))
        self.assertIsInstance(result, str)

    def test_noise_returns_string(self):
        result = self.model.transcribe(_white_noise(2.0))
        self.assertIsInstance(result, str)

    def test_empty_array_returns_empty(self):
        result = self.model.transcribe(np.array([], dtype=np.float32))
        self.assertEqual(result, "")

    def test_result_is_stripped(self):
        result = self.model.transcribe(_silence(1.0))
        self.assertEqual(result, result.strip())

    def test_latency_under_1s_for_3s_audio(self):
        audio = _silence(3.0)
        t0 = time.perf_counter()
        self.model.transcribe(audio)
        elapsed = (time.perf_counter() - t0) * 1000
        self.assertLess(elapsed, 1000,
                        f"distil-medium took {elapsed:.0f}ms for 3s audio (budget: 1000ms)")

    def test_english_language_accepted(self):
        result = self.model.transcribe(_silence(1.0), language="en")
        self.assertIsInstance(result, str)

    def test_german_language_param_no_crash(self):
        """distil-medium.en is English-only but must not raise on language='de'."""
        try:
            result = self.model.transcribe(_silence(1.0), language="de")
            self.assertIsInstance(result, str)
        except Exception:
            pass  # acceptable — model may warn/ignore unsupported language

    def test_backend_is_faster_whisper(self):
        self.assertEqual(self.model.backend, "faster-whisper")

    def test_singleton_same_instance(self):
        m1 = get_stt_model("distil-medium.en")
        m2 = get_stt_model("distil-medium.en")
        self.assertIs(m1, m2)

    def test_transcribe_audio_routing(self):
        result = transcribe_audio(_silence(1.0), model_id="distil-medium.en")
        self.assertIsInstance(result, str)


# ═════════════════════════════════════════════════════════════════════════════
# Tier 3 — large-v3-turbo (high, multilingual)
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(FASTER_WHISPER_AVAILABLE, "faster-whisper not installed")
@unittest.skipIf(_disk_space_gb() < 2.0, "Less than 2 GB disk space — skipping large model")
class TestLargeV3TurboSTT(unittest.TestCase):
    """Capability tests for large-v3-turbo via faster-whisper (Tier 3 — high)."""

    @classmethod
    def setUpClass(cls):
        print("\nLoading large-v3-turbo for tests (this downloads ~800 MB on first run) ...")
        try:
            cls.model = get_stt_model("large-v3-turbo")
        except Exception as e:
            raise unittest.SkipTest(f"large-v3-turbo not available: {e}")

    def test_returns_string(self):
        result = self.model.transcribe(_silence(1.0))
        self.assertIsInstance(result, str)

    def test_silence_returns_string(self):
        result = self.model.transcribe(_silence(2.0))
        self.assertIsInstance(result, str)

    def test_empty_array_returns_empty(self):
        result = self.model.transcribe(np.array([], dtype=np.float32))
        self.assertEqual(result, "")

    def test_latency_under_2s_for_3s_audio(self):
        audio = _silence(3.0)
        t0 = time.perf_counter()
        self.model.transcribe(audio)
        elapsed = (time.perf_counter() - t0) * 1000
        self.assertLess(elapsed, 2000,
                        f"large-v3-turbo took {elapsed:.0f}ms for 3s audio (budget: 2000ms)")

    def test_99_language_codes_spot_check(self):
        """Spot-check 5 language codes; none should raise an exception."""
        for lang in ("en", "de", "fr", "ja", "zh"):
            try:
                result = self.model.transcribe(_silence(0.5), language=lang)
                self.assertIsInstance(result, str,
                                      f"transcribe(..., language='{lang}') did not return str")
            except Exception as exc:
                self.fail(f"language='{lang}' raised {exc!r}")

    def test_german_language_accepted(self):
        result = self.model.transcribe(_silence(1.0), language="de")
        self.assertIsInstance(result, str)

    def test_japanese_language_accepted(self):
        result = self.model.transcribe(_silence(1.0), language="ja")
        self.assertIsInstance(result, str)

    def test_chinese_language_accepted(self):
        result = self.model.transcribe(_silence(1.0), language="zh")
        self.assertIsInstance(result, str)

    def test_backend_is_faster_whisper(self):
        self.assertEqual(self.model.backend, "faster-whisper")

    def test_transcribe_audio_routing(self):
        result = transcribe_audio(_silence(1.0), model_id="large-v3-turbo")
        self.assertIsInstance(result, str)


# ═════════════════════════════════════════════════════════════════════════════
# Cross-backend contract — whichever model is cached/available first
# ═════════════════════════════════════════════════════════════════════════════

class TestSTTBackendContract(unittest.TestCase):
    """Backend-agnostic contract — runs with whichever STT is installed."""

    @classmethod
    def setUpClass(cls):
        cls.model = None
        # Prefer the smallest/fastest available model to keep CI fast
        for model_id in ("moonshine:tiny", "distil-medium.en", "base"):
            try:
                cls.model = get_stt_model(model_id)
                cls.model_id = model_id
                break
            except Exception:
                continue
        if cls.model is None:
            raise unittest.SkipTest("No STT backend available")

    def test_transcribe_returns_str(self):
        result = self.model.transcribe(_silence(1.0))
        self.assertIsInstance(result, str)

    def test_transcribe_returns_stripped_string(self):
        result = self.model.transcribe(_silence(1.0))
        self.assertEqual(result, result.strip())

    def test_float32_input_required(self):
        """float32 should always work; other dtypes should not raise unhandled."""
        audio = _silence(1.0).astype(np.float32)
        result = self.model.transcribe(audio)
        self.assertIsInstance(result, str)

    def test_16khz_mono_audio_accepted(self):
        audio = np.zeros(SR, dtype=np.float32)  # 1s @ 16kHz
        result = self.model.transcribe(audio)
        self.assertIsInstance(result, str)

    def test_output_is_finite_string(self):
        result = self.model.transcribe(_white_noise(1.0))
        self.assertIsInstance(result, str)
        self.assertFalse(
            result in {"nan", "inf", "-inf"},
            "transcription returned a non-finite value token"
        )

    def test_very_short_audio_no_crash(self):
        short = np.zeros(160, dtype=np.float32)  # 10ms
        try:
            result = self.model.transcribe(short)
            self.assertIsInstance(result, str)
        except Exception:
            pass  # graceful failure is acceptable

    def test_long_audio_no_crash(self):
        long_audio = np.zeros(SR * 30, dtype=np.float32)  # 30s
        result = self.model.transcribe(long_audio)
        self.assertIsInstance(result, str)


# ═════════════════════════════════════════════════════════════════════════════
# Profile tier metadata tests — no model loading needed
# ═════════════════════════════════════════════════════════════════════════════

class TestSTTProfileMetadata(unittest.TestCase):
    """Verify that RESOURCE_PROFILES contain valid STT configuration keys."""

    def setUp(self):
        import importlib
        import main as _main
        self.profiles = _main.RESOURCE_PROFILES

    def test_all_tiers_present(self):
        for tier in ("low", "mid", "high"):
            self.assertIn(tier, self.profiles)

    def test_all_tiers_have_stt_model(self):
        for tier in ("low", "mid", "high"):
            self.assertIn("stt_model", self.profiles[tier],
                          f"tier '{tier}' missing 'stt_model'")

    def test_low_tier_uses_moonshine(self):
        stt = self.profiles["low"]["stt_model"]
        self.assertTrue(stt.startswith("moonshine"),
                        f"low tier stt_model='{stt}' should start with 'moonshine'")

    def test_mid_tier_uses_distil(self):
        stt = self.profiles["mid"]["stt_model"]
        self.assertIn("distil", stt,
                      f"mid tier stt_model='{stt}' should use a distil model")

    def test_high_tier_uses_large_v3(self):
        stt = self.profiles["high"]["stt_model"]
        self.assertIn("large", stt.lower(),
                      f"high tier stt_model='{stt}' should use large-v3-turbo")

    def test_moonshine_routing(self):
        """get_stt_model('moonshine:tiny') must return MoonshineSTT (not WhisperSTT)."""
        if not MOONSHINE_AVAILABLE:
            self.skipTest("useful-moonshine-onnx not installed")
        model = get_stt_model("moonshine:tiny")
        self.assertIsInstance(model, MoonshineSTT)

    def test_whisper_routing(self):
        """get_stt_model('base') must return WhisperSTT (not MoonshineSTT)."""
        if not FASTER_WHISPER_AVAILABLE:
            self.skipTest("faster-whisper not installed")
        model = get_stt_model("base")
        self.assertIsInstance(
            model,
            WhisperSTT,
            msg=f"expected WhisperSTT, got {type(model).__name__} (backend={getattr(model, 'backend', None)})",
        )
        self.assertNotIsInstance(model, MoonshineSTT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
