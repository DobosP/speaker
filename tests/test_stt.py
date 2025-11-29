"""
Unit tests for the Speech-to-Text module.
"""
import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stt import WhisperSTT, get_stt_model, transcribe_audio


class TestWhisperSTT(unittest.TestCase):
    """Test the Whisper STT model."""
    
    @classmethod
    def setUpClass(cls):
        """Load model once for all tests."""
        print("\n🔄 Loading Whisper model for tests (this may take a moment)...")
        cls.model = get_stt_model("openai/whisper-base")
    
    def test_singleton_pattern(self):
        """Should return the same instance (singleton)."""
        model1 = get_stt_model("openai/whisper-base")
        model2 = get_stt_model("openai/whisper-base")
        self.assertIs(model1, model2, "Should be the same instance")
    
    def test_model_initialized(self):
        """Model should be properly initialized."""
        self.assertTrue(self.model._initialized)
        self.assertIsNotNone(self.model.pipe)
    
    def test_transcribe_silence(self):
        """Should handle silence gracefully."""
        silence = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        result = self.model.transcribe(silence)
        # Silence might return empty string or some minimal output
        self.assertIsInstance(result, str)
    
    def test_transcribe_noise(self):
        """Should handle random noise gracefully."""
        noise = np.random.randn(16000).astype(np.float32) * 0.01
        result = self.model.transcribe(noise)
        self.assertIsInstance(result, str)
    
    def test_transcribe_empty_array(self):
        """Should handle empty array gracefully."""
        empty = np.array([], dtype=np.float32)
        result = self.model.transcribe(empty)
        self.assertEqual(result, "")
    
    def test_transcribe_audio_function(self):
        """Test the backward-compatible transcribe_audio function."""
        silence = np.zeros(8000, dtype=np.float32)
        result = transcribe_audio(silence, model_id="openai/whisper-base")
        self.assertIsInstance(result, str)
    
    def test_transcribe_synthetic_tone(self):
        """Should transcribe a synthetic tone without crashing."""
        # Generate a 440Hz sine wave (1 second)
        sample_rate = 16000
        t = np.linspace(0, 1, sample_rate, dtype=np.float32)
        tone = np.sin(2 * np.pi * 440 * t) * 0.3
        
        result = self.model.transcribe(tone)
        self.assertIsInstance(result, str)


class TestModelTypes(unittest.TestCase):
    """Test model type validation."""
    
    def test_invalid_model_type(self):
        """Should raise error for unsupported model type."""
        with self.assertRaises(ValueError):
            transcribe_audio(
                np.zeros(1000, dtype=np.float32),
                model_type="invalid_type"
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)

