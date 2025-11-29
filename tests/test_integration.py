"""
Integration tests for the Voice Assistant.
"""
import unittest
import numpy as np
import threading
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio import AudioRecorder, AudioPlayer
from utils.stt import get_stt_model


class TestAudioSTTIntegration(unittest.TestCase):
    """Test audio recording + STT integration."""
    
    @classmethod
    def setUpClass(cls):
        """Load model once for all tests."""
        cls.model = get_stt_model("openai/whisper-base")
    
    def test_recorded_audio_can_be_transcribed(self):
        """Audio format from recorder should be compatible with STT."""
        # Simulate what the recorder produces
        simulated_audio = np.random.randn(16000 * 2).astype(np.float32) * 0.01
        
        # This should not crash
        result = self.model.transcribe(simulated_audio)
        self.assertIsInstance(result, str)
    
    def test_barge_in_workflow(self):
        """Test the barge-in workflow for interrupting assistant."""
        received = []
        interrupts = []
        
        def callback(audio):
            received.append(audio)
        
        def on_interrupt():
            interrupts.append(True)
        
        recorder = AudioRecorder(
            callback=callback,
            vad_threshold=0.5,
            on_interrupt=on_interrupt
        )
        player = AudioPlayer()
        
        try:
            recorder.start()
            
            # Simulate assistant speaking
            recorder.set_assistant_speaking(True)
            self.assertTrue(recorder.assistant_is_speaking)
            
            # In barge-in mode, recorder should still be active
            self.assertTrue(recorder.is_recording)
            
            # Simulate assistant done speaking
            recorder.set_assistant_speaking(False)
            self.assertFalse(recorder.assistant_is_speaking)
            
            recorder.stop()
        except Exception as e:
            recorder.stop()
            if "PortAudio" in str(e):
                self.skipTest(f"No audio device: {e}")
            raise
        finally:
            player.cleanup()


class TestFullPipeline(unittest.TestCase):
    """Test the full pipeline without actual audio."""
    
    def test_synthetic_speech_pipeline(self):
        """Test transcribing synthetic audio."""
        model = get_stt_model("openai/whisper-base")
        
        # Generate synthetic "speech-like" audio (modulated noise)
        sample_rate = 16000
        duration = 2  # seconds
        t = np.linspace(0, duration, sample_rate * duration, dtype=np.float32)
        
        # Amplitude-modulated noise (simulates speech envelope)
        envelope = np.abs(np.sin(2 * np.pi * 3 * t))  # 3Hz modulation
        audio = np.random.randn(len(t)).astype(np.float32) * envelope * 0.1
        
        result = model.transcribe(audio)
        self.assertIsInstance(result, str)
        print(f"\nSynthetic audio transcription: '{result}'")


if __name__ == '__main__':
    unittest.main(verbosity=2)

