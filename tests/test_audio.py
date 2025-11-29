"""
Unit tests for the cross-platform audio module.
"""
import unittest
import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio import (
    AudioRecorder,
    AudioPlayer,
    list_audio_devices,
    get_default_input_device,
    get_default_output_device,
)


class TestAudioDeviceDetection(unittest.TestCase):
    """Test audio device detection works on any platform."""
    
    def test_list_devices_returns_without_error(self):
        """Should list devices without crashing."""
        try:
            devices = list_audio_devices()
            self.assertIsNotNone(devices)
        except Exception as e:
            self.fail(f"list_audio_devices() raised an exception: {e}")
    
    def test_get_default_input_device(self):
        """Should return a valid device index or None."""
        try:
            device = get_default_input_device()
            # Device can be None if no input device, or an integer
            self.assertTrue(device is None or isinstance(device, (int, type(None))))
        except Exception as e:
            self.fail(f"get_default_input_device() raised: {e}")
    
    def test_get_default_output_device(self):
        """Should return a valid device index or None."""
        try:
            device = get_default_output_device()
            self.assertTrue(device is None or isinstance(device, (int, type(None))))
        except Exception as e:
            self.fail(f"get_default_output_device() raised: {e}")


class TestAudioRecorder(unittest.TestCase):
    """Test the AudioRecorder class."""
    
    def setUp(self):
        self.received_audio = []
        self.callback_called = False
    
    def audio_callback(self, audio_data):
        self.callback_called = True
        self.received_audio.append(audio_data)
    
    def test_recorder_initialization(self):
        """Should initialize without errors."""
        recorder = AudioRecorder(callback=self.audio_callback)
        self.assertIsNotNone(recorder)
        self.assertFalse(recorder.is_recording)
        self.assertFalse(recorder.is_paused)
    
    def test_recorder_start_stop(self):
        """Should start and stop recording without errors."""
        recorder = AudioRecorder(callback=self.audio_callback)
        
        try:
            recorder.start()
            self.assertTrue(recorder.is_recording)
            time.sleep(0.5)  # Record briefly
            recorder.stop()
            self.assertFalse(recorder.is_recording)
        except Exception as e:
            recorder.stop()
            # Skip if no audio device available (e.g., CI environment)
            if "No Default Input Device" in str(e) or "PortAudio" in str(e):
                self.skipTest(f"No audio device available: {e}")
            self.fail(f"Start/stop raised: {e}")
    
    def test_recorder_pause_resume(self):
        """Should pause and resume without errors."""
        recorder = AudioRecorder(callback=self.audio_callback)
        
        try:
            recorder.start()
            recorder.pause()
            self.assertTrue(recorder.is_paused)
            recorder.resume()
            self.assertFalse(recorder.is_paused)
            recorder.stop()
        except Exception as e:
            recorder.stop()
            # Skip if no audio device available
            if "No Default Input Device" in str(e) or "PortAudio" in str(e):
                self.skipTest(f"No audio device available: {e}")
            self.fail(f"Pause/resume raised: {e}")
    
    def test_recorder_with_custom_threshold(self):
        """Should accept custom VAD threshold."""
        recorder = AudioRecorder(
            callback=self.audio_callback,
            vad_threshold=0.05,
            silence_duration=2.0,
        )
        self.assertEqual(recorder.vad_threshold, 0.05)
        self.assertEqual(recorder.silence_duration, 2.0)
    
    def test_recorder_detects_device_sample_rate(self):
        """Should detect device's native sample rate."""
        recorder = AudioRecorder(callback=self.audio_callback)
        # Device sample rate should be a positive integer
        self.assertIsInstance(recorder.device_sample_rate, int)
        self.assertGreater(recorder.device_sample_rate, 0)
        # Common sample rates
        common_rates = [8000, 16000, 22050, 44100, 48000, 96000]
        self.assertIn(recorder.device_sample_rate, common_rates,
                     f"Unusual sample rate: {recorder.device_sample_rate}")
    
    def test_recorder_sample_rate_property(self):
        """Sample rate property should return target rate (16kHz for Whisper)."""
        recorder = AudioRecorder(callback=self.audio_callback)
        # sample_rate property should return target (16000) regardless of device rate
        self.assertEqual(recorder.sample_rate, 16000)
    
    def test_recorder_barge_in_callback(self):
        """Should accept and store barge-in callback."""
        interrupt_called = []
        
        def on_interrupt():
            interrupt_called.append(True)
        
        recorder = AudioRecorder(
            callback=self.audio_callback,
            on_interrupt=on_interrupt
        )
        self.assertEqual(recorder.on_interrupt, on_interrupt)
    
    def test_recorder_set_assistant_speaking(self):
        """Should track assistant speaking state."""
        recorder = AudioRecorder(callback=self.audio_callback)
        
        self.assertFalse(recorder.assistant_is_speaking)
        
        recorder.set_assistant_speaking(True)
        self.assertTrue(recorder.assistant_is_speaking)
        
        recorder.set_assistant_speaking(False)
        self.assertFalse(recorder.assistant_is_speaking)


class TestAudioPlayer(unittest.TestCase):
    """Test the AudioPlayer class."""
    
    def test_player_initialization(self):
        """Should initialize without errors."""
        try:
            player = AudioPlayer()
            self.assertIsNotNone(player)
            self.assertFalse(player.is_playing())
            player.cleanup()
        except Exception as e:
            self.fail(f"AudioPlayer init raised: {e}")
    
    def test_player_stop_when_not_playing(self):
        """Should handle stop() when nothing is playing."""
        player = AudioPlayer()
        try:
            player.stop()  # Should not raise
        except Exception as e:
            self.fail(f"stop() when not playing raised: {e}")
        finally:
            player.cleanup()


if __name__ == '__main__':
    unittest.main(verbosity=2)

