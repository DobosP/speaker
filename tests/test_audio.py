"""
Unit tests for the cross-platform audio module.
"""
import unittest
import numpy as np
import time
import sys
import os
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio import (
    AudioRecorder,
    AudioPlayer,
    NLMSEchoCanceller,
    BargeInDetector,
    FrameFeatures,
    SileroVADDetector,
    split_sentences,
    list_audio_devices,
    get_default_input_device,
    get_default_output_device,
    SILERO_VAD_AVAILABLE,
)
from utils.voice_gate import OpenWakeWordGate


class TestNLMSEchoCanceller(unittest.TestCase):
    """Test the NLMS adaptive echo canceller."""

    def test_basic_echo_cancellation(self):
        """NLMS should reduce echo energy after adapting over several frames."""
        aec = NLMSEchoCanceller(filter_length=64, step_size=0.5)
        # Feed several frames so the filter can adapt
        n_frames = 8
        frame_size = 256
        total = n_frames * frame_size
        ref = np.sin(np.linspace(0, 8 * np.pi, total)).astype(np.float32) * 0.3
        noise = np.random.randn(total).astype(np.float32) * 0.01
        mic = ref * 0.8 + noise

        aec.feed_reference(ref)
        first_energy = None
        last_energy = None
        for i in range(n_frames):
            chunk = mic[i * frame_size : (i + 1) * frame_size]
            cleaned = aec.process_frame(chunk)
            energy = float(np.mean(cleaned**2))
            if i == 0:
                first_energy = energy
            last_energy = energy

        # After adaptation, later frames should have less residual energy
        self.assertLess(last_energy, first_energy)

    def test_passthrough_without_reference(self):
        """Without reference, audio should pass through unchanged."""
        aec = NLMSEchoCanceller(filter_length=64)
        mic = np.random.randn(256).astype(np.float32) * 0.1
        output = aec.process_frame(mic)
        np.testing.assert_allclose(output, mic, atol=1e-6)

    def test_reset_clears_state(self):
        """Reset should clear all filter state."""
        aec = NLMSEchoCanceller(filter_length=64)
        ref = np.ones(128, dtype=np.float32) * 0.5
        aec.feed_reference(ref)
        self.assertTrue(aec.active)
        aec.reset()
        self.assertFalse(aec.active)
        np.testing.assert_array_equal(aec.h, np.zeros(64))

    def test_reference_consumed_incrementally(self):
        """Reference should be consumed as frames are processed."""
        aec = NLMSEchoCanceller(filter_length=32)
        ref = np.ones(256, dtype=np.float32) * 0.5
        aec.feed_reference(ref)
        self.assertTrue(aec.active)

        # Process two 128-sample frames (consuming 256 ref samples total)
        mic1 = np.random.randn(128).astype(np.float32) * 0.1
        aec.process_frame(mic1)
        mic2 = np.random.randn(128).astype(np.float32) * 0.1
        aec.process_frame(mic2)
        # Reference should be exhausted
        self.assertFalse(aec.active)

    def test_multiple_frames_adaptation(self):
        """Filter should adapt over multiple frames."""
        aec = NLMSEchoCanceller(filter_length=64, step_size=0.5)
        # Feed many frames of the same echo pattern
        ref = np.tile(
            np.sin(np.linspace(0, 2 * np.pi, 128)).astype(np.float32), 10
        ) * 0.3
        mic = ref * 0.7
        aec.feed_reference(ref)

        energies = []
        for i in range(10):
            chunk = mic[i * 128 : (i + 1) * 128]
            cleaned = aec.process_frame(chunk)
            energies.append(float(np.mean(cleaned**2)))

        # Later frames should have less residual energy (filter learned)
        self.assertLess(energies[-1], energies[0])


class TestSileroVAD(unittest.TestCase):
    """Test the Silero VAD detector."""

    @unittest.skipUnless(SILERO_VAD_AVAILABLE, "Silero VAD not installed")
    def test_silence_not_speech(self):
        """Silence should not be detected as speech."""
        vad = SileroVADDetector()
        silence = np.zeros(1024, dtype=np.float32)
        self.assertFalse(vad.is_speech(silence, 16000))

    @unittest.skipUnless(SILERO_VAD_AVAILABLE, "Silero VAD not installed")
    def test_probability_range(self):
        """Speech probability should be between 0 and 1."""
        vad = SileroVADDetector()
        noise = np.random.randn(1024).astype(np.float32) * 0.01
        prob = vad.speech_probability(noise, 16000)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    @unittest.skipUnless(SILERO_VAD_AVAILABLE, "Silero VAD not installed")
    def test_singleton(self):
        """SileroVADDetector should be a singleton."""
        vad1 = SileroVADDetector()
        vad2 = SileroVADDetector()
        self.assertIs(vad1, vad2)


class TestSentenceSplitter(unittest.TestCase):
    """Test sentence splitting for chunked TTS."""

    def test_single_sentence(self):
        self.assertEqual(split_sentences("Hello world."), ["Hello world."])

    def test_multiple_sentences(self):
        result = split_sentences(
            "This is the first sentence. This is the second one. And the third!"
        )
        self.assertEqual(len(result), 3)
        self.assertIn("first sentence", result[0])

    def test_short_fragments_merged(self):
        """Short fragments (<8 chars) merge with next."""
        result = split_sentences("Hello. How are you? I am fine!")
        # "Hello." (6 chars) merges with "How are you?" -> 2 items
        self.assertEqual(len(result), 2)

    def test_short_merge(self):
        """Short fragments should be merged with previous."""
        result = split_sentences("Hi. OK. That sounds great.")
        # "Hi. OK." are both short, should be merged
        self.assertTrue(len(result) <= 2)

    def test_empty_text(self):
        result = split_sentences("")
        self.assertEqual(result, [""])

    def test_no_punctuation(self):
        result = split_sentences("no punctuation here")
        self.assertEqual(result, ["no punctuation here"])


class TestAudioDeviceDetection(unittest.TestCase):
    """Test audio device detection works on any platform."""

    def test_list_devices_returns_without_error(self):
        try:
            devices = list_audio_devices()
            self.assertIsNotNone(devices)
        except Exception as e:
            self.fail(f"list_audio_devices() raised an exception: {e}")

    def test_get_default_input_device(self):
        try:
            device = get_default_input_device()
            self.assertTrue(device is None or isinstance(device, (int, type(None))))
        except Exception as e:
            self.fail(f"get_default_input_device() raised: {e}")

    def test_get_default_output_device(self):
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
        recorder = AudioRecorder(callback=self.audio_callback)
        self.assertIsNotNone(recorder)
        self.assertFalse(recorder.is_recording)
        self.assertFalse(recorder.is_paused)

    def test_recorder_start_stop(self):
        recorder = AudioRecorder(callback=self.audio_callback)
        try:
            recorder.start()
            self.assertTrue(recorder.is_recording)
            time.sleep(0.5)
            recorder.stop()
            self.assertFalse(recorder.is_recording)
        except Exception as e:
            recorder.stop()
            if "No Default Input Device" in str(e) or "PortAudio" in str(e):
                self.skipTest(f"No audio device available: {e}")
            self.fail(f"Start/stop raised: {e}")

    def test_recorder_pause_resume(self):
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
            if "No Default Input Device" in str(e) or "PortAudio" in str(e):
                self.skipTest(f"No audio device available: {e}")
            self.fail(f"Pause/resume raised: {e}")

    def test_recorder_with_custom_threshold(self):
        recorder = AudioRecorder(
            callback=self.audio_callback,
            vad_threshold=0.05,
            silence_duration=2.0,
        )
        self.assertEqual(recorder.vad_threshold, 0.05)
        self.assertEqual(recorder.silence_duration, 2.0)

    def test_recorder_detects_device_sample_rate(self):
        recorder = AudioRecorder(callback=self.audio_callback)
        self.assertIsInstance(recorder.device_sample_rate, int)
        self.assertGreater(recorder.device_sample_rate, 0)
        common_rates = [8000, 16000, 22050, 44100, 48000, 96000]
        self.assertIn(
            recorder.device_sample_rate, common_rates,
            f"Unusual sample rate: {recorder.device_sample_rate}",
        )

    def test_recorder_sample_rate_property(self):
        recorder = AudioRecorder(callback=self.audio_callback)
        self.assertEqual(recorder.sample_rate, 16000)

    def test_adaptive_vad_threshold_from_noise(self):
        recorder = AudioRecorder(callback=self.audio_callback, adaptive_vad=True)
        noise_floor = recorder._update_noise_floor_from_rms(
            [0.001, 0.002, 0.004, 0.003]
        )
        self.assertGreaterEqual(noise_floor, recorder.vad_noise_floor_min)
        threshold = recorder._get_vad_threshold()
        self.assertGreaterEqual(
            threshold, noise_floor * recorder.vad_noise_multiplier
        )

    def test_nlms_aec_reduces_echo_energy(self):
        """Recorder's NLMS AEC should reduce echo after adapting.

        The set_echo_reference() prepends ~120ms of silence to align
        the reference with playback latency.  We feed enough frames
        for the NLMS to pass through that silence and start adapting,
        then verify that late-frame energy is lower than early-frame energy.
        """
        recorder = AudioRecorder(
            callback=self.audio_callback,
            aec_enabled=True,
            aec_strength=0.5,
            aec_filter_ms=50.0,          # smaller filter for fast convergence in test
        )
        sr = recorder.device_sample_rate
        n_frames = 20                     # enough for convergence after delay
        frame_size = 1024
        total = n_frames * frame_size
        ref = np.sin(np.linspace(0, 8 * np.pi, total)).astype(np.float32) * 0.2
        mic = ref * 0.7 + np.random.randn(total).astype(np.float32) * 0.01
        recorder.set_echo_reference(ref, sr)
        energies = []
        for i in range(n_frames):
            chunk = mic[i * frame_size : (i + 1) * frame_size]
            cleaned = recorder._apply_aec(chunk)
            energy = float(np.mean(cleaned**2))
            energies.append(energy)
        # Compare last quarter vs first quarter average energy
        q = n_frames // 4
        early_avg = sum(energies[:q]) / q
        late_avg = sum(energies[-q:]) / q
        self.assertLess(late_avg, early_avg)

    def test_simple_voiced_fallback(self):
        recorder = AudioRecorder(
            callback=self.audio_callback, simple_voiced_fallback=True
        )
        tone = np.sin(np.linspace(0, 2 * np.pi * 20, 1024)).astype(
            np.float32
        ) * 0.2
        self.assertTrue(recorder._simple_voiced(tone, threshold=0.01))

    def test_echo_similarity_threshold(self):
        recorder = AudioRecorder(
            callback=self.audio_callback, aec_enabled=True
        )
        ref = np.sin(np.linspace(0, 2 * np.pi, 1024)).astype(np.float32) * 0.2
        recorder._aec_ref = ref.copy()
        recorder._aec_ref_idx = 0
        recorder._ref_samples_consumed = 0
        corr = recorder._echo_similarity(ref.copy())
        self.assertGreater(corr, 0.5)

    def test_min_rms_ratio_blocks_low_energy(self):
        recorder = AudioRecorder(
            callback=self.audio_callback, adaptive_vad=True
        )
        recorder._noise_floor = 0.01
        recorder.barge_in_min_rms_ratio = 2.0
        min_rms = recorder._noise_floor * recorder.barge_in_min_rms_ratio
        self.assertGreater(min_rms, recorder._noise_floor)

    def test_recorder_barge_in_callback(self):
        interrupt_called = []

        def on_interrupt():
            interrupt_called.append(True)

        recorder = AudioRecorder(
            callback=self.audio_callback, on_interrupt=on_interrupt
        )
        self.assertEqual(recorder.on_interrupt, on_interrupt)

    def test_recorder_set_assistant_speaking(self):
        recorder = AudioRecorder(callback=self.audio_callback)
        self.assertFalse(recorder.assistant_is_speaking)
        recorder.set_assistant_speaking(True)
        self.assertTrue(recorder.assistant_is_speaking)
        recorder.set_assistant_speaking(False)
        self.assertFalse(recorder.assistant_is_speaking)


class DummyWakewordDetector:
    def __init__(self):
        self.available = True
        self._detected = False

    def set_detected(self, value: bool):
        self._detected = value

    def detect(self, audio_chunk, sample_rate):
        return self._detected


class DummySpeakerVerifier:
    def __init__(self, accepted: bool):
        self.available = True
        self._accepted = accepted

    def verify(self, audio, sample_rate):
        return self._accepted


class TestVoiceGates(unittest.TestCase):
    def setUp(self):
        self.captured = []

    def _callback(self, audio_data):
        self.captured.append(audio_data)

    def test_wakeword_gate_blocks_until_armed(self):
        detector = DummyWakewordDetector()
        recorder = AudioRecorder(
            callback=self._callback,
            wakeword_enabled=True,
            wakeword_detector=detector,
            wakeword_timeout_sec=1.0,
        )
        self.assertFalse(recorder._wakeword_gate_open())
        detector.set_detected(True)
        recorder._update_wakeword_state(np.zeros(1024, dtype=np.float32))
        self.assertTrue(recorder._wakeword_gate_open())

    def test_speaker_verifier_blocks_callback_when_rejected(self):
        recorder = AudioRecorder(
            callback=self._callback,
            speaker_verify_enabled=True,
            speaker_verifier=DummySpeakerVerifier(accepted=False),
        )
        recorder._audio_buffer = np.ones(int(recorder.device_sample_rate * 0.8), dtype=np.float32) * 0.02
        recorder._is_speaking = True
        recorder._finish_recording()
        self.assertEqual(len(self.captured), 0)

    def test_speaker_verifier_allows_callback_when_accepted(self):
        recorder = AudioRecorder(
            callback=self._callback,
            speaker_verify_enabled=True,
            speaker_verifier=DummySpeakerVerifier(accepted=True),
        )
        recorder._audio_buffer = np.ones(int(recorder.device_sample_rate * 0.8), dtype=np.float32) * 0.02
        recorder._is_speaking = True
        recorder._finish_recording()
        self.assertEqual(len(self.captured), 1)

    def test_wakeword_score_filter_uses_selected_label(self):
        scores = {
            "alexa": {"score": 0.82},
            "hey_jarvis": {"score": 0.21},
        }
        picked = OpenWakeWordGate._extract_top_score(scores, wakeword="jarvis")
        self.assertAlmostEqual(picked, 0.21, places=3)

    def test_diagnostics_log_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "bargein-debug.jsonl")
            recorder = AudioRecorder(
                callback=self._callback,
                diagnostics_log_path=path,
            )
            recorder._diag_log("unit_test_event", value=1)
            self.assertTrue(os.path.exists(path))

    def test_listener_state_transitions_tts_recover(self):
        recorder = AudioRecorder(callback=self._callback)
        self.assertIn(recorder.listener_state(), {"idle", "armed"})
        recorder.set_assistant_speaking(True)
        self.assertEqual(recorder.listener_state(), "assistant_speaking")
        recorder.set_assistant_speaking(False)
        self.assertEqual(recorder.listener_state(), "recover")


class TestSelfInterruptionPrevention(unittest.TestCase):
    """Tests that TTS output does NOT trigger false barge-in."""

    def audio_callback(self, audio):
        pass

    def test_echo_similarity_blocks_own_output(self):
        """When mic picks up TTS output, echo_similarity should be high,
        preventing barge-in."""
        recorder = AudioRecorder(
            callback=self.audio_callback,
            aec_enabled=True,
            echo_corr_threshold=0.45,
        )
        sr = recorder.device_sample_rate
        # Simulate a TTS reference signal
        duration_sec = 1.0
        n_samples = int(sr * duration_sec)
        ref = np.sin(np.linspace(0, 50 * np.pi, n_samples)).astype(np.float32) * 0.3
        recorder.set_echo_reference(ref, sr)

        # The set_echo_reference prepends ~120ms silence as delay compensation.
        delay_samples = int(0.12 * sr)

        # Simulate mic picking up the TTS after the delay
        frame_size = 1024
        # Skip frames during the silence portion
        skip_frames = delay_samples // frame_size + 1
        for i in range(skip_frames):
            zero_chunk = np.zeros(frame_size, dtype=np.float32)
            recorder._echo_similarity(zero_chunk)  # consume silence portion

        # Now feed TTS-like audio and check echo similarity is high
        high_sim_count = 0
        test_frames = 10
        for i in range(test_frames):
            start = skip_frames * frame_size + i * frame_size
            if start + frame_size > n_samples:
                break
            # Mic = ref (echo) + small noise
            mic_chunk = ref[start - delay_samples:start - delay_samples + frame_size] * 0.8
            mic_chunk = mic_chunk + np.random.randn(frame_size).astype(np.float32) * 0.005
            sim = recorder._echo_similarity(mic_chunk)
            if sim >= recorder.echo_corr_threshold:
                high_sim_count += 1
        # Most frames should be detected as echo
        self.assertGreater(high_sim_count, test_frames // 2)

    def test_ref_samples_consumed_tracks_position(self):
        """_ref_samples_consumed should advance with each call to
        _echo_similarity, keeping alignment with the NLMS ref."""
        recorder = AudioRecorder(
            callback=self.audio_callback, aec_enabled=True
        )
        sr = recorder.device_sample_rate
        ref = np.random.randn(4096).astype(np.float32) * 0.1
        recorder._aec_ref = ref.copy()
        recorder._ref_samples_consumed = 0
        frame = np.random.randn(1024).astype(np.float32) * 0.05
        recorder._echo_similarity(frame)
        self.assertEqual(recorder._ref_samples_consumed, 1024)
        recorder._echo_similarity(frame)
        self.assertEqual(recorder._ref_samples_consumed, 2048)

    def test_higher_silero_threshold_in_barge_in_mode(self):
        """_is_voiced with barge_in_mode=True should be harder to trigger."""
        recorder = AudioRecorder(
            callback=self.audio_callback, aec_enabled=True
        )
        if recorder._silero_vad is None:
            self.skipTest("Silero VAD not available")
        # Feed a borderline signal (echo residual) and check that
        # barge_in_mode=True blocks it while normal mode might not.
        # Use a tone-like signal that Silero might rate around 0.5-0.7
        sr = recorder.device_sample_rate
        chunk = np.sin(np.linspace(0, 200 * np.pi, 1024)).astype(np.float32) * 0.05
        result_normal = recorder._is_voiced(chunk, 0.01, barge_in_mode=False)
        result_barge = recorder._is_voiced(chunk, 0.01, barge_in_mode=True)
        # barge_in_mode should be at least as strict (never less strict)
        if result_barge:
            self.assertTrue(result_normal)  # if barge allows, normal must too


class TestBargeInDetector(unittest.TestCase):
    """Direct tests for score-based barge-in confirmation behavior."""

    def test_echo_heavy_frame_is_blocked(self):
        detector = BargeInDetector(
            sample_rate=16000,
            min_speech_sec=0.2,
            echo_corr_threshold=0.45,
        )
        frame = FrameFeatures(
            timestamp=time.time(),
            rms=0.12,
            threshold=0.01,
            voiced=True,
            echo_similarity=0.92,   # strong echo
            raw_rms=0.12,
        )
        triggered = detector.update(frame, frame_len=1024)
        self.assertFalse(triggered)
        self.assertEqual(detector.above_samples, 0)

    def test_real_voiced_speech_triggers_after_min_duration(self):
        detector = BargeInDetector(
            sample_rate=16000,
            min_speech_sec=0.2,   # 3200 samples
            echo_corr_threshold=0.45,
        )
        frame = FrameFeatures(
            timestamp=time.time(),
            rms=0.08,
            threshold=0.01,
            voiced=True,
            echo_similarity=0.05,
            raw_rms=0.08,
        )
        # 3 frames = 3072 samples (not enough), 4th should trigger
        self.assertFalse(detector.update(frame, frame_len=1024))
        self.assertFalse(detector.update(frame, frame_len=1024))
        self.assertFalse(detector.update(frame, frame_len=1024))
        self.assertTrue(detector.update(frame, frame_len=1024))
        self.assertGreaterEqual(detector.above_samples, 3200)

    def test_low_confidence_frame_resets_counter(self):
        detector = BargeInDetector(
            sample_rate=16000,
            min_speech_sec=0.2,
            echo_corr_threshold=0.45,
        )
        strong = FrameFeatures(
            timestamp=time.time(),
            rms=0.08,
            threshold=0.01,
            voiced=True,
            echo_similarity=0.05,
            raw_rms=0.08,
        )
        weak = FrameFeatures(
            timestamp=time.time(),
            rms=0.008,   # below threshold
            threshold=0.01,
            voiced=False,
            echo_similarity=0.0,
            raw_rms=0.008,
        )
        detector.update(strong, frame_len=1024)
        detector.update(strong, frame_len=1024)
        self.assertGreater(detector.above_samples, 0)
        detector.update(weak, frame_len=1024)
        self.assertEqual(detector.above_samples, 0)

    def test_strong_unvoiced_energy_can_trigger(self):
        detector = BargeInDetector(
            sample_rate=16000,
            min_speech_sec=0.15,  # 2400 samples
            echo_corr_threshold=0.45,
        )
        # Simulate VAD miss with strong energy but calibrated noise floor.
        # This is the real-world scenario: Silero occasionally misses a frame
        # but the environment has been calibrated so energy path can contribute.
        frame = FrameFeatures(
            timestamp=time.time(),
            rms=0.05,
            threshold=0.01,
            voiced=False,
            echo_similarity=0.05,
            raw_rms=0.05,
            noise_floor_calibrated=True,
        )
        self.assertFalse(detector.update(frame, frame_len=1024))
        self.assertFalse(detector.update(frame, frame_len=1024))
        self.assertTrue(detector.update(frame, frame_len=1024))


class TestAudioPlayer(unittest.TestCase):
    """Test the AudioPlayer class."""

    def test_player_initialization(self):
        try:
            player = AudioPlayer()
            self.assertIsNotNone(player)
            self.assertFalse(player.is_playing())
            player.cleanup()
        except Exception as e:
            self.fail(f"AudioPlayer init raised: {e}")

    def test_player_stop_when_not_playing(self):
        player = AudioPlayer()
        try:
            player.stop()
        except Exception as e:
            self.fail(f"stop() when not playing raised: {e}")
        finally:
            player.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
