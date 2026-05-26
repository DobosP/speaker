"""Hardware-free tests for TTS file prep and playback helpers."""
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio import (  # noqa: E402
    AudioPlayer,
    KOKORO_AVAILABLE,
    _tts_file_suffix,
    _WAV_TTS_BACKENDS,
)


class TestTTSPlaybackHelpers(unittest.TestCase):
    def test_wav_suffix_covers_local_backends(self):
        for backend in ("kokoro", "piper", "melotts", "supertonic"):
            self.assertEqual(_tts_file_suffix(backend), ".wav")
            self.assertIn(backend, _WAV_TTS_BACKENDS)

    def test_mp3_suffix_for_edge(self):
        self.assertEqual(_tts_file_suffix("edge-tts"), ".mp3")


@unittest.skipUnless(KOKORO_AVAILABLE, "kokoro-onnx not installed")
class TestKokoroPlaybackPrep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.player = AudioPlayer(
            tts_backend="kokoro",
            voice="en-US",
            playback_backend="sounddevice",
        )
        if cls.player.tts_backend != "kokoro":
            raise unittest.SkipTest("Kokoro model unavailable")

    def test_prepare_speech_non_silent(self):
        path = self.player.prepare_speech_file("Audio test.")
        try:
            data, sr = self.player._load_audio_data(path)
            rms = float(np.sqrt(np.mean(data**2)))
            self.assertGreater(len(data), 0)
            self.assertGreater(rms, 1e-4, f"silent wav rms={rms}")
            self.assertGreater(sr, 0)
        finally:
            os.unlink(path)

    def test_play_uses_sounddevice_when_configured(self):
        path = self.player.prepare_speech_file("Hi.")
        try:
            with mock.patch.object(self.player, "_play_with_sounddevice", return_value=True) as sd_play:
                with mock.patch.object(self.player, "_play_with_pygame") as pg_play:
                    ok = self.player.play_prepared_file(path)
            self.assertTrue(ok)
            sd_play.assert_called_once()
            pg_play.assert_not_called()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_play_falls_back_to_pygame_on_sd_error(self):
        path = self.player.prepare_speech_file("Hi.")
        try:
            with mock.patch.object(
                self.player,
                "_play_with_sounddevice",
                side_effect=RuntimeError("sd fail"),
            ):
                with mock.patch.object(self.player, "_play_with_pygame", return_value=True) as pg_play:
                    ok = self.player._play_audio_blocking(path)
            self.assertTrue(ok)
            pg_play.assert_called_once()
        finally:
            if os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
