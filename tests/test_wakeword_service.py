"""
Tests for wakeword service boundary adapters.
"""

import unittest
import numpy as np

from utils.wakeword_service import build_wakeword_service


class _DummyDetector:
    def __init__(self):
        self.available = True
        self.last_score = 0.0
        self.available_labels = ["hey_dobby"]
        self.wakeword = "hey_dobby"
        self._count = 0

    def detect(self, audio_chunk, sample_rate):
        self._count += 1
        if self._count >= 2:
            self.last_score = 0.9
            return True
        self.last_score = 0.1
        return False


class TestWakewordService(unittest.TestCase):
    def test_local_service_emits_event(self):
        svc = build_wakeword_service(
            mode="local",
            wakeword="hey_dobby",
            threshold=0.5,
            model_path=None,
            detector_override=_DummyDetector(),
        )
        audio = np.zeros(1024, dtype=np.float32)
        svc.submit_audio(audio, 16000)
        self.assertIsNone(svc.poll_event())
        svc.submit_audio(audio, 16000)
        evt = svc.poll_event()
        self.assertIsNotNone(evt)
        self.assertTrue(evt.detected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
