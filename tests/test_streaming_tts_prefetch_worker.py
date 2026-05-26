"""Regression: prefetch TTS worker must wait for LLM chunks, not exit on an empty queue."""
import os
import queue
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.dev

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import VoiceAssistant  # noqa: E402


class _PrefetchHarness:
    """Minimal VoiceAssistant surface for _streaming_tts_worker_prefetch."""

    _TTS_DEQUEUE_SHUTDOWN = VoiceAssistant._TTS_DEQUEUE_SHUTDOWN

    def __init__(self):
        self._shutdown_event = threading.Event()
        self._cancel_generation = threading.Event()
        self._tts_session_lock = threading.Lock()
        self._active_tts_session = 1
        self._streaming_controller_lock = threading.Lock()
        self._streaming_controller_tts_active = False
        self._controller = None
        self._recorder = None
        self._player = MagicMock()
        self._player.prepare_speech_file.return_value = "/tmp/prefetch_test.wav"
        self.spoken = []
        self.streaming_tts_prefetch = True
        self.streaming_llm = True

    def _speak(self, text, *, defer_assistant_speaking=False, prepared_path=None):
        self.spoken.append((text, prepared_path))

    def _invalidate_tts_session(self):
        self._active_tts_session = -1

    def _streaming_tts_dequeue(self, tts_queue, **kwargs):
        return VoiceAssistant._streaming_tts_dequeue(self, tts_queue, **kwargs)


class TestStreamingTtsPrefetchWorker(unittest.TestCase):
    def test_worker_waits_for_late_enqueue(self):
        """Reproduces live bug: worker started before first LLM phrase (>100ms later)."""
        harness = _PrefetchHarness()
        tts_queue = queue.Queue()
        worker = threading.Thread(
            target=VoiceAssistant._streaming_tts_worker_prefetch,
            args=(harness, tts_queue),
            daemon=True,
        )
        worker.start()
        time.sleep(0.25)
        self.assertTrue(worker.is_alive(), "prefetch worker exited before any TTS was enqueued")

        tts_queue.put((1, "hello from the assistant"))
        tts_queue.put(None)
        worker.join(timeout=5.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(harness.spoken), 1)
        self.assertEqual(harness.spoken[0][0], "hello from the assistant")

    def test_dequeue_blocks_past_initial_timeout(self):
        harness = _PrefetchHarness()
        tts_queue = queue.Queue()
        result = {}

        def run():
            result["item"] = VoiceAssistant._streaming_tts_dequeue(harness, tts_queue)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        time.sleep(0.2)
        self.assertTrue(t.is_alive())
        tts_queue.put((2, "second phrase"))
        t.join(timeout=2.0)
        self.assertEqual(result["item"], (2, "second phrase"))


if __name__ == "__main__":
    unittest.main()
