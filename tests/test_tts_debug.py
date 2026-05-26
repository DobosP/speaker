"""Tests for SPEAKER_TTS_DEBUG / tts_debug configuration."""
import logging
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import tts_debug  # noqa: E402


class TestTtsDebug(unittest.TestCase):
    def tearDown(self):
        tts_debug.configure(enabled=False)
        if "SPEAKER_TTS_DEBUG" in os.environ:
            del os.environ["SPEAKER_TTS_DEBUG"]

    def test_resolve_from_env(self):
        os.environ["SPEAKER_TTS_DEBUG"] = "1"
        self.assertTrue(tts_debug.resolve_enabled())
        self.assertFalse(tts_debug.resolve_enabled(config_value=False))

    def test_resolve_from_config(self):
        self.assertTrue(tts_debug.resolve_enabled(config_value=True))
        self.assertTrue(tts_debug.resolve_enabled(config_value="yes"))

    def test_log_emits_when_enabled(self):
        tts_debug.configure(enabled=True)
        lg = logging.getLogger("speaker.tts")
        with self.assertLogs(lg, level="INFO") as cm:
            tts_debug.log_tts("test_event", text_chars=12)
        self.assertIn("test_event", cm.output[0])
        self.assertIn("text_chars=12", cm.output[0])

    def test_log_noop_when_disabled(self):
        tts_debug.configure(enabled=False)
        lg = logging.getLogger("speaker.tts")
        with mock.patch.object(lg, "log") as m:
            tts_debug.log_tts("test_event")
        m.assert_not_called()

    def test_console_noop_when_disabled(self):
        tts_debug.configure(enabled=False)
        with mock.patch("builtins.print") as m:
            tts_debug.console("enqueued", "hi")
        m.assert_not_called()

    def test_console_when_enabled(self):
        tts_debug.configure(enabled=True)
        with mock.patch("builtins.print") as m:
            tts_debug.console("enqueued", "phrase")
        m.assert_called_once()
        self.assertIn("enqueued", m.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
