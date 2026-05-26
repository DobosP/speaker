"""
Tests for wakeword/speaker gate helpers.
"""

import unittest

from utils.voice_gate import (
    _sample_rate_candidates,
    validate_wakeword_name,
    list_known_wakewords,
)


class TestVoiceGateHelpers(unittest.TestCase):
    def test_sample_rate_candidates_prioritize_and_dedupe(self):
        rates = _sample_rate_candidates(16000, 44100)
        self.assertEqual(rates[0], 16000)
        self.assertEqual(rates[1], 44100)
        self.assertEqual(len(rates), len(set(rates)))

    def test_sample_rate_candidates_ignore_invalid_values(self):
        rates = _sample_rate_candidates(-1, 0)
        self.assertIn(16000, rates)
        self.assertIn(48000, rates)
        self.assertNotIn(-1, rates)
        self.assertNotIn(0, rates)

    def test_validate_wakeword_name_accepts_known(self):
        ok, _ = validate_wakeword_name("hey_jarvis", list_known_wakewords())
        self.assertTrue(ok)

    def test_validate_wakeword_name_rejects_unknown(self):
        ok, msg = validate_wakeword_name("not_a_model", ["hey_jarvis", "alexa"])
        self.assertFalse(ok)
        self.assertIn("Unknown wakeword", msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
