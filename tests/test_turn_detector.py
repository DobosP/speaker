"""
Tests for turn detector module.
"""

import time
import unittest

from utils.turn_detector import TurnDetector


class TestTurnDetector(unittest.TestCase):
    def test_recent_partial_blocks_turn_end(self):
        detector = TurnDetector(min_endpointing_delay=0.2, max_endpointing_delay=1.0)
        detector.on_partial_text("hello")
        decision = detector.evaluate(silence_sec=0.3)
        self.assertFalse(decision.should_end_turn)
        self.assertEqual(decision.reason, "recent_partial")

    def test_max_endpoint_timeout_ends_turn(self):
        detector = TurnDetector(min_endpointing_delay=0.2, max_endpointing_delay=0.5)
        time.sleep(0.01)
        decision = detector.evaluate(silence_sec=0.6)
        self.assertTrue(decision.should_end_turn)


if __name__ == "__main__":
    unittest.main(verbosity=2)
