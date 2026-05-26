"""
Unit tests for the dialogue controller.
"""
import unittest
import time

from utils.dialogue_controller import DialogueController, BargeInInfo
from main import VoiceAssistant


class TestDialogueController(unittest.TestCase):
    def test_ignore_transcript_rules(self):
        controller = DialogueController()
        self.assertTrue(controller.should_ignore_transcript(""))
        self.assertTrue(controller.should_ignore_transcript("  "))
        self.assertTrue(controller.should_ignore_transcript("uh"))
        self.assertFalse(controller.should_ignore_transcript("hello"))

    def test_should_stop_speaking_requires_active_tts(self):
        controller = DialogueController(min_interrupt_delay_sec=0.0)
        info = BargeInInfo(
            rms=0.5, threshold=0.1, voiced=True,
            duration_sec=0.3, timestamp=time.time(),
        )
        self.assertFalse(controller.should_stop_speaking(info))

    def test_should_stop_speaking_voiced(self):
        controller = DialogueController(min_interrupt_delay_sec=0.0)
        controller.on_tts_start()
        info = BargeInInfo(
            rms=0.5, threshold=0.1, voiced=True,
            duration_sec=0.3, timestamp=time.time(),
        )
        self.assertTrue(controller.should_stop_speaking(info))

    def test_should_stop_speaking_with_partial(self):
        controller = DialogueController(
            min_interrupt_delay_sec=0.0, min_partial_chars=3
        )
        controller.on_tts_start()
        controller.on_partial_transcript("hello")
        info = BargeInInfo(
            rms=0.15, threshold=0.12, voiced=False,
            duration_sec=0.25, timestamp=time.time(),
        )
        self.assertTrue(controller.should_stop_speaking(info))

    def test_echo_partial_does_not_stop_speaking(self):
        controller = DialogueController(
            min_interrupt_delay_sec=0.0,
            min_partial_chars=2,
            echo_similarity_threshold=0.6,
        )
        controller.on_assistant_text("I can help you with that right now.")
        controller.on_tts_start()
        controller.on_partial_transcript("help you with that")
        info = BargeInInfo(
            rms=0.2, threshold=0.1, voiced=False,
            duration_sec=0.3, timestamp=time.time(),
        )
        self.assertFalse(controller.should_stop_speaking(info))

    def test_non_echo_partial_stops_speaking(self):
        controller = DialogueController(
            min_interrupt_delay_sec=0.0,
            min_partial_chars=2,
            echo_similarity_threshold=0.9,
        )
        controller.on_assistant_text("I can help you with that right now.")
        controller.on_tts_start()
        controller.on_partial_transcript("what about tomorrow")
        info = BargeInInfo(
            rms=0.2, threshold=0.1, voiced=False,
            duration_sec=0.3, timestamp=time.time(),
        )
        self.assertTrue(controller.should_stop_speaking(info))

    def test_voice_allows_barge_in_without_partial(self):
        """With simplified controller, voiced=True should allow barge-in."""
        controller = DialogueController(
            min_interrupt_delay_sec=0.0,
            require_partial_for_barge_in=False,
        )
        controller.on_tts_start()
        info = BargeInInfo(
            rms=0.4, threshold=0.1, voiced=True,
            duration_sec=0.4, timestamp=time.time(),
        )
        self.assertTrue(controller.should_stop_speaking(info))

    def test_rms_fallback_disabled_blocks(self):
        controller = DialogueController(
            min_interrupt_delay_sec=0.0,
            allow_rms_fallback=False,
        )
        controller.on_tts_start()
        info = BargeInInfo(
            rms=0.5, threshold=0.1, voiced=False,
            duration_sec=0.3, timestamp=time.time(),
        )
        self.assertFalse(controller.should_stop_speaking(info))

    def test_rms_fallback_enabled_allows_strong(self):
        controller = DialogueController(
            min_interrupt_delay_sec=0.0,
            allow_rms_fallback=True,
        )
        controller.on_tts_start()
        info = BargeInInfo(
            rms=0.5, threshold=0.1, voiced=False,
            duration_sec=0.3, timestamp=time.time(),
        )
        self.assertTrue(controller.should_stop_speaking(info))

    def test_echo_info_blocks_even_if_voiced(self):
        controller = DialogueController(min_interrupt_delay_sec=0.0)
        controller.on_tts_start()
        info = BargeInInfo(
            rms=0.5, threshold=0.1, voiced=True,
            duration_sec=0.4, timestamp=time.time(),
            echo=True,
        )
        self.assertFalse(controller.should_stop_speaking(info))

    def test_should_stop_speaking_rejects_short_barge_in(self):
        controller = DialogueController(
            min_interrupt_delay_sec=0.0, min_barge_in_sec=0.3
        )
        controller.on_tts_start()
        info = BargeInInfo(
            rms=0.2, threshold=0.1, voiced=True,
            duration_sec=0.1, timestamp=time.time(),
        )
        self.assertFalse(controller.should_stop_speaking(info))

    def test_state_machine_transitions(self):
        controller = DialogueController(min_interrupt_delay_sec=0.0)
        self.assertEqual(controller.state(), "idle")
        controller.on_tts_start()
        self.assertEqual(controller.state(), "assistant_speaking")
        info = BargeInInfo(
            rms=0.5, threshold=0.1, voiced=True,
            duration_sec=0.4, timestamp=time.time(),
        )
        self.assertTrue(controller.should_stop_speaking(info))
        self.assertEqual(controller.state(), "user_takeover")
        controller.on_tts_end()
        self.assertEqual(controller.state(), "recover")


class DummyPlayer:
    def __init__(self):
        self.stop_called = False

    def stop(self):
        self.stop_called = True

    def cleanup(self):
        pass


class TestVoiceAssistantBargeIn(unittest.TestCase):
    def test_on_barge_in_stops_when_controller_allows(self):
        assistant = VoiceAssistant(mode="controller", enable_memory=False)
        assistant._player = DummyPlayer()
        assistant._controller = DialogueController(min_interrupt_delay_sec=0.0)
        assistant._controller.on_tts_start()
        info = {
            "rms": 0.5, "threshold": 0.1, "voiced": True,
            "duration_sec": 0.4, "timestamp": time.time(),
        }
        result = assistant._on_barge_in(info)
        self.assertTrue(result)
        self.assertTrue(assistant._player.stop_called)

    def test_on_barge_in_skips_when_controller_blocks(self):
        assistant = VoiceAssistant(mode="controller", enable_memory=False)
        assistant._player = DummyPlayer()
        assistant._controller = DialogueController(
            min_interrupt_delay_sec=0.0, min_barge_in_sec=0.5
        )
        assistant._controller.on_tts_start()
        info = {
            "rms": 0.5, "threshold": 0.1, "voiced": True,
            "duration_sec": 0.1, "timestamp": time.time(),
        }
        result = assistant._on_barge_in(info)
        self.assertFalse(result)
        self.assertFalse(assistant._player.stop_called)

    def test_shutdown_stops_player_and_cancels_generation(self):
        assistant = VoiceAssistant(mode="controller", enable_memory=False)
        assistant._player = DummyPlayer()
        self.assertFalse(assistant._cancel_generation.is_set())
        assistant.shutdown()
        self.assertTrue(assistant._cancel_generation.is_set())
        self.assertTrue(assistant._shutdown_event.is_set())
        self.assertTrue(assistant._player.stop_called)


if __name__ == "__main__":
    unittest.main(verbosity=2)
