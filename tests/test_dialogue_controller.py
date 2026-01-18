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
            rms=0.5,
            threshold=0.1,
            voiced=True,
            duration_sec=0.3,
            timestamp=time.time(),
        )
        self.assertFalse(controller.should_stop_speaking(info))

    def test_should_stop_speaking_voiced(self):
        controller = DialogueController(min_interrupt_delay_sec=0.0)
        controller.on_tts_start()
        info = BargeInInfo(
            rms=0.5,
            threshold=0.1,
            voiced=True,
            duration_sec=0.3,
            timestamp=time.time(),
        )
        self.assertTrue(controller.should_stop_speaking(info))

    def test_should_stop_speaking_with_partial(self):
        controller = DialogueController(min_interrupt_delay_sec=0.0, min_partial_chars=3)
        controller.on_tts_start()
        controller.on_partial_transcript("hello")
        info = BargeInInfo(
            rms=0.15,
            threshold=0.12,
            voiced=False,
            duration_sec=0.25,
            timestamp=time.time(),
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
            rms=0.2,
            threshold=0.1,
            voiced=False,
            duration_sec=0.3,
            timestamp=time.time(),
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
            rms=0.2,
            threshold=0.1,
            voiced=False,
            duration_sec=0.3,
            timestamp=time.time(),
        )
        self.assertTrue(controller.should_stop_speaking(info))

    def test_voice_barrier_overrides_echo(self):
        controller = DialogueController(
            min_interrupt_delay_sec=0.0,
            min_partial_chars=2,
            echo_similarity_threshold=0.6,
        )
        controller.on_assistant_text("I can help you with that right now.")
        controller.on_tts_start()
        controller.on_partial_transcript("help you with that")
        info = BargeInInfo(
            rms=0.4,
            threshold=0.1,
            voiced=True,
            duration_sec=0.4,
            timestamp=time.time(),
        )
        self.assertTrue(controller.should_stop_speaking(info))

    def test_rms_fallback_disabled_blocks_echo(self):
        controller = DialogueController(
            min_interrupt_delay_sec=0.0,
            min_partial_chars=2,
            echo_similarity_threshold=0.6,
            allow_rms_fallback=False,
        )
        controller.on_assistant_text("This is a test response for echo.")
        controller.on_tts_start()
        controller.on_partial_transcript("test response")
        info = BargeInInfo(
            rms=0.5,
            threshold=0.1,
            voiced=False,
            duration_sec=0.3,
            timestamp=time.time(),
        )
        self.assertFalse(controller.should_stop_speaking(info))

    def test_rms_fallback_enabled_allows_strong_non_echo(self):
        controller = DialogueController(
            min_interrupt_delay_sec=0.0,
            min_partial_chars=2,
            echo_similarity_threshold=0.9,
            allow_rms_fallback=True,
        )
        controller.on_assistant_text("This is a test response for echo.")
        controller.on_tts_start()
        controller.on_partial_transcript("different words")
        info = BargeInInfo(
            rms=0.5,
            threshold=0.1,
            voiced=False,
            duration_sec=0.3,
            timestamp=time.time(),
        )
        self.assertTrue(controller.should_stop_speaking(info))

    def test_echo_info_blocks_even_if_voiced(self):
        controller = DialogueController(min_interrupt_delay_sec=0.0)
        controller.on_tts_start()
        info = BargeInInfo(
            rms=0.5,
            threshold=0.1,
            voiced=True,
            duration_sec=0.4,
            timestamp=time.time(),
            echo=True,
        )
        self.assertFalse(controller.should_stop_speaking(info))

    def test_require_partial_blocks_weak_voiced(self):
        controller = DialogueController(
            min_interrupt_delay_sec=0.0,
            require_partial_for_barge_in=True,
            strong_voiced_multiplier=3.0,
        )
        controller.on_tts_start()
        info = BargeInInfo(
            rms=0.2,
            threshold=0.1,
            voiced=True,
            duration_sec=0.4,
            timestamp=time.time(),
            echo=False,
        )
        self.assertFalse(controller.should_stop_speaking(info))

    def test_should_stop_speaking_rejects_short_barge_in(self):
        controller = DialogueController(min_interrupt_delay_sec=0.0, min_barge_in_sec=0.3)
        controller.on_tts_start()
        info = BargeInInfo(
            rms=0.2,
            threshold=0.1,
            voiced=True,
            duration_sec=0.1,
            timestamp=time.time(),
        )
        self.assertFalse(controller.should_stop_speaking(info))


class DummyPlayer:
    def __init__(self):
        self.stop_called = False

    def stop(self):
        self.stop_called = True


class TestVoiceAssistantBargeIn(unittest.TestCase):
    def test_on_barge_in_stops_when_controller_allows(self):
        assistant = VoiceAssistant(mode="controller", enable_memory=False)
        assistant._player = DummyPlayer()
        assistant._controller = DialogueController(min_interrupt_delay_sec=0.0)
        assistant._controller.on_tts_start()
        info = {
            "rms": 0.5,
            "threshold": 0.1,
            "voiced": True,
            "duration_sec": 0.4,
            "timestamp": time.time(),
        }
        result = assistant._on_barge_in(info)
        self.assertTrue(result)
        self.assertTrue(assistant._player.stop_called)

    def test_on_barge_in_skips_when_controller_blocks(self):
        assistant = VoiceAssistant(mode="controller", enable_memory=False)
        assistant._player = DummyPlayer()
        assistant._controller = DialogueController(min_interrupt_delay_sec=0.0, min_barge_in_sec=0.5)
        assistant._controller.on_tts_start()
        info = {
            "rms": 0.5,
            "threshold": 0.1,
            "voiced": True,
            "duration_sec": 0.1,
            "timestamp": time.time(),
        }
        result = assistant._on_barge_in(info)
        self.assertFalse(result)
        self.assertFalse(assistant._player.stop_called)


if __name__ == "__main__":
    unittest.main(verbosity=2)

