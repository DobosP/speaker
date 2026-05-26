"""Smoke checks for live partial / utterance pipeline wiring (no audio)."""
import pytest

from main import VoiceAssistant

pytestmark = pytest.mark.dev


def test_utterance_pipeline_lock_exists():
    a = VoiceAssistant(enable_memory=False, live_partial_log=False)
    lk = a._utterance_pipeline_lock
    assert lk is not None and hasattr(lk, "acquire") and hasattr(lk, "release")


def test_live_partial_defaults_normalize():
    a = VoiceAssistant(enable_memory=False, live_partial_mode="newline")
    assert a.live_partial_mode == "newline"
    a2 = VoiceAssistant(enable_memory=False, live_partial_mode="invalid")
    assert a2.live_partial_mode == "overwrite"
