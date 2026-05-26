"""Regression: prefetch TTS worker must wait for LLM chunks (not exit on empty queue)."""
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from main import VoiceAssistant

pytestmark = pytest.mark.dev


class _StubPlayer:
    tts_backend = "kokoro"

    def prepare_speech_file(self, text: str) -> str:
        return "/tmp/stub-tts.wav"


def _assistant_with_stub_player() -> VoiceAssistant:
    assistant = VoiceAssistant(
        enable_memory=False,
        streaming_llm=True,
        streaming_tts_prefetch=True,
        audio_player=_StubPlayer(),
    )
    assistant._player = _StubPlayer()
    assistant._recorder = MagicMock()
    assistant._recorder.is_barge_in_active = MagicMock(return_value=False)
    return assistant


def test_prefetch_worker_waits_for_late_enqueue():
    """Prefetch worker used to return after 100ms with an empty queue (no audio)."""
    assistant = _assistant_with_stub_player()
    assistant._active_tts_session = 7

    spoken: list[str] = []

    def _capture_speak(text, **kwargs):
        spoken.append(text)
        return True

    tts_queue: queue.Queue = queue.Queue()
    with patch.object(assistant, "_speak", side_effect=_capture_speak):
        with patch.object(
            assistant._player, "prepare_speech_file", return_value="/tmp/stub.wav"
        ):
            worker = threading.Thread(
                target=assistant._streaming_tts_worker_prefetch,
                args=(tts_queue,),
                daemon=True,
            )
            worker.start()
            time.sleep(0.25)
            tts_queue.put((7, "How can I help you?"))
            tts_queue.put(None)
            worker.join(timeout=10.0)

    assert not worker.is_alive()
    assert spoken == ["How can I help you?"]


def test_prefetch_worker_waits_for_second_chunk():
    assistant = _assistant_with_stub_player()
    assistant._active_tts_session = 3

    spoken: list[str] = []

    def _capture_speak(text, **kwargs):
        spoken.append(text)
        return True

    tts_queue: queue.Queue = queue.Queue()
    with patch.object(assistant, "_speak", side_effect=_capture_speak):
        with patch.object(
            assistant._player, "prepare_speech_file", return_value="/tmp/stub.wav"
        ):
            worker = threading.Thread(
                target=assistant._streaming_tts_worker_prefetch,
                args=(tts_queue,),
                daemon=True,
            )
            worker.start()
            tts_queue.put((3, "First phrase."))
            time.sleep(0.05)
            tts_queue.put((3, "Second phrase."))
            tts_queue.put(None)
            worker.join(timeout=15.0)

    assert spoken == ["First phrase.", "Second phrase."]
