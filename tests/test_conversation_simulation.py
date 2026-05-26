"""
Conversation-level simulations for the voice assistant.

These tests exercise the real recorder and VoiceAssistant control flow while
scripted STT/LLM/TTS doubles make the scenarios deterministic and offline.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from tests.conversation_harness import ConversationRunner
from tests.fixtures import (
    SR,
    babble_noise,
    mix,
    pad_to,
    real_speech,
    real_tts_echo,
    tv_noise,
    voiced_speech,
)
from tests.harness import MockWakewordService

pytestmark = [pytest.mark.dev, pytest.mark.audio]


def _finishable_turn_audio(duration_sec: float = 0.4) -> np.ndarray:
    return voiced_speech(duration_sec, amplitude=0.35)


def test_happy_path_streams_llm_sentences_to_tts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with ConversationRunner(
        transcripts=["tell me a joke"],
        llm_responses=[["It is test time.", "I am ready."]],
        tts_playback_sec=0.03,
    ) as runner:
        runner.inject_user_turn(
            _finishable_turn_audio(),
            trailing_silence_sec=0.35,
            inter_chunk_delay=0.02,
        )

        assert runner.wait_for_response()

    assert runner.stt.calls
    assert runner.llm.prompts == ["tell me a joke"]
    assert runner.player.spoken_texts == ["It is test time.", "I am ready."]
    runner.assert_event_order("user_audio_injected", "stt_final", "llm_start", "tts_start")


def test_stop_command_short_circuits_llm_and_tts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with ConversationRunner(
        transcripts=["stop"],
        llm_responses=[["This should not be spoken."]],
        tts_playback_sec=0.03,
    ) as runner:
        runner.inject_user_turn(
            _finishable_turn_audio(0.32),
            trailing_silence_sec=0.35,
            inter_chunk_delay=0.02,
        )

        assert runner.wait_for_response()

    assert not runner.assistant._shutdown_event.is_set()
    assert runner.llm.prompts == []
    assert runner.player.spoken_texts == []


def test_quit_command_shuts_down_without_llm(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with ConversationRunner(
        transcripts=["quit"],
        llm_responses=[["This should not be spoken."]],
        tts_playback_sec=0.03,
    ) as runner:
        runner.inject_user_turn(
            _finishable_turn_audio(0.32),
            trailing_silence_sec=0.35,
            inter_chunk_delay=0.02,
        )

        assert runner.wait_for_response()

    assert runner.assistant._shutdown_event.is_set()
    assert runner.llm.prompts == []
    assert runner.player.spoken_texts == []


def test_stop_near_miss_still_routes_to_llm(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with ConversationRunner(
        transcripts=["started"],
        llm_responses=[["I heard started."]],
        tts_playback_sec=0.03,
    ) as runner:
        runner.inject_user_turn(
            _finishable_turn_audio(0.32),
            trailing_silence_sec=0.35,
            inter_chunk_delay=0.02,
        )

        assert runner.wait_for_response()

    assert runner.llm.prompts == ["started"]
    assert runner.player.spoken_texts == ["I heard started."]


def test_time_request_uses_capability_instead_of_llm(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with ConversationRunner(
        transcripts=["what time is it"],
        llm_responses=[["The LLM should not answer this."]],
        tts_playback_sec=0.03,
    ) as runner:
        runner.inject_user_turn(
            _finishable_turn_audio(0.32),
            trailing_silence_sec=0.35,
            inter_chunk_delay=0.02,
        )

        assert runner.wait_for_response()

    assert runner.llm.prompts == []
    assert len(runner.player.spoken_texts) == 1
    assert runner.player.spoken_texts[0].startswith("The current time is ")


def test_user_barge_in_cancels_streaming_tts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with ConversationRunner(
        transcripts=["tell me a long story"],
        llm_responses=[["This answer keeps playing while the user interrupts."]],
        tts_playback_sec=0.65,
        recorder_kwargs={"barge_in_min_speech_sec": 0.08},
    ) as runner:
        runner.inject_user_turn(
            _finishable_turn_audio(),
            trailing_silence_sec=0.35,
            inter_chunk_delay=0.02,
        )
        assert runner.wait_for_tts_start()

        tts_echo = runner.player.current_audio
        assert tts_echo is not None
        user = real_speech(0.45, amplitude=0.35, fallback_amplitude=0.42)
        mic = mix(pad_to(user, len(tts_echo)), tts_echo, snr_db=8.0)
        runner.inject_barge_in(mic, inter_chunk_delay=0.01)

        assert runner.wait_for_response()

    assert runner.interrupts, "expected user speech during TTS to trigger barge-in"
    assert runner.player.stop_count >= 1
    stop_at = runner.timeline.first_time("tts_stop")
    barge_at = runner.timeline.first_time("barge_in")
    assert stop_at is not None and barge_at is not None
    assert stop_at - barge_at < 0.20


def test_partial_stop_fast_path_stops_tts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with ConversationRunner(
        transcripts=["tell me a long story"],
        llm_responses=[["First sentence.", "Second sentence.", "Third sentence."]],
        tts_playback_sec=0.65,
    ) as runner:
        runner.inject_user_turn(
            _finishable_turn_audio(),
            trailing_silence_sec=0.35,
            inter_chunk_delay=0.02,
        )
        assert runner.wait_for_tts_start()

        decision = runner.assistant._router.route_partial(
            runner.assistant._route_context("stop", is_partial=True)
        )
        runner.assistant._execute_route_decision(decision, "stop", None, None)
        assert runner.wait_for_response()

    assert runner.player.stop_count >= 1
    assert len(runner.player.spoken_texts) < 3


def test_tts_self_echo_does_not_trigger_barge_in(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with ConversationRunner(
        transcripts=["unused"],
        llm_responses=[],
        tts_playback_sec=0.1,
        recorder_kwargs={"barge_in_min_delay_after_ref_sec": 0.20},
    ) as runner:
        echo = real_tts_echo(0.5, amplitude=0.08)
        runner.harness.set_tts_speaking(audio_ref=echo, zero_delays=False)
        runner.inject_barge_in(echo, inter_chunk_delay=1024 / SR)

    assert runner.interrupts == []
    assert runner.player.stop_count == 0


def test_noisy_room_still_completes_conversation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with ConversationRunner(
        transcripts=["turn on the lights"],
        llm_responses=[["Turning on the lights."]],
        tts_playback_sec=0.03,
    ) as runner:
        speech = real_speech(0.5, amplitude=0.28, fallback_amplitude=0.35)
        noise = mix(
            tv_noise(0.5, amplitude=0.05),
            babble_noise(0.5, amplitude=0.03),
            snr_db=3.0,
        )
        runner.inject_user_turn(
            mix(speech, noise, snr_db=8.0),
            trailing_silence_sec=0.35,
            inter_chunk_delay=0.02,
        )

        assert runner.wait_for_response()

    assert runner.llm.prompts == ["turn on the lights"]
    assert runner.player.spoken_texts == ["Turning on the lights."]


def test_wakeword_gate_blocks_then_allows_conversation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with ConversationRunner(
        transcripts=["after wakeword"],
        llm_responses=[["Wakeword accepted."]],
        wakeword_enabled=True,
        tts_playback_sec=0.03,
    ) as runner:
        wakeword = MockWakewordService()
        runner.recorder._wakeword_service = wakeword

        runner.inject_user_turn(
            _finishable_turn_audio(),
            trailing_silence_sec=0.35,
            inter_chunk_delay=0.02,
        )
        time.sleep(0.1)
        assert runner.stt.calls == []

        wakeword.arm()
        runner.inject_user_turn(
            _finishable_turn_audio(),
            trailing_silence_sec=0.35,
            inter_chunk_delay=0.02,
        )
        assert runner.wait_for_response()

    assert runner.llm.prompts == ["after wakeword"]
    assert runner.player.spoken_texts == ["Wakeword accepted."]
