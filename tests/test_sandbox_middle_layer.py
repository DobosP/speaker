"""Realistic-timing tests for the middle-layer decision agent.

These drive the real threaded brain through a simulated device (incremental
STT partials, endpoint delay, weight-dependent LLM latency, TTS playback
duration) and assert the stop/continue/speak decisions hold up under
concurrency. Profiles are time-scaled so the suite stays fast.

Assertions wait on real signals (output produced, task started) rather than a
single idle check, because the runtime is genuinely multi-threaded.
"""

from __future__ import annotations

import pytest

from always_on_agent.events import Mode

from tests.sandbox import DESKTOP_HIGH, DESKTOP_MID, PHONE_LOW
from tests.sandbox.scenario import Sandbox

# Scale latencies down ~10x: relative timing (and thus concurrency) is
# preserved, but the wall-clock cost per scenario stays small.
PROFILES = [PHONE_LOW.scaled(0.1), DESKTOP_MID.scaled(0.1), DESKTOP_HIGH.scaled(0.1)]
PROFILE_IDS = ["phone_low", "desktop_mid", "desktop_high"]


@pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
def test_captures_full_question_after_incremental_partials(profile):
    """The brain must act on the *final* utterance, not a mid-stream partial."""
    captured = []

    def reply(prompt: str) -> str:
        captured.append(prompt)
        return "The time is noon."

    with Sandbox(profile, reply_fn=reply) as sb:
        sb.user_says("what time is it in tokyo")
        assert sb.wait_spoke_count(1)
        assert sb.spoken == ["The time is noon."]
        # assistant.answer was invoked with the full final question, not "what".
        assert captured == ["what time is it in tokyo"]


@pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
def test_no_response_before_endpoint(profile):
    """While the user is still mid-utterance, nothing should be spoken yet."""
    with Sandbox(profile) as sb:
        sb.engine._cb.on_partial("tell me")  # noqa: SLF001
        sb.engine._cb.on_partial("tell me about")  # noqa: SLF001
        assert not sb.wait_speaking(timeout=profile.stt_endpoint_delay_sec + 0.3)
        assert sb.spoken == []


@pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
def test_barge_in_during_playback_stops_immediately(profile):
    with Sandbox(profile, reply_fn=lambda p: "here is a fairly long spoken answer") as sb:
        sb.user_says("tell me something")
        assert sb.wait_speaking()
        sb.barge_in()
        assert sb.wait_not_speaking(timeout=1.0)


@pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
def test_barge_in_during_llm_generation_suppresses_stale_answer(profile):
    """The hard case: user interrupts *while the LLM is still generating*.

    We wait until the task is actually running (LLM mid-generate), then barge
    in. The half-formed answer must never be spoken. This is the concurrency
    bug a synchronous test cannot see.
    """
    with Sandbox(profile, reply_fn=lambda p: "a long answer the user no longer wants") as sb:
        sb.user_says("explain quantum computing")
        assert sb.wait_task_active()  # task is running the (slow) LLM step
        assert not sb.engine.is_speaking  # still generating, not yet speaking
        sb.barge_in()
        sb.settle()
        assert sb.spoken == []  # stale answer suppressed


def test_barge_in_cancels_llm_generation_early():
    """Barge-in should interrupt the LLM mid-stream, not wait for the full
    answer to generate. Uses the slow-phone profile so the generation window is
    wide and easy to interrupt."""
    long_reply = "this is a deliberately long answer with many tokens to generate"
    total_tokens = len(long_reply.split())
    profile = PHONE_LOW.scaled(0.1)
    with Sandbox(profile, reply_fn=lambda p: long_reply) as sb:
        sb.user_says("explain something at length")
        assert sb.wait_task_active()
        sb.barge_in()
        sb.settle()
        assert sb.spoken == []
        # Generation stopped well before producing the whole answer.
        assert sb.llm.tokens_yielded < total_tokens


@pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
def test_voice_mode_switch_then_research(profile):
    with Sandbox(profile, start_mode=Mode.ASSISTANT) as sb:
        sb.user_says("research mode")
        assert sb._poll(lambda: sb.mode == Mode.RESEARCH, 2.0)

        sb.user_says("local speech to text engines")
        assert sb.wait_spoke_count(1)


def test_concurrent_followup_while_research_runs():
    """A new question arriving mid-task shouldn't deadlock the runtime."""
    profile = DESKTOP_MID.scaled(0.1)
    with Sandbox(profile, start_mode=Mode.ASSISTANT) as sb:
        sb.user_says("research edge voice models")
        sb.user_says("what time is it")  # follow-up before the first result
        assert sb.wait_spoke_count(1, timeout=8.0)
        assert sb.wait_idle(timeout=8.0)


def test_passive_mode_ignores_unaddressed_then_activates():
    profile = DESKTOP_HIGH.scaled(0.1)
    with Sandbox(profile, start_mode=Mode.PASSIVE, reply_fn=lambda p: "Yes?") as sb:
        sb.user_says("what time is it")
        sb.settle()
        assert sb.spoken == []  # no wake word -> ignored

        sb.user_says("assistant please help")
        assert sb.wait_spoke_count(1)
        assert sb.spoken == ["Yes?"]


@pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
def test_runtime_records_first_audio_latency(profile):
    """A full turn populates the metrics recorder end-to-end: speech_end,
    asr_final, llm_first_token, tts_first_audio -- so first-audio latency is a
    real measured delta, not a model estimate."""
    with Sandbox(profile, reply_fn=lambda p: "The time is noon.") as sb:
        sb.user_says("what time is it")
        assert sb.wait_spoke_count(1)
        assert sb.wait_idle()
        [record] = sb.runtime.metrics.records()
        assert record.first_audio_latency is not None
        assert record.first_audio_latency > 0
        assert record.final_to_first_token is not None
        assert record.endpoint_latency is not None
        # Faster profiles must measure as faster first-audio than slow ones.
        assert record.first_audio_latency >= profile.llm_ttft_sec


@pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
def test_runtime_records_barge_in_latency(profile):
    with Sandbox(profile, reply_fn=lambda p: "here is a fairly long spoken answer") as sb:
        sb.user_says("tell me something")
        assert sb.wait_speaking()
        sb.barge_in()
        assert sb.wait_not_speaking(timeout=1.0)
        records = sb.runtime.metrics.records()
        assert records and records[-1].barge_in_latency is not None
        assert records[-1].barge_in_latency >= 0
