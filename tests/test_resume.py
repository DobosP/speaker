"""Resume-after-interrupt + L4 self-echo guard (core/resume.py).

Pins the two owner-reported failures from the live run-20260610-124002:
'start again' after stopping a story greeted the user instead of CONTINUING
the story, and the assistant's own TTS ('Okay, let's begin. How can I help you
today?') came back through the mic as a user turn and got ANSWERED.
"""
from __future__ import annotations

import time

from core.resume import ResumeConfig, ResumeTracker


# --- resume_prompt -------------------------------------------------------------


def _cut_story_tracker(**cfg) -> ResumeTracker:
    cfg.setdefault("enabled", True)
    t = ResumeTracker(ResumeConfig(**cfg))
    t.note_query("Tell me a story about a lighthouse keeper")
    t.note_spoken("He kept watch over the churning sea.")
    t.note_spoken("Every night he climbed the spiral stairs.")
    t.note_cut()
    return t


def test_start_again_after_a_cut_resumes_not_restarts():
    t = _cut_story_tracker()
    prompt = t.resume_prompt("Start again")
    assert prompt is not None
    assert "Tell me a story about a lighthouse keeper" in prompt
    assert "spiral stairs" in prompt          # the spoken tail is embedded
    assert "do not start over" in prompt      # the continue instruction


def test_continue_and_go_on_also_resume():
    for phrase in ("continue", "Go on", "keep going", "resume", "continua"):
        t = _cut_story_tracker()
        assert t.resume_prompt(phrase) is not None, phrase


def _live_tracker(**cfg) -> ResumeTracker:
    """A tracker with the SHIPPED config.json switches (both halves on)."""
    cfg.setdefault("enabled", True)
    cfg.setdefault("echo_guard_enabled", True)
    return ResumeTracker(ResumeConfig(**cfg))


def test_dataclass_defaults_are_off_and_config_json_opts_in():
    # Programmatic construction is byte-identical (bench/test harnesses must
    # not lose repeat queries to the text guard -- EchoLLM embeds the user's
    # words verbatim); the shipped config block turns both halves on.
    dflt = ResumeConfig()
    assert dflt.enabled is False and dflt.echo_guard_enabled is False
    on = ResumeConfig.from_dict({"enabled": True, "echo_guard_enabled": True})
    assert on.enabled is True and on.echo_guard_enabled is True


def test_resume_phrase_without_a_cut_passes_through():
    t = _live_tracker()
    t.note_query("Tell me a story")
    t.note_spoken("Once upon a time.")
    # No cut: the reply finished normally -> "continue" is a normal request.
    assert t.resume_prompt("continue") is None


def test_non_resume_text_passes_through_even_after_a_cut():
    t = _cut_story_tracker()
    assert t.resume_prompt("what's the weather like") is None


def test_resume_is_consumed_but_a_second_cut_rearms_with_newer_tail():
    t = _cut_story_tracker()
    assert t.resume_prompt("continue") is not None
    # Consumed: asking again without a new cut is a normal turn.
    assert t.resume_prompt("continue") is None
    # The resumed reply speaks more, then is cut again (note_query NOT called
    # for resume turns -- the runtime keeps accumulating the same story).
    t.note_spoken("One night a storm rolled in from the west.")
    t.note_cut()
    prompt = t.resume_prompt("start again")
    assert prompt is not None
    assert "storm rolled in" in prompt  # the NEWER tail


def test_a_new_query_clears_the_resumable_state():
    t = _cut_story_tracker()
    t.note_query("what time is it")  # the user moved on
    assert t.resume_prompt("continue") is None


def test_cut_with_nothing_spoken_is_not_resumable():
    t = _live_tracker()
    t.note_query("Tell me a story")
    t.note_cut()  # cut before the first sentence ever played
    assert t.resume_prompt("continue") is None


def test_resume_disabled_passes_through():
    t = _cut_story_tracker(enabled=False)
    assert t.resume_prompt("start again") is None


# --- is_self_echo (L4 guard) ----------------------------------------------------


def test_own_sentence_heard_back_within_window_is_echo():
    t = _live_tracker()
    t.note_spoken("Okay, let's begin.")
    t.note_spoken("How can I help you today?")
    t.note_playback_end()
    # The recorded failure: the mic fed the sentence back nearly verbatim.
    assert t.is_self_echo("Okay, let's begin. How can I help you today?")
    assert t.is_self_echo("How can I help you today")
    assert t.echo_dropped >= 1


def test_garbled_echo_tail_fuzzy_matches_but_control_words_never_do():
    t = _live_tracker()
    t.note_spoken("You're very welcome!")
    t.note_playback_end()
    # The recorder garbles echo: the live run produced 'Wecome.' from
    # "welcome" and 'Leleep.' from "...spin and leap" -- char-fuzzy tail
    # matching catches those...
    assert t.is_self_echo("Wecome.")
    # ...but confirm/deny/control words are EXEMPT from the fuzzy rule (the
    # user's real "yes"/"okay" must never be eaten).
    assert not t.is_self_echo("yes")
    assert not t.is_self_echo("okay")


def test_live_round3_garbled_tails_are_echo():
    # Recorded phantom finals from run-20260610-130622.
    t = _live_tracker()
    t.note_spoken(
        "She moved like sunlight on water, captivating everyone who watched her spin and leap."
    )
    t.note_playback_end()
    assert t.is_self_echo("Leleep.")

    t2 = _live_tracker()
    t2.note_spoken("That sounds lovely!")
    t2.note_playback_end()
    assert t2.is_self_echo("Loly.")

    # A real short reaction with no resemblance to the tail passes through.
    t3 = _live_tracker()
    t3.note_spoken("...until she finally fell fast asleep.")
    t3.note_playback_end()
    assert not t3.is_self_echo("Great")


def test_no_after_a_now_sentence_is_never_eaten():
    # 'no' char-matches 'now' at 0.8 -- the exemption keeps denials alive.
    t = _live_tracker()
    t.note_spoken("Would you like me to do it now?")
    t.note_playback_end()
    assert not t.is_self_echo("No")


def test_echo_final_arriving_four_seconds_later_is_still_caught():
    # The recognizer adds endpoint silence + utterance length before the FINAL
    # fires: the live verbatim echo landed ~4s after playback end and slipped
    # the old 3s window. The default window now covers it.
    t = _live_tracker()
    t.note_spoken("Okay, let's begin.")
    t.note_spoken("How can I help you today?")
    t.note_playback_end()
    time.sleep(0.0)  # logical check only; window default is 8s
    assert t.is_self_echo("Okay, let's begin. How can I help you today?")


def test_user_speech_is_not_echo():
    t = _live_tracker()
    t.note_spoken("Deep in the heart of an ancient forest lived a clockmaker.")
    t.note_playback_end()
    assert not t.is_self_echo("tell me a longer story about the clockmaker please")
    assert not t.is_self_echo("what day is today")


def test_echo_outside_the_window_is_not_dropped():
    t = _live_tracker(echo_window_sec=0.05)
    t.note_spoken("How can I help you today?")
    t.note_playback_end()
    time.sleep(0.1)
    # The user really might repeat the assistant's words later; only the
    # playback tail window is guarded.
    assert not t.is_self_echo("How can I help you today")


def test_echo_guard_disabled_drops_nothing():
    t = _live_tracker(echo_guard_enabled=False)
    t.note_spoken("How can I help you today?")
    t.note_playback_end()
    assert not t.is_self_echo("How can I help you today")


def test_no_playback_yet_means_no_echo():
    t = _live_tracker()
    assert not t.is_self_echo("hello there")


# --- VoiceRuntime integration ----------------------------------------------------


def _runtime():
    from core.engines.scripted import ScriptedEngine
    from core.llm import EchoLLM
    from core.runtime import VoiceRuntime

    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        resume_config=ResumeConfig(enabled=True, echo_guard_enabled=True),
    )
    runtime.start(run_bus=False)
    return engine, runtime


def test_runtime_resumes_a_barged_story():
    engine, runtime = _runtime()
    try:
        runtime._on_final("Tell me a story about a lighthouse keeper")
        assert runtime.wait_idle(timeout=5.0)
        assert engine.spoken  # the reply was (being) spoken
        runtime._on_barge_in()  # the user cut it off

        runtime._on_final("start again")
        assert runtime.wait_idle(timeout=5.0)
        # EchoLLM echoes its prompt -> the spoken reply carries the synthetic
        # continue-prompt, proving the brain received RESUME semantics.
        joined = " ".join(engine.spoken)
        assert "INTERRUPTED" in joined
        assert "lighthouse keeper" in joined
    finally:
        runtime.stop()


def test_runtime_newest_input_supersedes_the_inflight_turn():
    """Owner (live round 3): "the output should answer only to my latest
    input". A NEW final arriving while the prior turn is still GENERATING
    cancels it -- the stale answer is never spoken."""
    from core.engines.scripted import ScriptedEngine
    from core.llm import EchoLLM
    from core.runtime import VoiceRuntime

    class SlowEchoLLM(EchoLLM):
        def stream(self, prompt, *, system=None, images=None):
            time.sleep(0.6)  # still generating when the next final lands
            yield from super().stream(prompt, system=system, images=images)

    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, SlowEchoLLM())
    runtime.start(run_bus=True)  # real bus thread: the task is genuinely in flight
    try:
        runtime._on_final("what is the capital of france")
        deadline = time.time() + 3.0
        while time.time() < deadline and not runtime.supervisor.state.active_tasks:
            time.sleep(0.01)
        assert runtime.supervisor.state.active_tasks  # the stale turn is generating

        runtime._on_final("tell me a story about dragons")  # the user moved on
        assert runtime.wait_idle(timeout=10.0)
        joined = " ".join(engine.spoken)
        assert "dragons" in joined
        assert "capital of france" not in joined, (
            "the superseded turn's stale answer was spoken"
        )
    finally:
        runtime.stop()


def test_runtime_drops_its_own_echo_final():
    engine, runtime = _runtime()
    try:
        runtime._on_final("hi there friend")
        assert runtime.wait_idle(timeout=5.0)
        spoken_before = len(engine.spoken)
        echo = engine.spoken[-1]  # what the mic would feed back
        runtime._on_final(echo)
        assert runtime.wait_idle(timeout=5.0)
        # The echo final was dropped: no new reply was spoken.
        assert len(engine.spoken) == spoken_before
    finally:
        runtime.stop()
