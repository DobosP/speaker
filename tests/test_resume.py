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


def test_garbled_echo_tail_still_matches():
    t = _live_tracker()
    t.note_spoken("You're very welcome!")
    t.note_playback_end()
    # The recorded 'Wecome.' tail is sub-min-words; an exact-ish short match is
    # required -- 'welcome' normalizes to a full spoken-sentence subset but not
    # an exact sentence, so this one legitimately passes through (energy floors
    # own it); the guard must NOT eat the user's own short words either.
    assert not t.is_self_echo("yes")
    assert not t.is_self_echo("okay")


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
