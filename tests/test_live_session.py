"""Tests for the live-validation harness's NON-AUDIO logic (no models/hardware).

The live run itself (tools/live_session) needs real ASR/TTS/LLM + audio and is
run only on request; these pin the pure pieces -- scenario data, timing parsing,
answer filtering, and report generation -- so they don't rot.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import tools.live_session.__main__ as live_main
from tools.live_session.driver import (
    InjectingInputStream,
    _is_answer,
    _parse_pause,
    apply_inject_profile,
)
from tools.live_session.report import (
    BARGE_STOP_RATE_MIN,
    HEARD_OK_THRESHOLD,
    RESPONSE_OK_THRESHOLD,
    _pctls,
    build_suite_report,
    grade_barge_turns,
    response_aggregate,
    response_rows,
    response_score,
    stt_score,
    summarize_barge,
    summarize_capture,
    write_grade,
    write_latency_report,
    write_suite_report,
    write_summary,
    write_timeline,
)
from tools.live_session.scenarios import (
    BARGE_SCENARIOS,
    LONG_ANSWER,
    LONG_ANSWER_ALT,
    SCENARIOS,
    Scenario,
    Turn,
    by_name,
    resolve_suite,
    suite_names,
)


def test_scenarios_are_well_formed():
    # Bumped for the added barge-in scenarios (early-vs-late, stop-vs-new-topic,
    # multiple-in-one-turn, no-barge control) on top of the original 7.
    assert len(SCENARIOS) >= 11
    names = {s.name for s in SCENARIOS}
    assert len(names) == len(SCENARIOS)  # unique
    for s in SCENARIOS:
        assert s.turns and all(t.text.strip() for t in s.turns)
        for t in s.turns:
            # No new timing token is introduced -- all stay in the allowed set.
            assert t.timing in ("wait_for_response", "immediately", "barge_in") or t.timing.startswith("pause:")


def test_new_barge_scenarios_exist_and_use_long_answer():
    names = {s.name for s in SCENARIOS}
    expected = {
        "barge_in_interrupt_stop",
        "barge_in_early_vs_late",
        "barge_stop_command_vs_new_topic",
        "barge_multiple_in_one_turn",
        "barge_no_barge_control",
    }
    assert expected <= names
    # Every barge scenario uses a shared long-answer target so there is always
    # speech to cut (the short-response-race fix).
    for name in expected:
        s = by_name(name)
        assert any(t.text in (LONG_ANSWER, LONG_ANSWER_ALT) for t in s.turns), name

    # Multi-interrupt scenarios must not repeat the exact prompt: EchoLLM speaks
    # that prompt verbatim, and the runtime should then classify a quick repeat as
    # its own TTS echo rather than creating a second answer.
    for name in (
        "barge_in_interrupt_stop",
        "barge_in_early_vs_late",
        "barge_stop_command_vs_new_topic",
    ):
        targets = [
            t.text for t in by_name(name).turns
            if t.text in (LONG_ANSWER, LONG_ANSWER_ALT)
        ]
        assert targets == [LONG_ANSWER, LONG_ANSWER_ALT]

    stop_turn = by_name("barge_in_interrupt_stop").turns[1]
    assert stop_turn.text == "Stop."
    assert stop_turn.timing == "barge_in"


def test_no_barge_control_scenario_has_no_barge_line():
    s = by_name("barge_no_barge_control")
    # The control must NOT script a barge_in -- its whole point is to prove zero
    # self-interrupts when nothing barges.
    assert all(t.timing != "barge_in" for t in s.turns)
    assert any(t.text == LONG_ANSWER for t in s.turns)


def test_by_name():
    assert by_name("baseline_latency_single_turn_qa").capability.startswith("Single-turn")
    try:
        by_name("nope")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_parse_pause():
    assert _parse_pause("pause:1.5") == 1.5
    assert _parse_pause("pause:3s") == 3.0
    assert _parse_pause("wait_for_response") == 0.0
    assert _parse_pause("barge_in") == 0.0
    assert _parse_pause("pause:garbage") == 0.0


def test_is_answer_filters_control_keeps_apology():
    assert _is_answer("Paris is the capital of France.")
    assert not _is_answer("Mode: assistant")
    assert not _is_answer("Queued research: foo")
    assert not _is_answer("[cancelled]")
    assert not _is_answer("")
    # the timeout/abstain apology is content (not a control prefix) -> kept.
    assert _is_answer("Sorry, that took too long -- let's try again.")
    assert _is_answer("Sorry, I don't have an answer for that.")


def _fake_events():
    return [
        {"idx": 1, "speaker": "user", "text": "what's the capital of france", "timing": "wait_for_response",
         "audio": "user/01.wav", "t_start": 0.1, "t_end": 1.4, "asr_final": "what's the capital of france"},
        {"idx": 1, "speaker": "assistant", "text": "Paris.", "audio": "assistant/01.wav", "t_start": 2.0,
         "latency": {"first_audio_latency": 0.62, "endpoint_latency": 0.18,
                     "final_to_first_token": 0.30, "first_token_to_audio": 0.14, "barge_in_latency": None}},
        {"idx": 2, "speaker": "user", "text": "how many days in a week", "timing": "pause:1.5",
         "audio": "user/02.wav", "t_start": 3.5, "t_end": 4.6, "asr_final": "how many days in a week"},
        {"idx": 2, "speaker": "assistant", "text": "Seven.", "audio": "assistant/02.wav", "t_start": 5.1,
         "latency": {"first_audio_latency": 0.41, "endpoint_latency": 0.16,
                     "final_to_first_token": 0.18, "first_token_to_audio": 0.07, "barge_in_latency": None}},
    ]


def test_write_timeline(tmp_path):
    events = _fake_events()
    path = write_timeline(events, tmp_path)
    loaded = json.loads(path.read_text())
    assert loaded == events
    # attribution: two user + two assistant, each with an audio file
    users = [e for e in loaded if e["speaker"] == "user"]
    assistants = [e for e in loaded if e["speaker"] == "assistant"]
    assert len(users) == 2 and len(assistants) == 2
    assert all(e["audio"] for e in loaded)


def test_write_latency_report_aggregates(tmp_path):
    path = write_latency_report(_fake_events(), tmp_path)
    rep = json.loads(path.read_text())
    assert len(rep["per_turn"]) == 2
    assert rep["per_turn"][0]["first_audio_ms"] == 620.0
    agg = rep["aggregate_first_audio"]
    assert agg["n"] == 2
    assert agg["first_audio_ms_min"] == 410.0
    assert agg["first_audio_ms_max"] == 620.0


# --- capture/attribution logic (with fakes -- no models/audio) ----------------


class _FakeRec:
    def __init__(self, stamps):
        self.stamps = stamps

    def as_dict(self):
        return {"first_audio_latency": 0.5, "endpoint_latency": 0.1,
                "final_to_first_token": 0.3, "first_token_to_audio": 0.1,
                "barge_in_latency": None}


class _FakeMetrics:
    def __init__(self, recs):
        self._recs = recs

    def records(self):
        return self._recs


class _FakeState:
    def __init__(self):
        self.active_tasks = {}
        self.queued_tasks = []
        self.transcript_log = []
        self.spoken_outputs = []


class _FakeSupervisor:
    def __init__(self):
        self.state = _FakeState()


class _FakeEngine:
    is_speaking = False

    def __init__(self):
        self._spoken = []
        self._stops = []

    def speak(self, text, t):
        self._spoken.append((text, t))

    def stop(self, t):
        self._stops.append(t)

    def spoken_count(self):
        return len(self._spoken)

    def spoken_since(self, n):
        return list(self._spoken[n:])

    def stopped_after(self, t):
        return any(s >= t for s in self._stops)


class _FakeRuntime:
    def __init__(self, engine, supervisor, metrics):
        self.engine = engine
        self.supervisor = supervisor
        self.metrics = metrics


def _bare_convo(engine, runtime, tmp_path):
    from tools.live_session.driver import LiveConversation

    c = LiveConversation.__new__(LiveConversation)
    c.engine = engine
    c.runtime = runtime
    c.events = []
    c._t0 = 0.0
    c._uidx = c._aidx = 0
    c._flush_base = c._metrics_consumed = 0
    c._ln_metrics = c._ln_transcript = c._ln_speak = 0
    c._cur_user_event = None
    c._assistant_voice = None
    c._last_user_timing = None
    c._out = tmp_path
    return c


def test_flush_assistant_joins_streamed_sentences_into_one_event(tmp_path):
    from core.metrics import LLM_FIRST_TOKEN, TTS_FIRST_AUDIO

    eng = _FakeEngine()
    eng.speak("Paris is the capital.", 2.0)
    eng.speak("It has about 2 million people.", 2.4)
    sup = _FakeSupervisor()
    # An answered turn stamps a first token (the pairing key) and, when the TTS
    # stamp lands, first audio too.
    metrics = _FakeMetrics([_FakeRec({LLM_FIRST_TOKEN: 1.4, TTS_FIRST_AUDIO: 1.9})])
    c = _bare_convo(eng, _FakeRuntime(eng, sup, metrics), tmp_path)

    c._flush_assistant()
    assert len(c.events) == 1
    ev = c.events[0]
    assert ev["speaker"] == "assistant"
    assert "Paris is the capital." in ev["text"] and "2 million" in ev["text"]
    assert ev["interrupted"] is False
    assert ev["latency"]["first_audio_latency"] == 0.5


def test_flush_assistant_marks_interrupted_when_stopped(tmp_path):
    eng = _FakeEngine()
    eng.speak("Once upon a time there were three little pigs", 3.0)
    eng.stop(3.5)  # barge-in cut it off
    c = _bare_convo(eng, _FakeRuntime(eng, _FakeSupervisor(), _FakeMetrics([])), tmp_path)
    c._flush_assistant()
    assert c.events and c.events[0]["interrupted"] is True


def test_flush_assistant_noop_when_nothing_spoken(tmp_path):
    eng = _FakeEngine()
    c = _bare_convo(eng, _FakeRuntime(eng, _FakeSupervisor(), _FakeMetrics([])), tmp_path)
    c._flush_assistant()
    assert c.events == []


def test_flush_assistant_stamps_barge_intended_from_upcoming_user_timing(tmp_path):
    # A barge_in line interrupts the PRECEDING answer, so the answer is flagged
    # barge_intended by the UPCOMING line's timing (the barge that hits it), not
    # the line before it. The grader uses this to tell a real interrupt from a
    # self-interrupt.
    eng = _FakeEngine()
    eng.speak("Once upon a time...", 1.0)
    eng.stop(1.2)
    c = _bare_convo(eng, _FakeRuntime(eng, _FakeSupervisor(), _FakeMetrics([])), tmp_path)
    c._uidx = 2  # target user 1, then interrupter user 2
    c._upcoming_user_timing = "barge_in"
    c._flush_assistant()
    assert c.events[0]["barge_intended"] is True
    assert c.events[0]["response_to_user_idx"] == 1

    eng2 = _FakeEngine()
    eng2.speak("Paris.", 2.0)
    c2 = _bare_convo(eng2, _FakeRuntime(eng2, _FakeSupervisor(), _FakeMetrics([])), tmp_path)
    c2._uidx = 1
    c2._upcoming_user_timing = "wait_for_response"
    c2._flush_assistant()
    assert c2.events[0]["barge_intended"] is False
    assert c2.events[0]["response_to_user_idx"] == 1


def test_poll_capture_ignores_stale_last_partial_across_turns(tmp_path):
    # supervisor.last_partial is never cleared between turns -- it only ever gets
    # overwritten on a NEW partial. The per-turn probe must baseline it (partial0)
    # so a stale carry-over from an earlier turn does NOT read as a live partial.
    import threading

    eng = _FakeEngine()
    sup = _FakeSupervisor()
    c = _bare_convo(eng, _FakeRuntime(eng, sup, _FakeMetrics([])), tmp_path)
    # _poll_capture_once snapshots heartbeat state; wire the minimal seam.
    c._hb_lock = threading.Lock()
    c._hb_last = None
    c._hb_count = 0

    # Turn 1: a fresh partial appears during the user line -> counted.
    sup.state.last_partial = ""
    c._probe = {"transcript0": 0, "partial0": "", "partials_during_user": 0,
                "hb_gap_max_s": 0.0}
    sup.state.last_partial = "what's the capital"
    c._poll_capture_once(None)
    assert c._probe["partials_during_user"] == 1

    # Turn 2: the mic hears NOTHING, so last_partial keeps its stale turn-1 value.
    # Baselined against that stale value, every poll must count 0.
    c._probe = {"transcript0": len(sup.state.transcript_log),
                "partial0": (sup.state.last_partial or "").strip(),
                "partials_during_user": 0, "hb_gap_max_s": 0.0}
    for _ in range(5):
        c._poll_capture_once(None)  # last_partial unchanged from partial0
    assert c._probe["partials_during_user"] == 0

    # Turn 2, later: a genuinely new partial arrives -> counted again.
    sup.state.last_partial = "how many days in a week"
    c._poll_capture_once(None)
    assert c._probe["partials_during_user"] == 1


def test_consume_latency_pairs_turns_in_order(tmp_path):
    # Pairing is on LLM_FIRST_TOKEN (every answered turn stamps it), NOT
    # TTS_FIRST_AUDIO (TTS-side, racy under load in --inject). A record with a
    # first token but no first-audio stamp must STILL be paired+reported (its
    # endpoint/final->token numbers are the reliable ones); only a record the
    # model never answered (no first token) is skipped.
    from core.metrics import LLM_FIRST_TOKEN, TTS_FIRST_AUDIO

    recs = [
        _FakeRec({LLM_FIRST_TOKEN: 0.5, TTS_FIRST_AUDIO: 1.0}),  # answered + audio
        _FakeRec({}),                                            # never answered -> skip
        _FakeRec({LLM_FIRST_TOKEN: 4.0}),                        # answered, audio MISSING -> still paired
    ]
    c = _bare_convo(_FakeEngine(), _FakeRuntime(_FakeEngine(), _FakeSupervisor(), _FakeMetrics(recs)), tmp_path)
    assert c._consume_latency() is not None  # first answered turn
    assert c._consume_latency() is not None  # third (paired despite no first-audio; skips empty middle)
    assert c._consume_latency() is None      # no more


def test_has_work_and_idle(tmp_path):
    eng = _FakeEngine()
    sup = _FakeSupervisor()
    c = _bare_convo(eng, _FakeRuntime(eng, sup, _FakeMetrics([])), tmp_path)
    # nothing in flight -> no work, idle. (The capture loop won't honor idle until
    # the assistant has ENGAGED -- a task/speech -- which is the premature-idle fix:
    # a bare transcript or empty state no longer counts as "responded".)
    assert c._has_work() is False
    assert c._idle() is True
    # a task in flight -> work, not idle
    sup.state.active_tasks = {"t": object()}
    assert c._has_work() is True
    assert c._idle() is False


def test_inject_barge_waits_for_target_first_audio_and_effective_grace(tmp_path):
    from core.metrics import TTS_FIRST_AUDIO

    now = [10.0]
    stale = _FakeRec({TTS_FIRST_AUDIO: 1.0})
    target = _FakeRec({})
    eng = _FakeEngine()
    eng.is_speaking = True
    eng.config = type("Cfg", (), {
        "barge_in_playback_onset_grace_sec": 0.4,
    })()
    eng._playback_onset_at = 10.0
    runtime = _FakeRuntime(
        eng, _FakeSupervisor(), _FakeMetrics([stale, target])
    )
    c = _bare_convo(eng, runtime, tmp_path)
    c._inject = True
    c._ln_metrics = 1  # the stale reply predates the target user line

    def _sleep(delay):
        now[0] += delay
        if now[0] >= 10.2:
            target.stamps.setdefault(TTS_FIRST_AUDIO, 77.0)

    stamp = c._await_inject_barge_window(
        clock=lambda: now[0], sleeper=_sleep, timeout=1.0, poll_sec=0.05
    )

    assert stamp == 77.0
    # True audio arrived at 10.2, but the effective engine guard is anchored at
    # synthesis onset 10.0 and therefore does not expire until 10.4.
    assert round(now[0], 6) == 10.4


def test_inject_barge_wait_returns_at_late_first_audio_after_grace(tmp_path):
    from core.metrics import TTS_FIRST_AUDIO

    now = [20.0]
    target = _FakeRec({})
    eng = _FakeEngine()
    eng.is_speaking = True
    eng.config = type("Cfg", (), {
        "barge_in_playback_onset_grace_sec": 0.4,
    })()
    eng._playback_onset_at = 20.0
    c = _bare_convo(
        eng,
        _FakeRuntime(eng, _FakeSupervisor(), _FakeMetrics([target])),
        tmp_path,
    )
    c._inject = True
    c._ln_metrics = 0

    def _sleep(delay):
        now[0] += delay
        if now[0] >= 20.6:
            target.stamps.setdefault(TTS_FIRST_AUDIO, 88.0)

    stamp = c._await_inject_barge_window(
        clock=lambda: now[0], sleeper=_sleep, timeout=1.0, poll_sec=0.1
    )

    assert stamp == 88.0
    # The core grace already expired at 20.4; do not add a second 0.4 seconds
    # after true first audio finally arrives.
    assert round(now[0], 6) == 20.6


def test_acoustic_barge_window_helper_is_a_strict_noop(tmp_path):
    eng = _FakeEngine()
    c = _bare_convo(
        eng, _FakeRuntime(eng, _FakeSupervisor(), _FakeMetrics([])), tmp_path
    )
    c._inject = False

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("acoustic scheduling touched inject wait machinery")

    assert c._await_inject_barge_window(
        clock=_must_not_run, sleeper=_must_not_run
    ) is None


def test_inject_barge_wait_fails_when_target_never_reaches_first_audio(tmp_path):
    import pytest

    now = [30.0]
    eng = _FakeEngine()
    eng.is_speaking = True
    eng.config = type("Cfg", (), {
        "barge_in_playback_onset_grace_sec": 0.4,
    })()
    eng._playback_onset_at = 30.0
    c = _bare_convo(
        eng,
        _FakeRuntime(eng, _FakeSupervisor(), _FakeMetrics([_FakeRec({})])),
        tmp_path,
    )
    c._inject = True
    c._ln_metrics = 0

    def _sleep(delay):
        now[0] += delay

    with pytest.raises(RuntimeError, match="true first audio"):
        c._await_inject_barge_window(
            clock=lambda: now[0], sleeper=_sleep, timeout=0.2, poll_sec=0.05
        )


# --- CLI flags: --smart-endpoint (experimental semantic endpoint A/B) ----------
#
# These pin the in-memory config mutation only (no audio/models). main() short-
# circuits on --check after applying the device profile + the flag mutations and
# before constructing LiveConversation, so we capture the config in a stubbed
# _preflight. The committed config.json default must stay OFF; the flag flips it
# ON only for the live run, and its absence is byte-identical to before.


def _run_main_capturing_config(monkeypatch, argv, *, initial_sherpa=None):
    """Drive main(argv) with a minimal stub config + a captured _preflight.

    Returns the config dict as it was when _preflight saw it (i.e. AFTER the
    device-profile merge and the CLI flag mutations, BEFORE LiveConversation).
    """
    import core.config as core_config

    captured: dict = {}

    monkeypatch.setattr(
        core_config,
        "_load_config",
        lambda *a, **k: {
            "device": "desktop",
            "sherpa": dict(initial_sherpa or {}),
        },
    )
    monkeypatch.setattr(core_config, "_apply_device_profile",
                        lambda config, device, **_kwargs: config)

    def _stub_preflight(config, **_kwargs):
        captured["config"] = config
        return ["stop here -- captured config, do not build the live runtime"]

    monkeypatch.setattr(live_main, "_preflight", _stub_preflight)

    # --check makes main() return after _preflight without touching scenarios.
    live_main.main([*argv, "--check"])
    return captured["config"]


def test_smart_endpoint_flag_enables_endpoint(monkeypatch):
    config = _run_main_capturing_config(monkeypatch, ["--smart-endpoint"])
    assert config["sherpa"]["endpoint_enabled"] is True


def test_no_smart_endpoint_flag_leaves_endpoint_untouched(monkeypatch):
    config = _run_main_capturing_config(monkeypatch, [])
    # Default-OFF must be byte-identical to before: the flag's absence does not
    # introduce the key at all (the engine then defaults endpoint_enabled False).
    assert "endpoint_enabled" not in config.get("sherpa", {})


def test_inject_disables_every_echo_only_barge_discriminator(monkeypatch):
    config = _run_main_capturing_config(
        monkeypatch,
        ["--inject"],
        initial_sherpa={
            "aec_enabled": True,
            "capture_voice_comm": True,
            "denoise_enabled": True,
            "barge_word_cut_enabled": True,
            "coherence_barge_in_enabled": True,
            "dtd_enabled": True,
            "barge_in_output_margin_db": 99.0,
            "barge_confirm_enabled": True,
        },
    )
    assert config["sherpa"]["aec_enabled"] is False
    assert config["sherpa"]["capture_voice_comm"] is False
    assert config["sherpa"]["denoise_enabled"] is False
    assert config["sherpa"]["barge_word_cut_enabled"] is False
    assert config["sherpa"]["coherence_barge_in_enabled"] is False
    assert config["sherpa"]["dtd_enabled"] is False
    assert config["sherpa"]["barge_in_output_margin_db"] == 0.0
    assert config["sherpa"]["barge_confirm_enabled"] is False
    assert config["sherpa"]["coherence_warmup_frames"] == 0
    assert config["sherpa"]["coherence_confirm_frames"] == 1
    assert config["sherpa"]["coherence_max_delay_ms"] == 0.0
    assert config["sherpa"]["barge_in_min_speech_sec"] == 0.1


def test_shared_inject_profile_preserves_unrelated_config():
    config = {
        "device": "desktop",
        "sherpa": {
            "aec_enabled": True,
            "capture_voice_comm": True,
            "denoise_enabled": True,
            "barge_word_cut_enabled": True,
            "coherence_barge_in_enabled": True,
            "dtd_enabled": True,
            "barge_in_output_margin_db": 6.0,
            "barge_confirm_enabled": True,
            "coherence_warmup_frames": 5,
            "coherence_confirm_frames": 2,
            "coherence_max_delay_ms": 400.0,
            "barge_in_min_speech_sec": 0.2,
            "tts_speaker_id": 17,
        },
    }
    assert apply_inject_profile(config) is config
    assert config["device"] == "desktop"
    assert config["sherpa"]["tts_speaker_id"] == 17
    assert config["sherpa"]["aec_enabled"] is False
    assert config["sherpa"]["capture_voice_comm"] is False
    assert config["sherpa"]["denoise_enabled"] is False
    assert config["sherpa"]["barge_word_cut_enabled"] is False
    assert config["sherpa"]["coherence_barge_in_enabled"] is False
    assert config["sherpa"]["dtd_enabled"] is False
    assert config["sherpa"]["barge_in_output_margin_db"] == 0.0
    assert config["sherpa"]["barge_confirm_enabled"] is False
    assert config["sherpa"]["coherence_warmup_frames"] == 0
    assert config["sherpa"]["coherence_confirm_frames"] == 1
    assert config["sherpa"]["coherence_max_delay_ms"] == 0.0
    assert config["sherpa"]["barge_in_min_speech_sec"] == 0.1


def test_non_inject_keeps_conservative_physical_coherence_profile(monkeypatch):
    config = _run_main_capturing_config(
        monkeypatch,
        [],
        initial_sherpa={
            "denoise_enabled": True,
            "coherence_warmup_frames": 5,
            "coherence_confirm_frames": 2,
            "coherence_max_delay_ms": 400.0,
            "barge_in_min_speech_sec": 0.2,
            "coherence_barge_in_enabled": True,
            "dtd_enabled": True,
            "barge_in_output_margin_db": 6.0,
            "barge_confirm_enabled": True,
        },
    )
    assert config["sherpa"]["coherence_warmup_frames"] == 5
    assert config["sherpa"]["coherence_confirm_frames"] == 2
    assert config["sherpa"]["coherence_max_delay_ms"] == 400.0
    assert config["sherpa"]["barge_in_min_speech_sec"] == 0.2
    assert config["sherpa"]["denoise_enabled"] is True
    assert config["sherpa"]["coherence_barge_in_enabled"] is True
    assert config["sherpa"]["dtd_enabled"] is True
    assert config["sherpa"]["barge_in_output_margin_db"] == 6.0
    assert config["sherpa"]["barge_confirm_enabled"] is True


def test_inject_profile_uses_one_vad_block_without_echo_discrimination():
    """The no-device topology retains real sustain but has no echo to classify."""
    from core.engines._dtd import BargeSustain

    config = {"sherpa": {}}
    apply_inject_profile(config)
    sherpa = config["sherpa"]
    assert sherpa["coherence_barge_in_enabled"] is False
    assert sherpa["dtd_enabled"] is False
    assert sherpa["barge_in_output_margin_db"] == 0.0
    assert sherpa["barge_confirm_enabled"] is False
    sustain = BargeSustain(
        window_sec=0.5,
        block_sec=0.1,
        min_voiced_sec=sherpa["barge_in_min_speech_sec"],
    )
    assert sustain.update(True) is True


def test_inject_input_stream_paces_to_absolute_device_deadlines():
    now = [0.0]
    sleeps = []

    def _sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    stream = InjectingInputStream(
        100, clock=lambda: now[0], sleeper=_sleep
    )
    stream.start()
    stream.read(10)
    now[0] += 0.03  # caller processing time between device reads
    stream.read(10)

    assert sleeps == [0.1, 0.07]


def test_inject_floor_counter_publishes_only_returned_floor_samples():
    now = [0.0]

    def _sleep(delay):
        now[0] += delay

    stream = InjectingInputStream(10, clock=lambda: now[0], sleeper=_sleep)
    stream.push([1.0, 2.0, 3.0])
    stream.start()

    stream.read(2)
    assert stream.floor_samples_delivered == 0
    stream.read(2)
    assert stream.floor_samples_delivered == 1
    stream.read(2)
    assert stream.floor_samples_delivered == 3


def test_inject_push_receipts_track_partial_and_cross_buffer_consumption():
    now = [0.0]

    def _sleep(delay):
        now[0] += delay

    stream = InjectingInputStream(
        10, clock=lambda: now[0], sleeper=_sleep
    )
    first = stream.push([1.0, 2.0, 3.0])
    second = stream.push([4.0, 5.0])

    assert (first.start_sample, first.end_sample) == (0, 3)
    assert (second.start_sample, second.end_sample) == (3, 5)
    assert first.sample_count == 3
    assert first.duration_seconds == 0.3
    assert first.enqueued_at == second.enqueued_at == 0.0
    assert first.wait_started(0) is False
    assert first.wait_drained(0) is False

    stream.start()
    stream.read(2)
    assert first.first_consumed_at == 0.2
    assert first.fully_consumed_at is None
    assert second.first_consumed_at is None
    assert first.wait_started(0) is True
    assert first.wait_drained(0) is False

    # This device block contains the final sample of the first push and the
    # first sample of the second, so both receipts share its delivery time.
    stream.read(2)
    assert first.fully_consumed_at == 0.4
    assert first.last_consumed_at == 0.4
    assert second.first_consumed_at == 0.4
    assert second.fully_consumed_at is None
    assert first.wait_drained(0) is True

    stream.read(1)
    assert second.fully_consumed_at == 0.5
    assert second.last_consumed_at == 0.5
    assert second.wait_started(0) is True
    assert second.wait_drained(0) is True


def test_inject_push_receipt_publishes_only_after_paced_read_returns():
    now = [0.0]
    observed_during_sleep = []
    receipt = None

    def _sleep(delay):
        observed_during_sleep.append(
            (receipt.first_consumed_at, receipt.fully_consumed_at)
        )
        now[0] += delay

    stream = InjectingInputStream(
        10, clock=lambda: now[0], sleeper=_sleep
    )
    receipt = stream.push([1.0, 2.0])
    stream.start()
    stream.read(2)

    assert observed_during_sleep == [(None, None)]
    assert receipt.first_consumed_at == 0.2
    assert receipt.fully_consumed_at == 0.2


def test_inject_empty_push_receipt_is_immediately_drained():
    now = [7.5]
    stream = InjectingInputStream(10, clock=lambda: now[0])

    receipt = stream.push([])

    assert (receipt.start_sample, receipt.end_sample) == (0, 0)
    assert receipt.sample_count == 0
    assert receipt.duration_seconds == 0.0
    assert receipt.enqueued_at == 7.5
    assert receipt.first_consumed_at == 7.5
    assert receipt.fully_consumed_at == 7.5
    assert receipt.wait_started(0) is True
    assert receipt.wait_drained(0) is True


def test_inject_input_stream_does_not_replay_unbounded_stall_backlog():
    now = [0.0]
    sleeps = []

    def _sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    stream = InjectingInputStream(
        100, clock=lambda: now[0], sleeper=_sleep
    )
    stream.start()
    stream.read(10)
    now[0] += 1.0  # a real capture-thread stall, not routine processing jitter
    stream.read(10)
    stream.read(10)

    # The late read returns once, then the fake device resumes normal cadence;
    # it cannot instantly backfill ten blocks and hide the missing wall time.
    assert [round(delay, 9) for delay in sleeps] == [0.1, 0.1]


def test_invalid_device_profile_fails_cleanly_before_preflight(monkeypatch, capsys):
    import core.config as core_config

    monkeypatch.setattr(core_config, "_load_config", lambda *args, **kwargs: {
        "device": "desktop",
        "device_profiles": {"desktop": {}},
        "sherpa": {},
    })
    rc = live_main.main(["--device", "does-not-exist", "--check"])
    assert rc == 2
    text = capsys.readouterr().out.lower()
    assert "invalid device profile" in text
    assert "desktop" in text


def test_preflight_missing_models_points_at_config_local():
    problems = live_main._preflight(
        {"sherpa": {}},
        llm="echo",
        import_fn=lambda _name: object(),
        exists=lambda _path: False,
        platform="win32",
        require_audio_devices=False,
        require_os_echo_route=False,
    )
    text = "\n".join(problems)
    assert "tools.setup_models" in text
    assert "config.local.json" in text


def test_preflight_rejects_importable_sounddevice_with_no_usable_devices():
    class _NoDevices:
        def query_devices(self, *args, **kwargs):
            raise RuntimeError("no default audio route")

    required = {
        key: f"/m/{key}"
        for key in (
            "asr_tokens", "asr_encoder", "asr_decoder", "asr_joiner",
            "tts_model", "tts_tokens",
        )
    }
    problems = live_main._preflight(
        {"sherpa": required},
        llm="echo",
        sd=_NoDevices(),
        import_fn=lambda _name: object(),
        exists=lambda _path: True,
        platform="win32",
    )
    text = "\n".join(problems).lower()
    assert "audio input" in text
    assert "audio output" in text


def test_preflight_rejects_word_cut_when_ec_module_is_not_routed():
    from tools.doctor import PipeWireState

    required = {
        key: f"/m/{key}"
        for key in (
            "asr_tokens", "asr_encoder", "asr_decoder", "asr_joiner",
            "tts_model", "tts_tokens",
        )
    }

    class _Devices:
        def query_devices(self, *args, **kwargs):
            return {"name": "pipewire", "default_samplerate": 48000.0}

    state = PipeWireState(
        modules=(
            "42 module-echo-cancel source_name=ec_source sink_name=ec_sink"
        ),
        sources="51 ec_source PipeWire float32le RUNNING",
        sinks="52 ec_sink PipeWire float32le RUNNING",
        default_source="alsa_input.raw",
        default_sink="alsa_output.raw",
    )
    problems = live_main._preflight(
        {"sherpa": {
            **required,
            "barge_word_cut_enabled": True,
            "speaker_embedding_model": "/m/speaker.onnx",
            "speaker_enroll_embedding": "/m/enrollment.json",
            "aec_enabled": False,
            "vad_model": "/m/vad.onnx",
        }},
        llm="echo",
        sd=_Devices(),
        import_fn=lambda _name: object(),
        exists=lambda _path: True,
        platform="linux",
        pipewire_state=state,
    )
    text = "\n".join(problems).lower()
    assert "echo-cancel route" in text
    assert "capture" in text and "playback" in text


def test_explicit_inject_preflight_skips_only_physical_devices_and_ec_route():
    class _MustNotQueryDevices:
        def query_devices(self, *args, **kwargs):
            raise AssertionError("inject preflight queried physical audio")

    required = {
        key: f"/m/{key}"
        for key in (
            "asr_tokens", "asr_encoder", "asr_decoder", "asr_joiner",
            "tts_model", "tts_tokens",
        )
    }
    problems = live_main._preflight(
        {"sherpa": {
            **required,
            "barge_word_cut_enabled": True,
            "speaker_embedding_model": "/m/speaker.onnx",
            "speaker_enroll_embedding": "/m/enrollment.json",
            "aec_enabled": False,
            "vad_model": "/m/vad.onnx",
        }},
        llm="echo",
        sd=_MustNotQueryDevices(),
        import_fn=lambda _name: object(),
        exists=lambda _path: True,
        platform="linux",
        require_audio_devices=False,
        require_os_echo_route=False,
    )
    assert problems == []


def test_committed_config_endpoint_enabled_with_validated_min_silence():
    # Smart endpoint was validated on-device (docs/archive/live_validation_run_2026-05-30.md:
    # ~300ms first-audio win, no tail clipping, no sentence splitting) and ENABLED.
    # min_silence MUST stay >= 0.7s: it has to exceed BOTH the decoder lookahead and a
    # typical intra-sentence comma pause -- at 0.5 a run-on ("Hey, what are you, and...")
    # split at the comma. Lowering it risks regressing that, so this pins the floor.
    config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
    assert config["sherpa"]["endpoint_enabled"] is True
    assert config["sherpa"]["endpoint_min_silence_sec"] >= 0.7
    # Adaptive confidence-tiered floor (validated 2026-06-01): a high-confidence
    # completion commits at 0.6 instead of 0.7 (~110ms endpoint win). The floor MUST
    # stay below min_silence (so it's an actual shortening) but not so low it splits
    # a comma run-on -- 0.55 truncated "In one sentence, ...", 0.6 did not.
    floor = config["sherpa"].get("endpoint_high_confidence_floor", 0.0)
    if floor > 0.0:
        assert 0.6 <= floor < config["sherpa"]["endpoint_min_silence_sec"]


def test_smart_endpoint_flag_is_in_help(capsys):
    import pytest

    # --help raises SystemExit(0) after printing usage to stdout.
    with pytest.raises(SystemExit) as exc:
        live_main.main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--smart-endpoint" in help_text
    assert "EXPERIMENTAL" in help_text  # the validation warning is surfaced


def test_write_summary_is_gradeable(tmp_path):
    scenario = Scenario(
        name="demo", capability="cap", goal="g",
        turns=(Turn("hi"),),
        expected_behavior="answers hi",
        pass_signals=("answered once",),
        failure_modes=("dead air",),
    )
    path = write_summary(scenario, _fake_events(), tmp_path, voice={"speaker_id": 1, "speed": 1.1})
    text = path.read_text()
    assert "USER" in text and "ASSISTANT" in text
    assert "answers hi" in text  # expected_behavior surfaced for grading
    assert "answered once" in text and "dead air" in text
    assert "first_audio" in text  # latency table present
    # New honest-grading + full-duplex sections render even without a capture verdict.
    assert "STT accuracy (over-the-air)" in text
    assert "Full-duplex / continuous capture" in text


def test_write_summary_renders_full_duplex_verdict(tmp_path):
    scenario = Scenario(
        name="demo", capability="cap", goal="g",
        turns=(Turn("hi"),),
        expected_behavior="answers hi",
    )
    capture = summarize_capture(
        rec_seconds=30.0, wall_seconds=31.0,
        partials_during_user_total=4, capture_silent_warned=False,
    )
    path = write_summary(
        scenario, _fake_events(), tmp_path,
        voice={"speaker_id": 1, "speed": 1.1, "volume": 0.5}, capture=capture,
    )
    text = path.read_text()
    assert "full_duplex: ok" in text
    assert "partials produced WHILE the user spoke: 4" in text
    assert "volume=0.5" in text  # the SNR the run used is recorded


# --- CLI flags: --input-gain (capture-side SNR knob) ---------------------------
#
# Mirror the --smart-endpoint coverage: --input-gain N mutates the in-memory
# config['sherpa']['input_gain'] only (the exact pattern --no-input-gate uses),
# and its ABSENCE leaves the key untouched so a run with no new flags is
# byte-identical to before.


def test_input_gain_flag_sets_config(monkeypatch):
    config = _run_main_capturing_config(monkeypatch, ["--input-gain", "1.0"])
    assert config["sherpa"]["input_gain"] == 1.0


def test_input_gain_flag_accepts_fractional(monkeypatch):
    config = _run_main_capturing_config(monkeypatch, ["--input-gain", "2.5"])
    assert config["sherpa"]["input_gain"] == 2.5


def test_no_input_gain_flag_leaves_config_untouched(monkeypatch):
    config = _run_main_capturing_config(monkeypatch, [])
    # Default: the key is NOT introduced, so the engine keeps its configured gain.
    assert "input_gain" not in config.get("sherpa", {})


def test_input_gain_flag_is_in_help(capsys):
    import pytest

    with pytest.raises(SystemExit) as exc:
        live_main.main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--input-gain" in help_text
    assert "--user-volume" in help_text


def test_user_volume_out_of_range_errors(capsys):
    import pytest

    # argparse error() exits with code 2 and writes usage to stderr.
    with pytest.raises(SystemExit) as exc:
        live_main.main(["--user-volume", "1.5", "--check"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--user-volume" in err


def test_user_volume_in_range_parses(monkeypatch):
    # A valid --user-volume does not raise and does not mutate sherpa config
    # (it is a playback-side knob threaded into LiveConversation, not config).
    config = _run_main_capturing_config(monkeypatch, ["--user-volume", "0.4"])
    assert "input_gain" not in config.get("sherpa", {})


# --- SyntheticUser playback volume scaling (fake sd, no audio/models) ----------


class _FakeTTSAudio:
    def __init__(self, samples, sample_rate):
        self.samples = samples
        self.sample_rate = sample_rate


class _FakeTTS:
    sample_rate = 22050

    def generate(self, text, sid=0, speed=1.0):
        import numpy as np

        # A deterministic full-scale-ish tone so peak scaling is observable.
        return _FakeTTSAudio(
            (np.ones(2205, dtype="float32") * 0.8), 22050
        )


def _make_user(monkeypatch, volume):
    """Build a SyntheticUser with the TTS + device-norm seams faked out."""
    import core.engines._sherpa_models as sherpa_models
    import core.engines.sherpa as sherpa_engine
    from tools.live_session.synthetic_user import SyntheticUser

    monkeypatch.setattr(sherpa_models, "build_tts", lambda cfg: _FakeTTS())
    monkeypatch.setattr(sherpa_engine, "_norm_device", lambda d: None)

    class _Cfg:
        tts_speaker_id = 0
        tts_speed = 1.0

    return SyntheticUser(_Cfg(), speaker_id=0, speed=1.0, volume=volume)


def test_user_volume_scales_played_peak_not_saved(monkeypatch):
    import numpy as np

    played = {}

    class _FakeSD:
        @staticmethod
        def query_devices(dev, kind):
            return {"default_samplerate": 22050}

        @staticmethod
        def play(buf, rate, device=None):
            played["peak"] = float(np.max(np.abs(np.asarray(buf))))

        @staticmethod
        def wait():
            pass

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", _FakeSD)

    user_full = _make_user(monkeypatch, volume=1.0)
    samples_full, sr = user_full.say("hello")
    full_played_peak = played["peak"]
    full_saved_peak = float(np.max(np.abs(samples_full)))

    user_half = _make_user(monkeypatch, volume=0.5)
    samples_half, _ = user_half.say("hello")
    half_played_peak = played["peak"]
    half_saved_peak = float(np.max(np.abs(samples_half)))

    # The PLAYED peak scales with volume...
    assert half_played_peak < full_played_peak
    assert abs(half_played_peak - 0.5 * full_played_peak) < 1e-3
    # ...but the RETURNED/saved samples stay full-scale (clean reference).
    assert abs(half_saved_peak - full_saved_peak) < 1e-6


def test_user_volume_clamps_to_unit_interval(monkeypatch):
    u_hi = _make_user(monkeypatch, volume=5.0)
    assert u_hi.voice["volume"] == 1.0
    u_lo = _make_user(monkeypatch, volume=-2.0)
    assert u_lo.voice["volume"] == 0.0
    u_bad = _make_user(monkeypatch, volume=None)  # type: ignore[arg-type]
    assert u_bad.voice["volume"] == 1.0


# --- over-the-air STT scorer (pure) --------------------------------------------


def test_stt_score_exact_match_is_one():
    assert stt_score("What's the capital of France?", "what's the capital of france") == 1.0
    # Normalization handles punctuation + case + whitespace.
    assert stt_score("Paris.", "  PARIS  ") == 1.0


def test_stt_score_total_garbage_is_near_zero():
    score = stt_score("What's the capital of France?", "I did those completely")
    assert score < 0.25


def test_stt_score_empty_heard_is_zero():
    assert stt_score("anything", None) == 0.0
    assert stt_score("anything", "") == 0.0


def test_stt_score_partial_overlap_is_monotonic():
    truth = "what is the capital of france"
    garbage = stt_score(truth, "completely unrelated words here")
    partial = stt_score(truth, "what is the capital")
    perfect = stt_score(truth, "what is the capital of france")
    assert garbage < partial < perfect
    assert perfect == 1.0


def test_stt_score_heard_ok_threshold():
    # A close-but-imperfect transcript still clears the heard-ok bar.
    assert stt_score("the capital of france", "the capital of franc") >= HEARD_OK_THRESHOLD


# --- continuous-capture / full-duplex verdict (pure) ---------------------------


def test_capture_verdict_healthy_is_ok():
    v = summarize_capture(
        rec_seconds=30.0, wall_seconds=31.0,
        partials_during_user_total=3, capture_silent_warned=False,
    )
    assert v["full_duplex"] == "ok" and v["ok"] is True
    assert v["transcribed_during_user"] is True
    assert v["covered_whole_session"] is True


def test_capture_verdict_silent_gap_fails():
    v = summarize_capture(
        rec_seconds=30.0, wall_seconds=31.0,
        partials_during_user_total=3, capture_silent_warned=True,
    )
    assert v["full_duplex"] == "FAIL" and v["ok"] is False


def test_capture_verdict_no_partials_during_user_fails():
    # Recording covered the session but the recognizer never produced a partial
    # while the user audio played -> we can't claim full-duplex transcription.
    v = summarize_capture(
        rec_seconds=30.0, wall_seconds=31.0,
        partials_during_user_total=0, capture_silent_warned=False,
    )
    assert v["ok"] is False
    assert v["transcribed_during_user"] is False


def test_capture_verdict_short_recording_fails():
    # The recorder fell far short of wall-clock -> capture paused -> FAIL.
    v = summarize_capture(
        rec_seconds=5.0, wall_seconds=31.0,
        partials_during_user_total=3, capture_silent_warned=False,
    )
    assert v["ok"] is False
    assert v["covered_whole_session"] is False


# --- write_grade artifact ------------------------------------------------------


def test_write_grade_emits_per_turn_and_aggregate(tmp_path):
    events = _fake_events()
    capture = summarize_capture(
        rec_seconds=10.0, wall_seconds=10.5,
        partials_during_user_total=2, capture_silent_warned=False,
    )
    report = write_grade(events, tmp_path, capture=capture)
    loaded = json.loads((tmp_path / "grade.json").read_text())
    assert loaded == report
    assert len(report["per_turn"]) == 2  # two user turns
    # Both fake user turns are heard verbatim -> perfect scores.
    assert all(r["stt_score"] == 1.0 and r["heard_ok"] for r in report["per_turn"])
    agg = report["aggregate"]
    assert agg["n"] == 2 and agg["n_correct"] == 2
    assert agg["stt_score_median"] == 1.0
    assert report["full_duplex"]["full_duplex"] == "ok"
    # First-audio latency is paired in order onto the user turn.
    assert report["per_turn"][0]["first_audio_ms"] == 620.0


def test_write_grade_uses_explicit_response_links_after_no_answer_turn(tmp_path):
    events = [
        {"idx": 1, "speaker": "user", "text": "first", "asr_final": "first"},
        {
            "idx": 1, "speaker": "assistant", "text": "answer one",
            "response_to_user_idx": 1,
            "latency": {"first_audio_latency": 0.5},
        },
        {"idx": 2, "speaker": "user", "text": "Stop.", "asr_final": None},
        {"idx": 3, "speaker": "user", "text": "third", "asr_final": "third"},
        {
            "idx": 2, "speaker": "assistant", "text": "answer three",
            "response_to_user_idx": 3,
            "latency": {"first_audio_latency": 0.7},
        },
    ]
    rows = write_grade(events, tmp_path)["per_turn"]
    assert [row["first_audio_ms"] for row in rows] == [500.0, None, 700.0]


# --- barge-in / interrupt grade (pure, inject-mode-only logic) ------------------
#
# Fakes only -- the same shape the driver writes onto each assistant event:
#   * barge_intended (the PRECEDING user line's timing == "barge_in"),
#   * interrupted (engine.stopped_after -- proof a stop call occurred),
#   * latency.barge_in_latency (BARGE_IN_STOP minus BARGE_IN, in SECONDS; None when
#     stop_speaking found no live FIFO to cut -- the short-answer race).


def _user(idx, timing="wait_for_response"):
    return {"idx": idx, "speaker": "user", "text": "x", "timing": timing}


def _assistant(idx, *, barge_intended, interrupted, barge_in_latency=None):
    return {
        "idx": idx, "speaker": "assistant", "text": "...",
        "interrupted": interrupted, "barge_intended": barge_intended,
        "latency": {"first_audio_latency": 0.5, "barge_in_latency": barge_in_latency},
    }


def test_grade_barge_turns_marks_stop_and_self_interrupt():
    events = [
        # A real, intended barge that stopped a live stream -> stopped, NOT a
        # self-interrupt; stop_ms is the latency * 1000.
        _user(1, "barge_in"),
        _assistant(1, barge_intended=True, interrupted=True, barge_in_latency=0.12),
        # A NON-barge turn that nonetheless carries a stray barge_in_latency (a
        # barge fired with no intended barge) -> self-interrupt.
        _user(2, "wait_for_response"),
        _assistant(2, barge_intended=False, interrupted=False, barge_in_latency=0.20),
    ]
    rows = grade_barge_turns(events)
    assert len(rows) == 2
    r1, r2 = rows
    assert r1["barge_intended"] is True and r1["stopped"] is True
    assert r1["stop_ms"] == 120.0 and r1["self_interrupt"] is False
    # The stray barge on a non-intended turn is the self-interruption signal.
    assert r2["barge_intended"] is False and r2["self_interrupt"] is True
    assert r2["stop_ms"] == 200.0


def test_grade_barge_turns_interrupted_without_intended_is_self_interrupt():
    # interrupted=True with no intended barge before it (and no latency stamp) is
    # STILL a self-interrupt -- interrupted is the primary signal.
    events = [
        _user(1, "wait_for_response"),
        _assistant(1, barge_intended=False, interrupted=True, barge_in_latency=None),
    ]
    rows = grade_barge_turns(events)
    assert rows[0]["self_interrupt"] is True
    assert rows[0]["stop_called"] is True
    assert rows[0]["stopped"] is False
    assert rows[0]["stop_ms"] is None  # the short-answer race: no STOP stamp


def test_intended_stop_call_without_fifo_cut_cannot_pass():
    events = [
        _user(1, "barge_in"),
        _assistant(1, barge_intended=True, interrupted=True, barge_in_latency=None),
    ]
    rows = grade_barge_turns(events)
    assert rows[0]["stop_called"] is True
    assert rows[0]["stopped"] is False
    summary = summarize_barge(rows)
    assert summary["n_stopped"] == 0
    assert summary["verdict"] == "FAIL"


def test_summarize_barge_headline_ok_path():
    events = [
        _user(1, "barge_in"),
        _assistant(1, barge_intended=True, interrupted=True, barge_in_latency=0.10),
        _user(2, "barge_in"),
        _assistant(2, barge_intended=True, interrupted=True, barge_in_latency=0.30),
    ]
    s = summarize_barge(grade_barge_turns(events))
    assert s["n_intended_barges"] == 2 and s["n_stopped"] == 2
    assert s["stops_when_barged_rate"] == 1.0
    assert s["stop_latency_ms_median"] == 200.0  # median of 100, 300
    assert s["self_interrupt_count"] == 0
    assert s["verdict"] == "ok"
    assert s["inject_mode_only"] is True


def test_summarize_barge_fail_on_missed_stop():
    # An intended barge that did NOT stop the answer drops the rate below the
    # threshold -> FAIL.
    events = [
        _user(1, "barge_in"),
        _assistant(1, barge_intended=True, interrupted=True, barge_in_latency=0.10),
        _user(2, "barge_in"),
        _assistant(2, barge_intended=True, interrupted=False, barge_in_latency=None),
    ]
    s = summarize_barge(grade_barge_turns(events))
    assert s["n_stopped"] == 1 and s["n_intended_barges"] == 2
    assert s["stops_when_barged_rate"] == 0.5
    assert s["stops_when_barged_rate"] < BARGE_STOP_RATE_MIN
    assert s["verdict"] == "FAIL"


def test_summarize_barge_fail_on_self_interrupt():
    # Even with a perfect stop rate, a single self-interrupt FAILs the verdict.
    events = [
        _user(1, "barge_in"),
        _assistant(1, barge_intended=True, interrupted=True, barge_in_latency=0.10),
        _user(2, "wait_for_response"),
        _assistant(2, barge_intended=False, interrupted=True, barge_in_latency=None),
    ]
    s = summarize_barge(grade_barge_turns(events))
    assert s["stops_when_barged_rate"] == 1.0  # the one intended barge stopped
    assert s["self_interrupt_count"] == 1
    assert s["verdict"] == "FAIL"


def test_summarize_barge_no_barge_control_zero_self_interrupts():
    # The CONTROL: a long answer + a closer, NO barge anywhere. n_intended==0, the
    # rate is None ("n/a"), and the verdict keys ONLY off zero self-interrupts.
    events = [
        _user(1, "wait_for_response"),
        _assistant(1, barge_intended=False, interrupted=False, barge_in_latency=None),
        _user(2, "wait_for_response"),
        _assistant(2, barge_intended=False, interrupted=False, barge_in_latency=None),
    ]
    s = summarize_barge(grade_barge_turns(events))
    assert s["n_intended_barges"] == 0
    assert s["stops_when_barged_rate"] is None
    assert s["self_interrupt_count"] == 0
    assert s["verdict"] == "ok"


def test_summarize_barge_control_with_self_interrupt_fails():
    # The regression catch: the no-barge control where the assistant interrupted
    # ITSELF must FAIL even though no barge was intended.
    events = [
        _user(1, "wait_for_response"),
        _assistant(1, barge_intended=False, interrupted=True, barge_in_latency=0.15),
    ]
    s = summarize_barge(grade_barge_turns(events))
    assert s["n_intended_barges"] == 0
    assert s["self_interrupt_count"] == 1
    assert s["verdict"] == "FAIL"


def test_summarize_barge_multiple_barges_benign_second_not_penalized():
    # The multiple-in-one-turn shape: the first intended barge stops a long answer;
    # the second intended barge hits an idle assistant and produces NO new answer
    # (a benign no-op -- no assistant event). Only the first turn exists, intended
    # and stopped -> rate 1/1, zero self-interrupts, verdict ok.
    events = [
        _user(1, "barge_in"),
        _assistant(1, barge_intended=True, interrupted=True, barge_in_latency=0.11),
        _user(2, "barge_in"),
        # second barge: no assistant event produced (idle) -- nothing to grade.
    ]
    s = summarize_barge(grade_barge_turns(events))
    assert s["n_intended_barges"] == 1 and s["n_stopped"] == 1
    assert s["stops_when_barged_rate"] == 1.0
    assert s["self_interrupt_count"] == 0
    assert s["verdict"] == "ok"


def test_write_grade_includes_barge_block(tmp_path):
    events = [
        _user(1, "barge_in"),
        _assistant(1, barge_intended=True, interrupted=True, barge_in_latency=0.12),
        _user(2, "wait_for_response"),
        _assistant(2, barge_intended=False, interrupted=False, barge_in_latency=None),
    ]
    report = write_grade(events, tmp_path)
    loaded = json.loads((tmp_path / "grade.json").read_text())
    assert loaded == report
    assert "barge_in" in report
    b = report["barge_in"]
    assert b["n_intended_barges"] == 1 and b["n_stopped"] == 1
    assert b["self_interrupt_count"] == 0
    assert b["verdict"] == "ok"
    assert b["inject_mode_only"] is True


def test_write_summary_renders_barge_section(tmp_path):
    scenario = Scenario(
        name="barge_demo", capability="cap", goal="g",
        turns=(Turn(LONG_ANSWER), Turn("Stop.", "barge_in")),
        expected_behavior="stops on barge",
        pass_signals=("FIFO cut",),
        failure_modes=("answer continued",),
    )
    events = [
        _user(1, "wait_for_response"),
        _assistant(1, barge_intended=True, interrupted=True, barge_in_latency=0.13),
    ]
    path = write_summary(
        scenario, events, tmp_path, voice={"speaker_id": 1, "speed": 1.0},
    )
    text = path.read_text()
    assert "Barge-in / interrupt" in text
    assert "INJECT MODE ONLY" in text  # the inject-mode caveat is loud
    assert "stops-when-barged" in text
    assert "self-interrupts" in text
    assert "excludes user-onset-to-detection" in text
    assert "Manual pass checks" in text
    assert "[pass]" not in text and "[fail]" not in text
    # The per-turn table header is present.
    assert "barge_intended" in text and "stop_ms" in text


def test_write_summary_labels_inject_user_time_as_enqueue(tmp_path):
    scenario = Scenario(
        name="inject_time", capability="cap", goal="g",
        turns=(Turn("hello"),),
    )
    events = [{
        "idx": 1,
        "speaker": "user",
        "text": "hello",
        "timing": "wait_for_response",
        "time_basis": "capture_enqueue",
        "t_enqueue": 1.25,
        "capture_buffered_seconds": 0.8,
        "asr_final": "hello",
    }]
    text = write_summary(
        scenario, events, tmp_path, voice={"speaker_id": 1, "speed": 1.0},
    ).read_text()
    assert "USER ENQUEUED" in text
    assert "capture enqueue times" in text
    assert "not a measured consumption/overlap interval" in text


def test_write_summary_renders_no_barge_control_verdict(tmp_path):
    scenario = Scenario(
        name="control", capability="cap", goal="g",
        turns=(Turn(LONG_ANSWER),), expected_behavior="no self-interrupt",
    )
    events = [
        _user(1, "wait_for_response"),
        _assistant(1, barge_intended=False, interrupted=False, barge_in_latency=None),
    ]
    text = write_summary(
        scenario, events, tmp_path, voice={"speaker_id": 1, "speed": 1.0},
    ).read_text()
    # The control's barge verdict is "ok" with an n/a rate (no intended barge).
    assert "Verdict: ok" in text
    assert "control" in text  # the n/a-rate annotation names the control case


# --- response-quality scorer (pure) --------------------------------------------


def test_response_score_all_concepts_present_is_one():
    r = response_score(("paris",), (), "The capital of France is Paris.")
    assert r["score"] == 1.0 and r["ok"] is True
    assert r["matched"] == ["paris"] and r["missing"] == []
    assert r["forbidden_hit"] == []


def test_response_score_fraction_of_concepts():
    # Two expected concepts, only one present -> 0.5.
    r = response_score(("red", "blue"), (), "The flag is red and white.")
    assert r["score"] == 0.5
    assert "red" in r["matched"] and "blue" in r["missing"]
    assert r["ok"] is False  # 0.5 < RESPONSE_OK_THRESHOLD


def test_response_score_alternatives_with_pipe():
    # "seven|7" matches either spelling.
    assert response_score(("seven|7",), (), "There are seven days.")["score"] == 1.0
    assert response_score(("seven|7",), (), "There are 7 days.")["score"] == 1.0
    assert response_score(("seven|7",), (), "There are five days.")["score"] == 0.0


def test_response_score_digit_matches_whole_token_not_substring():
    # A digit alternative must match a whole token: "4" must NOT hit inside "40".
    assert response_score(("4",), (), "The answer is 40 apples.")["score"] == 0.0
    assert response_score(("4",), (), "The answer is 4.")["score"] == 1.0


def test_response_score_alpha_matches_substring_plural():
    # An alphabetic concept matches as a substring, so "moon" hits "moons" and
    # "freeze" hits "freezes" (the realistic-scenario plural/inflection case).
    assert response_score(("moon",), (), "Jupiter has 95 moons.")["score"] == 1.0
    assert response_score(("freeze|ice",), (), "Water freezes into ice.")["score"] == 1.0


def test_response_score_forbidden_hit_flags_and_fails():
    # The honesty probe: the assistant falsely claims it saved a note.
    r = response_score(
        (), ("i've saved|i saved|noted that",),
        "Got it, I've saved that note for you.",
    )
    assert r["forbidden_hit"]  # the false claim is caught
    assert r["ok"] is False
    # Score is 1.0 (nothing was *expected*) but ok is killed by the forbidden hit.
    assert r["score"] == 1.0


def test_response_score_forbid_only_clean_answer_is_ok():
    # A truthful decline (no forbidden phrase) on a forbid-only turn passes.
    r = response_score(
        (), ("i've saved|i saved",),
        "I can't actually save notes, but I can help you remember it now.",
    )
    assert r["forbidden_hit"] == [] and r["ok"] is True and r["score"] == 1.0


def test_response_score_empty_answer_scores_zero_when_expected():
    assert response_score(("paris",), (), "")["score"] == 0.0
    assert response_score(("paris",), (), None)["score"] == 0.0
    # ...but a forbid-only turn with no answer is vacuously ok (no false claim).
    assert response_score((), ("i saved",), "")["ok"] is True


def test_response_score_normalizes_punctuation_and_case():
    # Spelling answers come back hyphenated/spaced; normalization makes them match.
    assert response_score(("m-a-r-s",), (), "You spell it M-A-R-S.")["score"] == 1.0


def test_response_score_normalizes_temperature_symbols():
    result = response_score(
        ("water", "zero|0", "celsius"),
        (),
        "Water freezes at 0°C.",
    )

    assert result["score"] == 1.0
    assert result["ok"] is True


# --- response_rows pairing (scenario turns -> the answer that followed) ---------


def _scenario_with(turns):
    return Scenario(name="resp_demo", capability="c", goal="g", turns=tuple(turns))


def test_response_rows_pairs_turn_to_following_answer():
    scenario = _scenario_with([
        Turn("What's the capital of France?", "wait_for_response", expect=("paris",)),
        Turn("How many days in a week?", "wait_for_response", expect=("seven|7",)),
    ])
    events = [
        {"idx": 1, "speaker": "user", "text": "what's the capital of france"},
        {"idx": 1, "speaker": "assistant", "text": "Paris is the capital."},
        {"idx": 2, "speaker": "user", "text": "how many days in a week"},
        {"idx": 2, "speaker": "assistant", "text": "There are seven days."},
    ]
    rows = response_rows(scenario, events)
    assert len(rows) == 2
    assert rows[0]["turn"] == 1 and rows[0]["score"] == 1.0
    assert rows[0]["answer"] == "Paris is the capital."
    assert rows[1]["turn"] == 2 and rows[1]["score"] == 1.0


def test_response_rows_skips_ungraded_turns():
    scenario = _scenario_with([
        Turn("Good morning, weather?", "wait_for_response"),  # no expect/forbid
        Turn("Capital of Japan?", "wait_for_response", expect=("tokyo",)),
    ])
    events = [
        {"idx": 1, "speaker": "user", "text": "good morning weather"},
        {"idx": 1, "speaker": "assistant", "text": "I don't have live weather."},
        {"idx": 2, "speaker": "user", "text": "capital of japan"},
        {"idx": 2, "speaker": "assistant", "text": "Tokyo."},
    ]
    rows = response_rows(scenario, events)
    # Only the graded (expect-bearing) turn appears.
    assert len(rows) == 1 and rows[0]["turn"] == 2 and rows[0]["score"] == 1.0


def test_response_rows_joins_multi_sentence_answer():
    scenario = _scenario_with([
        Turn("Primary colors?", "wait_for_response", expect=("red", "blue", "yellow")),
    ])
    events = [
        {"idx": 1, "speaker": "user", "text": "primary colors"},
        {"idx": 1, "speaker": "assistant", "text": "The primary colors are red."},
        {"idx": 1, "speaker": "assistant", "text": "Also blue and yellow."},
    ]
    rows = response_rows(scenario, events)
    assert rows[0]["score"] == 1.0  # all three across the joined sentences


def test_response_rows_skips_immediately_and_barge_in_turns():
    # A graded turn scheduled as a merge/interrupt does NOT pair 1:1 with the next
    # answer, so it must be skipped (mis-attribution guard) even if it has expect.
    scenario = _scenario_with([
        Turn("clean turn", "wait_for_response", expect=("paris",)),
        Turn("addon turn", "immediately", expect=("tokyo",)),
        Turn("barge turn", "barge_in", expect=("london",)),
    ])
    events = [
        {"idx": 1, "speaker": "user", "text": "clean turn"},
        {"idx": 1, "speaker": "assistant", "text": "Paris."},
        {"idx": 2, "speaker": "user", "text": "addon turn"},
        {"idx": 2, "speaker": "assistant", "text": "Tokyo."},
        {"idx": 3, "speaker": "user", "text": "barge turn"},
        {"idx": 3, "speaker": "assistant", "text": "London."},
    ]
    rows = response_rows(scenario, events)
    assert len(rows) == 1 and rows[0]["turn"] == 1  # only the wait_for_response turn


def test_response_rows_short_run_does_not_crash():
    # The run timed out: the second turn never produced a user event.
    scenario = _scenario_with([
        Turn("Q1", "wait_for_response", expect=("a",)),
        Turn("Q2", "wait_for_response", expect=("b",)),
    ])
    events = [
        {"idx": 1, "speaker": "user", "text": "q1"},
        {"idx": 1, "speaker": "assistant", "text": "a"},
    ]
    rows = response_rows(scenario, events)
    assert len(rows) == 1 and rows[0]["turn"] == 1


def test_response_aggregate_headline():
    rows = [
        {"score": 1.0, "ok": True, "forbidden_hit": []},
        {"score": 0.5, "ok": False, "forbidden_hit": []},
        {"score": 1.0, "ok": True, "forbidden_hit": []},
    ]
    agg = response_aggregate(rows)
    assert agg["n"] == 3 and agg["n_ok"] == 2
    assert agg["response_score_median"] == 1.0
    assert agg["response_score_min"] == 0.5
    assert agg["n_forbidden_hits"] == 0
    assert agg["verdict"] == "ok"


def test_response_aggregate_forbidden_hit_fails_verdict():
    rows = [
        {"score": 1.0, "ok": False, "forbidden_hit": ["i saved"]},
        {"score": 1.0, "ok": True, "forbidden_hit": []},
    ]
    agg = response_aggregate(rows)
    assert agg["n_forbidden_hits"] == 1 and agg["verdict"] == "FAIL"


def test_response_aggregate_empty_is_n_zero():
    assert response_aggregate([]) == {"n": 0}


# --- latency percentiles (pure) ------------------------------------------------


def test_pctls_basic_distribution():
    d = _pctls([100, 200, 300, 400, 500])
    assert d["n"] == 5 and d["min"] == 100.0 and d["max"] == 500.0
    assert d["p50"] == 300.0
    assert d["mean"] == 300.0


def test_pctls_single_value():
    d = _pctls([1184.2])
    assert d["n"] == 1
    assert d["p50"] == d["p90"] == d["p99"] == d["min"] == d["max"] == 1184.2


def test_pctls_empty_is_empty():
    assert _pctls([]) == {}
    assert _pctls([None, None]) == {}  # type: ignore[list-item]


def test_pctls_p90_interpolates():
    # 10 values 0..90; p90 sits at the 9th order statistic edge.
    d = _pctls(list(range(0, 100, 10)))  # 0,10,...,90
    assert d["p50"] == 45.0
    assert d["p90"] >= 80.0  # high tail


def test_latency_report_has_distribution_and_stages(tmp_path):
    path = write_latency_report(_fake_events(), tmp_path)
    rep = json.loads(path.read_text())
    agg = rep["aggregate_first_audio"]
    # Original keys preserved...
    assert agg["n"] == 2 and agg["first_audio_ms_min"] == 410.0
    # ...new distribution keys added.
    assert "first_audio_ms_p50" in agg and "first_audio_ms_p90" in agg
    assert "first_audio_ms_p99" in agg and "first_audio_ms_mean" in agg
    # Per-stage distribution present for endpoint/LLM/TTS.
    assert "endpoint_ms" in rep["stages"]
    assert rep["stages"]["endpoint_ms"]["n"] == 2


# --- write_grade with a scenario includes the response block -------------------


def test_write_grade_includes_response_when_scenario_given(tmp_path):
    scenario = _scenario_with([
        Turn("what's the capital of france", "wait_for_response", expect=("paris",)),
        Turn("how many days in a week", "pause:1.5", expect=("seven|7",)),
    ])
    # _fake_events answers "Paris." then "Seven." -> both graded ok.
    report = write_grade(_fake_events(), tmp_path, scenario=scenario)
    assert "response" in report
    resp = report["response"]
    assert resp["aggregate"]["n"] == 2
    assert resp["aggregate"]["verdict"] == "ok"
    assert all(r["score"] == 1.0 for r in resp["per_turn"])


def test_write_grade_without_scenario_has_empty_response(tmp_path):
    # Backward compat: no scenario -> response section present but empty.
    report = write_grade(_fake_events(), tmp_path)
    assert report["response"]["per_turn"] == []
    assert report["response"]["aggregate"] == {"n": 0}


def test_write_summary_renders_response_section(tmp_path):
    scenario = _scenario_with([
        Turn("what's the capital of france", "wait_for_response", expect=("paris",)),
        Turn("how many days in a week", "pause:1.5", expect=("seven|7",)),
    ])
    text = write_summary(
        scenario, _fake_events(), tmp_path, voice={"speaker_id": 1, "speed": 1.0},
    ).read_text()
    assert "Response quality" in text
    assert "median response score" in text
    # The latency distribution line is rendered too.
    assert "first_audio distribution" in text
    assert "where the time goes" in text


# --- suite report (pooled cross-scenario dashboard) ----------------------------


def test_build_suite_report_pools_across_scenarios():
    s1 = _scenario_with([Turn("capital of france", "wait_for_response", expect=("paris",))])
    s2 = _scenario_with([Turn("days in a week", "wait_for_response", expect=("seven|7",))])
    run1 = {
        "scenario": s1,
        "events": [
            {"idx": 1, "speaker": "user", "text": "capital of france",
             "asr_final": "capital of france"},
            {"idx": 1, "speaker": "assistant", "text": "Paris.",
             "latency": {"first_audio_latency": 1.0, "endpoint_latency": 0.9,
                         "final_to_first_token": 0.08, "first_token_to_audio": 0.02,
                         "barge_in_latency": None}},
        ],
        "capture": summarize_capture(rec_seconds=10.0, wall_seconds=10.5,
                                     partials_during_user_total=2, capture_silent_warned=False),
    }
    run2 = {
        "scenario": s2,
        "events": [
            {"idx": 1, "speaker": "user", "text": "days in a week",
             "asr_final": "days in a week"},
            {"idx": 1, "speaker": "assistant", "text": "Seven.",
             "latency": {"first_audio_latency": 2.0, "endpoint_latency": 1.8,
                         "final_to_first_token": 0.15, "first_token_to_audio": 0.05,
                         "barge_in_latency": None}},
        ],
        "capture": None,
    }
    report = build_suite_report([run1, run2])
    assert report["n_scenarios"] == 2
    fa = report["latency"]["first_audio_ms"]
    assert fa["n"] == 2 and fa["min"] == 1000.0 and fa["max"] == 2000.0
    assert fa["p50"] == 1500.0
    # endpoint dominates first-audio in the pooled per-stage view.
    assert report["latency"]["stages"]["endpoint_ms"]["n"] == 2
    # Pooled STT + response.
    assert report["stt"]["n"] == 2
    assert report["response"]["n"] == 2 and report["response"]["verdict"] == "ok"
    assert len(report["per_scenario"]) == 2


def test_write_suite_report_writes_files(tmp_path):
    s1 = _scenario_with([Turn("capital of france", "wait_for_response", expect=("paris",))])
    run1 = {
        "scenario": s1,
        "events": [
            {"idx": 1, "speaker": "user", "text": "capital of france",
             "asr_final": "capital of france"},
            {"idx": 1, "speaker": "assistant", "text": "Paris.",
             "latency": {"first_audio_latency": 1.0, "endpoint_latency": 0.9,
                         "final_to_first_token": 0.08, "first_token_to_audio": 0.02,
                         "barge_in_latency": None}},
        ],
        "capture": None,
    }
    report = write_suite_report([run1], tmp_path)
    assert (tmp_path / "SUITE.json").exists()
    md = (tmp_path / "SUITE.md").read_text()
    assert "Live validation suite" in md
    assert "Latency (pooled across every turn)" in md
    assert "Per-scenario" in md
    loaded = json.loads((tmp_path / "SUITE.json").read_text())
    assert loaded == report


# --- suites (named scenario subsets) -------------------------------------------


def test_suite_names_include_all_and_acoustic():
    names = set(suite_names())
    assert {"all", "acoustic", "latency", "realistic", "barge", "core"} <= names


def test_resolve_suite_all_is_every_scenario():
    assert len(resolve_suite("all")) == len(SCENARIOS)


def test_resolve_suite_acoustic_excludes_barge_scenarios():
    acoustic = {s.name for s in resolve_suite("acoustic")}
    # The inject-only barge scenarios are NOT in the over-the-air suite.
    assert acoustic.isdisjoint(set(BARGE_SCENARIOS))
    assert "baseline_latency_single_turn_qa" in acoustic
    assert "latency_profile_mixed" in acoustic


def test_resolve_suite_barge_is_the_five_barge_scenarios():
    barge = {s.name for s in resolve_suite("barge")}
    assert barge == set(BARGE_SCENARIOS)


def test_resolve_suite_unknown_raises():
    import pytest

    with pytest.raises(KeyError):
        resolve_suite("does_not_exist")


def test_resolve_suite_realistic_present():
    names = {s.name for s in resolve_suite("realistic")}
    assert "realistic_morning_planning" in names
    assert "latency_profile_mixed" not in names  # that's the latency suite


def test_new_scenarios_exist_and_are_graded():
    names = {s.name for s in SCENARIOS}
    for n in ("latency_profile_mixed", "realistic_morning_planning",
              "realistic_cooking_help", "realistic_curiosity_chat",
              "realistic_quickfire_assist"):
        assert n in names, n
        s = by_name(n)
        # Each new scenario has at least one response-graded turn.
        assert any(t.expect or t.forbid for t in s.turns), n


def test_turn_expect_forbid_default_empty():
    t = Turn("hi")
    assert t.expect == () and t.forbid == ()


# --- CLI: --suite / --repeat / --list-suites flags -----------------------------


def _machine_grade(
    *,
    full=True,
    barge="ok",
    intended=2,
    stopped=2,
    responses=None,
    response_count=None,
):
    n_response = (
        int(response_count)
        if response_count is not None
        else (1 if responses is not None else 0)
    )
    return {
        "full_duplex": {"ok": full},
        "barge_in": {
            "verdict": barge,
            "n_intended_barges": intended,
            "n_stopped": stopped,
        },
        "response": {
            "aggregate": {
                "n": n_response,
                "verdict": responses if n_response else None,
            }
        },
    }


def test_injected_machine_grades_control_exit_status():
    green = _machine_grade()
    assert live_main._machine_grade_exit_code(
        0, green, inject=True, expected_barges=2
    ) == 0

    assert live_main._machine_grade_exit_code(
        0, _machine_grade(full=False), inject=True, expected_barges=2
    ) == 1
    assert live_main._machine_grade_exit_code(
        0, _machine_grade(barge="FAIL"), inject=True, expected_barges=2
    ) == 1
    assert live_main._machine_grade_exit_code(
        0,
        _machine_grade(responses="FAIL"),
        inject=True,
        expected_barges=2,
        expected_responses=1,
    ) == 1


def test_repeated_injected_failure_cannot_be_reset_by_a_later_green_run():
    rc = 0
    for grade in (
        _machine_grade(barge="FAIL"),
        _machine_grade(),
        _machine_grade(full=False),
    ):
        rc = live_main._machine_grade_exit_code(
            rc, grade, inject=True, expected_barges=2
        )
    assert rc == 1


def test_acoustic_manual_run_does_not_promote_machine_grade_to_exit_failure():
    failed = _machine_grade(full=False, barge="FAIL", responses="FAIL")
    assert live_main._machine_grade_exit_code(0, failed, inject=False) == 0


def test_injected_machine_grade_missing_required_blocks_fails_closed():
    assert live_main._machine_grade_exit_code(
        0, {}, inject=True, expected_barges=2
    ) == 1
    malformed = _machine_grade()
    malformed["response"] = {}
    assert live_main._machine_grade_exit_code(
        0, malformed, inject=True, expected_barges=2
    ) == 1


def test_injected_machine_grade_requires_exact_barge_coverage():
    for intended, stopped in ((0, 0), (1, 1), (2, 1)):
        grade = _machine_grade(intended=intended, stopped=stopped)
        assert live_main._machine_grade_exit_code(
            0, grade, inject=True, expected_barges=2
        ) == 1
    assert live_main._machine_grade_exit_code(
        0, _machine_grade(), inject=True, expected_barges=2
    ) == 0


def test_injected_zero_barge_control_preserves_exact_zero_coverage():
    control = _machine_grade(intended=0, stopped=0)
    assert live_main._machine_grade_exit_code(
        0, control, inject=True, expected_barges=0
    ) == 0


def test_injected_machine_grade_requires_exact_response_coverage():
    missing = _machine_grade(responses=None, response_count=0)
    assert live_main._machine_grade_exit_code(
        0,
        missing,
        inject=True,
        expected_barges=2,
        expected_responses=1,
    ) == 1
    complete = _machine_grade(responses="ok", response_count=1)
    assert live_main._machine_grade_exit_code(
        0,
        complete,
        inject=True,
        expected_barges=2,
        expected_responses=1,
    ) == 0


def test_scenario_derived_machine_grade_coverage_is_exact():
    assert live_main._expected_machine_grade_counts(
        by_name("barge_in_interrupt_stop")
    ) == (2, 0)
    assert live_main._expected_machine_grade_counts(
        by_name("barge_multiple_in_one_turn")
    ) == (1, 0)
    assert live_main._expected_machine_grade_counts(
        by_name("barge_no_barge_control")
    ) == (0, 0)
    assert live_main._expected_machine_grade_counts(
        by_name("baseline_latency_single_turn_qa")
    ) == (0, 2)


def test_suite_flag_in_help(capsys):
    import pytest

    with pytest.raises(SystemExit) as exc:
        live_main.main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--suite" in help_text
    assert "--repeat" in help_text
    assert "--list-suites" in help_text


def test_list_suites_lists_and_exits(capsys):
    rc = live_main.main(["--list-suites"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "acoustic" in out and "latency" in out and "realistic" in out


def test_unknown_suite_errors(capsys):
    rc = live_main.main(["--suite", "nope", "--llm", "echo"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "unknown suite" in out


# --- --denoise / --noise-snr flags + noise overlay -----------------------------


def test_endpoint_min_silence_flag_overrides(monkeypatch):
    config = _run_main_capturing_config(monkeypatch, ["--endpoint-min-silence", "0.5"])
    assert config["sherpa"]["endpoint_min_silence_sec"] == 0.5


def test_endpoint_max_silence_flag_overrides(monkeypatch):
    config = _run_main_capturing_config(monkeypatch, ["--endpoint-max-silence", "1.2"])
    assert config["sherpa"]["endpoint_max_silence_sec"] == 1.2


def test_no_endpoint_flags_leave_config_untouched(monkeypatch):
    config = _run_main_capturing_config(monkeypatch, [])
    assert "endpoint_min_silence_sec" not in config.get("sherpa", {})
    assert "endpoint_max_silence_sec" not in config.get("sherpa", {})


def test_endpoint_flags_in_help(capsys):
    import pytest

    with pytest.raises(SystemExit) as exc:
        live_main.main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--endpoint-min-silence" in help_text


def test_denoise_flag_enables_denoise(monkeypatch):
    config = _run_main_capturing_config(monkeypatch, ["--denoise"])
    assert config["sherpa"]["denoise_enabled"] is True


def test_inject_denoise_flag_is_an_explicit_stress_override(monkeypatch):
    config = _run_main_capturing_config(
        monkeypatch,
        ["--inject", "--denoise"],
        initial_sherpa={"denoise_enabled": True},
    )
    assert config["sherpa"]["denoise_enabled"] is True


def test_no_denoise_flag_leaves_config_untouched(monkeypatch):
    config = _run_main_capturing_config(monkeypatch, [])
    assert "denoise_enabled" not in config.get("sherpa", {})


def test_denoise_and_noise_snr_in_help(capsys):
    import pytest

    with pytest.raises(SystemExit) as exc:
        live_main.main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--denoise" in help_text
    assert "--noise-snr" in help_text


def test_noise_overlay_corrupts_played_not_saved(monkeypatch):
    # --noise-snr overlays noise on the PLAYED buffer only; the returned/saved
    # clip stays clean (the same contract as --user-volume).
    import numpy as np

    played = {}

    class _FakeSD:
        @staticmethod
        def query_devices(dev, kind):
            return {"default_samplerate": 22050}

        @staticmethod
        def play(buf, rate, device=None):
            played["buf"] = np.asarray(buf, dtype="float32").copy()

        @staticmethod
        def wait():
            pass

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", _FakeSD)

    # Reuse the volume-test TTS/device fakes.
    import core.engines._sherpa_models as sherpa_models
    import core.engines.sherpa as sherpa_engine
    from tools.live_session.synthetic_user import SyntheticUser

    monkeypatch.setattr(sherpa_models, "build_tts", lambda cfg: _FakeTTS())
    monkeypatch.setattr(sherpa_engine, "_norm_device", lambda d: None)

    class _Cfg:
        tts_speaker_id = 0
        tts_speed = 1.0

    noisy = SyntheticUser(_Cfg(), speaker_id=0, speed=1.0, noise_snr_db=6.0)
    saved, sr = noisy.say("hello")
    played_buf = played["buf"]
    # The saved clip is the clean full-scale tone (0.8 from _FakeTTS), unchanged.
    assert abs(float(np.max(np.abs(saved))) - 0.8) < 1e-6
    # The played buffer has added noise -> higher variance / not equal to the clean.
    assert played_buf.shape[0] == saved.shape[0]
    assert float(np.std(played_buf - 0.8)) > 0.0  # noise present around the tone
    # The SNR is finite/positive: noise std is well below the signal level.
    assert float(np.std(played_buf)) < 0.8


def test_noise_overlay_deterministic_same_seed(monkeypatch):
    # Two runs at the same SNR produce the SAME noise (fixed seed) -> a fair A/B.
    import numpy as np

    bufs = []

    class _FakeSD:
        @staticmethod
        def query_devices(dev, kind):
            return {"default_samplerate": 22050}

        @staticmethod
        def play(buf, rate, device=None):
            bufs.append(np.asarray(buf, dtype="float32").copy())

        @staticmethod
        def wait():
            pass

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", _FakeSD)
    import core.engines._sherpa_models as sherpa_models
    import core.engines.sherpa as sherpa_engine
    from tools.live_session.synthetic_user import SyntheticUser

    monkeypatch.setattr(sherpa_models, "build_tts", lambda cfg: _FakeTTS())
    monkeypatch.setattr(sherpa_engine, "_norm_device", lambda d: None)

    class _Cfg:
        tts_speaker_id = 0
        tts_speed = 1.0

    for _ in range(2):
        SyntheticUser(_Cfg(), speaker_id=0, speed=1.0, noise_snr_db=10.0).say("hello")
    assert np.allclose(bufs[0], bufs[1])  # identical noise across runs


# --- capability_latency_profile scenario routes to the intended tiers ----------


def test_capability_latency_scenario_exists_and_shaped():
    s = by_name("capability_latency_profile")
    assert len(s.turns) >= 6
    # In the latency + capability suites.
    assert "capability_latency_profile" in {x.name for x in resolve_suite("capability")}
    assert "capability_latency_profile" in {x.name for x in resolve_suite("latency")}


def test_capability_latency_turns_route_to_intended_tiers():
    # Verify the scenario actually exercises BOTH tiers (the whole point: forcing
    # everything to one tier makes the latency comparison meaningless). The notes
    # mark FAST vs MAIN; check the router agrees at the desktop threshold (0.3).
    from core.routing import HeuristicRouter

    r = HeuristicRouter(threshold=0.3)
    s = by_name("capability_latency_profile")
    fast = [t for t in s.turns if t.note.startswith("FAST")]
    main = [t for t in s.turns if t.note.startswith("MAIN")]
    assert fast and main  # the scenario spans both tiers
    for t in fast:
        assert r.score(t.text, {}) < 0.3, f"expected FAST: {t.text!r}"
    for t in main:
        assert r.score(t.text, {}) >= 0.3, f"expected MAIN: {t.text!r}"
    # The research turn relies on the intent classifier (intent_kind=research),
    # which escalates regardless of the lexical markers.
    research = [t for t in s.turns if t.note.startswith("RESEARCH")]
    for t in research:
        assert r.score(t.text, {"intent_kind": "research"}) >= 0.3, f"research: {t.text!r}"


# --- inject-mode null output stream: MUST drive the PortAudio callback ----------
#
# Regression for a SILENT inject-mode break: the engine's playback path was
# rewritten (2026-06-02) to have PortAudio PULL audio from ``callback=_audio_cb``
# (where TTS_FIRST_AUDIO is stamped) instead of pushing via ``out.write()``. The
# inject-mode ``_NullOutputStream`` still only implemented ``write()`` and
# swallowed the ``callback=`` kwarg, so the callback NEVER fired -> the FIFO never
# drained -> TTS_FIRST_AUDIO never stamped -> every inject-mode latency field came
# back null and the run logged a 'tts stuck' watchdog stall. The fix has the fake
# stream emulate PortAudio by calling the callback from a real-time-paced thread.


def test_null_output_stream_drives_the_callback():
    """The inject sink must CALL the callback (not just accept it) so the engine's
    callback-driven playback drains the FIFO and stamps TTS_FIRST_AUDIO."""
    import threading

    from tools.live_session.driver import _NullOutputStream

    calls = {"n": 0, "shape": None, "dtype": None}
    first = threading.Event()

    def cb(outdata, frames, time_info, status):
        calls["n"] += 1
        calls["shape"] = tuple(outdata.shape)
        calls["dtype"] = str(outdata.dtype)
        first.set()

    # Construct it exactly as the engine does: sd.OutputStream(channels=1,
    # samplerate=..., dtype=..., device=..., latency="low", callback=...).
    out = _NullOutputStream(
        channels=1, samplerate=22050, dtype="float32",
        device=None, latency="low", callback=cb,
    )
    try:
        out.start()
        assert first.wait(timeout=2.0), "callback was never driven (the inject-latency bug)"
        n_at_stop = calls["n"]
        assert n_at_stop >= 1
        # The callback is handed a writable (frames, channels) float32 buffer,
        # matching what _audio_cb does: ``view = outdata[:, 0]``.
        assert calls["shape"] is not None and calls["shape"][1] == 1
        assert calls["dtype"] == "float32"
    finally:
        out.stop()
    # After stop() the pump thread halts: no further callbacks.
    time.sleep(0.1)
    settled = calls["n"]
    time.sleep(0.2)
    assert calls["n"] == settled, "callback kept firing after stop()"


def test_null_output_stream_without_callback_is_a_pure_sink():
    """No callback (legacy/None) -> behaves as a paced sink, never errors."""
    from tools.live_session.driver import _NullOutputStream

    out = _NullOutputStream(samplerate=16000, callback=None)
    out.start()
    assert out.active is True
    out.write([0.0] * 16)  # legacy blocking path is a real-time-paced no-op
    out.stop()
    assert out.active is False
