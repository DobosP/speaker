"""Tests for the live-validation harness's NON-AUDIO logic (no models/hardware).

The live run itself (tools/live_session) needs real ASR/TTS/LLM + audio and is
run only on request; these pin the pure pieces -- scenario data, timing parsing,
answer filtering, and report generation -- so they don't rot.
"""
from __future__ import annotations

import json
from pathlib import Path

import tools.live_session.__main__ as live_main
from tools.live_session.driver import _is_answer, _parse_pause
from tools.live_session.report import write_latency_report, write_summary, write_timeline
from tools.live_session.scenarios import SCENARIOS, Scenario, Turn, by_name


def test_scenarios_are_well_formed():
    assert len(SCENARIOS) >= 5
    names = {s.name for s in SCENARIOS}
    assert len(names) == len(SCENARIOS)  # unique
    for s in SCENARIOS:
        assert s.turns and all(t.text.strip() for t in s.turns)
        for t in s.turns:
            assert t.timing in ("wait_for_response", "immediately", "barge_in") or t.timing.startswith("pause:")


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
    c._out = tmp_path
    return c


def test_flush_assistant_joins_streamed_sentences_into_one_event(tmp_path):
    from core.metrics import TTS_FIRST_AUDIO

    eng = _FakeEngine()
    eng.speak("Paris is the capital.", 2.0)
    eng.speak("It has about 2 million people.", 2.4)
    sup = _FakeSupervisor()
    metrics = _FakeMetrics([_FakeRec({TTS_FIRST_AUDIO: 1.9})])
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


def test_consume_latency_pairs_turns_in_order(tmp_path):
    from core.metrics import TTS_FIRST_AUDIO

    recs = [_FakeRec({TTS_FIRST_AUDIO: 1.0}), _FakeRec({}), _FakeRec({TTS_FIRST_AUDIO: 5.0})]
    c = _bare_convo(_FakeEngine(), _FakeRuntime(_FakeEngine(), _FakeSupervisor(), _FakeMetrics(recs)), tmp_path)
    assert c._consume_latency() is not None  # first audio turn
    assert c._consume_latency() is not None  # third (skips the empty middle)
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


# --- CLI flags: --smart-endpoint (experimental semantic endpoint A/B) ----------
#
# These pin the in-memory config mutation only (no audio/models). main() short-
# circuits on --check after applying the device profile + the flag mutations and
# before constructing LiveConversation, so we capture the config in a stubbed
# _preflight. The committed config.json default must stay OFF; the flag flips it
# ON only for the live run, and its absence is byte-identical to before.


def _run_main_capturing_config(monkeypatch, argv):
    """Drive main(argv) with a minimal stub config + a captured _preflight.

    Returns the config dict as it was when _preflight saw it (i.e. AFTER the
    device-profile merge and the CLI flag mutations, BEFORE LiveConversation).
    """
    import core.config as core_config

    captured: dict = {}

    monkeypatch.setattr(core_config, "_load_config",
                        lambda *a, **k: {"device": "desktop", "sherpa": {}})
    monkeypatch.setattr(core_config, "_apply_device_profile",
                        lambda config, device: config)

    def _stub_preflight(config):
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


def test_committed_config_default_endpoint_is_off():
    # The global committed default MUST stay False -- the flag is an in-memory,
    # per-run override and never writes config.json.
    config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
    assert config["sherpa"]["endpoint_enabled"] is False


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
