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
from tools.live_session.report import (
    BARGE_STOP_RATE_MIN,
    HEARD_OK_THRESHOLD,
    grade_barge_turns,
    stt_score,
    summarize_barge,
    summarize_capture,
    write_grade,
    write_latency_report,
    write_summary,
    write_timeline,
)
from tools.live_session.scenarios import LONG_ANSWER, SCENARIOS, Scenario, Turn, by_name


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
    # Every barge scenario uses the shared LONG_ANSWER as its barge target so
    # there is always speech to cut (the short-response-race fix).
    for name in expected:
        s = by_name(name)
        assert any(t.text == LONG_ANSWER for t in s.turns), name


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


def test_flush_assistant_stamps_barge_intended_from_upcoming_user_timing(tmp_path):
    # A barge_in line interrupts the PRECEDING answer, so the answer is flagged
    # barge_intended by the UPCOMING line's timing (the barge that hits it), not
    # the line before it. The grader uses this to tell a real interrupt from a
    # self-interrupt.
    eng = _FakeEngine()
    eng.speak("Once upon a time...", 1.0)
    eng.stop(1.2)
    c = _bare_convo(eng, _FakeRuntime(eng, _FakeSupervisor(), _FakeMetrics([])), tmp_path)
    c._upcoming_user_timing = "barge_in"
    c._flush_assistant()
    assert c.events[0]["barge_intended"] is True

    eng2 = _FakeEngine()
    eng2.speak("Paris.", 2.0)
    c2 = _bare_convo(eng2, _FakeRuntime(eng2, _FakeSupervisor(), _FakeMetrics([])), tmp_path)
    c2._upcoming_user_timing = "wait_for_response"
    c2._flush_assistant()
    assert c2.events[0]["barge_intended"] is False


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


def test_committed_config_endpoint_enabled_with_validated_min_silence():
    # Smart endpoint was validated on-device (docs/live_validation_run_2026-05-30.md:
    # ~300ms first-audio win, no tail clipping, no sentence splitting) and ENABLED.
    # min_silence MUST stay >= 0.7s: it has to exceed BOTH the decoder lookahead and a
    # typical intra-sentence comma pause -- at 0.5 a run-on ("Hey, what are you, and...")
    # split at the comma. Lowering it risks regressing that, so this pins the floor.
    config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
    assert config["sherpa"]["endpoint_enabled"] is True
    assert config["sherpa"]["endpoint_min_silence_sec"] >= 0.7


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


# --- barge-in / interrupt grade (pure, inject-mode-only logic) ------------------
#
# Fakes only -- the same shape the driver writes onto each assistant event:
#   * barge_intended (the PRECEDING user line's timing == "barge_in"),
#   * interrupted (engine.stopped_after -- the primary "stopped" signal),
#   * latency.barge_in_latency (BARGE_IN_STOP minus BARGE_IN, in SECONDS; None when
#     stop_speaking found no live stream to abort -- the short-answer race).


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
    assert rows[0]["stop_ms"] is None  # the short-answer race: no STOP stamp


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
    # The per-turn table header is present.
    assert "barge_intended" in text and "stop_ms" in text


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
