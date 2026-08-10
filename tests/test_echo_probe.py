from __future__ import annotations

import ast
import hashlib
import json
import re
import struct
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from core.engine import (
    PlaybackCapabilities,
    PlaybackOutcome,
    PlaybackReceipt,
    TrackedSpeech,
)
from core.engines._aec import AecDelaySnapshot
from tools import echo_probe as echo_probe_module
from tools.echo_probe import (
    _PLAYBACK_TERMINAL_PROTOCOL,
    _PlaybackTerminalProtocolError,
    _ProbeLifecycleResult,
    _StimulusIdBuilder,
    _StimulusPlan,
    _TerminalReceiptTracker,
    _TrackedStimulusResult,
    _aec_reference_delay_summary,
    _build_stimulus_plan,
    _emit_probe_result,
    _make_scaled_synthesizer,
    _require_tracked_terminal,
    _run_probe_lifecycle,
    _run_tracked_stimulus,
)
from tools.interrupt_suite import summarize


_STIMULUS_ID_PREFIX = "echo-probe-spoken-text-v1:sha256:"
_STIMULUS_DOMAIN = b"speaker.echo-probe.spoken-text.v1\0"
_EXPECTED_SENTENCE_BYTES = (
    b"This is a live audio calibration test of the speaker assistant.",
    b"I am checking whether my own voice is captured back by the microphone.",
    b"The barge in gate should not interrupt me while I am still speaking.",
    b"This final sentence completes the quiet playback portion of the diagnostic.",
)
_EXPECTED_STIMULUS_DIGESTS = {
    1: "f6d8f6a9701c09d8d87307e9884bb2ad7ddd8aec16b9353732ba03cf9b75f3fd",
    3: "27c744fbb14bbcb74f9256a2a4db2af25c6e7ad6c70e5ea224c3c8708a06a3c6",
    4: "4366d6a869714ed767964bda1dc682b5264c109d5d3f12260f6e07276c502704",
    5: "b17929f5d51bb8ece730dcfb2094ec85ab195e6a451c79a63fdd955bd59875fb",
}


def _independent_stimulus_id(texts) -> str:
    digest = hashlib.new("sha256", _STIMULUS_DOMAIN)
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    return _STIMULUS_ID_PREFIX + digest.hexdigest()


def _built_stimulus_id(plan: _StimulusPlan) -> str:
    builder = _StimulusIdBuilder()
    for text in plan:
        builder.record(text)
    return builder.stimulus_id()


def _terminal_action(
    outcome: object = PlaybackOutcome.COMPLETED,
    *,
    fragment_id: str | None = None,
    duplicate: bool = False,
):
    def action(speech, on_terminal):
        receipt = PlaybackReceipt(
            fragment_id=speech.fragment_id if fragment_id is None else fragment_id,
            outcome=outcome,  # type: ignore[arg-type]
        )
        on_terminal(receipt)
        if duplicate:
            on_terminal(receipt)

    return action


class _ScriptedTrackedEngine:
    def __init__(
        self,
        actions=(),
        *,
        capabilities: object = PlaybackCapabilities(tracked_terminal=True),
    ) -> None:
        self.playback_capabilities = capabilities
        self.actions = list(actions)
        self.submissions: list[TrackedSpeech] = []
        self.callbacks = []
        self.legacy_calls = 0

    def speak(self, *_args, **_kwargs):
        self.legacy_calls += 1
        raise AssertionError("legacy speak must never be called")

    def speak_tracked(self, speech, *, on_terminal, on_started=None):
        assert type(speech) is TrackedSpeech
        assert on_started is None
        self.submissions.append(speech)
        self.callbacks.append(on_terminal)
        if not self.actions:
            raise AssertionError("unexpected tracked submission")
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        if action is not None:
            action(speech, on_terminal)


def _run_scripted_stimulus(
    engine,
    *,
    count: int = 1,
    timeout_seconds: float = 0.2,
    on_wait=lambda: None,
):
    builder = _StimulusIdBuilder()
    result = _run_tracked_stimulus(
        engine,
        _build_stimulus_plan(count),
        builder,
        timeout_seconds=timeout_seconds,
        on_wait=on_wait,
    )
    return result, builder


class _SnapshotCalibrator:
    def __init__(self, snapshot: AecDelaySnapshot) -> None:
        self.value = snapshot
        self.calls = 0

    def snapshot(self) -> AecDelaySnapshot:
        self.calls += 1
        return self.value


class _ForbiddenSnapshot:
    def snapshot(self):
        raise AssertionError("capture-owned snapshot read before quiescence")


def _event(*, set_: bool = False) -> threading.Event:
    event = threading.Event()
    if set_:
        event.set()
    return event


def _config(
    *,
    aec_enabled: bool = True,
    auto_delay: bool = True,
    sample_rate: int = 16000,
    seed_ms: int = 80,
):
    return SimpleNamespace(
        aec_enabled=aec_enabled,
        aec_auto_delay=auto_delay,
        sample_rate=sample_rate,
        aec_ref_delay_ms=seed_ms,
    )


def _engine(*, aec_active: bool, calibrator=None, operating: int = 1280):
    return SimpleNamespace(
        _aec=object() if aec_active else None,
        _aec_delay_cal=calibrator,
        _aec_ref_delay=operating,
        _running=_event(),
        _capture_resource_hold=_event(),
    )


class _MainTrackedEngine(_ScriptedTrackedEngine):
    def __init__(
        self,
        actions=(),
        *,
        capabilities: object = PlaybackCapabilities(tracked_terminal=True),
    ) -> None:
        super().__init__(actions, capabilities=capabilities)
        self.events: list[str] = []
        self.stop_calls = 0
        self._aec = None
        self._aec_delay_cal = None
        self._aec_ref_delay = 1280
        self._running = _event()
        self._capture_resource_hold = _event()
        self._echo_coherence = None
        self._dtd = None
        self._playback_level = 0.9
        self._synthesize = lambda *_args, **_kwargs: None
        self.looks_like_user_calls = []
        self.capture_samples = [0.25, -0.25]
        self.raw_samples = [0.5, -0.5]
        self.capture_playback_level = 0.125

    def _looks_like_user(self, samples, mic_raw=None, *, playback_level=None):
        self.looks_like_user_calls.append((samples, mic_raw, playback_level))
        return True

    def start(self, _callbacks):
        self.events.append("start")
        self._looks_like_user(
            self.capture_samples,
            self.raw_samples,
            playback_level=self.capture_playback_level,
        )

    def stop(self):
        self.stop_calls += 1
        self.events.append("stop")

    def speak_tracked(self, speech, *, on_terminal, on_started=None):
        self.events.append(f"submit-{len(self.submissions) + 1}")
        super().speak_tracked(
            speech,
            on_terminal=on_terminal,
            on_started=on_started,
        )


class _TrackedTerminalAccessorFault:
    def __init__(self, fault: BaseException) -> None:
        self.fault = fault

    @property
    def tracked_terminal(self):
        raise self.fault


class _ControlledTerminalEvent:
    def __init__(self) -> None:
        self.wait_entered = threading.Event()
        self.resume_waiter = threading.Event()
        self._is_set = False

    def set(self) -> None:
        self._is_set = True

    def wait(self, _timeout=None) -> bool:
        self.wait_entered.set()
        if not self.resume_waiter.wait(1.0):
            raise AssertionError("terminal waiter did not resume")
        return self._is_set


class _ControlledClock:
    def __init__(self, value: float) -> None:
        self._lock = threading.Lock()
        self._value = value

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def monotonic(self) -> float:
        with self._lock:
            return self._value


def _install_fake_main(monkeypatch, engine, *, argv=()):
    sherpa_cfg = SimpleNamespace(
        tts_model="fake-model",
        sample_rate=16000,
        aec_ref_delay_ms=80,
        aec_enabled=False,
        aec_auto_delay=False,
        input_device="fake-input",
    )

    config_module = ModuleType("core.config")
    config_module.load_config = lambda _path: {"device": "fake-device", "sherpa": {}}
    config_module.apply_device_profile = lambda cfg, _device: cfg

    class FakeSherpaConfig:
        @classmethod
        def from_dict(cls, _config):
            return sherpa_cfg

    sherpa_module = ModuleType("core.engines.sherpa")
    sherpa_module.SherpaConfig = FakeSherpaConfig
    sherpa_module.SherpaOnnxEngine = lambda _config: engine
    monkeypatch.setitem(sys.modules, "core.config", config_module)
    monkeypatch.setitem(sys.modules, "core.engines.sherpa", sherpa_module)
    monkeypatch.setattr(echo_probe_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(sys, "argv", ["echo_probe", *argv])
    return sherpa_cfg


def test_stimulus_catalog_bytes_are_exactly_locked():
    assert tuple(text.encode("utf-8") for text in echo_probe_module.SENTENCES) == (
        _EXPECTED_SENTENCE_BYTES
    )


@pytest.mark.parametrize(
    ("requested_count", "effective_count"),
    [(-7, 1), (0, 1), (1, 1), (3, 3), (4, 4), (5, 5)],
)
def test_stimulus_plan_and_id_match_independent_vectors(
    requested_count, effective_count
):
    plan = _build_stimulus_plan(requested_count)
    expected_texts = tuple(
        _EXPECTED_SENTENCE_BYTES[index % len(_EXPECTED_SENTENCE_BYTES)].decode("utf-8")
        for index in range(effective_count)
    )

    assert type(plan) is _StimulusPlan
    assert type(plan.cycle) is tuple
    assert plan.count == effective_count
    assert tuple(plan) == expected_texts
    assert _built_stimulus_id(plan) == (
        _STIMULUS_ID_PREFIX + _EXPECTED_STIMULUS_DIGESTS[effective_count]
    )
    assert _built_stimulus_id(plan) == _independent_stimulus_id(expected_texts)


def test_stimulus_plan_is_lazy_frozen_and_preserves_text_object_identity(monkeypatch):
    first = "".join(("first", " sentence"))
    second = "".join(("second", " sentence"))
    monkeypatch.setattr(echo_probe_module, "SENTENCES", [first, second])
    plan = _build_stimulus_plan(5)
    iterator = iter(plan)

    monkeypatch.setattr(echo_probe_module, "SENTENCES", ["later mutation"])
    yielded = tuple(iterator)

    assert yielded == (first, second, first, second, first)
    assert yielded[0] is first
    assert yielded[1] is second
    assert yielded[2] is first


def test_stimulus_id_binds_order_repetition_and_utf8_byte_length():
    first = _EXPECTED_SENTENCE_BYTES[0].decode("utf-8")
    second = _EXPECTED_SENTENCE_BYTES[1].decode("utf-8")
    forward = _independent_stimulus_id((first, second))

    assert forward != _independent_stimulus_id((second, first))
    assert forward != _independent_stimulus_id((first, second, first))

    multibyte = "caf\N{LATIN SMALL LETTER E WITH ACUTE}"
    encoded = multibyte.encode("utf-8")
    builder = _StimulusIdBuilder()
    builder.record(multibyte)
    expected = hashlib.sha256(
        _STIMULUS_DOMAIN + struct.pack(">Q", len(encoded)) + encoded
    ).hexdigest()
    character_length = hashlib.sha256(
        _STIMULUS_DOMAIN + struct.pack(">Q", len(multibyte)) + encoded
    ).hexdigest()

    assert builder.stimulus_id() == _STIMULUS_ID_PREFIX + expected
    assert expected != character_length


def test_neutral_fourth_sentence_changes_n4_without_changing_n3():
    current = tuple(value.decode("utf-8") for value in _EXPECTED_SENTENCE_BYTES)
    legacy = (
        *current[:3],
        "If you hear this whole message without it cutting off, suppression works.",
    )
    current_n3 = _independent_stimulus_id(current[:3])
    legacy_n3 = _independent_stimulus_id(legacy[:3])
    current_n4 = _independent_stimulus_id(current)
    legacy_n4 = _independent_stimulus_id(legacy)

    assert current_n3 == (
        _STIMULUS_ID_PREFIX
        + "27c744fbb14bbcb74f9256a2a4db2af25c6e7ad6c70e5ea224c3c8708a06a3c6"
    )
    assert legacy_n3 == current_n3
    assert legacy_n4 == (
        _STIMULUS_ID_PREFIX
        + "be277009edac56b0473730c1e5326a1b765976386aca9f26c500eab9f00822eb"
    )
    assert current_n4 == (
        _STIMULUS_ID_PREFIX
        + "4366d6a869714ed767964bda1dc682b5264c109d5d3f12260f6e07276c502704"
    )
    assert current_n4 != legacy_n4


def test_tracked_stimulus_accepts_synchronous_terminals_and_counts_outcomes():
    engine = _ScriptedTrackedEngine(
        [
            _terminal_action(PlaybackOutcome.COMPLETED),
            _terminal_action(PlaybackOutcome.INTERRUPTED),
            _terminal_action(PlaybackOutcome.COMPLETED),
        ]
    )

    result, builder = _run_scripted_stimulus(engine, count=3)

    assert type(result) is _TrackedStimulusResult
    assert (result.submitted, result.completed, result.interrupted) == (3, 2, 1)
    result.validate()
    assert [speech.text for speech in engine.submissions] == list(
        _build_stimulus_plan(3)
    )
    assert builder.stimulus_id() == (
        _STIMULUS_ID_PREFIX + _EXPECTED_STIMULUS_DIGESTS[3]
    )
    assert engine.actions == []
    assert engine.legacy_calls == 0


def test_next_sentence_and_stop_wait_for_each_exact_terminal_receipt():
    events: list[str] = []
    submitted = [threading.Event(), threading.Event()]
    callbacks = []

    class BlockingEngine(_ScriptedTrackedEngine):
        def __init__(self):
            super().__init__(())

        def speak_tracked(self, speech, *, on_terminal, on_started=None):
            assert type(speech) is TrackedSpeech
            assert on_started is None
            index = len(self.submissions)
            self.submissions.append(speech)
            callbacks.append(on_terminal)
            events.append(f"submit-{index + 1}")
            submitted[index].set()

    engine = BlockingEngine()
    thread_result: list[_ProbeLifecycleResult] = []
    thread_error: list[BaseException] = []

    def run_lifecycle():
        try:
            thread_result.append(
                _run_probe_lifecycle(
                    lambda: events.append("start"),
                    lambda: _run_scripted_stimulus(
                        engine,
                        count=2,
                        timeout_seconds=2.0,
                    ),
                    lambda: events.append("stop"),
                )
            )
        except BaseException as exc:
            thread_error.append(exc)

    worker = threading.Thread(target=run_lifecycle)
    worker.start()
    assert submitted[0].wait(1.0)
    assert not submitted[1].is_set()
    assert "stop" not in events

    first = engine.submissions[0]
    events.append("terminal-1")
    callbacks[0](
        PlaybackReceipt(
            fragment_id=first.fragment_id,
            outcome=PlaybackOutcome.COMPLETED,
        )
    )
    assert submitted[1].wait(1.0)
    assert "stop" not in events

    second = engine.submissions[1]
    events.append("terminal-2")
    callbacks[1](
        PlaybackReceipt(
            fragment_id=second.fragment_id,
            outcome=PlaybackOutcome.INTERRUPTED,
        )
    )
    worker.join(1.0)

    assert not worker.is_alive()
    assert thread_error == []
    assert len(thread_result) == 1
    assert thread_result[0].report_allowed is True
    assert events == [
        "start",
        "submit-1",
        "terminal-1",
        "submit-2",
        "terminal-2",
        "stop",
    ]
    assert engine.legacy_calls == 0


@pytest.mark.parametrize(
    ("arrival_time", "expected_error_type"),
    [
        (100.75, None),
        (101.25, TimeoutError),
    ],
)
def test_terminal_deadline_uses_callback_arrival_not_delayed_waiter_observation(
    arrival_time,
    expected_error_type,
    monkeypatch,
):
    clock = _ControlledClock(100.0)
    monkeypatch.setattr(
        echo_probe_module,
        "time",
        SimpleNamespace(monotonic=clock.monotonic),
    )
    tracker = _TerminalReceiptTracker()
    slot = tracker.new_slot("opaque-fragment")
    controlled_event = _ControlledTerminalEvent()
    slot.event = controlled_event  # type: ignore[assignment]
    receipt = PlaybackReceipt(
        fragment_id=slot.fragment_id,
        outcome=PlaybackOutcome.COMPLETED,
    )
    callback = tracker.callback_for(slot)
    returned = []
    errors: list[BaseException] = []

    def wait_for_terminal():
        try:
            returned.append(
                tracker.wait_for_terminal(
                    slot,
                    timeout_seconds=1.0,
                    on_wait=lambda: None,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    waiter = threading.Thread(target=wait_for_terminal)
    waiter.start()
    assert controlled_event.wait_entered.wait(1.0)
    clock.set(arrival_time)
    callback(receipt)
    clock.set(102.0)
    controlled_event.resume_waiter.set()
    waiter.join(1.0)

    assert not waiter.is_alive()
    assert slot.received_at == arrival_time
    if expected_error_type is None:
        assert returned == [receipt]
        assert errors == []
        tracker.validate()
    else:
        assert returned == []
        assert len(errors) == 1
        assert type(errors[0]) is expected_error_type
        with pytest.raises(_PlaybackTerminalProtocolError):
            tracker.validate()


def test_fragment_ids_are_unique_ordinal_only_and_privacy_safe():
    engine = _ScriptedTrackedEngine([_terminal_action() for _ in range(5)])

    _run_scripted_stimulus(engine, count=5)

    fragment_ids = [speech.fragment_id for speech in engine.submissions]
    stimulus_terms = {
        token.strip(".").lower()
        for sentence in echo_probe_module.SENTENCES
        for token in sentence.split()
        if len(token.strip(".")) >= 4
    }
    assert len(fragment_ids) == len(set(fragment_ids)) == 5
    assert all(type(fragment_id) is str and fragment_id for fragment_id in fragment_ids)
    assert all(
        re.fullmatch(r"[a-z0-9._:-]+", fragment_id) for fragment_id in fragment_ids
    )
    assert all(
        term not in fragment_id.lower()
        for fragment_id in fragment_ids
        for term in stimulus_terms
    )


@pytest.mark.parametrize("tracked_terminal", [False, None, 0, "true"])
def test_tracked_terminal_capability_is_exact_and_required(tracked_terminal):
    engine = _ScriptedTrackedEngine(
        [],
        capabilities=SimpleNamespace(tracked_terminal=tracked_terminal),
    )

    with pytest.raises(_PlaybackTerminalProtocolError):
        _require_tracked_terminal(engine)
    with pytest.raises(_PlaybackTerminalProtocolError):
        _run_scripted_stimulus(engine)

    assert engine.submissions == []
    assert engine.legacy_calls == 0


def _faulting_tracked_engine(fault: str) -> _ScriptedTrackedEngine:
    if fault == "missing-capability":
        return _ScriptedTrackedEngine(
            [], capabilities=PlaybackCapabilities(tracked_terminal=False)
        )
    if fault == "dropped":
        action = _terminal_action(PlaybackOutcome.DROPPED)
    elif fault == "failed":
        action = _terminal_action(PlaybackOutcome.FAILED)
    elif fault == "wrong-outcome-type":
        action = _terminal_action("completed")
    elif fault == "fragment-mismatch":
        action = _terminal_action(fragment_id="different-fragment")
    elif fault == "wrong-receipt-type":

        def action(_speech, callback):
            callback(object())

    elif fault == "duplicate":
        action = _terminal_action(duplicate=True)
    elif fault == "timeout":
        action = None
    elif fault == "submission-error":
        action = RuntimeError("submission failed")
    else:
        raise AssertionError(f"unknown fault fixture: {fault}")
    return _ScriptedTrackedEngine([action])


@pytest.mark.parametrize(
    ("fault", "expected_error_type"),
    [
        ("missing-capability", "_PlaybackTerminalProtocolError"),
        ("dropped", "_PlaybackTerminalProtocolError"),
        ("failed", "_PlaybackTerminalProtocolError"),
        ("wrong-outcome-type", "_PlaybackTerminalProtocolError"),
        ("fragment-mismatch", "_PlaybackTerminalProtocolError"),
        ("wrong-receipt-type", "_PlaybackTerminalProtocolError"),
        ("duplicate", "_PlaybackTerminalProtocolError"),
        ("timeout", "TimeoutError"),
        ("submission-error", "RuntimeError"),
    ],
)
def test_terminal_anomalies_fail_closed_and_lifecycle_stops_exactly_once(
    fault,
    expected_error_type,
):
    engine = _faulting_tracked_engine(fault)
    calls: list[str] = []

    lifecycle = _run_probe_lifecycle(
        lambda: calls.append("start"),
        lambda: _run_scripted_stimulus(
            engine,
            timeout_seconds=0.001,
        ),
        lambda: calls.append("stop"),
    )

    assert calls == ["start", "stop"]
    assert lifecycle.probe_error_type == expected_error_type
    assert lifecycle.stop_error_type is None
    assert lifecycle.report_allowed is False
    assert engine.legacy_calls == 0
    payload = _lifecycle_error_payload(lifecycle)
    assert set(payload).isdisjoint(
        {
            "stimulus_id",
            "playback_terminal_protocol",
            "sentences_submitted",
            "sentences_completed",
            "sentences_interrupted",
            "sentences_spoken",
        }
    )


def test_delayed_duplicate_during_stop_fails_closed_after_one_stop():
    engine = _ScriptedTrackedEngine([_terminal_action()])
    tracked: list[_TrackedStimulusResult] = []
    stop_calls = 0

    def probe():
        result, _builder = _run_scripted_stimulus(engine)
        tracked.append(result)

    def stop():
        nonlocal stop_calls
        stop_calls += 1
        speech = engine.submissions[0]
        engine.callbacks[0](
            PlaybackReceipt(
                fragment_id=speech.fragment_id,
                outcome=PlaybackOutcome.COMPLETED,
            )
        )
        tracked[0].validate()

    lifecycle = _run_probe_lifecycle(lambda: None, probe, stop)

    assert stop_calls == 1
    assert lifecycle.probe_error_type is None
    assert lifecycle.stop_error_type == "_PlaybackTerminalProtocolError"
    assert lifecycle.report_allowed is False
    assert engine.legacy_calls == 0


def test_gain_wrapper_forwards_generation_and_directives_and_scales_only_audio():
    import numpy as np

    text = "exact text"
    generation = object()
    directives = object()
    samples = np.asarray([0.25, -0.5, 1.0], dtype="float32")
    original_samples = samples.copy()
    seen = {}
    written = []
    sentinel = object()

    def synthesize(received_text, write, *, gen=None, directives=None):
        seen.update(
            text=received_text,
            gen=gen,
            directives=directives,
        )
        write(samples)
        return sentinel

    scaled = _make_scaled_synthesizer(synthesize, 0.4)
    returned = scaled(
        text,
        written.append,
        gen=generation,
        directives=directives,
    )

    assert returned is sentinel
    assert seen["text"] is text
    assert seen["gen"] is generation
    assert seen["directives"] is directives
    assert len(written) == 1
    np.testing.assert_array_equal(samples, original_samples)
    np.testing.assert_allclose(written[0], original_samples * 0.4, rtol=0, atol=0)


def test_main_success_reports_exact_terminal_protocol_counts_and_wrapper_level(
    monkeypatch,
    capsys,
):
    engine = _MainTrackedEngine(
        [
            _terminal_action(PlaybackOutcome.COMPLETED),
            _terminal_action(PlaybackOutcome.INTERRUPTED),
        ]
    )
    _install_fake_main(
        monkeypatch,
        engine,
        argv=("--sentences", "2", "--label", "fake-probe"),
    )

    rc = echo_probe_module.main()

    assert rc == 0
    report = _decode_exactly_one_json_object(capsys.readouterr().out)
    assert report["playback_terminal_protocol"] == _PLAYBACK_TERMINAL_PROTOCOL
    assert report["playback_terminal_protocol"] == "tracked-sink-terminal-v1"
    assert {
        key: report[key]
        for key in (
            "sentences_submitted",
            "sentences_completed",
            "sentences_interrupted",
            "sentences_spoken",
        )
    } == {
        "sentences_submitted": 2,
        "sentences_completed": 1,
        "sentences_interrupted": 1,
        "sentences_spoken": 2,
    }
    assert report["stimulus_id"] == _independent_stimulus_id(
        tuple(_build_stimulus_plan(2))
    )
    assert report["vad_flagged_during_play"] == 1
    assert report["gate_passed_count"] == 1
    assert report["median_mic_over_playback_dB"] == 6.0
    assert engine.events == ["start", "submit-1", "submit-2", "stop"]
    assert engine.stop_calls == 1
    assert engine.legacy_calls == 0
    assert len(engine.looks_like_user_calls) == 1
    samples, mic_raw, playback_level = engine.looks_like_user_calls[0]
    assert samples is engine.capture_samples
    assert mic_raw is engine.raw_samples
    assert playback_level is engine.capture_playback_level


def test_main_missing_terminal_capability_fails_preflight_without_start_or_stop(
    monkeypatch,
    capsys,
):
    engine = _MainTrackedEngine(
        [],
        capabilities=PlaybackCapabilities(tracked_terminal=False),
    )
    _install_fake_main(monkeypatch, engine)

    rc = echo_probe_module.main()

    assert rc == 1
    assert _decode_exactly_one_json_object(capsys.readouterr().out) == {
        "error": "probe-preflight-failed",
        "preflight_error_type": "_PlaybackTerminalProtocolError",
    }
    assert engine.events == []
    assert engine.stop_calls == 0
    assert engine.submissions == []
    assert engine.legacy_calls == 0


@pytest.mark.parametrize(
    ("fault_kind", "expected_error_type"),
    [
        ("malformed", "_PlaybackTerminalProtocolError"),
        ("runtime-error", "RuntimeError"),
        ("system-exit-zero", "SystemExit"),
    ],
)
def test_main_malformed_or_faulting_capability_fails_identity_free_preflight(
    fault_kind,
    expected_error_type,
    monkeypatch,
    capsys,
):
    if fault_kind == "malformed":
        capabilities = object()
    elif fault_kind == "runtime-error":
        capabilities = _TrackedTerminalAccessorFault(RuntimeError("capability"))
    else:
        capabilities = _TrackedTerminalAccessorFault(SystemExit(0))
    engine = _MainTrackedEngine([], capabilities=capabilities)
    _install_fake_main(monkeypatch, engine)

    rc = echo_probe_module.main()

    assert rc == 1
    assert _decode_exactly_one_json_object(capsys.readouterr().out) == {
        "error": "probe-preflight-failed",
        "preflight_error_type": expected_error_type,
    }
    assert engine.events == []
    assert engine.stop_calls == 0
    assert engine.submissions == []
    assert engine.legacy_calls == 0


def test_main_capability_keyboard_interrupt_preserves_identity_without_output(
    monkeypatch,
    capsys,
):
    interrupt = KeyboardInterrupt("capability")
    engine = _MainTrackedEngine(
        [],
        capabilities=_TrackedTerminalAccessorFault(interrupt),
    )
    _install_fake_main(monkeypatch, engine)

    with pytest.raises(KeyboardInterrupt) as raised:
        echo_probe_module.main()

    assert raised.value is interrupt
    assert capsys.readouterr().out == ""
    assert engine.events == []
    assert engine.stop_calls == 0
    assert engine.submissions == []
    assert engine.legacy_calls == 0


@pytest.mark.parametrize(
    ("action", "expected_error_type"),
    [
        (_terminal_action(PlaybackOutcome.DROPPED), "_PlaybackTerminalProtocolError"),
        (RuntimeError("submission"), "RuntimeError"),
    ],
)
def test_main_terminal_or_submission_failure_stops_once_and_is_identity_free(
    action,
    expected_error_type,
    monkeypatch,
    capsys,
):
    engine = _MainTrackedEngine([action])
    _install_fake_main(monkeypatch, engine, argv=("--sentences", "1"))

    rc = echo_probe_module.main()

    assert rc == 1
    report = _decode_exactly_one_json_object(capsys.readouterr().out)
    assert report == {
        "error": "probe-lifecycle-failed",
        "start_error_type": None,
        "probe_error_type": expected_error_type,
        "stop_error_type": None,
    }
    assert set(report).isdisjoint(
        {
            "stimulus_id",
            "playback_terminal_protocol",
            "sentences_submitted",
            "sentences_completed",
            "sentences_interrupted",
            "sentences_spoken",
        }
    )
    assert engine.events == ["start", "submit-1", "stop"]
    assert engine.stop_calls == 1
    assert engine.legacy_calls == 0


def test_main_has_one_success_only_stimulus_scalar():
    tree = ast.parse(Path(echo_probe_module.__file__).read_text(encoding="utf-8"))
    main_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    stimulus_keys = [
        key
        for node in ast.walk(main_node)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and key.value == "stimulus_id"
    ]
    normal_out = next(
        node.value
        for node in ast.walk(main_node)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "out"
            for target in node.targets
        )
        and isinstance(node.value, ast.Dict)
    )

    assert len(stimulus_keys) == 1
    assert any(
        isinstance(key, ast.Constant) and key.value == "stimulus_id"
        for key in normal_out.keys
    )


@pytest.mark.parametrize(
    ("configured", "expected_state"),
    [(False, "aec_disabled"), (True, "aec_unavailable")],
)
def test_delay_summary_distinguishes_disabled_from_fail_open(
    configured, expected_state
):
    report = _aec_reference_delay_summary(
        _engine(aec_active=False),
        _config(aec_enabled=configured),
        stop_completed=True,
    )

    assert report["measurement_state"] == expected_state
    assert report["aec_active"] is False
    assert report["runtime_snapshot_available"] is False
    assert report["runtime_estimate_accepted"] is False
    assert report["operating_delay_samples"] is None
    assert report["operating_delay_ms"] is None


def test_delay_summary_reports_fixed_configured_seed_without_acceptance():
    report = _aec_reference_delay_summary(
        _engine(aec_active=True, operating=1280),
        _config(auto_delay=False),
        stop_completed=True,
    )

    assert report == {
        "measurement_state": "fixed_configured_delay",
        "configured_seed_ms": 80,
        "configured_seed_samples": 1280,
        "sample_rate_hz": 16000,
        "aec_enabled_configured": True,
        "aec_active": True,
        "auto_delay_enabled": False,
        "auto_delay_active": False,
        "runtime_snapshot_available": True,
        "runtime_estimate_accepted": False,
        "effective_seed_samples": 1280,
        "operating_delay_samples": 1280,
        "operating_delay_ms": 80.0,
    }


def test_delay_summary_rejects_fixed_operating_value_that_is_not_the_seed():
    report = _aec_reference_delay_summary(
        _engine(aec_active=True, operating=1920),
        _config(auto_delay=False),
        stop_completed=True,
    )

    assert report["measurement_state"] == "snapshot_unavailable"
    assert report["runtime_snapshot_available"] is False
    assert report["runtime_estimate_accepted"] is None
    assert report["operating_delay_samples"] is None


@pytest.mark.parametrize(
    ("accepted", "operating", "expected_state"),
    [
        (False, 1280, "awaiting_measurement"),
        (True, 1280, "accepted_measurement"),
        (True, 1920, "accepted_measurement"),
    ],
)
def test_delay_summary_uses_explicit_acceptance_even_when_value_equals_seed(
    accepted, operating, expected_state
):
    calibrator = _SnapshotCalibrator(AecDelaySnapshot(16000, 1280, operating, accepted))
    report = _aec_reference_delay_summary(
        _engine(aec_active=True, calibrator=calibrator, operating=operating),
        _config(),
        stop_completed=True,
    )

    assert report["measurement_state"] == expected_state
    assert report["runtime_snapshot_available"] is True
    assert report["runtime_estimate_accepted"] is accepted
    assert report["effective_seed_samples"] == 1280
    assert report["operating_delay_samples"] == operating
    assert report["operating_delay_ms"] == round(operating / 16.0, 3)
    assert calibrator.calls == 1


def test_delay_summary_preserves_exact_samples_for_non_16khz_rate():
    calibrator = _SnapshotCalibrator(AecDelaySnapshot(22050, 1543, 1544, True))
    report = _aec_reference_delay_summary(
        _engine(aec_active=True, calibrator=calibrator, operating=1544),
        _config(sample_rate=22050, seed_ms=70),
        stop_completed=True,
    )

    assert report["sample_rate_hz"] == 22050
    assert report["configured_seed_samples"] == 1543
    assert report["effective_seed_samples"] == 1543
    assert report["operating_delay_samples"] == 1544
    assert report["operating_delay_ms"] == 70.023


@pytest.mark.parametrize(
    ("stop_completed", "running", "retained"),
    [(False, False, False), (True, True, False), (True, False, True)],
)
def test_delay_summary_never_reads_retained_or_incomplete_stop_snapshot(
    stop_completed, running, retained
):
    engine = _engine(aec_active=True, calibrator=_ForbiddenSnapshot())
    if running:
        engine._running.set()
    if retained:
        engine._capture_resource_hold.set()
    report = _aec_reference_delay_summary(
        engine,
        _config(),
        stop_completed=stop_completed,
    )

    assert report["measurement_state"] == "snapshot_unavailable"
    assert report["runtime_snapshot_available"] is False
    assert report["runtime_estimate_accepted"] is None
    assert report["effective_seed_samples"] is None
    assert report["operating_delay_samples"] is None
    assert report["operating_delay_ms"] is None


def test_delay_summary_marks_missing_enabled_calibrator_unavailable():
    report = _aec_reference_delay_summary(
        _engine(aec_active=True, calibrator=None),
        _config(auto_delay=True),
        stop_completed=True,
    )

    assert report["measurement_state"] == "calibrator_unavailable"
    assert report["auto_delay_enabled"] is True
    assert report["auto_delay_active"] is False
    assert report["runtime_snapshot_available"] is False
    assert report["runtime_estimate_accepted"] is None
    assert report["operating_delay_samples"] is None


def test_delay_summary_rejects_snapshot_that_disagrees_with_engine_operating_delay():
    calibrator = _SnapshotCalibrator(AecDelaySnapshot(16000, 1280, 1920, True))
    report = _aec_reference_delay_summary(
        _engine(aec_active=True, calibrator=calibrator, operating=1280),
        _config(),
        stop_completed=True,
    )

    assert report["measurement_state"] == "snapshot_unavailable"
    assert report["runtime_snapshot_available"] is False
    assert report["runtime_estimate_accepted"] is None
    assert report["operating_delay_samples"] is None


def test_delay_summary_rejects_unaccepted_non_seed_operating_value():
    calibrator = _SnapshotCalibrator(AecDelaySnapshot(16000, 1280, 1920, False))
    report = _aec_reference_delay_summary(
        _engine(aec_active=True, calibrator=calibrator, operating=1920),
        _config(),
        stop_completed=True,
    )

    assert report["measurement_state"] == "snapshot_unavailable"
    assert report["runtime_snapshot_available"] is False
    assert report["runtime_estimate_accepted"] is None
    assert report["operating_delay_samples"] is None


def test_delay_summary_is_json_native_scalar_only():
    calibrator = _SnapshotCalibrator(AecDelaySnapshot(16000, 1280, 1920, True))
    report = _aec_reference_delay_summary(
        _engine(aec_active=True, calibrator=calibrator, operating=1920),
        _config(),
        stop_completed=True,
    )

    assert set(report) == {
        "measurement_state",
        "configured_seed_ms",
        "configured_seed_samples",
        "sample_rate_hz",
        "aec_enabled_configured",
        "aec_active",
        "auto_delay_enabled",
        "auto_delay_active",
        "runtime_snapshot_available",
        "runtime_estimate_accepted",
        "effective_seed_samples",
        "operating_delay_samples",
        "operating_delay_ms",
    }
    assert all(
        value is None or type(value) in (bool, float, int, str)
        for value in report.values()
    )
    assert json.loads(json.dumps(report)) == report


def test_echo_probe_source_keeps_open_speaker_guidance_diagnostic_only():
    source = Path(echo_probe_module.__file__).read_text(encoding="utf-8").lower()

    forbidden = (
        "headphone",
        "headset",
        "python -m core --session local",
        "logs/runs/*.txt",
        "self-calibrated safely",
        "safe for this room",
        "safe on echo",
        "healthy result",
        "raise the floor",
        "lower it only",
        "raise dtd_k",
        "lower dtd_k",
        "pick k between",
        "echo never approaches",
        "recommended",
        "winner",
    )
    required = (
        "quiet echo-only",
        "inconclusive",
        "does not select or tune",
        "bare-speaker",
        "talk over playback",
        "./live.sh",
        "python -m tools.live_audio_ab logs/runs/run-<id>.txt",
    )

    for fragment in forbidden:
        assert fragment not in source
    for fragment in required:
        assert fragment in source


def test_interrupt_suite_summary_ignores_additive_delay_and_guidance():
    cell = {
        "label": "mic/strategy",
        "self_interruptions": 0,
        "stimulus_id": (_STIMULUS_ID_PREFIX + _EXPECTED_STIMULUS_DIGESTS[3]),
        "playback_terminal_protocol": "tracked-sink-terminal-v1",
        "sentences_submitted": 3,
        "sentences_completed": 2,
        "sentences_interrupted": 1,
        "sentences_spoken": 3,
        "coherence": {
            "coherence_fired_on_own_tts": 0,
            "hint": "Quiet echo-only observation is inconclusive.",
            "note": "Diagnostic context only.",
        },
        "vad_flagged_during_play": 4,
        "peak_playback_level": 0.25,
        "median_mic_over_playback_dB": -12.0,
        "aec_reference_delay": {"measurement_state": "accepted_measurement"},
        "adaptive_dtd": {
            "hint": "This does not select or tune thresholds.",
            "note": "Run the bare-speaker live acceptance separately.",
        },
        "note": (
            "Use ./live.sh and inspect exactly one run with "
            "python -m tools.live_audio_ab logs/runs/run-<id>.txt"
        ),
    }

    assert summarize(cell) == {
        "label": "mic/strategy",
        "self_int": 0,
        "coh_fired": 0,
        "headroom_p95": None,
        "self_cal_margin": None,
        "vad_flagged": 4,
        "peak_play": 0.25,
        "mic_over_play_dB": -12.0,
        "error": None,
    }


def _step(calls, name, fault=None):
    def run():
        calls.append(name)
        if fault is not None:
            raise fault

    return run


def _assert_lifecycle_result(
    result,
    *,
    start_error_type=None,
    probe_error_type=None,
    stop_error_type=None,
    stop_completed,
    report_allowed,
):
    assert type(result) is _ProbeLifecycleResult
    assert result.start_error_type == start_error_type
    assert result.probe_error_type == probe_error_type
    assert result.stop_error_type == stop_error_type
    assert result.stop_completed is stop_completed
    assert result.report_allowed is report_allowed


def _decode_exactly_one_json_object(stdout):
    stripped = stdout.lstrip()
    value, end = json.JSONDecoder().raw_decode(stripped)
    assert stripped[end:].strip() == ""
    assert type(value) is dict
    return value


def _lifecycle_error_payload(result):
    return {
        "error": "probe-lifecycle-failed",
        "start_error_type": result.start_error_type,
        "probe_error_type": result.probe_error_type,
        "stop_error_type": result.stop_error_type,
    }


def test_probe_lifecycle_success_orders_start_probe_stop_once():
    calls = []

    result = _run_probe_lifecycle(
        _step(calls, "start"),
        _step(calls, "probe"),
        _step(calls, "stop"),
    )

    assert calls == ["start", "probe", "stop"]
    _assert_lifecycle_result(
        result,
        stop_completed=True,
        report_allowed=True,
    )


def test_probe_lifecycle_stop_runtime_error_disallows_report():
    calls = []

    result = _run_probe_lifecycle(
        _step(calls, "start"),
        _step(calls, "probe"),
        _step(calls, "stop", RuntimeError("stop-fault")),
    )

    assert calls == ["start", "probe", "stop"]
    _assert_lifecycle_result(
        result,
        stop_error_type="RuntimeError",
        stop_completed=False,
        report_allowed=False,
    )


@pytest.mark.parametrize("fault_phase", ["start", "probe"])
def test_probe_lifecycle_runtime_error_fails_closed_and_stops_once(fault_phase):
    calls = []
    fault = RuntimeError(f"{fault_phase}-fault")
    start = _step(calls, "start", fault if fault_phase == "start" else None)
    probe = _step(calls, "probe", fault if fault_phase == "probe" else None)

    result = _run_probe_lifecycle(start, probe, _step(calls, "stop"))

    assert calls == (
        ["start", "stop"] if fault_phase == "start" else ["start", "probe", "stop"]
    )
    _assert_lifecycle_result(
        result,
        start_error_type="RuntimeError" if fault_phase == "start" else None,
        probe_error_type="RuntimeError" if fault_phase == "probe" else None,
        stop_completed=True,
        report_allowed=False,
    )


@pytest.mark.parametrize("fault_phase", ["start", "probe"])
def test_probe_lifecycle_keeps_primary_and_stop_ordinary_failure_types(fault_phase):
    calls = []
    primary = RuntimeError("primary")
    start = _step(calls, "start", primary if fault_phase == "start" else None)
    probe = _step(calls, "probe", primary if fault_phase == "probe" else None)

    result = _run_probe_lifecycle(
        start,
        probe,
        _step(calls, "stop", ValueError("secondary")),
    )

    assert calls == (
        ["start", "stop"] if fault_phase == "start" else ["start", "probe", "stop"]
    )
    _assert_lifecycle_result(
        result,
        start_error_type="RuntimeError" if fault_phase == "start" else None,
        probe_error_type="RuntimeError" if fault_phase == "probe" else None,
        stop_error_type="ValueError",
        stop_completed=False,
        report_allowed=False,
    )


@pytest.mark.parametrize("fault_phase", ["start", "probe", "stop"])
def test_probe_lifecycle_keyboard_interrupt_preserves_identity_after_cleanup(
    fault_phase,
):
    calls = []
    interrupt = KeyboardInterrupt(fault_phase)
    start = _step(calls, "start", interrupt if fault_phase == "start" else None)
    probe = _step(calls, "probe", interrupt if fault_phase == "probe" else None)
    stop = _step(calls, "stop", interrupt if fault_phase == "stop" else None)

    with pytest.raises(KeyboardInterrupt) as raised:
        _run_probe_lifecycle(start, probe, stop)

    assert raised.value is interrupt
    assert calls == (
        ["start", "stop"] if fault_phase == "start" else ["start", "probe", "stop"]
    )


def test_warmup_keyboard_interrupt_still_stops_once_and_preserves_interrupt():
    calls = []
    interrupt = KeyboardInterrupt("warmup")

    def warmup_analogue():
        calls.append("warmup")
        raise interrupt

    with pytest.raises(KeyboardInterrupt) as raised:
        _run_probe_lifecycle(
            _step(calls, "start"),
            warmup_analogue,
            _step(calls, "stop"),
        )

    assert raised.value is interrupt
    assert calls == ["start", "warmup", "stop"]


@pytest.mark.parametrize(
    "stop_fault",
    [RuntimeError("stop"), KeyboardInterrupt("secondary-stop")],
)
def test_primary_keyboard_interrupt_wins_over_any_stop_failure(stop_fault):
    calls = []
    interrupt = KeyboardInterrupt("primary")

    with pytest.raises(KeyboardInterrupt) as raised:
        _run_probe_lifecycle(
            _step(calls, "start"),
            _step(calls, "probe", interrupt),
            _step(calls, "stop", stop_fault),
        )

    assert raised.value is interrupt
    assert calls == ["start", "probe", "stop"]


def test_stop_keyboard_interrupt_wins_over_ordinary_primary_failure():
    calls = []
    interrupt = KeyboardInterrupt("stop")

    with pytest.raises(KeyboardInterrupt) as raised:
        _run_probe_lifecycle(
            _step(calls, "start"),
            _step(calls, "probe", RuntimeError("probe")),
            _step(calls, "stop", interrupt),
        )

    assert raised.value is interrupt
    assert calls == ["start", "probe", "stop"]


@pytest.mark.parametrize("fault_phase", ["start", "probe", "stop"])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_system_exit_is_a_lifecycle_error_and_never_false_success(
    fault_phase,
    exit_code,
    capsys,
):
    calls = []
    fault = SystemExit(exit_code)
    result = _run_probe_lifecycle(
        _step(calls, "start", fault if fault_phase == "start" else None),
        _step(calls, "probe", fault if fault_phase == "probe" else None),
        _step(calls, "stop", fault if fault_phase == "stop" else None),
    )

    assert calls == (
        ["start", "stop"] if fault_phase == "start" else ["start", "probe", "stop"]
    )
    _assert_lifecycle_result(
        result,
        start_error_type="SystemExit" if fault_phase == "start" else None,
        probe_error_type="SystemExit" if fault_phase == "probe" else None,
        stop_error_type="SystemExit" if fault_phase == "stop" else None,
        stop_completed=fault_phase != "stop",
        report_allowed=False,
    )

    error_payload = _lifecycle_error_payload(result)
    assert "stimulus_id" not in error_payload
    rc = _emit_probe_result(error_payload, success=result.report_allowed)
    assert rc == 1
    assert _decode_exactly_one_json_object(capsys.readouterr().out) == error_payload


def test_returned_stop_with_retained_hold_allows_report_but_not_aec_snapshot():
    calls = []
    engine = _engine(aec_active=True, calibrator=_ForbiddenSnapshot())
    engine._capture_resource_hold.set()

    result = _run_probe_lifecycle(
        _step(calls, "start"),
        _step(calls, "probe"),
        _step(calls, "stop"),
    )
    report = _aec_reference_delay_summary(
        engine,
        _config(),
        stop_completed=result.stop_completed,
    )

    assert calls == ["start", "probe", "stop"]
    _assert_lifecycle_result(
        result,
        stop_completed=True,
        report_allowed=True,
    )
    assert report["measurement_state"] == "snapshot_unavailable"
    assert report["runtime_snapshot_available"] is False
    assert report["runtime_estimate_accepted"] is None
    assert report["operating_delay_samples"] is None


def test_emit_probe_result_success_is_one_json_object_and_returns_zero(capsys):
    stimulus_id = _STIMULUS_ID_PREFIX + _EXPECTED_STIMULUS_DIGESTS[4]
    payload = {
        "label": "probe",
        "self_interruptions": 0,
        "stimulus_id": stimulus_id,
    }

    rc = _emit_probe_result(payload, success=True)

    assert rc == 0
    emitted = _decode_exactly_one_json_object(capsys.readouterr().out)
    assert emitted == payload
    assert type(emitted["stimulus_id"]) is str
    assert re.fullmatch(
        r"echo-probe-spoken-text-v1:sha256:[0-9a-f]{64}",
        emitted["stimulus_id"],
    )


def test_emit_probe_result_lifecycle_error_is_one_json_object_and_returns_one(
    capsys,
):
    payload = {
        "error": "probe-lifecycle-failed",
        "start_error_type": None,
        "probe_error_type": "RuntimeError",
        "stop_error_type": None,
    }

    rc = _emit_probe_result(payload, success=False)

    assert rc == 1
    emitted = _decode_exactly_one_json_object(capsys.readouterr().out)
    assert emitted == payload
    assert "stimulus_id" not in emitted


@pytest.mark.parametrize(
    "payload",
    [
        {"value": float("nan")},
        {"value": object()},
    ],
)
def test_emit_probe_result_invalid_success_falls_back_to_one_error_object(
    payload,
    capsys,
):
    rc = _emit_probe_result(payload, success=True)

    assert rc == 1
    emitted = _decode_exactly_one_json_object(capsys.readouterr().out)
    assert emitted == {"error": "probe-result-serialization-failed"}
