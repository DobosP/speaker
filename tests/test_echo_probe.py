from __future__ import annotations

import ast
import hashlib
import json
import re
import struct
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.engines._aec import AecDelaySnapshot
from tools import echo_probe as echo_probe_module
from tools.echo_probe import (
    _ProbeLifecycleResult,
    _StimulusIdBuilder,
    _StimulusPlan,
    _aec_reference_delay_summary,
    _build_stimulus_plan,
    _emit_probe_result,
    _run_probe_lifecycle,
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


def test_main_hashes_the_same_lazy_plan_text_only_after_speak_returns():
    tree = ast.parse(Path(echo_probe_module.__file__).read_text(encoding="utf-8"))
    main_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    loop = next(
        node
        for node in ast.walk(main_node)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "text"
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "stimulus_plan"
    )
    calls = [node for node in ast.walk(loop) if isinstance(node, ast.Call)]
    speak = next(
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "speak"
    )
    record = next(
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "record"
    )

    assert isinstance(speak.args[0], ast.Name)
    assert isinstance(record.args[0], ast.Name)
    assert speak.args[0].id == record.args[0].id == loop.target.id
    assert speak.lineno < record.lineno


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
