from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from core.engines._aec import AecDelaySnapshot
from tools.echo_probe import (
    _ProbeLifecycleResult,
    _aec_reference_delay_summary,
    _emit_probe_result,
    _run_probe_lifecycle,
)
from tools.interrupt_suite import summarize


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


def test_interrupt_suite_summary_ignores_additive_delay_block():
    cell = {
        "label": "mic/strategy",
        "self_interruptions": 0,
        "coherence": {"coherence_fired_on_own_tts": 0},
        "vad_flagged_during_play": 4,
        "peak_playback_level": 0.25,
        "median_mic_over_playback_dB": -12.0,
        "aec_reference_delay": {"measurement_state": "accepted_measurement"},
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
    payload = {"label": "probe", "self_interruptions": 0}

    rc = _emit_probe_result(payload, success=True)

    assert rc == 0
    assert _decode_exactly_one_json_object(capsys.readouterr().out) == payload


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
    assert _decode_exactly_one_json_object(capsys.readouterr().out) == payload


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
