from __future__ import annotations

from dataclasses import replace
import json
from threading import Barrier, Thread

import pytest

from always_on_agent.logical_turn import (
    LogicalTurnAbortReason,
    LogicalTurnBoundary,
)
from always_on_agent.logical_turn_shadow import (
    LogicalTurnCompositionShadow,
    LogicalTurnShadowScope,
    aggregate_logical_turn_shadows,
    validate_logical_turn_shadow_report,
)


def _observe(
    shadow: LogicalTurnCompositionShadow,
    epoch: int,
    text: str,
    *,
    held: bool,
) -> bool:
    return shadow.observe_native_final(
        native_epoch=epoch,
        native_sequence=0,
        text=text,
        held=held,
    )


def _closed_multi_epoch(
    *,
    legacy_text: str = "FIND IN MY VAULT",
) -> LogicalTurnCompositionShadow:
    shadow = LogicalTurnCompositionShadow()
    assert _observe(shadow, 1, "FIND IN", held=True)
    assert _observe(shadow, 2, "MY VAULT", held=False)
    assert shadow.compare_legacy_composition(
        boundary=LogicalTurnBoundary.SEMANTIC_COMPLETE,
        legacy_text=legacy_text,
    )
    shadow.close()
    return shadow


def test_single_epoch_match_is_closed_but_not_multi_epoch_evidence() -> None:
    shadow = LogicalTurnCompositionShadow()

    assert _observe(shadow, 1, "  SEARCH MY VAULT  ", held=False)
    assert shadow.compare_legacy_composition(
        boundary=LogicalTurnBoundary.NATIVE_FINAL,
        legacy_text="SEARCH MY VAULT",
    )
    shadow.close()

    snapshot = shadow.snapshot()
    assert snapshot.closed
    assert not snapshot.active
    assert snapshot.started == snapshot.committed == 1
    assert snapshot.native_epochs == snapshot.native_finals == 1
    assert snapshot.multi_epoch_commits == 0
    assert snapshot.composition_matches == 1
    report = aggregate_logical_turn_shadows((snapshot,))
    assert report["parity"]["evidence_sufficient"] is False
    assert report["parity"]["exact"] is False


def test_multiple_native_epochs_match_and_produce_sufficient_evidence() -> None:
    shadow = LogicalTurnCompositionShadow()

    assert _observe(shadow, 1, "FIND", held=True)
    assert _observe(shadow, 2, "IN MY", held=True)
    assert _observe(shadow, 3, "VAULT", held=False)
    assert shadow.compare_legacy_composition(
        boundary=LogicalTurnBoundary.HARD_LIMIT,
        legacy_text="FIND IN MY VAULT",
    )
    shadow.close()

    snapshot = shadow.snapshot()
    assert snapshot.native_epochs == snapshot.native_finals == 3
    assert snapshot.held_native_finals == 2
    assert snapshot.multi_epoch_commits == 1
    assert snapshot.boundary_hard_limit == 1
    report = aggregate_logical_turn_shadows((snapshot,))
    assert report["parity"]["evidence_sufficient"] is True
    assert report["parity"]["exact"] is True
    assert report["health"]["ok"] is True


def test_empty_held_and_terminal_epochs_preserve_prefix_once() -> None:
    shadow = LogicalTurnCompositionShadow()

    assert _observe(shadow, 1, "GO IN MY VAULT", held=True)
    assert shadow.observe_empty_native_epoch(native_epoch=2, held=True)
    assert shadow.observe_empty_native_epoch(native_epoch=3, held=False)
    assert shadow.compare_legacy_composition(
        boundary=LogicalTurnBoundary.SILENCE_LIMIT,
        legacy_text="GO IN MY VAULT",
    )
    shadow.close()

    snapshot = shadow.snapshot()
    assert snapshot.native_epochs == 3
    assert snapshot.native_finals == 1
    assert snapshot.empty_native_epochs == 2
    assert snapshot.multi_epoch_commits == 1
    assert snapshot.boundary_silence_limit == 1


def test_exact_mismatch_is_retained_only_as_a_counter() -> None:
    marker = "PRIVATE SENTINEL TRANSCRIPT"
    shadow = _closed_multi_epoch(legacy_text=marker)

    snapshot = shadow.snapshot()
    assert snapshot.composition_matches == 0
    assert snapshot.composition_mismatches == 1
    report = aggregate_logical_turn_shadows((snapshot,))
    assert report["parity"]["exact"] is False
    assert marker not in repr(shadow)
    assert marker not in repr(snapshot)
    assert marker not in json.dumps(report, sort_keys=True)


def test_rejected_epochs_and_bounds_do_not_half_admit_state() -> None:
    initial = LogicalTurnCompositionShadow(max_text_chars=3)
    assert not _observe(initial, 1, "LONG", held=False)
    initial_snapshot = initial.snapshot()
    assert initial_snapshot.started == 0
    assert initial_snapshot.native_epochs == 0
    assert initial_snapshot.bounds_overflows == 1
    assert not initial_snapshot.active
    initial.close()

    successor = LogicalTurnCompositionShadow(
        max_native_epochs=2,
        max_native_finals=2,
        max_text_chars=7,
    )
    assert _observe(successor, 1, "ONE", held=True)
    before = successor.snapshot()
    assert not _observe(successor, 3, "TWO", held=False)
    skipped = successor.snapshot()
    assert (
        replace(
            skipped,
            coordinator_rejections=before.coordinator_rejections,
        )
        == before
    )
    assert not _observe(successor, 2, "TOOLONG", held=False)
    bounded = successor.snapshot()
    assert bounded.active
    assert bounded.native_epochs == 1
    assert bounded.bounds_overflows == 1
    assert _observe(successor, 2, "TWO", held=False)
    assert successor.compare_legacy_composition(
        boundary=LogicalTurnBoundary.NATIVE_FINAL,
        legacy_text="ONE TWO",
    )
    successor.close()
    assert successor.snapshot().committed == 1


def test_epoch_disposition_rejects_revisions_successors_and_premature_commit() -> None:
    terminal = LogicalTurnCompositionShadow()
    assert _observe(terminal, 1, "FINAL", held=False)
    before = terminal.snapshot()
    assert not terminal.observe_native_final(
        native_epoch=1,
        native_sequence=1,
        text="REVISION",
        held=False,
    )
    assert not _observe(terminal, 2, "IMPOSSIBLE", held=False)
    assert not terminal.observe_empty_native_epoch(native_epoch=2, held=False)
    after = terminal.snapshot()
    assert after.native_epochs == before.native_epochs
    assert after.active
    assert after.coordinator_rejections == before.coordinator_rejections + 3
    assert terminal.compare_legacy_composition(
        boundary=LogicalTurnBoundary.NATIVE_FINAL,
        legacy_text="FINAL",
    )
    terminal.close()

    held = LogicalTurnCompositionShadow()
    assert _observe(held, 1, "WAIT", held=True)
    assert not held.compare_legacy_composition(
        boundary=LogicalTurnBoundary.NATIVE_FINAL,
        legacy_text="WAIT",
    )
    held.close()
    held_snapshot = held.snapshot()
    assert held_snapshot.aborted == 1
    assert held_snapshot.abort_control == 1
    assert held_snapshot.extra_legacy_terminals == 1


@pytest.mark.parametrize("reason", tuple(LogicalTurnAbortReason))
def test_every_abort_reason_closes_one_active_turn(
    reason: LogicalTurnAbortReason,
) -> None:
    shadow = LogicalTurnCompositionShadow()
    assert _observe(shadow, 1, "PRIVATE", held=True)

    assert shadow.abort(reason)
    shadow.close()

    snapshot = shadow.snapshot()
    assert snapshot.started == snapshot.aborted == 1
    assert snapshot.committed == 0
    field = {
        LogicalTurnAbortReason.TRANSCRIPT_ABORT: "abort_transcript",
        LogicalTurnAbortReason.CONTROL: "abort_control",
        LogicalTurnAbortReason.STOP: "abort_stop",
        LogicalTurnAbortReason.SHUTDOWN: "abort_shutdown",
    }[reason]
    assert getattr(snapshot, field) == 1


def test_close_is_idempotent_and_late_input_is_counted() -> None:
    shadow = LogicalTurnCompositionShadow()
    assert _observe(shadow, 1, "UNFINISHED", held=True)

    shadow.close()
    shadow.close()
    assert not _observe(shadow, 2, "LATE", held=False)

    snapshot = shadow.snapshot()
    assert snapshot.closed
    assert snapshot.abort_shutdown == 1
    assert snapshot.missing_legacy_terminals == 1
    assert snapshot.late_observations == 1


def test_internal_abort_failure_still_erases_and_closes_state(monkeypatch) -> None:
    shadow = LogicalTurnCompositionShadow()
    assert _observe(shadow, 1, "PRIVATE", held=True)

    def fail_abort(*_args, **_kwargs):
        raise RuntimeError("fixed internal failure")

    monkeypatch.setattr(shadow._coordinator, "abort", fail_abort)
    shadow.close()

    snapshot = shadow.snapshot()
    assert snapshot.closed
    assert not snapshot.active
    assert snapshot.started == snapshot.aborted == 1
    assert snapshot.abort_shutdown == 1
    assert snapshot.internal_errors == 1


@pytest.mark.parametrize("operation", ("abort", "observe"))
def test_close_race_terminalizes_once_without_deadlock(operation: str) -> None:
    shadow = LogicalTurnCompositionShadow()
    assert _observe(shadow, 1, "PRIVATE", held=True)
    barrier = Barrier(3)

    def close() -> None:
        barrier.wait()
        shadow.close()

    def compete() -> None:
        barrier.wait()
        if operation == "abort":
            shadow.abort(LogicalTurnAbortReason.CONTROL)
        else:
            _observe(shadow, 2, "SUCCESSOR", held=False)

    workers = (Thread(target=close), Thread(target=compete))
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=2.0)

    assert all(not worker.is_alive() for worker in workers)
    snapshot = shadow.snapshot()
    assert snapshot.closed
    assert not snapshot.active
    assert snapshot.started == snapshot.aborted == 1
    assert snapshot.committed == 0
    assert snapshot.abort_control + snapshot.abort_shutdown == 1


def test_string_subclasses_are_rejected_without_running_user_code() -> None:
    touched: list[str] = []

    class Hostile(str):
        def strip(self, *args, **kwargs):
            touched.append("strip")
            return super().strip(*args, **kwargs)

        def __eq__(self, other):
            touched.append("equal")
            return super().__eq__(other)

    shadow = LogicalTurnCompositionShadow()
    assert not _observe(shadow, 1, Hostile("PRIVATE"), held=False)
    assert touched == []
    assert _observe(shadow, 1, "SAFE", held=False)
    assert not shadow.compare_legacy_composition(
        boundary=LogicalTurnBoundary.NATIVE_FINAL,
        legacy_text=Hostile("SAFE"),
    )
    assert touched == []
    assert shadow.compare_legacy_composition(
        boundary=LogicalTurnBoundary.NATIVE_FINAL,
        legacy_text="SAFE",
    )
    shadow.close()
    assert shadow.snapshot().coordinator_rejections == 2


def test_aggregate_requires_closed_snapshots_and_fixed_capabilities() -> None:
    open_shadow = LogicalTurnCompositionShadow()
    with pytest.raises(ValueError, match="closed"):
        aggregate_logical_turn_shadows((open_shadow.snapshot(),))

    first = _closed_multi_epoch()
    second = _closed_multi_epoch()
    report = aggregate_logical_turn_shadows((first.snapshot(), second.snapshot()))
    assert report["coverage"] == {
        "runs": 2,
        "closed_runs": 2,
        "active_at_report": 0,
        "complete": True,
    }
    assert report["capabilities"]["runtime_authority"] is False
    assert report["capabilities"]["abort_terminal_parity"] is False
    assert report["capabilities"]["selected_final_parity"] is False
    validate_logical_turn_shadow_report(report)


@pytest.mark.parametrize(
    ("section", "name", "value"),
    (
        ("coverage", "runs", 0),
        ("coverage", "closed_runs", 0),
        ("lifecycle", "held_native_finals", 99),
        ("lifecycle", "held_native_finals", 0),
        ("lifecycle", "committed", 0),
        ("parity", "paired_terminals", 0),
        ("parity", "multi_epoch_commits", 99),
        ("parity", "evidence_sufficient", False),
        ("parity", "exact", False),
        ("health", "ok", False),
    ),
)
def test_report_validator_rejects_impossible_mutations(
    section: str,
    name: str,
    value: object,
) -> None:
    report = aggregate_logical_turn_shadows((_closed_multi_epoch().snapshot(),))
    mutated = json.loads(json.dumps(report))
    mutated[section][name] = value

    with pytest.raises((TypeError, ValueError)):
        validate_logical_turn_shadow_report(mutated)


def test_report_validator_rejects_coordinated_causal_forgeries() -> None:
    report = aggregate_logical_turn_shadows((_closed_multi_epoch().snapshot(),))

    without_readiness = json.loads(json.dumps(report))
    without_readiness["lifecycle"]["readiness"] = 0
    without_readiness["boundaries"]["semantic_complete"] = 0
    with pytest.raises(ValueError, match="readiness"):
        validate_logical_turn_shadow_report(without_readiness)

    one_epoch_multi = json.loads(json.dumps(report))
    one_epoch_multi["lifecycle"]["native_epochs"] = 1
    one_epoch_multi["lifecycle"]["native_finals"] = 1
    with pytest.raises(ValueError, match="native epoch"):
        validate_logical_turn_shadow_report(one_epoch_multi)

    commit_without_final = json.loads(json.dumps(report))
    commit_without_final["lifecycle"]["native_finals"] = 0
    commit_without_final["lifecycle"]["empty_native_epochs"] = 2
    commit_without_final["lifecycle"]["held_native_finals"] = 0
    commit_without_final["parity"]["multi_epoch_commits"] = 0
    commit_without_final["parity"]["evidence_sufficient"] = False
    commit_without_final["parity"]["exact"] = False
    with pytest.raises(ValueError, match="final"):
        validate_logical_turn_shadow_report(commit_without_final)

    readiness_without_terminal = json.loads(json.dumps(report))
    readiness_without_terminal["lifecycle"]["readiness"] = 2
    readiness_without_terminal["boundaries"]["native_final"] = 1
    with pytest.raises(ValueError, match="exceed"):
        validate_logical_turn_shadow_report(readiness_without_terminal)

    held_without_terminal_epoch = json.loads(json.dumps(report))
    held_without_terminal_epoch["lifecycle"]["native_epochs"] = 1
    held_without_terminal_epoch["lifecycle"]["native_finals"] = 1
    held_without_terminal_epoch["parity"]["multi_epoch_commits"] = 0
    held_without_terminal_epoch["parity"]["evidence_sufficient"] = False
    held_without_terminal_epoch["parity"]["exact"] = False
    with pytest.raises(ValueError, match="exceed"):
        validate_logical_turn_shadow_report(held_without_terminal_epoch)

    all_empty = json.loads(json.dumps(report))
    all_empty["lifecycle"].update(
        {
            "native_finals": 0,
            "empty_native_epochs": 2,
            "held_native_finals": 0,
            "readiness": 0,
            "committed": 0,
            "aborted": 1,
        }
    )
    all_empty["boundaries"]["semantic_complete"] = 0
    all_empty["abort_reasons"]["control"] = 1
    all_empty["parity"].update(
        {
            "paired_terminals": 0,
            "extra_legacy_terminals": 1,
            "composition_matches": 0,
            "multi_epoch_commits": 0,
            "evidence_sufficient": False,
            "exact": False,
        }
    )
    with pytest.raises(ValueError, match="non-empty final"):
        validate_logical_turn_shadow_report(all_empty)

    failed_ready_without_control_abort = json.loads(json.dumps(report))
    failed_ready_without_control_abort["lifecycle"].update(
        {
            "started": 2,
            "native_epochs": 3,
            "native_finals": 3,
            "readiness": 2,
            "aborted": 1,
        }
    )
    failed_ready_without_control_abort["boundaries"]["native_final"] = 1
    failed_ready_without_control_abort["abort_reasons"]["shutdown"] = 1
    failed_ready_without_control_abort["parity"]["legacy_terminals"] = 2
    failed_ready_without_control_abort["parity"]["extra_legacy_terminals"] = 1
    with pytest.raises(ValueError, match="control aborts"):
        validate_logical_turn_shadow_report(failed_ready_without_control_abort)

    zero_evidence_abort = json.loads(json.dumps(report))
    zero_evidence_abort["lifecycle"]["started"] = 2
    zero_evidence_abort["lifecycle"]["aborted"] = 1
    zero_evidence_abort["abort_reasons"]["control"] = 1
    with pytest.raises(ValueError, match="started and multi-epoch"):
        validate_logical_turn_shadow_report(zero_evidence_abort)

    too_many_terminal_finals = json.loads(json.dumps(report))
    too_many_terminal_finals["lifecycle"]["native_epochs"] = 3
    too_many_terminal_finals["lifecycle"]["native_finals"] = 3
    with pytest.raises(ValueError, match="unheld finals exceed started"):
        validate_logical_turn_shadow_report(too_many_terminal_finals)

    single_without_terminal_final = json.loads(json.dumps(report))
    single_without_terminal_final["lifecycle"].update(
        {
            "started": 2,
            "native_epochs": 4,
            "native_finals": 2,
            "empty_native_epochs": 2,
            "held_native_finals": 2,
            "readiness": 2,
            "committed": 2,
        }
    )
    single_without_terminal_final["boundaries"]["native_final"] = 1
    single_without_terminal_final["parity"].update(
        {
            "legacy_terminals": 2,
            "paired_terminals": 2,
            "composition_matches": 2,
        }
    )
    with pytest.raises(ValueError, match="single-epoch commits"):
        validate_logical_turn_shadow_report(single_without_terminal_final)


def test_report_validator_rejects_unknown_fields_and_capability_changes() -> None:
    report = aggregate_logical_turn_shadows((_closed_multi_epoch().snapshot(),))
    unknown = json.loads(json.dumps(report))
    unknown["private_metadata"] = {}
    with pytest.raises(ValueError, match="shape"):
        validate_logical_turn_shadow_report(unknown)

    capability = json.loads(json.dumps(report))
    capability["capabilities"]["runtime_authority"] = True
    with pytest.raises(ValueError, match="capabilities"):
        validate_logical_turn_shadow_report(capability)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", True),
        ("scope", LogicalTurnShadowScope.SHERPA_RAW_COMPOSITION),
    ),
)
def test_report_validator_rejects_json_scalar_type_aliases(
    field: str,
    value: object,
) -> None:
    report = aggregate_logical_turn_shadows((_closed_multi_epoch().snapshot(),))
    report[field] = value

    with pytest.raises(ValueError, match="identity"):
        validate_logical_turn_shadow_report(report)


@pytest.mark.parametrize("value", (0, 1))
def test_report_validator_requires_boolean_capabilities(value: int) -> None:
    report = aggregate_logical_turn_shadows((_closed_multi_epoch().snapshot(),))
    report["capabilities"]["runtime_authority"] = value

    with pytest.raises(ValueError, match="capabilities"):
        validate_logical_turn_shadow_report(report)
