"""Backlog: staged confirmations get a TTL (the pending-confirmation sibling of
reap_overdue_tasks). An abandoned "Confirm command: ..." must expire with a spoken
cancellation instead of waiting forever for a stray later "yes"."""

from __future__ import annotations

import time

from always_on_agent.models import IntentDecision, IntentKind
from always_on_agent.supervisor import AgentSupervisor


def _stage(sup, *, text="delete my files"):
    """Stage a requires-confirmation task through the real staging path so the
    TTL stamp is applied exactly as in production."""
    task = sup.tasks.create_task(
        IntentDecision(kind=IntentKind.COMMAND, confidence=1.0, text=text, reason="t")
    )
    task.metadata["requires_confirmation"] = True
    if sup._confirmation_ttl_sec > 0:
        task.metadata["confirmation_expires_at"] = time.monotonic() + sup._confirmation_ttl_sec
    sup.state.pending_confirmations[task.task_id] = task
    return task


def test_expired_confirmation_is_swept_with_spoken_cancellation():
    sup = AgentSupervisor(confirmation_ttl_sec=60.0)
    task = _stage(sup, text="empty the trash")

    swept = sup.sweep_expired_confirmations(now=time.monotonic() + 61.0)

    assert swept == 1
    assert task.task_id not in sup.state.pending_confirmations
    assert task.cancel_event.is_set()
    assert any("Confirmation expired" in s and "empty the trash" in s
               for s in sup.state.spoken_outputs)


def test_unexpired_confirmation_survives_the_sweep_and_still_confirms():
    sup = AgentSupervisor(confirmation_ttl_sec=60.0)
    task = _stage(sup)

    assert sup.sweep_expired_confirmations(now=time.monotonic() + 30.0) == 0
    assert task.task_id in sup.state.pending_confirmations

    sup._confirm_next(owner_verified=False)  # pre-expiry confirm still works
    assert task.task_id not in sup.state.pending_confirmations


def test_ttl_zero_disables_expiry():
    sup = AgentSupervisor(confirmation_ttl_sec=0.0)
    task = _stage(sup)
    assert "confirmation_expires_at" not in task.metadata
    assert sup.sweep_expired_confirmations(now=time.monotonic() + 10_000.0) == 0
    assert task.task_id in sup.state.pending_confirmations


def test_legacy_staged_task_without_stamp_never_expires():
    # A confirmation staged by an older path (no expires_at) must not be dropped.
    sup = AgentSupervisor(confirmation_ttl_sec=60.0)
    task = sup.tasks.create_task(
        IntentDecision(kind=IntentKind.COMMAND, confidence=1.0, text="legacy", reason="t")
    )
    sup.state.pending_confirmations[task.task_id] = task
    assert sup.sweep_expired_confirmations(now=time.monotonic() + 10_000.0) == 0
    assert task.task_id in sup.state.pending_confirmations


def test_real_staging_path_stamps_expiry_and_expires_end_to_end():
    # Drive the REAL staging branch: a COMMAND decision plans with
    # requires_confirmation=True, so _execute_decision stages it — the TTL stamp
    # must be applied there, and the sweep must then expire it.
    sup = AgentSupervisor(confirmation_ttl_sec=45.0)
    before = time.monotonic()
    sup._execute_decision(
        IntentDecision(kind=IntentKind.COMMAND, confidence=1.0, text="risky thing", reason="t")
    )

    assert len(sup.state.pending_confirmations) == 1
    task = next(iter(sup.state.pending_confirmations.values()))
    deadline = task.metadata["confirmation_expires_at"]
    assert before + 44.0 <= deadline <= time.monotonic() + 46.0

    assert sup.sweep_expired_confirmations(now=deadline + 1.0) == 1
    assert not sup.state.pending_confirmations
