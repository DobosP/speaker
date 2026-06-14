"""P1b: owner-verified action plumbing + bound/gated confirm.

The action chokepoint (always_on_agent.origin, wired into command.stage in P0) is
fed by a turn's speaker-ID trust threaded event -> supervisor -> task -> capability
context. These pin that the trust is FAIL-CLOSED by default and that the confirm of
an owner-verified staged action requires an owner-verified "yes". Tier-0, no audio.
"""
from __future__ import annotations

from always_on_agent.capabilities import CapabilityRegistry, CapabilityResult, CapabilitySpec
from always_on_agent.events import AgentEvent, EventKind
from always_on_agent.models import IntentDecision, IntentKind
from always_on_agent.supervisor import AgentSupervisor
from always_on_agent.tasks import TaskRuntime


# --- event payload defaults (fail-closed) ------------------------------------

def test_final_event_owner_verified_defaults_fail_closed():
    ev = AgentEvent.final("delete my files")
    assert ev.payload["owner_verified"] is False
    assert ev.payload["origin"] == "unknown"
    ev2 = AgentEvent.final("x", owner_verified=True, origin="live_audio")
    assert ev2.payload["owner_verified"] is True and ev2.payload["origin"] == "live_audio"


def test_confirm_event_owner_verified_defaults_fail_closed():
    assert AgentEvent.confirm().payload["owner_verified"] is False
    assert AgentEvent.confirm(owner_verified=True).payload["owner_verified"] is True


# --- supervisor stamps the turn trust onto created tasks ----------------------

def test_create_task_stamps_turn_trust():
    sup = AgentSupervisor()
    sup.state.turn_owner_verified = True
    sup.state.turn_origin = "live_audio"
    decision = IntentDecision(kind=IntentKind.ASSISTANT, confidence=1.0, text="hi", reason="t")
    task = sup._create_task(decision)
    assert task.metadata["owner_verified"] is True
    assert task.metadata["origin"] == "live_audio"


def test_create_task_fail_closed_default_trust():
    sup = AgentSupervisor()  # fresh: turn trust defaults False/unknown
    decision = IntentDecision(kind=IntentKind.ASSISTANT, confidence=1.0, text="hi", reason="t")
    task = sup._create_task(decision)
    assert task.metadata["owner_verified"] is False
    assert task.metadata["origin"] == "unknown"


def test_handle_speech_sets_turn_trust_from_final():
    sup = AgentSupervisor()
    sup.handle_event(AgentEvent.final("hello", owner_verified=True, origin="live_audio"))
    assert sup.state.turn_owner_verified is True and sup.state.turn_origin == "live_audio"
    # a later unverified final RE-establishes (never inherits) trust
    sup.handle_event(AgentEvent.final("hello again"))
    assert sup.state.turn_owner_verified is False and sup.state.turn_origin == "unknown"


# --- tasks._invoke forwards trust into the capability context -----------------

def test_invoke_forwards_owner_verified_to_capability_context():
    seen = {}
    reg = CapabilityRegistry()

    def cap(query, context):
        seen.update(context)
        return CapabilityResult(True, "ok")

    reg.register("x.do", cap, spec=CapabilitySpec(name="x.do", summary="t", side_effecting=True))
    rt = TaskRuntime(lambda ev: None, reg)
    decision = IntentDecision(kind=IntentKind.COMMAND, confidence=1.0, text="do it", reason="t")
    task = rt.create_task(decision)
    task.metadata["owner_verified"] = True
    task.metadata["origin"] = "live_audio"
    rt._invoke(task, "x.do")
    assert seen["owner_verified"] is True and seen["origin"] == "live_audio"


def test_invoke_fail_closed_when_unstamped():
    seen = {}
    reg = CapabilityRegistry()
    reg.register("x.do", lambda q, c: (seen.update(c) or CapabilityResult(True, "ok")))
    rt = TaskRuntime(lambda ev: None, reg)
    task = rt.create_task(IntentDecision(kind=IntentKind.COMMAND, confidence=1.0, text="do", reason="t"))
    rt._invoke(task, "x.do")
    assert seen["owner_verified"] is False and seen["origin"] == "unknown"


# --- bound + owner-gated confirm ---------------------------------------------

def _stage(sup, *, owner_verified, text="delete my files"):
    """Put a fake staged (requires-confirmation) task into pending_confirmations."""
    task = sup.tasks.create_task(
        IntentDecision(kind=IntentKind.COMMAND, confidence=1.0, text=text, reason="t")
    )
    task.metadata["owner_verified"] = owner_verified
    sup.state.pending_confirmations[task.task_id] = task
    return task


def test_owner_verified_staged_action_needs_owner_verified_confirm():
    sup = AgentSupervisor()
    task = _stage(sup, owner_verified=True)
    # an ambient/unverified "yes" must NOT approve it
    sup._confirm_next(owner_verified=False)
    assert task.task_id in sup.state.pending_confirmations  # still pending
    assert any("verified-owner" in s for s in sup.state.spoken_outputs)
    # an owner-verified "yes" approves it
    sup._confirm_next(owner_verified=True)
    assert task.task_id not in sup.state.pending_confirmations
    assert any(s.startswith("Confirmed:") for s in sup.state.spoken_outputs)


def test_non_owner_gated_staged_action_confirms_freely():
    # A staged task NOT marked owner_verified (legacy / pre-enrollment) confirms as
    # before -- its execution is still gated by the capability chokepoint.
    sup = AgentSupervisor()
    task = _stage(sup, owner_verified=False)
    sup._confirm_next(owner_verified=False)
    assert task.task_id not in sup.state.pending_confirmations


def test_confirm_reads_back_specific_action():
    sup = AgentSupervisor()
    _stage(sup, owner_verified=False, text="empty the trash")
    sup._confirm_next(owner_verified=False)
    assert any("empty the trash" in s for s in sup.state.spoken_outputs)


def test_confirm_event_carries_owner_verified_through_handler():
    sup = AgentSupervisor()
    task = _stage(sup, owner_verified=True)
    # CONTROL_CONFIRM with owner_verified=False -> refused
    sup.handle_event(AgentEvent.confirm(owner_verified=False))
    assert task.task_id in sup.state.pending_confirmations
    sup.handle_event(AgentEvent.confirm(owner_verified=True))
    assert task.task_id not in sup.state.pending_confirmations
