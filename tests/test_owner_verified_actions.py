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


# --- review fixes: trust-laundering paths ------------------------------------

def test_partial_yes_cannot_launder_prior_turn_trust():
    # BLOCKER fix: a CONFIRM from a PARTIAL would carry the prior final's trust.
    # The analyzer must NOT emit CONFIRM/DENY on a partial.
    from always_on_agent.speech_analyzer import LiveSpeechAnalyzer
    from always_on_agent.events import Mode

    an = LiveSpeechAnalyzer()
    partial = an.observe("yes", is_final=False)
    d_partial = an.decide(partial, Mode.ASSISTANT, has_pending_confirmation=True)
    assert d_partial.kind != IntentKind.CONFIRM  # partial "yes" does NOT confirm
    final = an.observe("yes", is_final=True)
    d_final = an.decide(final, Mode.ASSISTANT, has_pending_confirmation=True)
    assert d_final.kind == IntentKind.CONFIRM  # the final does


def test_followup_task_is_fail_closed_not_owner_verified():
    # A proactive followup (assistant-initiated) must never inherit the prior turn's
    # owner trust. Drive the REAL _emit_followup with followups enabled.
    from always_on_agent.followups import FollowupConfig

    sup = AgentSupervisor(followup_config=FollowupConfig(enabled=True))
    sup.state.turn_owner_verified = True  # prior turn was owner-verified
    sup.state.turn_origin = "live_audio"
    captured = {}
    sup._start_task = lambda task, *a, **k: captured.update(  # type: ignore[assignment]
        owner_verified=task.metadata.get("owner_verified"), origin=task.metadata.get("origin")
    )
    sup._emit_followup()
    assert captured.get("owner_verified") is False
    assert captured.get("origin") == "system"


def test_fold_continuation_demotes_unverified_addon():
    # FOLD must DEMOTE: an unverified add-on folded into an owner-verified queued
    # continuation revokes its owner trust (real _maybe_continue FOLD branch).
    from always_on_agent.continuation import CONTINUE, ContinuationConfig
    from always_on_agent.events import Mode

    class _AlwaysContinue:
        def classify(self, *a, **k):
            return CONTINUE

    sup = AgentSupervisor(
        continuation_config=ContinuationConfig(enabled=True),
        continuation=_AlwaysContinue(),
    )
    # A victim task that is already "speaking" (after first audio) with a queued
    # owner-verified continuation behind it.
    victim = sup.tasks.create_task(
        IntentDecision(kind=IntentKind.ASSISTANT, confidence=1.0, text="tell a story", reason="t", mode=Mode.ASSISTANT)
    )
    victim.started_speaking = True  # after first audio -> FOLD/queue path, not MERGE
    sup.state.active_tasks[victim.task_id] = victim
    queued = sup.tasks.create_task(
        IntentDecision(kind=IntentKind.ASSISTANT, confidence=1.0, text="make it longer", reason="t", mode=Mode.ASSISTANT)
    )
    queued.metadata["owner_verified"] = True
    queued.metadata["origin"] = "live_audio"
    queued.metadata["continue_after"] = victim.task_id  # marker _pending_continuation_behind matches
    sup.state.queued_tasks.append(queued)
    # An UNVERIFIED add-on turn folds into the queued continuation.
    sup.state.turn_owner_verified = False
    sup.state.turn_origin = "unknown"
    decision = IntentDecision(kind=IntentKind.ASSISTANT, confidence=1.0, text="and funnier", reason="t", mode=Mode.ASSISTANT)
    handled = sup._maybe_continue(decision)
    assert handled is True
    assert queued.metadata["owner_verified"] is False  # demoted
