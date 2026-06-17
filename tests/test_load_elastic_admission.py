"""control-plane-2: load-elastic task admission.

Under sustained system load the supervisor tightens its concurrent-task ceiling
to 1 so a second turn can't thrash an already-saturated CPU/GPU. Pure logic --
the load reader is a fake () -> Optional[float]; no models, no audio, no threads.
"""
from __future__ import annotations

from always_on_agent.events import Mode
from always_on_agent.models import IntentDecision, IntentKind
from always_on_agent.supervisor import AgentSupervisor


def _task(sup, text="hello"):
    return sup.tasks.create_task(
        IntentDecision(IntentKind.ASSISTANT, 0.9, text, "assistant_mode", mode=Mode.ASSISTANT)
    )


def _seat_active(sup, text="active"):
    t = _task(sup, text)
    sup.state.active_tasks[t.task_id] = t
    return t


def test_inert_without_a_load_reader():
    sup = AgentSupervisor()  # default: no load_fraction -> behaves exactly as before
    _seat_active(sup)
    assert sup._should_queue(_task(sup)) is False


def test_queues_a_second_task_under_load():
    sup = AgentSupervisor(load_fraction=lambda: 0.95, admission_load_ceiling=0.85)
    _seat_active(sup)
    assert sup._should_queue(_task(sup)) is True   # load high + 1 active -> cap to 1


def test_first_task_always_admits_even_under_load():
    sup = AgentSupervisor(load_fraction=lambda: 0.99, admission_load_ceiling=0.85)
    assert sup._should_queue(_task(sup)) is False   # no active task to be the "1"


def test_admits_below_the_load_ceiling():
    sup = AgentSupervisor(load_fraction=lambda: 0.40, admission_load_ceiling=0.85)
    _seat_active(sup)
    assert sup._should_queue(_task(sup)) is False


def test_none_load_sample_does_not_throttle():
    sup = AgentSupervisor(load_fraction=lambda: None, admission_load_ceiling=0.85)
    _seat_active(sup)
    assert sup._should_queue(_task(sup)) is False   # unknown load -> never starves admission


def test_load_throttled_task_admits_once_the_active_one_clears():
    # No permanent starvation: a turn queued under load is admitted as soon as the
    # active task drains (event-driven, not load-edge driven).
    sup = AgentSupervisor(load_fraction=lambda: 0.95, admission_load_ceiling=0.85)
    active = _seat_active(sup)
    queued = _task(sup, "second")
    assert sup._should_queue(queued) is True            # throttled while 1 is active
    sup.state.active_tasks.pop(active.task_id)           # the active task completes
    assert sup._should_queue(queued) is False            # now admits even under load
