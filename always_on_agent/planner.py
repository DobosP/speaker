from __future__ import annotations

from dataclasses import dataclass, field

from .events import Mode
from .models import IntentDecision, IntentKind
from .text import keywords


@dataclass(frozen=True)
class PlanStep:
    name: str
    capability: str
    speak_result: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskPlan:
    intent: IntentKind
    mode: Mode
    input_text: str
    steps: tuple[PlanStep, ...]
    priority: int = 100
    requires_confirmation: bool = False
    speak_final: bool = True
    tags: tuple[str, ...] = ()


class TaskPlanner:
    """Builds explicit plans from intent decisions."""

    def plan(self, decision: IntentDecision) -> TaskPlan:
        mode = decision.mode or _mode_for_intent(decision.kind)
        tags = keywords(decision.text)
        if decision.kind == IntentKind.RESEARCH:
            if decision.metadata.get("search_scope") == "vault":
                return TaskPlan(
                    intent=decision.kind,
                    mode=mode,
                    input_text=decision.text,
                    steps=(
                        PlanStep("search", "vault.search"),
                        PlanStep("synthesize", "research.local", speak_result=True),
                    ),
                    priority=60,
                    speak_final=decision.speak,
                    tags=tags,
                )
            return TaskPlan(
                intent=decision.kind,
                mode=mode,
                input_text=decision.text,
                steps=(
                    PlanStep("scope", "research.scope"),
                    PlanStep("search", "web.search"),
                    PlanStep("synthesize", "research.local", speak_result=True),
                ),
                priority=60,
                speak_final=decision.speak,
                tags=tags,
            )
        if decision.kind == IntentKind.SEARCH:
            if decision.metadata.get("search_scope") == "vault":
                return TaskPlan(
                    intent=decision.kind,
                    mode=mode,
                    input_text=decision.text,
                    steps=(
                        PlanStep("search", "vault.search"),
                        PlanStep("synthesize", "research.local", speak_result=True),
                    ),
                    priority=70,
                    speak_final=decision.speak,
                    tags=tags,
                )
            return TaskPlan(
                intent=decision.kind,
                mode=mode,
                input_text=decision.text,
                steps=(PlanStep("search", "web.search", speak_result=True),),
                priority=70,
                speak_final=decision.speak,
                tags=tags,
            )
        if decision.kind == IntentKind.COMMAND:
            return TaskPlan(
                intent=decision.kind,
                mode=mode,
                input_text=decision.text,
                steps=(PlanStep("stage", "command.stage", speak_result=True),),
                priority=30,
                requires_confirmation=True,
                speak_final=decision.speak,
                tags=tags,
            )
        if decision.kind == IntentKind.DICTATION:
            return TaskPlan(
                intent=decision.kind,
                mode=mode,
                input_text=decision.text,
                steps=(PlanStep("clean", "dictation.clean"),),
                priority=80,
                speak_final=False,
                tags=tags,
            )
        if decision.kind == IntentKind.MEETING_NOTE:
            return TaskPlan(
                intent=decision.kind,
                mode=mode,
                input_text=decision.text,
                steps=(PlanStep("store", "meeting.note"),),
                priority=85,
                speak_final=False,
                tags=("meeting",) + tags,
            )
        return TaskPlan(
            intent=IntentKind.ASSISTANT,
            mode=mode,
            input_text=decision.text,
            steps=(PlanStep("answer", "assistant.answer", speak_result=True),),
            priority=90,
            speak_final=decision.speak,
            tags=tags,
        )


def _mode_for_intent(intent: IntentKind) -> Mode:
    if intent == IntentKind.SEARCH:
        return Mode.SEARCH
    if intent == IntentKind.RESEARCH:
        return Mode.RESEARCH
    if intent == IntentKind.COMMAND:
        return Mode.COMMAND
    if intent == IntentKind.DICTATION:
        return Mode.DICTATION
    if intent == IntentKind.MEETING_NOTE:
        return Mode.MEETING
    return Mode.ASSISTANT
