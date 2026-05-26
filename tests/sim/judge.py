"""LLM-as-judge: advisory scoring of a finished conversation against a rubric.

The judge is ADVISORY only. Per recent findings that one non-deterministic model
is a poor sole arbiter of another, conversation success is gated by the
deterministic ``GoalCheck`` predicates (see runner.state_success). The judge's
verdict is recorded for reporting and for measuring state/judge agreement -- a
quality signal on the simulator itself.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re

from tests.sim.persona import Goal
from tests.sim.runner import SimTranscript


@dataclass(frozen=True)
class Rubric:
    criteria: tuple[str, ...] = (
        "goal_satisfied: did the assistant accomplish the user's stated goal?",
        "voice_format_ok: are replies short (1-2 sentences), with no emoji or filler?",
        "relevance: were replies on-topic and free of obvious hallucination?",
        "control_obeyed: if the user asked to stop/quit, did the assistant comply?",
    )


@dataclass(frozen=True)
class Verdict:
    goal_satisfied: bool
    voice_format_ok: bool
    relevance_score: int  # 1-5
    reasoning: str
    raw: str


def _render(transcript: SimTranscript) -> str:
    return "\n".join(f"{role}: {text}" for role, text in transcript.turns)


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


class LLMJudge:
    def __init__(self, chat, rubric: Rubric = Rubric()):
        self.chat = chat
        self.rubric = rubric

    def evaluate(self, goal: Goal, transcript: SimTranscript) -> Verdict:
        system = (
            "You are a strict evaluator of a voice assistant conversation. "
            "Think step by step about each criterion, THEN output a single JSON object "
            'with keys: goal_satisfied (bool), voice_format_ok (bool), '
            'relevance_score (int 1-5), reasoning (string).'
        )
        user = (
            f"User goal: {goal.description}\n\n"
            f"Rubric:\n- " + "\n- ".join(self.rubric.criteria) + "\n\n"
            f"Conversation:\n{_render(transcript)}\n\n"
            "Return only the JSON object."
        )
        raw = self.chat.complete(system, [{"role": "user", "content": user}], json_mode=True)
        data = _extract_json(raw)
        return Verdict(
            goal_satisfied=bool(data.get("goal_satisfied", False)),
            voice_format_ok=bool(data.get("voice_format_ok", False)),
            relevance_score=int(data.get("relevance_score", 0) or 0),
            reasoning=str(data.get("reasoning", "")),
            raw=raw,
        )
