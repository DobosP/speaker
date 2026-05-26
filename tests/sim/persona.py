"""Persona / Goal / GoalCheck model for the LLM-driven user simulator.

A scenario is a (Persona, Goal) pair. The Persona describes *who* is talking
(style, language, and either an LLM role-play prompt or deterministic scripted
turns); the Goal describes *what success looks like* as a set of deterministic
``GoalCheck`` predicates over the conversation transcript.

The GoalChecks are the anchor: per recent findings that LLM-simulated users are
imperfect proxies, conversation success is decided by these deterministic
predicates, not by the (advisory) LLM judge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:  # avoid import cycle at runtime
    from tests.sim.runner import SimTranscript


@dataclass(frozen=True)
class GoalCheck:
    """One deterministic success predicate over a finished conversation."""

    name: str
    # kind in: transcript_contains | tts_spoken | route_action | no_shutdown
    #          | shutdown | turn_count_max | custom
    kind: str
    target: Optional[str] = None
    predicate: Optional[Callable[["SimTranscript"], bool]] = None

    def passed(self, transcript: "SimTranscript") -> bool:
        spoken = " ".join(transcript.assistant_spoken).lower()
        if self.kind in ("transcript_contains", "tts_spoken"):
            return (self.target or "").lower() in spoken
        if self.kind == "route_action":
            return self.target in transcript.route_actions
        if self.kind == "no_shutdown":
            return not transcript.shutdown
        if self.kind == "shutdown":
            return transcript.shutdown
        if self.kind == "turn_count_max":
            return len(transcript.turns) <= int(self.target)
        if self.kind == "custom":
            assert self.predicate is not None, f"custom check {self.name} needs a predicate"
            return self.predicate(transcript)
        raise ValueError(f"unknown GoalCheck kind: {self.kind}")


@dataclass(frozen=True)
class Goal:
    """A user goal plus the deterministic checks that confirm it was met."""

    description: str  # natural-language goal handed to the user-LLM
    checks: tuple[GoalCheck, ...]
    max_turns: int = 4


@dataclass(frozen=True)
class Persona:
    """Who is talking. ``scripted_turns`` powers the deterministic CI user."""

    name: str
    style: str = "neutral"
    language: str = "en"
    system_prompt: str = ""  # role-play instructions for the user-LLM (real tier)
    scripted_turns: tuple[str, ...] = ()  # deterministic turns (mock tier)


@dataclass(frozen=True)
class Scenario:
    persona: Persona
    goal: Goal
    # Assistant replies for the MOCK tier, one per LLM-invoked turn (capability /
    # control turns consume none). Ignored by the real tier (real LocalLLM answers).
    assistant_script: tuple[str, ...] = field(default_factory=tuple)
