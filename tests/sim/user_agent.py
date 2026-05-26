"""User agents that produce the user's side of a conversation, turn by turn.

* ``ScriptedUser`` -- replays ``Persona.scripted_turns``; fully deterministic (CI).
* ``SimulatedUser`` -- an LLM role-plays the persona toward the goal (real tier).

Both expose ``next_turn(history) -> str | None`` where ``None`` means the user is
finished/satisfied. ``history`` is the list of (role, text) turns so far.
"""
from __future__ import annotations

from typing import Optional

from tests.sim.persona import Goal, Persona


class BaseUser:
    def next_turn(self, history: list[tuple[str, str]]) -> Optional[str]:
        raise NotImplementedError


class ScriptedUser(BaseUser):
    """Deterministic CI user: replays the persona's scripted turns in order."""

    def __init__(self, persona: Persona):
        self._turns = list(persona.scripted_turns)

    def next_turn(self, history: list[tuple[str, str]]) -> Optional[str]:
        return self._turns.pop(0) if self._turns else None


class SimulatedUser(BaseUser):
    """LLM-backed user. From the simulator's POV the assistant's replies are its
    input, so we role-flip the transcript when prompting the user-LLM."""

    _DONE = "<<DONE>>"

    def __init__(self, persona: Persona, goal: Goal, chat, max_turns: Optional[int] = None):
        self.persona = persona
        self.goal = goal
        self.chat = chat
        self.max_turns = max_turns if max_turns is not None else goal.max_turns
        self._count = 0

    def _system(self) -> str:
        return (
            f"You are role-playing a HUMAN user talking to a voice assistant.\n"
            f"Persona: {self.persona.name} ({self.persona.style}, language={self.persona.language}).\n"
            f"{self.persona.system_prompt}\n"
            f"Your goal: {self.goal.description}\n"
            f"Speak naturally and briefly, one short utterance per turn, as if talking out loud.\n"
            f"Output ONLY your next spoken line. When your goal is satisfied, output exactly {self._DONE}."
        )

    def next_turn(self, history: list[tuple[str, str]]) -> Optional[str]:
        if self._count >= self.max_turns:
            return None
        self._count += 1
        # Role-flip: assistant turns become "user" input to the user-LLM, and the
        # user-LLM's own prior turns become "assistant".
        messages = [
            {"role": "assistant" if role == "user" else "user", "content": text}
            for role, text in history
        ]
        if not messages:
            messages = [{"role": "user", "content": "(the assistant is listening)"}]
        reply = self.chat.complete(self._system(), messages).strip()
        if not reply or self._DONE in reply:
            return None
        # keep only the first line; models sometimes add stage directions
        return reply.splitlines()[0].strip()
