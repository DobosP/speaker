"""Mock tier of the LLM-driven user simulator -- deterministic, CI-safe.

Both the user (ScriptedUser) and the assistant LLM (ScriptedStreamingLLM) are
scripted, so the whole conversation is deterministic and needs no network. The
real recorder, endpointing, router, and TTS player stay in the loop, so this
proves the SimulationRunner / GoalCheck / pass^k machinery end-to-end.
"""
from __future__ import annotations

import pytest

from tests.conversation_harness import ScriptedStreamingLLM
from tests.sim.runner import SimulationRunner, failed_checks, pass_hat_k
from tests.sim.scenarios import MOCK_SCENARIOS
from tests.sim.user_agent import ScriptedUser

pytestmark = [pytest.mark.dev, pytest.mark.audio]


@pytest.mark.parametrize("scenario", MOCK_SCENARIOS, ids=lambda s: s.persona.name)
def test_mock_persona_reaches_goal(scenario, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    # One scripted assistant response per LLM-invoked turn (capability/control
    # turns consume none). Rebuilt fresh per trial since the double is stateful.
    script = [[text] for text in scenario.assistant_script]
    runner = SimulationRunner(
        scenario.persona,
        scenario.goal,
        assistant_llm_factory=lambda: ScriptedStreamingLLM([list(r) for r in script]),
        user_factory=lambda: ScriptedUser(scenario.persona),
    )

    result = pass_hat_k(runner, k=3)

    assert result["pass_hat_k"] == 1.0, {
        "scenario": scenario.persona.name,
        "failures": [failed_checks(scenario.goal, t) for t in result["transcripts"]],
        "spoken": [t.assistant_spoken for t in result["transcripts"]],
        "actions": [t.route_actions for t in result["transcripts"]],
    }
