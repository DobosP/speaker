"""Real-Ollama tier of the LLM-driven user simulator (opt-in, slow).

An LLM role-plays the persona/goal (SimulatedUser) and drives a REAL LocalLLM
assistant through the full pipeline; an LLM judge reports advisory verdicts.

Success is still gated by the deterministic GoalChecks (state_success inside
pass_hat_k). The hard assertion is a soft reliability floor so a small local
model does not red-bar CI; pass^k and judge verdicts are printed for inspection.

The whole module skips cleanly when Ollama (or the model) is unavailable, exactly
like the fixture graceful-degrade pattern in tests/conftest.py. Run with:
    ollama serve & ; ollama pull llama3.2:3b
    python -m pytest tests/test_user_simulation_llm.py -m "llm and slow" -v -s
"""
from __future__ import annotations

import json
import os

import pytest

from tests.sim.judge import LLMJudge
from tests.sim.ollama_adapter import DEFAULT_SIM_MODEL, OllamaChat, ollama_available
from tests.sim.runner import SimulationRunner, failed_checks, pass_hat_k
from tests.sim.scenarios import LLM_SCENARIOS
from tests.sim.user_agent import SimulatedUser

_OK, _REASON = ollama_available()

pytestmark = [
    pytest.mark.llm,
    pytest.mark.slow,
    pytest.mark.network,
    pytest.mark.skipif(not _OK, reason=f"Ollama unavailable: {_REASON}"),
]


@pytest.mark.parametrize("scenario", LLM_SCENARIOS, ids=lambda s: s.persona.name)
def test_real_persona_reaches_goal(scenario, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from utils.llm import LocalLLM

    chat = OllamaChat()
    runner = SimulationRunner(
        scenario.persona,
        scenario.goal,
        assistant_llm_factory=lambda: LocalLLM(model=DEFAULT_SIM_MODEL),
        user_factory=lambda: SimulatedUser(scenario.persona, scenario.goal, chat),
    )

    k = int(os.environ.get("SPEAKER_SIM_K", "3"))
    result = pass_hat_k(runner, k=k)
    judge = LLMJudge(chat)
    verdicts = [judge.evaluate(scenario.goal, t) for t in result["transcripts"]]

    # Advisory report (visible with -s); NOT the gate.
    print(
        json.dumps(
            {
                "scenario": scenario.persona.name,
                "pass_at_1": result["pass_at_1"],
                "pass_hat_k": result["pass_hat_k"],
                "failed_checks": [failed_checks(scenario.goal, t) for t in result["transcripts"]],
                "judge": [v.__dict__ for v in verdicts],
            },
            indent=2,
            default=str,
        )
    )

    # Deterministic anchor: a soft reliability floor.
    assert result["pass_at_1"] >= 0.5, "scenario fell below the reliability floor"
