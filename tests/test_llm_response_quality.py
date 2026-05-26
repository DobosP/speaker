"""Response-quality evals for the assistant LLM (utils.llm.LocalLLM).

Combines three SOTA techniques on the REAL model:
  * property checks  -- the voice-format contract (<=2-3 sentences, no emoji,
    no filler) asserted on real output (deterministic predicates);
  * metamorphic      -- format-robustness: an adversarial "write an essay" prompt
    must STILL yield a short spoken reply (the pipeline's guarantee);
  * LLM-as-judge     -- advisory relevance scoring against a reference, reported
    and held to a soft floor (a single weak local model is not a hard oracle).

The predicate logic is also unit-tested on fabricated strings, so that part runs
in CI with no model. The model-dependent tests skip cleanly without Ollama.
"""
from __future__ import annotations

import json
import pathlib

import pytest
import yaml

from tests.sim.ollama_adapter import DEFAULT_SIM_MODEL, OllamaChat, ollama_available
from tests.sim.quality import (
    count_sentences,
    has_emoji,
    is_voice_format_ok,
    voice_format_violations,
)

_DATA = pathlib.Path(__file__).parent / "router_data" / "llm_eval.yaml"
CASES = yaml.safe_load(_DATA.read_text(encoding="utf-8"))["cases"]
_OK, _REASON = ollama_available()
_needs_ollama = pytest.mark.skipif(not _OK, reason=f"Ollama unavailable: {_REASON}")


# ── Pure predicate unit tests (run everywhere, no model) ─────────────────────
def test_quality_predicates_flag_violations():
    assert is_voice_format_ok("Paris is the capital of France.")
    assert not is_voice_format_ok("")
    assert "emoji" in voice_format_violations("All good \U0001f600")
    assert "filler_prefix" in voice_format_violations("Ah, here is the answer.")
    long_essay = ". ".join(f"Sentence number {i}" for i in range(12)) + "."
    issues = voice_format_violations(long_essay)
    assert any(v.startswith("too_many_sentences") for v in issues)


def test_count_sentences_and_emoji_helpers():
    assert count_sentences("One. Two! Three?") == 3
    assert count_sentences("just one clause") == 1
    assert has_emoji("nice \U0001f44d") is True
    assert has_emoji("plain text") is False


# ── Real-model tier (llm + slow) ─────────────────────────────────────────────
@pytest.mark.llm
@pytest.mark.slow
@_needs_ollama
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_response_obeys_voice_format(case):
    from utils.llm import LocalLLM

    llm = LocalLLM(model=DEFAULT_SIM_MODEL)
    reply = llm.get_response(case["prompt"])
    issues = voice_format_violations(reply)
    assert not issues, f"{case['id']}: format violations {issues} in reply {reply!r}"


@pytest.mark.llm
@pytest.mark.slow
@_needs_ollama
@pytest.mark.parametrize(
    "case", [c for c in CASES if c.get("format_attack")], ids=lambda c: c["id"]
)
def test_format_robust_against_long_answer_attack(case):
    # Metamorphic relation: a prompt designed to elicit a long answer must not
    # break the short-spoken-reply contract.
    from utils.llm import LocalLLM

    llm = LocalLLM(model=DEFAULT_SIM_MODEL)
    reply = llm.get_response(case["prompt"])
    assert is_voice_format_ok(reply), f"{case['id']}: essay attack produced {reply!r}"


@pytest.mark.llm
@pytest.mark.slow
@_needs_ollama
def test_judge_relevance_floor():
    # LLM-as-judge over the non-attack prompts; advisory, soft floor.
    from utils.llm import LocalLLM

    llm = LocalLLM(model=DEFAULT_SIM_MODEL)
    judge = OllamaChat()
    qa_cases = [c for c in CASES if not c.get("format_attack")]
    results = []
    relevant = 0
    for case in qa_cases:
        reply = llm.get_response(case["prompt"])
        verdict_raw = judge.complete(
            "You are a strict grader. Output JSON only: "
            '{"relevant": bool, "reason": string}.',
            [
                {
                    "role": "user",
                    "content": (
                        f"Question: {case['prompt']}\n"
                        f"Reference answer: {case['reference']}\n"
                        f"Assistant reply: {reply}\n"
                        "Is the assistant reply relevant and consistent with the reference?"
                    ),
                }
            ],
            json_mode=True,
        )
        try:
            ok = bool(json.loads(verdict_raw).get("relevant", False))
        except json.JSONDecodeError:
            ok = False
        relevant += int(ok)
        results.append({"id": case["id"], "relevant": ok, "reply": reply})

    score = relevant / len(qa_cases)
    print(json.dumps({"relevance_score": score, "results": results}, indent=2, default=str))
    assert score >= 0.6, f"relevance below floor: {score:.2f}"
