"""LLM evals for the memory subsystem (real Ollama tier).

Applies the golden-set + soft-threshold pattern to the three LLM-backed memory
decisions:
  * LocalLLM.extract_user_info   -- entity extraction (name/location/...);
  * LocalLLM.summarize_conversation -- structured summary + topics;
  * OllamaMemoryCleanup.cleanup  -- the save/discard gate (precision on a small
    labeled set: substantive content kept, filler/control discarded).

Thresholds are soft (a small local model is noisy) and results are printed for
inspection. Skips cleanly when Ollama is unavailable.
"""
from __future__ import annotations

import json

import pytest

from tests.sim.ollama_adapter import DEFAULT_SIM_MODEL, ollama_available

_OK, _REASON = ollama_available()
pytestmark = [
    pytest.mark.llm,
    pytest.mark.slow,
    pytest.mark.skipif(not _OK, reason=f"Ollama unavailable: {_REASON}"),
]

# (message, substring that should appear somewhere in the extracted values)
_EXTRACTION_GOLDEN = [
    ("My name is Sarah and I live in Berlin.", "sarah"),
    ("I work as a software engineer.", "engineer"),
    ("I really love playing the guitar on weekends.", "guitar"),
]

# (utterance, expected worth_saving when gating substantive content)
_GATE_GOLDEN = [
    ("I prefer tea over coffee in the morning.", True),
    ("My flight to Tokyo is next Tuesday.", True),
    ("Remember that my dog's name is Rex.", True),
    ("uh um hmm", False),
    ("stop", False),
    ("...", False),
]


def test_extract_user_info_finds_entities():
    from utils.llm import LocalLLM

    llm = LocalLLM(model=DEFAULT_SIM_MODEL)
    hits = 0
    report = []
    for message, needle in _EXTRACTION_GOLDEN:
        info = llm.extract_user_info(message)
        blob = " ".join(str(v).lower() for v in info.values())
        found = needle in blob
        hits += int(found)
        report.append({"message": message, "needle": needle, "info": info, "found": found})
    print(json.dumps(report, indent=2, default=str))
    # Soft floor: most entities recovered (small models miss some).
    assert hits / len(_EXTRACTION_GOLDEN) >= 0.66


def test_summarize_conversation_is_structured_and_nonempty():
    from utils.llm import LocalLLM

    llm = LocalLLM(model=DEFAULT_SIM_MODEL)
    convo = [
        {"role": "user", "content": "I just moved to Madrid for a new data science job."},
        {"role": "assistant", "content": "Congratulations on the move and the new role."},
        {"role": "user", "content": "Yes, I am excited to explore the city on weekends."},
    ]
    result = llm.summarize_conversation(convo)
    print(json.dumps(result, indent=2, default=str))
    assert isinstance(result.get("summary", ""), str) and result["summary"].strip()
    assert isinstance(result.get("topics", []), list)


def test_cleanup_gate_precision_on_labeled_set():
    from utils.memory_writer import OllamaMemoryCleanup

    cleanup = OllamaMemoryCleanup(DEFAULT_SIM_MODEL, enabled=True)
    correct = 0
    report = []
    for text, expected in _GATE_GOLDEN:
        result = cleanup.cleanup(text, gate=True)
        ok = result.worth_saving == expected
        correct += int(ok)
        report.append(
            {"text": text, "expected": expected, "got": result.worth_saving, "reason": result.reason}
        )
    print(json.dumps(report, indent=2, default=str))
    # Soft floor: the gate should agree with labels most of the time.
    assert correct / len(_GATE_GOLDEN) >= 0.7
