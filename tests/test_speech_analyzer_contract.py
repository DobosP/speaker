"""Cross-language CONTRACT golden for the deterministic speech analyzer.

This is the P5 anti-drift gate (cross-platform-1 / architecture-quality-4). The
control-plane decision (``observe`` -> ``decide``) and the Mode state machine are
frozen as transcript -> expected (SpeechObservation subset + IntentDecision)
fixtures in ``tests/golden/speech_analyzer_contract.json``.

The PYTHON analyzer must reproduce them (this test). When the mobile **Dart**
shell ports ``always_on_agent``'s analyzer onto the shared ``AgentEvent``/``Mode``
contract, its test MUST load the SAME JSON and reproduce it too -- so the desktop
and mobile runtimes provably cannot drift (the lenient mobile stop-check drift the
review flagged is exactly what this prevents). Authoring the fixtures Python-side
first is the gate; the Dart port lands behind it (needs a Flutter toolchain).

Regenerate ONLY on an intentional contract change, then review the JSON diff:

    python tests/test_speech_analyzer_contract.py
"""
from __future__ import annotations

import json
from pathlib import Path

from always_on_agent.events import Mode
from always_on_agent.models import IntentKind
from always_on_agent.speech_analyzer import LiveSpeechAnalyzer

GOLDEN = Path(__file__).parent / "golden" / "speech_analyzer_contract.json"

# Single-utterance decision cases covering every branch of decide():
# (name, text, is_final, current_mode).
CASES = [
    ("stop_phrase", "stop", True, Mode.ASSISTANT),
    ("confirm_phrase", "yes", True, Mode.COMMAND),
    ("deny_phrase", "no", True, Mode.COMMAND),
    ("mode_alias_research", "research mode", True, Mode.PASSIVE),
    ("research_prefix", "research local TTS options", True, Mode.ASSISTANT),
    ("search_prefix", "search for pipecat", True, Mode.ASSISTANT),
    ("dictate_prefix", "dictate hello world", True, Mode.ASSISTANT),
    ("command_prefix", "run the backup script", True, Mode.ASSISTANT),
    ("partial_non_control", "what is the weather", False, Mode.ASSISTANT),
    ("passive_no_activation", "the sky is blue", True, Mode.PASSIVE),
    ("passive_wake_activation", "assistant please help me", True, Mode.PASSIVE),
    ("search_mode_default", "find the report", True, Mode.SEARCH),
    ("research_mode_default", "the history of rome", True, Mode.RESEARCH),
    ("dictation_mode_default", "dear team comma", True, Mode.DICTATION),
    ("meeting_mode_default", "we agreed on friday", True, Mode.MEETING),
    ("command_mode_default", "lights on", True, Mode.COMMAND),
    ("assistant_mode_default", "what time is it", True, Mode.ASSISTANT),
    ("empty", "", True, Mode.ASSISTANT),
]

# Mode state machine: a sequence of finals from PASSIVE. A MODE_SWITCH decision
# advances the current mode to its target (the supervisor's _set_mode contract);
# every other decision leaves the mode unchanged. Each step records the decision
# kind + the resulting mode so the Dart port can prove the same transitions.
SEQUENCE = [
    "assistant mode",   # MODE_SWITCH -> assistant
    "what is pipecat",  # ASSISTANT
    "research mode",    # MODE_SWITCH -> research
    "the moon landing", # RESEARCH
    "stop",             # STOP (mode unchanged)
    "passive mode",     # MODE_SWITCH -> passive
    "the sky is blue",  # IGNORE (passive, no activation)
]


def _ser_decision(d) -> dict:
    return {
        "kind": d.kind.value,
        "confidence": round(d.confidence, 4),
        "text": d.text,
        "reason": d.reason,
        "mode": d.mode.value if d.mode else None,
        "target_mode": d.target_mode.value if d.target_mode else None,
        "requires_confirmation": d.requires_confirmation,
        "speak": d.speak,
        "starts_task": d.starts_task,
    }


def _ser_observation(o) -> dict:
    return {
        "normalized": o.normalized,
        "is_final": o.is_final,
        "language": o.language,
        "activation_score": round(o.activation_score, 4),
        "keywords": list(o.keywords),
    }


def run_case(case) -> dict:
    name, text, is_final, mode = case
    analyzer = LiveSpeechAnalyzer()  # fresh -> stateless/deterministic
    obs = analyzer.observe(text, is_final=is_final)
    dec = analyzer.decide(obs, mode)
    return {
        "name": name,
        "input": {"text": text, "is_final": is_final, "mode": mode.value},
        "observation": _ser_observation(obs),
        "decision": _ser_decision(dec),
    }


def run_sequence(utterances) -> list:
    analyzer = LiveSpeechAnalyzer()
    mode = Mode.PASSIVE
    steps = []
    for text in utterances:
        obs = analyzer.observe(text, is_final=True)
        dec = analyzer.decide(obs, mode)
        if dec.kind == IntentKind.MODE_SWITCH and dec.target_mode is not None:
            mode = dec.target_mode
        steps.append({"text": text, "kind": dec.kind.value, "mode": mode.value})
    return steps


def build_golden() -> dict:
    return {
        "cases": [run_case(c) for c in CASES],
        "sequence": {
            "start_mode": "passive",
            "utterances": list(SEQUENCE),
            "steps": run_sequence(SEQUENCE),
        },
    }


def test_speech_analyzer_matches_golden_contract():
    assert GOLDEN.exists(), (
        f"missing golden {GOLDEN}; regenerate with `python {Path(__file__).name}`"
    )
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    actual = build_golden()
    assert actual == expected, (
        "speech analyzer drifted from the frozen cross-language contract. If the "
        "change is intentional, regenerate the golden (python "
        "tests/test_speech_analyzer_contract.py) and review the diff -- and update "
        "the Dart port to match."
    )


if __name__ == "__main__":
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(build_golden(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN}")
