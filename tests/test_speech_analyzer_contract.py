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
# (name, text, is_final, current_mode, has_pending_confirmation).
# CONFIRM/DENY only fire when a confirmation is actually pending; the
# *_no_pending cases pin the fall-through (a bare "yes"/"no" answering the
# assistant is a normal turn, not a dropped control reply).
CASES = [
    ("stop_phrase", "stop", True, Mode.ASSISTANT, False),
    ("confirm_phrase", "yes", True, Mode.COMMAND, True),
    ("deny_phrase", "no", True, Mode.COMMAND, True),
    ("confirm_no_pending", "yes", True, Mode.ASSISTANT, False),
    ("deny_no_pending", "no", True, Mode.ASSISTANT, False),
    ("mode_alias_research", "research mode", True, Mode.PASSIVE, False),
    ("research_prefix", "research local TTS options", True, Mode.ASSISTANT, False),
    ("search_prefix", "search for pipecat", True, Mode.ASSISTANT, False),
    ("dictate_prefix", "dictate hello world", True, Mode.ASSISTANT, False),
    ("command_prefix", "run the backup script", True, Mode.ASSISTANT, False),
    ("partial_non_control", "what is the weather", False, Mode.ASSISTANT, False),
    ("passive_no_activation", "the sky is blue", True, Mode.PASSIVE, False),
    ("passive_wake_activation", "assistant please help me", True, Mode.PASSIVE, False),
    ("search_mode_default", "find the report", True, Mode.SEARCH, False),
    ("research_mode_default", "the history of rome", True, Mode.RESEARCH, False),
    ("dictation_mode_default", "dear team comma", True, Mode.DICTATION, False),
    ("meeting_mode_default", "we agreed on friday", True, Mode.MEETING, False),
    ("command_mode_default", "lights on", True, Mode.COMMAND, False),
    ("assistant_mode_default", "what time is it", True, Mode.ASSISTANT, False),
    ("empty", "", True, Mode.ASSISTANT, False),
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
    name, text, is_final, mode, has_pending = case
    analyzer = LiveSpeechAnalyzer()  # fresh -> stateless/deterministic
    obs = analyzer.observe(text, is_final=is_final)
    dec = analyzer.decide(obs, mode, has_pending_confirmation=has_pending)
    return {
        "name": name,
        "input": {
            "text": text, "is_final": is_final, "mode": mode.value,
            "has_pending_confirmation": has_pending,
        },
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


def test_confirm_deny_only_fire_with_pending_confirmation():
    """The missed-confirm-trigger fix: a bare 'yes'/'no' is a control-plane
    CONFIRM/DENY ONLY when a confirmation is actually pending. With nothing
    pending it must fall through to a normal answerable turn (ASSISTANT), not be
    dropped against an empty queue."""
    a = LiveSpeechAnalyzer()
    for word in ("yes", "no"):
        d = a.decide(a.observe(word, is_final=True), Mode.ASSISTANT, has_pending_confirmation=False)
        assert d.kind == IntentKind.ASSISTANT, (word, d.kind)
    assert a.decide(a.observe("yes", is_final=True), Mode.COMMAND,
                    has_pending_confirmation=True).kind == IntentKind.CONFIRM
    assert a.decide(a.observe("no", is_final=True), Mode.COMMAND,
                    has_pending_confirmation=True).kind == IntentKind.DENY


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
