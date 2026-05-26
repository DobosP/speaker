"""Metamorphic relations for the router-intelligence layers.

Metamorphic testing needs no oracle: instead of asserting an exact output for
an input, it asserts a *relation* between the outputs of related inputs
(e.g. paraphrasing the input must not change the routing decision). This is the
state-of-the-art way to test non-deterministic / hard-to-oracle behavior, and
it fits these pure decision functions perfectly.

Relations are split into:
* ``R*`` -- ConversationRouter invariants (passing)
* ``A*`` -- LiveSpeechAnalyzer invariants (passing)
* ``X*`` -- known-gap relations encoded as ``xfail(strict=True)``; an XPASS means
  the underlying bug was fixed and the relation should be promoted to a hard
  assertion. See the plan's "latent bugs" section (B1/B2/B3).
"""
from __future__ import annotations

import pytest

from always_on_agent.events import Mode
from always_on_agent.models import IntentKind
from always_on_agent.speech_analyzer import LiveSpeechAnalyzer
from tests.router_relations import (
    analyze,
    case_variants,
    observe,
    punctuation_variants,
    route,
    route_partial,
    whitespace_variants,
    with_trailing_filler,
)
from utils.conversation_router import RouteAction

pytestmark = [pytest.mark.dev, pytest.mark.audio]


# Representative seeds spanning every router branch.
_ROUTER_SEEDS = [
    "stop",
    "stop talking",
    "cancel",
    "quit",
    "goodbye",
    "what time is it",
    "tell me a joke",
]


# ── ConversationRouter relations ────────────────────────────────────────────
@pytest.mark.parametrize("seed", _ROUTER_SEEDS)
def test_R1_case_invariance(seed):
    actions = {route(v).action for v in case_variants(seed)}
    assert len(actions) == 1, f"{seed!r} routed differently across letter cases: {actions}"


@pytest.mark.parametrize("seed", _ROUTER_SEEDS)
def test_R2_punctuation_invariance(seed):
    actions = {route(v).action for v in punctuation_variants(seed)}
    assert len(actions) == 1, f"{seed!r} routed differently across punctuation: {actions}"


@pytest.mark.parametrize("seed", _ROUTER_SEEDS)
def test_R3_whitespace_invariance(seed):
    actions = {route(v).action for v in whitespace_variants(seed)}
    assert len(actions) == 1, f"{seed!r} routed differently across whitespace: {actions}"


def test_R4_apostrophe_equivalence():
    # normalize_transcript strips apostrophes, so these must be identical.
    assert route("that's enough").action == route("thats enough").action == RouteAction.STOP_OUTPUT


@pytest.mark.parametrize(
    "noisy,clean",
    [
        ("stop stop talking", "stop talking"),
        ("what what time is it", "what time is it"),
    ],
)
def test_R5_repeated_word_idempotence(noisy, clean):
    # normalize_transcript collapses consecutive duplicate words.
    assert route(noisy).action == route(clean).action


@pytest.mark.parametrize(
    "control,expected",
    [
        ("stop", RouteAction.STOP_OUTPUT),
        ("quit", RouteAction.SHUTDOWN),
    ],
)
def test_R6_trailing_filler_preserves_control(control, expected):
    for variant in with_trailing_filler(control):
        assert route(variant).action == expected, f"{variant!r} lost control routing"


def test_R7_capability_stable_under_noise():
    for variant in ["what time is it", "What time is it?", "what   time is  it", "WHAT TIME IS IT"]:
        decision = route(variant)
        assert decision.action == RouteAction.CAPABILITY
        assert decision.capability == "system.time"


@pytest.mark.parametrize(
    "text", ["stop", "quit", "cancel", "goodbye", "what time is it", "tell me a joke"]
)
def test_R8_partial_is_subset_of_final(text):
    # A partial transcript may only mirror the final control decision or IGNORE;
    # it must never escalate a non-control utterance to LLM/CAPABILITY.
    final_action = route(text).action
    partial_action = route_partial(text).action
    assert partial_action in {final_action, RouteAction.IGNORE}


@pytest.mark.parametrize(
    "near_miss", ["started", "stopping", "exited", "quitting", "cancellation", "silently"]
)
def test_R9_near_miss_stability(near_miss):
    # Morphological neighbours of control words must NOT trigger control routing.
    assert route(near_miss).action == RouteAction.LLM


# ── LiveSpeechAnalyzer relations ────────────────────────────────────────────
@pytest.mark.parametrize(
    "english,romanian,kind",
    [
        ("stop", "opreste", IntentKind.STOP),
        ("cancel", "anuleaza", IntentKind.STOP),
        ("yes", "da", IntentKind.CONFIRM),
        ("no", "nu", IntentKind.DENY),
    ],
)
def test_A1_bilingual_control_equivalence(english, romanian, kind):
    assert analyze(english).kind == analyze(romanian).kind == kind


@pytest.mark.parametrize(
    "english,romanian,kind",
    [
        ("search local stt", "cauta moonshine local", IntentKind.SEARCH),
        ("research voice pipelines", "cerceteaza voice pipelines", IntentKind.RESEARCH),
        ("dictate this note", "scrie aceasta nota", IntentKind.DICTATION),
    ],
)
def test_A2_bilingual_prefix_equivalence(english, romanian, kind):
    assert analyze(english).kind == analyze(romanian).kind == kind


@pytest.mark.parametrize(
    "english,romanian,target",
    [
        ("search mode", "mod cautare", Mode.SEARCH),
        ("research mode", "mod cercetare", Mode.RESEARCH),
        ("assistant mode", "mod asistent", Mode.ASSISTANT),
    ],
)
def test_A3_bilingual_mode_equivalence(english, romanian, target):
    en = analyze(english)
    ro = analyze(romanian)
    assert en.kind == ro.kind == IntentKind.MODE_SWITCH
    assert en.target_mode == ro.target_mode == target


def test_A4_activation_score_monotonicity():
    # Adding wake/intent terms must never lower the activation score.
    ladder = [
        "help me",
        "please help me",
        "assistant please help me",
        "assistant please help me search",
    ]
    scores = [observe(t).activation_score for t in ladder]
    assert scores == sorted(scores), f"activation score not monotonic: {scores}"


@pytest.mark.parametrize(
    "text,language",
    [
        ("search assistant mode", "en"),
        ("cauta asistent mod", "ro"),
    ],
)
def test_A5_language_detection_consistency(text, language):
    assert observe(text).language == language


def test_A6_stability_self_similarity():
    # Stability compares each partial to the previous one, so it is stateful;
    # one analyzer instance is reused across the three observations.
    analyzer_pair = LiveSpeechAnalyzer()
    analyzer_pair.observe("turn on the lights", is_final=False)
    same = analyzer_pair.observe("turn on the lights", is_final=False)
    assert same.stability == pytest.approx(1.0)
    different = analyzer_pair.observe("completely unrelated phrase", is_final=False)
    assert different.stability < 1.0


# ── Known-gap relations (xfail strict = fix-trackers) ───────────────────────
@pytest.mark.xfail(strict=True, reason="B1: analyzer matches control phrases by exact string only")
def test_X1_analyzer_trailing_filler_preserves_control():
    # Desired: "stop please" should still be a STOP. Today it falls through to IGNORE.
    assert analyze("stop please").kind == IntentKind.STOP


@pytest.mark.xfail(strict=True, reason="B2: router prefix-match misses leading filler")
def test_X2_router_leading_filler_preserves_control():
    # Desired: "please stop" should be STOP_OUTPUT. Today it routes to the LLM.
    assert route("please stop").action == RouteAction.STOP_OUTPUT


@pytest.mark.discovery
@pytest.mark.xfail(strict=True, reason="B3: router has no bilingual phrase support")
def test_X3_router_bilingual_stop():
    # Desired: Romanian "opreste" should stop output (the analyzer already does).
    assert route("opreste").action == RouteAction.STOP_OUTPUT
