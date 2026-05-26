"""Golden-dataset regression gate for the router-intelligence layers.

A versioned table of trusted (input -> expected decision) rows, run parametrized.
This is the eval-driven-development pattern: the YAML files in ``router_data/``
are the single source of truth for expected routing/intent behavior, and any
production change that shifts a decision fails CI here. A coverage guard ensures
every enum value stays represented as the layers evolve.
"""
from __future__ import annotations

import pathlib

import pytest
import yaml

from always_on_agent.events import Mode
from always_on_agent.models import IntentKind
from always_on_agent.speech_analyzer import LiveSpeechAnalyzer
from tests.router_relations import route, route_partial
from utils.conversation_router import RouteAction

pytestmark = [pytest.mark.dev, pytest.mark.audio]

_DATA_DIR = pathlib.Path(__file__).parent / "router_data"


def _load(name: str) -> dict:
    with open(_DATA_DIR / name, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


_ROUTER = _load("router_golden.yaml")
_ANALYZER = _load("analyzer_golden.yaml")
ROUTER_CASES = _ROUTER["cases"]
ANALYZER_CASES = _ANALYZER["cases"]
_DEFAULT_CAPS = tuple(_ROUTER.get("default_capabilities", ()))


@pytest.mark.parametrize("case", ROUTER_CASES, ids=lambda c: c["id"])
def test_router_golden(case):
    caps = tuple(case["capabilities"]) if "capabilities" in case else _DEFAULT_CAPS
    if case.get("is_partial"):
        decision = route_partial(case["transcript"], caps=caps)
    else:
        decision = route(case["transcript"], caps=caps)
    assert decision.action.value == case["expect_action"], case["id"]
    if "expect_reason" in case:
        assert decision.reason == case["expect_reason"], case["id"]
    if "expect_capability" in case:
        assert decision.capability == case["expect_capability"], case["id"]
    if "expect_payload" in case:
        assert decision.payload == case["expect_payload"], case["id"]


@pytest.mark.parametrize("case", ANALYZER_CASES, ids=lambda c: c["id"])
def test_analyzer_golden(case):
    analyzer = LiveSpeechAnalyzer()
    obs = analyzer.observe(case["text"], is_final=case.get("is_final", True))
    decision = analyzer.decide(obs, Mode(case.get("mode", "passive")))
    assert decision.kind.value == case["expect_kind"], case["id"]
    if "expect_language" in case:
        assert obs.language == case["expect_language"], case["id"]
    if "expect_target_mode" in case:
        assert decision.target_mode is not None and decision.target_mode.value == case[
            "expect_target_mode"
        ], case["id"]
    if "expect_requires_confirmation" in case:
        assert decision.requires_confirmation == case["expect_requires_confirmation"], case["id"]


# ── Coverage guard: keep the golden tables exhaustive as the enums grow ──────
def test_router_golden_ids_unique():
    ids = [c["id"] for c in ROUTER_CASES]
    assert len(ids) == len(set(ids)), "duplicate router golden ids"


def test_analyzer_golden_ids_unique():
    ids = [c["id"] for c in ANALYZER_CASES]
    assert len(ids) == len(set(ids)), "duplicate analyzer golden ids"


def test_router_golden_covers_every_action():
    covered = {c["expect_action"] for c in ROUTER_CASES}
    expected = {action.value for action in RouteAction}
    missing = expected - covered
    assert not missing, f"RouteAction values with no golden row: {missing}"


def test_analyzer_golden_covers_every_intent_kind():
    covered = {c["expect_kind"] for c in ANALYZER_CASES}
    expected = {kind.value for kind in IntentKind}
    missing = expected - covered
    assert not missing, f"IntentKind values with no golden row: {missing}"
