"""ConversationRouter tested at scale over real NLU corpora (CLINC150 + MASSIVE).

By default these run over the committed sample slices (hundreds of real
utterances); set ``SPEAKER_DATASET_DOWNLOAD=1`` to expand to the full datasets
(thousands). The per-record tests assert universal invariants (valid output,
determinism, normalization contract, case/punctuation invariance); the aggregate
tests measure control false-trigger rate and capability recall and surface the
offending utterances.
"""
from __future__ import annotations

import pytest

from tests.datasets.loaders import load_clinc150, router_corpus
from tests.router_relations import route, route_partial
from utils.conversation_router import RouteAction, normalize_transcript

pytestmark = [pytest.mark.dev, pytest.mark.audio]

_CORPUS = router_corpus()
_IDS = [f"{(r.source or 'rec').split()[0][:6].lower()}-{i}" for i, r in enumerate(_CORPUS)]

# CLINC intents that may legitimately map to control/volume, excluded from the
# false-trigger precision metric (we don't want to penalize a genuine "cancel").
_CONTROL_LIKE_INTENTS = {"cancel", "no", "yes", "maybe", "change_volume", "oos"}


@pytest.mark.parametrize("rec", _CORPUS, ids=_IDS)
def test_router_returns_valid_action(rec):
    decision = route(rec.text)
    assert isinstance(decision.action, RouteAction)
    assert isinstance(decision.normalized_text, str)


@pytest.mark.parametrize("rec", _CORPUS, ids=_IDS)
def test_router_is_deterministic_and_partial_safe(rec):
    first = route(rec.text).action
    assert route(rec.text).action == first
    # A partial may only mirror the final decision or be IGNORE; never escalate.
    assert route_partial(rec.text).action in {first, RouteAction.IGNORE}


@pytest.mark.parametrize("rec", _CORPUS, ids=_IDS)
def test_normalize_contract_holds(rec):
    norm = normalize_transcript(rec.text)
    assert normalize_transcript(norm) == norm, "normalization not idempotent"
    assert all(c.isalnum() or c == " " for c in norm), f"stray chars in {norm!r}"
    assert "  " not in norm, "whitespace not collapsed"


@pytest.mark.parametrize("rec", _CORPUS, ids=_IDS)
def test_router_case_and_punctuation_invariant(rec):
    base = route(rec.text).action
    assert route(rec.text.lower()).action == base
    assert route(f"{rec.text}.").action == base
    assert route(f"{rec.text}?").action == base


def test_control_false_trigger_rate():
    """General/OOS utterances must rarely be mistaken for stop/shutdown control."""
    triggers = []
    considered = 0
    for rec in _CORPUS:
        if rec.intent in _CONTROL_LIKE_INTENTS:
            continue
        considered += 1
        action = route(rec.text).action
        if action in {RouteAction.STOP_OUTPUT, RouteAction.SHUTDOWN}:
            triggers.append((rec.text, rec.intent, action.value))
    rate = len(triggers) / max(considered, 1)
    print(f"\ncontrol false-trigger rate: {rate:.4f} ({len(triggers)}/{considered})")
    for utt, intent, action in triggers[:25]:
        print(f"  [{action}] ({intent}) {utt!r}")
    assert rate < 0.03, f"control layer over-fires on {rate:.2%} of general utterances"


def test_time_capability_recall():
    """How many real 'time' queries the (deliberately narrow) matcher catches."""
    time_recs = [r for r in load_clinc150() if r.intent == "time"]
    if not time_recs:
        pytest.skip("no time-intent utterances available")
    hits = sum(1 for r in time_recs if route(r.text).capability == "system.time")
    recall = hits / len(time_recs)
    print(f"\ntime->capability recall: {recall:.2f} ({hits}/{len(time_recs)})")
    # Soft floor: the matcher is intentionally narrow; this documents the gap.
    assert recall >= 0.15


def test_corpus_is_nontrivial():
    assert len(_CORPUS) >= 100, f"router corpus unexpectedly small: {len(_CORPUS)}"
