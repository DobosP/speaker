from __future__ import annotations

from itertools import permutations

import pytest

import core.asr_verifier as asr_verifier
from always_on_agent import speech_analyzer


def _verify(
    baseline: str,
    streaming: str,
    verifier: str,
    offline: str | None = None,
) -> asr_verifier.AsrConsensusDecision:
    return asr_verifier.verify_asr_consensus(
        baseline_selected=baseline,
        streaming=streaming,
        offline=offline,
        verifier=verifier,
    )


def test_exact_normalized_quorum_chooses_existing_offline_rendering():
    decision = _verify(
        "find my bolt",
        "unrelated words",
        "find my vault",
        "Find My Vault!",
    )

    assert decision.chosen == "Find My Vault!"
    assert decision.outcome is asr_verifier.AsrConsensusOutcome.CONSENSUS
    assert decision.source is asr_verifier.AsrConsensusSource.OFFLINE
    assert decision.support == 2
    assert decision.changed is True


def test_all_three_independent_sources_can_form_quorum():
    decision = _verify(
        "wrong baseline",
        "search my vault",
        "Search my vault!",
        "SEARCH MY VAULT",
    )

    assert decision.chosen == "SEARCH MY VAULT"
    assert decision.source is asr_verifier.AsrConsensusSource.OFFLINE
    assert decision.support == 3


def test_baseline_rendering_wins_when_it_belongs_to_consensus():
    baseline = "Find My Vault, please."
    decision = _verify(
        baseline,
        "find my vault please",
        "find my vault please!",
        "different",
    )

    assert decision.chosen == baseline
    assert decision.source is asr_verifier.AsrConsensusSource.BASELINE
    assert decision.support == 2
    assert decision.changed is False
    assert decision.outcome is asr_verifier.AsrConsensusOutcome.CONSENSUS


def test_renderer_priority_is_baseline_then_offline_then_verifier():
    offline_wins = _verify(
        "other",
        "distractor",
        "find my vault",
        "Find My Vault!",
    )
    verifier_wins = _verify(
        "other",
        "find my vault",
        "Find My Vault!",
        "distractor",
    )

    assert offline_wins.chosen == "Find My Vault!"
    assert offline_wins.source is asr_verifier.AsrConsensusSource.OFFLINE
    assert verifier_wins.chosen == "Find My Vault!"
    assert verifier_wins.source is asr_verifier.AsrConsensusSource.VERIFIER


def test_baseline_is_not_counted_as_an_independent_vote():
    baseline = "find my vault"
    decision = _verify(
        baseline,
        "streaming disagrees",
        "find my vault",
        "offline disagrees too",
    )

    assert decision.chosen == baseline
    assert decision.outcome is asr_verifier.AsrConsensusOutcome.TIE
    assert decision.source is asr_verifier.AsrConsensusSource.BASELINE
    assert decision.support == 1
    assert decision.changed is False


def test_optional_offline_source_can_be_absent():
    decision = _verify(
        "find my bolt",
        "find my vault",
        "Find my vault!",
    )

    assert decision.chosen == "Find my vault!"
    assert decision.source is asr_verifier.AsrConsensusSource.VERIFIER
    assert decision.support == 2
    assert decision.changed is True


@pytest.mark.parametrize(
    ("streaming", "offline", "verifier", "outcome", "support"),
    [
        ("", None, "...", asr_verifier.AsrConsensusOutcome.NO_QUORUM, 0),
        ("one", None, "", asr_verifier.AsrConsensusOutcome.NO_QUORUM, 1),
        ("one", None, "two", asr_verifier.AsrConsensusOutcome.TIE, 1),
        ("one", "two", "three", asr_verifier.AsrConsensusOutcome.TIE, 1),
    ],
)
def test_ties_and_missing_quorum_keep_baseline(
    streaming,
    offline,
    verifier,
    outcome,
    support,
):
    baseline = "production baseline"
    decision = _verify(baseline, streaming, verifier, offline)

    assert decision.chosen == baseline
    assert decision.outcome is outcome
    assert decision.source is asr_verifier.AsrConsensusSource.BASELINE
    assert decision.support == support
    assert decision.changed is False


def test_near_match_is_not_an_exact_normalized_quorum():
    baseline = "production baseline"
    decision = _verify(
        baseline,
        "find my vault",
        "find the vault",
        "find vault",
    )

    assert decision.chosen == baseline
    assert decision.outcome is asr_verifier.AsrConsensusOutcome.TIE
    assert decision.support == 1


def test_distinct_unicode_words_cannot_collapse_into_false_quorum():
    baseline = "production baseline"
    decision = _verify(
        baseline,
        "unrelated",
        "find 公開 vault",
        "find 秘密 vault",
    )

    assert decision.chosen == baseline
    assert decision.outcome is asr_verifier.AsrConsensusOutcome.TIE
    assert decision.support == 1


def test_unicode_and_symbol_content_is_preserved_in_exact_tokens():
    assert asr_verifier._exact_tokens("Café — 秘密 🔒") == (
        "café",
        "秘密",
        "🔒",
    )
    assert asr_verifier._exact_tokens("Paul’s vault") == (
        "paul's",
        "vault",
    )


def test_keyword_argument_permutations_are_deterministic():
    arguments = (
        ("baseline_selected", "find my bolt"),
        ("streaming", "unrelated"),
        ("offline", "Find My Vault!"),
        ("verifier", "find my vault"),
    )
    decisions = {
        asr_verifier.verify_asr_consensus(**dict(order))
        for order in permutations(arguments)
    }

    assert len(decisions) == 1
    decision = decisions.pop()
    assert decision.chosen == "Find My Vault!"
    assert decision.support == 2


def test_acoustic_source_placement_permutations_keep_same_winning_text():
    decisions = []
    for streaming, offline, verifier in set(
        permutations(("agreed words", "agreed words", "distractor"))
    ):
        decisions.append(
            _verify(
                "baseline words",
                streaming,
                verifier,
                offline,
            )
        )

    assert {decision.chosen for decision in decisions} == {"agreed words"}
    assert {decision.support for decision in decisions} == {2}
    assert {decision.outcome for decision in decisions} == {
        asr_verifier.AsrConsensusOutcome.CONSENSUS
    }


def test_decision_repr_never_contains_chosen_transcript():
    private = "SENTINEL_PRIVATE_TRANSCRIPT"
    decision = _verify(
        "baseline",
        private,
        private,
    )

    assert decision.chosen == private
    assert private not in repr(decision)
    assert private not in str(decision)
    assert "support=2" in repr(decision)


def test_invalid_input_fails_closed_without_exposing_details():
    private = "SENTINEL_PRIVATE_BASELINE"
    decision = asr_verifier.verify_asr_consensus(
        baseline_selected=private,
        streaming="streaming",
        offline=object(),
        verifier="verifier",
    )

    assert decision.chosen == private
    assert decision.outcome is asr_verifier.AsrConsensusOutcome.ERROR
    assert decision.source is asr_verifier.AsrConsensusSource.BASELINE
    assert decision.support == 0
    assert decision.changed is False
    assert private not in repr(decision)


def test_internal_failure_keeps_baseline_and_sanitizes_exception(monkeypatch):
    baseline = "SENTINEL_PRIVATE_BASELINE"

    def fail_without_echoing(_text):
        raise RuntimeError("SENTINEL_PRIVATE_FAILURE_DETAIL")

    monkeypatch.setattr(asr_verifier, "_exact_tokens", fail_without_echoing)
    decision = _verify(baseline, "one", "one")

    assert decision.chosen == baseline
    assert decision.outcome is asr_verifier.AsrConsensusOutcome.ERROR
    assert "SENTINEL" not in repr(decision)


def test_consensus_cannot_create_stop_semantics():
    baseline = "please keep talking"
    decision = _verify(
        baseline,
        "Stop!",
        "different",
        "stop",
    )

    assert decision.chosen == baseline
    assert decision.outcome is asr_verifier.AsrConsensusOutcome.CONTROL_GUARD
    assert decision.source is asr_verifier.AsrConsensusSource.BASELINE
    assert decision.support == 2
    assert decision.changed is False


@pytest.mark.parametrize(
    "phrase",
    sorted(speech_analyzer._STOP_PHRASES | {"oprește"}),
)
def test_every_runtime_stop_phrase_is_fenced_from_creation(phrase):
    baseline = "please keep talking"
    decision = _verify(baseline, phrase, phrase)

    assert speech_analyzer.exact_control_class(phrase) == ("stop", "")
    assert decision.chosen == baseline
    assert decision.outcome is asr_verifier.AsrConsensusOutcome.CONTROL_GUARD


@pytest.mark.parametrize(
    "control",
    ["yes", "no", "confirm", "deny", "command mode", "mod pasiv"],
)
def test_consensus_cannot_create_desktop_control_semantics(control):
    baseline = "ordinary conversation"
    decision = _verify(baseline, control, control)

    assert speech_analyzer.exact_control_class(control) is not None
    assert decision.chosen == baseline
    assert decision.outcome is asr_verifier.AsrConsensusOutcome.CONTROL_GUARD


def test_consensus_cannot_change_mode_switch_target():
    baseline = "command mode"
    decision = _verify(baseline, "assistant mode", "assistant mode")

    assert decision.chosen == baseline
    assert decision.outcome is asr_verifier.AsrConsensusOutcome.CONTROL_GUARD


def test_consensus_cannot_remove_stop_semantics():
    baseline = "Cancel that!"
    decision = _verify(
        baseline,
        "keep going",
        "Keep going!",
    )

    assert decision.chosen == baseline
    assert decision.outcome is asr_verifier.AsrConsensusOutcome.CONTROL_GUARD
    assert decision.support == 2
    assert decision.changed is False


def test_consensus_may_change_rendering_when_stop_semantics_are_preserved():
    decision = _verify(
        "stop",
        "Cancel!",
        "cancel",
    )

    assert decision.chosen == "cancel"
    assert decision.source is asr_verifier.AsrConsensusSource.VERIFIER
    assert decision.outcome is asr_verifier.AsrConsensusOutcome.CONSENSUS
    assert decision.support == 2
    assert decision.changed is True


def test_existing_stop_consensus_prefers_unchanged_baseline_rendering():
    baseline = "Stop!"
    decision = _verify(
        baseline,
        "stop",
        "STOP",
    )

    assert decision.chosen == baseline
    assert decision.source is asr_verifier.AsrConsensusSource.BASELINE
    assert decision.outcome is asr_verifier.AsrConsensusOutcome.CONSENSUS
    assert decision.changed is False
