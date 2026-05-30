"""Unit tests for the ADD-ON / continuation classifier (always_on_agent.continuation).

The classifier is the deterministic gate that lets the supervisor tell an add-on
("and also...", "make it shorter", "what about Mars too") apart from a fresh
question. Detection is conservative: clear continuation cues -> CONTINUE,
everything else -> NEW (so a miss degrades to today's behaviour, never worse).
"""
from __future__ import annotations

from always_on_agent.continuation import (
    CONTINUE,
    NEW,
    ContinuationConfig,
    HeuristicContinuationClassifier,
    ScriptedContinuationClassifier,
)


def _classifier(**cfg) -> HeuristicContinuationClassifier:
    return HeuristicContinuationClassifier(ContinuationConfig(enabled=True, **cfg))


def test_leading_markers_are_continuations():
    c = _classifier()
    for text in (
        "and also the forecast",
        "and then book it",
        "oh wait make it shorter",
        "actually use celsius",
        "what about tomorrow",
        "plus the humidity",
        "instead show me Friday",
    ):
        assert c.classify(text, "whats the weather") == CONTINUE, text


def test_trailing_markers_are_continuations():
    c = _classifier()
    assert c.classify("the forecast too", "weather") == CONTINUE
    assert c.classify("in spanish as well", "tell me a joke") == CONTINUE


def test_short_modifier_fragments_are_continuations():
    c = _classifier()
    assert c.classify("in spanish", "tell me a joke") == CONTINUE
    assert c.classify("make it funnier", "tell me a joke") == CONTINUE
    assert c.classify("for tomorrow", "whats the weather") == CONTINUE


def test_fresh_questions_are_new():
    c = _classifier()
    for text in (
        "what is the capital of France",
        "tell me a joke",
        "set a timer for five minutes",
        "who won the game last night",
        "play some music",
    ):
        assert c.classify(text, "whats the weather") == NEW, text


def test_long_utterance_starting_with_modifier_is_not_swallowed():
    # A modifier-lead only counts when the utterance is short; a full sentence
    # that merely opens with a preposition is a fresh turn, not an add-on.
    c = _classifier(addon_max_words=4)
    assert c.classify("for the love of all that is holy explain quantum", "hi") == NEW


def test_long_question_starting_with_weak_conjunction_is_new():
    # The headline default-ON risk: a full fresh question that merely opens with
    # "and"/"but"/"or" must NOT be swallowed into the in-flight turn.
    c = _classifier(addon_max_words=6)
    assert c.classify("and what is the population of the entire country", "hi") == NEW
    assert c.classify("but who actually won the world cup in nineteen ninety", "hi") == NEW
    assert c.classify("or should we look at the quarterly revenue figures instead", "hi") == NEW


def test_weak_conjunction_followed_by_modifier_is_continuation_even_if_longer():
    # "and in spanish ..." -- a weak cue immediately followed by a modifier still
    # reads as an add-on past the length bound.
    c = _classifier(addon_max_words=3)
    assert c.classify("and in spanish please if you can", "tell me a joke") == CONTINUE


def test_long_question_ending_in_too_is_new():
    # Trailing markers are bounded too: a long fresh question that ends in "too"
    # is not an add-on.
    c = _classifier(addon_max_words=4)
    assert c.classify("what is the gross domestic product of france too", "hi") == NEW
    # ...but a short tail still merges.
    assert c.classify("the forecast too", "weather") == CONTINUE


def test_strong_markers_merge_regardless_of_length():
    c = _classifier(addon_max_words=3)
    assert (
        c.classify("and also rank them by price and availability in stock", "compare laptops")
        == CONTINUE
    )
    assert (
        c.classify("what about the long term effects on coastal cities worldwide", "sea level")
        == CONTINUE
    )


def test_romanian_marker_is_continuation():
    c = _classifier()
    # "de asemenea" survives normalize_text (de-diacritic'd convention).
    assert c.classify("de asemenea ce zici de soare", "vremea") == CONTINUE


def test_romanian_diacritic_input_matches_markers():
    # normalize_text now folds diacritics, so real accented RO input matches the
    # de-diacritic'd marker table.
    c = _classifier()
    assert c.classify("în schimb arată altceva", "vremea") == CONTINUE  # strong 'in schimb'
    assert c.classify("și de asemenea soarele", "vremea") == CONTINUE  # strong 'si de asemenea'
    assert c.classify("și ploaia", "vremea") == CONTINUE  # weak 'si', short


def test_config_empty_marker_list_is_honoured():
    # An explicit empty list disables that marker class (not treated as "unset").
    cfg = ContinuationConfig.from_dict({"enabled": True, "weak_markers": []})
    assert cfg.weak_markers == ()
    c = HeuristicContinuationClassifier(cfg)
    assert c.classify("and tomorrow", "weather") == NEW  # weak 'and' disabled
    assert c.classify("and also tomorrow", "weather") == CONTINUE  # strong still on


def test_empty_addon_is_new():
    assert _classifier().classify("", "weather") == NEW
    assert _classifier().classify("   ", "weather") == NEW


def test_custom_markers_override_defaults():
    c = HeuristicContinuationClassifier(
        ContinuationConfig(enabled=True, strong_markers=("furthermore",), weak_markers=())
    )
    assert c.classify("furthermore add detail", "x") == CONTINUE
    # Previously-default markers are no longer recognised once overridden.
    assert c.classify("and also this", "x") == NEW


def test_scripted_fake_records_calls_and_maps():
    fake = ScriptedContinuationClassifier({"and also X": CONTINUE}, default=NEW)
    assert fake.classify("and also X", "prev") == CONTINUE
    assert fake.classify("something else", "prev") == NEW
    assert fake.calls == [("and also X", "prev"), ("something else", "prev")]


def test_config_from_dict_defaults_off():
    assert ContinuationConfig.from_dict(None).enabled is False
    assert ContinuationConfig.from_dict({}).enabled is False
    cfg = ContinuationConfig.from_dict({"enabled": True, "addon_max_words": 5})
    assert cfg.enabled is True
    assert cfg.addon_max_words == 5
