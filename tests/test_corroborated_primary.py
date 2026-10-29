"""Exact per-edit primary support, without model imports or reference hints."""
from __future__ import annotations

from itertools import product

import pytest

from core.asr_verifier import corroborated_primary, _unique_edits


STREAM = "Tell me the whether in London today"
PRIMARY = "Tell me the weather in London today"
WITNESS = "Tell me the weather in London tomorrow"


def choose(streaming=STREAM, primary=PRIMARY, verifier=WITNESS):
    return corroborated_primary(streaming=streaming, primary=primary, verifier=verifier)


def test_exact_changed_span_supported_despite_separate_verifier_error():
    assert choose() == PRIMARY


def test_return_original_rendering_without_stitching_or_casing_rewrite():
    primary = "Tell me the Weather, in London today!"
    assert choose(primary=primary) is primary


def test_exact_inserted_span_with_same_streaming_index_is_supported():
    assert choose("Tell me weather in London today", PRIMARY, WITNESS) == PRIMARY


def test_unicode_and_symbols_remain_exact_supported_tokens():
    assert choose("Explain α and δ today", "Explain β and δ today", "Explain β and δ tomorrow") == "Explain β and δ today"
    assert choose("Read x + y today", "Read x $ y today", "Read x $ y tomorrow") == "Read x $ y today"


def test_negation_insertion_needs_the_same_independent_exact_edit():
    assert choose("Tell me to call today", "Tell me not to call today", "Tell me not to call tomorrow") == "Tell me not to call today"


def test_disjoint_verifier_deletion_is_not_a_primary_deletion_vote():
    assert choose(verifier="Tell me the weather in London") == PRIMARY


@pytest.mark.parametrize(("streaming", "primary", "verifier"), [
    ("weather", "the weather", "the weather tomorrow"),
    ("weather today", "the weather today", "the weather today please"),
    (STREAM, PRIMARY, "Tell me the climate in London tomorrow"),
    (STREAM, PRIMARY, "Tell me the weather around London today"),
    ("Tell me weather today", "Tell me local weather today", "Tell me weather local today"),
    ("read red and blue today", "read blue and red today", "read blue and red tomorrow"),
    ("read red and blue today", "read blue and red today please", "read blue and red tomorrow please"),
    ("read red and blue today", "read blue and green today", "read blue and green tomorrow"),
    ("Tell me the the whether today", "Tell me the the weather today", "Tell me the weather tomorrow"),
    ("a b a", "a a", "a a please"),
    ("Tell me not to do that", "Tell me to do that", "Tell me to do this"),
    ("Tell me not to call today", "Tell me call today", "Tell me call tomorrow"),
    ("Set a timer for twenty minutes please", "Set a timer for ten minutes please", "Set a timer for 10 minutes now"),
    ("Read twenty pages please", "Read twenty five pages please", "Read twenty pages five now"),
    ("Explain α and δ today", "Explain β and δ today", "Explain gamma and δ tomorrow"),
    ("Read x + y today", "Read x $ y today", "Read x + y tomorrow"),
])
def test_unsupported_ambiguous_deleted_reordered_changes_abstain(streaming, primary, verifier):
    assert choose(streaming, primary, verifier) is None


def test_adjacent_broader_verifier_edit_is_not_support_for_smaller_primary_edit():
    assert choose("read old words today", "read new words today", "read new phrases tomorrow") is None


def test_verifier_ambiguity_anywhere_rejects_even_separate_supported_edit():
    assert choose("read old words then a a", "read new words then a a", "read new words then a") is None


@pytest.mark.parametrize("control", [
    "stop", "stop talking", "cancel", "cancel that", "quiet", "be quiet",
    "yes", "no", "confirm", "do it", "assistant mode", "research mode", "dictation mode",
])
@pytest.mark.parametrize("position", range(3))
def test_any_original_control_class_stays_outside_dialogue_fallback(control, position):
    texts = [STREAM, PRIMARY, WITNESS]
    texts[position] = control
    assert choose(*texts) is None


@pytest.mark.parametrize("value", ["", " ", None, 7, object(), "x " * 129, "x" * 8193])
@pytest.mark.parametrize("position", range(3))
def test_invalid_empty_or_bounded_input_abstains_without_cropping(value, position):
    texts = [STREAM, PRIMARY, WITNESS]
    texts[position] = value
    assert choose(*texts) is None


def test_helper_exception_is_silent_and_detail_free(capsys, caplog):
    class Hostile(str):
        def __len__(self):
            raise RuntimeError("private source must not escape")
    assert choose(primary=Hostile(PRIMARY)) is None
    assert "private source" not in caplog.text
    assert capsys.readouterr() == ("", "")


def test_every_optimal_path_is_counted_not_only_local_chosen_backtrace():
    assert _unique_edits(("a", "b", "a"), ("a", "a"), allow_deletions=True) == ((1, 2, ()),)
    assert _unique_edits(("a", "a", "b"), ("a", "b"), allow_deletions=True) is None


def test_exhaustive_short_alignments_never_accept_primary_deletion():
    for a, b in product(product(("a", "b"), repeat=3), product(("a", "b"), repeat=2)):
        assert _unique_edits(a, b) is None
