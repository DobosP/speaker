"""Unit tests for the SenseVoice 2nd-pass agreement guard (core.asr_text).

Pure text, no audio/model deps -- Tier-0 safe. These pin the fail-closed
agreement guard that demotes hallucinated 2nd-pass text back to the streaming
final while preserving punctuation/casing cleanup and legit long corrections.
The hallucination pairs come straight from ``logs/runs/run-20260608-181250``
(open-speaker echo producing plausible-but-invented finals).
"""
from __future__ import annotations

from core.asr_text import agreement_guard


# --- short-clip hallucinations rejected (keep the streaming final) -----------

def test_short_hallucination_being_to_i():
    # The headline cascade trigger: 'BEING' (raw zipformer) <- echo, SenseVoice
    # invents 'I.'. Short clip, no shared content token -> reject.
    assert agreement_guard("BEING", "I.", segment_sec=0.4) == "BEING"


def test_short_hallucination_thirteen_to_whatt():
    assert agreement_guard("THIRTEEN", "Whatt.", segment_sec=0.4) == "THIRTEEN"


def test_short_hallucination_come_to_ten_to_one():
    assert agreement_guard("COME TO TEN", "1.", segment_sec=0.5) == "COME TO TEN"


def test_short_hallucination_ah_to_okay():
    assert agreement_guard("AH", "Okay.", segment_sec=0.3) == "AH"


def test_short_hallucination_w_story_old_to_o():
    # '1.'/'O.' style: the 2nd pass collapses to an empty content-token set, so it
    # can never agree on a short clip.
    assert agreement_guard("W STORY OLD", "O.", segment_sec=0.5) == "W STORY OLD"


# --- legit long garbled-correction accepted (take the 2nd pass) --------------

def test_long_low_overlap_correction_accepted():
    # The legit case the 2nd pass exists for: a real, longer utterance the
    # streaming pass mangled, with near-zero token overlap but high phrase
    # similarity. A long, clear improvement is accepted.
    assert (
        agreement_guard("Ario der", "are you there", segment_sec=2.0)
        == "are you there"
    )


def test_long_correction_full_sentence_accepted():
    assert (
        agreement_guard(
            "THE LOW IS THIS CORDOOR KING",
            "Hello, is this code working?",
            segment_sec=2.5,
        )
        == "Hello, is this code working?"
    )


def test_long_unrelated_second_pass_rejected():
    assert (
        agreement_guard("please turn on the kitchen lights", "The weather is lovely.", segment_sec=2.2)
        == "please turn on the kitchen lights"
    )


# --- LONG clip but the 2nd pass collapses to no real words -> hallucination ----


def test_long_clip_single_letter_hallucination_rejected():
    # The open-speaker live failure (run-20260617-225622): a >1.2s clip the
    # streaming pass heard as real words, the 2nd pass invented into a bare letter.
    # No content tokens in the 2nd pass + real words in the streaming -> keep the
    # streaming final, even though the clip is "not short".
    assert agreement_guard("MANY OWN", "H.", segment_sec=1.8) == "MANY OWN"
    assert agreement_guard("IT IS", "I.", segment_sec=1.5) == "IT IS"


def test_long_clip_real_correction_still_accepted_over_the_new_rule():
    # The new no-content-token rule must NOT block a legit correction that HAS
    # real words (it only bites when the 2nd pass collapses to punctuation).
    assert (
        agreement_guard("Ario der", "are you there", segment_sec=2.0)
        == "are you there"
    )


# --- short clip but the 2nd pass agrees -> accept the (cleaner) 2nd pass ------

def test_short_agreeing_pair_accepted():
    # 'stop' vs 'Stop.' -> {'stop'} & {'stop'} non-empty -> take the punctuated 2nd pass.
    assert agreement_guard("stop", "Stop.", segment_sec=0.4) == "Stop."


def test_short_agreeing_pair_case_insensitive():
    assert agreement_guard("YES", "Yes.", segment_sec=0.4) == "Yes."


# --- token-count fallback when segment_sec is None ---------------------------

def test_token_fallback_short_rejected():
    # No duration -> fall back to streaming-final token count (<=2 == short).
    assert agreement_guard("BEING", "I.") == "BEING"


def test_token_fallback_long_accepted():
    # >2 tokens -> not short, but still accepted because the normalized words agree.
    assert (
        agreement_guard(
            "please set a timer for ten minutes",
            "Please set a timer for 10 minutes.",
        )
        == "Please set a timer for 10 minutes."
    )


def test_token_fallback_two_words_is_short():
    # Exactly short_words (2) counts as short -> agreement required; here they
    # disagree on every >=2-char content token -> keep streaming.
    assert agreement_guard("OH NO", "Yep.") == "OH NO"


# --- empty / degenerate inputs -----------------------------------------------

def test_empty_second_pass_returns_streaming():
    assert agreement_guard("hello", "") == "hello"


def test_blank_second_pass_returns_streaming():
    assert agreement_guard("hello", "   ") == "hello"


def test_empty_streaming_returns_second_pass():
    assert agreement_guard("", "Hello.") == "Hello."


def test_blank_streaming_returns_second_pass():
    assert agreement_guard("   ", "Hello.") == "Hello."


def test_both_empty_returns_streaming():
    # Both blank: second_pass empty branch wins first -> returns streaming_final ("").
    assert agreement_guard("", "") == ""


# --- single-char trap: 'I.'/'1.'/'O.'/'i' collapse to an empty content set ----

def test_single_char_artifacts_have_empty_content_tokens():
    from core.asr_text import _content_tokens

    for artifact in ("I.", "1.", "O.", "i", "A.", "."):
        assert _content_tokens(artifact) == set(), artifact


def test_single_char_2nd_pass_never_agrees_on_short_clip():
    # Even if the streaming final shares the *letter*, the len>=2 filter means a
    # single-char 2nd pass cannot agree -> streaming kept.
    assert agreement_guard("I AM", "I.", segment_sec=0.4) == "I AM"


# --- short_sec / short_words are tunable on the signature ---------------------

def test_short_sec_override_makes_clip_long():
    # With a stricter short_sec, a 0.5s clip is no longer "short" -> a 2nd pass
    # with REAL WORDS is trusted. (A bare-letter 2nd pass like "I." is now always
    # rejected by the no-content-token rule, regardless of length, so this uses a
    # content-bearing correction to isolate the length gate.)
    assert (
        agreement_guard("Ario der", "are you there", segment_sec=0.5, short_sec=0.4)
        == "are you there"
    )


def test_short_words_override_token_fallback():
    # Raising short_words makes a 3-token streaming final "short" again -> needs
    # agreement; 'WHO IS IT' vs 'Yo.' disagree -> keep streaming.
    assert agreement_guard("WHO IS IT", "Yo.", short_words=3) == "WHO IS IT"


# --- FIX-3: shared-token fabrications rejected, grounded corrections kept ------
# Real SenseVoice failures where the 2nd pass shares a token or two with the raw
# and rides word_agreement / the char-similarity hatch into the brain while
# inventing the rest. The grounding gates must separate these from genuine
# corrections. Long clips so the length gate is out of the way -- the point is
# the CONTENT test, not duration.


def test_fabrication_like_a_question_kept_raw():
    # raw 'LIKE A QUESTION' -> a 12-word invention sharing only 'question'.
    assert (
        agreement_guard(
            "LIKE A QUESTION",
            "And did you I could pressure in if you found this question",
            segment_sec=2.5,
        )
        == "LIKE A QUESTION"
    )


def test_fabrication_long_story_to_ceiling_kept_raw():
    # raw 'TEND ME A LONG STORY ABOUT HER' -> 'A long story about the ceiling':
    # shares 3 of 6 content tokens (exactly half) and fabricates 'ceiling'.
    assert (
        agreement_guard(
            "TEND ME A LONG STORY ABOUT HER",
            "A long story about the ceiling",
            segment_sec=2.5,
        )
        == "TEND ME A LONG STORY ABOUT HER"
    )


def test_legit_itn_cleanup_accepted():
    # Pure ITN/punctuation cleanup: every content token preserved, 'ten' -> '10'.
    assert (
        agreement_guard(
            "please set a timer for ten minutes",
            "Please set a timer for 10 minutes.",
            segment_sec=2.5,
        )
        == "Please set a timer for 10 minutes."
    )


def test_legit_garble_repair_accepted():
    # Zero token overlap phonetic repair -> the char-similarity hatch (behind
    # _low_overlap) still lands it.
    assert (
        agreement_guard("Ario der", "are you there", segment_sec=2.0)
        == "are you there"
    )


def test_legit_full_sentence_repair_accepted():
    # 2 of 6 content tokens shared -> word_agreement fires but grounding rejects
    # it; the low-overlap char-similarity hatch then accepts the real repair.
    assert (
        agreement_guard(
            "THE LOW IS THIS CORDOOR KING",
            "Hello is this code working",
            segment_sec=2.5,
        )
        == "Hello is this code working"
    )


def test_unrelated_second_pass_kept_raw():
    # No shared content of substance and low phrase similarity -> keep raw.
    assert (
        agreement_guard(
            "please turn on the kitchen lights",
            "The weather is lovely.",
            segment_sec=2.2,
        )
        == "please turn on the kitchen lights"
    )


# --- FIX-3: the pure grounding predicates in isolation ------------------------


def test_grounded_rewrite_exact_normalized_match():
    from core.asr_text import _grounded_rewrite

    # Punctuation/casing only -> exact normalized-word match.
    assert _grounded_rewrite("stop", "Stop.")
    assert _grounded_rewrite("please set a timer for ten minutes",
                             "Please set a timer for 10 minutes.")


def test_grounded_rewrite_rejects_half_or_fewer_kept():
    from core.asr_text import _grounded_rewrite

    # 'LIKE A QUESTION': 1 of 2 content tokens kept (2*1 not > 2) -> not grounded.
    assert not _grounded_rewrite(
        "LIKE A QUESTION",
        "And did you I could pressure in if you found this question",
    )
    # 'TEND ME A LONG STORY ABOUT HER': 3 of 6 kept (2*3 not > 6) -> not grounded.
    assert not _grounded_rewrite(
        "TEND ME A LONG STORY ABOUT HER", "A long story about the ceiling"
    )


def test_grounded_rewrite_rejects_balloon_even_if_majority_kept():
    from core.asr_text import _grounded_rewrite

    # Keeps both content tokens but more than doubles the word count -> ballooned.
    assert not _grounded_rewrite(
        "help now", "help me right now please with everything you can"
    )


def test_low_overlap_true_only_for_garbled_raw():
    from core.asr_text import _low_overlap

    # Zero/near-zero shared tokens -> the hatch is trustworthy.
    assert _low_overlap("Ario der", "are you there")
    assert _low_overlap("THE LOW IS THIS CORDOOR KING", "Hello is this code working")
    # Half the raw's tokens reproduced -> NOT low overlap, hatch blocked.
    assert not _low_overlap(
        "TEND ME A LONG STORY ABOUT HER", "A long story about the ceiling"
    )
