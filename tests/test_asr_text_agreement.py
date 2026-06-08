"""Unit tests for the SenseVoice 2nd-pass agreement guard (core.asr_text).

Pure text, no audio/model deps -- Tier-0 safe. These pin the length-keyed (NOT
overlap-keyed) discriminator that demotes a short-clip hallucinated 2nd pass back
to the streaming final while preserving the legit long garbled-correction case.
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
    # streaming pass mangled, with near-zero token overlap. Not short -> trusted.
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
    # >2 tokens -> not short -> trust the 2nd pass even without a duration.
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
    # With a stricter short_sec, a 0.5s clip is no longer "short" -> 2nd pass trusted.
    assert (
        agreement_guard("BEING", "I.", segment_sec=0.5, short_sec=0.4) == "I."
    )


def test_short_words_override_token_fallback():
    # Raising short_words makes a 3-token streaming final "short" again -> needs
    # agreement; 'WHO IS IT' vs 'Yo.' disagree -> keep streaming.
    assert agreement_guard("WHO IS IT", "Yo.", short_words=3) == "WHO IS IT"
