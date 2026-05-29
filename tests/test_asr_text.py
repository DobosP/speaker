"""Unit tests for readable-text restoration (core.asr_text).

Pure functions, no models -- they pin the casing behavior applied to every
streaming partial/final.
"""
from __future__ import annotations

from core.asr_text import looks_all_caps, restore_casing


def test_looks_all_caps():
    assert looks_all_caps("HE MURMURED HIS MURDERING")
    assert not looks_all_caps("he murmured")
    assert not looks_all_caps("Hello there")
    assert not looks_all_caps("")          # no letters -> not all-caps
    assert not looks_all_caps("123 !!!")   # digits/symbols only
    # One stray lowercase token still reads as all-caps (ratio, not isupper).
    assert looks_all_caps("HELLO THERE wORLD GOODBYE NOW", threshold=0.9)


def test_restore_casing_all_caps_input():
    # The exact shape from logs/runs/run-20260528-004726.
    assert restore_casing("HE MURMURED HIS MURDERING") == "He murmured his murdering"


def test_restore_casing_capitalizes_each_sentence():
    assert restore_casing("WHAT TIME IS IT. SET A TIMER") == "What time is it. Set a timer"


def test_restore_casing_uppercases_pronoun_i():
    assert restore_casing("CAN I ASK YOU SOMETHING") == "Can I ask you something"
    # Standalone "i" anywhere, not a letter inside a word.
    assert restore_casing("WHERE AM I") == "Where am I"


def test_restore_casing_preserves_mixed_case_unless_forced():
    # Already nicely-cased text: only the always-safe sentence-start fix applies,
    # proper-noun casing is preserved (not lowercased).
    assert restore_casing("set a timer for Paris") == "Set a timer for Paris"


def test_restore_casing_force_lowercases_first():
    # force=True is what the engine uses after a punctuation model ran.
    assert restore_casing("HELLO THERE", force=True) == "Hello there"


def test_restore_casing_empty_and_blank():
    assert restore_casing("") == ""
    assert restore_casing("   ") == "   "
