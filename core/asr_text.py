"""Readable-text restoration for streaming ASR output.

The streaming zipformer transducer emits unpunctuated text and, with many
English models, ALL-CAPS BPE tokens -- e.g. ``"HE MURMURED HIS MURDERING"`` in
``logs/runs/run-20260528-004726``. That is hard to read in the transcript and
mildly degrades the LLM prompt. These are pure, dependency-free functions that
restore conventional casing with no model, so they run cheaply on every partial
and final on the hot path. An optional sherpa-onnx punctuation model (wired in
the engine, off by default) adds real punctuation to finals on top of this.
"""
from __future__ import annotations

import re

# Sentence terminators after which the next letter is capitalized.
_TERMINATORS = ".!?"
# Standalone lowercase pronoun "i" (incl. contractions like "i'm", "i'll");
# always uppercased. Case-sensitive so it never touches an already-correct "I".
_I_WORD = re.compile(r"\bi\b")


def looks_all_caps(text: str, *, threshold: float = 0.9) -> bool:
    """True if (almost) every letter is uppercase -- the all-caps ASR shape.

    Uses a ratio rather than ``str.isupper`` so a stray digit/symbol or a single
    lowercased token doesn't flip the verdict on otherwise all-caps output."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    uppers = sum(1 for c in letters if c.isupper())
    return uppers / len(letters) >= threshold


def restore_casing(text: str, *, force: bool = False) -> str:
    """Restore sentence casing on unpunctuated / all-caps ASR text.

    Lowercases first **only** when the text looks all-caps (or ``force`` is set),
    then capitalizes the first letter of each sentence and the standalone pronoun
    "I". Genuinely mixed-case input (a model that already cases proper nouns) is
    left as-is apart from the always-safe sentence-start and "I" fixes -- so we
    never destroy existing casing."""
    if not text or not text.strip():
        return text
    if force or looks_all_caps(text):
        text = text.lower()
    # Capitalize the first alphabetic character of the string and of each
    # sentence (the first letter after a .!? terminator).
    out = list(text)
    capitalize_next = True
    for i, ch in enumerate(out):
        if ch.isalpha():
            if capitalize_next:
                out[i] = ch.upper()
                capitalize_next = False
        elif ch in _TERMINATORS:
            capitalize_next = True
    cased = "".join(out)
    # The pronoun "I" (and its contractions) is always uppercase, anywhere.
    return _I_WORD.sub("I", cased)
