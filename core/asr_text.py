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


# Content-token extractor for the agreement guard: lowercase alphanumeric runs,
# keeping only tokens of length >= 2 so single-letter artifacts ("I", "1", "O")
# collapse to nothing and can never spuriously "agree" with a real word.
_CONTENT_TOKEN = re.compile(r"[a-z0-9]+")


def _content_tokens(text: str) -> set[str]:
    """Normalized content tokens (lowercased, len>=2 alphanumeric runs).

    A SenseVoice hallucination on a short echo clip ("I.", "1.", "O.") yields an
    empty set under this filter, so it can never share a token with the streaming
    final -- which is exactly how the short-clip agreement test rejects it."""
    return {t for t in _CONTENT_TOKEN.findall(text.lower()) if len(t) >= 2}


def agreement_guard(
    streaming_final: str,
    second_pass: str,
    *,
    segment_sec: float | None = None,
    short_sec: float = 1.2,
    short_words: int = 2,
) -> str:
    """Pick the final transcript, guarding against short-clip 2nd-pass hallucination.

    The offline 2nd-pass recognizer (SenseVoice) exists to *correct* the streaming
    final on genuinely garbled but real speech -- e.g. ``"Ario der" -> "are you
    there"`` (a real, longer utterance the streaming pass mangled, with near-zero
    token overlap). But on a SHORT clip that is really just open-speaker echo /
    ambient noise it instead *invents* a plausible sentence from nothing -- the
    pairs logged in ``logs/runs/run-20260608-181250``: ``"BEING" -> "I."``,
    ``"THIRTEEN" -> "Whatt."``, ``"THE LOW IS THIS CORDOOR KING" -> "Hello, is
    this code working?"``. Those echo-borne finals then reach the brain and
    cascade into a self-interrupt runaway.

    The discriminator keys on clip *length* (a text/duration risk, never energy),
    not on overlap, so the legit long correction is preserved:

    1. If either side is empty/blank, return the non-empty side.
    2. Decide whether this is a short, hallucination-prone clip: by ``segment_sec``
       when known (``< short_sec``), else by the streaming final's token count
       (``<= short_words``). KNOWN LIMITATION: a real but garbled correction whose
       clip lands in ``[asr_final_min_sec, short_sec)`` (the 2nd pass runs but the
       clip is still "short") with zero >=2-char token overlap would be demoted to
       the streaming final. Dormant while the 2nd pass is off; if re-enabled, shrink
       the band (raise ``asr_final_min_sec`` toward ``short_sec``) or let a
       high-overlap 2nd pass through even when short.
    3. If NOT short, trust the 2nd pass unconditionally (keeps ``"Ario der" ->
       "are you there"`` even though the tokens don't overlap).
    4. If short, accept the 2nd pass only when it AGREES with the streaming final
       -- i.e. they share at least one normalized content token (lowercased,
       len>=2 alphanumeric; single-letter artifacts like ``"I."`` collapse to the
       empty set). On disagreement, return the streaming final raw: it is left
       un-postprocessed because the L1/L2 energy/refractory gates then drop the
       echo final anyway.

    Pure, dependency-free, no I/O -- mirrors ``restore_casing`` so it runs on the
    final path with no model or audio import."""
    if not second_pass or not second_pass.strip():
        return streaming_final
    if not streaming_final or not streaming_final.strip():
        return second_pass
    if segment_sec is not None:
        short = segment_sec < short_sec
    else:
        short = len(streaming_final.split()) <= short_words
    if not short:
        return second_pass
    if _content_tokens(streaming_final) & _content_tokens(second_pass):
        return second_pass
    return streaming_final
