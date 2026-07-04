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

from difflib import SequenceMatcher
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


def _normalized_words(text: str) -> list[str]:
    """Lowercased alphanumeric words for punctuation/casing-insensitive checks."""
    return _CONTENT_TOKEN.findall(text.lower())


def _compact_words(words: list[str]) -> str:
    return "".join(words)


def word_agreement(streaming_final: str, second_pass: str) -> bool:
    """Whether two transcripts agree after punctuation/casing normalization.

    This is intentionally lexical and conservative: exact normalized words always
    agree (``"stop"`` vs ``"Stop."``), and longer phrases may agree by sharing
    enough >=2-character content words. Single-letter artifacts are ignored for
    shared-token agreement so ``"I."`` cannot validate an invented final.
    """
    st_words = _normalized_words(streaming_final)
    sp_words = _normalized_words(second_pass)
    if st_words and st_words == sp_words:
        return True

    st_tokens = {w for w in st_words if len(w) >= 2}
    sp_tokens = {w for w in sp_words if len(w) >= 2}
    if not st_tokens or not sp_tokens:
        return False

    shared = len(st_tokens & sp_tokens)
    shorter = min(len(st_tokens), len(sp_tokens))
    required = 1 if shorter <= 2 else 2
    return shared >= required


def _grounded_rewrite(raw: str, rewrite: str) -> bool:
    """Whether a lexical (``word_agreement``) 2nd-pass acceptance is actually
    *grounded* in the raw, not a fabrication that merely shares a token or two.

    Live, SenseVoice rewrote the short garbled raw ``"LIKE A QUESTION"`` into
    ``"And did you I could pressure in if you found this question"`` and ``"TEND
    ME A LONG STORY ABOUT HER"`` into ``"A long story about the ceiling"``: each
    slips past ``word_agreement`` on one/two shared content tokens while
    inventing the rest. A genuine correction either reproduces the raw words
    exactly (pure punctuation/casing/ITN cleanup) or KEEPS more than half of the
    raw's content tokens without ballooning in length. Dimensionless ratios of
    the raw itself -- no machine constant."""
    raw_words = _normalized_words(raw)
    rewrite_words = _normalized_words(rewrite)
    if raw_words and raw_words == rewrite_words:
        return True
    raw_tokens = _content_tokens(raw)
    shared = len(raw_tokens & _content_tokens(rewrite))
    preserves_majority = 2 * shared > len(raw_tokens)
    not_ballooned = len(rewrite_words) <= 2 * len(raw_words)
    return preserves_majority and not_ballooned


def _low_overlap(raw: str, rewrite: str) -> bool:
    """Whether the raw is garbled *enough* (few shared content tokens) that the
    whole-phrase char-similarity escape hatch (``_clear_long_improvement``) is
    trustworthy. The hatch exists for ``"Ario der" -> "are you there"`` (zero
    overlap, phonetic repair); it must NOT become a second door for a
    fabrication that already reproduces much of the raw's content. Trust it only
    when at most a third of the raw's content tokens survive."""
    raw_tokens = _content_tokens(raw)
    shared = len(raw_tokens & _content_tokens(rewrite))
    return 3 * shared <= len(raw_tokens)


def drops_and_invents(raw: str, cleaned: str) -> bool:
    """True iff a cleanup rewrite DROPS most of the raw's content and INVENTS
    new words in its place: it shares at least one content token (so it isn't a
    total non-sequitur the expansion/self-echo guards already cover) yet keeps
    at most half of them.

    Live, the fast-tier cleaner turned ``"TEND ME A LONG STORY ABOUT HER"`` into
    ``"A long story about the ceiling"`` -- same length, so the expansion check
    misses it -- dropping three of six content tokens and fabricating "ceiling".
    Pure text, a dimensionless ratio of the raw's own token count."""
    raw_tokens = _content_tokens(raw)
    shared = len(raw_tokens & _content_tokens(cleaned))
    return shared >= 1 and 2 * shared <= len(raw_tokens)


def _clear_long_improvement(
    streaming_final: str,
    second_pass: str,
    *,
    segment_sec: float | None,
    short_sec: float,
) -> bool:
    """Narrow escape hatch for real longer corrections with low word overlap."""
    if segment_sec is None or segment_sec < short_sec:
        return False

    st_words = _normalized_words(streaming_final)
    sp_words = _normalized_words(second_pass)
    if len(sp_words) < 3:
        return False

    st_compact = _compact_words(st_words)
    sp_compact = _compact_words(sp_words)
    if not st_compact or not sp_compact:
        return False

    # Whole-phrase character similarity catches common ASR segmentation/phonetic
    # repairs such as "Ario der" -> "are you there" without opening the door to
    # unrelated short acknowledgements like "Okay.".
    return SequenceMatcher(None, st_compact, sp_compact).ratio() >= 0.55


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
    2. Reject a 2nd pass that collapses to no content words while the streaming
       final has real content.
    3. Accept punctuation/casing cleanup or a lexical correction only when the two
       transcripts agree after normalization.
    4. For short clips (by duration, or by streaming word count when duration is
       unavailable), require that agreement. For longer clips, also allow a narrow
       whole-phrase similarity escape hatch so a real garbled correction like
       ``"Ario der" -> "are you there"`` still lands.

    Pure, dependency-free, no I/O -- mirrors ``restore_casing`` so it runs on the
    final path with no model or audio import."""
    if not second_pass or not second_pass.strip():
        return streaming_final
    if not streaming_final or not streaming_final.strip():
        return second_pass
    # A 2nd pass that collapses to NO content tokens -- just punctuation / single
    # letters ("H.", "I.", "O.") -- while the streaming final HAS real words is a
    # hallucination on garbled/echoey audio, not a correction. Never let it
    # override real words, even on a LONG clip (the open-speaker live failure
    # run-20260617-225622: a >1.2s clip the streaming pass heard as 'MANY OWN'
    # / 'IT IS' the 2nd pass invented into 'H.' / 'I.', which the length gate
    # below would otherwise have trusted unconditionally).
    st_tokens = _content_tokens(streaming_final)
    sp_tokens = _content_tokens(second_pass)
    if st_tokens and not sp_tokens:
        return streaming_final

    if segment_sec is not None:
        short = segment_sec < short_sec
    else:
        short = len(_normalized_words(streaming_final)) <= short_words

    # A lexical agreement is trusted only when the 2nd pass is GROUNDED in the
    # raw -- else SenseVoice fabrications that share a token or two ("LIKE A
    # QUESTION" -> "...if you found this question") ride the shared-token path
    # into the brain.
    if word_agreement(streaming_final, second_pass) and _grounded_rewrite(
        streaming_final, second_pass
    ):
        return second_pass

    # The whole-phrase char-similarity hatch is only trustworthy for a genuinely
    # garbled raw (few shared tokens); gate it behind _low_overlap so a
    # fabrication that already reproduces much of the raw can't sneak through it.
    if (
        not short
        and _low_overlap(streaming_final, second_pass)
        and _clear_long_improvement(
            streaming_final,
            second_pass,
            segment_sec=segment_sec,
            short_sec=short_sec,
        )
    ):
        return second_pass

    return streaming_final
