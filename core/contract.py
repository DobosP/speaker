"""The cross-language behavior contract every shell shares.

These are small, pure, dependency-free functions whose behavior is pinned by the
fixtures in ``tests/golden/`` and **must match the Dart port in
``mobile/lib/contract.dart`` exactly**. They are the "share the contract, not a
binary core" seam from ``docs/target_architecture.md`` §9: the Python core and
the Dart mobile shell each implement the brain in their own language, and these
fixtures keep the two from silently drifting.

Covered (genuinely shared by both runtimes):
- streaming-TTS sentence splitting (``stream_sentences`` / ``drain_complete_sentences``)
- control-command normalization + stop recognition (``normalize_command`` / ``is_stop_command``)

Not covered here: modes, confirm/deny, and the priority event bus are part of the
Python brain only (the mobile shell has no supervisor), so they stay outside the
cross-language contract by design.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

# --- streaming-TTS sentence splitting -------------------------------------

# A boundary is a newline, or a sentence terminator immediately followed by
# whitespace. Requiring the trailing whitespace is what keeps "3.14" (or a "."
# that is simply the latest token so far) from being split mid-number.
_WHITESPACE = " \t\n\r"
_TERMINATORS = ".!?"


def _is_space(ch: str) -> bool:
    return ch in _WHITESPACE


def _next_cut(buffer: str) -> Optional[tuple[int, int]]:
    """``(emit_end, resume)`` for the first complete boundary, or ``None``.

    ``emit_end`` is the slice end of the spoken text (terminator included, the
    newline excluded); ``resume`` is where the remaining buffer continues (the
    boundary characters are consumed)."""
    for i, ch in enumerate(buffer):
        if ch == "\n":
            return (i, i + 1)
        if ch in _TERMINATORS and i + 1 < len(buffer) and _is_space(buffer[i + 1]):
            return (i + 1, i + 2)
    return None


def drain_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """Pull every *complete* sentence out of ``buffer``.

    Returns ``(sentences, remaining)``; ``remaining`` is the trailing,
    not-yet-terminated text to keep buffering. Each sentence is stripped, and
    empty sentences are dropped."""
    out: list[str] = []
    while True:
        cut = _next_cut(buffer)
        if cut is None:
            break
        emit_end, resume = cut
        sentence = buffer[:emit_end].strip()
        buffer = buffer[resume:]
        if sentence:
            out.append(sentence)
    return out, buffer


def stream_sentences(tokens: Iterable[str]) -> list[str]:
    """The full streaming-TTS emission for a token stream.

    Concatenates ``tokens`` in order, emitting complete sentences as boundaries
    arrive and flushing the trailing remainder at the end — the exact ordered
    list of chunks that would be spoken."""
    out: list[str] = []
    buffer = ""
    for token in tokens:
        buffer += token
        sentences, buffer = drain_complete_sentences(buffer)
        out.extend(sentences)
    tail = buffer.strip()
    if tail:
        out.append(tail)
    return out


# --- control-command normalization + stop recognition ---------------------

# Canonical normalization: lowercase, drop anything that isn't a-z or space,
# collapse runs of spaces, trim. This is what lets "Stop!", "  cancel. " and
# "be   quiet" all resolve to their bare command form.
_NON_COMMAND = re.compile(r"[^a-z ]")
_SPACE_RUN = re.compile(r" +")

# The stop-class control phrases both runtimes recognize. Mode/confirm/deny
# commands are config-driven and desktop-only, so they are not listed here.
STOP_COMMANDS = frozenset({
    "stop", "cancel", "cancel that", "quiet", "stop talking", "stop speaking",
    "be quiet",
})


def normalize_command(text: str) -> str:
    normalized = _NON_COMMAND.sub("", (text or "").lower())
    return _SPACE_RUN.sub(" ", normalized).strip()


def is_stop_command(text: str) -> bool:
    return normalize_command(text) in STOP_COMMANDS
