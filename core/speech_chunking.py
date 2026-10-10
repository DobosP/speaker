"""Optional desktop first-fragment scheduling; the shared sentence contract stays intact.

Only the first neutral sentence may gain one extra boundary. No tokenizer/model,
background timer, text rewrite, or acoustic/default qualification is implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Mapping

from .contract import _next_cut, drain_complete_sentences

SPEECH_LATENCY_MODES = ("normal", "fast")
_SPACES = " \t\r\n"
_PROTECTED = frozenset('[]{}<>`"“”‘’()')
_MAX_FIRST_SCAN = 1024


@dataclass(frozen=True)
class SpeechChunkingConfig:
    mode: str = "normal"
    min_chars: int = 32
    max_words: int = 28

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in SPEECH_LATENCY_MODES:
            raise ValueError("unsupported speech latency mode")
        if type(self.min_chars) is not int or not 8 <= self.min_chars <= 160:
            raise ValueError("first-fragment minimum must be 8..160 characters")
        if type(self.max_words) is not int or not 8 <= self.max_words <= 80:
            raise ValueError("first-fragment word budget must be 8..80")

    @classmethod
    def from_tts(cls, tts: Mapping | None) -> SpeechChunkingConfig:
        if tts is None:
            return cls()
        if not isinstance(tts, Mapping):
            raise ValueError("tts config must be a mapping")
        return cls(
            mode=tts.get("speech_latency", "normal"),
            min_chars=tts.get("first_fragment_min_chars", 32),
            max_words=tts.get("first_fragment_max_words", 28),
        )


def apply_speech_latency(
    config: dict, requested: str | None = None
) -> tuple[dict, SpeechChunkingConfig]:
    """Pure selection shared by core, doctor and launcher; never changes assets."""
    selected = config
    if requested is not None:
        if type(requested) is not str or requested not in SPEECH_LATENCY_MODES:
            raise ValueError("unsupported speech latency mode")
        selected = deepcopy(config)
        tts = selected.setdefault("tts", {})
        if type(tts) is not dict:
            raise ValueError("tts config must be a dictionary")
        tts["speech_latency"] = requested
    return selected, SpeechChunkingConfig.from_tts(selected.get("tts"))


def _first_fragment_cut(
    text: str, config: SpeechChunkingConfig
) -> tuple[int, int] | None:
    """Select a bounded neutral clause/word cut with a known safe continuation.

    Lookahead prevents a mid-sentence directive becoming a new fragment-leading
    directive. Encountered markup, quotes and grouping abstain rather than lose
    expression/code context. Numeric punctuation and incomplete words stay whole.
    """
    words = 0
    in_word = False
    for i, char in enumerate(text[:_MAX_FIRST_SCAN]):
        if char in _PROTECTED:
            return None
        if char == "'" and not (
            i > 0
            and i + 1 < len(text)
            and text[i - 1].isalnum()
            and text[i + 1].isalnum()
        ):
            return None
        if char not in _SPACES:
            if not in_word:
                words += 1
                in_word = True
        else:
            in_word = False
        clause = char in ",;:" and i > 0 and not text[i - 1].isdigit()
        word_cut = char in _SPACES and words >= config.max_words
        if not (clause or word_cut) or len(text[: i + 1].strip()) < config.min_chars:
            continue
        end = i + 1 if clause else i
        if clause and (i + 1 >= len(text) or text[i + 1] not in _SPACES):
            continue
        after = i + 1
        while after < min(len(text), _MAX_FIRST_SCAN) and text[after] in _SPACES:
            after += 1
        if after >= min(len(text), _MAX_FIRST_SCAN):
            return None  # wait for real continuation, not a new timer/thread
        if text[after] in _PROTECTED or text[after] == "'":
            return None
        if word_cut and (text[end - 1].isdigit() or text[after].isdigit()):
            continue
        return end, after
    return None


class SpeechChunker:
    """Per-reply bounded policy state; full response text stays with its caller."""

    def __init__(self, config: SpeechChunkingConfig | None = None) -> None:
        self.config = config or SpeechChunkingConfig()
        self._buffer = ""
        self._first_pending = True
        self._early_possible = self.config.mode == "fast"

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        if self._first_pending and self._early_possible:
            self._buffer = self._buffer.lstrip(_SPACES)
            normal = _next_cut(self._buffer)
            candidate = _first_fragment_cut(self._buffer, self.config)
            if candidate is None and (
                len(self._buffer) >= _MAX_FIRST_SCAN or self._buffer[:1] in _PROTECTED
            ):
                self._early_possible = False
            if candidate is not None and (normal is None or candidate[0] < normal[0]):
                end, resume = candidate
                first = self._buffer[:end].strip()
                self._buffer = self._buffer[resume:]
                self._first_pending = False
                rest, self._buffer = drain_complete_sentences(self._buffer)
                return [first, *rest]
        complete, self._buffer = drain_complete_sentences(self._buffer)
        if complete:
            self._first_pending = False
        return complete

    def finish(self) -> str:
        tail = self._buffer.strip()
        self._buffer = ""
        self._first_pending = False
        return tail
