"""Deterministic speech-to-intent fast-path (the no-LLM 'speech to action' tier).

Frequent commands -- "what time is it", "set a timer for 5 minutes", and any
custom literal phrases -- are matched by a small grammar and answered/acted on
immediately, with no model and no task in the loop. This is the lowest-latency
path to an action. It runs on final transcripts (and unmapped keyword-spotter
hits) before the brain; a miss falls through to the normal analyzer/LLM path so
nothing is captured silently.

It complements the keyword fast-path (control phrases like "stop" in
``runtime.py``): KWS handles instant control words; this handles short literal
commands and slot-bearing intents (durations, etc.) from the ASR transcript.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional


@dataclass(frozen=True)
class IntentMatch:
    name: str
    slots: dict


_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "ninety": 90,
}


def _to_number(value: str) -> Optional[int]:
    value = (value or "").strip().lower()
    if value.isdigit():
        return int(value)
    return _WORD_NUMBERS.get(value)


def parse_duration(value: str, unit: str) -> Optional[int]:
    """Return a duration in seconds from a value+unit pair, or None if unparseable."""
    n = _to_number(value)
    if n is None:
        return None
    u = (unit or "").lower()
    if u.startswith("sec"):
        return n
    if u.startswith("min"):
        return n * 60
    if u.startswith(("hour", "hr")):
        return n * 3600
    return None


def human_duration(seconds: int) -> str:
    if seconds >= 3600 and seconds % 3600 == 0:
        n, unit = seconds // 3600, "hour"
    elif seconds >= 60 and seconds % 60 == 0:
        n, unit = seconds // 60, "minute"
    else:
        n, unit = seconds, "second"
    return f"{n} {unit}" + ("s" if n != 1 else "")


# Loose, case-insensitive patterns over the lowercased transcript. `\b` after
# "time" keeps "timer" from matching the time intent. Named groups feed slots.
_DEFAULT_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    ("time", re.compile(r"\b(?:what(?:'s| is)?\s+(?:the\s+)?time|what time is it|current time|tell me the time)\b")),
    ("date", re.compile(r"\b(?:what(?:'s| is)?\s+(?:the\s+|today'?s\s+)?date|what day is it|what'?s today|today'?s date)\b")),
    ("timer", re.compile(r"\bset (?:a |an )?(?:timer|alarm)(?: for)? (?P<value>\d+|[a-z]+) (?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?)\b")),
]


class IntentGrammar:
    """Maps a transcript to the first matching :class:`IntentMatch`, or None."""

    def __init__(self, patterns: Optional[list[tuple[str, "re.Pattern[str]"]]] = None):
        self._patterns = patterns if patterns is not None else list(_DEFAULT_PATTERNS)

    @classmethod
    def default(cls) -> "IntentGrammar":
        return cls()

    def match(self, text: str) -> Optional[IntentMatch]:
        t = (text or "").lower().strip()
        for name, rx in self._patterns:
            m = rx.search(t)
            if m:
                return IntentMatch(name, dict(m.groupdict()))
        return None


def _default_scheduler(delay: float, fn: Callable[[], None]) -> Callable[[], None]:
    timer = threading.Timer(delay, fn)
    timer.daemon = True
    timer.start()
    return timer.cancel


class LocalIntentHandler:
    """Resolves matched intents to immediate spoken actions, no LLM involved.

    ``speak`` is the engine's playback entry point. ``phrases`` adds custom
    literal commands (normalized phrase -> spoken response). ``clock`` and
    ``scheduler`` are injectable for deterministic tests. ``handle`` returns True
    when it took the utterance (caller should stop), False to fall through.
    """

    def __init__(
        self,
        speak: Callable[[str], None],
        *,
        grammar: Optional[IntentGrammar] = None,
        phrases: Optional[dict] = None,
        clock: Optional[Callable[[], datetime]] = None,
        scheduler: Optional[Callable[[float, Callable[[], None]], Callable[[], None]]] = None,
        enabled: bool = True,
    ):
        self._speak = speak
        self._grammar = grammar or IntentGrammar.default()
        self._phrases = {str(k).lower().strip(): str(v) for k, v in (phrases or {}).items()}
        self._clock = clock or datetime.now
        self._schedule = scheduler or _default_scheduler
        self.enabled = enabled
        self._timers: list[Callable[[], None]] = []

    def handle(self, text: str) -> bool:
        if not self.enabled or not text:
            return False
        phrase = self._phrases.get(text.lower().strip())
        if phrase is not None:
            self._speak(phrase)
            return True
        match = self._grammar.match(text)
        if match is None:
            return False
        handler = getattr(self, f"_do_{match.name}", None)
        return bool(handler and handler(match.slots))

    def cancel_all(self) -> None:
        """Cancel pending scheduled actions (called on stop / barge-in)."""
        for cancel in self._timers:
            try:
                cancel()
            except Exception:
                pass
        self._timers.clear()

    def _do_time(self, _slots: dict) -> bool:
        now = self._clock()
        hour = now.hour % 12 or 12
        ampm = "AM" if now.hour < 12 else "PM"
        self._speak(f"It's {hour}:{now.minute:02d} {ampm}.")
        return True

    def _do_date(self, _slots: dict) -> bool:
        now = self._clock()
        self._speak(now.strftime("Today is %A, %B ") + f"{now.day}.")
        return True

    def _do_timer(self, slots: dict) -> bool:
        seconds = parse_duration(slots.get("value", ""), slots.get("unit", ""))
        if not seconds or seconds <= 0:
            return False  # unparseable duration -> let the normal path handle it
        label = human_duration(seconds)
        cancel = self._schedule(
            float(seconds), lambda: self._speak(f"Time's up. Your {label} timer is done.")
        )
        self._timers.append(cancel)
        self._speak(f"Okay, timer set for {label}.")
        return True
