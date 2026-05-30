"""Continuation (ADD-ON) classifier: decides whether a follow-up final extends
the turn that is still in flight, or starts a fresh one.

The always-on loop turns every ASR final into its own task. When the user *adds
on* to a request that is still being answered ("...and also tomorrow", "oh wait,
make it shorter", "what about Mars too"), the old behaviour spawned a SECOND,
competing task that raced the first -- two LLM generations, two TTS streams, two
overlapping spoken answers. This module is the gate that lets the supervisor
recognise an add-on so it can *merge* it into the in-flight turn instead of
racing it (see :meth:`AgentSupervisor._maybe_continue`).

Design constraints (mirrors :mod:`core.addressing`):

* **Deterministic + cheap + local.** Continuation classification runs on the bus
  thread for every follow-up while a task is in flight; it must add no latency
  and make no network/LLM call -- it stays on the §9.7 always-on local path. The
  default classifier is a pure marker/length heuristic.
* **Pluggable.** :class:`ContinuationClassifier` is a ``Protocol`` so a future
  ``LLMContinuationClassifier`` (fast tier, off by default, never on the local
  critical path) can drop in via the same supervisor seam with no other change.
* **Safe by omission.** Anything the heuristic doesn't clearly recognise is
  :data:`NEW`, which falls back to today's behaviour (a clean second task) -- a
  miss is never *worse* than the status quo.

The supervisor only ever consults this for ``IntentKind.ASSISTANT`` follow-ups,
*after* the deterministic STOP/CONFIRM/DENY/MODE_SWITCH forks have already fired
(see :meth:`AgentSupervisor._execute_decision`), so a real "stop" can never be
misread as a continuation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol, runtime_checkable

from .text import normalize_text

CONTINUE = "CONTINUE"
NEW = "NEW"

_VALID = (CONTINUE, NEW)

# STRONG leading cues: multi-word / unambiguous add-on phrases that mark a
# continuation regardless of utterance length ("and also rank them by price",
# "what about the long-term effects on coastal cities"). Matched as a normalised
# token-prefix. Romanian entries use the de-diacritic'd forms that survive
# ``normalize_text`` (e.g. "de asemenea"), matching the existing bilingual
# convention in ``speech_analyzer`` ("opreste", "cauta", ...).
DEFAULT_STRONG_MARKERS: tuple[str, ...] = (
    "and also", "and then", "and add", "and make", "oh wait", "oh and",
    "what about", "how about", "make it", "change it", "as well",
    "on second thought", "no i mean",
    # Romanian. ``normalize_text`` folds diacritics to ASCII, so these match real
    # accented input too ("și de asemenea" -> "si de asemenea", "în schimb" ->
    # "in schimb").
    "si de asemenea", "de asemenea", "de fapt", "ce zici de", "in schimb",
)

# WEAK leading cues: single ambiguous conjunctions that also begin plenty of
# fresh questions ("and what's the capital of France"). They only mark a
# continuation when the utterance is SHORT (<= addon_max_words) or the next word
# is a modifier -- so a full new question that merely opens with "and"/"but"/"or"
# is NOT swallowed (the headline default-ON false-merge risk). A miss here just
# degrades to today's behaviour (a clean second task), never worse.
DEFAULT_WEAK_MARKERS: tuple[str, ...] = (
    "and", "also", "plus", "but", "or", "actually", "wait", "instead",
    "rather", "i mean",
    # Romanian: "si" (and), "sau" (or), "stai" (wait/hold on).
    "si", "sau", "stai",
)

# Trailing cues ("...and the forecast too", "...in spanish as well"). Bounded by
# length like the weak markers so a long fresh question that merely ends in
# "too"/"also" isn't swallowed.
DEFAULT_TRAILING_MARKERS: tuple[str, ...] = ("too", "as well", "also")

# Leading words that mark a SHORT utterance as a modifier of the running turn
# rather than a fresh question ("in spanish", "make it shorter", "for tomorrow").
# Only consulted for utterances at or under ``addon_max_words`` so a full new
# question that merely starts with a preposition isn't swallowed.
DEFAULT_MODIFIER_LEADS: tuple[str, ...] = (
    "in", "with", "without", "about", "for", "like", "to", "from", "on",
    "more", "less", "shorter", "longer", "faster", "slower", "not",
    "make", "add", "include", "exclude", "only", "just", "maybe",
)

DEFAULT_MERGE_TEMPLATE = (
    'The user first asked: "{prev}". In the same breath they added: "{addon}". '
    "Answer both as one coherent reply, addressing the original request and the "
    "addition together."
)
DEFAULT_CONTINUE_TEMPLATE = (
    'A moment ago you were asked: "{prev}". The user is now adding to that same '
    'request, in the same conversation: "{addon}". Answer just this addition, '
    "keeping it consistent with the earlier reply."
)


@dataclass
class ContinuationConfig:
    """Config for the ADD-ON / continuation feature (the ``continuation`` block).

    ``enabled`` defaults to **false** so programmatic / test construction of the
    supervisor is byte-identical to today; the shipped ``config.json`` opts in.
    ``addon_max_words`` bounds the short-modifier-fragment branch. The marker /
    template fields are overridable per device profile via the shallow merge.
    """

    enabled: bool = False
    addon_max_words: int = 8
    strong_markers: tuple[str, ...] = DEFAULT_STRONG_MARKERS
    weak_markers: tuple[str, ...] = DEFAULT_WEAK_MARKERS
    trailing_markers: tuple[str, ...] = DEFAULT_TRAILING_MARKERS
    modifier_leads: tuple[str, ...] = DEFAULT_MODIFIER_LEADS
    merge_template: str = DEFAULT_MERGE_TEMPLATE
    continue_template: str = DEFAULT_CONTINUE_TEMPLATE

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, object]]) -> "ContinuationConfig":
        data = data or {}
        _missing = object()

        def _tuple(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
            # Distinguish "key absent" (use default) from "key present but empty"
            # (honour the explicit empty override -- e.g. weak_markers: [] to run
            # strong markers only).
            value = data.get(key, _missing)
            if value is _missing:
                return default
            return tuple(str(v) for v in value)  # type: ignore[union-attr]

        return cls(
            enabled=bool(data.get("enabled", False)),
            addon_max_words=int(data.get("addon_max_words", 8) or 8),
            strong_markers=_tuple("strong_markers", DEFAULT_STRONG_MARKERS),
            weak_markers=_tuple("weak_markers", DEFAULT_WEAK_MARKERS),
            trailing_markers=_tuple("trailing_markers", DEFAULT_TRAILING_MARKERS),
            modifier_leads=_tuple("modifier_leads", DEFAULT_MODIFIER_LEADS),
            merge_template=str(data.get("merge_template") or DEFAULT_MERGE_TEMPLATE),
            continue_template=str(data.get("continue_template") or DEFAULT_CONTINUE_TEMPLATE),
        )


@runtime_checkable
class ContinuationClassifier(Protocol):
    """Returns :data:`CONTINUE` (extend the in-flight turn) or :data:`NEW`."""

    def classify(self, addon: str, prev: str = "") -> str: ...


class HeuristicContinuationClassifier:
    """Deterministic marker + short-fragment continuation classifier.

    Pure, stateless, no I/O: a normalised token compare over a small marker set.
    ``CONTINUE`` iff the follow-up *clearly* extends the current turn -- a leading
    or trailing continuation marker, or a short utterance that opens with a
    modifier word. Everything else is :data:`NEW` (safe fallback). ``prev`` is
    accepted for parity with the ``Protocol`` (and future similarity-based or LLM
    classifiers) but the heuristic decides from the add-on alone.
    """

    def __init__(self, config: Optional[ContinuationConfig] = None) -> None:
        cfg = config or ContinuationConfig()
        self._max_words = max(1, int(cfg.addon_max_words))
        # Normalise markers once so they compare against normalize_text() output
        # (lowercased, de-diacritic'd, punctuation stripped).
        self._strong = tuple(
            tuple(n.split()) for m in cfg.strong_markers if (n := normalize_text(m))
        )
        self._weak = tuple(
            tuple(n.split()) for m in cfg.weak_markers if (n := normalize_text(m))
        )
        self._trailing = tuple(
            tuple(n.split()) for m in cfg.trailing_markers if (n := normalize_text(m))
        )
        self._modifier_leads = frozenset(
            n for m in cfg.modifier_leads if (n := normalize_text(m))
        )

    def classify(self, addon: str, prev: str = "") -> str:
        words = normalize_text(addon).split()
        if not words:
            return NEW
        short = len(words) <= self._max_words
        # Strong multi-word add-on cues: always a continuation.
        for marker in self._strong:
            if tuple(words[: len(marker)]) == marker:
                return CONTINUE
        # Weak conjunction cues: only when the utterance is short, or the cue is
        # immediately followed by a modifier word -- otherwise a long fresh
        # question that merely opens with "and"/"but"/"or" is left as NEW.
        for marker in self._weak:
            k = len(marker)
            if tuple(words[:k]) == marker and (
                short or (len(words) > k and words[k] in self._modifier_leads)
            ):
                return CONTINUE
        if short:
            # Trailing cue ("the forecast too") and short modifier-led fragment
            # ("in spanish", "make it shorter") -- both bounded to short utterances.
            for marker in self._trailing:
                if len(words) >= len(marker) and tuple(words[-len(marker):]) == marker:
                    return CONTINUE
            if words[0] in self._modifier_leads:
                return CONTINUE
        return NEW


class ScriptedContinuationClassifier:
    """Test fake: maps add-on text -> decision. Anything not in the map is
    ``default`` (defaults to :data:`NEW`). Records ``.calls`` for assertions."""

    def __init__(
        self, mapping: Optional[dict[str, str]] = None, *, default: str = NEW
    ) -> None:
        self._mapping = dict(mapping or {})
        self._default = default
        self.calls: list[tuple[str, str]] = []

    def classify(self, addon: str, prev: str = "") -> str:
        self.calls.append((addon, prev))
        return self._mapping.get(addon, self._default)
