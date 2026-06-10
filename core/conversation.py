"""Recent-conversation context for the answering model (short-term memory).

``assistant()`` used to build the prompt from the CURRENT utterance alone, so the
model had no idea what was just said: "what's its population?" after "what's the
capital of France?" had no referent, and "make it shorter" / "the second one"
couldn't resolve. This assembles a compact ``=== Recent conversation ===`` block
from the last few user/assistant turns kept in the working-window memory, which
``assistant()`` prepends to the system prompt -- so the model resolves the
references itself (subsuming a separate coreference pass).

Distinct from semantic *recall* (``memory.context_for_llm``), which surfaces
relevant snippets from PAST sessions: this is the immediate, chronological
this-conversation history. Bounded (turns + chars) so it stays cheap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from always_on_agent.text import normalize_text

# Default tags the conversation turns carry in memory (assistant() ingests user
# queries as ("user",); the supervisor ingests replies as ("assistant_output",)).
_USER_TAGS = ("user",)
_ASSISTANT_TAGS = ("assistant_output",)

# Phrases that mean "drop the current thread and start fresh". When a USER turn
# matches one (see ``is_topic_reset``), the recent-conversation block is cut off
# at that turn so the OLD topic cannot bleed into the new one -- the live failure
# (run-20260610-003800): after a barge + "Start again" the model re-apologized
# about the stale "volume" topic because the block still carried it. Matched as
# a contiguous normalized-token phrase, only on SHORT utterances
# (``reset_max_words``) so a long sentence that merely *mentions* "never mind"
# is not misread as a reset. EN + RO (de-diacritic'd to match normalize_text).
DEFAULT_RESET_PHRASES: tuple[str, ...] = (
    # English
    "start again", "start over", "start fresh", "never mind", "nevermind",
    "forget it", "forget that", "forget about it", "new topic",
    "different topic", "change the subject", "new question", "clean slate",
    # Romanian (normalize_text folds diacritics): "de la început" -> "de la
    # inceput", "de la capăt" -> "de la capat", "las-o baltă" -> "las o balta".
    "de la inceput", "de la capat", "alt subiect", "subiect nou", "las o balta",
)

# Placeholder/abstain assistant replies that carry no conversational content --
# excluded from the recent-context block so they don't pollute it. These mirror
# the constants in core/capabilities.py (ABSTAIN_REPLY) and
# always_on_agent/supervisor.py (_TIMEOUT_APOLOGY); kept as literals here to
# avoid an import cycle.
_EXCLUDED_REPLIES = (
    "Sorry, I don't have an answer for that.",
    "Sorry, that took too long -- let's try again.",
)


@dataclass(frozen=True)
class RecentContextConfig:
    """Gating for recent-conversation injection (the flat ``memory`` block).

    ``enabled`` defaults **on**: short-term context is the headline fix and is
    cheap (no embeddings, just the last few turns). ``max_turns`` / ``max_chars``
    bound the volume so TTFT stays in check on a small model."""

    enabled: bool = True
    max_turns: int = 6
    max_chars: int = 800
    per_turn_chars: int = 240
    user_tags: tuple[str, ...] = field(default=_USER_TAGS)
    assistant_tags: tuple[str, ...] = field(default=_ASSISTANT_TAGS)
    excluded_replies: tuple[str, ...] = field(default=_EXCLUDED_REPLIES)
    # Topic reset: a user turn matching a reset phrase cuts the block off there
    # (and a reset CURRENT query suppresses the block for its own turn).
    reset_enabled: bool = True
    reset_phrases: tuple[str, ...] = field(default=DEFAULT_RESET_PHRASES)
    reset_max_words: int = 8

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, object]]) -> "RecentContextConfig":
        data = data or {}
        phrases = data.get("recent_context_reset_phrases")
        return cls(
            enabled=bool(data.get("recent_context_enabled", True)),
            max_turns=int(data.get("recent_context_turns", 6) or 6),
            max_chars=int(data.get("recent_context_max_chars", 800) or 800),
            per_turn_chars=int(data.get("recent_context_per_turn_chars", 240) or 240),
            reset_enabled=bool(data.get("recent_context_reset_enabled", True)),
            reset_phrases=(
                tuple(str(p) for p in phrases)
                if isinstance(phrases, (list, tuple))
                else DEFAULT_RESET_PHRASES
            ),
            reset_max_words=int(data.get("recent_context_reset_max_words", 8) or 8),
        )


def _contains_phrase(words: Sequence[str], phrase: Sequence[str]) -> bool:
    n = len(phrase)
    if n == 0 or n > len(words):
        return False
    target = tuple(phrase)
    return any(tuple(words[i : i + n]) == target for i in range(len(words) - n + 1))


def is_topic_reset(text: str, config: Optional[RecentContextConfig] = None) -> bool:
    """True iff ``text`` is a short "drop the thread, start fresh" utterance.

    Normalized contiguous-phrase containment ("can we start again please" counts)
    bounded to ``reset_max_words`` so a long sentence that merely *mentions* a
    phrase ("the hero says never mind and walks away...") is not a reset. A false
    positive only costs this turn's recent-context block (the model answers
    fresh); a miss is the status quo. Pure + deterministic."""
    config = config or RecentContextConfig()
    if not config.reset_enabled:
        return False
    words = normalize_text(text).split()
    if not words or len(words) > config.reset_max_words:
        return False
    for phrase in config.reset_phrases:
        if _contains_phrase(words, normalize_text(phrase).split()):
            return True
    return False


def collect_recent_turns(
    memory: object,
    config: Optional[RecentContextConfig] = None,
    *,
    current_query: Optional[str] = None,
) -> list[tuple[str, str]]:
    """The recent conversation as ``[(role, text), ...]`` (role is "User"/"You").

    The data behind the block: the last ``max_turns`` user/assistant turns in
    order, each text truncated to ``per_turn_chars``, ambient/ingested/meeting
    items and placeholder/abstain replies excluded. ``[]`` when disabled / empty.
    Best-effort: any error yields ``[]`` so it can never break a turn. The caller
    needs the per-turn texts (not just the formatted block) to float the prompt's
    sensitivity over each prior turn (§9.7).

    Topic reset (two layers, both via ``is_topic_reset``): a reset
    ``current_query`` ("start again") suppresses the block for ITS turn -- the
    model must answer fresh, not re-explain the stale thread -- and a reset USER
    turn already in memory cuts the block off there for every LATER turn, so the
    pre-reset topic never bleeds back in."""
    config = config or RecentContextConfig()
    if not config.enabled or memory is None:
        return []
    if current_query is not None and is_topic_reset(current_query, config):
        return []
    get_all = getattr(memory, "all", None)
    if not callable(get_all):
        return []
    try:
        items = list(get_all())
    except Exception:  # noqa: BLE001 - context is best-effort, never fatal
        return []
    user_tags = set(config.user_tags)
    assistant_tags = set(config.assistant_tags)
    excluded = set(config.excluded_replies)
    turns: list[tuple[str, str]] = []
    for item in items:
        tags = set(getattr(item, "tags", ()) or ())
        text = (getattr(item, "text", "") or "").strip()
        if not text or text in excluded:
            continue
        if tags & user_tags:
            role = "User"
        elif tags & assistant_tags:
            role = "You"
        else:
            continue  # ambient / ingested / meeting notes are not conversation turns
        turns.append((role, text[: config.per_turn_chars]))
    # Cut the thread at the most recent reset USER turn: everything at/before it
    # (including the reset utterance itself) is the abandoned topic.
    if config.reset_enabled:
        for i in range(len(turns) - 1, -1, -1):
            role, text = turns[i]
            if role == "User" and is_topic_reset(text, config):
                turns = turns[i + 1:]
                break
    return turns[-config.max_turns:]


def format_recent_block(
    turns: list[tuple[str, str]], config: Optional[RecentContextConfig] = None
) -> str:
    """Render collected turns as a bounded ``=== Recent conversation ===`` block,
    or ``''`` when there are none. Drops oldest turns until it fits ``max_chars``."""
    config = config or RecentContextConfig()
    if not turns:
        return ""
    rows = [f"{role}: {text}" for role, text in turns]
    header = "=== Recent conversation (most recent last) ==="
    block = header + "\n" + "\n".join(rows)
    while len(block) > config.max_chars and len(rows) > 1:
        rows.pop(0)
        block = header + "\n" + "\n".join(rows)
    return block


def build_recent_context(
    memory: object,
    config: Optional[RecentContextConfig] = None,
    *,
    current_query: Optional[str] = None,
) -> str:
    """A bounded recent-conversation block, or ``''`` when disabled / empty."""
    config = config or RecentContextConfig()
    return format_recent_block(
        collect_recent_turns(memory, config, current_query=current_query), config
    )
