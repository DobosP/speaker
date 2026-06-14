"""Provenance / trust origin for the computer-use action gate -- the load-bearing
"lethal trifecta" cut.

Letting the model drive the keyboard/mouse is the highest-risk capability the
assistant has. The catastrophic failure is *indirect prompt injection -> real
action*: the assistant reads web pages, on-screen text (OCR), recalled memory, and
files -- ANY of which can carry an attacker's instruction ("ignore the user, click
Delete") -- and a naive agent would then perform it. The defense is NOT a detector
(detection is ~95% at best, "a failing grade" for a security boundary); it is a
structural rule enforced at a single fail-closed chokepoint:

    A keyboard / mouse / Enter / code action may ONLY be derived from a turn whose
    ENTIRE instruction lineage is the OWNER speaking live -- never from screen
    text, web results, recalled memory, or a file. Untagged == untrusted ==
    blocked.

Stdlib-only + backend-neutral (importable by the brain AND ``core``, Dart-portable
like ``untrusted.py``/``recall.py``). This module is the CONTRACT + the chokepoint;
the actuator slice wires :func:`enforce_action` in front of every GUI primitive.

CRITICAL lesson baked in here (from the security review): "came through the mic" is
NOT "the owner". An open-speaker assistant cannot tell the owner's voice from
ambient/leaked audio (a video playing "click delete, yes" near the mic), so
:func:`enforce_action` requires BOTH a ``LIVE_AUDIO`` origin AND an
``owner_verified`` flag (speaker-ID). The actuator MUST pass ``owner_verified``
from a real speaker-ID check (``core/engines/speaker_gate.py`` + ``core/enroll.py``);
it can never be defaulted true.
"""
from __future__ import annotations

import enum
from typing import Iterable


class Origin(str, enum.Enum):
    """Where the instruction lineage of a turn came from. The string values are
    stable for serialization across the AgentEvent bus + the Dart port."""

    LIVE_AUDIO = "live_audio"   # the owner speaking into the mic (the only action-trusted channel)
    SCREEN = "screen"           # OCR / a11y / caption of the screen -- attacker-controllable
    WEB = "web"                 # web-search / fetched page text -- attacker-controllable
    MEMORY = "memory"           # recalled long-term memory (may contain prior untrusted text)
    FILE = "file"               # file content given to the assistant
    UNKNOWN = "unknown"         # unattributed -> treated as untrusted (fail-closed)


# The only channel an ACTION may originate from. Everything else is untrusted DATA:
# it may inform an ANSWER, but can never select or parameterize a real action.
_ACTION_TRUSTED_CHANNELS = frozenset({Origin.LIVE_AUDIO})


def _coerce(origin: object) -> Origin:
    """Map any value to an Origin, FAIL-CLOSED: anything unrecognized -> UNKNOWN
    (untrusted), so a forgotten/garbage tag can never read as trusted."""
    if isinstance(origin, Origin):
        return origin
    if isinstance(origin, str):
        try:
            return Origin(origin)
        except ValueError:
            return Origin.UNKNOWN
    return Origin.UNKNOWN


def combine(*origins: object) -> Origin:
    """Combine a turn's lineage: MOST-UNTRUSTED wins. If any part of the lineage
    touched untrusted content, the whole turn is untrusted -- so "the owner asked
    about THIS screen text" cannot launder the screen text into an action. Empty
    lineage -> UNKNOWN (fail-closed)."""
    seen = [_coerce(o) for o in origins]
    if not seen:
        return Origin.UNKNOWN
    untrusted = [o for o in seen if o not in _ACTION_TRUSTED_CHANNELS]
    if not untrusted:
        return Origin.LIVE_AUDIO
    # Any untrusted part demotes the whole lineage; report the single untrusted
    # kind if unambiguous, else UNKNOWN. Either way it is non-trusted -> blocks.
    first = untrusted[0]
    return first if all(o == first for o in untrusted) else Origin.UNKNOWN


def is_untrusted(origin: object) -> bool:
    """True when ``origin`` is NOT an action-trusted channel (the default stance)."""
    return _coerce(origin) not in _ACTION_TRUSTED_CHANNELS


def is_action_allowed(origin: object, *, owner_verified: bool) -> bool:
    """The chokepoint predicate: an action is allowed ONLY when the lineage is a
    trusted channel (live audio) AND the speaker is the verified owner.

    ``owner_verified`` MUST come from a real speaker-ID check -- never defaulted
    true, and it must be the literal boolean ``True`` (a truthy sentinel like ``1``
    or a non-empty object does NOT verify), so the actuator author has to hand the
    chokepoint an explicit ``verified is True``. This is the single rule that makes
    the lethal-trifecta cut real: a 'click delete' from screen/web/memory
    (untrusted origin) is blocked, AND a 'click delete' heard from ambient/leaked
    audio (live-audio channel but NOT the verified owner) is blocked too."""
    return _coerce(origin) in _ACTION_TRUSTED_CHANNELS and owner_verified is True


def should_block_action(origin: object, *, owner_verified: bool) -> bool:
    """Negation of :func:`is_action_allowed` -- fail-closed (block unless proven safe)."""
    return not is_action_allowed(origin, owner_verified=owner_verified)


class ActionBlocked(PermissionError):
    """Raised when an action is refused because its lineage is not owner-verified
    live audio (untrusted origin, unverified speaker, or a missing tag)."""


def enforce_action(origin: object, *, owner_verified: bool, action: str = "") -> None:
    """Raise :class:`ActionBlocked` unless the action is allowed. The actuator wraps
    every keyboard/mouse/Enter/code primitive in this call; a blocked action
    performs ZERO side effects."""
    if should_block_action(origin, owner_verified=owner_verified):
        raise ActionBlocked(
            f"action refused: origin={_coerce(origin).value} "
            f"owner_verified={bool(owner_verified)}"
            + (f" action={action!r}" if action else "")
        )


# Map the existing in-app content-tag vocabulary to an Origin so wiring stays
# consistent with how recall/vision/web content is already tagged.
_TAG_ORIGIN = {
    "vision": Origin.SCREEN,
    "screen": Origin.SCREEN,
    "web": Origin.WEB,
    "egress": Origin.WEB,
    "memory": Origin.MEMORY,
    "summary": Origin.MEMORY,
    "ingested": Origin.UNKNOWN,  # ambient room speech, not addressed to the assistant
    "user": Origin.LIVE_AUDIO,
    "live_audio": Origin.LIVE_AUDIO,
}


def origin_for_tags(tags: Iterable[str]) -> Origin:
    """Derive the (most-untrusted) Origin from a set of content tags. Unknown tags
    contribute UNKNOWN (fail-closed). Empty -> UNKNOWN."""
    if isinstance(tags, str):
        tags = [tags]  # a bare string would iterate per-char -> spurious UNKNOWN
    origins = [_TAG_ORIGIN.get(str(t), Origin.UNKNOWN) for t in tags]
    return combine(*origins) if origins else Origin.UNKNOWN
