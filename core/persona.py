"""The assistant's identity + capability-aware system prompt.

Two jobs the model couldn't do before:

1. **Know what it is.** A configurable persona (name + one-line character) instead
   of an anonymous "local voice assistant".
2. **Know what skills it has.** The system prompt enumerates the assistant's
   *actually-registered* user-facing capabilities (from the
   :class:`~always_on_agent.capabilities.CapabilityRegistry` manifest), and its
   web-access limit reflects the runtime §9.7 egress state rather than a
   hardcoded denial.

``DEFAULT_SYSTEM`` (the byte-identical legacy prompt, composed here from the same
named parts) stays the default for direct callers/tests;
:func:`build_system_prompt` is what the runtime feeds the answering model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

# --- prompt building blocks (one source of truth per sentence) ----------------
_IDENTITY = "You are a local, on-device voice assistant."
_STYLE = (
    "Default to one or two short, natural spoken sentences, no markdown or "
    "lists. BUT when the user asks for a story, a poem, a joke, an explanation, "
    "or to go into detail, give them the full thing directly -- generate it "
    "yourself; never summarize it or offer to find it elsewhere."
)
_ASR = (
    "Your input is from speech recognition and may be garbled or misheard: if a "
    "request is unclear or sounds like a fragment, ask one short clarifying "
    "question instead of guessing, and never invent facts you're unsure of."
)
# Web-access limit, runtime-conditional (the §9.7 egress state decides which).
_NO_WEB = (
    "You answer from your own knowledge -- you have no web access and cannot "
    "open files or apps, so never claim you searched, found, or will look "
    "something up online."
)
_WEB = (
    "You answer mainly from your own knowledge, but you can search the web when a "
    "question needs current information; you cannot open files or apps."
)
_NO_COMMENT = "Don't comment on the user's name, tone, or mood."

# --- skills block framing (registry-backed path ONLY; never part of DEFAULT_SYSTEM) ---
# These wrap the live, §9.7-filtered capability bullets in build_system_prompt so
# the answering model reads them as an instruction ("describe exactly these") rather
# than loose background data. They MUST NOT be folded into DEFAULT_SYSTEM, whose
# bytes are pinned by tests/test_memory_contract.py + test_goal_alignment_fixes.py.
_SKILLS_HEADER = (
    "For reference, here is what you can do for the user (this matters ONLY if they "
    "ask what you can do -- otherwise ignore it):"
)
_SKILLS_GUIDANCE = (
    "Almost always, just answer the user's actual request directly and say nothing "
    "about this list. ONLY if they explicitly ask what you are or what you can do, "
    "give a brief, natural one-sentence summary of the list above, and do not claim "
    "any ability that is not on it -- telling a story, poem, joke, or explanation is "
    "part of answering, not a separate skill. If a message is unclear or sounds like "
    "a fragment, ask one short question -- do not recite what you can do."
)

# The legacy one-paragraph prompt, recomposed from the parts above so it stays
# byte-identical to what shipped (pinned by tests/test_memory_contract.py and
# tests/test_goal_alignment_fixes.py) -- now with a single source per sentence.
DEFAULT_SYSTEM = " ".join([_IDENTITY, _STYLE, _ASR, _NO_WEB, _NO_COMMENT])


@dataclass
class PersonaConfig:
    """Optional identity for the assistant (the ``assistant`` config block).

    All fields default empty -> the anonymous legacy identity, so behaviour is
    unchanged unless the user configures a persona."""

    name: str = ""
    persona: str = ""
    extra: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, object]]) -> "PersonaConfig":
        data = data if isinstance(data, Mapping) else {}
        return cls(
            name=str(data.get("name", "") or ""),
            persona=str(data.get("persona", "") or ""),
            extra=str(data.get("extra", "") or ""),
        )


def render_skills(registry: object, *, web_enabled: bool = False) -> str:
    """A user-facing "what you can do" block from the capability manifest.

    Only lists skills that are actually available right now: a cloud/web-egress
    skill (web.search) is omitted when web is disabled, so the prompt never tells
    the model it can search the web while the limits line says it can't (§9.7).
    Empty when the registry has no manifest (e.g. a bare/stub registry) so the
    prompt degrades to identity + guidance with no skills section."""
    manifest = getattr(registry, "manifest", None)
    if not callable(manifest):
        return ""
    specs = []
    for spec in manifest():
        if not getattr(spec, "user_facing", False):
            continue
        if getattr(spec, "egress", "local") == "cloud" and not web_enabled:
            continue
        specs.append(spec)
    if not specs:
        return ""
    lines = "\n".join(f"- {spec.summary}" for spec in specs)
    # Header + bullets, then the faithful-enumeration instruction so the model
    # reads this as "describe exactly these and claim nothing else" -- not as
    # loose background it can free-associate a self-intro from.
    return _SKILLS_HEADER + "\n" + lines + "\n\n" + _SKILLS_GUIDANCE


def build_system_prompt(
    registry: object = None,
    *,
    persona: Optional[PersonaConfig] = None,
    web_enabled: bool = False,
) -> str:
    """Compose the answering model's system prompt.

    Identity (persona-aware) + the live skill list + the shared interaction
    guidance + a web-access line that reflects the actual §9.7 egress state.
    Reuses the same sentence constants as :data:`DEFAULT_SYSTEM`, so there's one
    source per sentence."""
    if persona is not None and persona.name:
        identity = f"You are {persona.name}, a local, on-device voice assistant."
    else:
        identity = _IDENTITY
    parts: list[str] = [identity]
    if persona is not None and persona.persona:
        parts.append(persona.persona)
    skills = render_skills(registry, web_enabled=web_enabled) if registry is not None else ""
    if skills:
        parts.append(skills)
    parts.append(_STYLE)
    parts.append(_ASR)
    parts.append(_WEB if web_enabled else _NO_WEB)
    parts.append(_NO_COMMENT)
    if persona is not None and persona.extra:
        parts.append(persona.extra)
    return "\n\n".join(parts)
