from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .memory import Memory, SessionMemory
from .text import keywords, normalize_text


@dataclass(frozen=True)
class CapabilityResult:
    ok: bool
    text: str
    data: dict[str, object] = field(default_factory=dict)
    citations: tuple[str, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class CapabilitySpec:
    """Self-describing metadata for a registered capability.

    The single source of truth the controller and the model reason over: it
    drives the ReAct planner's tool catalog, the assistant's "here is what I can
    do" system prompt, and the startup reconciliation check -- so the three can
    no longer drift from the actually-registered providers.

    - ``summary``: a short, user/model-facing "what it does" line.
    - ``when_to_use``: planner-facing guidance ("use this when ..."); falls back
      to ``summary`` when empty.
    - ``egress``: ``"local"`` or ``"cloud"`` -- the §9.7 data-boundary class.
    - ``speaks``: produces spoken output (vs a silent side-effect like dictation).
    - ``side_effecting``: takes an action / changes state (command, note).
    - ``planner_tool``: exposed to the ReAct planner as a callable tool.
    - ``user_facing``: enumerated in the model's self-description.
    """

    name: str
    summary: str
    when_to_use: str = ""
    egress: str = "local"
    speaks: bool = True
    side_effecting: bool = False
    planner_tool: bool = False
    user_facing: bool = True


CapabilityProvider = Callable[[str, dict[str, object]], CapabilityResult]


class CapabilityRegistry:
    def __init__(self):
        self._providers: dict[str, CapabilityProvider] = {}
        self._specs: dict[str, CapabilitySpec] = {}

    def register(
        self,
        name: str,
        provider: CapabilityProvider,
        *,
        spec: Optional[CapabilitySpec] = None,
    ) -> None:
        self._providers[name] = provider
        if spec is not None:
            # Allow a spec to carry its own name, but key it under ``name`` so
            # registration and lookup never disagree.
            self._specs[name] = spec
        # When re-registering WITHOUT a spec (an override -- e.g. the LLM-backed
        # assistant.answer replacing the stub, or the §9.7 web.search), keep the
        # existing spec so capability metadata survives the swap.

    def invoke(self, name: str, query: str, context: dict[str, object] | None = None) -> CapabilityResult:
        provider = self._providers.get(name)
        if provider is None:
            return CapabilityResult(False, "", error=f"missing capability: {name}")
        try:
            return provider(query, context or {})
        except Exception as exc:
            return CapabilityResult(False, "", error=str(exc))

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def spec(self, name: str) -> Optional[CapabilitySpec]:
        return self._specs.get(name)

    def manifest(self) -> tuple[CapabilitySpec, ...]:
        """All registered specs, name-sorted (stable for prompts/logs)."""
        return tuple(self._specs[name] for name in sorted(self._specs))

    def planner_tools(self) -> tuple[str, ...]:
        """Capability names the ReAct planner may call (planner_tool=True)."""
        return tuple(spec.name for spec in self.manifest() if spec.planner_tool)

    def describe(self, names: Optional[Sequence[str]] = None, *, planner: bool = False) -> str:
        """Render a ``- name: description`` catalog for prompts / logs.

        ``planner`` picks the planner-facing ``when_to_use`` (falling back to
        ``summary``); otherwise the user-facing ``summary``. Names without a spec
        degrade to a generic line rather than vanishing."""
        if names is None:
            names = [spec.name for spec in self.manifest()]
        lines: list[str] = []
        for name in names:
            spec = self._specs.get(name)
            if spec is None:
                lines.append(f"- {name}: a local capability")
                continue
            desc = (spec.when_to_use or spec.summary) if planner else spec.summary
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)


def create_default_capabilities(memory: Memory | None = None) -> CapabilityRegistry:
    memory = memory or SessionMemory()
    registry = CapabilityRegistry()
    corpus = {
        "moonshine": "Moonshine is an open-source ASR family optimized for fast speech recognition on edge devices.",
        "pipecat": "Pipecat is an open-source Python framework for realtime voice and multimodal conversational agents.",
        "livekit": "LiveKit Agents is an open-source framework for realtime voice agents, WebRTC sessions, and STT-LLM-TTS pipelines.",
        "wyoming": "Wyoming is a protocol used by Home Assistant to connect local voice services such as wake word, STT, and TTS.",
        "ollama": "Ollama runs local LLMs behind a simple API and is useful for offline assistant reasoning.",
    }

    def assistant(query: str, context: dict[str, object]) -> CapabilityResult:
        hits = memory.search(query, limit=2)
        suffix = ""
        if hits:
            suffix = " Relevant memory: " + " | ".join(item.text for item in hits)
        return CapabilityResult(True, f"I can help with: {query}.{suffix}".strip())

    def search(query: str, context: dict[str, object]) -> CapabilityResult:
        q = set(normalize_text(query).split())
        results = []
        for key, value in corpus.items():
            haystack = set(normalize_text(key + " " + value).split())
            score = len(q & haystack)
            if key in q:
                score += 3
            if score:
                results.append((score, key, value))
        results.sort(reverse=True)
        if not results:
            return CapabilityResult(True, f"No local result for: {query}", data={"results": []})
        top = results[:3]
        text = " ".join(value for _, _, value in top)
        return CapabilityResult(
            True,
            text,
            data={"results": [{"name": key, "summary": value} for _, key, value in top]},
            citations=tuple(key for _, key, _ in top),
        )

    def research(query: str, context: dict[str, object]) -> CapabilityResult:
        topic_words = keywords(query, limit=5)
        plan = [
            f"define scope around {', '.join(topic_words) or query}",
            "collect local/open-source candidates",
            "compare latency, license, integration cost, and offline support",
            "produce recommendation and next implementation step",
        ]
        search_result = search(query, context)
        text = (
            f"Research summary for {query}: "
            f"{search_result.text} "
            f"Recommended path: keep local Python audio, add adapters and supervisor tasks."
        )
        return CapabilityResult(True, text, data={"plan": plan, "search": search_result.data})

    def research_scope(query: str, context: dict[str, object]) -> CapabilityResult:
        topic_words = keywords(query, limit=6)
        scope = {
            "topic": query,
            "keywords": topic_words,
            "questions": [
                f"What is the current best OSS option for {query}?",
                "What can run locally with low latency?",
                "What adapter boundary avoids locking the project to one framework?",
            ],
        }
        return CapabilityResult(
            True,
            f"Research scope: {', '.join(topic_words) or query}",
            data=scope,
        )

    def command(query: str, context: dict[str, object]) -> CapabilityResult:
        return CapabilityResult(
            True,
            f"Command staged for confirmation: {query}",
            data={"requires_confirmation": True},
        )

    def dictation(query: str, context: dict[str, object]) -> CapabilityResult:
        return CapabilityResult(True, query, data={"silent": True})

    def meeting_note(query: str, context: dict[str, object]) -> CapabilityResult:
        memory.add(query, tags=("meeting",) + keywords(query))
        return CapabilityResult(True, "", data={"stored": True, "silent": True})

    registry.register(
        "assistant.answer", assistant,
        spec=CapabilitySpec(
            "assistant.answer",
            summary="answer questions and chat directly from your own knowledge",
            egress="local", speaks=True, planner_tool=False, user_facing=True,
        ),
    )
    registry.register(
        "search.local", search,
        spec=CapabilitySpec(
            "search.local",
            summary="look something up in the on-device knowledge corpus",
            when_to_use="look up a topic in the local knowledge corpus",
            egress="local", speaks=True, planner_tool=True, user_facing=False,
        ),
    )
    # ``web.search`` defaults to the local corpus so the core-free brain stays
    # self-contained (RESEARCH/SEARCH plans + the ReAct catalog reference it).
    # The integration layer (core/websearch.attach_web_search_capability, wired
    # in core/runtime.py) OVERRIDES this with the §9.7-gated SearXNG-backed
    # provider -- mirroring how attach_llm_capabilities overrides assistant.answer.
    # The override re-registers WITHOUT a spec, so this metadata persists.
    registry.register(
        "web.search", search,
        spec=CapabilitySpec(
            "web.search",
            summary="search the web for current information",
            when_to_use="search the web for current information on a topic",
            egress="cloud", speaks=True, planner_tool=True, user_facing=True,
        ),
    )
    registry.register(
        "research.scope", research_scope,
        spec=CapabilitySpec(
            "research.scope",
            summary="outline the scope and key questions for a topic",
            when_to_use="outline the scope and key questions for a topic",
            egress="local", speaks=False, planner_tool=True, user_facing=False,
        ),
    )
    registry.register(
        "research.local", research,
        spec=CapabilitySpec(
            "research.local",
            summary="research a topic and give a recommendation",
            when_to_use="synthesize gathered findings into a recommendation",
            egress="local", speaks=True, planner_tool=True, user_facing=True,
        ),
    )
    # The side-effecting, mode-gated capabilities below are NOT user_facing: a
    # turn only reaches them via an explicit prefix ("dictate ...", "run ...") or
    # a prior mode switch, chosen deterministically by the analyzer -- NOT by the
    # answering LLM. Advertising them in the answering model's self-description
    # would make it CLAIM it took a note / ran a command when the turn was
    # actually a plain text reply with no side-effect (a confabulation). They
    # stay in the manifest (for the planner + reconciliation), just not in the
    # "what you can do" block. command.stage is also a no-op stub by default
    # (real execution needs the opt-in agent brain) and contradicts the prompt's
    # "cannot open files or apps" limit.
    registry.register(
        "command.stage", command,
        spec=CapabilitySpec(
            "command.stage",
            summary="run a system command (asks you to confirm first)",
            egress="local", speaks=True, side_effecting=True,
            planner_tool=False, user_facing=False,
        ),
    )
    registry.register(
        "dictation.clean", dictation,
        spec=CapabilitySpec(
            "dictation.clean",
            summary="take dictation and write down what you say",
            egress="local", speaks=False, side_effecting=True,
            planner_tool=False, user_facing=False,
        ),
    )
    registry.register(
        "meeting.note", meeting_note,
        spec=CapabilitySpec(
            "meeting.note",
            summary="take meeting notes and remember them",
            egress="local", speaks=False, side_effecting=True,
            planner_tool=False, user_facing=False,
        ),
    )
    return registry
