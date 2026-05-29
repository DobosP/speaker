from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .memory import Memory, SessionMemory
from .text import keywords, normalize_text


@dataclass(frozen=True)
class CapabilityResult:
    ok: bool
    text: str
    data: dict[str, object] = field(default_factory=dict)
    citations: tuple[str, ...] = ()
    error: str = ""


CapabilityProvider = Callable[[str, dict[str, object]], CapabilityResult]


class CapabilityRegistry:
    def __init__(self):
        self._providers: dict[str, CapabilityProvider] = {}

    def register(self, name: str, provider: CapabilityProvider) -> None:
        self._providers[name] = provider

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

    registry.register("assistant.answer", assistant)
    registry.register("search.local", search)
    # ``web.search`` defaults to the local corpus so the core-free brain stays
    # self-contained (RESEARCH/SEARCH plans + the ReAct catalog reference it).
    # The integration layer (core/websearch.attach_web_search_capability, wired
    # in core/runtime.py) OVERRIDES this with the §9.7-gated SearXNG-backed
    # provider -- mirroring how attach_llm_capabilities overrides assistant.answer.
    registry.register("web.search", search)
    registry.register("research.scope", research_scope)
    registry.register("research.local", research)
    registry.register("command.stage", command)
    registry.register("dictation.clean", dictation)
    registry.register("meeting.note", meeting_note)
    return registry
