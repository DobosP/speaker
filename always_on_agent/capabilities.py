from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
import time
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Optional, Sequence

from .memory import Memory, SessionMemory
from .origin import Origin
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
    - ``authority``: controller-enforced action provenance. ``none`` is read-only
      or legacy behavior, ``direct_live`` accepts only an unmodified live-audio
      request, and ``verified_owner`` additionally requires speaker identity.
    - ``requires_confirmation``: the controller must have completed its staged
      readback/confirmation before the provider may run.
    """

    name: str
    summary: str
    when_to_use: str = ""
    egress: str = "local"
    speaks: bool = True
    side_effecting: bool = False
    planner_tool: bool = False
    user_facing: bool = True
    authority: Literal["none", "direct_live", "verified_owner"] = "none"
    requires_confirmation: bool = False

    def __post_init__(self) -> None:
        if self.authority not in {"none", "direct_live", "verified_owner"}:
            raise ValueError("invalid capability authority")
        if self.requires_confirmation and self.authority == "none":
            raise ValueError("confirmation requires an explicit capability authority")


CapabilityProvider = Callable[[str, dict[str, object]], CapabilityResult]


@dataclass(frozen=True)
class CapabilityInvocationResult:
    """Detached, recursively immutable result exposed to observers."""

    ok: bool
    text: str
    data: Mapping[str, object]
    citations: tuple[str, ...]
    error: str


def _freeze_observation(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool, Enum)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_observation(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_observation(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_observation(item) for item in value)
    return f"<{type(value).__name__}>"


def _invocation_result(result: CapabilityResult) -> CapabilityInvocationResult:
    try:
        frozen = _freeze_observation(result.data)
        if not isinstance(frozen, Mapping):
            raise TypeError("capability result data is not a mapping")
    except Exception as exc:
        # Observation is best-effort and must never change provider semantics,
        # including for cyclic or custom mapping values.
        frozen = MappingProxyType(
            {"observation_error": f"<{type(exc).__name__}>"}
        )
    return CapabilityInvocationResult(
        ok=result.ok,
        text=result.text,
        data=frozen,
        citations=tuple(result.citations),
        error=result.error,
    )


@dataclass(frozen=True)
class CapabilityInvocation:
    """Opt-in observation of one registry invocation boundary.

    The registry does not log or retain these records.  Evaluation and
    diagnostics callers may subscribe explicitly when they need an exact tool
    trajectory, including nested ReAct calls.  ``context`` is deliberately not
    exposed: it can contain cancellation primitives and private runtime state.
    Observer failures are isolated from capability execution.
    """

    invocation_id: int
    phase: Literal["started", "finished"]
    name: str
    query: str
    task_id: str
    planner_tool: bool
    timestamp: float
    monotonic: float
    result: CapabilityInvocationResult | None = None


CapabilityInvocationObserver = Callable[[CapabilityInvocation], None]


class CapabilityRegistry:
    def __init__(self):
        self._providers: dict[str, CapabilityProvider] = {}
        self._specs: dict[str, CapabilitySpec] = {}
        self._invocation_lock = Lock()
        self._invocation_sequence = 0
        self._invocation_observers: list[CapabilityInvocationObserver] = []

    def observe_invocations(
        self,
        observer: CapabilityInvocationObserver,
    ) -> Callable[[], None]:
        """Subscribe to invocation boundaries and return an idempotent remover.

        Observation is strictly opt-in.  Callbacks run outside the registry
        lock, and a callback exception cannot fail or alter the capability.
        """

        with self._invocation_lock:
            self._invocation_observers.append(observer)

        def remove() -> None:
            with self._invocation_lock:
                try:
                    self._invocation_observers.remove(observer)
                except ValueError:
                    pass

        return remove

    def _next_invocation_id(self) -> int:
        with self._invocation_lock:
            self._invocation_sequence += 1
            return self._invocation_sequence

    def _invocation_observer_snapshot(self) -> tuple[CapabilityInvocationObserver, ...]:
        with self._invocation_lock:
            return tuple(self._invocation_observers)

    @staticmethod
    def _notify_invocation(
        event: CapabilityInvocation,
        observers: tuple[CapabilityInvocationObserver, ...],
    ) -> None:
        for observer in observers:
            try:
                observer(event)
            except Exception:
                # Diagnostics must never become an execution dependency.
                continue

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

    @staticmethod
    def _authorized(
        spec: Optional[CapabilitySpec], context: Mapping[str, object]
    ) -> CapabilityResult | None:
        """Return a refusal for an unauthorized action, otherwise ``None``.

        This is the common provider boundary, so direct registry calls and both
        planner implementations share the same fail-closed rule. Read tools and
        legacy specs (``authority=none``) remain byte-for-byte unaffected.
        """
        if spec is None or spec.authority == "none":
            return None
        origin = context.get("origin")
        if isinstance(origin, Origin):
            origin_value: str | None = origin.value
        elif isinstance(origin, str):
            origin_value = origin
        else:
            # Objects that merely expose ``value = 'live_audio'`` are not a
            # provenance tag. Accept only the concrete enum or serialized str.
            origin_value = None
        direct = context.get("direct_user_instruction") is True
        confirmed = context.get("confirmed") is True
        owner = context.get("owner_verified") is True
        allowed = origin_value == "live_audio" and direct
        if spec.authority == "verified_owner":
            allowed = allowed and owner
        if spec.requires_confirmation:
            allowed = allowed and confirmed
        if allowed:
            return None
        return CapabilityResult(
            True,
            "I can't perform that action from this request.",
            data={"executed": False, "blocked": "action_authority"},
        )

    def _call_provider(
        self,
        name: str,
        provider: Optional[CapabilityProvider],
        query: str,
        context: dict[str, object],
    ) -> CapabilityResult:
        if provider is None:
            return CapabilityResult(False, "", error=f"missing capability: {name}")
        refusal = self._authorized(self._specs.get(name), context)
        if refusal is not None:
            return refusal
        try:
            return provider(query, context)
        except Exception as exc:
            return CapabilityResult(False, "", error=str(exc))

    def invoke(self, name: str, query: str, context: dict[str, object] | None = None) -> CapabilityResult:
        context = context or {}
        provider = self._providers.get(name)
        # The common production path stays timestamp/sequence/lock free.
        # A subscriber added concurrently begins with the next invocation.
        if not self._invocation_observers:
            return self._call_provider(name, provider, query, context)

        observers = self._invocation_observer_snapshot()
        if not observers:
            return self._call_provider(name, provider, query, context)

        invocation_id = self._next_invocation_id()
        spec = self._specs.get(name)
        common = {
            "invocation_id": invocation_id,
            "name": name,
            "query": query,
            "task_id": str(context.get("task_id", "") or ""),
            "planner_tool": bool(spec is not None and spec.planner_tool),
        }
        self._notify_invocation(
            CapabilityInvocation(
                phase="started",
                timestamp=time.time(),
                monotonic=time.monotonic(),
                **common,
            ),
            observers,
        )
        result = self._call_provider(name, provider, query, context)
        self._notify_invocation(
            CapabilityInvocation(
                phase="finished",
                timestamp=time.time(),
                monotonic=time.monotonic(),
                result=_invocation_result(result),
                **common,
            ),
            observers,
        )
        return result

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def spec(self, name: str) -> Optional[CapabilitySpec]:
        return self._specs.get(name)

    def manifest(self) -> tuple[CapabilitySpec, ...]:
        """All registered specs, name-sorted (stable for prompts/logs)."""
        return tuple(self._specs[name] for name in sorted(self._specs))

    def planner_tools(self) -> tuple[str, ...]:
        """Read-only capability names the ReAct planner may call."""
        return tuple(
            spec.name
            for spec in self.manifest()
            if (
                spec.planner_tool
                and not spec.side_effecting
                and spec.authority == "none"
            )
        )

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
