from __future__ import annotations

import time
from typing import Optional

from always_on_agent.capabilities import create_default_capabilities
from always_on_agent.event_bus import EventBus
from always_on_agent.events import AgentEvent, EventKind, Mode
from always_on_agent.followups import FollowupConfig
from always_on_agent.memory import SessionMemory
from always_on_agent.react import PlannerConfig, attach_react_capability, should_escalate
from always_on_agent.supervisor import AgentSupervisor

from .capabilities import attach_llm_capabilities
from .engine import AudioEngine, EngineCallbacks
from .llm import EchoLLM, LLMClient
from .routing import Router


class VoiceRuntime:
    """Thin orchestrator: ``AudioEngine`` <-> ``AgentSupervisor`` <-> TTS.

    Replaces the ``main.py`` ``VoiceAssistant`` monolith. It owns no DSP and no
    model code. Responsibilities:

    - feed engine transcripts onto the brain's event bus,
    - play ``TTS_REQUEST`` events the brain emits,
    - on barge-in, stop playback and cancel in-flight work.

    The brain (modes, intent decisions, planner, cancellable tasks) is reused
    unchanged from ``always_on_agent``.
    """

    def __init__(
        self,
        engine: AudioEngine,
        llm: Optional[LLMClient] = None,
        *,
        fast_llm: Optional[LLMClient] = None,
        start_mode: Mode = Mode.ASSISTANT,
        agent_config=None,
        router: Optional[Router] = None,
        planner_config: Optional[PlannerConfig] = None,
        stream_tts: bool = False,
        followup_config: Optional[FollowupConfig] = None,
        command_map: Optional[dict[str, str]] = None,
    ):
        self.engine = engine
        # Command fast-path policy: maps a spotted keyword phrase to a control
        # action ("stop", "confirm", "deny", or "mode:<name>"). Keys are matched
        # case-insensitively. Empty -> unmapped keywords fall back to the normal
        # transcript path so nothing is silently dropped.
        self._command_map = {k.strip().lower(): v for k, v in (command_map or {}).items()}
        self.bus = EventBus()
        memory = SessionMemory()
        llm = llm or EchoLLM()
        registry = create_default_capabilities(memory)
        planner_on = planner_config is not None and planner_config.enabled
        escalate = should_escalate if (planner_on and planner_config.escalate) else None
        attach_llm_capabilities(
            registry, llm, fast_llm=fast_llm, router=router, escalate=escalate
        )
        if planner_on:
            attach_react_capability(registry, llm, config=planner_config)
        if agent_config is not None:
            # Opt-in: route command-mode through the Open Interpreter action brain.
            from .agent import attach_agent_capability

            attach_agent_capability(registry, agent_config)
        self.supervisor = AgentSupervisor(
            bus=self.bus,
            capabilities=registry,
            memory=memory,
            stream_tts=stream_tts,
            followup_config=followup_config,
        )
        self.supervisor.state.mode = start_mode
        self.bus.subscribe(self._on_event)
        self._bus_threaded = False

    # --- lifecycle ---
    def start(self, *, run_bus: bool = True) -> None:
        """Start the engine. ``run_bus=True`` runs the event loop on a
        background thread (production). Tests pass ``run_bus=False`` and pump
        the bus via :meth:`wait_idle`."""
        self.engine.start(
            EngineCallbacks(
                on_partial=self._on_partial,
                on_final=self._on_final,
                on_barge_in=self._on_barge_in,
                on_command=self._on_command,
            )
        )
        if run_bus:
            self.bus.start()
            self._bus_threaded = True

    def stop(self) -> None:
        self.supervisor.shutdown()
        if self._bus_threaded:
            self.bus.stop()
            self._bus_threaded = False
        self.engine.stop()

    @property
    def mode(self) -> Mode:
        return self.supervisor.state.mode

    # --- engine callbacks (may run on an audio thread) ---
    def _on_partial(self, text: str) -> None:
        self.bus.publish(AgentEvent.partial(text))

    def _on_final(self, text: str) -> None:
        self.bus.publish(AgentEvent.final(text))

    def _on_barge_in(self) -> None:
        # User spoke over the assistant: cut playback now, cancel in-flight work.
        self.engine.stop_speaking()
        self.bus.publish(AgentEvent.stop("barge_in"))

    def _on_command(self, keyword: str) -> None:
        # Spotted control phrase: act immediately, bypassing analyzer + LLM.
        action = self._command_map.get(keyword.strip().lower())
        if action is None:
            # Unmapped keyword: don't drop it -- hand it to the normal path.
            self.bus.publish(AgentEvent.final(keyword))
            return
        if action == "stop":
            self.engine.stop_speaking()
            self.bus.publish(AgentEvent.stop("command"))
        elif action == "confirm":
            self.bus.publish(AgentEvent.confirm("command"))
        elif action == "deny":
            self.bus.publish(AgentEvent.deny("command"))
        elif action.startswith("mode:"):
            try:
                self.bus.publish(AgentEvent.mode(Mode(action.split(":", 1)[1]), source="command"))
            except ValueError:
                pass

    # --- bus subscriber ---
    def _on_event(self, event: AgentEvent) -> None:
        if event.kind == EventKind.TTS_REQUEST:
            text = str(event.payload.get("text", "")).strip()
            if text:
                self.engine.speak(text)
        elif event.kind == EventKind.CONTROL_STOP:
            self.engine.stop_speaking()

    # --- synchronous helper for tests / console demo ---
    def wait_idle(self, timeout: float = 3.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.bus.drain()
            if not self.supervisor.state.active_tasks and not self.supervisor.state.queued_tasks:
                self.bus.drain()
                return True
            time.sleep(0.005)
        self.bus.drain()
        return (
            not self.supervisor.state.active_tasks and not self.supervisor.state.queued_tasks
        )
