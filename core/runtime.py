from __future__ import annotations

import time
from typing import Optional

from always_on_agent.capabilities import create_default_capabilities
from always_on_agent.event_bus import EventBus
from always_on_agent.events import AgentEvent, EventKind, Mode
from always_on_agent.memory import SessionMemory
from always_on_agent.supervisor import AgentSupervisor

from .capabilities import attach_llm_capabilities
from .engine import AudioEngine, EngineCallbacks
from .llm import EchoLLM, LLMClient


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
        start_mode: Mode = Mode.ASSISTANT,
    ):
        self.engine = engine
        self.bus = EventBus()
        memory = SessionMemory()
        registry = create_default_capabilities(memory)
        attach_llm_capabilities(registry, llm or EchoLLM())
        self.supervisor = AgentSupervisor(bus=self.bus, capabilities=registry, memory=memory)
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
            )
        )
        if run_bus:
            self.bus.start()
            self._bus_threaded = True

    def stop(self) -> None:
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
