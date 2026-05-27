from __future__ import annotations

import logging
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
from .contract import is_stop_command, normalize_command
from .engine import AudioEngine, EngineCallbacks
from .intents import LocalIntentHandler
from .llm import EchoLLM, LLMClient
from .metrics import ASR_FINAL, BARGE_IN, MetricsRecorder
from .routing import Router

log = logging.getLogger("speaker.runtime")


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
        intents: Optional[LocalIntentHandler] = None,
    ):
        self.engine = engine
        # Optional deterministic speech-to-intent fast-path. When present it
        # answers frequent commands directly (no LLM); a miss falls through to
        # the brain. ``None`` -> disabled (the default, preserving the bare
        # transcript->brain path).
        self._intents = intents
        # Command fast-path policy: maps a spotted keyword phrase to a control
        # action ("stop", "confirm", "deny", or "mode:<name>"). Keys are matched
        # case-insensitively. Empty -> unmapped keywords fall back to the normal
        # transcript path so nothing is silently dropped.
        self._command_map = {normalize_command(k): v for k, v in (command_map or {}).items()}
        self.bus = EventBus()
        # Per-turn latency recorder, fed by this runtime (asr_final, barge_in),
        # the engine (speech_end, tts_first_audio, barge_in_stop via on_metric),
        # and the LLM capability (llm_first_token). Read via ``runtime.metrics``.
        self.metrics = MetricsRecorder()
        memory = SessionMemory()
        llm = llm or EchoLLM()
        registry = create_default_capabilities(memory)
        planner_on = planner_config is not None and planner_config.enabled
        escalate = should_escalate if (planner_on and planner_config.escalate) else None
        attach_llm_capabilities(
            registry, llm, fast_llm=fast_llm, router=router, escalate=escalate,
            recorder=self.metrics,
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
                on_metric=self.metrics.mark,
            )
        )
        if run_bus:
            self.bus.start()
            self._bus_threaded = True

    def stop(self) -> None:
        # Guard each step so a teardown error never prevents the engine from
        # stopping -- that is what flushes the session recording to disk.
        try:
            self.supervisor.shutdown()
        except Exception:  # noqa: BLE001
            log.exception("supervisor shutdown failed")
        if self._bus_threaded:
            try:
                self.bus.stop()
            except Exception:  # noqa: BLE001
                log.exception("bus stop failed")
            self._bus_threaded = False
        self.engine.stop()

    @property
    def mode(self) -> Mode:
        return self.supervisor.state.mode

    # --- engine callbacks (may run on an audio thread) ---
    def _on_partial(self, text: str) -> None:
        self.bus.publish(AgentEvent.partial(text))

    def _on_final(self, text: str) -> None:
        self.metrics.mark(ASR_FINAL)
        log.info(
            "final -> brain: %r (mode=%s)", text, self.mode.value,
            extra={"transcript": {"role": "user", "text": text, "mode": self.mode.value}},
        )
        # Try the no-LLM fast-path first; only fall through to the brain on a miss.
        if self._intents is not None and self._intents.handle(text):
            log.debug("handled by intent fast-path: %r", text)
            return
        self.bus.publish(AgentEvent.final(text))

    def _on_barge_in(self) -> None:
        # User spoke over the assistant: cut playback now, cancel in-flight work.
        self.metrics.mark(BARGE_IN)
        self.engine.stop_speaking()
        self.bus.publish(AgentEvent.stop("barge_in"))

    def _on_command(self, keyword: str) -> None:
        # Spotted control phrase: act immediately, bypassing analyzer + LLM.
        # Normalization and the stop-class fall back to the shared contract so
        # the desktop and mobile shells recognize the same control phrases.
        action = self._command_map.get(normalize_command(keyword))
        if action is None and is_stop_command(keyword):
            action = "stop"
        if action is None:
            # Unmapped keyword: try the intent fast-path, else the normal path.
            if self._intents is not None and self._intents.handle(keyword):
                return
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
                log.info(
                    "assistant: %r", text,
                    extra={"transcript": {"role": "assistant", "text": text}},
                )
                self.engine.speak(text)
        elif event.kind == EventKind.CONTROL_STOP:
            self.engine.stop_speaking()
            # Stop also cancels pending fast-path actions (e.g. a running timer).
            if self._intents is not None:
                self._intents.cancel_all()

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
