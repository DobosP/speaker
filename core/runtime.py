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

from .addressing import ACT, INGEST, UNSURE, AddressingClassifier
from .capabilities import attach_llm_capabilities
from .cleanup import TranscriptCleaner
from .contract import is_stop_command, normalize_command
from .engine import AudioEngine, EngineCallbacks
from .intents import LocalIntentHandler
from .llm import EchoLLM, LLMClient
from .metrics import ASR_FINAL, BARGE_IN, MetricsRecorder
from .routing import Router
from .watchdog import StuckWatchdog

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
        addressing: Optional[AddressingClassifier] = None,
        unsure_acts: bool = True,
        cleaner: Optional[TranscriptCleaner] = None,
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
        # Input gate (optional). When set, every ASR final is classified before
        # reaching the brain so background speech / read-aloud quotations don't
        # trigger replies. See core/addressing.py and docs/target_architecture.md
        # §9.8. ``unsure_acts`` decides what to do with UNSURE -- True (default)
        # preserves prior behavior (respond on ambiguity); False is conservative.
        self._addressing = addressing
        self._unsure_acts = unsure_acts
        # Optional transcript cleaner. When present, every ACT'd final goes
        # through an LLM rewrite to drop disfluencies and resolve self-
        # corrections before the brain sees it. See core/cleanup.py.
        self._cleaner = cleaner
        self.bus = EventBus()
        # Per-turn latency recorder, fed by this runtime (asr_final, barge_in),
        # the engine (speech_end, tts_first_audio, barge_in_stop via on_metric),
        # and the LLM capability (llm_first_token). Read via ``runtime.metrics``.
        self.metrics = MetricsRecorder()
        self.memory = SessionMemory()
        memory = self.memory
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
        # Live watchdog: warns when a turn stalls or the barge-in gate flaps.
        # See core.watchdog for the heuristics; WARNINGs land in the run
        # bundle so the next stuck reproduction leaves visible evidence.
        self._watchdog = StuckWatchdog(self.metrics)

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
                on_heartbeat=self._watchdog.note_heartbeat,
            )
        )
        if run_bus:
            self.bus.start()
            self._bus_threaded = True
        self._watchdog.start()

    def stop(self) -> None:
        # Guard each step so a teardown error never prevents the engine from
        # stopping -- that is what flushes the session recording to disk.
        try:
            self._watchdog.stop()
        except Exception:  # noqa: BLE001
            log.exception("watchdog stop failed")
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
        log.info(
            "final -> brain: %r (mode=%s)", text, self.mode.value,
            extra={"transcript": {"role": "user", "text": text, "mode": self.mode.value}},
        )
        # Input gate: classify before opening a metrics turn so an INGEST'd
        # utterance doesn't trip the watchdog's "no llm_first_token" check.
        # When no classifier is configured this is a no-op (legacy behavior).
        if self._addressing is not None:
            recent = [item.text for item in self.memory.all()[-4:]]
            decision = self._addressing.classify(text, recent=recent)
            log.info("addressing decision: %s for %r", decision, text)
            if decision == INGEST or (decision == UNSURE and not self._unsure_acts):
                self.memory.add(text, tags=("ingested",))
                return
        self.metrics.mark(ASR_FINAL)
        # Cleanup pass: rewrite disfluencies / self-corrections so the brain
        # acts on what the user meant, not on every "um" + word-repeat. The
        # raw text was already logged above; emit a second transcript entry
        # with the cleaned version (and the raw retained in 'raw') so the
        # user can audit every rewrite in run-<id>.summary.json.
        final_text = text
        if self._cleaner is not None:
            recent_for_cleaner = [item.text for item in self.memory.all()[-4:]]
            try:
                cleaned = self._cleaner.clean(text, recent=recent_for_cleaner)
            except Exception:  # noqa: BLE001
                log.exception("transcript cleaner failed; passing raw text through")
                cleaned = text
            if cleaned and cleaned != text:
                log.info(
                    "cleaned: %r -> %r", text, cleaned,
                    extra={"transcript": {
                        "role": "user", "text": cleaned, "raw": text,
                        "mode": self.mode.value,
                    }},
                )
                final_text = cleaned
        # Try the no-LLM fast-path next; only fall through to the brain on a miss.
        if self._intents is not None and self._intents.handle(final_text):
            log.debug("handled by intent fast-path: %r", final_text)
            return
        self.bus.publish(AgentEvent.final(final_text))

    def _on_barge_in(self) -> None:
        # User spoke over the assistant: cut playback now, cancel in-flight work.
        self.metrics.mark(BARGE_IN)
        self._watchdog.note_barge_in()
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
