from __future__ import annotations

import logging
import time
from threading import Event, Thread
from typing import Callable, Mapping, Optional

from always_on_agent.capabilities import create_default_capabilities
from always_on_agent.continuation import ContinuationConfig
from always_on_agent.event_bus import EventBus
from always_on_agent.events import AgentEvent, EventKind, Mode
from always_on_agent.followups import FollowupConfig
from always_on_agent.memory import Memory, SessionMemory
from always_on_agent.react import PlannerConfig, attach_react_capability, should_escalate
from always_on_agent.supervisor import AgentSupervisor

from .addressing import ACT, INGEST, UNSURE, AddressingClassifier
from .capabilities import RecallConfig, _answers_locally, attach_llm_capabilities
from .capability_router import CapabilityRouter, CapabilityTierRouter, escalate_predicate
from .conversation import RecentContextConfig
from .cleanup import TranscriptCleaner
from .contract import is_stop_command, normalize_command
from .engine import AudioEngine, EngineCallbacks
from .intents import LocalIntentHandler
from .llm import EchoLLM, LLMClient
from .metrics import ASR_FINAL, BARGE_IN, LLM_FIRST_TOKEN, MetricsRecorder
from .persona import PersonaConfig, build_system_prompt
from .routing import Router
from .watchdog import StuckWatchdog
from .websearch import WebSearchConfig, attach_web_search_capability

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
        memory: Optional[Memory] = None,
        recall_config: Optional[RecallConfig] = None,
        recent_context_config: Optional[RecentContextConfig] = None,
        web_search_config: Optional[WebSearchConfig] = None,
        start_mode: Mode = Mode.ASSISTANT,
        agent_config=None,
        router: Optional[Router] = None,
        capability_router: Optional[CapabilityRouter] = None,
        planner_config: Optional[PlannerConfig] = None,
        stream_tts: bool = False,
        followup_config: Optional[FollowupConfig] = None,
        continuation_config: Optional[ContinuationConfig] = None,
        command_map: Optional[dict[str, str]] = None,
        intents: Optional[LocalIntentHandler] = None,
        addressing: Optional[AddressingClassifier] = None,
        unsure_acts: bool = True,
        cleaner: Optional[TranscriptCleaner] = None,
        live_routing: bool = False,
        load_snapshot: Optional[Callable[[], Optional[float]]] = None,
        warm_on_start: bool = False,
        persona: Optional[PersonaConfig] = None,
        task_timeouts: Optional[Mapping[str, float]] = None,
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
        # One Memory instance shared by the capability recall, the default
        # capabilities corpus, and the supervisor. Desktop defaults to in-RAM
        # unless app.py built a Postgres-backed adapter and passed it in.
        self.memory = memory or SessionMemory()
        memory = self.memory
        llm = llm or EchoLLM()
        registry = create_default_capabilities(memory)
        # Register the pluggable web.search provider on top of the corpus
        # search.local (left intact as the offline fallback). The provider
        # routes every query through the §9.7 egress gate before any network
        # call; a missing/disabled config builds a corpus-only provider with no
        # httpx dependency. See core/websearch.py.
        web_cfg = web_search_config or WebSearchConfig()
        attach_web_search_capability(registry, web_cfg)
        planner_on = planner_config is not None and planner_config.enabled
        escalate = should_escalate if (planner_on and planner_config.escalate) else None
        # Unified capability router (the "middle layer"): when configured it BACKS
        # both the tier choice and the escalate decision, so one coherent module
        # decides simple/research/act + fast/main. Absent -> the existing per-gate
        # routing (the ``router`` arg + ``should_escalate``) stands, byte-identical.
        self._capability_router = capability_router
        if capability_router is not None:
            router = CapabilityTierRouter(capability_router)
            if planner_on and planner_config.escalate:
                escalate = escalate_predicate(capability_router)
        # Capability-aware system prompt: the answering model is told who it is
        # (the configured persona) and what skills it ACTUALLY has, enumerated
        # from the live capability manifest, with a web-access line that reflects
        # the real §9.7 egress state instead of a hardcoded denial. Built from the
        # registry's user-facing specs (all registered by create_default_*), so it
        # can never drift from the real providers.
        # web_enabled reflects REAL backend availability (the same condition the
        # web.search provider uses to actually go to the network), not just the
        # enabled flag -- so the model isn't told it can search the web when the
        # provider would silently fall back to the local corpus.
        web_enabled = bool(getattr(web_cfg, "enabled", False) and getattr(web_cfg, "base_url", ""))
        system_prompt = build_system_prompt(registry, persona=persona, web_enabled=web_enabled)
        # Kept so the startup pre-warm prefills the model's cacheable system
        # prefix with the REAL prompt (not a throwaway "hi" with no system), so
        # turn 1's first token isn't paying to fill a cold KV-cache prefix.
        self._system_prompt = system_prompt
        attach_llm_capabilities(
            registry, llm, fast_llm=fast_llm, system=system_prompt, router=router,
            escalate=escalate, recorder=self.metrics, memory=memory, recall=recall_config,
            recent_context=recent_context_config,
            live_routing=live_routing, load_snapshot=load_snapshot,
        )
        if planner_on:
            # Thread the persona name (a plain string -- no core import in the
            # brain) so an escalated ReAct turn's final answer keeps the persona,
            # and a first-token hook so an escalated turn stamps LLM_FIRST_TOKEN
            # (B3 -- otherwise the watchdog false-flags it as "llm stuck").
            attach_react_capability(
                registry, llm, config=planner_config,
                persona_name=persona.name if persona is not None else "",
                first_token_hook=lambda: self.metrics.mark(LLM_FIRST_TOKEN),
            )
        if agent_config is not None:
            # Opt-in: route command-mode through the Open Interpreter action brain.
            from .agent import attach_agent_capability

            attach_agent_capability(registry, agent_config)
        # Startup reconciliation: log the capability manifest once and warn if the
        # planner is configured with a tool that isn't actually registered (the
        # drift the manifest is meant to prevent).
        self._reconcile_capabilities(registry, planner_config if planner_on else None)
        self.supervisor = AgentSupervisor(
            bus=self.bus,
            capabilities=registry,
            memory=memory,
            stream_tts=stream_tts,
            followup_config=followup_config,
            continuation_config=continuation_config,
            task_timeouts=task_timeouts,
        )
        self.supervisor.state.mode = start_mode
        self.bus.subscribe(self._on_event)
        self._bus_threaded = False
        # Live watchdog: warns when a turn stalls or the barge-in gate flaps.
        # See core.watchdog for the heuristics; WARNINGs land in the run
        # bundle so the next stuck reproduction leaves visible evidence.
        # Wire the storm hook so a detected barge-in / TTS-echo storm briefly
        # debounces the engine's barge-in gate (realtime-concurrency-5). Engines
        # without the hook (scripted/livekit) pass None and the watchdog just
        # logs the diagnosis as before.
        # on_tick drives the supervisor's overdue-task reap on the watchdog's
        # existing 1 s cadence, so a hung task is killed (the controller "heals")
        # rather than merely diagnosed.
        self._watchdog = StuckWatchdog(
            self.metrics,
            on_storm=getattr(self.engine, "note_barge_in_storm", None),
            on_tick=self._on_watchdog_tick,
        )
        # Optional startup pre-warm (lat-2): without it, turn 1 pays the model
        # cold-load (~3s for a GPU Gemma, measured ttft 3.17s vs ~0.27s warm) on
        # the user's first utterance. When enabled, start() kicks a *background*
        # warm-up of the answering models (and the engine, if it exposes
        # ``warm()``) so the load is paid before the user speaks. Off by default
        # so library/test construction is byte-identical; the CLI opts in via
        # ``config.warm_on_start``. ``fast_llm`` is warmed first (it answers the
        # common case); duplicates are collapsed by identity so a collapsed
        # fast/main pair is only warmed once.
        #
        # §9.7 gate: warm ONLY purely-local tiers. ``_answers_locally`` is the
        # canonical "this model cannot reach cloud" predicate (False for a
        # cloud-backed HedgeLLM / SensitivityRouterLLM), so a throwaway warm-up
        # "hi" can never fire a billed cloud completion before the user has
        # invoked anything -- cloud egress still happens only on a real turn.
        self._warm_on_start = warm_on_start
        # Readiness signal: set once the background warm-up has paid the model +
        # engine cold-start costs (or immediately when warm-up is off). A
        # programmatic handle a caller/test can block on until "ready to fire".
        self.warm_ready = Event()
        warm_models: list[LLMClient] = []

        def _add_warm(candidate: Optional[LLMClient]) -> None:
            if candidate is None:
                return
            if _answers_locally(candidate):
                if all(candidate is not m for m in warm_models):
                    warm_models.append(candidate)
            else:
                # A cloud-backed model (HedgeLLM / SensitivityRouterLLM): warming
                # the whole thing could fire a billed cloud completion before any
                # real turn, so we don't. But its purely-LOCAL leg can be warmed
                # safely (§9.7: no egress), so the local tier of a cloud-hybrid
                # isn't left cold. Best-effort: only if it exposes a `.local`.
                local = getattr(candidate, "local", None)
                if local is not None and _answers_locally(local):
                    if all(local is not m for m in warm_models):
                        warm_models.append(local)

        for model in (fast_llm, llm):
            _add_warm(model)
        self._warm_models = warm_models

    def _reconcile_capabilities(self, registry, planner_config) -> None:
        """Log the capability manifest once and warn on planner-tool drift.

        Single source of truth: the controller and the model both reason over the
        registry manifest, so a configured planner tool that isn't actually
        registered (the drift the manifest exists to kill) is surfaced loudly
        rather than failing silently mid-plan."""
        manifest = getattr(registry, "manifest", None)
        if callable(manifest):
            specs = manifest()
            log.info(
                "capabilities (%d): %s",
                len(specs), ", ".join(spec.name for spec in specs),
            )
        if planner_config is not None:
            registered = set(registry.names())
            missing = [t for t in getattr(planner_config, "tools", ()) if t not in registered]
            if missing:
                log.warning(
                    "planner configured with unregistered tool(s) %s -- they will be "
                    "reported unavailable to the planner", ", ".join(missing),
                )

    def _on_watchdog_tick(self) -> None:
        """Periodic maintenance on the watchdog's cadence: reap hung tasks so the
        controller never stays blocked on a capability that won't return."""
        try:
            self.supervisor.reap_overdue_tasks()
        except Exception:  # noqa: BLE001 - maintenance must never kill the watchdog
            log.exception("overdue-task reap failed")

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
                on_capture_state=self._on_capture_state,
            )
        )
        if run_bus:
            self.bus.start()
            self._bus_threaded = True
        self._watchdog.start()
        if self._warm_on_start:
            # Background daemon so startup never blocks on a model load: by the
            # time the user finishes their first utterance the model is resident.
            Thread(target=self._warm, name="speaker-warm", daemon=True).start()
        else:
            # Nothing to warm -> ready immediately, so a waiter never blocks.
            self.warm_ready.set()

    def _warm(self) -> None:
        """Pre-load the answering models, the gate/cleaner, and the engine so
        turn 1 isn't cold; then raise ``warm_ready``.

        Best-effort: a warm-up failure (model not pulled yet, server down) must
        never take down the live pipeline -- it just means turn 1 pays the cold
        cost as before. The model warm uses the REAL system prompt so the
        cacheable system prefix is prefilled (not a bare ``hi`` that leaves the
        prefix cold), and the input gate / cleaner are exercised once so their
        first live classify/clean isn't cold either."""
        try:
            for model in self._warm_models:
                try:
                    model.generate("hi", system=self._system_prompt)
                except TypeError:
                    # A minimal LLM stub may not accept system=; fall back.
                    try:
                        model.generate("hi")
                    except Exception:  # noqa: BLE001 - warm-up is best-effort
                        log.debug("warm-up failed for %s", type(model).__name__, exc_info=True)
                except Exception:  # noqa: BLE001 - warm-up is best-effort
                    log.debug("warm-up failed for %s", type(model).__name__, exc_info=True)
            # Warm the pre-brain gate + cleaner (they use the fast tier with a
            # different system prefix, so they have their own cold cost).
            if self._addressing is not None:
                try:
                    self._addressing.classify("hi", recent=())
                except Exception:  # noqa: BLE001 - best-effort
                    log.debug("addressing warm-up failed", exc_info=True)
            if self._cleaner is not None:
                try:
                    self._cleaner.clean("hi", recent=())
                except Exception:  # noqa: BLE001 - best-effort
                    log.debug("cleaner warm-up failed", exc_info=True)
            engine_warm = getattr(self.engine, "warm", None)
            if callable(engine_warm):
                try:
                    engine_warm()
                except Exception:  # noqa: BLE001 - engine warm is best-effort
                    log.debug("engine warm-up failed", exc_info=True)
            log.info("startup pre-warm complete (%d model(s))", len(self._warm_models))
        finally:
            # Always signal readiness, even if a warm step failed -- "warm-up
            # finished" is the signal, not "warm-up perfect".
            self.warm_ready.set()

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
        # Flush + release the memory backend at close-time (R6). prune() runs
        # the close-time retention pass (a no-op stub in P2a); close() flushes
        # any pending writes and releases the pool. Guarded so a teardown error
        # never prevents the engine from stopping (the recording flush).
        try:
            self.memory.prune()
            self.memory.close()
        except Exception:  # noqa: BLE001
            log.exception("memory close failed")
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
                # A short add-on to a turn that's still in flight ("make it
                # shorter", "in spanish") reads as ambient to the addressing
                # gate, but it IS addressed -- let a genuine continuation reach
                # the brain so it can merge/continue instead of being dropped.
                if self.supervisor.looks_like_continuation(text):
                    log.info("addressing override: continuation of in-flight turn")
                else:
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
        # Unified capability-router decision (the "middle layer"). It DRIVES
        # behaviour via the tier router + escalate predicate wired in __init__;
        # this call records the decision for the run summary AND primes the
        # fast-LLM action cache, so the in-capability escalate/tier consults this
        # turn don't re-call the model. Best-effort: routing never breaks a turn.
        if self._capability_router is not None:
            try:
                decision = self._capability_router.route(
                    final_text, {"mode": self.mode.value}
                )
                log.info(
                    "route: action=%s tier=%s conf=%.2f source=%s (%s) for %r",
                    decision.action, decision.tier, decision.confidence,
                    decision.source, decision.reason, final_text,
                    extra={"route": {
                        "action": decision.action, "tier": decision.tier,
                        "confidence": decision.confidence,
                        "source": decision.source, "reason": decision.reason,
                    }},
                )
                self.metrics.mark(f"route_{decision.action}")
            except Exception:  # noqa: BLE001 - routing observability is best-effort
                log.exception("capability router failed; default routing stands")
        # Try the no-LLM fast-path next; only fall through to the brain on a miss.
        if self._intents is not None and self._intents.handle(final_text):
            log.debug("handled by intent fast-path: %r", final_text)
            return
        self.bus.publish(AgentEvent.final(final_text))

    def _on_barge_in(self) -> None:
        # User spoke over the assistant: cancel in-flight work, then cut playback.
        # Cancellation MUST be set before stop_speaking() returns so the
        # streaming emitter stops producing sentences and any TTS_REQUEST still
        # in flight is dropped by _on_event -- otherwise a stale sentence from
        # the interrupted turn could be spoken after the barge-in
        # (realtime-concurrency-1). cancel_all() only sets cancel_events under a
        # short-lived lock; it never blocks on an engine call or a thread join,
        # so it is safe to run here on the audio thread before stop_speaking().
        self.metrics.mark(BARGE_IN)
        self._watchdog.note_barge_in()
        self.supervisor.cancel_all()
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
            # Same determinism as barge-in: set cancellation before playback is
            # cut so no stale sentence from the interrupted turn is spoken
            # (realtime-concurrency-1).
            self.supervisor.cancel_all()
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

    def _on_capture_state(self, state: str, message: str) -> None:
        """Engine reports a change in the capture stream's lifecycle.

        Three forks: ``"open"`` means we're capturing audio normally;
        ``"recovering"`` means the engine hit a PortAudio error and is
        retrying with backoff; ``"fatal"`` means recovery exhausted and
        the capture loop will not produce more audio.

        Publishes a ``CAPTURE_STATE`` :class:`AgentEvent` so the brain
        can react (today: just log + tell the watchdog; future: spoken
        feedback "reconnecting microphone"), and tells the watchdog so
        it skips the false "audio thread stalled" warning during a
        legitimate reopen."""
        self._watchdog.note_capture_state(state, message)
        log.info("capture state: %s (%s)", state, message)
        self.bus.publish(
            AgentEvent(
                EventKind.CAPTURE_STATE,
                {"state": state, "message": message},
                priority=30,
            )
        )

    # --- bus subscriber ---
    def _on_event(self, event: AgentEvent) -> None:
        if event.kind == EventKind.TTS_REQUEST:
            text = str(event.payload.get("text", "")).strip()
            if not text:
                return
            # Deterministic barge-in (realtime-concurrency-1): a sentence that
            # was queued or emitted just before an interrupt belongs to a turn
            # the user has cancelled. Drop it so no stale sentence is spoken
            # after stop_speaking(). The supervisor's epoch/active-task check is
            # the single source of truth -- cancellation is set there before the
            # barge-in path returns (see _on_barge_in / _on_command).
            task_id = str(event.payload.get("task_id", ""))
            epoch = event.payload.get("epoch")
            if not self.supervisor.tts_request_allowed(task_id, epoch):
                log.debug("dropping stale TTS_REQUEST (task_id=%r): %r", task_id, text)
                return
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
