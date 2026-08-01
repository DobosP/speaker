from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from threading import Event, Thread
from typing import Callable, Mapping, Optional

from always_on_agent.capabilities import create_default_capabilities
from always_on_agent.acoustic import AcousticLineage
from always_on_agent.continuation import ContinuationConfig
from always_on_agent.event_bus import EventBus
from always_on_agent.events import AgentEvent, EventKind, Mode
from always_on_agent.followups import FollowupConfig
from always_on_agent.memory import Memory, SessionMemory
from always_on_agent.post_barge import PostBargeResponseGate
from always_on_agent.react import PlannerConfig, attach_react_capability, should_escalate
from always_on_agent.session_actor import PlaybackAdmission, TaskIdentity
from always_on_agent.speech_analyzer import (
    LiveSpeechAnalyzer,
    ModePolicy,
    is_assistant_mode_final_candidate,
    is_pending_confirmation_response,
)
from always_on_agent.supervisor import AgentSupervisor, ArrivalContinuation

from always_on_agent.text import normalize_text

from .addressing import ACT, INGEST, UNSURE, AddressingClassifier
from .capabilities import RecallConfig, _answers_locally, attach_llm_capabilities
from .capability_router import CapabilityRouter, CapabilityTierRouter, escalate_predicate
from .conversation import RecentContextConfig
from .cleanup import TranscriptCleaner, rewrite_is_overreach
from .contract import is_stop_command, normalize_command
from .engine import (
    AcousticSignal,
    AudioEngine,
    CommandDetection,
    EngineCallbacks,
    FinalTranscript,
    OwnerVerification,
    PartialTranscript,
    PlaybackOutcome,
    PlaybackReceipt,
    TrackedSpeech,
    TranscriptAbort,
)
from .intents import LocalIntentHandler
from .input_lineage import AcousticRevisionGate
from .llm import EchoLLM, LLMCallCancelled, LLMClient
from .metrics import (
    ASR_FINAL,
    BARGE_IN,
    HANDLED_LOCAL,
    HELD,
    LLM_FIRST_TOKEN,
    TTS_REQUESTED,
    MetricsRecorder,
)
from .obsidian import ObsidianConfig, attach_obsidian_capability
from .persona import PersonaConfig, build_system_prompt
from .reminders import (
    ReminderConfig,
    attach_reminder_capabilities,
    build_reminder_manager,
)
from .playback_history import (
    PlaybackCommit,
    PlaybackContext,
    PlaybackFinalization,
    PlaybackHistory,
)
from .tts_markup import (
    ReplySpeechId,
    ReplyVoiceContinuity,
    build_markup_guidance,
)
from .resume import ResumeConfig, ResumeTracker
from .routing import LatencyPolicy, Router, classify_latency_policy
from .turn_merge import FinalDispatcher, FinalDispatchLease, TurnMergeConfig
from .trusted_apps import (
    TrustedAppsConfig,
    attach_device_tool_command_dispatcher,
    attach_trusted_app_capability,
    build_trusted_app_manager,
)
from .watchdog import StuckWatchdog
from .websearch import WebSearchConfig, attach_web_search_capability

log = logging.getLogger("speaker.runtime")

_LLM_BACKED_TASK_CAPABILITIES = frozenset({"assistant.answer", "research.local"})
# Internal marker for the legacy callback that carries text but no typed origin.
# It is normalized back to ``unknown`` before any event leaves the runtime.
_LEGACY_TEXT_ORIGIN = "__legacy_text__"


@dataclass(frozen=True)
class _PublishedUnheardFinal:
    """One committed final that remains eligible for unheard rollback."""

    generation: int
    reservation: ArrivalContinuation
    event: AgentEvent
    claimed: Event = field(default_factory=Event, repr=False, compare=False)


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
        obsidian_config: Optional[ObsidianConfig] = None,
        reminder_config: Optional[ReminderConfig] = None,
        trusted_apps_config: Optional[TrustedAppsConfig] = None,
        start_mode: Mode = Mode.ASSISTANT,
        agent_config=None,
        computer_use_config=None,
        watch_config=None,
        router: Optional[Router] = None,
        capability_router: Optional[CapabilityRouter] = None,
        planner_config: Optional[PlannerConfig] = None,
        stream_tts: bool = False,
        followup_config: Optional[FollowupConfig] = None,
        continuation_config: Optional[ContinuationConfig] = None,
        turn_merge_config: Optional[TurnMergeConfig] = None,
        resume_config: Optional[ResumeConfig] = None,
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
        confirmation_ttl_sec: float = 180.0,
        admission_load: Optional[Callable[[], Optional[float]]] = None,
        post_barge_response_window_sec: float = 8.0,
        diagnostic_observer: Optional[Callable[[object], object]] = None,
        diagnostic_invalidator: Optional[Callable[[str], object]] = None,
    ):
        self.engine = engine
        # Optional private, transcript-free evidence sink.  It is deliberately
        # not an AgentEvent subscriber: playback receipts arrive on a priority
        # lifecycle lane and diagnostics must never add control-plane work.
        self._diagnostic_observer = diagnostic_observer
        self._diagnostic_invalidator = diagnostic_invalidator
        self._diagnostic_acoustic_by_input: dict[
            int, tuple[str, str, int]
        ] = {}
        self._diagnostic_acoustic_order: deque[int] = deque()
        # Playout receipts are opt-in.  Legacy engines retain the established
        # admission-time history path; a capable engine moves memory/resume/
        # follow-up truth to its terminal sink attestations.
        playback_capabilities = self.engine.playback_capabilities
        self._tracked_playback = bool(playback_capabilities.tracked_terminal)
        self._playback_history = PlaybackHistory() if self._tracked_playback else None
        _eng_cfg = getattr(self.engine, "config", None)
        self._tts_lock_speaker_id = bool(
            _eng_cfg is not None
            and getattr(_eng_cfg, "tts_lock_speaker_id", False)
        )
        self._reply_voice_continuity = (
            ReplyVoiceContinuity(
                voices=(getattr(_eng_cfg, "tts_speaker_voices", {}) or {}).keys(),
                emotions=(
                    getattr(_eng_cfg, "tts_emotion_speed_map", {}) or {}
                ).keys(),
                lock_speaker_id=self._tts_lock_speaker_id,
            )
            if (
                _eng_cfg is not None
                and getattr(_eng_cfg, "tts_markup", False)
                and playback_capabilities.tracked_terminal
                and playback_capabilities.speech_style_hints
            )
            else None
        )
        # Linearizes "ledger became terminal" with publication of the resulting
        # MEMORY_COMMIT.  Shutdown samples both facts under this same lock so it
        # cannot close the bus/memory in the tiny resolve->publish handoff gap.
        self._playback_effect_lock = threading.RLock()
        self._playback_effect_changed = threading.Condition(
            self._playback_effect_lock
        )
        self._pending_playback_memory_commits = 0
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
        self._stream_tts = bool(stream_tts)
        self.bus = EventBus()
        # Set by stop() before any teardown step; _on_event drops
        # action-producing events (TTS_REQUEST) once true.
        self._stopping = False
        # Serialize the complete teardown, not only the initial effect gate.
        # A concurrent caller waits until capability drains and backend closes
        # have both completed; re-entry by the owning thread is a no-op.
        self._stop_lifecycle = threading.Condition()
        self._stop_started = False
        self._stop_complete = False
        self._stop_owner_ident: int | None = None
        self._stop_waiters = 0
        # Linearizes the short post-preprocessing commit section against
        # barge-in/stop. It is never held across an LLM/provider call; local
        # intents and bus publication are the only potentially visible effects.
        self._terminal_effect_lock = threading.RLock()
        self._acoustic_revisions = AcousticRevisionGate()
        # A confirmed acoustic barge is conversational provenance, not speaker
        # identity.  Keep exactly one bounded response-only grant across the
        # final merger; it can never authorize controls, tools, or actions.
        self._post_barge_response = PostBargeResponseGate(
            post_barge_response_window_sec
        )
        self._input_generation_lock = threading.Lock()
        self._next_input_generation = 0
        self._arrival_superseded_generations: set[int] = set()
        self._arrival_continuations: dict[
            int, tuple[ArrivalContinuation, bool]
        ] = {}
        # Finals that have committed and entered the bus but do not yet have an
        # AgentTask.  A new partial can reserve this otherwise invisible lineage
        # while its generation fence prevents the old bus event from starting.
        self._published_unheard: dict[int, ArrivalContinuation] = {}
        self._published_unheard_finals: dict[
            int, _PublishedUnheardFinal
        ] = {}
        self._partial_fence_active = False
        self._partial_fence_generation: int | None = None
        self._partial_fence_acoustic_keys: frozenset[
            tuple[str, int, int, str]
        ] | None = None
        self._partial_fence_unheard: _PublishedUnheardFinal | None = None
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
        # Local Obsidian access is read-only, bounded, and registered only when
        # the configured vault exists. Attach it before building both the
        # capability-aware system prompt and ReAct planner so neither can claim
        # or offer a missing machine-local tool.
        vault_cfg = obsidian_config or ObsidianConfig()
        attach_obsidian_capability(registry, vault_cfg)
        self._vault_enabled = "vault.search" in registry.names()
        # Optional device functions live in the same capability registry as
        # chat, web, and vault reads.  Setup decides which providers exist;
        # there is no separate reminder/vault/app chatbot or live entry point.
        self._reminder_manager = build_reminder_manager(
            reminder_config or ReminderConfig()
        )
        if self._reminder_manager is not None:
            attach_reminder_capabilities(registry, self._reminder_manager)
        self._trusted_app_manager = build_trusted_app_manager(
            trusted_apps_config or TrustedAppsConfig()
        )
        if self._trusted_app_manager is not None:
            attach_trusted_app_capability(registry, self._trusted_app_manager)
        self._device_tools_enabled = bool(
            self._reminder_manager is not None
            or self._trusted_app_manager is not None
        )
        if (
            planner_config is not None
            and self._vault_enabled
            and "vault.search" not in planner_config.tools
        ):
            # Preserve ADR-0033's verified first four phone-native tools: the
            # optional desktop vault tool is appended and therefore remains off
            # the four-schema MiniCPM seam unless that ADR is revisited.
            planner_config = replace(
                planner_config,
                tools=(*planner_config.tools, "vault.search"),
            )
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
        self._latency_router = router
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
        # Opt-in expressive-TTS markup: when the engine has tts_markup on, teach the
        # answering model the leading-tag grammar + its configured voices/emotions
        # so it can emit [emotion:.. voice:..] tags. Duck-typed off the engine
        # config (a sherpa-only feature), so a non-sherpa engine yields "" and the
        # prompt is unchanged. Off by default -> the model stays tag-unaware.
        markup_guidance = ""
        if _eng_cfg is not None and getattr(_eng_cfg, "tts_markup", False):
            markup_guidance = build_markup_guidance(
                voices=list(getattr(_eng_cfg, "tts_speaker_voices", {}) or {}),
                emotions=list(getattr(_eng_cfg, "tts_emotion_speed_map", {}) or {}),
                voice_continuity=self._reply_voice_continuity is not None,
                lock_speaker_id=self._tts_lock_speaker_id,
            )
        system_prompt = build_system_prompt(
            registry, persona=persona, web_enabled=web_enabled,
            markup_guidance=markup_guidance,
        )
        # Kept so the startup pre-warm prefills the model's cacheable system
        # prefix with the REAL prompt (not a throwaway "hi" with no system), so
        # turn 1's first token isn't paying to fill a cold KV-cache prefix.
        self._system_prompt = system_prompt
        # Visual context feed (machine -> model). A host process (screen grabber,
        # camera, app) sets the current frame via set_current_frame(); the latest
        # frame rides AMBIENT on every assistant turn so the model has visual
        # context of what the user is doing. None by default -> the image_provider
        # returns None and behaviour is byte-identical (text-only). Must be set
        # BEFORE attach_llm_capabilities so the bound provider sees it.
        self._frame_lock = threading.Lock()
        self._current_frame: Optional[object] = None
        attach_llm_capabilities(
            registry, llm, fast_llm=fast_llm, system=system_prompt, router=router,
            escalate=escalate, recorder=self.metrics, memory=memory, recall=recall_config,
            recent_context=recent_context_config,
            live_routing=live_routing, load_snapshot=load_snapshot,
            image_provider=self._current_images,
            before_conversation_read=self._await_playback_history,
        )
        if planner_on:
            # Thread the persona name (a plain string -- no core import in the
            # brain) so an escalated ReAct turn's final answer keeps the persona,
            # and a first-token hook so an escalated turn stamps LLM_FIRST_TOKEN
            # (B3 -- otherwise the watchdog false-flags it as "llm stuck").
            # Phone profiles may additionally opt their *bare local* MiniCPM
            # client into its verified native XML step backend. Desktop
            # Gemma/Ollama and every unconfigured model retain textual ReAct.
            from .minicpm_tools import build_minicpm_planner_backend

            attach_react_capability(
                registry, llm, config=planner_config,
                persona_name=persona.name if persona is not None else "",
                first_token_hook=lambda: self.metrics.mark(LLM_FIRST_TOKEN),
                step_backend=build_minicpm_planner_backend(llm),
            )
        if agent_config is not None:
            # Opt-in: route command-mode through the Open Interpreter action brain.
            from .agent import attach_agent_capability

            attach_agent_capability(registry, agent_config)
        # The typed, setup-approved dispatcher is a separate internal route for
        # exact reminder/app requests.  It never replaces the historical
        # command.stage action brain; unmatched commands keep their old path.
        self._device_tool_dispatcher = attach_device_tool_command_dispatcher(
            registry,
            reminder_manager=self._reminder_manager,
            trusted_app_manager=self._trusted_app_manager,
        )
        if computer_use_config is not None and computer_use_config.get("enabled"):
            # Opt-in (default OFF, separate from --agent): the READ-ONLY
            # screen.identify capability. NO actuating gui.* is registered here --
            # input control is a separately-gated slice (owner speaker-ID + the
            # always_on_agent.origin action chokepoint).
            from .ui_grounding import attach_computer_use_capability

            attach_computer_use_capability(
                registry,
                enabled=True,
                monitor=int(computer_use_config.get("monitor", 1) or 1),
            )
        # Watch/monitor capability (default OFF, separate gate from --agent and
        # computer-use). Owner-verified grants let the assistant watch SPECIFIC
        # granted app windows for a described event; it observes + speaks only,
        # never actuates, and the model can never arm one (planner_tool=False).
        # Grants are machine-local; capture is window-scoped (never full screen).
        self._watch_manager = None
        if watch_config and watch_config.get("enabled"):
            from .watch import GrantStore, WatchManager, attach_watch_capability
            from .watch_source import make_watch_source

            self._watch_manager = WatchManager(
                GrantStore(list(watch_config.get("grants") or [])),
                make_watch_source(),
                publish=self._publish_auxiliary_event,
                # Lazily read the supervisor's epoch at fire time (it is constructed
                # just below); a watch alert is epoch-stamped so barge-in can silence it.
                current_epoch=lambda: self.supervisor.speech_epoch,
                max_active=int(watch_config.get("max_active", 2) or 2),
                min_poll_sec=float(watch_config.get("min_poll_sec", 5.0) or 5.0),
            )
            attach_watch_capability(registry, watch_config, manager=self._watch_manager)
        # Startup reconciliation: log the capability manifest once and warn if the
        # planner is configured with a tool that isn't actually registered (the
        # drift the manifest is meant to prevent).
        self._reconcile_capabilities(registry, planner_config if planner_on else None)
        self.supervisor = AgentSupervisor(
            bus=self.bus,
            capabilities=registry,
            analyzer=LiveSpeechAnalyzer(
                ModePolicy(
                    vault_search_enabled=self._vault_enabled,
                    device_tools_enabled=self._device_tools_enabled,
                    device_tool_matcher=(
                        self._device_tool_dispatcher.match
                        if self._device_tool_dispatcher is not None
                        else None
                    ),
                )
            ),
            memory=memory,
            stream_tts=stream_tts,
            followup_config=followup_config,
            continuation_config=continuation_config,
            task_timeouts=task_timeouts,
            confirmation_ttl_sec=confirmation_ttl_sec,
            # control-plane-2: load-elastic admission. Ungated (unlike the
            # routing-only load_snapshot, which is live_routing-gated), since
            # tightening the concurrency ceiling under load is always safe.
            load_fraction=admission_load,
            on_turn_merged=self.metrics.mark_merged_turn,
            on_continuation_admitted=self._clear_arrival_continuation,
            on_input_claimed=self._mark_published_unheard_claimed,
            on_input_resolved=self._clear_published_unheard,
            record_user_memory=self._record_user_memory_ordered,
            runtime_owns_stop=True,
            defer_output_until_tts_admission=True,
            defer_output_until_playback_receipt=self._tracked_playback,
        )
        self.supervisor.state.mode = start_mode
        if self._intents is not None and hasattr(self._intents, "bind_speak"):
            self._intents.bind_speak(self._speak_local_intent)
        self.bus.subscribe(self._on_event)
        self._bus_threaded = False
        # Off-thread final dispatch (core/turn_merge.py). LLM-backed addressing,
        # cleanup, or routing always runs behind cancellable generation leases;
        # when turn merging is enabled, a final that reads mid-thought ("A long
        # story about") is also HELD briefly so the next words form ONE query.
        # A runtime with no gate/router and merging off keeps the synchronous
        # legacy path.
        self._dispatcher: Optional[FinalDispatcher] = None
        dispatch_config = turn_merge_config or TurnMergeConfig()
        needs_preprocessing_dispatch = any(
            gate is not None
            for gate in (addressing, cleaner, capability_router, router)
        )
        if dispatch_config.enabled or needs_preprocessing_dispatch:
            self._dispatcher = FinalDispatcher(
                self._process_final,
                dispatch_config,
                on_hold=lambda: self.metrics.mark(HELD),
                cancellable=True,
            )
        # Resume-after-interrupt + L4 self-echo guard (core/resume.py): tracks
        # the current turn's query + the sentences actually SPOKEN so (a) a
        # "start again"/"continue" after a cut resumes the reply from where it
        # stopped, and (b) a final that is the assistant's own TTS echo (the
        # mic hearing the reply's tail at high speaker volume) is dropped
        # instead of being answered. Default on; both halves off-switchable.
        self._resume = ResumeTracker(resume_config or ResumeConfig())
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

    # --- visual context feed (machine -> model) -----------------------------
    def set_current_frame(self, image: Optional[object]) -> None:
        """Set (or clear, with ``None``) the current visual frame fed to the model.

        ``image`` is whatever the configured multimodal LLM accepts -- raw image
        bytes (e.g. a PNG/JPEG) or a file path. A host process (a screen grabber,
        camera, or app) calls this; the latest frame is then attached AMBIENTLY to
        every subsequent assistant turn's context, so the model has visual context
        of what the user is doing without the user having to ask for a capture each
        time. Thread-safe. ``None`` clears it (back to text-only).

        Only the main/multimodal model receives the frame -- the fast tier (e.g.
        gemma3:1b) can't see images -- so an image-bearing turn is forced to the
        main tier and treated as PRIVATE (a screen capture never rides a public
        cloud chain; docs/target_architecture.md §9.7). A per-turn
        ``context['images']`` still overrides this ambient frame when set."""
        with self._frame_lock:
            self._current_frame = image

    def clear_current_frame(self) -> None:
        """Stop feeding a visual frame (same as ``set_current_frame(None)``)."""
        self.set_current_frame(None)

    def _current_images(self) -> Optional[list]:
        """``image_provider`` hook for the capability layer: the latest frame as a
        one-item list, or ``None`` when no frame is set (text-only)."""
        with self._frame_lock:
            frame = self._current_frame
        return [frame] if frame is not None else None

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
        controller never stays blocked on a capability that won't return, and
        expire abandoned staged confirmations so a stray later "yes" can't approve
        a forgotten action.  A running agent also claims recent durable reminder
        deliveries and speaks them through the normal cancellable TTS path."""
        try:
            self.supervisor.reap_overdue_tasks()
        except Exception:  # noqa: BLE001 - maintenance must never kill the watchdog
            log.exception("overdue-task reap failed")
        try:
            self.supervisor.sweep_expired_confirmations()
        except Exception:  # noqa: BLE001 - maintenance must never kill the watchdog
            log.exception("confirmation-TTL sweep failed")
        if self._reminder_manager is not None:
            try:
                self._reminder_manager.reconcile_stale_deliveries()
            except Exception:  # noqa: BLE001 - maintenance must keep running
                log.exception("stale reminder delivery reconciliation failed")
            try:
                reminders = self._reminder_manager.claim_voice_announcements()
                for reminder in reminders:
                    self._publish_auxiliary_event(
                        AgentEvent(
                            EventKind.TTS_REQUEST,
                            {
                                "task_id": f"reminder:{reminder.reminder_id}",
                                "text": f"Reminder: {reminder.message}",
                                "reminder_due": True,
                                "reminder_id": reminder.reminder_id,
                                "reminder_claim_token": reminder.voice_claim_token,
                            },
                        )
                    )
            except Exception:  # noqa: BLE001 - maintenance must keep running
                log.exception("reminder voice-announcement poll failed")

    # --- lifecycle ---
    def start(self, *, run_bus: bool = True) -> None:
        """Start the engine. ``run_bus=True`` runs the event loop on a
        background thread (production). Tests pass ``run_bus=False`` and pump
        the bus via :meth:`wait_idle`."""
        if self._dispatcher is not None:
            self._dispatcher.start()  # before the engine: callbacks may fire at once
        self.engine.start(
            EngineCallbacks(
                on_partial=self._on_partial,
                on_partial_result=self._on_partial_result,
                on_final=self._on_final,
                on_final_result=self._on_final_result,
                on_barge_in=self._on_barge_in,
                on_barge_in_result=self._on_barge_in_result,
                on_command=self._on_command,
                on_command_result=self._on_command_result,
                on_transcript_abort=self._on_transcript_abort,
                on_metric=self.metrics.mark,
                on_heartbeat=self._watchdog.note_heartbeat,
                on_capture_state=self._on_capture_state,
                # Anchors the L4 self-echo window: a final arriving shortly
                # after playback ends that reads like a just-spoken sentence
                # is the assistant's own echo (core/resume.py).
                on_speech_end=self._resume.note_playback_end,
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
        caller = threading.get_ident()
        with self._stop_lifecycle:
            if self._stop_complete:
                return
            if self._stop_started:
                if self._stop_owner_ident == caller:
                    return
                self._stop_waiters += 1
                self._stop_lifecycle.notify_all()
                try:
                    while not self._stop_complete:
                        self._stop_lifecycle.wait()
                finally:
                    self._stop_waiters -= 1
                return
            self._stop_started = True
            self._stop_owner_ident = caller
        try:
            self._stop_once()
        finally:
            with self._stop_lifecycle:
                self._stop_complete = True
                self._stop_owner_ident = None
                self._stop_lifecycle.notify_all()

    def _stop_once(self) -> None:
        # Shutdown gate FIRST: the threaded bus keeps dispatching until
        # bus.stop() below, so a queued TTS_REQUEST could otherwise start
        # speaking mid-teardown (codex-review 2026-07-06). _on_event drops
        # action-producing events once this is set.
        with self._terminal_effect_lock:
            self._stopping = True
            self._post_barge_response.invalidate()
            if self._reply_voice_continuity is not None:
                self._reply_voice_continuity.clear()
            self._clear_arrival_continuation()
            self._clear_published_unheard()
            self._clear_partial_fence()
            self._acoustic_revisions.clear()
            if self._intents is not None:
                self._intents.cancel_all()
        # Guard each step so a teardown error never prevents the engine from
        # stopping -- that is what flushes the session recording to disk.
        if self._dispatcher is not None:
            try:
                # Retire uncommitted preprocessing and stop the coordinator.
                self._dispatcher.stop()
            except Exception:  # noqa: BLE001
                log.exception("final dispatcher stop failed")
        try:
            self._watchdog.stop()
        except Exception:  # noqa: BLE001
            log.exception("watchdog stop failed")
        engine_stopped = False
        if self._playback_history is not None:
            # Receipt-capable shutdown must stop/terminalize the sink while the
            # bus and memory are still alive.  Legacy engines retain the old
            # teardown order below.  Bound the drain so a broken engine contract
            # can never turn shutdown into a hang.
            self._interrupt_playback_history()
            receipt_deadline = time.monotonic() + 0.5
            try:
                # Ask for terminal receipts while the diagnostic writer is still
                # accepting invalidation. Engine.stop() may close that writer.
                self.engine.stop_speaking()
            except Exception:  # noqa: BLE001
                log.exception("receipt-capable engine stop_speaking failed")
            while time.monotonic() < receipt_deadline:
                if not self._bus_threaded:
                    self.bus.drain()
                with self._playback_effect_lock:
                    if (
                        not self._playback_history.pending
                        and not self._pending_playback_memory_commits
                        and self.bus.idle()
                    ):
                        break
                time.sleep(0.005)
            with self._playback_effect_lock:
                playback_pending_before_engine_stop = bool(
                    self._playback_history.pending
                    or self._pending_playback_memory_commits
                )
            if playback_pending_before_engine_stop:
                invalidator = self._diagnostic_invalidator
                if invalidator is not None:
                    try:
                        invalidator("receipt_timeout")
                    except Exception:  # noqa: BLE001 - teardown must continue
                        log.exception("could not invalidate pending receipt evidence")
            try:
                self.engine.stop()
                engine_stopped = True
            except Exception:  # noqa: BLE001
                log.exception("receipt-capable engine stop failed")
            post_stop_receipt_deadline = time.monotonic() + 0.5
            while time.monotonic() < post_stop_receipt_deadline:
                if not self._bus_threaded:
                    self.bus.drain()
                with self._playback_effect_lock:
                    if (
                        not self._playback_history.pending
                        and not self._pending_playback_memory_commits
                        and self.bus.idle()
                    ):
                        break
                time.sleep(0.005)
            with self._playback_effect_lock:
                playback_pending = bool(
                    self._playback_history.pending
                    or self._pending_playback_memory_commits
                )
            if playback_pending:
                log.warning(
                    "playback receipts still pending after bounded shutdown drain"
                )
        try:
            self.supervisor.shutdown()
        except Exception:  # noqa: BLE001
            log.exception("supervisor shutdown failed")
        if self._watch_manager is not None:
            try:
                # The supervisor's terminal actor fence signals and boundedly
                # drains cooperative mutating providers before any backend
                # they may still be using is torn down.
                self._watch_manager.shutdown()
            except Exception:  # noqa: BLE001
                log.exception("watch manager shutdown failed")
        if self._reminder_manager is not None:
            try:
                # Closing the local DB does not cancel durable systemd timers;
                # scheduled reminders intentionally survive agent restarts.
                self._reminder_manager.close()
            except Exception:  # noqa: BLE001
                log.exception("reminder manager close failed")
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
        if not engine_stopped:
            self.engine.stop()

    @property
    def mode(self) -> Mode:
        return self.supervisor.state.mode

    # --- engine callbacks (may run on an audio thread) ---
    def _publish_playback_commits(
        self, commits: tuple[PlaybackCommit, ...]
    ) -> None:
        """Hand receipt-proven history back to the serialized brain thread."""

        for commit in commits:
            # Increment before publish while the playback-effect lock is held:
            # the bus subscriber's acknowledgement blocks on that same lock.
            # Roll back immediately when a terminally faulted bus rejects it.
            self._pending_playback_memory_commits += 1
            accepted = self.bus.publish(
                AgentEvent(
                    EventKind.MEMORY_COMMIT,
                    {
                        "source": (
                            "playback_receipt"
                            if commit.role == "assistant"
                            else "playback_ordered_user"
                        ),
                        "text": commit.text,
                        "is_followup": commit.is_followup,
                        "schedule_followup": commit.schedule_followup,
                        "epoch": commit.epoch,
                        "input_generation": commit.input_generation,
                        "followup_generation": commit.followup_generation,
                    },
                    priority=30,
                )
            )
            if not accepted:
                self._pending_playback_memory_commits -= 1
        if commits:
            self._playback_effect_changed.notify_all()

    def _record_user_memory_ordered(self, text: str) -> None:
        """Preserve assistant-before-user chronology without blocking the bus."""

        cleaned = (text or "").strip()
        if not cleaned:
            return
        history = self._playback_history
        if history is None:
            self.memory.add(cleaned, tags=("user",))
            return
        with self._playback_effect_changed:
            if history.conversation_pending or self._pending_playback_memory_commits:
                commits = history.stage_user(
                    cleaned,
                    epoch=self.supervisor.speech_epoch,
                    input_generation=self.supervisor.latest_arrival_generation,
                    followup_generation=self.supervisor.followup_generation,
                )
                self._publish_playback_commits(commits)
                self._playback_effect_changed.notify_all()
                return
        self.memory.add(cleaned, tags=("user",))

    def _await_playback_history(self, timeout: float = 1.0) -> bool:
        """Bounded provider-side barrier before reading/adding conversation.

        A new answer must not read recent context or append its user query ahead
        of an older sink receipt. Receipt callbacks and the bus remain free to
        run while this provider thread waits. A broken engine degrades after the
        bound; the capability then skips conversation memory for that turn.
        """

        history = self._playback_history
        if history is None:
            return True
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._playback_effect_changed:
            while (
                history.conversation_pending
                or self._pending_playback_memory_commits
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0 or self._stopping:
                    log.warning(
                        "conversation memory barrier timed out with playback history pending"
                    )
                    return False
                self._playback_effect_changed.wait(timeout=min(0.05, remaining))
            return True

    def _observe_playback_diagnostic(
        self,
        *,
        stage: str,
        fragment_id: str,
        context: Optional[PlaybackContext] = None,
        acoustic_identity: Optional[tuple[str, str, int]] = None,
        playback_kind: Optional[str] = None,
        outcome: Optional[str] = None,
        played_fraction: Optional[float] = None,
    ) -> None:
        observer = self._diagnostic_observer
        if observer is None:
            return
        try:
            from .diagnostic_bundle import DiagnosticObservation, DiagnosticStage

            if stage == "admitted":
                diagnostic_stage = DiagnosticStage.PLAYBACK_ADMITTED
            elif stage == "started":
                diagnostic_stage = DiagnosticStage.PLAYBACK_STARTED
            else:
                diagnostic_stage = DiagnosticStage.PLAYBACK_TERMINAL
            stream_id = utterance_id = None
            acoustic_revision = None
            if acoustic_identity is not None:
                stream_id, utterance_id, acoustic_revision = acoustic_identity
            observer(
                DiagnosticObservation(
                    stage=diagnostic_stage,
                    monotonic_ns=time.perf_counter_ns(),
                    stream_id=stream_id,
                    utterance_id=utterance_id,
                    revision=acoustic_revision,
                    correlation_id=fragment_id,
                    agent_session_id=(
                        context.session_id if context and context.session_id else None
                    ),
                    agent_turn_id=(
                        context.turn_id if context and context.turn_id else None
                    ),
                    agent_revision=(context.revision if context else None),
                    playback_kind=playback_kind,
                    outcome=outcome,
                    numeric_value=played_fraction,
                )
            )
        except Exception:  # noqa: BLE001 - evidence never owns playback lifecycle
            invalidator = self._diagnostic_invalidator
            if invalidator is not None:
                try:
                    invalidator("observation_error")
                except Exception:  # noqa: BLE001 - retain playback lifecycle
                    pass
            log.warning("playback diagnostic observation failed", exc_info=True)

    def _remember_diagnostic_acoustic(
        self,
        input_generation: int,
        acoustic: AcousticLineage | None,
        revision: int,
    ) -> None:
        if self._diagnostic_observer is None or acoustic is None or not acoustic.spans:
            return
        span = acoustic.spans[-1]
        generation = int(input_generation)
        if generation not in self._diagnostic_acoustic_by_input:
            self._diagnostic_acoustic_order.append(generation)
        self._diagnostic_acoustic_by_input[generation] = (
            span.stream_id,
            span.utterance_id,
            max(0, int(revision)),
        )
        while len(self._diagnostic_acoustic_order) > 256:
            expired = self._diagnostic_acoustic_order.popleft()
            self._diagnostic_acoustic_by_input.pop(expired, None)

    def _on_playback_started(self, fragment_id: str) -> None:
        """Exact sink-onset callback for one receipt-owned fragment."""

        history = self._playback_history
        if history is None:
            return
        with self._playback_effect_lock:
            context = history.fragment_context(fragment_id)
            attempted = history.mark_started(fragment_id)
            if attempted is None or context is None:
                return
            self._observe_playback_diagnostic(
                stage="started",
                fragment_id=fragment_id,
                context=context,
            )
            self._resume.note_playback_started(fragment_id)
            if not context.remember:
                return
            # Receipt-dispatch thread, not the audio callback: expose an exact
            # sink-onset marker to the autonomous harness without logging from
            # PortAudio's real-time thread.
            log.debug(
                "playback receipt started: fragment=%s task=%s input_generation=%d",
                fragment_id,
                context.task_id,
                context.input_generation,
            )

    def _on_playback_terminal(self, receipt: PlaybackReceipt) -> None:
        """Resolve one terminal receipt without doing memory I/O on its thread."""

        history = self._playback_history
        if history is None:
            return
        with self._playback_effect_lock:
            playback_context = history.fragment_context(receipt.fragment_id)
            playback_admission_id = history.fragment_playback_admission(
                receipt.fragment_id
            )
            try:
                try:
                    resolution = history.resolve(receipt)
                except Exception:  # noqa: BLE001 - adapter contract violation
                    log.exception(
                        "malformed playback receipt for %r; forcing failure",
                        receipt.fragment_id,
                    )
                    try:
                        resolution = history.force_fail(receipt.fragment_id)
                    except Exception:  # noqa: BLE001 - release capacity anyway
                        log.exception(
                            "failed to force-terminalize playback %r",
                            receipt.fragment_id,
                        )
                        resolution = None
                if resolution is None:
                    log.warning(
                        "ignoring unknown/duplicate playback receipt %r",
                        receipt.fragment_id,
                    )
                    return
                played_fraction = None
                if (
                    resolution.played_samples is not None
                    and resolution.total_samples is not None
                    and resolution.total_samples > 0
                ):
                    played_fraction = (
                        float(resolution.played_samples)
                        / float(resolution.total_samples)
                    )
                self._observe_playback_diagnostic(
                    stage="terminal",
                    fragment_id=receipt.fragment_id,
                    context=playback_context,
                    outcome=resolution.outcome.value,
                    played_fraction=played_fraction,
                )
                self._resume.note_playback_receipt(
                    resolution.fragment_id,
                    resolution.safe_text_prefix,
                    played=resolution.played,
                )
                self._publish_playback_commits(resolution.commits)
                self._log_playback_finalizations(history.drain_finalizations())
            finally:
                if playback_admission_id:
                    self.supervisor.finish_playback(playback_admission_id)
                self._playback_effect_changed.notify_all()

    def _rollback_registered_playback(self, fragment_id: str) -> None:
        """Resolve a history fragment when transfer to the resume ledger fails."""

        history = self._playback_history
        if history is None:
            return
        resolution = history.force_fail(fragment_id)
        if resolution is not None:
            self._resume.note_playback_receipt(
                resolution.fragment_id,
                resolution.safe_text_prefix,
                played=resolution.played,
            )
            self._publish_playback_commits(resolution.commits)
        self._log_playback_finalizations(history.drain_finalizations())
        self._playback_effect_changed.notify_all()

    @staticmethod
    def _log_playback_finalizations(
        finalizations: tuple[PlaybackFinalization, ...],
    ) -> None:
        """Expose one exact autonomous marker per remembered finalized group."""

        for finalization in finalizations:
            context = finalization.context
            if not context.remember:
                continue
            log.debug(
                "playback quiescent: tracked assistant reply terminal "
                "task=%s input_generation=%d outcome=%s",
                context.task_id,
                context.input_generation,
                finalization.outcome.value,
            )

    def _interrupt_playback_history(self) -> None:
        if self._reply_voice_continuity is not None:
            self._reply_voice_continuity.clear()
        history = self._playback_history
        if history is not None:
            with self._playback_effect_lock:
                self._publish_playback_commits(history.interrupt_all())
                self._playback_effect_changed.notify_all()
                self._log_playback_finalizations(history.drain_finalizations())

    def _new_input_generation(self) -> int:
        with self._input_generation_lock:
            generation = self._allocate_input_generation_locked()
        self.supervisor.note_input_arrival(generation)
        return generation

    def _allocate_input_generation_locked(self) -> int:
        """Allocate while ``_input_generation_lock`` is already held."""

        self._next_input_generation += 1
        # Only substantive finals/partial fences allocate a generation. They
        # retire any not-yet-consumed marker from an older cancelled lease.
        self._arrival_superseded_generations.clear()
        return self._next_input_generation

    def _latest_input_arrival_generation(self) -> int:
        with self._input_generation_lock:
            return self._next_input_generation

    @staticmethod
    def _acoustic_keys(
        acoustic: AcousticLineage | None,
    ) -> frozenset[tuple[str, int, int, str]] | None:
        if acoustic is None:
            return None
        return frozenset(span.key for span in acoustic.spans)

    def _begin_partial_fence(
        self,
        acoustic: AcousticLineage | None = None,
    ) -> Optional[int]:
        acoustic_keys = self._acoustic_keys(acoustic)
        with self._input_generation_lock:
            if self._partial_fence_active:
                if acoustic_keys is not None:
                    self._partial_fence_acoustic_keys = acoustic_keys
                return None
            self._partial_fence_active = True
            self._partial_fence_acoustic_keys = acoustic_keys
            generation = self._allocate_input_generation_locked()
            self._partial_fence_generation = generation
            self._partial_fence_unheard = self._published_unheard_finals.get(
                generation - 1
            )
        self.supervisor.note_input_arrival(generation)
        return generation

    def _clear_partial_fence(self) -> int | None:
        with self._input_generation_lock:
            self._partial_fence_active = False
            generation = self._partial_fence_generation
            self._partial_fence_generation = None
            self._partial_fence_acoustic_keys = None
            self._partial_fence_unheard = None
            return generation

    def _clear_matching_partial_fence(
        self,
        acoustic: AcousticLineage,
    ) -> tuple[int, _PublishedUnheardFinal | None] | None:
        acoustic_keys = self._acoustic_keys(acoustic)
        with self._input_generation_lock:
            if (
                not self._partial_fence_active
                or acoustic_keys is None
                or self._partial_fence_acoustic_keys != acoustic_keys
            ):
                return None
            self._partial_fence_active = False
            generation = self._partial_fence_generation
            self._partial_fence_generation = None
            self._partial_fence_acoustic_keys = None
            published_unheard = self._partial_fence_unheard
            self._partial_fence_unheard = None
            if generation is None:
                return None
            return generation, published_unheard

    def _note_arrival_supersede(self, generation: int) -> None:
        with self._input_generation_lock:
            self._arrival_superseded_generations.add(generation)

    def _take_arrival_supersede(self, generation: int) -> bool:
        with self._input_generation_lock:
            if generation not in self._arrival_superseded_generations:
                return False
            self._arrival_superseded_generations.remove(generation)
            return True

    def _note_arrival_continuation(
        self,
        generation: int,
        reservation: ArrivalContinuation,
        *,
        propagated: bool = False,
    ) -> None:
        with self._input_generation_lock:
            self._arrival_continuations.clear()
            self._arrival_continuations[generation] = (reservation, propagated)

    def _get_arrival_continuation(
        self,
        generation: int,
    ) -> tuple[ArrivalContinuation, bool] | None:
        with self._input_generation_lock:
            return self._arrival_continuations.get(generation)

    def _latest_arrival_continuation(
        self,
    ) -> tuple[ArrivalContinuation, bool] | None:
        with self._input_generation_lock:
            if not self._arrival_continuations:
                return None
            generation = max(self._arrival_continuations)
            return self._arrival_continuations[generation]

    def _clear_arrival_continuation(
        self,
        generation: int | None = None,
    ) -> None:
        with self._input_generation_lock:
            if generation is None:
                self._arrival_continuations.clear()
            else:
                self._arrival_continuations.pop(generation, None)

    def _note_published_unheard(
        self,
        generation: int,
        text: str,
        metrics_turn_token: int | None,
        event: AgentEvent,
    ) -> None:
        reservation = ArrivalContinuation(
            victim_task_id=f"input-{generation}",
            origin=text,
            addons=(),
            recorded_addon_count=0,
            merge_before_audio=True,
            record_origin=True,
            victim_owner_verified=False,
            victim_origin="unknown",
            victim_metrics_turn_token=metrics_turn_token,
            awaiting_addon=True,
        )
        with self._input_generation_lock:
            # Only the latest committed-but-unresolved final can be continued;
            # a newer publication has already generation-fenced every older one.
            self._published_unheard.clear()
            self._published_unheard_finals.clear()
            self._published_unheard[generation] = reservation
            self._published_unheard_finals[generation] = (
                _PublishedUnheardFinal(
                    generation=generation,
                    reservation=reservation,
                    event=event,
                )
            )

    def _mark_published_unheard_claimed(self, generation: int) -> None:
        """Transfer queue ownership to the supervisor before final analysis."""

        with self._input_generation_lock:
            published = self._published_unheard_finals.get(int(generation))
            if published is None:
                return
            published.claimed.set()

    def _restore_published_unheard(
        self,
        published: _PublishedUnheardFinal,
        *,
        generation: int,
        input_epoch: int,
    ) -> bool:
        """Transfer an unheard final across an aborted partial fence.

        STT/route/metric preparation has already committed. An event which is
        still queue-owned is retargeted in place. Once the supervisor has
        claimed it, publish one replacement and let the stale original fail its
        post-analysis generation check. Neither path reruns preprocessing.
        """

        reissue: AgentEvent | None = None
        with self._input_generation_lock:
            mapped = self._published_unheard_finals.get(
                published.generation
            )
            if mapped is not None and mapped.event is not published.event:
                return False
            if mapped is None and any(
                candidate.event is not published.event
                for candidate in self._published_unheard_finals.values()
            ):
                return False
            # The stale original may resolve between its partial and abort,
            # clearing the live map. The matching partial fence still owns this
            # immutable snapshot, and its shared claim token records whether
            # the supervisor had already dequeued it.
            current = mapped or published
            metadata = current.event.payload.get("metadata")
            if not isinstance(metadata, dict):
                return False
            reservation = replace(
                current.reservation,
                victim_task_id=f"input-{generation}",
            )
            self._published_unheard.clear()
            self._published_unheard[generation] = reservation
            if current.claimed.is_set():
                payload = dict(current.event.payload)
                replacement_metadata = dict(metadata)
                replacement_metadata["input_generation"] = int(generation)
                replacement_metadata["input_epoch"] = int(input_epoch)
                payload["metadata"] = replacement_metadata
                reissue = replace(current.event, payload=payload)
                self._published_unheard_finals[generation] = (
                    _PublishedUnheardFinal(
                        generation=generation,
                        reservation=reservation,
                        event=reissue,
                    )
                )
            else:
                metadata["input_generation"] = int(generation)
                metadata["input_epoch"] = int(input_epoch)
                # Keep the old-generation alias until the event resolves. The
                # bus may capture that generation concurrently with this
                # transfer; clearing either alias must retire the whole event.
                self._published_unheard_finals[generation] = replace(
                    current,
                    generation=generation,
                    reservation=reservation,
                )
        if reissue is not None:
            self.bus.publish(reissue)
        return True

    def _take_published_unheard(
        self,
        expected_generation: int,
    ) -> ArrivalContinuation | None:
        with self._input_generation_lock:
            reservation = self._published_unheard.pop(expected_generation, None)
            # Anything older is already fenced and can no longer be a sensible
            # continuation origin.
            for generation in tuple(self._published_unheard):
                if generation < expected_generation:
                    self._published_unheard.pop(generation, None)
            return reservation

    def _clear_published_unheard(self, generation: int | None = None) -> None:
        with self._input_generation_lock:
            if generation is None:
                self._published_unheard.clear()
                self._published_unheard_finals.clear()
            else:
                generation = int(generation)
                published = self._published_unheard_finals.get(generation)
                if published is None:
                    self._published_unheard.pop(generation, None)
                    return
                for candidate_generation, candidate in tuple(
                    self._published_unheard_finals.items()
                ):
                    if candidate.event is published.event:
                        self._published_unheard.pop(
                            candidate_generation, None
                        )
                        self._published_unheard_finals.pop(
                            candidate_generation, None
                        )

    def _publish_auxiliary_event(self, event: AgentEvent) -> None:
        """Route taskless/watch/local speech through cancellable TTS admission."""
        if event.kind != EventKind.TTS_REQUEST:
            self.bus.publish(event)
            return
        with self._terminal_effect_lock:
            if self._stopping:
                return
            payload = dict(event.payload)
            epoch = int(payload.get("epoch", self.supervisor.speech_epoch))
            aux_tts_id = self.supervisor.register_aux_tts(
                str(payload.get("task_id", "aux")),
                speech_epoch=epoch,
                input_generation=self.supervisor.latest_arrival_generation,
                input_epoch=self.supervisor.input_epoch,
            )
            payload.update(
                {
                    "epoch": epoch,
                    "auxiliary_tts": True,
                    "aux_tts_id": aux_tts_id,
                    **self.supervisor.auxiliary_tts_identity_payload(
                        aux_tts_id
                    ),
                }
            )
            self.bus.publish(
                AgentEvent(
                    event.kind,
                    payload,
                    priority=event.priority,
                    timestamp=event.timestamp,
                )
            )

    def _speak_local_intent(self, text: str) -> None:
        if not text:
            return
        self._publish_auxiliary_event(
            AgentEvent(
                EventKind.TTS_REQUEST,
                {"task_id": "local-intent", "text": text},
            )
        )

    def _on_partial_result(self, result: PartialTranscript) -> None:
        self._on_partial(
            result.text,
            acoustic=result.acoustic,
            revision=result.revision,
        )

    def _on_transcript_abort(self, result: TranscriptAbort) -> None:
        """Retire a typed partial turn without creating an answer."""
        with self._terminal_effect_lock:
            if self._stopping:
                return
            if not self._acoustic_revisions.accept(
                result.acoustic,
                result.revision,
                final=True,
            ):
                log.debug(
                    "dropping stale/duplicate acoustic abort: revision=%d",
                    result.revision,
                )
                return
            post_barge_observation = self._post_barge_response.inspect(
                self.supervisor.input_epoch,
                "unknown",
                acoustic=result.acoustic,
            )
            if post_barge_observation is not None:
                self._post_barge_response.consume(post_barge_observation)
            cleared_fence = self._clear_matching_partial_fence(
                result.acoustic
            )
            if cleared_fence is not None:
                partial_generation, published_unheard = cleared_fence
                self.supervisor.commit_input_generation(partial_generation)
                self._clear_arrival_continuation(partial_generation)
                dispatcher_restored = False
                if self._dispatcher is not None:
                    dispatcher_restored = self._dispatcher.abandon_partial(
                        input_generation=partial_generation,
                        input_epoch=self.supervisor.input_epoch,
                    )
                if not dispatcher_restored and published_unheard is not None:
                    self._restore_published_unheard(
                        published_unheard,
                        generation=partial_generation,
                        input_epoch=self.supervisor.input_epoch,
                    )
            self.bus.publish(
                AgentEvent.transcript_aborted(
                    reason=result.reason.value,
                    acoustic=result.acoustic,
                    revision=result.revision,
                )
            )

    def _on_partial(
        self,
        text: str,
        *,
        acoustic: AcousticLineage | None = None,
        revision: int = 0,
    ) -> None:
        with self._terminal_effect_lock:
            if self._stopping:
                return
            if not self._acoustic_revisions.accept(
                acoustic, revision, final=False
            ):
                log.debug(
                    "dropping stale acoustic partial: revision=%d", revision
                )
                return
            # A recognizer interim that already reads as our just-played TTS is
            # not new user speech.  Preview only: the final guard owns the echo
            # diagnostic and consumes it.  Most importantly, do not let an echo
            # partial cancel a valid answer still in preprocessing.
            if self._resume.preview_self_echo(text):
                self.bus.publish(
                    AgentEvent.partial(
                        text, acoustic=acoustic, revision=revision
                    )
                )
                return
            partial_generation = (
                self._begin_partial_fence(acoustic)
                if normalize_text(text)
                else None
            )
            if partial_generation is not None:
                self.supervisor.cancel_pending_aux_tts()
            if self._dispatcher is not None:
                # Resumed speech retires premature preprocessing; with turn merging,
                # its next final is folded into the same bounded held turn.
                self._dispatcher.note_partial()
            if partial_generation is not None:
                reservation = self.supervisor.reserve_unheard_for_partial()
                if (
                    reservation is None
                    and self.supervisor.continuation_enabled
                ):
                    reservation = self._take_published_unheard(
                        partial_generation - 1
                    )
                if reservation is not None:
                    self.metrics.mark_arrival_superseded_turn(
                        reservation.victim_metrics_turn_token
                    )
                    if self.supervisor.continuation_enabled:
                        self._note_arrival_continuation(
                            partial_generation,
                            reservation,
                        )
            self.bus.publish(
                AgentEvent.partial(
                    text, acoustic=acoustic, revision=revision
                )
            )

    def _on_final_result(self, result: FinalTranscript) -> None:
        """Typed engine-final seam; text-only engines remain fail-closed."""
        self._on_final(
            result.text,
            owner_verified=result.owner_verified,
            owner_rejected=(
                result.owner_verification is OwnerVerification.REJECTED
            ),
            origin=result.origin,
            acoustic=result.acoustic,
            revision=result.revision,
        )

    def _on_final(
        self,
        text: str,
        *,
        owner_verified: bool = False,
        owner_rejected: bool = False,
        origin: str = _LEGACY_TEXT_ORIGIN,
        acoustic: AcousticLineage | None = None,
        revision: int = 0,
    ) -> None:
        final_at = time.perf_counter()
        with self._terminal_effect_lock:
            if self._stopping:
                return
            if not self._acoustic_revisions.accept(
                acoustic, revision, final=True
            ):
                log.debug(
                    "dropping stale/duplicate acoustic final: revision=%d",
                    revision,
                )
                return
            if owner_rejected:
                # Sherpa's loudness-rescue path intentionally admits REJECTED
                # identity for ordinary unverified usability. Preserve that
                # contract, but only after this acoustic revision wins the
                # terminal gate: a stale callback must have no effect on a newer
                # post-barge turn.
                self._post_barge_response.invalidate()
                log.info(
                    "speaker-rejected final is ineligible for post-barge "
                    "response: %r",
                    text,
                )
            post_barge_armed = self._post_barge_response.is_armed(
                self.supervisor.input_epoch
            )
            # Validate the carrier floor before newest-input cancellation. A
            # punctuation-only recognizer blip must not retire a valid question.
            if not normalize_text(text):
                if post_barge_armed:
                    self._post_barge_response.invalidate()
                log.info("dropping empty/punctuation-only final: %r", text)
                self.metrics.mark(HANDLED_LOCAL)
                partial_generation = self._clear_partial_fence()
                if partial_generation is not None:
                    self.supervisor.commit_input_generation(partial_generation)
                    self._clear_arrival_continuation(partial_generation)
                return
            if (
                (not owner_verified or post_barge_armed)
                and self._resume.preview_self_echo(text)
            ):
                # Like punctuation noise, a recognized self-echo must not enter
                # the dispatcher and cancel valid preprocessing already in flight.
                if self._resume.is_self_echo(text):
                    if post_barge_armed:
                        self._post_barge_response.invalidate()
                    log.info("dropping self-echo final (own TTS heard back): %r", text)
                    self.metrics.mark(HANDLED_LOCAL)
                    partial_generation = self._clear_partial_fence()
                    if partial_generation is not None:
                        self.supervisor.commit_input_generation(partial_generation)
                        self._clear_arrival_continuation(partial_generation)
                    return
            self._clear_partial_fence()
            self.supervisor.cancel_pending_aux_tts()
            input_generation = self._new_input_generation()
            self._remember_diagnostic_acoustic(
                input_generation,
                acoustic,
                revision,
            )
            pending = self._latest_arrival_continuation()
            if pending is not None:
                reservation, _was_propagated = pending
                # FinalDispatcher is the authority on whether ASR finals are
                # separate add-ons or fragments of one utterance. Carry the
                # lineage forward unchanged; _process_final reconciles it with
                # the lease's coalesced text before any gate runs.
                self._note_arrival_continuation(
                    input_generation,
                    reservation,
                    propagated=True,
                )
            else:
                reservation = self.supervisor.reserve_arrival_continuation(text)
                if reservation is None and self.supervisor.continuation_enabled:
                    published = self._take_published_unheard(
                        input_generation - 1
                    )
                    if published is not None:
                        reservation = self.supervisor.extend_arrival_continuation(
                            published,
                            text,
                        )
                        if reservation is None:
                            self.metrics.mark_arrival_superseded_turn(
                                published.victim_metrics_turn_token
                            )
                if reservation is not None:
                    self._note_arrival_continuation(
                        input_generation,
                        reservation,
                    )
                    if reservation.merge_before_audio:
                        self.metrics.mark_arrival_superseded_turn(
                            reservation.victim_metrics_turn_token
                        )
                elif self.supervisor.should_preempt_for_final_arrival(text):
                    self.supervisor.cancel_all(invalidate_inputs=False)
                    self.metrics.mark_arrival_superseded_turn()
                    self._note_arrival_supersede(input_generation)
            if self._dispatcher is not None:
                # Cheap engine-thread handoff (lock + notify); the hold/merge
                # logic and full dispatch chain run on provider workers.
                self._dispatcher.submit(
                    text,
                    submitted_at=final_at,
                    input_generation=input_generation,
                    input_epoch=self.supervisor.input_epoch,
                    owner_verified=owner_verified,
                    origin=origin,
                    acoustic=acoustic,
                    revision=revision,
                )
                return
            self._process_final(
                text,
                final_at=final_at,
                input_generation=input_generation,
                owner_verified=owner_verified,
                origin=origin,
                acoustic=acoustic,
                revision=revision,
            )

    def _process_final(
        self,
        text: str,
        lease: Optional[FinalDispatchLease] = None,
        *,
        final_at: Optional[float] = None,
        input_generation: Optional[int] = None,
        owner_verified: bool = False,
        origin: str = "unknown",
        acoustic: AcousticLineage | None = None,
        revision: int = 0,
    ) -> None:
        cancel_event = lease.cancel_event if lease is not None else None
        terminal_input_epoch = (
            lease.input_epoch
            if lease is not None and lease.input_epoch is not None
            else self.supervisor.input_epoch
        )
        if lease is not None:
            final_at = lease.submitted_at
            input_generation = lease.input_generation
            owner_verified = lease.owner_verified
            origin = lease.origin
            acoustic = lease.acoustic
            revision = lease.revision
        if final_at is None:
            final_at = time.perf_counter()
        if input_generation is None:
            input_generation = self._new_input_generation()
        arrival_superseded = self._take_arrival_supersede(input_generation)
        continuation_state = self._get_arrival_continuation(input_generation)
        continuation = (
            continuation_state[0] if continuation_state is not None else None
        )
        continuation_propagated = bool(
            continuation_state is not None and continuation_state[1]
        )

        def retired() -> bool:
            return bool(
                self._stopping
                or self.supervisor.input_epoch != terminal_input_epoch
                or input_generation < self._latest_input_arrival_generation()
                or (cancel_event is not None and cancel_event.is_set())
            )

        def claim_terminal() -> bool:
            if retired():
                return False
            return lease.claim_commit() if lease is not None else True

        if retired():
            return
        # Inspect only after the dispatcher has produced its current merged
        # candidate.  This preserves the live run's multi-final override while
        # avoiding consumption by a held lease that a newer partial can retire.
        post_barge_observation = self._post_barge_response.inspect(
            terminal_input_epoch,
            "unknown" if origin == _LEGACY_TEXT_ORIGIN else origin,
            acoustic=acoustic,
        )
        legacy_text_final = origin == _LEGACY_TEXT_ORIGIN
        direct_user_instruction = bool(
            not legacy_text_final
            and origin == "live_audio"
            and post_barge_observation is None
            and continuation is None
            and not (lease is not None and lease.coalesced)
        )
        if legacy_text_final:
            origin = "unknown"
        if continuation is not None and continuation_propagated:
            if continuation.awaiting_addon:
                continuation = self.supervisor.extend_arrival_continuation(
                    continuation,
                    text,
                )
                if continuation is None:
                    self._clear_arrival_continuation(input_generation)
                    arrival_superseded = True
            elif lease is not None and lease.coalesced:
                continuation = self.supervisor.coalesce_arrival_continuation(
                    continuation,
                    text,
                )
            else:
                previous = continuation
                continuation = self.supervisor.extend_arrival_continuation(
                    continuation,
                    text,
                )
                if continuation is None:
                    self._clear_arrival_continuation(input_generation)
                    arrival_superseded = True
                    if (
                        not previous.merge_before_audio
                        and not self.supervisor.state.pending_confirmations
                    ):
                        with self._terminal_effect_lock:
                            if not retired():
                                self.supervisor.cancel_all(
                                    invalidate_inputs=False
                                )
        # Carrier-signal floor: a final with no words at all ('.', '?') is
        # recognizer noise -- live it got ACT'ed and answered ('.' -> "Ten.").
        # Nothing downstream can do anything sensible with it.
        if not normalize_text(text):
            if not claim_terminal():
                return
            if post_barge_observation is not None:
                self._post_barge_response.consume(post_barge_observation)
            with self._terminal_effect_lock:
                if retired():
                    return
                log.info("dropping empty/punctuation-only final: %r", text)
                if not self.supervisor.commit_input_generation(input_generation):
                    return
                self.metrics.mark(HANDLED_LOCAL)
                self._clear_arrival_continuation(input_generation)
            return
        # L4 self-echo guard (core/resume.py): a final that arrived within the
        # echo window of playback end AND reads like a just-spoken sentence is
        # the assistant hearing ITSELF (live: its own "Okay, let's begin. How
        # can I help you today?" came back as a user turn and got answered).
        # Volume-independent -- the energy floors (L1-L3) cannot catch a loud
        # echo, the text match can. Dropped before addressing/ingest/metrics.
        if (
            (not owner_verified or post_barge_observation is not None)
            and self._resume.is_self_echo(text)
        ):
            if not claim_terminal():
                return
            if post_barge_observation is not None:
                self._post_barge_response.consume(post_barge_observation)
            with self._terminal_effect_lock:
                if retired():
                    return
                log.info("dropping self-echo final (own TTS heard back): %r", text)
                if not self.supervisor.commit_input_generation(input_generation):
                    return
                self.metrics.mark(HANDLED_LOCAL)
                self._clear_arrival_continuation(input_generation)
            return
        # Resume-after-interrupt: "start again"/"continue" after a CUT reply
        # becomes a continue-from-where-you-stopped prompt instead of a fresh
        # turn (owner: a stopped story must resume, not restart or greet).
        resume_prompt = self._resume.preview_resume_prompt(text)
        log.info(
            "final -> brain: %r (mode=%s input_generation=%d)",
            text,
            self.mode.value,
            input_generation,
            extra={"transcript": {"role": "user", "text": text, "mode": self.mode.value}},
        )
        if resume_prompt is not None:
            if not claim_terminal():
                return
            post_barge_consumed = True
            if post_barge_observation is not None:
                post_barge_consumed = self._post_barge_response.consume(
                    post_barge_observation
                )
            with self._terminal_effect_lock:
                if retired():
                    return
                invalid_observation = bool(
                    post_barge_observation is not None
                    and (
                        (
                            not post_barge_observation.response_eligible
                            and not legacy_text_final
                        )
                        or not post_barge_consumed
                    )
                )
                if self.mode != Mode.ASSISTANT or invalid_observation:
                    # A foreign/replaced barge observation or mode transition
                    # cannot use the synthetic bypass. Clear the abandoned
                    # resume lineage so a second final cannot retry it after the
                    # one-shot observation has been consumed.
                    if not self.supervisor.commit_input_generation(
                        input_generation
                    ):
                        return
                    self._resume.clear_resume_lineage()
                    self.metrics.mark(HANDLED_LOCAL)
                    self._clear_arrival_continuation(input_generation)
                    return
                if not self.supervisor.commit_input_generation(input_generation):
                    return
                # Consume only after terminal ownership won against a newer final.
                resume_prompt = self._resume.resume_prompt(text) or resume_prompt
                log.info("resume request %r -> continuing the interrupted reply", text)
                self.metrics.mark(ASR_FINAL, at=final_at)
                if arrival_superseded:
                    self.metrics.mark_superseded_turn()
                metrics_turn_token = self.metrics.current_turn_token()
                # Every synthetic resume is response-only: it embeds an older
                # query but the current "continue" has no stored owner verdict.
                # Keep tracker lineage for repeat cut/resume UX while making each
                # future synthetic resume take this same no-tools path.
                self.bus.publish(
                    AgentEvent.final(
                        resume_prompt,
                        owner_verified=False,
                        origin="unknown",
                        metadata={
                            "latency_policy": LatencyPolicy.SNAPPY_ANSWER.value,
                            "metrics_turn_token": metrics_turn_token,
                            "input_epoch": terminal_input_epoch,
                            "input_generation": input_generation,
                            "post_barge_response_only": True,
                            "skip_user_memory": True,
                        },
                        acoustic=acoustic,
                        revision=revision,
                    )
                )
                self._clear_arrival_continuation(input_generation)
            return
        # Input gate: classify before opening a metrics turn so an INGEST'd
        # utterance doesn't trip the watchdog's "no llm_first_token" check.
        # When no classifier is configured this is a no-op (legacy behavior).
        post_barge_response_only = False
        exact_device_command = (
            self._device_tool_dispatcher.match(text)
            if direct_user_instruction and self._device_tool_dispatcher is not None
            else None
        )
        pending_confirmation_response = bool(
            direct_user_instruction
            and self.supervisor.state.pending_confirmations
            and is_pending_confirmation_response(text)
        )
        deterministic_live_control = bool(
            exact_device_command is not None or pending_confirmation_response
        )
        if self._addressing is not None and not deterministic_live_control:
            # Conversation window only -- exclude non-conversational 'vision'
            # (screen) and 'procedural' (standing-rule) memories so a caption/OCR
            # trace or a behavior rule never enters the addressing classifier.
            recent = [it.text for it in self.memory.all()
                      if "vision" not in it.tags and "procedural" not in it.tags][-4:]
            decision = self._addressing.classify(text, recent=recent)
            if retired():
                return
            log.info("addressing decision: %s for %r", decision, text)
            if decision == INGEST or (decision == UNSURE and not self._unsure_acts):
                if (
                    post_barge_observation is not None
                    and post_barge_observation.response_eligible
                    and self.mode == Mode.ASSISTANT
                ):
                    # The acoustic interruption proves conversational timing,
                    # not identity.  Let this one final receive an answer while
                    # stripping every action-trust bit and downstream action
                    # route.  Terminal consumption below is still required.
                    post_barge_response_only = True
                    owner_verified = False
                    origin = "unknown"
                    log.info(
                        "addressing override: bounded post-barge response-only final"
                    )
                # A short add-on to a turn that's still in flight ("make it
                # shorter", "in spanish") reads as ambient to the addressing
                # gate, but it IS addressed -- let a genuine continuation reach
                # the brain so it can merge/continue instead of being dropped.
                elif (
                    continuation is not None
                    or self.supervisor.looks_like_continuation(text)
                ):
                    log.info("addressing override: continuation of in-flight turn")
                else:
                    # A memory write is irreversible. Win terminal ownership and
                    # commit this input identity BEFORE starting it; the prior
                    # ordering wrote first and let a newer lease cancel the old
                    # one while its stale DB write still completed afterward.
                    if not claim_terminal():
                        return
                    if post_barge_observation is not None:
                        self._post_barge_response.consume(
                            post_barge_observation
                        )
                    with self._terminal_effect_lock:
                        if retired():
                            return
                        if not self.supervisor.commit_input_generation(
                            input_generation
                        ):
                            return
                        self.metrics.mark(HANDLED_LOCAL)
                        if arrival_superseded:
                            self.metrics.mark_arrival_superseded_turn()
                        self._clear_arrival_continuation(input_generation)
                    self.memory.add(text, tags=("ingested",))
                    return
        # Cleanup pass: rewrite disfluencies / self-corrections so the brain
        # acts on what the user meant, not on every "um" + word-repeat. The
        # raw text was already logged above; emit a second transcript entry
        # with the cleaned version (and the raw retained in 'raw') so the
        # user can audit every rewrite in run-<id>.summary.json.
        final_text = text
        if self._cleaner is not None:
            recent_for_cleaner = [it.text for it in self.memory.all()
                                  if "vision" not in it.tags and "procedural" not in it.tags][-4:]
            try:
                cleaned = self._cleaner.clean(text, recent=recent_for_cleaner)
            except LLMCallCancelled:
                raise
            except Exception:  # noqa: BLE001
                log.exception("transcript cleaner failed; passing raw text through")
                cleaned = text
            if retired():
                return
            # Any token-changing model attempt strips action authority, even if
            # the cleaner returns empty or a hallucination guard later keeps the
            # raw transcript. Casing/punctuation-only cleanup remains equivalent.
            if (
                cleaned != text
                and normalize_text(str(cleaned or "")) != normalize_text(text)
            ):
                owner_verified = False
                direct_user_instruction = False
            if cleaned and cleaned != text:
                # Cleaner-hallucination guards (live run-20260610-132603): the
                # fast-tier cleaner rewrote noise fragments into the
                # ASSISTANT'S OWN prior sentences from its recent-context
                # ('Well' -> "What would you like to know about your place?"),
                # manufacturing phantom turns the assistant then answered.
                # (1) A rewrite that reads as a just-spoken assistant sentence
                # is noise wearing the assistant's words -- drop the turn.
                # (2) A rewrite that GROWS a fragment materially (cleaning can
                # only shrink/reshape) is invented content -- keep the raw.
                if self._resume.is_self_echo(cleaned):
                    if not claim_terminal():
                        return
                    if post_barge_observation is not None:
                        self._post_barge_response.consume(
                            post_barge_observation
                        )
                    with self._terminal_effect_lock:
                        if retired():
                            return
                        if not self.supervisor.commit_input_generation(
                            input_generation
                        ):
                            return
                        log.info(
                            "dropping final: cleaner rewrote noise %r into the "
                            "assistant's own words %r", text, cleaned,
                        )
                        self.metrics.mark(HANDLED_LOCAL)
                        if arrival_superseded:
                            self.metrics.mark_arrival_superseded_turn()
                        self._clear_arrival_continuation(input_generation)
                    return
                if rewrite_is_overreach(text, cleaned):
                    log.info(
                        "ignoring cleaner over-rewrite %r -> %r (kept raw)",
                        text, cleaned,
                    )
                else:
                    log.info(
                        "cleaned: %r -> %r", text, cleaned,
                        extra={"transcript": {
                            "role": "user", "text": cleaned, "raw": text,
                            "mode": self.mode.value,
                        }},
                    )
                    final_text = cleaned
        if (
            direct_user_instruction
            and self._device_tool_dispatcher is not None
            and self._device_tool_dispatcher.match(final_text) is not None
            and exact_device_command is None
        ):
            # Punctuation cleanup may not manufacture an exact action grammar.
            # The raw live transcript itself must already have matched a typed
            # command before any model/cleaner touched it.
            owner_verified = False
            direct_user_instruction = False
        continuation_metadata: dict[str, object] = {}
        routed_text = final_text
        if continuation is not None:
            _merged_text, continuation_metadata = (
                self.supervisor.materialize_arrival_continuation(
                    continuation,
                    final_text,
                )
            )

        # Unified capability-router decision (the "middle layer"). It DRIVES
        # behaviour via the tier router + escalate predicate wired in __init__;
        # this call records the decision for the run summary AND primes the
        # fast-LLM action cache, so the in-capability escalate/tier consults this
        # turn don't re-call the model. Best-effort: routing never breaks a turn.
        route_decision = None
        if (
            self._capability_router is not None
            and not post_barge_response_only
            and not (
                direct_user_instruction
                and exact_device_command is not None
            )
        ):
            try:
                route_context: dict[str, object] = {
                    "mode": self.mode.value,
                    "stream_tts": self._stream_tts,
                }
                if self._vault_enabled:
                    route_context["vault_available"] = True
                if cancel_event is not None:
                    route_context["cancel_event"] = cancel_event
                route_decision = self._capability_router.route(
                    routed_text,
                    route_context,
                )
                log.info(
                    "route: action=%s tier=%s policy=%s conf=%.2f source=%s (%s) for %r",
                    route_decision.action, route_decision.tier,
                    route_decision.latency_policy, route_decision.confidence,
                    route_decision.source, route_decision.reason, routed_text,
                    extra={"route": {
                        "action": route_decision.action, "tier": route_decision.tier,
                        "latency_policy": route_decision.latency_policy,
                        "confidence": route_decision.confidence,
                        "source": route_decision.source, "reason": route_decision.reason,
                    }},
                )
            except LLMCallCancelled:
                raise
            except Exception:  # noqa: BLE001 - routing observability is best-effort
                log.exception("capability router failed; default routing stands")
        if retired():
            return
        latency_context: dict[str, object] = {
            "mode": self.mode.value,
            "stream_tts": self._stream_tts,
        }
        if self._vault_enabled:
            latency_context["vault_available"] = True
        if post_barge_response_only:
            # Response-only means exactly that: do not ask either routing layer
            # whether this transcript should control, act, search, or research.
            latency_policy = LatencyPolicy.SNAPPY_ANSWER.value
        elif direct_user_instruction and exact_device_command is not None:
            # Exact controller-owned local tools do not need a model-tier
            # latency classification. In particular, terminal question-mark
            # punctuation must not turn an app/list command into CLARIFY.
            latency_policy = LatencyPolicy.INSTANT_CONTROL.value
        elif route_decision is not None:
            latency_context.update({
                "route_action": route_decision.action,
                "tier": route_decision.tier,
            })
            latency_policy = route_decision.latency_policy
        else:
            # A custom latency router may itself block. Keep it inside the
            # cancellable preprocessing lease, before terminal ownership.
            latency_policy = classify_latency_policy(
                routed_text, latency_context, router=self._latency_router
            ).value
        if retired():
            return
        # All blocking preprocessing is complete. Atomically win terminal
        # ownership before any intent, memory, supervisor, metric, or bus side
        # effect so a newer final cannot be followed by this stale turn.
        if not claim_terminal():
            return
        post_barge_consumed = True
        if post_barge_observation is not None:
            post_barge_consumed = self._post_barge_response.consume(
                post_barge_observation
            )
        if post_barge_response_only and not post_barge_consumed:
            # A newer barge/invalidation replaced the inspected token. Resolve
            # this stale lease locally and DROP its garble: do not answer it and
            # do not preserve it as ambient memory.
            with self._terminal_effect_lock:
                if retired():
                    return
                if not self.supervisor.commit_input_generation(input_generation):
                    return
                self.metrics.mark(HANDLED_LOCAL)
                if arrival_superseded:
                    self.metrics.mark_arrival_superseded_turn()
                self._clear_arrival_continuation(input_generation)
            return
        if post_barge_response_only and self.mode != Mode.ASSISTANT:
            # A UI/external mode transition raced preprocessing after the grant
            # was inspected. Resolve the old assistant-mode lease silently;
            # never answer into the newly selected mode.
            with self._terminal_effect_lock:
                if retired():
                    return
                if not self.supervisor.commit_input_generation(input_generation):
                    return
                self.metrics.mark(HANDLED_LOCAL)
                self._clear_arrival_continuation(input_generation)
            return
        with self._terminal_effect_lock:
            if retired():
                return
            if not self.supervisor.commit_input_generation(input_generation):
                return
            self.metrics.mark(ASR_FINAL, at=final_at)
            if arrival_superseded:
                self.metrics.mark_superseded_turn()
            metrics_turn_token = self.metrics.current_turn_token()
            if route_decision is not None:
                self.metrics.mark(f"route_{route_decision.action}")

            # Try the no-LLM fast-path next; only fall through to the brain on a miss.
            if (
                not post_barge_response_only
                and self._intents is not None
                and self._intents.handle(final_text)
            ):
                self._clear_arrival_continuation(input_generation)
                log.debug("handled by intent fast-path: %r", final_text)
                if (
                    (
                        self.supervisor.state.active_tasks
                        or self.supervisor.state.queued_tasks
                    )
                    and not self.supervisor.state.pending_confirmations
                    and not self.supervisor.looks_like_continuation(final_text)
                ):
                    log.info(
                        "local intent supersedes the in-flight turn: %r",
                        final_text,
                    )
                    self.supervisor.cancel_all(invalidate_inputs=False)
                    self.metrics.mark_superseded_turn()
                # Stamp the turn as resolved WITHOUT the LLM so the watchdog skips
                # it -- this turn has an asr_final but will never reach
                # llm_first_token, which would otherwise read as a false "llm stuck"
                # (rc-5). Same intent as the BARGE_IN skip in core/watchdog.py.
                self.metrics.mark(HANDLED_LOCAL)
                return
            # Newest-input-wins (owner, live round 3): a NEW final arriving while a
            # prior turn is still generating/queued SUPERSEDES it -- the user has
            # moved on, and speaking the stale answer first reads as "answering my
            # old question". Skipped when a confirmation is pending (this final is
            # likely its answer -- cancel_all would clear the pending confirm) and
            # for CONTINUE add-ons (the supervisor merges those into the in-flight
            # turn instead). Playback isn't a case here: ASR never emits finals
            # while the assistant is speaking (barge-in owns that path).
            if (
                self.supervisor.has_live_tasks()
                and not self.supervisor.state.pending_confirmations
                and continuation is None
                and not self.supervisor.looks_like_continuation(final_text)
            ):
                log.info("newest input supersedes the in-flight turn: %r", final_text)
                self.supervisor.cancel_all(invalidate_inputs=False)
                # Mark the preempted turn so the watchdog doesn't read its missing
                # llm_first_token as a stalled LLM (rc-5). The new final's turn-start
                # mark (SPEECH_END on the sherpa engine, else ASR_FINAL) already
                # banked that turn, so it's the last completed one.
                self.metrics.mark_superseded_turn()
            # A NEW turn for the resume tracker (resets the spoken-text window; a
            # resume turn deliberately bypasses this above and keeps accumulating).
            # Every later synthetic resume is forced through the same response-
            # only path above, so this query remains resumable without gaining
            # routing, tool, durable-memory, or owner authority.
            self._resume.note_query(final_text)
            published_origin = str(
                continuation_metadata.get(
                    "continuation_merged_text",
                    final_text,
                )
            )
            final_event = AgentEvent.final(
                final_text,
                owner_verified=owner_verified,
                origin=origin,
                metadata={
                    "latency_policy": latency_policy,
                    "metrics_turn_token": metrics_turn_token,
                    "input_epoch": terminal_input_epoch,
                    "input_generation": input_generation,
                    **continuation_metadata,
                    "post_barge_response_only": post_barge_response_only,
                    "direct_user_instruction": bool(
                        direct_user_instruction
                        and not post_barge_response_only
                        and continuation is None
                        and resume_prompt is None
                    ),
                    "skip_user_memory": bool(
                        post_barge_response_only
                        or continuation_metadata.get("skip_user_memory", False)
                    ),
                },
                acoustic=acoustic,
                revision=revision,
            )
            # Snapshot only conversational assistant turns.  Do not call the
            # supervisor analyzer here: custom analyzers may block, and this is
            # still inside the runtime's terminal seam.  When unified routing is
            # available its non-simple actions give the stronger exclusion.
            if (
                not post_barge_response_only
                and self.mode == Mode.ASSISTANT
                # A custom analyzer may define private non-assistant intents
                # which this pure default preview cannot know. Fail closed: the
                # publish-gap continuation optimization is available only for
                # the exact shipped deterministic analyzer.
                and type(self.supervisor.analyzer) is LiveSpeechAnalyzer
                and is_assistant_mode_final_candidate(
                    final_text,
                    self.mode,
                    has_pending_confirmation=bool(
                        self.supervisor.state.pending_confirmations
                    ),
                    vault_search_enabled=self._vault_enabled,
                    device_tools_enabled=self._device_tools_enabled,
                )
                and (
                    route_decision is None
                    or route_decision.action == "simple"
                )
            ):
                self._note_published_unheard(
                    input_generation,
                    published_origin,
                    metrics_turn_token,
                    final_event,
                )
            self.bus.publish(final_event)

    def _on_barge_in_result(self, result: AcousticSignal) -> None:
        self._on_barge_in(
            acoustic=result.acoustic,
            revision=result.revision,
        )

    def _on_barge_in(
        self,
        *,
        acoustic: AcousticLineage | None = None,
        revision: int = 0,
    ) -> None:
        # User spoke over the assistant: cancel in-flight work, then cut playback.
        # Cancellation MUST be set before stop_speaking() returns so the
        # streaming emitter stops producing sentences and any TTS_REQUEST still
        # in flight is dropped by _on_event -- otherwise a stale sentence from
        # the interrupted turn could be spoken after the barge-in
        # (realtime-concurrency-1). cancel_all() only sets cancel_events under a
        # short-lived lock; it never blocks on an engine call or a thread join,
        # so it is safe to run here on the audio thread before stop_speaking().
        with self._terminal_effect_lock:
            if self._stopping:
                return
            if not self._acoustic_revisions.accept(
                acoustic, revision, final=False
            ):
                log.debug(
                    "dropping stale/duplicate acoustic barge: revision=%d",
                    revision,
                )
                return
            if self._dispatcher is not None:
                self._dispatcher.cancel_pending()
            self._clear_arrival_continuation()
            self._clear_published_unheard()
            self._clear_partial_fence()
            self.metrics.mark(BARGE_IN)
            self._watchdog.note_barge_in()
            self.supervisor.cancel_all()
            self._post_barge_response.arm(
                self.supervisor.input_epoch,
                acoustic=acoustic,
            )
            self._interrupt_playback_history()
            self.engine.stop_speaking()
            if self._intents is not None:
                self._intents.cancel_all()
            # The reply was interrupted mid-way: arm the resume tracker ("start
            # again"/"continue" now resumes it) + anchor the echo window (the cut
            # playback's tail still reaches the mic for a moment).
            self._resume.note_cut()
            self._resume.note_playback_end()
            self.bus.publish(
                AgentEvent.stop(
                    "barge_in",
                    already_cancelled=True,
                    acoustic=acoustic,
                    revision=revision,
                )
            )

    def _on_command_result(self, result: CommandDetection) -> None:
        self._on_command(
            result.text,
            acoustic=result.acoustic,
            revision=result.revision,
        )

    def _on_command(
        self,
        keyword: str,
        *,
        acoustic: AcousticLineage | None = None,
        revision: int = 0,
    ) -> None:
        # Spotted control phrase: act immediately, bypassing analyzer + LLM.
        # Normalization and the stop-class fall back to the shared contract so
        # the desktop and mobile shells recognize the same control phrases.
        if self._stopping:
            return
        action = self._command_map.get(normalize_command(keyword))
        if action is None and is_stop_command(keyword):
            action = "stop"
        if action is None:
            # A KWS hit without a control mapping is ordinary user speech. Put
            # it through the exact same reservation/preemption/gate lifecycle as
            # an ASR final; bypassing that path left old active answers alive.
            self._on_final(
                keyword,
                acoustic=acoustic,
                revision=revision,
            )
            return
        with self._terminal_effect_lock:
            if self._stopping:
                return
            if not self._acoustic_revisions.accept(
                acoustic, revision, final=True
            ):
                log.debug(
                    "dropping stale/duplicate acoustic command: revision=%d",
                    revision,
                )
                return
            self._apply_mapped_command(
                action,
                acoustic=acoustic,
                revision=revision,
            )

    def _apply_mapped_command(
        self,
        action: str,
        *,
        acoustic: AcousticLineage | None,
        revision: int,
    ) -> None:
        """Apply one accepted command inside ``_terminal_effect_lock``.

        Keeping revision acceptance and the command effect in the same critical
        section makes their ordering linear: a newer partial cannot land in the
        gap and then be cancelled by an older accepted command.
        """
        # A deterministic command consumed the interruption itself; no later
        # ambient final may inherit the post-barge conversational grant.
        self._post_barge_response.invalidate()
        if action == "stop":
            # Same determinism as barge-in: set cancellation before playback is
            # cut so no stale sentence from the interrupted turn is spoken
            # (realtime-concurrency-1).
            with self._terminal_effect_lock:
                if self._stopping:
                    return
                if self._dispatcher is not None:
                    self._dispatcher.cancel_pending()
                self._clear_arrival_continuation()
                self._clear_published_unheard()
                self._clear_partial_fence()
                self.supervisor.cancel_all()
                self._interrupt_playback_history()
                self.engine.stop_speaking()
                if self._intents is not None:
                    self._intents.cancel_all()
                self._resume.note_cut()  # "stop" then "continue" resumes the reply
                self._resume.note_playback_end()
                self.bus.publish(
                    AgentEvent.stop(
                        "command",
                        already_cancelled=True,
                        acoustic=acoustic,
                        revision=revision,
                    )
                )
        elif action == "confirm":
            with self._terminal_effect_lock:
                if self._stopping:
                    return
                input_generation = self._new_input_generation()
                input_epoch = self.supervisor.input_epoch
                if not self.supervisor.commit_input_generation(input_generation):
                    return
                if self._dispatcher is not None:
                    self._dispatcher.cancel_pending()
                self._clear_arrival_continuation()
                self._clear_published_unheard()
                self._clear_partial_fence()
                self.supervisor.cancel_pending_aux_tts()
                if (
                    not self.supervisor.state.pending_confirmations
                    and self.supervisor.has_live_tasks()
                ):
                    self.supervisor.cancel_all(invalidate_inputs=False)
                self.bus.publish(
                    AgentEvent.confirm(
                        "command",
                        # Mapped KWS callbacks are emitted by the live capture
                        # engine, not by model/file/web content. They carry no
                        # biometric verdict, but they are a direct live control.
                        origin="live_audio",
                        direct_user_instruction=True,
                        input_generation=input_generation,
                        input_epoch=input_epoch,
                        acoustic=acoustic,
                        revision=revision,
                    )
                )
        elif action == "deny":
            with self._terminal_effect_lock:
                if self._stopping:
                    return
                input_generation = self._new_input_generation()
                input_epoch = self.supervisor.input_epoch
                if not self.supervisor.commit_input_generation(input_generation):
                    return
                if self._dispatcher is not None:
                    self._dispatcher.cancel_pending()
                self._clear_arrival_continuation()
                self._clear_published_unheard()
                self._clear_partial_fence()
                self.supervisor.cancel_pending_aux_tts()
                if (
                    not self.supervisor.state.pending_confirmations
                    and self.supervisor.has_live_tasks()
                ):
                    self.supervisor.cancel_all(invalidate_inputs=False)
                self.bus.publish(
                    AgentEvent.deny(
                        "command",
                        input_generation=input_generation,
                        input_epoch=input_epoch,
                        acoustic=acoustic,
                        revision=revision,
                    )
                )
        elif action.startswith("mode:"):
            with self._terminal_effect_lock:
                if self._stopping:
                    return
                try:
                    mode = Mode(action.split(":", 1)[1])
                except ValueError:
                    return
                if self._dispatcher is not None:
                    self._dispatcher.cancel_pending()
                self._clear_arrival_continuation()
                self._clear_published_unheard()
                self._clear_partial_fence()
                self.supervisor.cancel_pending_aux_tts()
                input_generation = self._new_input_generation()
                input_epoch = self.supervisor.input_epoch
                if not self.supervisor.commit_input_generation(input_generation):
                    return
                # A mode command is a new authoritative turn, not an add-on.
                # Retire registered/pre-audio work before changing how any older
                # bus event would be interpreted.  Its generation is committed
                # first so an old event that passed the initial bus check cannot
                # register in the has-live -> cancel decision gap.
                if self.supervisor.has_live_tasks():
                    self.supervisor.cancel_all(invalidate_inputs=False)
                self.bus.publish(
                    AgentEvent.mode(
                        mode,
                        source="command",
                        input_generation=input_generation,
                        input_epoch=input_epoch,
                        acoustic=acoustic,
                        revision=revision,
                    )
                )

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

    def _play_admitted_tts(
        self,
        event: AgentEvent,
        *,
        text: str,
        task_id: str,
        epoch: object,
        auxiliary_tts: bool,
        aux_tts_id: str,
        playback_admission: PlaybackAdmission,
    ) -> tuple[tuple[str, bool] | None, str, str]:
        """Own one playback admission until sink or history transfer succeeds."""

        admission_owned = True
        auxiliary_consumed = False
        output_consumed = False
        handoff_succeeded = False
        admitted_output: tuple[str, bool] | None = None
        admitted_reminder_id = ""
        admitted_reminder_token = ""
        style = None
        try:
            if self._reply_voice_continuity is not None:
                reply = None
                if (
                    not auxiliary_tts
                    and task_id
                    and epoch is not None
                    and bool(event.payload.get("streaming", False))
                ):
                    reply = ReplySpeechId(
                        task_id=task_id,
                        epoch=int(epoch),
                        binding_id=int(
                            event.payload.get("binding_id", 0) or 0
                        ),
                    )
                prepared = self._reply_voice_continuity.prepare(
                    text,
                    reply=reply,
                )
                text = prepared.text.strip()
                style = prepared.style
                if not text:
                    return None, "", ""

            if auxiliary_tts:
                if event.payload.get("reminder_due") is True:
                    reminder_id = str(
                        event.payload.get("reminder_id", "") or ""
                    )
                    reminder_token = str(
                        event.payload.get("reminder_claim_token", "") or ""
                    )
                    try:
                        reminder_committed = bool(
                            self._reminder_manager is not None
                            and self._reminder_manager.renew_voice_claim(
                                reminder_id,
                                claim_token=reminder_token,
                            )
                        )
                    except Exception:  # noqa: BLE001 - retry after claim lease
                        reminder_committed = False
                        log.exception("reminder voice-announcement commit failed")
                    if not reminder_committed:
                        log.debug("dropping stale reminder voice claim")
                        return None, "", ""
                    admitted_reminder_id = reminder_id
                    admitted_reminder_token = reminder_token
                self.supervisor.note_aux_tts_admitted(aux_tts_id)
                auxiliary_consumed = True
            elif not event.payload.get("latency_ack", False):
                admitted_output = self.supervisor.note_tts_admitted(
                    task_id,
                    epoch,
                    text,
                    event.payload,
                )
                output_consumed = True

            metrics_turn_token = event.payload.get("metrics_turn_token")
            if metrics_turn_token is None:
                self.metrics.mark(TTS_REQUESTED)
            else:
                self.metrics.mark(
                    TTS_REQUESTED,
                    turn_token=metrics_turn_token,
                )
            log.info(
                "assistant: %r",
                text,
                extra={"transcript": {"role": "assistant", "text": text}},
            )

            if self._playback_history is None:
                self._resume.note_spoken(text)
                self.engine.speak(text)
                handoff_succeeded = True
            else:
                epoch_value = int(
                    epoch if epoch is not None else self.supervisor.speech_epoch
                )
                streaming = bool(
                    not auxiliary_tts and task_id and admitted_output is None
                )
                is_followup = bool(
                    admitted_output[1]
                    if admitted_output is not None
                    else event.payload.get("followup", False)
                )
                with self._playback_effect_lock:
                    explicit_input_generation = event.payload.get("input_generation")
                    playback_input_generation = int(
                        explicit_input_generation
                        if explicit_input_generation is not None
                        else self.supervisor.latest_arrival_generation
                    )
                    fragment_id = self._playback_history.register(
                        task_id=task_id,
                        epoch=epoch_value,
                        input_generation=playback_input_generation,
                        followup_generation=self.supervisor.followup_generation,
                        text=text,
                        remember=bool(not auxiliary_tts and task_id),
                        is_followup=is_followup,
                        streaming=streaming,
                        playback_admission_id=(
                            playback_admission.playback_admission_id
                        ),
                        session_id=playback_admission.identity.session_id,
                        turn_id=playback_admission.identity.turn_id,
                        revision=playback_admission.identity.revision,
                        cancel_generation=(
                            playback_admission.identity.cancel_generation
                        ),
                        binding_id=playback_admission.identity.binding_id,
                    )
                    try:
                        # Both ledgers must own the fragment before the engine
                        # can synchronously callback from speak_tracked().
                        self._resume.stage_playback(fragment_id, text)
                        playback_context = self._playback_history.fragment_context(
                            fragment_id
                        )
                        self._observe_playback_diagnostic(
                            stage="admitted",
                            fragment_id=fragment_id,
                            context=playback_context,
                            playback_kind=(
                                "reminder"
                                if event.payload.get("reminder_due") is True
                                else "latency_ack"
                                if event.payload.get("latency_ack", False)
                                else "auxiliary"
                                if auxiliary_tts
                                else "reply"
                            ),
                            acoustic_identity=(
                                self._diagnostic_acoustic_by_input.get(
                                    playback_input_generation
                                )
                                if not auxiliary_tts
                                and explicit_input_generation is not None
                                else None
                            ),
                        )
                    except BaseException:
                        try:
                            self._rollback_registered_playback(fragment_id)
                        except Exception:  # noqa: BLE001 - retain stage failure
                            log.exception(
                                "failed to roll back staged playback %s",
                                fragment_id,
                            )
                        raise
                    # Actor capacity now belongs to the receipt ledger.
                    admission_owned = False
                try:
                    self.engine.speak_tracked(
                        TrackedSpeech(
                            fragment_id=fragment_id,
                            text=text,
                            style=style,
                        ),
                        on_started=self._on_playback_started,
                        on_terminal=self._on_playback_terminal,
                    )
                    handoff_succeeded = True
                except Exception:
                    self._on_playback_terminal(
                        PlaybackReceipt(
                            fragment_id=fragment_id,
                            outcome=PlaybackOutcome.FAILED,
                        )
                    )
                    raise
            return (
                admitted_output,
                admitted_reminder_id,
                admitted_reminder_token,
            )
        finally:
            if admission_owned:
                self.supervisor.finish_playback(
                    playback_admission.playback_admission_id
                )
            if auxiliary_tts and not auxiliary_consumed:
                self.supervisor.note_aux_tts_admitted(aux_tts_id)
            if not auxiliary_tts and not output_consumed:
                self.supervisor.note_tts_rejected(task_id, event.payload)
            if (
                admitted_reminder_id
                and not handoff_succeeded
                and self._reminder_manager is not None
            ):
                try:
                    self._reminder_manager.release_voice_announcement(
                        admitted_reminder_id,
                        claim_token=admitted_reminder_token,
                    )
                except Exception:  # noqa: BLE001 - retain primary failure
                    log.exception("reminder voice claim release failed")

    # --- bus subscriber ---
    def _on_event(self, event: AgentEvent) -> None:
        if event.kind == EventKind.MAILBOX_FAULT:
            # Supervisor (the earlier subscriber) has already retired actor/task
            # ownership. Cut output immediately, then let a separate thread own
            # full teardown so this bus thread never tries to join itself.
            log.critical("event mailbox fault; stopping voice runtime")
            with self._terminal_effect_lock:
                self._stopping = True
                self._post_barge_response.invalidate()
                self._clear_arrival_continuation()
                self._clear_published_unheard()
                self._clear_partial_fence()
                try:
                    self.engine.stop_speaking()
                except Exception:  # noqa: BLE001 - continue fail-closed teardown
                    log.exception("mailbox-fault speaker cut failed")
            discarded_kinds = event.payload.get("discarded_kinds")
            discarded_memory = (
                discarded_kinds.get(EventKind.MEMORY_COMMIT.value, 0)
                if isinstance(discarded_kinds, Mapping)
                else 0
            )
            if isinstance(discarded_memory, int) and not isinstance(
                discarded_memory,
                bool,
            ):
                with self._playback_effect_changed:
                    self._pending_playback_memory_commits = max(
                        0,
                        self._pending_playback_memory_commits
                        - max(0, discarded_memory),
                    )
                    self._playback_effect_changed.notify_all()
            try:
                Thread(
                    target=self.stop,
                    name="speaker-mailbox-fault-stop",
                    daemon=True,
                ).start()
            except Exception:  # noqa: BLE001
                log.exception("mailbox-fault runtime stop thread failed to start")
            return
        if (
            event.kind
            in {
                EventKind.CONTROL_MODE,
                EventKind.CONTROL_CONFIRM,
                EventKind.CONTROL_DENY,
            }
            or (
                event.kind == EventKind.CONTROL_STOP
                # The barge callback itself publishes this acknowledgement
                # after arming the grant. Other stop boundaries invalidate it.
                and not event.payload.get("already_cancelled", False)
            )
        ):
            self._post_barge_response.invalidate()
            if event.kind == EventKind.CONTROL_MODE:
                # A queued response-only STT_FINAL may already have opened its
                # ASR metrics turn before the higher-priority mode event fences
                # it at the supervisor. Resolve that turn locally so watchdog
                # diagnostics do not report a missing first token.
                self.metrics.mark(HANDLED_LOCAL)
        if (
            event.kind == EventKind.MEMORY_COMMIT
            and event.payload.get("source")
            in {"playback_receipt", "playback_ordered_user"}
        ):
            # Supervisor is subscribed before the runtime, so its memory write
            # has completed (or failed visibly) when this acknowledgement runs.
            with self._playback_effect_changed:
                if self._pending_playback_memory_commits > 0:
                    self._pending_playback_memory_commits -= 1
                self._playback_effect_changed.notify_all()
        terminal_kinds = {
            EventKind.TASK_COMPLETED,
            EventKind.TASK_CANCELLED,
            EventKind.TASK_FAILED,
            EventKind.TTS_STREAM_END,
        }
        terminal_identity: TaskIdentity | None = None
        terminal_success_current = True
        if event.kind in terminal_kinds:
            (
                terminal_accepted,
                terminal_identity,
                terminal_success_current,
            ) = self.supervisor.consume_terminal_event_acceptance(event)
            if not terminal_accepted:
                log.warning(
                    "dropping runtime effects for unaccepted terminal event: %s",
                    event.kind.value,
                )
                return
            if terminal_identity is None:
                try:
                    terminal_identity = TaskIdentity.from_payload(event.payload)
                except (TypeError, ValueError):
                    terminal_identity = None
        if event.kind == EventKind.TASK_COMPLETED and not terminal_success_current:
            # The actor identity was genuine, but a barge-in/superseding turn
            # won before this completion reached the bus.  Close only the exact
            # old binding as interrupted.  Successful-completion effects below
            # (metrics, stream metadata, continuity, and memory) must not run.
            task_id = str(event.payload.get("task_id", ""))
            binding_id = (
                terminal_identity.binding_id
                if terminal_identity is not None
                else None
            )
            if self._reply_voice_continuity is not None:
                self._reply_voice_continuity.close_task(task_id, binding_id)
            if self._playback_history is not None:
                with self._playback_effect_lock:
                    self._publish_playback_commits(
                        self._playback_history.close_task(
                            task_id,
                            interrupted=True,
                            binding_id=binding_id,
                        )
                    )
                    self._playback_effect_changed.notify_all()
                    self._log_playback_finalizations(
                        self._playback_history.drain_finalizations()
                    )
            return
        if event.kind == EventKind.TASK_COMPLETED:
            capability = str(event.payload.get("capability", "") or "")
            data = event.payload.get("data")
            handled_local = bool(
                isinstance(data, dict) and data.get("handled_local") is True
            )
            if handled_local or (
                capability and capability not in _LLM_BACKED_TASK_CAPABILITIES
            ):
                # The brain completed a local/non-LLM capability (dictation,
                # meeting note, local search/command staging), or an explicitly
                # controller-owned assistant.answer. The turn has an ASR_FINAL
                # but legitimately never gets an LLM_FIRST_TOKEN.
                metrics_turn_token = event.payload.get("metrics_turn_token")
                if metrics_turn_token is None:
                    self.metrics.mark(HANDLED_LOCAL)
                else:
                    self.metrics.mark(
                        HANDLED_LOCAL,
                        turn_token=metrics_turn_token,
                    )
            if self._playback_history is not None:
                if isinstance(data, dict) and data.get("streamed"):
                    completed_epoch = event.payload.get("epoch")
                    with self._playback_effect_lock:
                        self._playback_history.note_stream_metadata(
                            str(event.payload.get("task_id", "")),
                            int(
                                completed_epoch
                                if completed_epoch is not None
                                else self.supervisor.speech_epoch
                            ),
                            is_followup=bool(event.payload.get("followup", False)),
                            binding_id=(
                                terminal_identity.binding_id
                                if terminal_identity is not None
                                else 0
                            ),
                        )
        if self._reply_voice_continuity is not None:
            if event.kind == EventKind.TTS_STREAM_END:
                stream_epoch = event.payload.get("epoch")
                self._reply_voice_continuity.close(
                    ReplySpeechId(
                        task_id=str(event.payload.get("task_id", "")),
                        epoch=int(
                            stream_epoch
                            if stream_epoch is not None
                            else self.supervisor.speech_epoch
                        ),
                        binding_id=(
                            terminal_identity.binding_id
                            if terminal_identity is not None
                            else 0
                        ),
                    )
                )
            elif event.kind in {EventKind.TASK_CANCELLED, EventKind.TASK_FAILED}:
                self._reply_voice_continuity.close_task(
                    str(event.payload.get("task_id", "")),
                    (
                        terminal_identity.binding_id
                        if terminal_identity is not None
                        else None
                    ),
                )
            elif event.kind == EventKind.TASK_COMPLETED:
                data = event.payload.get("data")
                if not (isinstance(data, dict) and data.get("streamed")):
                    self._reply_voice_continuity.close_task(
                        str(event.payload.get("task_id", "")),
                        (
                            terminal_identity.binding_id
                            if terminal_identity is not None
                            else None
                        ),
                    )
        if self._playback_history is not None:
            if event.kind == EventKind.TTS_STREAM_END:
                stream_epoch = event.payload.get("epoch")
                stream_task_id = str(event.payload.get("task_id", ""))
                stream_epoch_value = int(
                    stream_epoch
                    if stream_epoch is not None
                    else self.supervisor.speech_epoch
                )
                with self._playback_effect_lock:
                    self._publish_playback_commits(
                        self._playback_history.close_stream(
                            stream_task_id,
                            stream_epoch_value,
                            binding_id=(
                                terminal_identity.binding_id
                                if terminal_identity is not None
                                else 0
                            ),
                        )
                    )
                    self._playback_effect_changed.notify_all()
                    self._log_playback_finalizations(
                        self._playback_history.drain_finalizations()
                    )
            elif event.kind in {EventKind.TASK_CANCELLED, EventKind.TASK_FAILED}:
                task_id = str(event.payload.get("task_id", ""))
                with self._playback_effect_lock:
                    self._publish_playback_commits(
                        self._playback_history.close_task(
                            task_id,
                            interrupted=True,
                            binding_id=(
                                terminal_identity.binding_id
                                if terminal_identity is not None
                                else None
                            ),
                        )
                    )
                    self._playback_effect_changed.notify_all()
                    self._log_playback_finalizations(
                        self._playback_history.drain_finalizations()
                    )
        if event.kind == EventKind.TTS_REQUEST:
            text = str(event.payload.get("text", "")).strip()
            if not text:
                return
            admitted_output: tuple[str, bool] | None = None
            admitted_reminder_id = ""
            admitted_reminder_token = ""
            with self._terminal_effect_lock:
                # Shutdown gate: stop() sets _stopping before the bus thread is
                # joined, so a reply racing teardown is dropped here instead of
                # starting playback that engine.stop() then has to kill.
                if self._stopping:
                    log.debug("dropping TTS_REQUEST during shutdown: %r", text)
                    return
                # Keep admission + playback atomic against barge/stop so a
                # sentence cannot pass the epoch check and start after the
                # interrupt has already returned.
                task_id = str(event.payload.get("task_id", ""))
                epoch = event.payload.get("epoch")
                auxiliary_tts = bool(event.payload.get("auxiliary_tts", False))
                aux_tts_id = str(event.payload.get("aux_tts_id", ""))
                allowed = (
                    self.supervisor.auxiliary_tts_allowed(aux_tts_id, epoch)
                    if auxiliary_tts
                    else self.supervisor.tts_request_allowed(
                        task_id,
                        epoch,
                        event.payload,
                    )
                )
                if not allowed:
                    if auxiliary_tts:
                        # Rejected emissions are terminal too.  Leaving their
                        # unique ID registered would keep wait_idle/follow-ups
                        # blocked forever after a cancellation race.
                        self.supervisor.note_aux_tts_admitted(aux_tts_id)
                    else:
                        self.supervisor.note_tts_rejected(
                            task_id,
                            event.payload,
                        )
                    log.debug(
                        "dropping stale TTS_REQUEST (task_id=%r): %r",
                        task_id,
                        text,
                    )
                    return
                playback_admission = self.supervisor.admit_playback(
                    event.payload,
                    task_id=task_id,
                    auxiliary_tts=auxiliary_tts,
                    aux_tts_id=aux_tts_id,
                )
                if playback_admission is None:
                    if auxiliary_tts:
                        self.supervisor.note_aux_tts_admitted(aux_tts_id)
                    else:
                        self.supervisor.note_tts_rejected(
                            task_id,
                            event.payload,
                        )
                    log.debug(
                        "dropping unowned/bounded TTS_REQUEST (task_id=%r): %r",
                        task_id,
                        text,
                    )
                    return
                (
                    admitted_output,
                    admitted_reminder_id,
                    admitted_reminder_token,
                ) = self._play_admitted_tts(
                    event,
                    text=text,
                    task_id=task_id,
                    epoch=epoch,
                    auxiliary_tts=auxiliary_tts,
                    aux_tts_id=aux_tts_id,
                    playback_admission=playback_admission,
                )
            if admitted_reminder_id and self._reminder_manager is not None:
                try:
                    committed = self._reminder_manager.mark_voice_announced(
                        admitted_reminder_id,
                        claim_token=admitted_reminder_token,
                    )
                    if not committed:
                        log.warning("reminder voice-announcement claim changed after handoff")
                except Exception:  # noqa: BLE001 - handoff succeeded; retry is safe
                    log.exception("reminder voice-announcement handoff commit failed")
            if admitted_output is not None and self._playback_history is None:
                output, is_followup = admitted_output
                self.supervisor.record_admitted_output(
                    output,
                    is_followup=is_followup,
                )
        elif event.kind == EventKind.CONTROL_STOP:
            if event.payload.get("already_cancelled", False):
                return
            with self._terminal_effect_lock:
                # Runtime owns analyzer STOP end-to-end: generation validation,
                # dispatcher retirement, supervisor cancellation, and physical
                # playback cut share this terminal critical section.  Splitting
                # those effects across subscribers let a newer turn land between
                # the state cancel and engine stop.
                if not self.supervisor.control_event_current(event.payload):
                    return
                if self._dispatcher is not None:
                    self._dispatcher.cancel_pending()
                if not self.supervisor.cancel_all_if_current(event.payload):
                    return
                self._clear_arrival_continuation()
                self._clear_published_unheard()
                self._clear_partial_fence()
                self._interrupt_playback_history()
                self.engine.stop_speaking()
                self._resume.note_cut()  # a spoken "stop" arms resume too
                self._resume.note_playback_end()
                # Stop also cancels pending fast-path actions (e.g. a running timer).
                if self._intents is not None:
                    self._intents.cancel_all()

    # --- synchronous helper for tests / console demo ---
    def wait_idle(
        self,
        timeout: float = 3.0,
        *,
        include_playback: bool = True,
    ) -> bool:
        """Wait until the turn is terminal, including tracked playout by default.

        ``include_playback=False`` is the explicit test/console seam for callers
        that need to wait only until the brain has begun a held utterance so they
        can inject a stop or barge-in.

        rc-1 fix: when the bus runs on its own thread (``run_bus=True`` -- the
        replay harness path), this must NOT also drain the queue from here:
        two threads racing ``get_nowait`` raised uncaught ``queue.Empty`` and
        double-dispatched events, poisoning the replay measurement loop. With
        the bus thread running we POLL ``bus.idle()``; only the test path
        (``run_bus=False``) pumps the queue from this thread."""

        def _playback_quiet() -> bool:
            if not include_playback:
                return True
            if self.supervisor.session_actor.active_playback_count:
                return False
            if self._playback_history is None:
                return True
            with self._playback_effect_lock:
                return bool(
                    not self._playback_history.pending
                    and not self._pending_playback_memory_commits
                )

        def _quiet() -> bool:
            return (
                (self._dispatcher is None or not self._dispatcher.has_pending)
                and self.bus.idle()
                and not self.supervisor.state.active_tasks
                and not self.supervisor.state.pending_audio_tasks
                and not self.supervisor.state.pending_aux_tts
                and not self.supervisor.state.queued_tasks
                and self.supervisor.tasks.active_count == 0
                and self.supervisor.session_actor.active_task_count == 0
                and self.supervisor.session_actor.active_provider_count == 0
                and self.supervisor.session_actor.active_tool_count == 0
                and _playback_quiet()
            )

        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self._bus_threaded:
                self.bus.drain()
            if _quiet():
                if not self._bus_threaded:
                    self.bus.drain()
                if _quiet():
                    return True
            time.sleep(0.005)
        if not self._bus_threaded:
            self.bus.drain()
        return _quiet()
