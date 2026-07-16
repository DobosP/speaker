from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from typing import Callable, Optional

from always_on_agent.continuation import ContinuationConfig
from always_on_agent.events import Mode
from always_on_agent.followups import FollowupConfig
from always_on_agent.memory import Memory, SessionMemory
from always_on_agent.react import PlannerConfig

from .capabilities import RecallConfig
from .config import (
    _apply_device_profile,
    _load_config,
    apply_device_profile,
    load_config,
    resolve_device,
)
from .engine import AudioEngine
from .llm import LLMClient, OllamaLLM
from .llm_factory import (
    _build_cloud_client,
    _build_llms,
    _preset_host,
    _wrap_cloud,
    build_llms,
    build_router_llm,
)
from .obsidian import ObsidianConfig
from .routing import build_router
from .capability_router import build_capability_router
from .runlog import setup_logging
from .runtime import VoiceRuntime
from .sysinfo import SystemMonitor
from .websearch import WebSearchConfig

log = logging.getLogger("speaker.app")


# Config loading + device-profile layering now live in ``core/config.py``; the
# LLM construction factory (``build_llms`` + the cloud-wrapping helpers) lives in
# ``core/llm_factory.py``. They are imported at module top and re-exported here so
# existing callers/tests that reach for ``core.app._load_config`` /
# ``core.app._build_llms`` / ``core.app._wrap_cloud`` keep working unchanged.


def _post_barge_response_window(config: dict) -> float:
    """Parse the optional response-only window, failing closed on bad config."""

    block = config.get("post_barge_response", {}) or {}
    if not isinstance(block, dict) or not block.get("enabled", True):
        return 0.0
    try:
        value = float(block.get("window_sec", 8.0))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0.0 else 0.0


def _require_asr_models(sherpa_cfg, engine_name: str) -> None:
    """Fail fast with an actionable message when the sherpa ASR models aren't
    configured, instead of starting an engine that can never hear anything."""
    if not getattr(sherpa_cfg, "asr_encoder", "") or not getattr(sherpa_cfg, "asr_tokens", ""):
        raise SystemExit(
            f"\n--engine {engine_name} needs the on-device speech models, but "
            "config.json/config.local.json have no sherpa model paths set.\n\n"
            "  Run once:   python -m tools.setup_models\n"
            "  (or full:   ./install.sh)\n\n"
            "That downloads the ASR/VAD/TTS models and writes their paths into "
            f"config.local.json. Then re-run:\n  python -m core --engine {engine_name}\n"
        )


def _require_sherpa_runtime_ready(
    config: dict,
    llm_mode: str,
    *,
    models_needed=None,
    run_checks=None,
    virtual_audio_binder=None,
) -> None:
    """Fail before model/device construction when sherpa cannot run correctly.

    Kept as a narrow CLI seam: console/replay/livekit and one-shot enrollment do
    not call it. ``run_checks`` is injectable so the control-plane decision is
    unit-tested without probing this test runner's audio devices.
    """
    if run_checks is None:
        from .readiness import run_runtime_checks

        run_checks = run_runtime_checks
    check_kwargs = {
        "resolved": True,
        "llm_mode": llm_mode,
        "models_needed": models_needed,
    }
    # The private delay harness supplies an already-validated, run-owned
    # topology. Keep this kwarg absent on every production call so injected
    # test doubles and the normal readiness contract remain byte-for-byte
    # compatible.
    if virtual_audio_binder is not None:
        check_kwargs["virtual_audio_binder"] = virtual_audio_binder
    checks = run_checks(config, **check_kwargs)
    failures = [check for check in checks if not check.ok]
    if not failures:
        return
    lines = []
    for check in failures:
        line = f"  - {check.name}: {check.detail}".rstrip()
        if check.hint:
            line += f"\n      fix: {check.hint}"
        lines.append(line)
    raise SystemExit(
        "\n--engine sherpa runtime preflight failed:\n"
        + "\n".join(lines)
        + "\n\nRun `python -m tools.doctor` after fixing these prerequisites."
    )


def _build_engine(args, config: dict, *, virtual_audio_binder=None) -> AudioEngine:
    if args.engine == "sherpa":
        from .engines.sherpa import SherpaConfig, SherpaOnnxEngine

        sherpa_cfg = SherpaConfig.from_dict(config.get("sherpa", {}))
        _require_asr_models(sherpa_cfg, "sherpa")
        return SherpaOnnxEngine(
            sherpa_cfg,
            virtual_audio_binder=virtual_audio_binder,
        )
    if args.engine == "replay":
        # Headless: run the real recognizer + TTS over recorded audio (no sound
        # card), for latency measurement on a server/CI. See tools/bench.
        from .engines.file_replay import FileReplayEngine
        from .engines.sherpa import SherpaConfig

        sherpa_cfg = SherpaConfig.from_dict(config.get("sherpa", {}))
        _require_asr_models(sherpa_cfg, "replay")
        return FileReplayEngine(sherpa_cfg)
    if args.engine == "livekit":
        # Remote (WebRTC) engine: same sherpa STT/TTS, audio over a LiveKit room.
        # URL + token come from the env so the base CLI stays decoupled from the
        # optional remote package; `python -m remote.worker` mints the token for
        # you. Install requirements-remote.txt to run it.
        from .engines.livekit import LiveKitEngine
        from .engines.sherpa import SherpaConfig

        url = os.environ.get("LIVEKIT_URL")
        token = os.environ.get("LIVEKIT_TOKEN")
        if not url or not token:
            raise SystemExit(
                "--engine livekit needs LIVEKIT_URL and LIVEKIT_TOKEN in the env. "
                "Run `python -m remote.worker` to mint a token automatically "
                "(after installing requirements-remote.txt and starting a LiveKit server)."
            )
        sherpa_cfg = SherpaConfig.from_dict(config.get("sherpa", {}))
        _require_asr_models(sherpa_cfg, "livekit")
        room = (config.get("remote", {}) or {}).get("room", "assistant")
        return LiveKitEngine(sherpa_cfg, url=url, token=token, room=room)
    from .engines.scripted import ScriptedEngine

    return ScriptedEngine()


def _apply_autotest_virtual_delay_profile(
    config: dict,
    *,
    input_device: str = "pipewire",
    output_device: str = "pipewire",
) -> dict:
    """Return the isolated synthetic-user profile for the silent delay gate.

    This seam is reachable only after the hidden CLI contract has been loaded
    and validated. It deliberately owns no persisted config or enrollment:
    synthetic clips exercise OS-EC + lexical word-cut/control flow, while owner
    identity remains the separate recorded/live gate.
    """

    merged = dict(config)
    sherpa = dict(config.get("sherpa", {}) or {})
    sherpa.update(
        input_device=input_device,
        output_device=output_device,
        aec_enabled=False,
        capture_voice_comm=False,
        barge_in_enabled=True,
        barge_word_cut_enabled=True,
        barge_word_cut_require_speaker=False,
        # The physical-device-free gate is the initial opt-in for the calibrated
        # energy/VAD substitution. Its private capture graph supplies a complete
        # pre-gain ambient window before scripted speech begins.
        input_agc=True,
        input_calibrate=True,
        input_calibrate_sec=1.5,
        barge_word_cut_energy_fallback_enabled=True,
        # The private native EC left one command block just above +9 dB and
        # starved the three-block gate. +6 dB retains the sustained debounce;
        # the production/library default remains disabled and unchanged.
        barge_word_cut_energy_margin_db=6.0,
        barge_word_cut_energy_min_blocks=3,
        # Synthetic speech has no owner embedding. Retain the production lexical
        # safety floor (observed garbled echo is 2-3 words) and require the S2
        # no-near-end scenario to prove zero self-cuts; identity remains a
        # separate recorded/physical gate.
        barge_word_cut_min_words=4,
        speaker_gate_input=False,
        speaker_embedding_model="",
        speaker_enroll_wav="",
        speaker_enroll_embedding="",
        release_output_when_idle=False,
    )
    merged["sherpa"] = sherpa
    llm = dict(config.get("llm", {}) or {})
    cloud = dict(llm.get("cloud", {}) or {})
    cloud.update(enabled=False, strategy="local_only")
    llm.update(
        host="http://127.0.0.1:11434",
        cloud=cloud,
        live_routing=False,
    )
    merged["llm"] = llm
    web_search = dict(config.get("web_search", {}) or {})
    web_search["enabled"] = False
    merged["web_search"] = web_search
    obsidian = dict(config.get("obsidian", {}) or {})
    obsidian["enabled"] = False
    merged["obsidian"] = obsidian
    screen_capture = dict(config.get("screen_capture", {}) or {})
    screen_capture.update(enabled=False, memorize=False)
    merged["screen_capture"] = screen_capture
    gui_actions = dict(config.get("gui_actions", {}) or {})
    gui_actions["enabled"] = False
    merged["gui_actions"] = gui_actions
    watch = dict(config.get("watch", {}) or {})
    watch.update(enabled=False, grants=[])
    merged["watch"] = watch
    memory = dict(config.get("memory", {}) or {})
    memory.update(
        backend="inmemory",
        embeddings=False,
        profile_enabled=False,
        procedural_enabled=False,
        cross_session_continuity=False,
        persist_assistant=False,
    )
    merged["memory"] = memory
    agent_brain = dict(config.get("agent_brain", {}) or {})
    agent_brain.update(local_only=True, offline=True, os_mode=False)
    merged["agent_brain"] = agent_brain
    return merged


def _redact_db_url(db_url: str) -> str:
    """Mask any password in a ``DATABASE_URL`` for safe logging (R12).

    Reuses ``setup_database._redact_db_url`` when importable; otherwise falls
    back to a host-only form. A connect error string may echo the DSN, so we
    only ever log the redacted URL -- never the raw error."""
    try:
        from setup_database import _redact_db_url as _redact  # type: ignore

        return _redact(db_url)
    except Exception:  # noqa: BLE001 - never let redaction failure leak the URL
        from urllib.parse import urlsplit, urlunsplit

        try:
            parts = urlsplit(db_url)
            if parts.password is None:
                return db_url
            user = parts.username or ""
            host = parts.hostname or ""
            netloc = f"{user}:***@{host}" if user else f"***@{host}"
            if parts.port:
                netloc = f"{netloc}:{parts.port}"
            return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
        except Exception:  # noqa: BLE001
            return db_url.split("@")[-1] if "@" in db_url else db_url


def _build_recall_budget(mem_cfg: dict):
    """Build the shared :class:`~always_on_agent.recall.RecallBudget` from the
    flat ``memory`` config block (already device-profile shallow-merged by the
    caller). Injected into BOTH backends so the in-RAM and Postgres recall paths
    are bounded by ONE config-derived token budget (true parity). Defaults match
    the dataclass so an absent block preserves today's volume; the ``phone``
    device profile can dial ``recall_max_tokens`` down without touching code."""
    from always_on_agent.recall import RecallBudget

    # Resolve max_tokens with the SAME legacy fallback RecallConfig.from_dict uses
    # (recall_max_tokens, else deprecated recall_max_chars//4, else 150) so the
    # memory budget and the injection-site cap derive from ONE value and can never
    # diverge -- a divergence would let the injection trim run looser than the
    # budget and inject an over-budget block.
    tok = mem_cfg.get("recall_max_tokens")
    if tok is None:
        ch = mem_cfg.get("recall_max_chars")
        tok = (int(ch) // 4) if ch else 150
    return RecallBudget(
        max_tokens=max(1, int(tok)),
        chars_per_token=float(mem_cfg.get("chars_per_token", 4.0) or 4.0),
        cutoff_k=float(mem_cfg.get("recall_cutoff_k", 0.0) or 0.0),
        dedup_ratio=float(mem_cfg.get("recall_dedup_ratio", 0.0) or 0.0),
        # Multi-signal recall scoring (recency-decay + kind-importance), default-OFF
        # (weights 0 -> relevance-only, byte-identical). recency is single-backend
        # by design (backends stamp different wall-clocks). See RecallBudget.
        recency_weight=float(mem_cfg.get("recall_recency_weight", 0.0) or 0.0),
        recency_half_life_days=float(mem_cfg.get("recall_recency_half_life_days", 7.0) or 7.0),
        importance_weight=float(mem_cfg.get("recall_importance_weight", 0.0) or 0.0),
    )


def _build_memory(config: dict, fast_llm: LLMClient | None = None) -> Memory:
    """Pick the memory backend (Locked Decision 4).

    Postgres-backed :class:`MemoryManagerAdapter` when ``memory.backend`` is
    ``postgres`` -- or when it is ``auto``/unset and ``$DATABASE_URL`` is set;
    in-RAM :class:`SessionMemory` otherwise. The adapter build is wrapped so a
    missing dependency (``ImportError``) or a connect failure falls back to the
    in-RAM store rather than crashing. Any connect error is logged **redacted**
    (R12) -- the message may carry the DSN.

    ``fast_llm`` (the snappy local tier) backs the P2b rolling-summary
    ``summarizer`` closure passed into the adapter; the actual LLM call runs on
    the ``MemoryWriter`` background thread, never the bus thread (R2). When it is
    ``None`` (e.g. ``--llm echo``) the manager falls back to keyword summaries.
    """
    mem_cfg = config.get("memory", {}) or {}
    # In-RAM working window: the addressing/cleaner recent buffer read via
    # memory.all()[-4:]. Independent of the Postgres Layer-1 cap, so size it from
    # its own optional key (default 200) rather than coupling it to max_recent.
    working_window = int(mem_cfg.get("working_window", 200) or 200)
    recall_budget = _build_recall_budget(mem_cfg)
    backend = str(mem_cfg.get("backend", "auto") or "auto").lower()
    db_url = os.environ.get("DATABASE_URL")

    # Persistent on-device SQLite backend (Decision D6): desktop-without-Postgres
    # continuity across restarts. Same Memory protocol + shared recall selection
    # as the in-RAM store; rows just survive the process. Degrades to in-RAM on
    # any error rather than crashing.
    if backend == "sqlite":
        try:
            from always_on_agent.sqlite_memory import SqliteVecMemory

            sqlite_path = os.path.expanduser(
                str(mem_cfg.get("sqlite_path") or "~/.speaker/memory.db")
            )
            if sqlite_path != ":memory:":
                os.makedirs(os.path.dirname(sqlite_path) or ".", exist_ok=True)
            return SqliteVecMemory(
                sqlite_path,
                max_items=working_window,
                budget=recall_budget,
                ttl_days=int(mem_cfg.get("episodic_ttl_days", 90) or 90),
            )
        except Exception as exc:  # noqa: BLE001 - degrade to in-RAM, never crash
            print(
                f"[memory] SQLite backend unavailable ({type(exc).__name__}); "
                "falling back to in-RAM."
            )
            return SessionMemory(max_items=working_window, budget=recall_budget)

    want_postgres = backend == "postgres" or (backend in ("auto", "") and bool(db_url))
    if not want_postgres:
        return SessionMemory(max_items=working_window, budget=recall_budget)

    # Rolling-summary closure over the fast tier. Built here where fast_llm is
    # available; the manager schedules the call on its writer thread (R2), so
    # this wrapper itself never runs on the bus thread.
    summarizer: Optional[Callable[[str], str]] = None
    if fast_llm is not None:
        def summarizer(text: str) -> str:
            return fast_llm.generate(text)

    try:
        from always_on_agent.memory import MemoryManagerAdapter
        from utils.memory_config import MemoryWriterConfig
        from utils.memory_writer import LLMClientMemoryCleanup

        # The Postgres writer's transcript cleanup/gate is an Ollama-specific
        # structured call. Reuse the *actual* fast Ollama role so persistence
        # cannot silently load a third, unprovisioned model beside the shipped
        # MiniCPM/Gemma pair. Non-Ollama/echo profiles retain the deterministic
        # junk, confidence, echo, control-phrase, and dedupe filters without
        # attempting a localhost Ollama call they cannot satisfy.
        fast_ollama_model = (
            str(fast_llm.model or "").strip()
            if isinstance(fast_llm, OllamaLLM)
            else ""
        )
        writer_llm_client = (
            LLMClientMemoryCleanup(fast_llm) if fast_ollama_model else None
        )
        writer_config = MemoryWriterConfig(
            llm_cleanup=bool(fast_ollama_model),
            llm_gate=bool(fast_ollama_model),
            cleanup_model=fast_ollama_model or MemoryWriterConfig().cleanup_model,
        )

        return MemoryManagerAdapter(
            summarizer=summarizer,
            profile_enabled=bool(mem_cfg.get("profile_enabled", False)),
            episodic_ttl_days=int(mem_cfg.get("episodic_ttl_days", 90) or 90),
            summary_ttl_days=int(mem_cfg.get("summary_ttl_days", 365) or 365),
            enable_embeddings=bool(mem_cfg.get("embeddings", False)),
            max_recent_messages=int(mem_cfg.get("max_recent", 20) or 20),
            recall_budget=recall_budget,
            # lm-7: size the all() ring off the SAME working_window as the other
            # backends so the addressing/cleaner recent buffer is consistent.
            working_window=working_window,
            # lm-2: cross-session continuity (default OFF) -- seed the rolling
            # summary head + fall back to prior-session recent messages on a fresh
            # process so memory survives a restart on the Postgres tier.
            cross_session_continuity=bool(mem_cfg.get("cross_session_continuity", False)),
            # lm-5: persist + recall assistant finals (default OFF) so the Postgres
            # tier matches the in-RAM/SQLite backends.
            persist_assistant=bool(mem_cfg.get("persist_assistant", False)),
            memory_writer_config=writer_config,
            memory_writer_llm_client=writer_llm_client,
        )
    except Exception as exc:  # noqa: BLE001 - degrade to in-RAM, never crash
        redacted = _redact_db_url(db_url) if db_url else "(no DATABASE_URL)"
        print(
            f"[memory] Postgres backend unavailable ({type(exc).__name__}); "
            f"falling back to in-RAM. db={redacted}"
        )
        return SessionMemory(max_items=working_window, budget=recall_budget)


def _run_console(runtime: VoiceRuntime, engine) -> None:
    print(f"[console] mode={runtime.mode.value}. Type to talk; Ctrl-D to quit.")
    print("[console] try: 'research mode', 'assistant please help', 'stop'")
    spoken_seen = 0
    try:
        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue
            engine.final(text)
            runtime.wait_idle()
            for utterance in engine.spoken[spoken_seen:]:
                print(f"assistant> {utterance}")
            spoken_seen = len(engine.spoken)
            print(f"[mode={runtime.mode.value}]")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        runtime.stop()


def _run_replay(runtime: VoiceRuntime, engine, replay_dir: str) -> None:
    import glob

    from .engines.file_replay import load_waveform

    paths = sorted(glob.glob(os.path.join(replay_dir, "*.npy")))
    paths += sorted(glob.glob(os.path.join(replay_dir, "*.wav")))
    if not paths:
        raise SystemExit(f"No .npy/.wav fixtures found in {replay_dir!r}")
    runtime.start(run_bus=True)
    print(f"[replay] {len(paths)} fixture(s) from {replay_dir}")
    try:
        for path in paths:
            samples, sample_rate = load_waveform(path)
            runtime.metrics.close_turn()
            engine.replay_samples(samples, sample_rate)
            runtime.wait_idle(timeout=30.0)
        runtime.metrics.close_turn()
        for i, record in enumerate(runtime.metrics.records()):
            fa = record.first_audio_latency
            print(f"  turn {i}: first_audio={fa:.3f}s" if fa is not None
                  else f"  turn {i}: (incomplete) {record.as_dict()}")
    finally:
        runtime.stop()


def _run_live(runtime: VoiceRuntime) -> None:
    import time

    # start() is inside the try so a failure mid-startup still flushes the
    # recorder + logs via runtime.stop() in finally (artifacts on crash too).
    try:
        runtime.start(run_bus=True)
        print(f"[live] engine running, mode={runtime.mode.value}. Ctrl-C to quit.")
        while True:
            failure = str(
                getattr(runtime.engine, "virtual_route_failure", "") or ""
            )
            if failure:
                raise RuntimeError(f"virtual audio route proof lost: {failure}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop()


def build_runtime(
    config: dict,
    *,
    engine: AudioEngine,
    llm: LLMClient,
    fast_llm: LLMClient | None,
    router,
    start_mode: Mode,
    agent_on: bool = False,
    gui_actions_on: bool = False,
    force_planner: bool = False,
    force_stream_tts: bool = False,
    load_fraction=None,
) -> VoiceRuntime:
    """Assemble the full VoiceRuntime from config + the built clients/engine.

    The single assembly point shared by the CLI (``main``) and the live
    on-hardware validation harness (``tools/live_session``), so both drive the
    exact same assistant (same continuation / capability-catalog / never-stuck /
    endpoint / context-aggregation wiring)."""
    agent_config = None
    if agent_on:
        from .agent import AgentBrainConfig

        agent_config = AgentBrainConfig.from_dict(config.get("agent_brain"))

    # Computer-use (read-only screen.identify) -- default OFF, separate opt-in from
    # the action brain. Enabled by the gui_actions config block OR the --gui-actions
    # flag; the runtime only registers the read-only capability (no actuator).
    gui_cfg = dict(config.get("gui_actions", {}) or {})
    computer_use_config = gui_cfg if (gui_cfg.get("enabled") or gui_actions_on) else None
    # Watch/monitor capability (default OFF). Grants live machine-locally in
    # config.local.json; base config.json ships enabled:false + grants:[].
    watch_cfg = dict(config.get("watch", {}) or {})
    if computer_use_config is not None:
        computer_use_config["enabled"] = True

    planner_config = PlannerConfig.from_dict(
        (config.get("agent", {}) or {}).get("planner")
    )
    if force_planner:
        planner_config.enabled = True

    stream_tts = bool((config.get("tts", {}) or {}).get("streaming", False)) or force_stream_tts
    followup_config = FollowupConfig.from_dict(config.get("followups"))
    # ADD-ON / continuation: merge a follow-up into the in-flight turn instead of
    # racing a competing cold task. Shipped on (config.json); a missing block
    # defaults off via from_dict so non-default configs stay unaffected.
    continuation_config = ContinuationConfig.from_dict(config.get("continuation"))
    # Hold-and-merge final dispatch: an incomplete-reading final waits briefly
    # for the user's next words instead of being answered as a fragment, and all
    # final processing moves off the audio capture thread. Shipped on
    # (config.json); a missing block defaults off via from_dict.
    from .turn_merge import TurnMergeConfig

    turn_merge_config = TurnMergeConfig.from_dict(config.get("turn_merge"))
    # Resume-after-interrupt + self-echo guard: "start again"/"continue" after
    # a cut resumes the reply; a final that is the assistant's own TTS echo is
    # dropped. Default on; the "resume" block overrides.
    from .resume import ResumeConfig

    resume_config = ResumeConfig.from_dict(config.get("resume"))

    intents_cfg = config.get("intents", {}) or {}
    intents = None
    if intents_cfg.get("enabled", False):
        from .intents import LocalIntentHandler

        intents = LocalIntentHandler(engine.speak, phrases=intents_cfg.get("phrases"))

    # Input gate: optional ACT/INGEST classifier between ASR_FINAL and the
    # brain (core/addressing.py). Requires a fast LLM client; if either is
    # missing, the runtime falls back to legacy behavior (every final acts).
    input_gate_cfg = config.get("input_gate", {}) or {}
    addressing = None
    if input_gate_cfg.get("enabled", False) and fast_llm is not None:
        from .addressing import LLMAddressingClassifier

        addressing = LLMAddressingClassifier(
            fast_llm, max_context=int(input_gate_cfg.get("max_context", 4))
        )
    unsure_acts = bool(input_gate_cfg.get("unsure_acts", True))
    post_barge_response_window_sec = _post_barge_response_window(config)

    # Transcript cleanup: optional LLM rewrite that drops fillers / resolves
    # self-corrections so the brain acts on the intended sentence. Same
    # opt-in shape as the input gate; needs llm.fast_model.
    cleanup_cfg = config.get("cleanup", {}) or {}
    cleaner = None
    if cleanup_cfg.get("enabled", False) and fast_llm is not None:
        from .cleanup import LLMTranscriptCleaner

        cleaner = LLMTranscriptCleaner(
            fast_llm, max_context=int(cleanup_cfg.get("max_context", 3))
        )

    # Assistant persona: optional name + character, so the model knows what it
    # is (its skills are enumerated automatically from the capability manifest).
    # Empty by default -> the anonymous "local voice assistant" identity.
    from .persona import PersonaConfig

    persona = PersonaConfig.from_dict(config.get("assistant"))

    memory = _build_memory(config, fast_llm)
    recall_config = RecallConfig.from_dict(config.get("memory"))
    # Short-term conversation context for the answering model (the recent turns,
    # so it can resolve "its"/"the second one"). Default ON; tuned in the memory
    # block. Distinct from semantic recall (past sessions) above.
    from .conversation import RecentContextConfig

    recent_context_config = RecentContextConfig.from_dict(config.get("memory"))
    # Headroom-aware live routing (smart-routing-2 + load follow-up). Flat flag
    # (P4 deep-merge note) so a device-profile override survives the merge;
    # default OFF so default behaviour is byte-identical. When on, the runtime
    # feeds the router the rolling local TTFT plus a cheap SystemMonitor load
    # snapshot (reads the last background sample -- no hot-path sampling) as an
    # additive-only, clamped nudge toward main/cloud (core/routing.py).
    live_routing = bool((config.get("llm", {}) or {}).get("live_routing", False))
    # Pluggable web.search (P3). The flat ``web_search`` block is disabled by
    # default, so the runtime ships corpus-only until a user points base_url at
    # their self-hosted SearXNG. Every query is gated (§9.7) before any egress.
    web_search_config = WebSearchConfig.from_dict(config.get("web_search"))
    # Bounded, read-only access to the local Obsidian vault. The committed Linux
    # path uses ``~`` and can be replaced machine-locally without baking an
    # absolute host path into the repository.
    obsidian_config = ObsidianConfig.from_dict(config.get("obsidian"))

    # Unified capability router (the "middle layer", core/capability_router.py).
    # Disabled by default; when enabled it backs the runtime's tier + escalate
    # decisions with one coherent module. Reuses the tier router built above and
    # the fast tier for optional LLM disambiguation of ambiguous turns, and the
    # configured command phrases so it recognizes the same control words.
    # P3: an optional dedicated (function-calling-tuned) LOCAL model for the routing
    # slot; defaults to the fast tier (byte-identical) when llm.router_model is unset.
    router_llm = build_router_llm(config, fast_llm)
    capability_router = build_capability_router(
        config,
        tier_router=router,
        fast_llm=router_llm,
        command_phrases=(config.get("commands") or {}).keys(),
    )

    runtime = VoiceRuntime(
        engine,
        llm,
        fast_llm=fast_llm,
        memory=memory,
        recall_config=recall_config,
        recent_context_config=recent_context_config,
        web_search_config=web_search_config,
        obsidian_config=obsidian_config,
        start_mode=start_mode,
        agent_config=agent_config,
        computer_use_config=computer_use_config,
        watch_config=watch_cfg,
        router=router,
        capability_router=capability_router,
        planner_config=planner_config,
        stream_tts=stream_tts,
        followup_config=followup_config,
        continuation_config=continuation_config,
        turn_merge_config=turn_merge_config,
        resume_config=resume_config,
        command_map=config.get("commands"),
        intents=intents,
        addressing=addressing,
        unsure_acts=unsure_acts,
        cleaner=cleaner,
        live_routing=live_routing,
        load_snapshot=load_fraction if live_routing else None,
        # control-plane-2: ungated load reader for load-elastic task admission
        # (separate from the live_routing-gated load_snapshot above).
        admission_load=load_fraction,
        # Pre-warm the LLM tiers at startup so turn 1 doesn't pay the ~3s model
        # cold-load on the user's first utterance (lat-2). On by default; set
        # config.warm_on_start=false to skip (e.g. to measure cold start).
        warm_on_start=bool(config.get("warm_on_start", True)),
        persona=persona,
        # Per-mode wall-clock task deadlines (never-stuck backstop). Optional --
        # the supervisor bakes in sensible defaults; config overrides per mode.
        task_timeouts=config.get("task_timeouts"),
        # TTL for staged owner confirmations (same never-stuck family): an
        # abandoned "Confirm command: ..." expires with a spoken notice instead
        # of waiting forever for a stray "yes". 0 disables.
        confirmation_ttl_sec=float(config.get("confirmation_ttl_sec", 180.0)),
        post_barge_response_window_sec=post_barge_response_window_sec,
    )

    return runtime


def _sigterm_to_keyboard_interrupt(signum, frame):  # noqa: ARG001 - signal ABI
    """Turn SIGTERM into the Ctrl-C shutdown path. Without this a killed run
    dies mid-flight and every ``finally`` is skipped: ``runtime.stop()`` never
    flushes the WAV recorders and ``runlog.finalize()`` never writes the
    summary -- exactly how run-20260706-231226 lost its audio evidence."""
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    # Best-effort: handlers may only be installed from the main thread, and
    # some embedders run main() elsewhere -- shutdown then relies on Ctrl-C.
    try:
        import signal

        signal.signal(signal.SIGTERM, _sigterm_to_keyboard_interrupt)
    except (ValueError, OSError):  # noqa: PERF203 - non-main thread / platform quirk
        pass
    parser = argparse.ArgumentParser(description="Lean local voice assistant runtime")
    parser.add_argument(
        "--engine", choices=["console", "sherpa", "livekit", "replay"], default="console"
    )
    parser.add_argument(
        "--replay-dir",
        dest="replay_dir",
        default=None,
        help="with --engine replay: a dir of .npy/.wav fixtures to run headless "
        "through the real recognizer + TTS and print per-turn latencies",
    )
    parser.add_argument("--llm", choices=["echo", "ollama"], default="ollama")
    parser.add_argument("--model", default=None, help="main Ollama model (research/vision)")
    parser.add_argument(
        "--fast-model",
        dest="fast_model",
        default=None,
        help="small Ollama model for quick spoken replies",
    )
    parser.add_argument(
        "--asr-final",
        dest="asr_final",
        choices=["streaming", "sense_voice", "whisper"],
        default=None,
        help="final-transcript backend override (A/B the second pass): "
        "'streaming' = streaming-only finals, NO offline re-transcribe (best for "
        "short/casual speech where the offline pass over-confidently hallucinates); "
        "'sense_voice'/'whisper' = offline second pass. Default: config asr_final_backend.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="device profile from config 'device_profiles' (e.g. desktop, phone)",
    )
    parser.add_argument(
        "--mode",
        choices=[m.value for m in Mode],
        default=Mode.ASSISTANT.value,
        help="starting mode",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="enable the Open Interpreter action brain for command-mode "
        "(voice -> machine control); reads the 'agent_brain' config block",
    )
    parser.add_argument(
        "--gui-actions",
        action="store_true",
        help="enable READ-ONLY computer-use (screen.identify: locate on-screen UI "
        "elements by voice; never clicks/types); reads the 'gui_actions' config block",
    )
    parser.add_argument(
        "--planner",
        action="store_true",
        help="enable the ReAct planner: assistant-mode reasoning/gathering "
        "queries escalate to a bounded plan->execute loop over the "
        "capabilities (reads the 'agent.planner' config block)",
    )
    parser.add_argument(
        "--stream-tts",
        dest="stream_tts",
        action="store_true",
        help="speak each sentence as it is generated (lower latency) instead "
        "of waiting for the full answer (reads the 'tts.streaming' config flag)",
    )
    # Three independent capture axes that share one run bundle (logs/runs/run-<id>.*):
    #   (always)   .txt full DEBUG log + .summary.json (timings, transcript, system stats)
    #   --debug    louder CONSOLE only (the file is always full DEBUG)
    #   --record   also save the session AUDIO (.wav), the heavy/opt-in artifact
    parser.add_argument(
        "--debug",
        action="store_true",
        help="louder console (mirror DEBUG to the terminal). The run bundle "
        "under logs/runs/ is written either way. Also SPEAKER_DEBUG=1.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="also save the session's 16 kHz mic audio beside the selected run "
        "bundle. Replays via `--engine replay` to become a test.",
    )
    parser.add_argument(
        "--record-playback-reference",
        action="store_true",
        help="also save a frame-aligned TTS reference beside the mic WAV for "
        "open-speaker/AEC replay; implies --record",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="print the available audio devices (with indices) and exit",
    )
    parser.add_argument(
        "--input-device",
        default=None,
        help="mic device index or name (see --list-devices); default = system default",
    )
    parser.add_argument(
        "--output-device",
        default=None,
        help="speaker device index or name; set this if the default is an HDMI "
        "monitor with no speakers",
    )
    parser.add_argument(
        "--input-gain",
        type=float,
        default=None,
        help="software gain on captured audio for a quiet mic (e.g. 4.0); "
        "prefer raising the OS mic level first",
    )
    parser.add_argument(
        "--enroll",
        action="store_true",
        help="record your voice and save a speaker-ID reference, then exit. "
        "Gates barge-in and (when enabled) input on your voice so the assistant "
        "stops answering ambient audio and its own TTS. Run once before --engine sherpa.",
    )
    parser.add_argument(
        "--enroll-seconds",
        dest="enroll_seconds",
        type=float,
        default=4.0,
        help="with --enroll: seconds of audio per clip (default: 4.0)",
    )
    parser.add_argument(
        "--enroll-passes",
        dest="enroll_passes",
        type=int,
        default=3,
        help="with --enroll: number of clips to average into the reference (default: 3)",
    )
    parser.add_argument(
        "--replace-enrollment",
        action="store_true",
        help="with --enroll: explicitly allow replacing a non-empty configured "
        "speaker reference (normally refused)",
    )
    parser.add_argument(
        "--require-prepared-enrollment",
        action="store_true",
        help="with --enroll: refuse unless tools.prepare_enrollment bound this "
        "checkout to a verified isolated candidate",
    )
    parser.add_argument(
        "--autotest-virtual-delay-contract",
        dest="autotest_virtual_delay_contract",
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.record_playback_reference:
        args.record = True

    virtual_audio_binder = None
    if args.autotest_virtual_delay_contract:
        if args.engine != "sherpa" or args.enroll:
            parser.error(
                "--autotest-virtual-delay-contract is private to non-enrollment "
                "--engine sherpa harness children"
            )
        from .virtual_audio import (
            PreparedVirtualAudioBinder,
            VirtualAudioContractError,
            load_virtual_audio_contract,
        )

        try:
            virtual_audio_binder = PreparedVirtualAudioBinder.prepare(
                load_virtual_audio_contract(args.autotest_virtual_delay_contract)
            )
        except VirtualAudioContractError as exc:
            parser.error(f"invalid virtual delay contract: {exc}")

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0

    runlog = setup_logging(args.debug or os.environ.get("SPEAKER_DEBUG") == "1")
    monitor = SystemMonitor(runlog.summary)
    monitor.start()  # baseline compute reading before anything heavy loads

    config = _load_config()
    # device-adapt-1: 'auto' (the shipped default) probes the host and picks the
    # matching profile, so an unconfigured low-spec box stops silently running the
    # heavy desktop tier. An explicit --device / config.device still wins.
    requested_device = args.device or config.get("device", "auto")
    device, rationale = resolve_device(config, requested_device)
    if rationale:
        msg = f"device profile auto-selected: {device} ({rationale})"
        runlog.logger.info(msg)
        print(f"[device] {msg}", file=sys.stderr)
    runlog.summary.note(
        engine=args.engine, llm=args.llm, device=device, mode=args.mode,
        model=args.model, fast_model=args.fast_model,
    )
    # cross-platform-8: strict -> a mistyped --device fails fast (lists valid
    # names) instead of silently no-opping to the heavy base config.
    try:
        config = _apply_device_profile(config, device, strict=True)
    except ValueError as exc:
        print(f"[device] {exc}", file=sys.stderr)
        return 2
    # CLI audio overrides win over config.json's sherpa block.
    sherpa_overrides = {
        "input_device": args.input_device,
        "output_device": args.output_device,
        "input_gain": args.input_gain,
        # --asr-final A/B: 'streaming' disables the offline second pass (empty
        # backend -> streaming-only finals). 'sense_voice'/'whisper' force it on.
        # None -> no override (use the config's asr_final_backend).
        "asr_final_backend": (
            "" if args.asr_final == "streaming"
            else args.asr_final
        ),
    }
    for key, val in sherpa_overrides.items():
        if val is not None:
            config.setdefault("sherpa", {})[key] = val
    if args.record_playback_reference:
        # Explicit evidence capture must not silently depend on an ignored
        # machine-local config key. The launcher uses this in-memory override;
        # normal runtime configuration remains unchanged on disk.
        config.setdefault("sherpa", {})["record_playback_reference"] = True

    if virtual_audio_binder is not None:
        contract = virtual_audio_binder.contract
        config = _apply_autotest_virtual_delay_profile(
            config,
            input_device=contract.capture_pcm,
            output_device=contract.playback_pcm,
        )
        runlog.logger.info(
            "[autotest-route] topology verified: %s",
            virtual_audio_binder.topology_detail,
        )

    # One-shot enrollment: record the user's voice, save the reference, exit.
    # Runs after the device profile + audio overrides so it uses the same mic
    # settings the live engine will, but before any model/LLM is built.
    if args.enroll:
        from .enroll import run_enrollment

        code = run_enrollment(
            config,
            passes=args.enroll_passes,
            seconds=args.enroll_seconds,
            replace_existing=args.replace_enrollment,
            require_prepared=args.require_prepared_enrollment,
        )
        try:
            monitor.stop()
        except Exception:  # noqa: BLE001 - telemetry must never mask the exit code
            pass
        runlog.finalize(None)
        return code

    # Native voice is the only normal CLI path that needs physical audio and
    # open-speaker routing. Fail before constructing models/streams so an active
    # word-cut recipe cannot silently run without its VAD or OS EC route. EchoLLM
    # deliberately skips Ollama readiness; console/replay/enroll never enter.
    if args.engine == "sherpa":
        llm_cfg = config.get("llm", {}) or {}
        requested_models = None
        if (
            args.llm != "echo"
            and str(llm_cfg.get("backend", "ollama")).lower() == "ollama"
        ):
            requested_models = tuple(dict.fromkeys(
                str(value) for value in (
                    args.model or llm_cfg.get("main_model"),
                    args.fast_model or llm_cfg.get("fast_model"),
                    llm_cfg.get("router_model"),
                )
                if value
            ))
        try:
            _require_sherpa_runtime_ready(
                config,
                args.llm,
                models_needed=requested_models,
                virtual_audio_binder=virtual_audio_binder,
            )
        except BaseException:
            # This check runs before the main runtime try/finally. Do not strand
            # the telemetry thread or an unfinalized run bundle on fail-fast.
            try:
                monitor.stop()
            except Exception:  # noqa: BLE001 - preserve the readiness error
                pass
            try:
                runlog.finalize(None)
            except Exception:  # noqa: BLE001 - preserve the readiness error
                pass
            raise

    llm, fast_llm = _build_llms(args, config)
    engine = _build_engine(
        args,
        config,
        virtual_audio_binder=virtual_audio_binder,
    )
    router = build_router(config)
    monitor.mark("after_build")  # compute reading once clients/engine are built

    record_path = None
    playback_reference_path = None
    if args.record and hasattr(engine, "set_record_path"):
        record_path = os.path.join(
            os.path.dirname(runlog.log_path), f"run-{runlog.run_id}.wav"
        )
        engine.set_record_path(record_path)
        runlog.summary.note(recording=record_path)
        if bool((config.get("sherpa", {}) or {}).get("record_playback_reference")):
            playback_reference_path = record_path[:-4] + ".ref.wav"
            runlog.summary.note(playback_reference=playback_reference_path)
    elif args.record:
        runlog.logger.warning("--record ignored: %s engine has no recorder", args.engine)

    runtime = build_runtime(
        config,
        engine=engine,
        llm=llm,
        fast_llm=fast_llm,
        router=router,
        start_mode=Mode(args.mode),
        agent_on=bool(args.agent),
        gui_actions_on=bool(getattr(args, "gui_actions", False)),
        force_planner=bool(args.planner),
        force_stream_tts=bool(args.stream_tts),
        load_fraction=monitor.load_fraction,
    )
    # Optional screen-capture feed (OFF by default; config.screen_capture.enabled).
    # When on, a background loop grabs the screen on a cadence and feeds the latest
    # frame to the model as ambient visual context (runtime.set_current_frame).
    # Optionally (screen_capture.memorize) a visual memorizer also turns each frame
    # into a caption+OCR 'vision' memory off the hot path (core/visual_memory.py).
    from .screen_capture import build_screen_feed
    from .visual_memory import build_visual_memorizer

    # Caption on the BARE LOCAL model only (§9.7: raw screen frames never leave the
    # device) -- never the cloud-wrapped main client. local_main is itself when
    # cloud is off; the getattr fallback keeps --llm echo working.
    memorizer = build_visual_memorizer(config, runtime, getattr(llm, "local_main", llm))
    screen_feed = build_screen_feed(
        config, runtime, observer=(memorizer.observe if memorizer is not None else None)
    )
    if memorizer is not None:
        memorizer.start()
    if screen_feed is not None:
        screen_feed.start()
    try:
        if args.engine == "replay":
            if not args.replay_dir:
                raise SystemExit("--engine replay needs --replay-dir <fixtures>")
            _run_replay(runtime, engine, args.replay_dir)
        elif args.engine in ("sherpa", "livekit"):
            _run_live(runtime)
        else:
            runtime.start(run_bus=False)
            _run_console(runtime, engine)
    finally:
        if screen_feed is not None:
            screen_feed.stop()
        if memorizer is not None:
            memorizer.stop()
        try:
            monitor.stop()  # folds baseline/peak/final/deltas into the summary
        except Exception:  # noqa: BLE001
            pass
        try:
            records = [r.as_dict() for r in runtime.metrics.records()]
        except Exception:  # noqa: BLE001 - never let logging hide the real error
            records = None
        runlog.finalize(records)
        print(f"[log] full log: {runlog.log_path}")
        print(f"[log] summary:  {runlog.summary_path}")
        if record_path is not None:
            print(f"[log] audio:    {record_path}")
        if playback_reference_path is not None:
            print(f"[log] playback: {playback_reference_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
