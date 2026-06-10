from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Callable, Optional

from always_on_agent.continuation import ContinuationConfig
from always_on_agent.events import Mode
from always_on_agent.followups import FollowupConfig
from always_on_agent.memory import Memory, SessionMemory
from always_on_agent.react import PlannerConfig

from .capabilities import RecallConfig
from .config import _apply_device_profile, _load_config, apply_device_profile, load_config
from .engine import AudioEngine
from .llm import LLMClient
from .llm_factory import (
    _build_cloud_client,
    _build_llms,
    _preset_host,
    _wrap_cloud,
    build_llms,
)
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


def _require_asr_models(sherpa_cfg, engine_name: str) -> None:
    """Fail fast with an actionable message when the sherpa ASR models aren't
    configured, instead of starting an engine that can never hear anything."""
    if not getattr(sherpa_cfg, "asr_encoder", "") or not getattr(sherpa_cfg, "asr_tokens", ""):
        raise SystemExit(
            f"\n--engine {engine_name} needs the on-device speech models, but "
            "config.json has no sherpa model paths set.\n\n"
            "  Run once:   python -m tools.setup_models\n"
            "  (or full:   ./install.sh)\n\n"
            "That downloads the ASR/VAD/TTS models and writes their paths into "
            f"config.json. Then re-run:\n  python -m core --engine {engine_name}\n"
        )


def _build_engine(args, config: dict) -> AudioEngine:
    if args.engine == "sherpa":
        from .engines.sherpa import SherpaConfig, SherpaOnnxEngine

        sherpa_cfg = SherpaConfig.from_dict(config.get("sherpa", {}))
        _require_asr_models(sherpa_cfg, "sherpa")
        return SherpaOnnxEngine(sherpa_cfg)
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
    backend = str(mem_cfg.get("backend", "auto") or "auto").lower()
    db_url = os.environ.get("DATABASE_URL")
    want_postgres = backend == "postgres" or (backend in ("auto", "") and bool(db_url))
    if not want_postgres:
        return SessionMemory(max_items=working_window)

    # Rolling-summary closure over the fast tier. Built here where fast_llm is
    # available; the manager schedules the call on its writer thread (R2), so
    # this wrapper itself never runs on the bus thread.
    summarizer: Optional[Callable[[str], str]] = None
    if fast_llm is not None:
        def summarizer(text: str) -> str:
            return fast_llm.generate(text)

    try:
        from always_on_agent.memory import MemoryManagerAdapter

        return MemoryManagerAdapter(
            summarizer=summarizer,
            profile_enabled=bool(mem_cfg.get("profile_enabled", False)),
            episodic_ttl_days=int(mem_cfg.get("episodic_ttl_days", 90) or 90),
            summary_ttl_days=int(mem_cfg.get("summary_ttl_days", 365) or 365),
            enable_embeddings=bool(mem_cfg.get("embeddings", False)),
            max_recent_messages=int(mem_cfg.get("max_recent", 20) or 20),
        )
    except Exception as exc:  # noqa: BLE001 - degrade to in-RAM, never crash
        redacted = _redact_db_url(db_url) if db_url else "(no DATABASE_URL)"
        print(
            f"[memory] Postgres backend unavailable ({type(exc).__name__}); "
            f"falling back to in-RAM. db={redacted}"
        )
        return SessionMemory(max_items=working_window)


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
            time.sleep(0.5)
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

    # Unified capability router (the "middle layer", core/capability_router.py).
    # Disabled by default; when enabled it backs the runtime's tier + escalate
    # decisions with one coherent module. Reuses the tier router built above and
    # the fast tier for optional LLM disambiguation of ambiguous turns, and the
    # configured command phrases so it recognizes the same control words.
    capability_router = build_capability_router(
        config,
        tier_router=router,
        fast_llm=fast_llm,
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
        start_mode=start_mode,
        agent_config=agent_config,
        router=router,
        capability_router=capability_router,
        planner_config=planner_config,
        stream_tts=stream_tts,
        followup_config=followup_config,
        continuation_config=continuation_config,
        turn_merge_config=turn_merge_config,
        command_map=config.get("commands"),
        intents=intents,
        addressing=addressing,
        unsure_acts=unsure_acts,
        cleaner=cleaner,
        live_routing=live_routing,
        load_snapshot=load_fraction if live_routing else None,
        # Pre-warm the LLM tiers at startup so turn 1 doesn't pay the ~3s model
        # cold-load on the user's first utterance (lat-2). On by default; set
        # config.warm_on_start=false to skip (e.g. to measure cold start).
        warm_on_start=bool(config.get("warm_on_start", True)),
        persona=persona,
        # Per-mode wall-clock task deadlines (never-stuck backstop). Optional --
        # the supervisor bakes in sensible defaults; config overrides per mode.
        task_timeouts=config.get("task_timeouts"),
    )

    return runtime


def main(argv: list[str] | None = None) -> int:
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
        help="also save the session's 16 kHz mic audio to the run bundle "
        "(logs/runs/run-<id>.wav). Replays via `--engine replay` to become a test.",
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
    args = parser.parse_args(argv)

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0

    runlog = setup_logging(args.debug or os.environ.get("SPEAKER_DEBUG") == "1")
    monitor = SystemMonitor(runlog.summary)
    monitor.start()  # baseline compute reading before anything heavy loads

    config = _load_config()
    device = args.device or config.get("device", "desktop")
    runlog.summary.note(
        engine=args.engine, llm=args.llm, device=device, mode=args.mode,
        model=args.model, fast_model=args.fast_model,
    )
    config = _apply_device_profile(config, device)
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

    # One-shot enrollment: record the user's voice, save the reference, exit.
    # Runs after the device profile + audio overrides so it uses the same mic
    # settings the live engine will, but before any model/LLM is built.
    if args.enroll:
        from .enroll import run_enrollment

        code = run_enrollment(
            config, passes=args.enroll_passes, seconds=args.enroll_seconds
        )
        try:
            monitor.stop()
        except Exception:  # noqa: BLE001 - telemetry must never mask the exit code
            pass
        runlog.finalize(None)
        return code

    llm, fast_llm = _build_llms(args, config)
    engine = _build_engine(args, config)
    router = build_router(config)
    monitor.mark("after_build")  # compute reading once clients/engine are built

    if args.record and hasattr(engine, "set_record_path"):
        record_path = os.path.join(
            os.path.dirname(runlog.log_path), f"run-{runlog.run_id}.wav"
        )
        engine.set_record_path(record_path)
        runlog.summary.note(recording=record_path)
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
        force_planner=bool(args.planner),
        force_stream_tts=bool(args.stream_tts),
        load_fraction=monitor.load_fraction,
    )
    # Optional screen-capture feed (OFF by default; config.screen_capture.enabled).
    # When on, a background loop grabs the screen on a cadence and feeds the latest
    # frame to the model as ambient visual context (runtime.set_current_frame).
    from .screen_capture import build_screen_feed

    screen_feed = build_screen_feed(config, runtime)
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
        if args.record:
            print(f"[log] audio:    {os.path.join('logs/runs', f'run-{runlog.run_id}.wav')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
