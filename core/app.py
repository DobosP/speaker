from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from always_on_agent.events import Mode
from always_on_agent.followups import FollowupConfig
from always_on_agent.memory import Memory, SessionMemory
from always_on_agent.react import PlannerConfig

from .capabilities import RecallConfig
from .engine import AudioEngine
from .llm import (
    EchoLLM,
    HedgeLLM,
    LlamaCppLLM,
    LLMClient,
    OllamaLLM,
    OpenAICompatLLM,
    SensitivityRouterLLM,
)
from .routing import build_chain_selector, build_router
from .runlog import setup_logging
from .runtime import VoiceRuntime
from .sysinfo import SystemMonitor


def _load_config(path: str = "config.json", *, local: str = "config.local.json") -> dict:
    """Load ``config.json`` (the committed template) and shallow-merge a
    machine-local ``config.local.json`` over it per top-level section. Keeping
    machine-specific values (e.g. the sherpa model paths written by
    ``tools.setup_models``) in the gitignored local file keeps the template
    portable and out of git."""
    config: dict = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    # Hermetic-test guard: when SPEAKER_NO_LOCAL_CONFIG is truthy, skip the
    # machine-local overlay entirely. Without this, a machine that has real
    # model paths in config.local.json makes `--engine sherpa` start the live
    # capture loop instead of failing fast, hanging the test suite. Production
    # and default behaviour are unchanged (the var is unset by default).
    _skip_local = os.environ.get("SPEAKER_NO_LOCAL_CONFIG", "").strip().lower() not in ("", "0", "false", "no")
    if os.path.exists(local) and not _skip_local:
        with open(local, "r", encoding="utf-8") as fh:
            overrides = json.load(fh)
        for section, value in overrides.items():
            base = config.get(section)
            if isinstance(value, dict) and isinstance(base, dict):
                config[section] = {**base, **value}
            else:
                config[section] = value
    return config


def _apply_device_profile(config: dict, device: str) -> dict:
    """Layer ``device_profiles[device]`` over the base config.

    A profile holds per-section overrides (``llm``, ``sherpa``); each is shallow-
    merged onto the base so a phone profile can swap the LLM backend/models and
    retune CPU threads without restating every field. Unknown device -> no-op.
    """
    profile = config.get("device_profiles", {}).get(device)
    if not profile:
        return config
    merged = dict(config)
    for section, overrides in profile.items():
        base = config.get(section)
        if isinstance(overrides, dict) and isinstance(base, dict):
            merged[section] = {**base, **overrides}
        else:
            merged[section] = overrides
    return merged


def _build_llms(args, config: dict) -> tuple[LLMClient, LLMClient | None]:
    """Return ``(main_llm, fast_llm)``.

    ``main_llm`` is the larger/multimodal model (research/vision); ``fast_llm``
    is a small model for snappy spoken replies. With ``--llm echo`` both are the
    same fake and ``fast_llm`` is ``None``. The backend is chosen by the ``llm``
    config block: ``ollama`` (desktop, GPU) or ``llamacpp`` (on-device GGUF).
    """
    if args.llm == "echo":
        return EchoLLM(), None
    llm_cfg = config.get("llm", {})
    options = llm_cfg.get("options")
    backend = llm_cfg.get("backend", "ollama")

    if backend == "llamacpp":
        common = dict(
            n_ctx=llm_cfg.get("n_ctx", 4096),
            n_threads=llm_cfg.get("n_threads"),
            n_gpu_layers=llm_cfg.get("n_gpu_layers", 0),
            chat_format=llm_cfg.get("chat_format"),
            options=options,
        )
        main_path = args.model or llm_cfg.get("main_model_path")
        fast_path = args.fast_model or llm_cfg.get("fast_model_path")
        if not main_path:
            raise SystemExit("llamacpp backend needs llm.main_model_path (a GGUF file).")
        main = LlamaCppLLM(main_path, **common)
        fast = LlamaCppLLM(fast_path, **common) if fast_path else None
        return _wrap_cloud(main, llm_cfg), fast

    host = llm_cfg.get("host")
    keep_alive = llm_cfg.get("keep_alive")
    main_model = args.model or llm_cfg.get("main_model") or config.get("llm_model", "gemma3:12b")
    fast_model = args.fast_model or llm_cfg.get("fast_model")
    main = OllamaLLM(model=main_model, host=host, options=options, keep_alive=keep_alive)
    fast = (
        OllamaLLM(model=fast_model, host=host, options=options, keep_alive=keep_alive)
        if fast_model
        else None
    )
    return _wrap_cloud(main, llm_cfg), fast


def _build_cloud_client(preset_name: str, preset: dict) -> Optional[OpenAICompatLLM]:
    """Build one :class:`OpenAICompatLLM` from a ``cloud_providers`` entry.

    Returns ``None`` when the entry's API-key env var isn't set -- so missing
    credentials disable that provider individually (the rest of the chain
    keeps working) rather than crashing the whole runtime.

    The preset's ``profile`` key (e.g. ``"cerebras"``, ``"deepseek_reasoning"``,
    ``"moonshot"``) selects a :class:`core.llm.ProviderProfile` so per-vendor
    quirks (forbidden params, extra_body routing, reasoning-field streaming,
    max_tokens caps) apply without per-call adapter logic. Unknown profile
    names fall back to the safe generic shape."""
    if not isinstance(preset, dict):
        return None
    model = preset.get("model")
    if not model:
        return None
    api_key_env = preset.get("api_key_env")
    if api_key_env and not os.environ.get(api_key_env):
        return None
    return OpenAICompatLLM(
        model=model,
        base_url=preset.get("base_url"),
        api_key_env=api_key_env,
        options=preset.get("options"),
        profile=preset.get("profile"),
    )


def _wrap_cloud(local_main: LLMClient, llm_cfg: dict) -> LLMClient:
    """Optionally route the main tier through cloud LLM(s) for lower latency.

    Off unless ``llm.cloud.enabled`` (or a populated ``cloud_providers`` +
    ``cloud_chains``) is configured, so the fully-local default is preserved.
    Only the main/reasoning tier is wrapped -- the fast tier is already
    snappy and stays on-device.

    Two configuration shapes:

    - **Multi-provider sensitivity-routed (preferred).** When
      ``llm.cloud_providers`` and ``llm.cloud_chains`` are present, build
      one :class:`HedgeLLM` per chain (each racing local + the chain's
      ordered list of clouds, with failover on error/timeout) and wrap
      them in a :class:`SensitivityRouterLLM` that picks per turn based on
      ``llm.cloud_routing.sensitivity_to_chain``. Missing API keys cause
      that provider to silently drop out of its chains.

    - **Single-cloud back-compat.** When only ``llm.cloud`` is set (no
      providers/chains), use the existing single-HedgeLLM path.
    """
    cloud_cfg = llm_cfg.get("cloud") or {}
    providers = llm_cfg.get("cloud_providers") or {}
    chains_cfg = llm_cfg.get("cloud_chains") or {}
    strategy = cloud_cfg.get("strategy", "hedge")

    if strategy == "local_only" or not cloud_cfg.get("enabled", False):
        return local_main

    hedge_kwargs = dict(
        strategy=strategy,
        hedge_delay_ms=int(cloud_cfg.get("hedge_delay_ms", 150)),
        ttft_deadline_ms=int(cloud_cfg.get("ttft_deadline_ms", 1200)),
    )

    # Multi-provider path.
    if providers and chains_cfg:
        # Resolve each provider lazily; drop ones with missing API keys.
        resolved: dict[str, OpenAICompatLLM] = {}
        for name, preset in providers.items():
            if isinstance(preset, dict) and name.startswith("_"):
                continue  # skip _comment / metadata keys
            client = _build_cloud_client(name, preset)
            if client is not None:
                resolved[name] = client

        hedged_chains: dict[str, LLMClient] = {}
        any_clouds = False
        for chain_name, preset_names in chains_cfg.items():
            if chain_name.startswith("_"):
                continue
            if not isinstance(preset_names, (list, tuple)):
                continue
            chain_clouds = [resolved[n] for n in preset_names if n in resolved]
            if chain_clouds:
                any_clouds = True
            hedged_chains[chain_name] = HedgeLLM(
                local=local_main, cloud=chain_clouds, **hedge_kwargs
            )

        if not hedged_chains or not any_clouds:
            # Every chain ended up empty (e.g. no API keys set): fall through
            # to the local main tier without surprise wrappers. Otherwise
            # SensitivityRouterLLM would wrap an all-local hedge that adds
            # threading overhead for no benefit.
            return local_main

        routing_cfg = llm_cfg.get("cloud_routing") or {}
        default_chain = str(routing_cfg.get("default_chain", "private") or "private")
        if default_chain not in hedged_chains:
            default_chain = next(iter(hedged_chains))
        selector = build_chain_selector({"llm": llm_cfg})
        return SensitivityRouterLLM(
            hedged_chains, selector=selector, default_chain=default_chain
        )

    # Single-cloud back-compat path.
    model = cloud_cfg.get("model")
    if not model:
        return local_main
    cloud = OpenAICompatLLM(
        model=model,
        base_url=cloud_cfg.get("base_url"),
        api_key_env=cloud_cfg.get("api_key_env"),
        options=cloud_cfg.get("options"),
    )
    return HedgeLLM(local=local_main, cloud=cloud, **hedge_kwargs)


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


def _build_memory(config: dict) -> Memory:
    """Pick the memory backend (Locked Decision 4).

    Postgres-backed :class:`MemoryManagerAdapter` when ``memory.backend`` is
    ``postgres`` -- or when it is ``auto``/unset and ``$DATABASE_URL`` is set;
    in-RAM :class:`SessionMemory` otherwise. The adapter build is wrapped so a
    missing dependency (``ImportError``) or a connect failure falls back to the
    in-RAM store rather than crashing. Any connect error is logged **redacted**
    (R12) -- the message may carry the DSN.
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

    try:
        from always_on_agent.memory import MemoryManagerAdapter

        return MemoryManagerAdapter(
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

    agent_config = None
    if args.agent:
        from .agent import AgentBrainConfig

        agent_config = AgentBrainConfig.from_dict(config.get("agent_brain"))

    planner_config = PlannerConfig.from_dict(
        (config.get("agent", {}) or {}).get("planner")
    )
    if args.planner:
        planner_config.enabled = True

    stream_tts = bool((config.get("tts", {}) or {}).get("streaming", False)) or args.stream_tts
    followup_config = FollowupConfig.from_dict(config.get("followups"))

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

    memory = _build_memory(config)
    recall_config = RecallConfig.from_dict(config.get("memory"))

    runtime = VoiceRuntime(
        engine,
        llm,
        fast_llm=fast_llm,
        memory=memory,
        recall_config=recall_config,
        start_mode=Mode(args.mode),
        agent_config=agent_config,
        router=router,
        planner_config=planner_config,
        stream_tts=stream_tts,
        followup_config=followup_config,
        command_map=config.get("commands"),
        intents=intents,
        addressing=addressing,
        unsure_acts=unsure_acts,
        cleaner=cleaner,
    )

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
