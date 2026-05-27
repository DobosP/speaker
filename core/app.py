from __future__ import annotations

import argparse
import json
import os
import sys

from always_on_agent.events import Mode
from always_on_agent.followups import FollowupConfig
from always_on_agent.react import PlannerConfig

from .engine import AudioEngine
from .llm import EchoLLM, HedgeLLM, LlamaCppLLM, LLMClient, OllamaLLM, OpenAICompatLLM
from .routing import build_router
from .runlog import setup_logging
from .runtime import VoiceRuntime
from .sysinfo import SystemMonitor


def _load_config(path: str = "config.json") -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


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


def _wrap_cloud(local_main: LLMClient, llm_cfg: dict) -> LLMClient:
    """Optionally hedge the main tier against a cloud provider for lower latency.

    Off unless ``llm.cloud.enabled`` is true, so the fully-local default is
    preserved. Only the main/reasoning tier is wrapped -- the fast tier is
    already snappy and stays on-device.
    """
    cloud_cfg = llm_cfg.get("cloud") or {}
    strategy = cloud_cfg.get("strategy", "hedge")
    model = cloud_cfg.get("model")
    if not cloud_cfg.get("enabled") or strategy == "local_only" or not model:
        return local_main
    cloud = OpenAICompatLLM(
        model=model,
        base_url=cloud_cfg.get("base_url"),
        api_key_env=cloud_cfg.get("api_key_env"),
        options=cloud_cfg.get("options"),
    )
    return HedgeLLM(
        local=local_main,
        cloud=cloud,
        strategy=strategy,
        hedge_delay_ms=int(cloud_cfg.get("hedge_delay_ms", 150)),
        ttft_deadline_ms=int(cloud_cfg.get("ttft_deadline_ms", 1200)),
    )


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

    runtime = VoiceRuntime(
        engine,
        llm,
        fast_llm=fast_llm,
        start_mode=Mode(args.mode),
        agent_config=agent_config,
        router=router,
        planner_config=planner_config,
        stream_tts=stream_tts,
        followup_config=followup_config,
        command_map=config.get("commands"),
        intents=intents,
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
