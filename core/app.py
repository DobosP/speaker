from __future__ import annotations

import argparse
import json
import os
import sys

from always_on_agent.events import Mode

from .engine import AudioEngine
from .llm import EchoLLM, LlamaCppLLM, LLMClient, OllamaLLM
from .routing import build_router
from .runtime import VoiceRuntime


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
        return main, fast

    host = llm_cfg.get("host")
    main_model = args.model or llm_cfg.get("main_model") or config.get("llm_model", "gemma3:12b")
    fast_model = args.fast_model or llm_cfg.get("fast_model")
    main = OllamaLLM(model=main_model, host=host, options=options)
    fast = OllamaLLM(model=fast_model, host=host, options=options) if fast_model else None
    return main, fast


def _build_engine(args, config: dict) -> AudioEngine:
    if args.engine == "sherpa":
        from .engines.sherpa import SherpaConfig, SherpaOnnxEngine

        sherpa_cfg = SherpaConfig.from_dict(config.get("sherpa", {}))
        return SherpaOnnxEngine(sherpa_cfg)
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


def _run_live(runtime: VoiceRuntime) -> None:
    import time

    runtime.start(run_bus=True)
    print(f"[live] engine running, mode={runtime.mode.value}. Ctrl-C to quit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lean local voice assistant runtime")
    parser.add_argument("--engine", choices=["console", "sherpa", "livekit"], default="console")
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
    args = parser.parse_args(argv)

    config = _load_config()
    device = args.device or config.get("device", "desktop")
    config = _apply_device_profile(config, device)
    llm, fast_llm = _build_llms(args, config)
    engine = _build_engine(args, config)
    router = build_router(config)

    agent_config = None
    if args.agent:
        from .agent import AgentBrainConfig

        agent_config = AgentBrainConfig.from_dict(config.get("agent_brain"))

    runtime = VoiceRuntime(
        engine,
        llm,
        fast_llm=fast_llm,
        start_mode=Mode(args.mode),
        agent_config=agent_config,
        router=router,
    )

    if args.engine in ("sherpa", "livekit"):
        _run_live(runtime)
    else:
        runtime.start(run_bus=False)
        _run_console(runtime, engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
