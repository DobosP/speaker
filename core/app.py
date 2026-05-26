from __future__ import annotations

import argparse
import json
import os
import sys

from always_on_agent.events import Mode

from .engine import AudioEngine
from .llm import EchoLLM, LLMClient, OllamaLLM
from .runtime import VoiceRuntime


def _load_config(path: str = "config.json") -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _build_llm(args, config: dict) -> LLMClient:
    if args.llm == "echo":
        return EchoLLM()
    model = args.model or config.get("llm_model", "gemma3:latest")
    return OllamaLLM(model=model)


def _build_engine(args, config: dict) -> AudioEngine:
    if args.engine == "sherpa":
        from .engines.sherpa import SherpaConfig, SherpaOnnxEngine

        sherpa_cfg = SherpaConfig.from_dict(config.get("sherpa", {}))
        return SherpaOnnxEngine(sherpa_cfg)
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
    parser.add_argument("--engine", choices=["console", "sherpa"], default="console")
    parser.add_argument("--llm", choices=["echo", "ollama"], default="ollama")
    parser.add_argument("--model", default=None, help="Ollama model name")
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
    llm = _build_llm(args, config)
    engine = _build_engine(args, config)

    agent_config = None
    if args.agent:
        from .agent import AgentBrainConfig

        agent_config = AgentBrainConfig.from_dict(config.get("agent_brain"))

    runtime = VoiceRuntime(engine, llm, start_mode=Mode(args.mode), agent_config=agent_config)

    if args.engine == "sherpa":
        _run_live(runtime)
    else:
        runtime.start(run_bus=False)
        _run_console(runtime, engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
