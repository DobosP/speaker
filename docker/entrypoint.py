#!/usr/bin/env python3
"""Container entrypoint: assemble a runtime config.json, then exec the real
command (worker or token server).

The image runs unprivileged with a read-only root filesystem, so /app is not
writable. We therefore read the baked-in /app/config.json, merge in:

  * the sherpa model paths fetched into the read-only /models volume (recorded
    by download_models.py at /models/sherpa_paths.json), and
  * OLLAMA_HOST -> llm.host, so the LLM points at your host-local Ollama,

and write the result to a writable tmpfs dir, which becomes the working
directory (core.app and token_server both load ./config.json from the cwd).
If the model-paths file is absent the worker simply runs without ASR/TTS, which
is what you want for `--llm echo` smoke tests.
"""
from __future__ import annotations

import json
import os
import sys

SRC_CONFIG = "/app/config.json"
RUN_DIR = os.environ.get("SPEAKER_RUNTIME_DIR", "/tmp/speaker")
RUN_CONFIG = os.path.join(RUN_DIR, "config.json")
MODEL_PATHS = "/models/sherpa_paths.json"


def build_config() -> str:
    with open(SRC_CONFIG, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    if os.path.exists(MODEL_PATHS):
        try:
            with open(MODEL_PATHS, "r", encoding="utf-8") as fh:
                paths = json.load(fh)
            sherpa = cfg.setdefault("sherpa", {})
            for key, value in paths.items():
                if value:
                    sherpa[key] = value
            print("[entrypoint] merged sherpa model paths from /models")
        except Exception as exc:  # noqa: BLE001 - surface and continue
            print(f"[entrypoint] could not merge model paths: {exc}", file=sys.stderr)
    else:
        print(f"[entrypoint] {MODEL_PATHS} not found; sherpa STT/TTS disabled", file=sys.stderr)

    ollama_host = os.environ.get("OLLAMA_HOST")
    if ollama_host:
        cfg.setdefault("llm", {})["host"] = ollama_host
        print(f"[entrypoint] llm.host -> {ollama_host}")

    os.makedirs(RUN_DIR, exist_ok=True)
    with open(RUN_CONFIG, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    return RUN_DIR


def main() -> int:
    run_dir = build_config()
    if len(sys.argv) < 2:
        print("[entrypoint] no command to run", file=sys.stderr)
        return 1
    os.chdir(run_dir)  # core.app / token_server read ./config.json from here
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
