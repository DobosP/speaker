#!/usr/bin/env python3
"""Container entrypoint: point config.json at the downloaded sherpa models, then
exec the real command (worker or token server).

The sherpa model paths are not known until they are fetched into the /models
volume, so the one-shot `download_models.py` step writes the resolved absolute
paths to /models/sherpa_paths.json and this script merges them into the image's
config.json at startup. If the file is absent (models not downloaded yet) the
config is left untouched -- the worker then runs without ASR/TTS rather than
crashing, which is what you want for `--llm echo` smoke tests.
"""
from __future__ import annotations

import json
import os
import sys

CONFIG = "/app/config.json"
PATHS = "/models/sherpa_paths.json"


def patch_config() -> None:
    if not os.path.exists(PATHS):
        print(f"[entrypoint] {PATHS} not found; leaving sherpa config empty", file=sys.stderr)
        return
    try:
        with open(CONFIG, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        with open(PATHS, "r", encoding="utf-8") as fh:
            paths = json.load(fh)
    except Exception as exc:  # noqa: BLE001 - surface and continue
        print(f"[entrypoint] could not merge model paths: {exc}", file=sys.stderr)
        return
    sherpa = cfg.setdefault("sherpa", {})
    for key, value in paths.items():
        if value:
            sherpa[key] = value
    with open(CONFIG, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    print("[entrypoint] merged sherpa model paths from /models/sherpa_paths.json")


def main() -> int:
    patch_config()
    if len(sys.argv) < 2:
        print("[entrypoint] no command to run", file=sys.stderr)
        return 1
    os.chdir("/app")
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
