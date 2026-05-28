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
# Optional deep-merge overlay: a JSON file (mounted in via compose) whose
# top-level sections are shallow-merged into the assembled config before it
# lands at /tmp/speaker/config.json. Used by the cloud-streaming console
# docker-compose to flip ``cloud.enabled=true`` on the desktop_gpu_4090
# profile without editing the committed config.json.
OVERLAY_PATH = os.environ.get("SPEAKER_CONFIG_OVERLAY", "/app/config.overlay.json")


def _shallow_merge(base: dict, overlay: dict) -> dict:
    """Deep-merge an overlay onto a base config (dicts recurse, non-dicts
    replace).

    Named ``_shallow_merge`` for continuity with the existing module
    history; semantically it's a deep merge. The previous shallow shape
    was insufficient -- to flip ``device_profiles.desktop_gpu_4090.llm.cloud.enabled``
    from an overlay, the merger has to walk three levels (sections ->
    profiles -> sub-sections) without flattening the base's siblings.
    """
    out = dict(base)
    for key, value in overlay.items():
        if (
            isinstance(value, dict)
            and isinstance(out.get(key), dict)
        ):
            out[key] = _shallow_merge(out[key], value)
        else:
            out[key] = value
    return out


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

    # Optional config overlay (e.g. enable cloud on the 4090 profile for the
    # docker-compose console flow). Skipped silently if absent.
    if os.path.exists(OVERLAY_PATH):
        try:
            with open(OVERLAY_PATH, "r", encoding="utf-8") as fh:
                overlay = json.load(fh)
            cfg = _shallow_merge(cfg, overlay)
            print(f"[entrypoint] merged config overlay from {OVERLAY_PATH}")
        except Exception as exc:  # noqa: BLE001
            print(f"[entrypoint] could not merge overlay: {exc}", file=sys.stderr)

    ollama_host = os.environ.get("OLLAMA_HOST")
    if ollama_host:
        cfg.setdefault("llm", {})["host"] = ollama_host
        print(f"[entrypoint] llm.host -> {ollama_host}")

    keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE")
    if keep_alive:
        cfg.setdefault("llm", {})["keep_alive"] = keep_alive
        print(f"[entrypoint] llm.keep_alive -> {keep_alive}")

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
