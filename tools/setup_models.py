#!/usr/bin/env python3
"""Download the sherpa-onnx ASR/VAD/TTS models for the native (`--engine sherpa`)
path and wire their paths into config.json.

Run once, then start the assistant:

    python -m tools.setup_models
    python -m core --engine sherpa

Reuses the shared model manifest (tools/bench/models.py) so the coordinates live
in one place (override with the SPEAKER_BENCH_*_REPO / _FILE env vars). Models
land in pretrained_models/sherpa/ with flat, predictable filenames. The LLM is
served by Ollama, so no GGUF is fetched here. Idempotent: re-running only fills
missing files unless you pass --force.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

DEST = os.path.join("pretrained_models", "sherpa")
CONFIG = "config.json"
FILE_KEYS = [
    "asr_tokens",
    "asr_encoder",
    "asr_decoder",
    "asr_joiner",
    "vad_model",
    "tts_model",
    "tts_tokens",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch sherpa models + wire config.json")
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    parser.add_argument("--dest", default=DEST, help=f"download dir (default: {DEST})")
    parser.add_argument(
        "--config", default=CONFIG, help=f"config file to update (default: {CONFIG})"
    )
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        raise SystemExit("Need huggingface_hub: python -m pip install huggingface-hub")
    from tools.bench.models import load_manifest

    token = os.environ.get("HUGGINGFACE_TOKEN") or None
    manifest = load_manifest(None)
    os.makedirs(args.dest, exist_ok=True)

    resolved: dict[str, str] = {}
    for key in FILE_KEYS:
        coords = manifest[key]
        print(f"[models] fetching {key}: {coords['repo']}/{coords['file']}")
        resolved[key] = hf_hub_download(
            repo_id=coords["repo"],
            filename=coords["file"],
            local_dir=args.dest,
            token=token,
            force_download=args.force,
        )

    # Piper VITS needs the espeak-ng-data phoneme tables (a directory subtree).
    tts_repo = manifest["tts_model"]["repo"]
    try:
        snapshot_download(
            repo_id=tts_repo,
            local_dir=args.dest,
            token=token,
            allow_patterns=["espeak-ng-data/*"],
        )
        data_dir = os.path.join(args.dest, "espeak-ng-data")
        resolved["tts_data_dir"] = data_dir if os.path.isdir(data_dir) else ""
    except Exception as exc:  # noqa: BLE001 - optional for some voices
        print(f"[models] espeak-ng-data not fetched ({exc}); continuing", file=sys.stderr)
        resolved["tts_data_dir"] = ""

    # Wire absolute paths into config.json, preserving everything else.
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    sherpa = cfg.setdefault("sherpa", {})
    for key, path in resolved.items():
        if path:
            sherpa[key] = os.path.abspath(path)
    with open(args.config, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    print(f"\n[models] {args.config} sherpa paths set:")
    for key in FILE_KEYS + ["tts_data_dir"]:
        print(f"  {key}: {sherpa.get(key, '')}")
    print("\nNow run:  python -m core --engine sherpa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
