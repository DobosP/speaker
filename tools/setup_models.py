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
from typing import Callable

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
# The ASR and TTS models each ship their OWN tokens.txt with different
# vocabularies, so they must land in separate folders -- a flat dir lets the
# second tokens.txt clobber the first, which loads the recognizer with the
# wrong vocab (garbage transcripts + get_result crashes).
SUBDIR = {
    "asr_tokens": "asr",
    "asr_encoder": "asr",
    "asr_decoder": "asr",
    "asr_joiner": "asr",
    "vad_model": "vad",
    "tts_model": "tts",
    "tts_tokens": "tts",
}


def dest_for(base: str, key: str) -> str:
    """Per-artifact download dir so same-named files (tokens.txt) don't collide."""
    return os.path.join(base, SUBDIR.get(key, ""))


def wire_sherpa_paths(
    config: dict, resolved: dict, *, abspath: Callable[[str], str] = os.path.abspath
) -> dict:
    """Merge resolved model paths into ``config['sherpa']`` in place.

    Empty paths are skipped (so a missing optional artifact like
    ``tts_data_dir`` leaves the existing value alone) and every other config
    section is preserved. Pure + injectable ``abspath`` so it is unit-testable
    without touching the filesystem."""
    sherpa = config.setdefault("sherpa", {})
    for key, path in resolved.items():
        if path:
            sherpa[key] = abspath(path)
    return config


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
        dest = dest_for(args.dest, key)
        os.makedirs(dest, exist_ok=True)
        print(f"[models] fetching {key}: {coords['repo']}/{coords['file']} -> {dest}")
        resolved[key] = hf_hub_download(
            repo_id=coords["repo"],
            filename=coords["file"],
            local_dir=dest,
            token=token,
            force_download=args.force,
        )

    # Piper VITS needs the espeak-ng-data phoneme tables (a directory subtree).
    tts_repo = manifest["tts_model"]["repo"]
    tts_dest = dest_for(args.dest, "tts_model")
    try:
        snapshot_download(
            repo_id=tts_repo,
            local_dir=tts_dest,
            token=token,
            allow_patterns=["espeak-ng-data/*"],
        )
        data_dir = os.path.join(tts_dest, "espeak-ng-data")
        resolved["tts_data_dir"] = data_dir if os.path.isdir(data_dir) else ""
    except Exception as exc:  # noqa: BLE001 - optional for some voices
        print(f"[models] espeak-ng-data not fetched ({exc}); continuing", file=sys.stderr)
        resolved["tts_data_dir"] = ""

    # Wire absolute paths into config.json, preserving everything else.
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    wire_sherpa_paths(cfg, resolved)
    with open(args.config, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    sherpa = cfg["sherpa"]
    print(f"\n[models] {args.config} sherpa paths set:")
    for key in FILE_KEYS + ["tts_data_dir"]:
        print(f"  {key}: {sherpa.get(key, '')}")
    print("\nNow run:  python -m core --engine sherpa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
