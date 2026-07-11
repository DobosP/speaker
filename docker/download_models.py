#!/usr/bin/env python3
"""One-shot: fetch the sherpa-onnx ASR/VAD/TTS models into the /models volume and
record their resolved paths to /models/sherpa_paths.json (read at startup by
entrypoint.py).

Reuses the project's model manifest (tools/bench/models.py) so coordinates stay
in one place and the same SPEAKER_BENCH_*_REPO / _FILE env overrides apply. The
LLM is served by Ollama, so the phone-only MiniCPM GGUF is deliberately skipped here.

Idempotent: skips if sherpa_paths.json already exists (set FORCE_DOWNLOAD=1 to
refetch).
"""
from __future__ import annotations

import json
import os
import sys

OUT = "/models"
PATHS = os.path.join(OUT, "sherpa_paths.json")
CACHE = os.path.join(OUT, "hf-cache")

# File artifacts (the espeak-ng-data directory is handled separately below).
FILE_KEYS = [
    "asr_tokens",
    "asr_encoder",
    "asr_decoder",
    "asr_joiner",
    "vad_model",
    "tts_model",
    "tts_tokens",
]


def main() -> int:
    if os.path.exists(PATHS) and os.environ.get("FORCE_DOWNLOAD") != "1":
        print(f"[models] {PATHS} present; skipping (set FORCE_DOWNLOAD=1 to refetch)")
        return 0

    sys.path.insert(0, "/app")
    from huggingface_hub import hf_hub_download, snapshot_download
    from tools.bench.models import load_manifest

    token = os.environ.get("HUGGINGFACE_TOKEN") or None
    manifest = load_manifest(None)
    os.makedirs(CACHE, exist_ok=True)

    resolved: dict[str, str] = {}
    for key in FILE_KEYS:
        coords = manifest[key]
        print(f"[models] fetching {key}: {coords['repo']}/{coords['file']}")
        resolved[key] = hf_hub_download(
            repo_id=coords["repo"], filename=coords["file"], cache_dir=CACHE, token=token
        )

    # Piper VITS needs the espeak-ng-data phoneme tables (a directory subtree).
    tts_repo = manifest["tts_model"]["repo"]
    try:
        snap = snapshot_download(
            repo_id=tts_repo, cache_dir=CACHE, token=token, allow_patterns=["espeak-ng-data/*"]
        )
        data_dir = os.path.join(snap, "espeak-ng-data")
        resolved["tts_data_dir"] = data_dir if os.path.isdir(data_dir) else ""
    except Exception as exc:  # noqa: BLE001 - data dir is optional for some voices
        print(f"[models] espeak-ng-data not fetched ({exc}); continuing", file=sys.stderr)
        resolved["tts_data_dir"] = ""

    with open(PATHS, "w", encoding="utf-8") as fh:
        json.dump(resolved, fh, indent=2)
    print(f"[models] wrote {PATHS}:")
    for key, value in resolved.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
