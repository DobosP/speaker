from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Model acquisition for the real benchmark. Each artifact is a (repo_id,
# filename) pair fetched from HuggingFace via huggingface_hub and cached. The
# coordinates are overridable so CI can pin exact versions without code changes:
#
#   * a JSON manifest file passed via --models-manifest / $SPEAKER_BENCH_MANIFEST
#   * per-artifact env overrides: $SPEAKER_BENCH_<ARTIFACT>_REPO / _FILE
#
# Gemma is gated -- pass a token (read from $HUGGINGFACE_TOKEN by the CLI).
# Sherpa ASR/VAD/TTS models are public.

# Small, on-device-representative defaults (phone tier). These are best-effort
# public coordinates; if a pull 404s, override via the manifest rather than
# editing code.
_ASR_REPO = "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26"
_ASR_SUFFIX = "epoch-99-avg-1-chunk-16-left-128"  # the streaming chunk variant
DEFAULT_MANIFEST: dict[str, dict[str, str]] = {
    "asr_tokens": {"repo": _ASR_REPO, "file": "tokens.txt"},
    "asr_encoder": {"repo": _ASR_REPO, "file": f"encoder-{_ASR_SUFFIX}.int8.onnx"},
    "asr_decoder": {"repo": _ASR_REPO, "file": f"decoder-{_ASR_SUFFIX}.onnx"},
    "asr_joiner": {"repo": _ASR_REPO, "file": f"joiner-{_ASR_SUFFIX}.int8.onnx"},
    "vad_model": {
        "repo": "csukuangfj/vad",
        "file": "silero_vad.onnx",
    },
    "tts_model": {
        "repo": "csukuangfj/vits-piper-en_US-libritts_r-medium",
        "file": "en_US-libritts_r-medium.onnx",
    },
    "tts_tokens": {
        "repo": "csukuangfj/vits-piper-en_US-libritts_r-medium",
        "file": "tokens.txt",
    },
    "fast_gguf": {
        "repo": "unsloth/gemma-3-1b-it-GGUF",
        "file": "gemma-3-1b-it-Q4_K_M.gguf",
    },
    "main_gguf": {
        "repo": "unsloth/gemma-3-4b-it-GGUF",
        "file": "gemma-3-4b-it-Q4_K_M.gguf",
    },
}


@dataclass
class ModelPaths:
    asr_tokens: str
    asr_encoder: str
    asr_decoder: str
    asr_joiner: str
    vad_model: str
    tts_model: str
    tts_tokens: str
    main_gguf: str
    fast_gguf: str
    tts_data_dir: str = ""  # espeak-ng-data dir (piper VITS phonemization)

    def sherpa_overrides(self) -> dict[str, str]:
        return {
            "asr_tokens": self.asr_tokens,
            "asr_encoder": self.asr_encoder,
            "asr_decoder": self.asr_decoder,
            "asr_joiner": self.asr_joiner,
            "vad_model": self.vad_model,
            "tts_model": self.tts_model,
            "tts_tokens": self.tts_tokens,
            "tts_data_dir": self.tts_data_dir,
        }

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def load_manifest(path: Optional[str]) -> dict[str, dict[str, str]]:
    """Merge env override file -> explicit path -> defaults."""
    manifest = {k: dict(v) for k, v in DEFAULT_MANIFEST.items()}
    file_path = path or os.environ.get("SPEAKER_BENCH_MANIFEST")
    if file_path:
        with open(file_path, "r", encoding="utf-8") as fh:
            override = json.load(fh)
        for key, coords in override.items():
            manifest.setdefault(key, {}).update(coords)
    # Per-artifact env overrides win last.
    for key in manifest:
        repo = os.environ.get(f"SPEAKER_BENCH_{key.upper()}_REPO")
        fname = os.environ.get(f"SPEAKER_BENCH_{key.upper()}_FILE")
        if repo:
            manifest[key]["repo"] = repo
        if fname:
            manifest[key]["file"] = fname
    return manifest


def fetch_models(
    cache_dir: str,
    *,
    token: Optional[str] = None,
    manifest_path: Optional[str] = None,
    which: Optional[list[str]] = None,
) -> ModelPaths:
    """Download every artifact in the manifest and return local paths.

    ``which`` limits the artifacts fetched (e.g. just ASR for a quick check).
    Raises a clear error if a coordinate is wrong so it can be fixed in the
    manifest, not the code.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download  # lazy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "tools.bench needs huggingface_hub: pip install huggingface_hub"
        ) from exc

    manifest = load_manifest(manifest_path)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    resolved: dict[str, str] = {}
    for key, coords in manifest.items():
        if which is not None and key not in which:
            continue
        try:
            resolved[key] = hf_hub_download(
                repo_id=coords["repo"],
                filename=coords["file"],
                cache_dir=cache_dir,
                token=token,
            )
        except Exception as exc:  # noqa: BLE001 - surface actionable guidance
            raise SystemExit(
                f"Failed to fetch model artifact {key!r} "
                f"({coords.get('repo')}/{coords.get('file')}): {exc}\n"
                f"Override its coordinates via a --models-manifest JSON or the "
                f"SPEAKER_BENCH_{key.upper()}_REPO / _FILE env vars."
            ) from exc

    # Piper VITS needs the espeak-ng-data phoneme tables (a whole directory),
    # which ships in the TTS repo. Pull just that subtree if the model wants it.
    if (which is None or "tts_data_dir" in which) and "tts_model" in resolved:
        tts_repo = manifest["tts_model"]["repo"]
        try:
            snap = snapshot_download(
                repo_id=tts_repo,
                cache_dir=cache_dir,
                token=token,
                allow_patterns=["espeak-ng-data/*"],
            )
            data_dir = os.path.join(snap, "espeak-ng-data")
            resolved["tts_data_dir"] = data_dir if os.path.isdir(data_dir) else ""
        except Exception:  # noqa: BLE001 - data dir is optional for some models
            resolved["tts_data_dir"] = ""

    return ModelPaths(**resolved)
