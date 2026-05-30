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
# Machine-local overrides (gitignored), merged over config.json at load time.
# Model paths are machine-specific, so they belong here, not in the committed
# config.json template.
CONFIG = "config.local.json"
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
# Speaker-embedding model for the speaker-ID gate (barge-in + input gating).
# It ships as a GitHub *release asset* (not a HF repo file), so it's fetched by
# direct URL rather than huggingface_hub. Override with --speaker-model-url or
# the SPEAKER_EMBEDDING_MODEL_URL env var. NOTE: the upstream release tag spells
# "recongition" -- that typo is real, keep it. CAM++ is small (~28 MB), CPU-only,
# ONNX (no torch), 16 kHz -- a good fit for the on-device gate.
SPEAKER_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
)
# Optional punctuation model for ASR finals (CT-Transformer). Ships as a
# .tar.bz2 release asset containing a model.onnx; we extract that one file.
# Override with --punct-model-url or the ASR_PUNCT_MODEL_URL env var.
PUNCT_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2"
)
# Optional speech denoiser for the capture front-end (sherpa-onnx GTCRN). Ships
# as a single ~523 KB .onnx GitHub *release asset* (not a HF repo file), so it's
# fetched by direct URL like the speaker model. Tiny, CPU-only, 16 kHz. Override
# with --denoise-model-url or the GTCRN_MODEL_URL env var.
GTCRN_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speech-enhancement-models/gtcrn_simple.onnx"
)


def dest_for(base: str, key: str) -> str:
    """Per-artifact download dir so same-named files (tokens.txt) don't collide."""
    return os.path.join(base, SUBDIR.get(key, ""))


def fetch_speaker_model(dest_dir: str, url: str, *, force: bool = False) -> str:
    """Download the speaker-embedding ONNX from a direct URL into ``dest_dir``.

    Returns the local path. Skips the download when the file already exists
    (unless ``force``). Streams to a ``.part`` file and renames on success so an
    interrupted fetch never leaves a truncated model in place. Uses urllib (no
    HF dependency) because the model is a GitHub release asset, not a repo file.
    """
    import shutil
    import urllib.request

    os.makedirs(dest_dir, exist_ok=True)
    filename = url.rsplit("/", 1)[-1] or "speaker.onnx"
    path = os.path.join(dest_dir, filename)
    if os.path.exists(path) and not force:
        return path
    tmp = path + ".part"
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:  # noqa: S310 - trusted release URL
        shutil.copyfileobj(resp, fh)
    os.replace(tmp, path)
    return path


def extract_member(archive: str, suffix: str, dest_dir: str) -> str:
    """Extract the first member of a ``.tar.bz2`` whose name ends with ``suffix``.

    Returns the extracted file's path (flattened into ``dest_dir``). Used for the
    punctuation release asset, which bundles ``model.onnx`` inside a directory.
    Guards against path-traversal members (a tar entry escaping ``dest_dir``)."""
    import tarfile

    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        member = next(
            (m for m in tar.getmembers() if m.isfile() and m.name.endswith(suffix)), None
        )
        if member is None:
            raise FileNotFoundError(f"no '*{suffix}' member in {archive}")
        out_path = os.path.join(dest_dir, os.path.basename(member.name))
        # Flatten: read the member and write it directly, so a malicious or
        # nested path can't place the file outside dest_dir.
        src = tar.extractfile(member)
        if src is None:
            raise FileNotFoundError(f"could not read {member.name} from {archive}")
        with src, open(out_path, "wb") as out:
            import shutil

            shutil.copyfileobj(src, out)
    return out_path


def fetch_punct_model(dest_dir: str, url: str, *, force: bool = False) -> str:
    """Download + unpack the punctuation model, returning the model.onnx path.

    Downloads the .tar.bz2 release asset and extracts its ``model.onnx``. Skips
    work when the extracted model already exists (unless ``force``)."""
    os.makedirs(dest_dir, exist_ok=True)
    model_path = os.path.join(dest_dir, "model.onnx")
    if os.path.exists(model_path) and not force:
        return model_path
    archive_name = url.rsplit("/", 1)[-1] or "punct.tar.bz2"
    archive = fetch_speaker_model(dest_dir, url, force=force)  # reuses the streamed download
    extracted = extract_member(archive, "model.onnx", dest_dir)
    if extracted != model_path:
        os.replace(extracted, model_path)
    try:
        os.remove(archive)  # the .tar.bz2 is no longer needed once unpacked
    except OSError:
        pass
    _ = archive_name  # (kept for clarity; download already named it)
    return model_path


def apply_accuracy(manifest: dict, accuracy: str) -> dict:
    """For ``high`` accuracy, use the non-quantized (fp32) ASR encoder/joiner
    from the same repo -- more accurate than the int8 default, cheap on a strong
    CPU/GPU. ``fast`` keeps the small int8 weights (phone-tier). Mutates + returns
    the manifest."""
    if accuracy == "high":
        for key in ("asr_encoder", "asr_joiner"):
            manifest[key]["file"] = manifest[key]["file"].replace(".int8.onnx", ".onnx")
    return manifest


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
    parser.add_argument(
        "--accuracy",
        choices=["high", "fast"],
        default="high",
        help="ASR weights: 'high' = fp32 (more accurate, default for a capable PC); "
        "'fast' = int8 (smaller, phone-tier)",
    )
    parser.add_argument("--dest", default=DEST, help=f"download dir (default: {DEST})")
    parser.add_argument(
        "--config", default=CONFIG, help=f"config file to update (default: {CONFIG})"
    )
    parser.add_argument(
        "--speaker-model-url",
        dest="speaker_model_url",
        default=os.environ.get("SPEAKER_EMBEDDING_MODEL_URL", SPEAKER_MODEL_URL),
        help="URL of the speaker-ID embedding model (default: the sherpa-onnx "
        "CAM++ release asset; override or set SPEAKER_EMBEDDING_MODEL_URL)",
    )
    parser.add_argument(
        "--no-speaker-model",
        dest="speaker_model",
        action="store_false",
        help="skip the optional speaker-ID model download",
    )
    parser.add_argument(
        "--punct-model-url",
        dest="punct_model_url",
        default=os.environ.get("ASR_PUNCT_MODEL_URL", PUNCT_MODEL_URL),
        help="URL of the optional ASR punctuation model .tar.bz2 (override or "
        "set ASR_PUNCT_MODEL_URL)",
    )
    parser.add_argument(
        "--punct-model",
        dest="punct_model",
        action="store_true",
        help="also download the optional punctuation model (adds real .,? to "
        "ASR finals; off by default since casing restoration already helps)",
    )
    parser.add_argument(
        "--denoise-model-url",
        dest="denoise_model_url",
        default=os.environ.get("GTCRN_MODEL_URL", GTCRN_MODEL_URL),
        help="URL of the optional GTCRN speech-denoise model .onnx (override or "
        "set GTCRN_MODEL_URL)",
    )
    parser.add_argument(
        "--denoise-model",
        dest="denoise_model",
        action="store_true",
        help="also download the optional speech-denoise model (GTCRN, ~523 KB) "
        "and wire denoise_model in config; off by default. After fetching, set "
        "sherpa.denoise_enabled=true to activate it on the capture path.",
    )
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        raise SystemExit("Need huggingface_hub: python -m pip install huggingface-hub")
    from tools.bench.models import load_manifest

    token = os.environ.get("HUGGINGFACE_TOKEN") or None
    manifest = apply_accuracy(load_manifest(None), args.accuracy)
    print(f"[models] accuracy={args.accuracy} (ASR encoder: {manifest['asr_encoder']['file']})")
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

    # Speaker-ID model (optional). Non-fatal: a failed fetch must not block the
    # core ASR/TTS setup -- the gate just stays unconfigured (fail-open) until
    # the user re-runs. It's a GitHub release asset, so fetched by direct URL.
    if args.speaker_model:
        speaker_dest = os.path.join(args.dest, "speaker")
        print(f"[models] fetching speaker-ID model: {args.speaker_model_url} -> {speaker_dest}")
        try:
            resolved["speaker_embedding_model"] = fetch_speaker_model(
                speaker_dest, args.speaker_model_url, force=args.force
            )
        except Exception as exc:  # noqa: BLE001 - optional enhancement
            print(
                f"[models] speaker-ID model not fetched ({exc}); continuing without it. "
                "Speaker gating stays off until you re-run.",
                file=sys.stderr,
            )

    # Punctuation model (optional, opt-in). Same non-fatal contract: a failed
    # fetch leaves the ASR path on casing-restoration only.
    if args.punct_model:
        punct_dest = os.path.join(args.dest, "punct")
        print(f"[models] fetching punctuation model: {args.punct_model_url} -> {punct_dest}")
        try:
            resolved["punct_model"] = fetch_punct_model(
                punct_dest, args.punct_model_url, force=args.force
            )
        except Exception as exc:  # noqa: BLE001 - optional enhancement
            print(
                f"[models] punctuation model not fetched ({exc}); continuing without it. "
                "ASR finals keep casing restoration but no punctuation.",
                file=sys.stderr,
            )

    # Speech-denoise model (optional, opt-in). Same non-fatal contract: a failed
    # fetch leaves the capture path WITHOUT denoise (the default), never blocks
    # core ASR/TTS setup. GitHub release asset -> direct-URL streamed download.
    if args.denoise_model:
        denoise_dest = os.path.join(args.dest, "denoise")
        print(f"[models] fetching speech-denoise model: {args.denoise_model_url} -> {denoise_dest}")
        try:
            resolved["denoise_model"] = fetch_speaker_model(
                denoise_dest, args.denoise_model_url, force=args.force
            )
        except Exception as exc:  # noqa: BLE001 - optional enhancement
            print(
                f"[models] speech-denoise model not fetched ({exc}); continuing without it. "
                "Denoise stays off (sherpa.denoise_enabled) until you re-run.",
                file=sys.stderr,
            )

    # Write the absolute model paths into the machine-local overrides file
    # (gitignored, merged over config.json at load). Start from whatever is
    # already there so other local overrides survive.
    cfg: dict = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    wire_sherpa_paths(cfg, resolved)
    with open(args.config, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    sherpa = cfg["sherpa"]
    print(f"\n[models] {args.config} sherpa paths set:")
    for key in FILE_KEYS + ["tts_data_dir", "speaker_embedding_model", "punct_model", "denoise_model"]:
        if key in ("punct_model", "denoise_model") and not sherpa.get(key):
            continue  # optional + opt-in; don't print an empty line by default
        print(f"  {key}: {sherpa.get(key, '')}")
    if sherpa.get("speaker_embedding_model"):
        print("\nSpeaker-ID model ready. Enroll your voice:  python -m core --enroll")
    if sherpa.get("denoise_model"):
        print(
            "\nSpeech-denoise model ready. To ACTIVATE it on the capture path, set "
            'sherpa.denoise_enabled=true in your config (it is OFF by default). '
            "Re-enroll your voice afterwards (the embedding shifts vs un-denoised audio)."
        )
    print("\nNow run:  python -m core --engine sherpa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
