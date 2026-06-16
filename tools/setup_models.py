#!/usr/bin/env python3
"""Download the sherpa-onnx ASR/VAD/TTS models for the native (`--engine sherpa`)
path and wire their paths into config.json.

Run once, then start the assistant:

    python -m tools.setup_models
    python -m core --engine sherpa

Reuses the shared model manifest (tools/bench/models.py) so the coordinates live
in one place (override with the SPEAKER_BENCH_*_REPO / _FILE env vars). Models
land in pretrained_models/sherpa/ with flat, predictable filenames. The desktop
default serves the LLM via Ollama, so no GGUF is fetched by default; pass
``--gguf`` to also fetch the on-device Gemma GGUF weights (the llamacpp backend
used by the phone / phone_lite device profiles) into ``models/``. Idempotent:
re-running only fills missing files unless you pass --force.

    python -m tools.setup_models --gguf   # + on-device LLM weights (phone tier)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable

DEST = os.path.join("pretrained_models", "sherpa")
# On-device LLM weights (llamacpp backend) live here; the phone/phone_lite device
# profiles already point llm.main_model_path / fast_model_path at models/<file>.
GGUF_DIR = "models"
GGUF_KEYS = ("main_gguf", "fast_gguf")
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
# Optional DTLN-aec deep echo-canceller (the AEC quality tier, core/engines/_aec.py
# aec_backend="dtln"). Upstream ships TFLite only, so we fetch the two stage tflite
# files (breizhn/DTLN-aec GitHub raw assets) and CONVERT them to ONNX with tf2onnx
# (a dev-time dep: pip install tf2onnx tensorflow-cpu; not needed at runtime, only
# onnxruntime is). Size 512 = highest quality (~10.4M params); 256/128 are lighter.
DTLN_AEC_BASE = os.environ.get(
    "DTLN_AEC_BASE_URL", "https://github.com/breizhn/DTLN-aec/raw/main/pretrained_models"
)

# Optional PROSODY turn-completion model for the semantic endpoint (Smart Turn v3,
# pipecat-ai/smart-turn-v3, BSD-2). A single ~8.7 MB .onnx HF repo file -- read the
# audio waveform, predict P(turn complete), so the endpoint floor can drop without
# splitting. Override with --turn-model-url or the SMART_TURN_MODEL_URL env var.
SMART_TURN_MODEL_URL = (
    "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/"
    "smart-turn-v3.2-cpu.onnx"
)

# Optional OFFLINE second-pass ASR for the FINAL transcript (SenseVoice). A
# tar.bz2 GitHub release asset; we extract model.int8.onnx + tokens.txt. Robust on
# run-on/casual speech with punctuation+casing+ITN; ~150ms/utterance. Override with
# --sense-voice-url or the SENSE_VOICE_MODEL_URL env var.
SENSE_VOICE_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
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


def fetch_gguf_models(
    manifest: dict,
    gguf_dir: str = GGUF_DIR,
    *,
    token: "str | None" = None,
    force: bool = False,
    download: "Callable | None" = None,
) -> dict:
    """Fetch the on-device Gemma GGUF weights (llamacpp backend) into ``gguf_dir``.

    Returns ``{key: local_path}`` for each of ``GGUF_KEYS``. Coordinates come
    from the shared bench manifest (one source of truth); the phone / phone_lite
    device profiles already point ``llm.main_model_path`` / ``fast_model_path``
    at ``gguf_dir/<file>``, so a successful fetch makes those profiles runnable
    with no config rewrite. ``download`` defaults to ``huggingface_hub``'s
    ``hf_hub_download`` (injected in tests)."""
    if download is None:
        from huggingface_hub import hf_hub_download as download  # gated -> needs token
    os.makedirs(gguf_dir, exist_ok=True)
    resolved: dict = {}
    for key in GGUF_KEYS:
        coords = manifest[key]
        print(f"[models] fetching {key}: {coords['repo']}/{coords['file']} -> {gguf_dir}")
        resolved[key] = download(
            repo_id=coords["repo"],
            filename=coords["file"],
            local_dir=gguf_dir,
            token=token,
            force_download=force,
        )
    return resolved


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
    parser.add_argument(
        "--turn-model-url",
        dest="turn_model_url",
        default=os.environ.get("SMART_TURN_MODEL_URL", SMART_TURN_MODEL_URL),
        help="URL of the optional Smart Turn prosody endpoint model .onnx (override "
        "or set SMART_TURN_MODEL_URL)",
    )
    parser.add_argument(
        "--turn-model",
        dest="turn_model",
        action="store_true",
        help="also download the optional prosody turn-completion model (Smart Turn "
        "v3, ~8.7 MB) and wire endpoint_prosody_model in config; off by default. "
        "After fetching, set sherpa.endpoint_detector=prosody to activate it.",
    )
    parser.add_argument(
        "--sense-voice-url",
        dest="sense_voice_url",
        default=os.environ.get("SENSE_VOICE_MODEL_URL", SENSE_VOICE_MODEL_URL),
        help="URL of the optional SenseVoice second-pass ASR tar.bz2 (override or "
        "set SENSE_VOICE_MODEL_URL)",
    )
    parser.add_argument(
        "--sense-voice",
        dest="sense_voice",
        action="store_true",
        help="also download the optional SenseVoice offline second-pass ASR (~230 MB) "
        "and wire asr_final_backend=sense_voice in config; off by default. Robust on "
        "run-on/casual speech (the streaming zipformer garbles it).",
    )
    parser.add_argument(
        "--aec-model",
        dest="aec_model",
        action="store_true",
        help="also download the DTLN-aec echo-canceller tflite models and CONVERT "
        "them to ONNX (the AEC 'dtln' quality tier) + wire aec_model in config; off "
        "by default. Needs tf2onnx + tensorflow-cpu (dev-time only). After fetching, "
        "set sherpa.aec_enabled=true + aec_backend='dtln' (and calibrate "
        "aec_ref_delay_ms) to activate it.",
    )
    parser.add_argument(
        "--aec-model-size",
        dest="aec_model_size",
        choices=["128", "256", "512"],
        default="512",
        help="DTLN-aec model size: 512 = best quality (~10.4M), 256/128 lighter (phone)",
    )
    parser.add_argument(
        "--gguf",
        action="store_true",
        help="also fetch the on-device Gemma GGUF weights (llamacpp backend; "
             "phone/phone_lite profiles) into --gguf-dir",
    )
    parser.add_argument(
        "--gguf-dir",
        dest="gguf_dir",
        default=GGUF_DIR,
        help=f"dir for the on-device GGUF weights (default: {GGUF_DIR})",
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

    # Prosody turn-completion model (optional, opt-in). Same non-fatal contract: a
    # failed fetch leaves the endpoint on the lexical detector (the default), never
    # blocks core ASR/TTS setup. HF repo file -> direct-URL streamed download.
    if args.turn_model:
        turn_dest = os.path.join(args.dest, "turn")
        print(f"[models] fetching prosody turn model: {args.turn_model_url} -> {turn_dest}")
        try:
            resolved["endpoint_prosody_model"] = fetch_speaker_model(
                turn_dest, args.turn_model_url, force=args.force
            )
        except Exception as exc:  # noqa: BLE001 - optional enhancement
            print(
                f"[models] prosody turn model not fetched ({exc}); continuing without it. "
                "The endpoint stays on the lexical detector until you re-run.",
                file=sys.stderr,
            )

    # SenseVoice offline second-pass ASR (optional, opt-in). tar.bz2 archive ->
    # extract model.int8.onnx + tokens.txt. Same non-fatal contract: a failed
    # fetch leaves the final on the streaming model. Sets the backend after wiring
    # (it's a mode string, not a path).
    want_sense_voice = False
    if args.sense_voice:
        sv_dest = os.path.join(args.dest, "sense_voice")
        print(f"[models] fetching SenseVoice second-pass ASR: {args.sense_voice_url} -> {sv_dest}")
        try:
            archive = fetch_speaker_model(sv_dest, args.sense_voice_url, force=args.force)
            resolved["asr_final_model"] = extract_member(archive, "model.int8.onnx", sv_dest)
            resolved["asr_final_tokens"] = extract_member(archive, "tokens.txt", sv_dest)
            want_sense_voice = True
        except Exception as exc:  # noqa: BLE001 - optional enhancement
            print(
                f"[models] SenseVoice not fetched ({exc}); continuing without it. "
                "The final transcript stays on the streaming model until you re-run.",
                file=sys.stderr,
            )

    # DTLN-aec echo canceller (optional, opt-in). Upstream ships TFLite only, so we
    # fetch the two stage tflite files and convert each to ONNX with tf2onnx. Same
    # non-fatal contract: a failed fetch/convert leaves AEC on the NumPy 'nlms'
    # backend (or off). tf2onnx + tensorflow are dev-time only (runtime uses
    # onnxruntime); a missing converter is reported, not fatal.
    if args.aec_model:
        import subprocess

        aec_dest = os.path.join(args.dest, "aec")
        os.makedirs(aec_dest, exist_ok=True)
        size = args.aec_model_size
        try:
            for stage, out_name in ((1, "dtln_aec_stage1.onnx"), (2, "dtln_aec_stage2.onnx")):
                url = f"{DTLN_AEC_BASE}/dtln_aec_{size}_{stage}.tflite"
                print(f"[models] fetching DTLN-aec stage {stage}: {url}")
                tflite = fetch_speaker_model(aec_dest, url, force=args.force)
                out_onnx = os.path.join(aec_dest, out_name)
                print(f"[models] converting {os.path.basename(tflite)} -> {out_name} (tf2onnx)")
                subprocess.run(
                    [sys.executable, "-m", "tf2onnx.convert", "--tflite", tflite,
                     "--output", out_onnx, "--opset", "13"],
                    check=True,
                )
                try:
                    os.remove(tflite)  # the tflite is only needed for the conversion
                except OSError:
                    pass
            resolved["aec_model"] = aec_dest
        except Exception as exc:  # noqa: BLE001 - optional enhancement
            print(
                f"[models] DTLN-aec model not prepared ({exc}); continuing without it. "
                "The 'dtln' AEC tier needs tf2onnx + tensorflow-cpu "
                "(pip install tf2onnx tensorflow-cpu); the 'nlms' backend needs nothing.",
                file=sys.stderr,
            )

    # On-device LLM weights (optional, opt-in): the Gemma GGUFs for the llamacpp
    # backend (phone / phone_lite profiles). The desktop default uses Ollama and
    # needs none. Same non-fatal contract: a failed fetch leaves the llamacpp
    # backend unrunnable until you re-run, but never blocks the sherpa setup.
    if args.gguf:
        try:
            gguf_paths = fetch_gguf_models(
                manifest, args.gguf_dir, token=token, force=args.force
            )
            print(
                "\nOn-device LLM weights ready:\n"
                f"  main: {gguf_paths['main_gguf']}\n"
                f"  fast: {gguf_paths['fast_gguf']}\n"
                "The phone / phone_lite device profiles already point "
                "llm.main_model_path / fast_model_path at these. Install the runtime "
                "with `pip install -r requirements-ondevice.txt`, then run:  "
                "python -m core --engine sherpa --device phone"
            )
        except Exception as exc:  # noqa: BLE001 - optional, non-fatal
            print(
                f"[models] GGUF weights not fetched ({exc}); the llamacpp backend "
                "(phone profiles) stays unrunnable until you re-run with --gguf. "
                "The gated Gemma repo needs $HUGGINGFACE_TOKEN.",
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
    if want_sense_voice:
        # backend is a mode string, not a path -> set it directly (wire_sherpa_paths
        # only handles paths). The model/tokens paths were wired above.
        cfg.setdefault("sherpa", {})["asr_final_backend"] = "sense_voice"
    with open(args.config, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    sherpa = cfg["sherpa"]
    print(f"\n[models] {args.config} sherpa paths set:")
    for key in FILE_KEYS + [
        "tts_data_dir", "speaker_embedding_model", "punct_model", "denoise_model",
        "endpoint_prosody_model", "aec_model",
    ]:
        if key in ("punct_model", "denoise_model", "endpoint_prosody_model", "aec_model") and not sherpa.get(key):
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
    if sherpa.get("endpoint_prosody_model"):
        print(
            "\nProsody turn model ready. To ACTIVATE the prosodic end-of-turn detector, set "
            "sherpa.endpoint_detector=prosody (it is OFF by default; needs onnxruntime). "
            "VALIDATE on device first:  "
            "python -m tools.live_session --all --inject --smart-endpoint"
        )
    if sherpa.get("aec_model"):
        print(
            "\nDTLN-aec model ready. To ACTIVATE the deep echo-canceller tier, set "
            "sherpa.aec_enabled=true + sherpa.aec_backend='dtln' (OFF by default) and "
            "CALIBRATE sherpa.aec_ref_delay_ms with tools/echo_probe.py. The "
            "dependency-free aec_backend='nlms' filter needs no model."
        )
    print("\nNow run:  python -m core --engine sherpa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
