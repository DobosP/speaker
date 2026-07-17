#!/usr/bin/env python3
"""Download the sherpa-onnx ASR/VAD/TTS models for the native (`--engine sherpa`)
path and wire their paths into config.local.json.

Run once, then start the assistant:

    python -m tools.setup_models
    ./live.sh  # normal Linux physical-session entry

Reuses the shared model manifest (tools/bench/models.py) so the coordinates live
in one place (override with the SPEAKER_BENCH_*_REPO / _FILE env vars). Models
land in pretrained_models/sherpa/ with flat, predictable filenames. The desktop
default serves the LLM via Ollama, so no GGUF is fetched by default; pass
``--gguf`` to also fetch the on-device MiniCPM5 GGUF weights (the llamacpp backend
used by the phone / phone_lite device profiles) into ``models/``. Idempotent:
re-running only fills missing files unless you pass --force.

    python -m tools.setup_models --gguf   # + on-device LLM weights (phone tier)

The cross-platform installer adds ``--sense-voice --denoise-model --kokoro
--require-selected``. ``--final-asr parakeet-unified-en`` replaces that
SenseVoice selection explicitly. In either mode the default speaker model and
every selected artifact must exist before one atomic config.local.json
publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
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
# Speaker-embedding model for normal-input gating and optional generic word-cut
# owner filtering (plus the own-TTS control-ambiguity guard).
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

# Optional high-accuracy English OFFLINE final recognizer. This is the exact
# sherpa-onnx NeMo transducer export measured on the private endpoint replay;
# its package checksum prevents a mutable/truncated archive from being wired.
PARAKEET_FINAL_MODEL_URL = os.environ.get(
    "PARAKEET_FINAL_MODEL_URL",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming.tar.bz2",
)
PARAKEET_FINAL_MODEL_SHA256 = (
    "99f63605b3a85a54c250c0869670a687b7d6598a47bf2421515e1f839a76e150"
)
PARAKEET_FINAL_DIR = (
    "sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming"
)
PARAKEET_FINAL_FILES = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
)
PARAKEET_FINAL_FILE_SHA256 = {
    "encoder.int8.onnx": "6716910b7a0833997fec7a410494c995d70124001a0e9b66d6370d6aced577e0",
    "decoder.int8.onnx": "a5e223392c90e75f8144cdb5eb95af7625db389e39edef2bd1a9c872b3298fe6",
    "joiner.int8.onnx": "869f43f7d24595c55581ad3bf249a935fb8a71389fbdaa7504b9f46f93140f8a",
    "tokens.txt": "dc0b4584ab2e4ddbf888425c076c61b736e7356a015250db7d307e6f1a8188ff",
}


def normal_voice_entry(platform_name: str | None = None) -> str:
    """Return the supported normal app entry for the current platform."""
    selected = sys.platform if platform_name is None else platform_name
    return "./live.sh" if selected.startswith("linux") else "python -m core --engine sherpa"

# Optional independent endpointed-ASR verifier. The empirically selected local
# development candidate is Faster-Whisper Small; it runs through CTranslate2 on
# CUDA and votes only in exact acoustic consensus. The setup publishes a local
# directory, never a remote model identifier, so normal runtime cannot download.
FASTER_WHISPER_VERIFIER_REPO = os.environ.get(
    "FASTER_WHISPER_VERIFIER_REPO",
    "Systran/faster-whisper-small",
)
# Exact snapshot used by the locked 2026-07-17 recording/GPU evaluation.
FASTER_WHISPER_VERIFIER_REVISION = (
    "536b0662742c02347bc0e980a01041f333bce120"
)
FASTER_WHISPER_REQUIRED_FILES = (
    "config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.txt",
)

# Optional Kokoro TTS (ADR-0010: the adopted desktop voice). Ships as a
# sherpa-onnx tts-models release .tar.bz2 bundling model.int8.onnx + voices.bin
# + tokens.txt + lexicon-*.txt + espeak-ng-data/ (+ dict/), so unlike the
# single-file assets the WHOLE package is unpacked (into <dest>/tts_kokoro/).
# build_tts keys the Kokoro path on tts_voices being set. Override with
# --kokoro-url or the KOKORO_TTS_URL env var.
KOKORO_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
    "kokoro-int8-multi-lang-v1_1.tar.bz2"
)


def dest_for(base: str, key: str) -> str:
    """Per-artifact download dir so same-named files (tokens.txt) don't collide."""
    return os.path.join(base, SUBDIR.get(key, ""))


def fetch_faster_whisper_verifier(
    dest_root: str,
    *,
    repo_id: str,
    revision: str,
    token: str | None,
    force: bool,
    snapshot_download_fn: Callable[..., str],
) -> str:
    """Stage and atomically publish one immutable verifier snapshot.

    Downloads never write into the active model directory. A failed fetch
    leaves both that directory and config untouched; a forced replacement
    retains the previous directory under a unique sibling name instead of
    deleting it. ``revision`` must be a full commit hash so a setup rerun cannot
    silently drift to a mutable repository head.
    """
    import re

    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Faster-Whisper revision must be a full commit hash")
    os.makedirs(dest_root, exist_ok=True)
    basename = f"faster_whisper_small-{revision[:12]}"
    destination = os.path.join(dest_root, basename)

    def complete(path: str) -> bool:
        return os.path.isdir(path) and all(
            os.path.isfile(os.path.join(path, name))
            for name in FASTER_WHISPER_REQUIRED_FILES
        )

    if complete(destination) and not force:
        return destination

    staging = tempfile.mkdtemp(prefix=f".{basename}-staging-", dir=dest_root)
    snapshot_download_fn(
        repo_id=repo_id,
        revision=revision,
        local_dir=staging,
        token=token,
        force_download=force,
    )
    if not complete(staging):
        raise FileNotFoundError("incomplete Faster-Whisper snapshot")

    backup = None
    if os.path.lexists(destination):
        backup = f"{destination}.previous-{os.path.basename(staging)}"
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if backup is not None:
            os.replace(backup, destination)
        raise
    return destination


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


def fetch_parakeet_final(
    dest_root: str,
    *,
    url: str,
    expected_sha256: str,
    force: bool = False,
) -> dict[str, str]:
    """Verify, stage, and atomically publish the selected Parakeet package.

    The release archive remains beside the published model for reproducible
    provenance. Extraction is flattened into a new staging directory, so tar
    paths cannot escape it and an interrupted setup never mutates the active
    model. Forced replacement preserves the old directory under a unique
    sibling name.
    """
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("Parakeet SHA-256 must be 64 lowercase hex characters")
    os.makedirs(dest_root, exist_ok=True)
    destination = os.path.join(dest_root, PARAKEET_FINAL_DIR)

    def resolved(path: str) -> dict[str, str]:
        return {
            "asr_final_model": os.path.join(path, "encoder.int8.onnx"),
            "asr_final_decoder": os.path.join(path, "decoder.int8.onnx"),
            "asr_final_joiner": os.path.join(path, "joiner.int8.onnx"),
            "asr_final_tokens": os.path.join(path, "tokens.txt"),
        }

    def complete(path: str) -> bool:
        return os.path.isdir(path) and all(
            os.path.isfile(os.path.join(path, name))
            for name in PARAKEET_FINAL_FILES
        )

    archive = fetch_speaker_model(dest_root, url, force=force)
    if _sha256_file(archive) != expected_sha256:
        raise ValueError("Parakeet archive checksum mismatch")
    if complete(destination) and not force:
        if not _parakeet_package_matches_archive(
            destination,
            archive,
            expected_sha256=expected_sha256,
        ):
            raise ValueError(
                "existing Parakeet model does not match the verified archive; "
                "rerun setup with --force"
            )
        return resolved(destination)

    staging = tempfile.mkdtemp(
        prefix=f".{PARAKEET_FINAL_DIR}-staging-",
        dir=dest_root,
    )
    for name in PARAKEET_FINAL_FILES:
        extract_member(archive, name, staging)
    if not complete(staging) or not _parakeet_package_matches_archive(
        staging,
        archive,
        expected_sha256=expected_sha256,
    ):
        raise FileNotFoundError("incomplete Parakeet model package")

    backup = None
    if os.path.lexists(destination):
        backup = f"{destination}.previous-{os.path.basename(staging)}"
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if backup is not None:
            os.replace(backup, destination)
        raise
    return resolved(destination)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parakeet_package_matches_archive(
    destination: str,
    archive: str,
    *,
    expected_sha256: str,
) -> bool:
    """Bind an installed four-file package to its checksum-verified archive."""
    import tarfile

    installed = {
        name: _sha256_file(os.path.join(destination, name))
        for name in PARAKEET_FINAL_FILES
    }
    # The selected package has hard-pinned extracted artifacts, avoiding a slow
    # bzip2 re-decompression on every idempotent setup. Explicit custom
    # URL/checksum overrides still compare each installed file to that archive.
    if expected_sha256 == PARAKEET_FINAL_MODEL_SHA256:
        return installed == PARAKEET_FINAL_FILE_SHA256
    archived: dict[str, str] = {}
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        for name in PARAKEET_FINAL_FILES:
            matches = [
                member
                for member in members
                if member.isfile() and member.name.endswith(name)
            ]
            if len(matches) != 1:
                return False
            source = tar.extractfile(matches[0])
            if source is None:
                return False
            digest = hashlib.sha256()
            with source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            archived[name] = digest.hexdigest()
    return installed == archived


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


def fetch_kokoro_package(dest_dir: str, url: str, *, force: bool = False) -> dict:
    """Download + unpack the Kokoro TTS package, returning its config paths.

    Returns ``{tts_model, tts_voices, tts_tokens, tts_data_dir, tts_lexicon}``
    (``tts_data_dir``/``tts_lexicon`` empty when the package ships none). The
    release .tar.bz2 nests everything under one top-level dir; members are
    re-rooted past that prefix into ``dest_dir`` with the same path-traversal
    guard as ``extract_member`` (each file is written via ``extractfile``,
    never ``tar.extract``). Idempotent: when the model + voices already exist
    the download and unpack are skipped (unless ``force``). The archive is
    removed after a successful unpack."""
    import shutil
    import tarfile

    os.makedirs(dest_dir, exist_ok=True)

    def _resolve() -> dict:
        model = ""
        for name in ("model.int8.onnx", "model.onnx"):
            cand = os.path.join(dest_dir, name)
            if os.path.exists(cand):
                model = cand
                break
        voices = os.path.join(dest_dir, "voices.bin")
        tokens = os.path.join(dest_dir, "tokens.txt")
        data_dir = os.path.join(dest_dir, "espeak-ng-data")
        lexicon = os.path.join(dest_dir, "lexicon-us-en.txt")
        if not os.path.exists(lexicon):
            others = sorted(
                f for f in os.listdir(dest_dir)
                if f.startswith("lexicon") and f.endswith(".txt")
            )
            lexicon = os.path.join(dest_dir, others[0]) if others else ""
        return {
            "tts_model": model,
            "tts_voices": voices if os.path.exists(voices) else "",
            "tts_tokens": tokens if os.path.exists(tokens) else "",
            "tts_data_dir": data_dir if os.path.isdir(data_dir) else "",
            "tts_lexicon": lexicon,
        }

    have = _resolve()
    # All three REQUIRED files must exist for the skip -- a partial prior unpack
    # (e.g. missing tokens.txt) must re-fetch, or wire_sherpa_paths would keep a
    # stale Piper tokens path beside Kokoro voices (codex-review 2026-07-06).
    if have["tts_model"] and have["tts_voices"] and have["tts_tokens"] and not force:
        return have

    archive = fetch_speaker_model(dest_dir, url, force=force)
    dest_root = os.path.realpath(dest_dir)
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            parts = [
                p for p in member.name.replace("\\", "/").split("/") if p and p != "."
            ]
            if any(p == ".." for p in parts):
                raise ValueError(f"unsafe member path in {archive}: {member.name}")
            # Strip the package's top-level dir; tolerate root-level members.
            rel = parts[1:] if len(parts) > 1 else parts
            if not rel:
                continue
            out_path = os.path.join(dest_dir, *rel)
            if not os.path.realpath(out_path).startswith(dest_root + os.sep):
                raise ValueError(f"unsafe member path in {archive}: {member.name}")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with src, open(out_path, "wb") as out:
                shutil.copyfileobj(src, out)
    try:
        os.remove(archive)  # the .tar.bz2 is no longer needed once unpacked
    except OSError:
        pass
    got = _resolve()
    if not (got["tts_model"] and got["tts_voices"] and got["tts_tokens"]):
        raise FileNotFoundError(
            f"Kokoro package from {url} lacked model/voices.bin/tokens.txt after unpack"
        )
    return got


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


def required_selected_artifact_errors(
    resolved: dict[str, str],
    *,
    speaker_model: bool,
    denoise_model: bool,
    sense_voice: bool,
    kokoro: bool,
    parakeet_final: bool = False,
    final_verifier: bool = False,
    exists: Callable[[str], bool] = os.path.exists,
) -> list[str]:
    """Return missing artifacts for a fail-closed selected-profile publish.

    The ordinary/manual downloader keeps its historical best-effort optional
    behavior. ``--require-selected`` calls this after every fetch and before it
    touches config, making the installer-selected desktop stack transactional
    from the configuration's point of view.
    """
    required = {key: key for key in FILE_KEYS}
    if speaker_model:
        required["speaker_embedding_model"] = "default speaker-ID model"
    if denoise_model:
        required["denoise_model"] = "selected GTCRN denoise model"
    if sense_voice:
        required["asr_final_model"] = "selected SenseVoice model"
        required["asr_final_tokens"] = "selected SenseVoice tokens"
    if parakeet_final:
        required.update(
            {
                "asr_final_model": "selected Parakeet encoder",
                "asr_final_decoder": "selected Parakeet decoder",
                "asr_final_joiner": "selected Parakeet joiner",
                "asr_final_tokens": "selected Parakeet tokens",
            }
        )
    if final_verifier:
        required["asr_final_verifier_model"] = (
            "selected Faster-Whisper verifier model"
        )
    if kokoro:
        # tts_model/tts_tokens are already in FILE_KEYS. Voices are the switch
        # that selects the Kokoro constructor instead of the fallback VITS one.
        required["tts_voices"] = "selected Kokoro voices"
        required["tts_data_dir"] = "selected Kokoro espeak data"
        required["tts_lexicon"] = "selected Kokoro lexicon"

    errors: list[str] = []
    for key, label in required.items():
        path = str(resolved.get(key, "") or "")
        if not path:
            errors.append(f"{label} was not resolved")
        elif not exists(path):
            errors.append(f"{label} is missing on disk")
    return errors


def publish_config_atomic(config: dict, path: str) -> None:
    """Atomically replace the machine-local JSON configuration.

    A failed serialization/fsync/replace leaves the previous config byte-for-
    byte intact. The temporary file is created beside the target so ``replace``
    stays on one filesystem on Linux, macOS, and Windows.
    """
    target = os.path.abspath(path)
    parent = os.path.dirname(target) or os.curdir
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.", suffix=".tmp", dir=parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, target)
        temporary = ""
        directory_fd = -1
        try:
            directory_fd = os.open(
                parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            os.fsync(directory_fd)
        except OSError:
            pass
        finally:
            if directory_fd >= 0:
                os.close(directory_fd)
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def fetch_gguf_models(
    manifest: dict,
    gguf_dir: str = GGUF_DIR,
    *,
    token: "str | None" = None,
    force: bool = False,
    download: "Callable | None" = None,
) -> dict:
    """Fetch the on-device MiniCPM5 GGUF weights into ``gguf_dir``.

    Returns ``{key: local_path}`` for each of ``GGUF_KEYS``. Coordinates come
    from the shared bench manifest (one source of truth); the phone / phone_lite
    device profiles already point ``llm.main_model_path`` / ``fast_model_path``
    at ``gguf_dir/<file>``, so a successful fetch makes those profiles runnable
    with no config rewrite. ``download`` defaults to ``huggingface_hub``'s
    ``hf_hub_download`` (injected in tests)."""
    if download is None:
        from huggingface_hub import hf_hub_download as download
    os.makedirs(gguf_dir, exist_ok=True)
    resolved: dict = {}
    fetched: dict[tuple[str, str], str] = {}
    for key in GGUF_KEYS:
        coords = manifest[key]
        identity = (coords["repo"], coords["file"])
        if identity not in fetched:
            print(f"[models] fetching {key}: {coords['repo']}/{coords['file']} -> {gguf_dir}")
            fetched[identity] = download(
                repo_id=coords["repo"],
                filename=coords["file"],
                local_dir=gguf_dir,
                token=token,
                force_download=force,
            )
        resolved[key] = fetched[identity]
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch sherpa models + wire config.local.json")
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
        "--require-selected",
        action="store_true",
        help=(
            "fail without changing config unless the base models, default "
            "speaker-ID model, and requested SenseVoice/GTCRN/Kokoro artifacts "
            "are all present"
        ),
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
        "--final-asr",
        choices=["parakeet-unified-en"],
        help=(
            "download and configure the checksum-pinned English Parakeet "
            "offline final recognizer; normal startup remains ./live.sh"
        ),
    )
    parser.add_argument(
        "--parakeet-url",
        default=PARAKEET_FINAL_MODEL_URL,
        help="release archive URL for --final-asr parakeet-unified-en",
    )
    parser.add_argument(
        "--parakeet-sha256",
        default=PARAKEET_FINAL_MODEL_SHA256,
        help="expected lowercase SHA-256 for the selected Parakeet archive",
    )
    parser.add_argument(
        "--final-verifier",
        choices=["faster-whisper-small"],
        help=(
            "download and configure the local CUDA Faster-Whisper Small final "
            "verifier; normal startup remains ./live.sh"
        ),
    )
    parser.add_argument(
        "--final-verifier-repo",
        default=FASTER_WHISPER_VERIFIER_REPO,
        help=(
            "Hugging Face repo for --final-verifier "
            f"(default: {FASTER_WHISPER_VERIFIER_REPO})"
        ),
    )
    parser.add_argument(
        "--final-verifier-revision",
        default=FASTER_WHISPER_VERIFIER_REVISION,
        help=(
            "full immutable commit for --final-verifier "
            f"(default: {FASTER_WHISPER_VERIFIER_REVISION})"
        ),
    )
    parser.add_argument(
        "--kokoro-url",
        dest="kokoro_url",
        default=os.environ.get("KOKORO_TTS_URL", KOKORO_URL),
        help="URL of the Kokoro TTS package .tar.bz2 (override or set KOKORO_TTS_URL)",
    )
    parser.add_argument(
        "--kokoro",
        dest="kokoro",
        action="store_true",
        help="also download the Kokoro TTS package (ADR-0010 adopted desktop voice, "
        "~330 MB unpacked) and wire tts_model/tts_voices/tts_tokens/tts_data_dir/"
        "tts_lexicon in config; off by default. build_tts keys on tts_voices, so a "
        "successful fetch switches synthesis from the Piper/VITS voice to Kokoro.",
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
        help="also fetch the on-device MiniCPM5 GGUF weights (llamacpp backend; "
             "phone/phone_lite profiles) into --gguf-dir",
    )
    parser.add_argument(
        "--gguf-dir",
        dest="gguf_dir",
        default=GGUF_DIR,
        help=f"dir for the on-device GGUF weights (default: {GGUF_DIR})",
    )
    args = parser.parse_args(argv)
    if args.require_selected and not args.speaker_model:
        parser.error("--require-selected cannot be combined with --no-speaker-model")
    if args.sense_voice and args.final_asr:
        parser.error("--sense-voice and --final-asr are mutually exclusive")

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
    selected_failures: list[str] = []
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
            if args.require_selected:
                selected_failures.append(
                    f"default speaker-ID model fetch failed: {type(exc).__name__}: {exc}"
                )
            else:
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
            if args.require_selected:
                selected_failures.append(
                    f"selected GTCRN fetch failed: {type(exc).__name__}: {exc}"
                )
            else:
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
            if args.require_selected:
                selected_failures.append(
                    f"selected SenseVoice fetch failed: {type(exc).__name__}: {exc}"
                )
            else:
                print(
                    f"[models] SenseVoice not fetched ({exc}); continuing without it. "
                    "The final transcript stays on the streaming model until you re-run.",
                    file=sys.stderr,
                )

    # Checksum-pinned NeMo Parakeet final recognizer (optional, opt-in). The
    # complete four-file package is staged and atomically published before its
    # paths or backend can enter the machine-local configuration.
    want_parakeet_final = False
    if args.final_asr:
        print(
            "[models] fetching Parakeet offline final ASR: "
            f"{args.parakeet_url}"
        )
        try:
            resolved.update(
                fetch_parakeet_final(
                    args.dest,
                    url=args.parakeet_url,
                    expected_sha256=args.parakeet_sha256,
                    force=args.force,
                )
            )
            want_parakeet_final = True
        except Exception as exc:  # noqa: BLE001 - optional enhancement
            if args.require_selected:
                selected_failures.append(
                    "selected Parakeet fetch failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                print(
                    "[models] Parakeet final ASR not fetched; continuing with "
                    "the established final recognizer until you re-run.",
                    file=sys.stderr,
                )

    # Independent Faster-Whisper acoustic verifier (optional, opt-in). The
    # complete CTranslate2 snapshot is staged in a dedicated directory and the
    # config receives only its absolute local path. A partial snapshot is never
    # published as an enabled verifier.
    want_final_verifier = False
    if args.final_verifier:
        print(
            "[models] fetching Faster-Whisper final verifier: "
            f"{args.final_verifier_repo}@{args.final_verifier_revision}"
        )
        try:
            verifier_path = fetch_faster_whisper_verifier(
                args.dest,
                repo_id=args.final_verifier_repo,
                revision=args.final_verifier_revision,
                token=token,
                force=args.force,
                snapshot_download_fn=snapshot_download,
            )
            resolved["asr_final_verifier_model"] = verifier_path
            want_final_verifier = True
        except Exception as exc:  # noqa: BLE001 - optional enhancement
            if args.require_selected:
                selected_failures.append(
                    "selected Faster-Whisper verifier fetch failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                print(
                    "[models] Faster-Whisper verifier not fetched; continuing "
                    "with the established final selector until you re-run.",
                    file=sys.stderr,
                )
    # Kokoro TTS (optional, opt-in): the ADR-0010 adopted desktop voice -- a whole
    # package (model + voices.bin + tokens + espeak-ng-data + lexicon), not a
    # single file. Same non-fatal contract: a failed fetch leaves synthesis on the
    # Piper/VITS voice (tts_voices stays unset). Runs AFTER the Piper paths were
    # resolved above so its tts_* entries overwrite them in wire_sherpa_paths.
    if args.kokoro:
        kokoro_dest = os.path.join(args.dest, "tts_kokoro")
        print(f"[models] fetching Kokoro TTS package: {args.kokoro_url} -> {kokoro_dest}")
        try:
            resolved.update(
                fetch_kokoro_package(kokoro_dest, args.kokoro_url, force=args.force)
            )
        except Exception as exc:  # noqa: BLE001 - optional enhancement
            if args.require_selected:
                selected_failures.append(
                    f"selected Kokoro fetch failed: {type(exc).__name__}: {exc}"
                )
            else:
                print(
                    f"[models] Kokoro not fetched ({exc}); continuing on the Piper/VITS "
                    "voice. tts_voices stays unset until you re-run.",
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

    # On-device LLM weights (optional, opt-in): the MiniCPM5 GGUF for the llamacpp
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
                "Set $HUGGINGFACE_TOKEN if the public download is rate-limited.",
                file=sys.stderr,
            )

    if args.require_selected:
        selected_failures.extend(
            required_selected_artifact_errors(
                resolved,
                speaker_model=args.speaker_model,
                denoise_model=args.denoise_model,
                sense_voice=args.sense_voice,
                kokoro=args.kokoro,
                parakeet_final=bool(args.final_asr),
                final_verifier=bool(args.final_verifier),
            )
        )
        # Preserve order while avoiding a fetch exception + path validation
        # printing the same root failure twice.
        selected_failures = list(dict.fromkeys(selected_failures))
        if selected_failures:
            print(
                "[models] required selected setup failed; config was not changed:",
                file=sys.stderr,
            )
            for failure in selected_failures:
                print(f"  - {failure}", file=sys.stderr)
            return 1

    # Stage the complete override in memory and publish it with one atomic
    # replace. Start from the existing machine-local file so unrelated local
    # overrides survive; any read/write error is a failed setup, never a
    # partially-truncated config that the installer could mistake for success.
    try:
        cfg: dict = {}
        if os.path.exists(args.config):
            with open(args.config, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            if not isinstance(cfg, dict):
                raise ValueError("existing local config must contain a JSON object")
        wire_sherpa_paths(cfg, resolved)
        if want_sense_voice:
            # backend is a mode string, not a path -> set it directly
            # (wire_sherpa_paths only handles paths).
            cfg.setdefault("sherpa", {})["asr_final_backend"] = "sense_voice"
        if want_parakeet_final:
            cfg.setdefault("sherpa", {})[
                "asr_final_backend"
            ] = "nemo_transducer"
        if want_final_verifier:
            cfg.setdefault("sherpa", {})[
                "asr_final_verifier_backend"
            ] = "faster_whisper"
        publish_config_atomic(cfg, args.config)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"[models] config publish failed; previous config was preserved "
            f"({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
        return 1

    sherpa = cfg["sherpa"]
    print(f"\n[models] {args.config} sherpa paths set:")
    for key in FILE_KEYS + [
        "tts_voices", "tts_data_dir", "tts_lexicon",
        "speaker_embedding_model", "punct_model", "denoise_model",
        "asr_final_model", "asr_final_tokens", "asr_final_decoder",
        "asr_final_joiner",
        "asr_final_verifier_model",
        "endpoint_prosody_model", "aec_model",
    ]:
        if key in (
            "tts_voices", "tts_data_dir", "tts_lexicon", "punct_model",
            "denoise_model", "asr_final_model", "asr_final_tokens",
            "asr_final_decoder", "asr_final_joiner",
            "asr_final_verifier_model",
            "endpoint_prosody_model", "aec_model",
        ) and not sherpa.get(key):
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
    print(f"\nNow run:  {normal_voice_entry()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
