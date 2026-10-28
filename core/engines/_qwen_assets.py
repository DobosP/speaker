"""Pinned local Qwen CPU assets. This module never downloads or loads a model."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import sys

MODEL_FILE = "Qwen3-ASR-0.6B-Q8_0.gguf"
MODEL_SIZE = 841502336
MODEL_SHA256 = "416e10c15b4a3d9002bd337d18fc450233fdf68502b6e10d1379d2789838afd0"
MODEL_URL = "https://huggingface.co/sakasegawa/Qwen3-ASR-0.6B-GGUF/resolve/ee78f8709547f39bea3b924a16c72403826cac92/" + MODEL_FILE
SDK_FILE = "speech-0.9.0-linux-x64-cpu.zip"
SDK_SIZE = 5471367
SDK_SHA256 = "e2e9a260fde6242e43f7fa57c658321c798d4f1f855b1adf973dd50361b5805b"
SDK_URL = "https://github.com/nyosegawa/speech.cpp/releases/download/v0.9.0/" + SDK_FILE
LIBRARY_FILE = "libspeech.so.3"
LIBRARY_SIZE = 6001680
LIBRARY_SHA256 = "935cf967511af4d4baf8e80a865edbcfa70ec4b305aa00695ea9cc191aa03b93"
SENSE_FILES = {
    "model.int8.onnx": (239233841, "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51"),
    "tokens.txt": (315894, "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc"),
}
BUNDLE_MANIFEST = "qwen-cpu.json"


class QwenAssetError(ValueError):
    """A fixed, detail-free local asset refusal."""


def supported_platform() -> bool:
    return sys.platform == "linux" and platform.machine().lower() in {"x86_64", "amd64"}


def _regular(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise QwenAssetError("qwen_asset_invalid")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise QwenAssetError("qwen_asset_invalid")
    return path


def read_regular(path: Path, maximum: int) -> bytes:
    """Read a small regular local file through a bounded stable no-follow FD."""
    try:
        path = _regular(path)
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= maximum:
                raise QwenAssetError("qwen_asset_invalid")
            raw = stream.read(maximum + 1)
            after = os.fstat(stream.fileno())
        def identity(value):
            return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
        if len(raw) != before.st_size or identity(before) != identity(after) or identity(after) != identity(path.lstat()):
            raise QwenAssetError("qwen_asset_invalid")
        return raw
    except (OSError, ValueError, TypeError):
        raise QwenAssetError("qwen_asset_invalid") from None


def verified_file(path: Path, expected_size: int, expected_sha256: str) -> Path:
    """Verify exact bytes through a bounded no-follow FD; reject changing files."""
    try:
        path = _regular(path)
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
                raise QwenAssetError("qwen_asset_invalid")
            checksum = hashlib.sha256()
            size = 0
            while chunk := stream.read(min(1024**2, expected_size + 1 - size)):
                checksum.update(chunk)
                size += len(chunk)
                if size > expected_size:
                    raise QwenAssetError("qwen_asset_invalid")
            after = os.fstat(stream.fileno())
        def identity(value):
            return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
        if size != expected_size or checksum.hexdigest() != expected_sha256 or identity(before) != identity(after) or identity(after) != identity(path.lstat()):
            raise QwenAssetError("qwen_asset_invalid")
        return path
    except (OSError, ValueError, TypeError):
        raise QwenAssetError("qwen_asset_invalid") from None


def validate_qwen_bundle(bundle_dir: Path) -> tuple[Path, Path]:
    """Return exact pinned (model_path, library_path); Linux CPU is qualified."""
    if not supported_platform():
        raise QwenAssetError("qwen_cpu_platform_unsupported")
    try:
        data = json.loads(read_regular(Path(bundle_dir).absolute() / BUNDLE_MANIFEST, 4096))
        if type(data) is not dict or set(data) != {"schema_version", "model", "library"} or type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise QwenAssetError("qwen_asset_invalid")
        if any(type(data[key]) is not str or not data[key] for key in ("model", "library")):
            raise QwenAssetError("qwen_asset_invalid")
        model = verified_file(Path(data["model"]), MODEL_SIZE, MODEL_SHA256)
        library = verified_file(Path(data["library"]), LIBRARY_SIZE, LIBRARY_SHA256)
        return model, library
    except (OSError, ValueError, TypeError):
        raise QwenAssetError("qwen_asset_invalid") from None


def validate_sense_voice(directory: Path) -> Path:
    root = Path(directory).absolute()
    for name, (size, sha256) in SENSE_FILES.items():
        verified_file(root / name, size, sha256)
    return root


PARAKEET_TDT_FILES = {
    "decoder.int8.onnx": (11845275, "179e50c43d1a9de79c8a24149a2f9bac6eb5981823f2a2ed88d655b24248db4e"),
    "encoder.int8.onnx": (652184281, "acfc2b4456377e15d04f0243af540b7fe7c992f8d898d751cf134c3a55fd2247"),
    "joiner.int8.onnx": (6355277, "3164c13fc2821009440d20fcb5fdc78bff28b4db2f8d0f0b329101719c0948b3"),
    "tokens.txt": (93939, "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d"),
}


def validate_parakeet_tdt(directory: Path) -> Path:
    """The measured INT8 TDT v3 voter, not a lookalike NeMo checkpoint."""
    root = Path(directory).absolute()
    for name, (size, sha256) in PARAKEET_TDT_FILES.items():
        verified_file(root / name, size, sha256)
    return root
