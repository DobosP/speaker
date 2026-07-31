"""Strict, hash-bound worker manifests for the streaming-STT harness."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    read_regular_bounded,
)
from .runtime_receipt import (
    RuntimeTreeReceiptError,
    verify_isolated_venv_marker,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_MAX_MANIFEST_BYTES = 64 * 1024
MAX_PYTHON_BYTES = 512 * 1024 * 1024
MAX_WORKER_BYTES = 4 * 1024 * 1024
MAX_FAKE_ARTIFACT_BYTES = 4 * 1024 * 1024
MAX_MOONSHINE_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_MOONSHINE_TOTAL_ARTIFACT_BYTES = 1024 * 1024 * 1024
MAX_NEMOTRON_ARTIFACT_BYTES = 3 * 1024 * 1024 * 1024
MAX_NEMOTRON_TOTAL_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024
_MANIFEST_V1_FIELDS = {
    "schema_version",
    "model_id",
    "adapter",
    "python",
    "worker",
    "artifacts",
    "limits",
}
_MANIFEST_V2_FIELDS = {*_MANIFEST_V1_FIELDS, "adapter_config"}
_MANIFEST_V3_FIELDS = _MANIFEST_V2_FIELDS
_FILE_FIELDS = {"path", "sha256", "size_bytes"}
_ARTIFACT_FIELDS = {"name", *_FILE_FIELDS}
_LIMIT_FIELDS = {"startup_timeout_sec", "case_timeout_sec"}
_MOONSHINE_CONFIG_FIELDS = {
    "package_version",
    "api_version",
    "model_arch",
    "provider",
    "language",
}
_NEMOTRON_CONFIG_FIELDS = {
    "python_version",
    "transformers_version",
    "librosa_version",
    "torch_version",
    "cuda_version",
    "model_revision",
    "lookahead_tokens",
    "language",
    "device",
    "dtype",
    "native_sample_rate_hz",
    "native_hop_length_samples",
    "native_n_fft_samples",
    "native_win_length_samples",
    "native_first_chunk_frames",
    "native_chunk_frames",
    "native_first_window_samples",
    "native_window_samples",
    "native_stride_samples",
    "streaming_latency_ms",
    "wheel_lock_sha256",
    "runtime_content_sha256",
    "runtime_file_count",
    "runtime_total_size_bytes",
    "runtime_maximum_file_bytes",
}
MOONSHINE_ADAPTER = "moonshine-voice-stream-v1"
NEMOTRON_ADAPTER = "transformers-nemotron-3.5-stream-v1"
NEMOTRON_WHEEL_LOCK_SHA256 = (
    "df268a2e268221428256b3ec525a3ad49da65b526b2e09b88df3802533b5af01"
)
NEMOTRON_RUNTIME_CONTENT_SHA256 = (
    "ebcf9cab82c93ab5503bb438fbc9585475fa138c4e18aaea73d9ab34fc48bb8f"
)
NEMOTRON_RUNTIME_FILE_COUNT = 24_273
NEMOTRON_RUNTIME_TOTAL_SIZE_BYTES = 6_752_927_292
NEMOTRON_RUNTIME_MAXIMUM_FILE_BYTES = 1_007_735_593
MOONSHINE_ARTIFACT_NAMES = (
    "runtime-receipt",
    "venv-marker",
    "release-wheel",
    "model-adapter",
    "model-cross-kv",
    "model-decoder-kv",
    "model-encoder",
    "model-frontend",
    "model-config",
    "model-tokenizer",
)
_MOONSHINE_ARTIFACT_BASENAMES = {
    "runtime-receipt": "runtime-receipt.json",
    "venv-marker": "pyvenv.cfg",
    "release-wheel": ("moonshine_voice-0.1.0-py3-none-manylinux_2_34_x86_64.whl"),
    "model-adapter": "adapter.ort",
    "model-cross-kv": "cross_kv.ort",
    "model-decoder-kv": "decoder_kv.ort",
    "model-encoder": "encoder.ort",
    "model-frontend": "frontend.ort",
    "model-config": "streaming_config.json",
    "model-tokenizer": "tokenizer.bin",
}
_MOONSHINE_SMALL_ARTIFACTS = {
    "runtime-receipt": 4 * 1024 * 1024,
    "venv-marker": 64 * 1024,
    "release-wheel": 128 * 1024 * 1024,
}
_MOONSHINE_RELEASE_RECEIPT = (
    "0f833deb43bad5dcfb4cfd3257b6df83ef9abd3f27be3199622fe41932e8d916",
    56_369_045,
)
_MOONSHINE_MODEL_RECEIPTS = {
    "tiny-streaming": {
        "model-adapter": (
            "df13e655b29d279911fcb42d8b91b0e655b8fe32b7ba1f463ece663ce55ae6eb",
            1_319_440,
        ),
        "model-cross-kv": (
            "5acfca68f7bb068c68c1960b54e215995ba07ee46b61645b78bff010a14e5a92",
            1_264_384,
        ),
        "model-decoder-kv": (
            "6e3828f1db4b634bc525cb8ba1f0b628ec56059168f0336ad060891c7c1c9154",
            32_403_688,
        ),
        "model-encoder": (
            "96dde726be90c4429f3bc458d04e3ea5bd1818a5fdcd0152edf4c07b8e405c07",
            7_569_200,
        ),
        "model-frontend": (
            "bbdf5edb120cb3df1adf9ebc07c35136539b007a7047fd148c6f2960fc56fcf1",
            8_324_600,
        ),
        "model-config": (
            "74fe5ddebd63b17caf59e8a3b18c17547ff7bce1642050edbb1c3962674f8950",
            509,
        ),
        "model-tokenizer": (
            "6884b35fd6377d4c4d32336a0bc152f36b64d1e45b6503683cdc238250a8472d",
            249_974,
        ),
    },
    "small-streaming": {
        "model-adapter": (
            "d8493e0ac76a198b309a8be6f74b3101e235f773ffe5d6b378278cd7e4177992",
            2_867_424,
        ),
        "model-cross-kv": (
            "6e57d1361717e00d73336a0c3beafedae784b1e537905ad253dee33db4007466",
            5_298_736,
        ),
        "model-decoder-kv": (
            "d5adfcfaa6e582144791f1568bd0f683852c7bfbb8c79acad97499da05e4ffcf",
            81_435_904,
        ),
        "model-encoder": (
            "3b21d02eff6aa5651524ada4271d37c1d7bba4eb3d256415074f2cfdbaeb526a",
            43_853_224,
        ),
        "model-frontend": (
            "e086451043c1c8652a9614e4a4a81d5807221b611584a3cf31f73779d5900003",
            30_984_200,
        ),
        "model-config": (
            "26f02b6afb22d60871a5efd85c3d38e569cc0ddb6c5eb6e93d3260152ae8a47a",
            512,
        ),
        "model-tokenizer": (
            "6884b35fd6377d4c4d32336a0bc152f36b64d1e45b6503683cdc238250a8472d",
            249_974,
        ),
    },
}
NEMOTRON_ARTIFACT_NAMES = (
    "runtime-receipt",
    "runtime-wheel-lock",
    "venv-marker",
    "model-config",
    "model-generation-config",
    "model-weights",
    "model-processor-config",
    "model-tokenizer",
    "model-tokenizer-config",
)
_NEMOTRON_ARTIFACT_BASENAMES = {
    "runtime-receipt": "runtime-receipt.json",
    "runtime-wheel-lock": "nemotron-runtime-wheels.lock.json",
    "venv-marker": "pyvenv.cfg",
    "model-config": "config.json",
    "model-generation-config": "generation_config.json",
    "model-weights": "model.safetensors",
    "model-processor-config": "processor_config.json",
    "model-tokenizer": "tokenizer.json",
    "model-tokenizer-config": "tokenizer_config.json",
}
_NEMOTRON_SMALL_ARTIFACTS = {
    "runtime-receipt": 32 * 1024 * 1024,
    "runtime-wheel-lock": 256 * 1024,
    "venv-marker": 64 * 1024,
    "model-config": 4 * 1024 * 1024,
    "model-generation-config": 4 * 1024 * 1024,
    "model-processor-config": 4 * 1024 * 1024,
    "model-tokenizer": 16 * 1024 * 1024,
    "model-tokenizer-config": 4 * 1024 * 1024,
}
_NEMOTRON_CONTROL_RECEIPTS = {
    "runtime-wheel-lock": (NEMOTRON_WHEEL_LOCK_SHA256, 32_099),
}
_NEMOTRON_MODEL_RECEIPTS = {
    "model-config": (
        "62d186fd91f518e00e7867500f1f5819225e8ee95ea3e21b546514bf2048e845",
        1_376,
    ),
    "model-generation-config": (
        "993e5d4cb74a6fe9d6e7084a76b3313c1446740679be4676570c23b664fdc07e",
        193,
    ),
    "model-weights": (
        "9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174",
        2_552_062_944,
    ),
    "model-processor-config": (
        "ec47870f1091ea4f25539208387b45b902c92d0e3f997a30061ef88f73437ab0",
        2_519,
    ),
    "model-tokenizer": (
        "3f3d481deb073b64c2082e8c7860d487a3a62774bf4e9e4faac83007e181f246",
        752_051,
    ),
    "model-tokenizer-config": (
        "5c641c5b3f50702a60082690d27c1ce7fcb5a92c4a624793bcae0f21eda3d6e0",
        881,
    ),
}
_NEMOTRON_STREAM_GEOMETRY = {
    0: {
        "native_first_chunk_frames": 1,
        "native_chunk_frames": 8,
        "native_first_window_samples": 200,
        "native_window_samples": 1_680,
        "native_stride_samples": 1_280,
        "streaming_latency_ms": 80,
    },
    3: {
        "native_first_chunk_frames": 25,
        "native_chunk_frames": 32,
        "native_first_window_samples": 4_040,
        "native_window_samples": 5_520,
        "native_stride_samples": 5_120,
        "streaming_latency_ms": 320,
    },
    6: {
        "native_first_chunk_frames": 49,
        "native_chunk_frames": 56,
        "native_first_window_samples": 7_880,
        "native_window_samples": 9_360,
        "native_stride_samples": 8_960,
        "streaming_latency_ms": 560,
    },
    13: {
        "native_first_chunk_frames": 105,
        "native_chunk_frames": 112,
        "native_first_window_samples": 16_840,
        "native_window_samples": 18_320,
        "native_stride_samples": 17_920,
        "streaming_latency_ms": 1_120,
    },
}


class ManifestError(RuntimeError):
    """A detail-free manifest or local-artifact validation failure."""


@dataclass(frozen=True)
class BoundFile:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class BoundArtifact(BoundFile):
    name: str


@dataclass(frozen=True)
class WorkerLimits:
    startup_timeout_sec: float
    case_timeout_sec: float


@dataclass(frozen=True)
class MoonshineConfig:
    package_version: str = "0.1.0"
    api_version: int = 30000
    model_arch: str = "tiny-streaming"
    provider: str = "cpu"
    language: str = "en"

    def __post_init__(self) -> None:
        if (
            self.package_version != "0.1.0"
            or type(self.api_version) is not int
            or self.api_version != 30000
            or self.model_arch not in {"tiny-streaming", "small-streaming"}
            or self.provider != "cpu"
            or self.language != "en"
        ):
            raise ManifestError()

    def as_dict(self) -> dict[str, object]:
        return {
            "package_version": self.package_version,
            "api_version": self.api_version,
            "model_arch": self.model_arch,
            "provider": self.provider,
            "language": self.language,
        }


@dataclass(frozen=True)
class NemotronConfig:
    python_version: str = "3.12.3"
    transformers_version: str = "5.13.1"
    librosa_version: str = "0.11.0"
    torch_version: str = "2.12.1+cu126"
    cuda_version: str = "12.6"
    model_revision: str = "f3d333391852ba876df169dcc9ba902d25b6ab0b"
    lookahead_tokens: int = 3
    language: str = "en-US"
    device: str = "cuda:0"
    dtype: str = "float32"
    native_sample_rate_hz: int | None = None
    native_hop_length_samples: int | None = None
    native_n_fft_samples: int | None = None
    native_win_length_samples: int | None = None
    native_first_chunk_frames: int | None = None
    native_chunk_frames: int | None = None
    native_first_window_samples: int | None = None
    native_window_samples: int | None = None
    native_stride_samples: int | None = None
    streaming_latency_ms: int | None = None
    wheel_lock_sha256: str = NEMOTRON_WHEEL_LOCK_SHA256
    runtime_content_sha256: str = NEMOTRON_RUNTIME_CONTENT_SHA256
    runtime_file_count: int = NEMOTRON_RUNTIME_FILE_COUNT
    runtime_total_size_bytes: int = NEMOTRON_RUNTIME_TOTAL_SIZE_BYTES
    runtime_maximum_file_bytes: int = NEMOTRON_RUNTIME_MAXIMUM_FILE_BYTES

    def __post_init__(self) -> None:
        if (
            self.python_version != "3.12.3"
            or self.transformers_version != "5.13.1"
            or self.librosa_version != "0.11.0"
            or self.torch_version != "2.12.1+cu126"
            or self.cuda_version != "12.6"
            or self.model_revision != "f3d333391852ba876df169dcc9ba902d25b6ab0b"
            or type(self.lookahead_tokens) is not int
            or self.lookahead_tokens not in {0, 3, 6, 13}
            or self.language not in {"en-US", "en-GB", "ro-RO"}
            or self.device != "cuda:0"
            or self.dtype != "float32"
            or self.wheel_lock_sha256 != NEMOTRON_WHEEL_LOCK_SHA256
            or self.runtime_content_sha256 != NEMOTRON_RUNTIME_CONTENT_SHA256
            or self.runtime_file_count != NEMOTRON_RUNTIME_FILE_COUNT
            or self.runtime_total_size_bytes != NEMOTRON_RUNTIME_TOTAL_SIZE_BYTES
            or self.runtime_maximum_file_bytes != NEMOTRON_RUNTIME_MAXIMUM_FILE_BYTES
        ):
            raise ManifestError()
        expected_geometry = {
            "native_sample_rate_hz": 16_000,
            "native_hop_length_samples": 160,
            "native_n_fft_samples": 512,
            "native_win_length_samples": 400,
            **_NEMOTRON_STREAM_GEOMETRY[self.lookahead_tokens],
        }
        for field_name, expected in expected_geometry.items():
            actual = getattr(self, field_name)
            if actual is None:
                object.__setattr__(self, field_name, expected)
            elif type(actual) is not int or actual != expected:
                raise ManifestError()

    def as_dict(self) -> dict[str, object]:
        return {
            "python_version": self.python_version,
            "transformers_version": self.transformers_version,
            "librosa_version": self.librosa_version,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "model_revision": self.model_revision,
            "lookahead_tokens": self.lookahead_tokens,
            "language": self.language,
            "device": self.device,
            "dtype": self.dtype,
            "native_sample_rate_hz": self.native_sample_rate_hz,
            "native_hop_length_samples": self.native_hop_length_samples,
            "native_n_fft_samples": self.native_n_fft_samples,
            "native_win_length_samples": self.native_win_length_samples,
            "native_first_chunk_frames": self.native_first_chunk_frames,
            "native_chunk_frames": self.native_chunk_frames,
            "native_first_window_samples": self.native_first_window_samples,
            "native_window_samples": self.native_window_samples,
            "native_stride_samples": self.native_stride_samples,
            "streaming_latency_ms": self.streaming_latency_ms,
            "wheel_lock_sha256": self.wheel_lock_sha256,
            "runtime_content_sha256": self.runtime_content_sha256,
            "runtime_file_count": self.runtime_file_count,
            "runtime_total_size_bytes": self.runtime_total_size_bytes,
            "runtime_maximum_file_bytes": self.runtime_maximum_file_bytes,
        }


@dataclass(frozen=True)
class WorkerManifest:
    path: Path
    digest: str
    schema_version: int
    model_id: str
    adapter: str
    python: BoundFile
    worker: BoundFile
    artifacts: tuple[BoundArtifact, ...]
    limits: WorkerLimits
    adapter_config: MoonshineConfig | NemotronConfig | None = None

    @property
    def artifact_by_name(self) -> Mapping[str, BoundArtifact]:
        return {artifact.name: artifact for artifact in self.artifacts}


def _bad() -> object:
    raise ManifestError()


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ManifestError()
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: _bad())
    except (UnicodeError, ValueError, OverflowError, ManifestError):
        raise ManifestError() from None


def _safe_id(value: object) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise ManifestError()
    return value


def _sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ManifestError()
    return value


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestError()
    return value


def _bounded_seconds(value: object, *, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError()
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ManifestError() from None
    if not math.isfinite(number) or number <= 0.0 or number > maximum:
        raise ManifestError()
    return number


def artifact_maximum_bytes(adapter: str, artifact_name: str) -> int:
    """Return the closed per-artifact read budget for one adapter."""

    if adapter == "fake-json-v1" and artifact_name == "fake-script":
        return MAX_FAKE_ARTIFACT_BYTES
    if adapter == MOONSHINE_ADAPTER and artifact_name in MOONSHINE_ARTIFACT_NAMES:
        return _MOONSHINE_SMALL_ARTIFACTS.get(
            artifact_name,
            MAX_MOONSHINE_ARTIFACT_BYTES,
        )
    if adapter == NEMOTRON_ADAPTER and artifact_name in NEMOTRON_ARTIFACT_NAMES:
        return _NEMOTRON_SMALL_ARTIFACTS.get(
            artifact_name,
            MAX_NEMOTRON_ARTIFACT_BYTES,
        )
    raise ManifestError()


def _bound_file(
    value: object,
    *,
    expected_fields: set[str],
    allow_symlink: bool,
    maximum_bytes: int,
    retain_lexical_path: bool = False,
) -> BoundFile:
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ManifestError()
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise ManifestError()
    candidate = Path(raw_path)
    if not candidate.is_absolute() or (candidate.is_symlink() and not allow_symlink):
        raise ManifestError()
    size = _positive_int(value.get("size_bytes"))
    expected_hash = _sha256(value.get("sha256"))
    try:
        digest = hash_regular_bounded(
            candidate,
            maximum_bytes=maximum_bytes,
            expected_bytes=size,
            allow_final_symlink=allow_symlink,
        )
    except BoundedReadError:
        raise ManifestError() from None
    if digest.sha256 != expected_hash:
        raise ManifestError()
    return BoundFile(
        path=candidate if retain_lexical_path else digest.path,
        sha256=expected_hash,
        size_bytes=size,
    )


def _artifact(value: object, *, adapter: str) -> BoundArtifact:
    if not isinstance(value, dict):
        raise ManifestError()
    name = _safe_id(value.get("name"))
    bound = _bound_file(
        value,
        expected_fields=_ARTIFACT_FIELDS,
        allow_symlink=False,
        maximum_bytes=artifact_maximum_bytes(adapter, name),
    )
    return BoundArtifact(
        path=bound.path,
        sha256=bound.sha256,
        size_bytes=bound.size_bytes,
        name=name,
    )


def _moonshine_config(value: object) -> MoonshineConfig:
    if not isinstance(value, dict) or set(value) != _MOONSHINE_CONFIG_FIELDS:
        raise ManifestError()
    try:
        return MoonshineConfig(
            package_version=value.get("package_version"),  # type: ignore[arg-type]
            api_version=value.get("api_version"),  # type: ignore[arg-type]
            model_arch=value.get("model_arch"),  # type: ignore[arg-type]
            provider=value.get("provider"),  # type: ignore[arg-type]
            language=value.get("language"),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError, ManifestError):
        raise ManifestError() from None


def _nemotron_config(value: object) -> NemotronConfig:
    if not isinstance(value, dict) or set(value) != _NEMOTRON_CONFIG_FIELDS:
        raise ManifestError()
    try:
        return NemotronConfig(
            python_version=value.get("python_version"),  # type: ignore[arg-type]
            transformers_version=value.get("transformers_version"),  # type: ignore[arg-type]
            librosa_version=value.get("librosa_version"),  # type: ignore[arg-type]
            torch_version=value.get("torch_version"),  # type: ignore[arg-type]
            cuda_version=value.get("cuda_version"),  # type: ignore[arg-type]
            model_revision=value.get("model_revision"),  # type: ignore[arg-type]
            lookahead_tokens=value.get("lookahead_tokens"),  # type: ignore[arg-type]
            language=value.get("language"),  # type: ignore[arg-type]
            device=value.get("device"),  # type: ignore[arg-type]
            dtype=value.get("dtype"),  # type: ignore[arg-type]
            native_sample_rate_hz=value.get("native_sample_rate_hz"),  # type: ignore[arg-type]
            native_hop_length_samples=value.get("native_hop_length_samples"),  # type: ignore[arg-type]
            native_n_fft_samples=value.get("native_n_fft_samples"),  # type: ignore[arg-type]
            native_win_length_samples=value.get("native_win_length_samples"),  # type: ignore[arg-type]
            native_first_chunk_frames=value.get("native_first_chunk_frames"),  # type: ignore[arg-type]
            native_chunk_frames=value.get("native_chunk_frames"),  # type: ignore[arg-type]
            native_first_window_samples=value.get("native_first_window_samples"),  # type: ignore[arg-type]
            native_window_samples=value.get("native_window_samples"),  # type: ignore[arg-type]
            native_stride_samples=value.get("native_stride_samples"),  # type: ignore[arg-type]
            streaming_latency_ms=value.get("streaming_latency_ms"),  # type: ignore[arg-type]
            wheel_lock_sha256=value.get("wheel_lock_sha256"),  # type: ignore[arg-type]
            runtime_content_sha256=value.get("runtime_content_sha256"),  # type: ignore[arg-type]
            runtime_file_count=value.get("runtime_file_count"),  # type: ignore[arg-type]
            runtime_total_size_bytes=value.get("runtime_total_size_bytes"),  # type: ignore[arg-type]
            runtime_maximum_file_bytes=value.get("runtime_maximum_file_bytes"),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError, ManifestError):
        raise ManifestError() from None


def _validate_venv_layout(
    python: BoundFile,
    marker: BoundArtifact,
) -> None:
    lexical_python = Path(os.path.abspath(python.path))
    venv_root = lexical_python.parent.parent
    if (
        lexical_python != python.path
        or lexical_python.parent.name != "bin"
        or marker.path.parent != venv_root
        or venv_root.resolve(strict=True) != venv_root
    ):
        raise ManifestError()
    try:
        verify_isolated_venv_marker(
            marker.path,
            expected_digest=marker.sha256,
            expected_bytes=marker.size_bytes,
        )
    except RuntimeTreeReceiptError:
        raise ManifestError() from None


def _validate_moonshine_layout(
    python: BoundFile,
    artifacts: tuple[BoundArtifact, ...],
    config: MoonshineConfig,
) -> None:
    by_name = {artifact.name: artifact for artifact in artifacts}
    model_receipts = _MOONSHINE_MODEL_RECEIPTS.get(config.model_arch)
    if (
        tuple(by_name) != MOONSHINE_ARTIFACT_NAMES
        or model_receipts is None
        or sum(artifact.size_bytes for artifact in artifacts)
        > MAX_MOONSHINE_TOTAL_ARTIFACT_BYTES
        or any(
            by_name[name].path.name != basename
            for name, basename in _MOONSHINE_ARTIFACT_BASENAMES.items()
        )
        or (
            by_name["release-wheel"].sha256,
            by_name["release-wheel"].size_bytes,
        )
        != _MOONSHINE_RELEASE_RECEIPT
        or any(
            (by_name[name].sha256, by_name[name].size_bytes) != receipt
            for name, receipt in model_receipts.items()
        )
    ):
        raise ManifestError()

    model_names = tuple(
        name for name in MOONSHINE_ARTIFACT_NAMES if name.startswith("model-")
    )
    model_parents = {by_name[name].path.parent for name in model_names}
    if len(model_parents) != 1:
        raise ManifestError()
    _validate_venv_layout(python, by_name["venv-marker"])


def _validate_nemotron_layout(
    python: BoundFile,
    artifacts: tuple[BoundArtifact, ...],
) -> None:
    by_name = {artifact.name: artifact for artifact in artifacts}
    if (
        tuple(by_name) != NEMOTRON_ARTIFACT_NAMES
        or sum(artifact.size_bytes for artifact in artifacts)
        > MAX_NEMOTRON_TOTAL_ARTIFACT_BYTES
        or any(
            by_name[name].path.name != basename
            for name, basename in _NEMOTRON_ARTIFACT_BASENAMES.items()
        )
        or any(
            (by_name[name].sha256, by_name[name].size_bytes) != receipt
            for name, receipt in _NEMOTRON_MODEL_RECEIPTS.items()
        )
        or any(
            (by_name[name].sha256, by_name[name].size_bytes) != receipt
            for name, receipt in _NEMOTRON_CONTROL_RECEIPTS.items()
        )
    ):
        raise ManifestError()
    model_names = tuple(
        name for name in NEMOTRON_ARTIFACT_NAMES if name.startswith("model-")
    )
    model_parents = {by_name[name].path.parent for name in model_names}
    if len(model_parents) != 1:
        raise ManifestError()
    _validate_venv_layout(python, by_name["venv-marker"])


def load_worker_manifest(path: Path | str) -> WorkerManifest:
    """Load and verify one immutable, machine-local worker receipt."""

    candidate = Path(path).expanduser()
    try:
        snapshot = read_regular_bounded(
            candidate,
            maximum_bytes=_MAX_MANIFEST_BYTES,
        )
    except BoundedReadError:
        raise ManifestError() from None
    resolved = snapshot.path
    raw = snapshot.data
    value = _strict_json(raw)
    if not isinstance(value, dict):
        raise ManifestError()
    schema_version = value.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2, 3}:
        raise ManifestError()
    expected_fields = {
        1: _MANIFEST_V1_FIELDS,
        2: _MANIFEST_V2_FIELDS,
        3: _MANIFEST_V3_FIELDS,
    }[schema_version]
    if set(value) != expected_fields:
        raise ManifestError()

    adapter = _safe_id(value.get("adapter"))
    if (
        (schema_version == 1 and adapter != "fake-json-v1")
        or (schema_version == 2 and adapter != MOONSHINE_ADAPTER)
        or (schema_version == 3 and adapter != NEMOTRON_ADAPTER)
    ):
        raise ManifestError()

    python = _bound_file(
        value.get("python"),
        expected_fields=_FILE_FIELDS,
        allow_symlink=True,
        maximum_bytes=MAX_PYTHON_BYTES,
        retain_lexical_path=True,
    )
    worker = _bound_file(
        value.get("worker"),
        expected_fields=_FILE_FIELDS,
        allow_symlink=False,
        maximum_bytes=MAX_WORKER_BYTES,
    )
    raw_artifacts = value.get("artifacts")
    expected_artifact_names = {
        1: ("fake-script",),
        2: MOONSHINE_ARTIFACT_NAMES,
        3: NEMOTRON_ARTIFACT_NAMES,
    }[schema_version]
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != len(
        expected_artifact_names
    ):
        raise ManifestError()
    artifacts = tuple(_artifact(item, adapter=adapter) for item in raw_artifacts)
    artifact_names = [artifact.name for artifact in artifacts]
    if artifact_names != list(expected_artifact_names):
        raise ManifestError()
    adapter_config = {
        1: lambda: None,
        2: lambda: _moonshine_config(value.get("adapter_config")),
        3: lambda: _nemotron_config(value.get("adapter_config")),
    }[schema_version]()
    if schema_version == 2:
        assert isinstance(adapter_config, MoonshineConfig)
        _validate_moonshine_layout(python, artifacts, adapter_config)
    elif schema_version == 3:
        assert isinstance(adapter_config, NemotronConfig)
        _validate_nemotron_layout(python, artifacts)

    raw_limits = value.get("limits")
    if not isinstance(raw_limits, dict) or set(raw_limits) != _LIMIT_FIELDS:
        raise ManifestError()
    limits = WorkerLimits(
        startup_timeout_sec=_bounded_seconds(
            raw_limits.get("startup_timeout_sec"),
            maximum=600.0,
        ),
        case_timeout_sec=_bounded_seconds(
            raw_limits.get("case_timeout_sec"),
            maximum=3600.0,
        ),
    )
    return WorkerManifest(
        path=resolved,
        digest=hashlib.sha256(raw).hexdigest(),
        schema_version=schema_version,
        model_id=_safe_id(value.get("model_id")),
        adapter=adapter,
        python=python,
        worker=worker,
        artifacts=artifacts,
        limits=limits,
        adapter_config=adapter_config,
    )
