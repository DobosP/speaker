"""Exact package-source contract for a future Anyreach CPU worker.

The committed wheel lock records a resolver-complete CPython 3.12/Linux x86_64
selection.  Loading this module does not download, install, import, or execute
any candidate package.  The lock is source provenance only: it is not a
wheelhouse measurement, installed-runtime receipt, graph contract, or model
compatibility claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Final, Mapping
from urllib.parse import unquote, urlsplit

from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from tools.streaming_stt.bounded_io import BoundedReadError, read_regular_bounded


RUNTIME_SOURCE_ID: Final = "anyreach-cpu-py312-linux-x86_64-source-v1"
RUNTIME_LOCK_KIND: Final = "exact-pypi-wheel-source-lock-v1"
DEFAULT_RUNTIME_LOCK: Final = (
    Path(__file__).resolve().parent
    / "anyreach-cpu-runtime-py312-linux-x86_64-v1.lock.json"
)
RUNTIME_LOCK_RAW_SHA256: Final = (
    "9709afd8b6386707621260fce123874e0db5b645c8311656cad59bff5ab79a97"
)
RUNTIME_LOCK_SIZE_BYTES: Final = 8_185
RUNTIME_WHEEL_COUNT: Final = 20
RUNTIME_WHEEL_BYTES: Final = 46_744_727
_MAX_LOCK_BYTES: Final = 64 * 1024
_MAX_WHEEL_BYTES: Final = 256 * 1024 * 1024
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_DISTRIBUTION_RE: Final = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_LOCK_FIELDS: Final = {"schema_version", "wheels"}
_WHEEL_FIELDS: Final = {
    "distribution",
    "filename",
    "sha256",
    "size_bytes",
    "source_url",
    "version",
    "wheel_tag",
}
_TARGET_PLATFORMS: Final = (
    "manylinux_2_28_x86_64",
    "manylinux_2_27_x86_64",
    "manylinux_2_17_x86_64",
    "manylinux2014_x86_64",
    "linux_x86_64",
)
_TARGET_TAGS: Final = frozenset(
    {
        *cpython_tags((3, 12), platforms=_TARGET_PLATFORMS),
        *compatible_tags((3, 12), interpreter="cp312", platforms=_TARGET_PLATFORMS),
    }
)
_EXPECTED_VERSIONS: Final = (
    ("anyio", "4.14.2"),
    ("certifi", "2026.7.22"),
    ("click", "8.4.2"),
    ("filelock", "3.32.2"),
    ("flatbuffers", "25.12.19"),
    ("fsspec", "2026.7.0"),
    ("h11", "0.16.0"),
    ("hf-xet", "1.5.2"),
    ("httpcore", "1.0.9"),
    ("httpx", "0.28.1"),
    ("huggingface-hub", "1.26.0"),
    ("idna", "3.18"),
    ("numpy", "2.4.6"),
    ("onnxruntime", "1.28.0"),
    ("packaging", "26.2"),
    ("protobuf", "7.35.1"),
    ("pyyaml", "6.0.3"),
    ("tokenizers", "0.23.1"),
    ("tqdm", "4.70.0"),
    ("typing-extensions", "4.16.0"),
)
_ROOT_VERSIONS: Final = MappingProxyType(
    {
        "numpy": "2.4.6",
        "onnxruntime": "1.28.0",
        "tokenizers": "0.23.1",
    }
)
_TARGET: Final = MappingProxyType(
    {
        "implementation": "cpython",
        "python_version": "3.12",
        "abi": "cp312",
        "operating_system": "linux",
        "architecture": "x86_64",
        "minimum_glibc": "2.28",
        "provider": "CPUExecutionProvider",
    }
)
_EVIDENCE_SCOPE: Final = MappingProxyType(
    {
        "package_sources_bound": True,
        "dependency_resolution_declared": True,
        "wheel_bytes_acquired": False,
        "wheelhouse_verified": False,
        "installed_runtime_verified": False,
        "package_imports_executed": False,
        "tokenizer_executed": False,
        "model_executed": False,
        "graph_contract_verified": False,
        "runtime_compatible_with_model": False,
        "production_evidence": False,
        "redistribution_authority": False,
    }
)


class AnyreachRuntimeContractError(RuntimeError):
    """A runtime source lock failed closed without exposing details."""


@dataclass(frozen=True)
class RuntimeWheelSource:
    distribution: str
    version: str
    filename: str
    wheel_tag: str
    source_url: str = field(repr=False)
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class AnyreachRuntimeSourceLock:
    path: Path = field(repr=False)
    digest: str
    runtime_source_id: str
    target: Mapping[str, str]
    root_versions: Mapping[str, str]
    wheels: tuple[RuntimeWheelSource, ...]
    total_size_bytes: int
    evidence_scope: Mapping[str, bool] = field(repr=False)

    @property
    def package_versions(self) -> Mapping[str, str]:
        return MappingProxyType(
            {item.distribution: item.version for item in self.wheels}
        )


def _bad() -> object:
    raise AnyreachRuntimeContractError()


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise AnyreachRuntimeContractError()
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: _bad(),
        )
    except AnyreachRuntimeContractError:
        raise
    except (OverflowError, RecursionError, UnicodeError, ValueError):
        raise AnyreachRuntimeContractError() from None


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (OverflowError, TypeError, UnicodeError, ValueError):
        raise AnyreachRuntimeContractError() from None


def _wheel_tag(filename: str) -> str:
    try:
        stem, extension = filename.rsplit(".", 1)
        _prefix, python_tag, abi_tag, platform_tag = stem.rsplit("-", 3)
    except ValueError:
        raise AnyreachRuntimeContractError() from None
    if extension != "whl":
        raise AnyreachRuntimeContractError()
    return f"{python_tag}-{abi_tag}-{platform_tag}"


def _source_url(value: object, *, filename: str) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise AnyreachRuntimeContractError()
    try:
        parsed = urlsplit(value)
        basename = unquote(PurePosixPath(parsed.path).name)
        port = parsed.port
    except (UnicodeError, ValueError):
        raise AnyreachRuntimeContractError() from None
    if (
        parsed.scheme != "https"
        or parsed.hostname != "files.pythonhosted.org"
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.query
        or parsed.fragment
        or basename != filename
    ):
        raise AnyreachRuntimeContractError()
    return value


def _wheel(value: object) -> RuntimeWheelSource:
    if not isinstance(value, dict) or set(value) != _WHEEL_FIELDS:
        raise AnyreachRuntimeContractError()
    distribution = value.get("distribution")
    version = value.get("version")
    filename = value.get("filename")
    wheel_tag = value.get("wheel_tag")
    size_bytes = value.get("size_bytes")
    sha256 = value.get("sha256")
    if (
        type(distribution) is not str
        or _DISTRIBUTION_RE.fullmatch(distribution) is None
        or canonicalize_name(distribution) != distribution
        or type(version) is not str
        or not version
        or type(filename) is not str
        or not filename.isascii()
        or PurePosixPath(filename).name != filename
        or type(wheel_tag) is not str
        or type(size_bytes) is not int
        or not 0 < size_bytes <= _MAX_WHEEL_BYTES
        or type(sha256) is not str
        or _SHA256_RE.fullmatch(sha256) is None
    ):
        raise AnyreachRuntimeContractError()
    try:
        parsed_name, parsed_version, _build, parsed_tags = parse_wheel_filename(
            filename
        )
        normalized_version = str(Version(version))
    except (InvalidVersion, ValueError):
        raise AnyreachRuntimeContractError() from None
    if (
        str(parsed_name) != distribution
        or str(parsed_version) != version
        or normalized_version != version
        or wheel_tag != _wheel_tag(filename)
        or not (parsed_tags & _TARGET_TAGS)
    ):
        raise AnyreachRuntimeContractError()
    return RuntimeWheelSource(
        distribution=distribution,
        version=version,
        filename=filename,
        wheel_tag=wheel_tag,
        source_url=_source_url(value.get("source_url"), filename=filename),
        size_bytes=size_bytes,
        sha256=sha256,
    )


def load_runtime_source_lock(
    path: Path | str = DEFAULT_RUNTIME_LOCK,
) -> AnyreachRuntimeSourceLock:
    """Load the exact public package-source lock without touching wheel bytes."""

    try:
        snapshot = read_regular_bounded(
            path,
            maximum_bytes=_MAX_LOCK_BYTES,
            expected_bytes=RUNTIME_LOCK_SIZE_BYTES,
        )
        raw = snapshot.data
        if hashlib.sha256(
            raw
        ).hexdigest() != RUNTIME_LOCK_RAW_SHA256 or raw != _canonical_json(
            _strict_json(raw)
        ):
            raise AnyreachRuntimeContractError()
        value = _strict_json(raw)
        if (
            not isinstance(value, dict)
            or set(value) != _LOCK_FIELDS
            or type(value.get("schema_version")) is not int
            or value.get("schema_version") != 1
            or not isinstance(value.get("wheels"), list)
        ):
            raise AnyreachRuntimeContractError()
        wheels = tuple(_wheel(item) for item in value["wheels"])
        identities = tuple(
            (item.distribution, item.version, item.filename) for item in wheels
        )
        filenames = tuple(item.filename for item in wheels)
        sources = tuple(item.source_url for item in wheels)
        if (
            len(wheels) != RUNTIME_WHEEL_COUNT
            or identities != tuple(sorted(identities))
            or tuple((item.distribution, item.version) for item in wheels)
            != _EXPECTED_VERSIONS
            or len(filenames) != len(set(filenames))
            or len(sources) != len(set(sources))
            or sum(item.size_bytes for item in wheels) != RUNTIME_WHEEL_BYTES
        ):
            raise AnyreachRuntimeContractError()
        return AnyreachRuntimeSourceLock(
            path=snapshot.path,
            digest=RUNTIME_LOCK_RAW_SHA256,
            runtime_source_id=RUNTIME_SOURCE_ID,
            target=_TARGET,
            root_versions=_ROOT_VERSIONS,
            wheels=wheels,
            total_size_bytes=RUNTIME_WHEEL_BYTES,
            evidence_scope=_EVIDENCE_SCOPE,
        )
    except AnyreachRuntimeContractError:
        raise
    except (BoundedReadError, OSError, RuntimeError, TypeError, ValueError):
        raise AnyreachRuntimeContractError() from None


__all__ = [
    "DEFAULT_RUNTIME_LOCK",
    "RUNTIME_LOCK_KIND",
    "RUNTIME_LOCK_RAW_SHA256",
    "RUNTIME_LOCK_SIZE_BYTES",
    "RUNTIME_SOURCE_ID",
    "RUNTIME_WHEEL_BYTES",
    "RUNTIME_WHEEL_COUNT",
    "AnyreachRuntimeContractError",
    "AnyreachRuntimeSourceLock",
    "RuntimeWheelSource",
    "load_runtime_source_lock",
]
