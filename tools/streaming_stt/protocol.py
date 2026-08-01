"""Versioned and bounded JSONL messages for isolated streaming STT."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


# Version 2 remains the wire contract for the existing fake, Moonshine,
# Nemotron, and Zipformer workers. Version 3 is opt-in for recognizers that can
# finish on a native endpoint before consuming the request's declared tail.
PROTOCOL_VERSION = 2
NATIVE_ENDPOINT_PROTOCOL_VERSION = 3
SUPPORTED_PROTOCOL_VERSIONS = frozenset(
    {PROTOCOL_VERSION, NATIVE_ENDPOINT_PROTOCOL_VERSION}
)
MAX_LINE_BYTES = 64 * 1024
MAX_HYPOTHESIS_CHARS = 4096
MAX_PARTIALS_PER_CASE = 256
MAX_PCM_BYTES = 8 * 1024 * 1024
MAX_CORPUS_BYTES = 32 * 1024 * 1024
MAX_STDOUT_BYTES = 4 * 1024 * 1024
MAX_STDERR_BYTES = 256 * 1024
# Protocol v2 keeps its original one-second tail and total-stream bounds.
# Native-endpoint protocol v3 may offer up to three seconds so the model can
# emit its own EOU/EOB marker without weakening any legacy worker contract.
MAX_TAIL_PADDING_SAMPLES = 16_000
MAX_NATIVE_TAIL_PADDING_SAMPLES = 48_000
MAX_MODEL_PADDING_SAMPLES = 32_000
MAX_STREAM_CHUNK_SAMPLES = 32_000
MAX_LEGACY_STREAM_SAMPLES = MAX_PCM_BYTES // 4 + MAX_TAIL_PADDING_SAMPLES
MAX_NATIVE_STREAM_SAMPLES = MAX_PCM_BYTES // 4 + MAX_NATIVE_TAIL_PADDING_SAMPLES
# Adapter-internal callers predate protocol v3 and use this as the widest
# process-local allocation guard. Wire parsing below selects the exact bound
# for each protocol version.
MAX_STREAM_SAMPLES = MAX_NATIVE_STREAM_SAMPLES
MAX_AUDIO_SECONDS = (MAX_PCM_BYTES // 4) / 16_000
MAX_MODEL_LOAD_MS = 10 * 60 * 1000.0
MAX_CASE_EVENT_MS = 60 * 60 * 1000.0
MAX_RESOURCE_MB = 1024 * 1024.0
# Partial and final progress share this domain: source PCM plus declared
# zero-valued tail padding. A successful final reports the exact total.
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PCM_FIELDS = {
    "path",
    "sha256",
    "encoding",
    "sample_rate",
    "channels",
    "samples",
}
_STREAM_FIELDS = {
    "chunk_samples",
    "pace",
    "partial_interval_ms",
    "tail_padding_samples",
}
_RESOURCE_FIELDS = {"rss_mb", "threads", "vram_mb"}
_RUNTIME_FIELDS = {"python", "platform"}
_NATIVE_ENDPOINT_REASONS = {"eou", "eob", "tail_exhausted"}


class ProtocolError(RuntimeError):
    """A detail-free wire contract failure."""


@dataclass(frozen=True)
class PcmInput:
    path: Path = field(repr=False)
    sha256: str
    samples: int
    sample_rate: int = 16_000
    channels: int = 1
    encoding: str = "f32le"

    @property
    def size_bytes(self) -> int:
        return self.samples * 4

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "encoding": self.encoding,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "samples": self.samples,
        }


@dataclass(frozen=True)
class StreamConfig:
    chunk_samples: int
    pace: str
    partial_interval_ms: int
    tail_padding_samples: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_samples": self.chunk_samples,
            "pace": self.pace,
            "partial_interval_ms": self.partial_interval_ms,
            "tail_padding_samples": self.tail_padding_samples,
        }


@dataclass(frozen=True)
class TranscribeRequest:
    request_id: str
    pcm: PcmInput
    stream: StreamConfig
    protocol_version: int = PROTOCOL_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "v": self.protocol_version,
            "id": self.request_id,
            "op": "transcribe",
            "pcm": self.pcm.as_dict(),
            "stream": self.stream.as_dict(),
        }


@dataclass(frozen=True)
class ShutdownRequest:
    request_id: str
    protocol_version: int = PROTOCOL_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "v": self.protocol_version,
            "id": self.request_id,
            "op": "shutdown",
        }


@dataclass(frozen=True)
class ResourceUsage:
    rss_mb: float
    threads: int
    vram_mb: float | None


@dataclass(frozen=True)
class ReadyEvent:
    model_id: str
    manifest_sha256: str
    source_bundle_sha256: str
    adapter: str
    model_load_ms: float
    resources: ResourceUsage
    runtime: Mapping[str, str]
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class PartialEvent:
    request_id: str
    seq: int
    text: str = field(repr=False)
    samples_seen: int
    elapsed_ms: float
    decode_ms: float
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class FinalEvent:
    request_id: str
    seq: int
    text: str = field(repr=False)
    samples_seen: int
    elapsed_ms: float
    finalization_ms: float
    compute_ms: float
    audio_seconds: float
    chunks: int
    deadline_misses: int
    max_backlog_ms: float
    resources: ResourceUsage
    model_padding_samples: int = 0
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class NativeFinalEvent:
    """One terminal with exact early-endpoint consumption evidence."""

    request_id: str
    seq: int
    text: str = field(repr=False)
    samples_seen: int
    elapsed_ms: float
    finalization_ms: float
    compute_ms: float
    audio_seconds: float
    chunks: int
    deadline_misses: int
    max_backlog_ms: float
    resources: ResourceUsage
    model_padding_samples: int
    source_samples: int
    source_samples_consumed: int
    declared_tail_samples: int
    tail_samples_consumed: int
    endpoint_reason: str
    native_endpoint: bool
    endpoint_probability: float | None
    endpoint_sample: int | None
    endpoint_latency_ms: float | None
    authoritative: bool
    protocol_version: int = NATIVE_ENDPOINT_PROTOCOL_VERSION


@dataclass(frozen=True)
class ErrorEvent:
    request_id: str
    code: str
    fatal: bool
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class ShutdownEvent:
    request_id: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class ByeEvent:
    protocol_version: int = PROTOCOL_VERSION


Response = (
    ReadyEvent
    | PartialEvent
    | FinalEvent
    | NativeFinalEvent
    | ErrorEvent
    | ShutdownEvent
    | ByeEvent
)


@dataclass(frozen=True)
class CaseTrace:
    partials: tuple[PartialEvent, ...] = field(repr=False)
    final: FinalEvent | NativeFinalEvent = field(repr=False)


def _bad() -> object:
    raise ProtocolError()


def strict_json_bytes(raw: bytes) -> object:
    """Parse one bounded UTF-8 JSON object with duplicate keys forbidden."""

    if not raw or len(raw) > MAX_LINE_BYTES:
        raise ProtocolError()

    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ProtocolError()
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _: _bad(),
        )
    except (UnicodeError, ValueError, OverflowError, ProtocolError):
        raise ProtocolError() from None


def encode_message(value: Mapping[str, object]) -> bytes:
    try:
        encoded = (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError):
        raise ProtocolError() from None
    if len(encoded) > MAX_LINE_BYTES:
        raise ProtocolError()
    return encoded


def _safe_id(value: object) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise ProtocolError()
    return value


def _sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProtocolError()
    return value


def _integer(
    value: object,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError()
    if maximum is not None and value > maximum:
        raise ProtocolError()
    return value


def _number(value: object, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError()
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ProtocolError() from None
    if not math.isfinite(number) or number < 0.0:
        raise ProtocolError()
    if maximum is not None and number > maximum:
        raise ProtocolError()
    return number


def _optional_number(value: object, *, maximum: float | None = None) -> float | None:
    if value is None:
        return None
    return _number(value, maximum=maximum)


def _optional_integer(
    value: object,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int | None:
    if value is None:
        return None
    return _integer(value, minimum=minimum, maximum=maximum)


def _protocol_version(value: object) -> int:
    if type(value) is not int or value not in SUPPORTED_PROTOCOL_VERSIONS:
        raise ProtocolError()
    return value


def _text(value: object) -> str:
    if not isinstance(value, str) or len(value) > MAX_HYPOTHESIS_CHARS:
        raise ProtocolError()
    return value


def _resources(value: object) -> ResourceUsage:
    if not isinstance(value, dict) or set(value) != _RESOURCE_FIELDS:
        raise ProtocolError()
    raw_vram = value.get("vram_mb")
    return ResourceUsage(
        rss_mb=_number(value.get("rss_mb"), maximum=MAX_RESOURCE_MB),
        threads=_integer(value.get("threads"), minimum=1, maximum=4096),
        vram_mb=(
            None if raw_vram is None else _number(raw_vram, maximum=MAX_RESOURCE_MB)
        ),
    )


def _pcm(value: object) -> PcmInput:
    if not isinstance(value, dict) or set(value) != _PCM_FIELDS:
        raise ProtocolError()
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise ProtocolError()
    path = Path(raw_path)
    if not path.is_absolute():
        raise ProtocolError()
    samples = _integer(
        value.get("samples"),
        minimum=1,
        maximum=MAX_PCM_BYTES // 4,
    )
    if (
        value.get("encoding") != "f32le"
        or value.get("sample_rate") != 16_000
        or type(value.get("sample_rate")) is not int
        or value.get("channels") != 1
        or type(value.get("channels")) is not int
    ):
        raise ProtocolError()
    return PcmInput(
        path=path,
        sha256=_sha256(value.get("sha256")),
        samples=samples,
    )


def _stream(value: object, *, protocol_version: int) -> StreamConfig:
    if not isinstance(value, dict) or set(value) != _STREAM_FIELDS:
        raise ProtocolError()
    pace = value.get("pace")
    if pace not in {"burst", "realtime"}:
        raise ProtocolError()
    return StreamConfig(
        chunk_samples=_integer(
            value.get("chunk_samples"),
            minimum=1,
            maximum=MAX_STREAM_CHUNK_SAMPLES,
        ),
        pace=str(pace),
        partial_interval_ms=_integer(
            value.get("partial_interval_ms"),
            minimum=1,
            maximum=5000,
        ),
        tail_padding_samples=_integer(
            value.get("tail_padding_samples"),
            maximum=(
                MAX_NATIVE_TAIL_PADDING_SAMPLES
                if protocol_version == NATIVE_ENDPOINT_PROTOCOL_VERSION
                else MAX_TAIL_PADDING_SAMPLES
            ),
        ),
    )


def parse_request(raw: bytes) -> TranscribeRequest | ShutdownRequest:
    value = strict_json_bytes(raw)
    if not isinstance(value, dict):
        raise ProtocolError()
    protocol_version = _protocol_version(value.get("v"))
    operation = value.get("op")
    if operation == "shutdown":
        if set(value) != {"v", "id", "op"}:
            raise ProtocolError()
        return ShutdownRequest(
            request_id=_safe_id(value.get("id")),
            protocol_version=protocol_version,
        )
    if operation != "transcribe" or set(value) != {"v", "id", "op", "pcm", "stream"}:
        raise ProtocolError()
    return TranscribeRequest(
        request_id=_safe_id(value.get("id")),
        pcm=_pcm(value.get("pcm")),
        stream=_stream(value.get("stream"), protocol_version=protocol_version),
        protocol_version=protocol_version,
    )


def parse_response(raw: bytes) -> Response:
    value = strict_json_bytes(raw)
    if not isinstance(value, dict):
        raise ProtocolError()
    protocol_version = _protocol_version(value.get("v"))
    maximum_stream_samples = (
        MAX_NATIVE_STREAM_SAMPLES
        if protocol_version == NATIVE_ENDPOINT_PROTOCOL_VERSION
        else MAX_LEGACY_STREAM_SAMPLES
    )
    event_type = value.get("type")
    if event_type == "ready":
        expected = {
            "v",
            "type",
            "model_id",
            "manifest_sha256",
            "source_bundle_sha256",
            "adapter",
            "model_load_ms",
            "resources",
            "runtime",
        }
        runtime = value.get("runtime")
        if (
            set(value) != expected
            or not isinstance(runtime, dict)
            or set(runtime) != _RUNTIME_FIELDS
            or any(not isinstance(item, str) or not item for item in runtime.values())
        ):
            raise ProtocolError()
        return ReadyEvent(
            model_id=_safe_id(value.get("model_id")),
            manifest_sha256=_sha256(value.get("manifest_sha256")),
            source_bundle_sha256=_sha256(value.get("source_bundle_sha256")),
            adapter=_safe_id(value.get("adapter")),
            model_load_ms=_number(
                value.get("model_load_ms"),
                maximum=MAX_MODEL_LOAD_MS,
            ),
            resources=_resources(value.get("resources")),
            runtime=dict(runtime),
            protocol_version=protocol_version,
        )
    if event_type == "partial":
        if set(value) != {
            "v",
            "id",
            "type",
            "seq",
            "text",
            "samples_seen",
            "elapsed_ms",
            "decode_ms",
        }:
            raise ProtocolError()
        return PartialEvent(
            request_id=_safe_id(value.get("id")),
            seq=_integer(value.get("seq"), maximum=MAX_PARTIALS_PER_CASE - 1),
            text=_text(value.get("text")),
            samples_seen=_integer(
                value.get("samples_seen"),
                maximum=maximum_stream_samples,
            ),
            elapsed_ms=_number(
                value.get("elapsed_ms"),
                maximum=MAX_CASE_EVENT_MS,
            ),
            decode_ms=_number(
                value.get("decode_ms"),
                maximum=MAX_CASE_EVENT_MS,
            ),
            protocol_version=protocol_version,
        )
    if event_type == "final":
        if protocol_version != PROTOCOL_VERSION:
            raise ProtocolError()
        expected_fields = {
            "v",
            "id",
            "type",
            "seq",
            "text",
            "samples_seen",
            "elapsed_ms",
            "finalization_ms",
            "compute_ms",
            "audio_seconds",
            "chunks",
            "deadline_misses",
            "max_backlog_ms",
            "model_padding_samples",
            "resources",
        }
        if set(value) != expected_fields:
            raise ProtocolError()
        return FinalEvent(
            request_id=_safe_id(value.get("id")),
            seq=_integer(value.get("seq"), maximum=MAX_PARTIALS_PER_CASE),
            text=_text(value.get("text")),
            samples_seen=_integer(
                value.get("samples_seen"),
                maximum=MAX_LEGACY_STREAM_SAMPLES,
            ),
            elapsed_ms=_number(
                value.get("elapsed_ms"),
                maximum=MAX_CASE_EVENT_MS,
            ),
            finalization_ms=_number(
                value.get("finalization_ms"),
                maximum=MAX_CASE_EVENT_MS,
            ),
            compute_ms=_number(
                value.get("compute_ms"),
                maximum=MAX_CASE_EVENT_MS,
            ),
            audio_seconds=_number(
                value.get("audio_seconds"),
                maximum=MAX_AUDIO_SECONDS,
            ),
            chunks=_integer(
                value.get("chunks"),
                minimum=1,
                maximum=MAX_LEGACY_STREAM_SAMPLES,
            ),
            deadline_misses=_integer(
                value.get("deadline_misses"),
                maximum=MAX_LEGACY_STREAM_SAMPLES,
            ),
            max_backlog_ms=_number(
                value.get("max_backlog_ms"),
                maximum=MAX_CASE_EVENT_MS,
            ),
            resources=_resources(value.get("resources")),
            model_padding_samples=_integer(
                value.get("model_padding_samples"),
                maximum=MAX_MODEL_PADDING_SAMPLES,
            ),
            protocol_version=protocol_version,
        )
    if event_type == "native_final":
        if protocol_version != NATIVE_ENDPOINT_PROTOCOL_VERSION:
            raise ProtocolError()
        expected_fields = {
            "v",
            "id",
            "type",
            "seq",
            "text",
            "samples_seen",
            "elapsed_ms",
            "finalization_ms",
            "compute_ms",
            "audio_seconds",
            "chunks",
            "deadline_misses",
            "max_backlog_ms",
            "model_padding_samples",
            "resources",
            "source_samples",
            "source_samples_consumed",
            "declared_tail_samples",
            "tail_samples_consumed",
            "endpoint_reason",
            "native_endpoint",
            "endpoint_probability",
            "endpoint_sample",
            "endpoint_latency_ms",
            "authoritative",
        }
        if set(value) != expected_fields:
            raise ProtocolError()
        endpoint_reason = value.get("endpoint_reason")
        native_endpoint = value.get("native_endpoint")
        authoritative = value.get("authoritative")
        if (
            endpoint_reason not in _NATIVE_ENDPOINT_REASONS
            or type(native_endpoint) is not bool
            or type(authoritative) is not bool
        ):
            raise ProtocolError()
        samples_seen = _integer(
            value.get("samples_seen"),
            maximum=MAX_NATIVE_STREAM_SAMPLES,
        )
        source_samples = _integer(
            value.get("source_samples"),
            minimum=1,
            maximum=MAX_PCM_BYTES // 4,
        )
        source_samples_consumed = _integer(
            value.get("source_samples_consumed"),
            maximum=MAX_PCM_BYTES // 4,
        )
        declared_tail_samples = _integer(
            value.get("declared_tail_samples"),
            maximum=MAX_NATIVE_TAIL_PADDING_SAMPLES,
        )
        tail_samples_consumed = _integer(
            value.get("tail_samples_consumed"),
            maximum=MAX_NATIVE_TAIL_PADDING_SAMPLES,
        )
        model_padding_samples = _integer(
            value.get("model_padding_samples"),
            maximum=MAX_MODEL_PADDING_SAMPLES,
        )
        endpoint_probability = _optional_number(
            value.get("endpoint_probability"), maximum=1.0
        )
        endpoint_sample = _optional_integer(
            value.get("endpoint_sample"),
            maximum=MAX_NATIVE_STREAM_SAMPLES + MAX_MODEL_PADDING_SAMPLES,
        )
        endpoint_latency_ms = _optional_number(
            value.get("endpoint_latency_ms"),
            maximum=MAX_CASE_EVENT_MS,
        )
        if (
            samples_seen <= 0
            or source_samples_consumed > source_samples
            or tail_samples_consumed > declared_tail_samples
            or (
                source_samples_consumed < source_samples
                and tail_samples_consumed != 0
            )
            or samples_seen != source_samples_consumed + tail_samples_consumed
        ):
            raise ProtocolError()
        if native_endpoint:
            source_complete = source_samples_consumed == source_samples
            expected_authoritative = endpoint_reason == "eou" and source_complete
            if (
                endpoint_reason not in {"eou", "eob"}
                or endpoint_probability is None
                or endpoint_sample != samples_seen + model_padding_samples
                or authoritative != expected_authoritative
                or (authoritative and endpoint_latency_ms is None)
                or (not authoritative and endpoint_latency_ms is not None)
            ):
                raise ProtocolError()
        elif (
            endpoint_reason != "tail_exhausted"
            or endpoint_probability is not None
            or endpoint_sample is not None
            or endpoint_latency_ms is not None
            or authoritative
            or source_samples_consumed != source_samples
            or tail_samples_consumed != declared_tail_samples
        ):
            raise ProtocolError()
        return NativeFinalEvent(
            request_id=_safe_id(value.get("id")),
            seq=_integer(value.get("seq"), maximum=MAX_PARTIALS_PER_CASE),
            text=_text(value.get("text")),
            samples_seen=samples_seen,
            elapsed_ms=_number(
                value.get("elapsed_ms"),
                maximum=MAX_CASE_EVENT_MS,
            ),
            finalization_ms=_number(
                value.get("finalization_ms"),
                maximum=MAX_CASE_EVENT_MS,
            ),
            compute_ms=_number(
                value.get("compute_ms"),
                maximum=MAX_CASE_EVENT_MS,
            ),
            audio_seconds=_number(
                value.get("audio_seconds"),
                maximum=MAX_AUDIO_SECONDS,
            ),
            chunks=_integer(
                value.get("chunks"),
                minimum=1,
                maximum=MAX_NATIVE_STREAM_SAMPLES,
            ),
            deadline_misses=_integer(
                value.get("deadline_misses"),
                maximum=MAX_NATIVE_STREAM_SAMPLES,
            ),
            max_backlog_ms=_number(
                value.get("max_backlog_ms"),
                maximum=MAX_CASE_EVENT_MS,
            ),
            resources=_resources(value.get("resources")),
            model_padding_samples=model_padding_samples,
            source_samples=source_samples,
            source_samples_consumed=source_samples_consumed,
            declared_tail_samples=declared_tail_samples,
            tail_samples_consumed=tail_samples_consumed,
            endpoint_reason=str(endpoint_reason),
            native_endpoint=bool(native_endpoint),
            endpoint_probability=endpoint_probability,
            endpoint_sample=endpoint_sample,
            endpoint_latency_ms=endpoint_latency_ms,
            authoritative=bool(authoritative),
            protocol_version=protocol_version,
        )
    if event_type == "error":
        if set(value) != {"v", "id", "type", "code", "fatal"}:
            raise ProtocolError()
        fatal = value.get("fatal")
        if type(fatal) is not bool:
            raise ProtocolError()
        return ErrorEvent(
            request_id=_safe_id(value.get("id")),
            code=_safe_id(value.get("code")),
            fatal=fatal,
            protocol_version=protocol_version,
        )
    if event_type == "shutdown":
        if set(value) != {"v", "id", "type"}:
            raise ProtocolError()
        return ShutdownEvent(
            request_id=_safe_id(value.get("id")),
            protocol_version=protocol_version,
        )
    if event_type == "bye":
        if set(value) != {"v", "type"}:
            raise ProtocolError()
        return ByeEvent(protocol_version=protocol_version)
    raise ProtocolError()
