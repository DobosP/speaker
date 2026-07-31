"""Deterministic JSON-backed adapter used only for harness validation."""

from __future__ import annotations

import json
import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from ..bounded_io import BoundedReadError, read_regular_bounded
from ..manifest import MAX_FAKE_ARTIFACT_BYTES
from ..protocol import (
    MAX_CASE_EVENT_MS,
    MAX_HYPOTHESIS_CHARS,
    MAX_MODEL_LOAD_MS,
    MAX_PARTIALS_PER_CASE,
    MAX_RESOURCE_MB,
    MAX_STREAM_SAMPLES,
    ResourceUsage,
    TranscribeRequest,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ROOT_FIELDS = {"schema_version", "ready", "cases"}
_READY_FIELDS = {"model_load_ms", "resources"}
_RESOURCE_FIELDS = {"rss_mb", "threads", "vram_mb"}
_CASE_FIELDS = {
    "partials",
    "final",
    "elapsed_ms",
    "finalization_ms",
    "compute_ms",
    "deadline_misses",
    "max_backlog_ms",
    "resources",
    "hang_sec",
}
_PARTIAL_FIELDS = {"after_samples", "text", "elapsed_ms", "decode_ms"}


class FakeAdapterError(RuntimeError):
    """A detail-free fake artifact or scripted-case failure."""


@dataclass(frozen=True)
class FakePartial:
    after_samples: int
    text: str = field(repr=False)
    elapsed_ms: float
    decode_ms: float


@dataclass(frozen=True)
class FakeCase:
    partials: tuple[FakePartial, ...] = field(repr=False)
    final: str = field(repr=False)
    elapsed_ms: float
    finalization_ms: float
    compute_ms: float
    deadline_misses: int
    max_backlog_ms: float
    resources: ResourceUsage
    hang_sec: float


@dataclass(frozen=True)
class FakeReady:
    model_load_ms: float
    resources: ResourceUsage


def _bad() -> object:
    raise FakeAdapterError()


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise FakeAdapterError()
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: _bad())
    except (UnicodeError, ValueError, OverflowError, FakeAdapterError):
        raise FakeAdapterError() from None


def _number(value: object, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FakeAdapterError()
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise FakeAdapterError() from None
    if not math.isfinite(number) or number < 0.0:
        raise FakeAdapterError()
    if maximum is not None and number > maximum:
        raise FakeAdapterError()
    return number


def _integer(value: object, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FakeAdapterError()
    if maximum is not None and value > maximum:
        raise FakeAdapterError()
    return value


def _text(value: object) -> str:
    if not isinstance(value, str) or len(value) > MAX_HYPOTHESIS_CHARS:
        raise FakeAdapterError()
    return value


def _resources(value: object) -> ResourceUsage:
    if not isinstance(value, dict) or set(value) != _RESOURCE_FIELDS:
        raise FakeAdapterError()
    raw_vram = value.get("vram_mb")
    return ResourceUsage(
        rss_mb=_number(value.get("rss_mb"), maximum=MAX_RESOURCE_MB),
        threads=_integer(value.get("threads"), minimum=1, maximum=4096),
        vram_mb=(
            None if raw_vram is None else _number(raw_vram, maximum=MAX_RESOURCE_MB)
        ),
    )


def _partial(value: object) -> FakePartial:
    if not isinstance(value, dict) or set(value) != _PARTIAL_FIELDS:
        raise FakeAdapterError()
    return FakePartial(
        after_samples=_integer(
            value.get("after_samples"),
            minimum=1,
            maximum=MAX_STREAM_SAMPLES,
        ),
        text=_text(value.get("text")),
        elapsed_ms=_number(
            value.get("elapsed_ms"),
            maximum=MAX_CASE_EVENT_MS,
        ),
        decode_ms=_number(
            value.get("decode_ms"),
            maximum=MAX_CASE_EVENT_MS,
        ),
    )


def _case(value: object) -> FakeCase:
    if not isinstance(value, dict) or set(value) != _CASE_FIELDS:
        raise FakeAdapterError()
    raw_partials = value.get("partials")
    if not isinstance(raw_partials, list) or len(raw_partials) > MAX_PARTIALS_PER_CASE:
        raise FakeAdapterError()
    partials = tuple(_partial(item) for item in raw_partials)
    after_samples = [partial.after_samples for partial in partials]
    if after_samples != sorted(after_samples) or len(set(after_samples)) != len(
        after_samples
    ):
        raise FakeAdapterError()
    return FakeCase(
        partials=partials,
        final=_text(value.get("final")),
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
        deadline_misses=_integer(
            value.get("deadline_misses"),
            maximum=MAX_STREAM_SAMPLES,
        ),
        max_backlog_ms=_number(
            value.get("max_backlog_ms"),
            maximum=MAX_CASE_EVENT_MS,
        ),
        resources=_resources(value.get("resources")),
        hang_sec=_number(value.get("hang_sec"), maximum=60.0),
    )


class FakeJsonAdapter:
    """Replay deterministic partial/final events keyed by exact PCM SHA-256."""

    def __init__(
        self,
        artifact_path: Path,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
    ) -> None:
        try:
            snapshot = read_regular_bounded(
                artifact_path,
                maximum_bytes=MAX_FAKE_ARTIFACT_BYTES,
                expected_bytes=expected_size_bytes,
            )
            raw = snapshot.data
        except BoundedReadError:
            raise FakeAdapterError() from None
        if (
            len(raw) != expected_size_bytes
            or hashlib.sha256(raw).hexdigest() != expected_sha256
        ):
            raise FakeAdapterError()
        value = _strict_json(raw)
        if (
            not isinstance(value, dict)
            or set(value) != _ROOT_FIELDS
            or type(value.get("schema_version")) is not int
            or value.get("schema_version") != 1
        ):
            raise FakeAdapterError()
        ready = value.get("ready")
        if not isinstance(ready, dict) or set(ready) != _READY_FIELDS:
            raise FakeAdapterError()
        self.ready = FakeReady(
            model_load_ms=_number(
                ready.get("model_load_ms"),
                maximum=MAX_MODEL_LOAD_MS,
            ),
            resources=_resources(ready.get("resources")),
        )
        raw_cases = value.get("cases")
        if not isinstance(raw_cases, dict) or not raw_cases:
            raise FakeAdapterError()
        cases: dict[str, FakeCase] = {}
        for digest, raw_case in raw_cases.items():
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise FakeAdapterError()
            cases[digest] = _case(raw_case)
        self._cases: Mapping[str, FakeCase] = cases

    def transcribe(
        self,
        request: TranscribeRequest,
        *,
        emit_partial: Callable[[int, FakePartial], None],
    ) -> FakeCase:
        case = self._cases.get(request.pcm.sha256)
        if case is None:
            raise FakeAdapterError()
        maximum_samples = request.pcm.samples + request.stream.tail_padding_samples
        if any(partial.after_samples > maximum_samples for partial in case.partials):
            raise FakeAdapterError()
        if case.hang_sec:
            time.sleep(case.hang_sec)
        for seq, partial in enumerate(case.partials):
            emit_partial(seq, partial)
        return case
