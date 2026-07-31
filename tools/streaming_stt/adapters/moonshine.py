"""Bounded direct-stream adapter for the isolated Moonshine candidate.

The candidate package is imported only while constructing a real adapter.
Tests inject a small API-shaped object and never install or import the runtime.
"""

from __future__ import annotations

from array import array
from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import importlib
from importlib import metadata
import math
from pathlib import Path
import sys
import threading
import time
from typing import TypeVar

from ..bounded_io import BoundedReadError, read_regular_bounded
from ..manifest import MoonshineConfig
from ..protocol import (
    MAX_HYPOTHESIS_CHARS,
    MAX_PARTIALS_PER_CASE,
    MAX_PCM_BYTES,
    MAX_RESOURCE_MB,
    MAX_STREAM_SAMPLES,
    ResourceUsage,
    TranscribeRequest,
)


_EXPECTED_DISTRIBUTION = "moonshine-voice"
_SAMPLE_RATE = 16_000
_FORCE_UPDATE_FLAG = 1
_MAX_TRANSCRIPT_LINES = 1024
_MAX_LINE_ID = (1 << 64) - 1
# The harness input bound is far shorter than this, so add_audio cannot trigger
# Moonshine's convenience auto-update. The adapter owns every update explicitly.
_DISABLED_AUTO_UPDATE_SECONDS = MAX_STREAM_SAMPLES / _SAMPLE_RATE + 1.0
_ARCH_ATTRIBUTE = {
    "tiny-streaming": "TINY_STREAMING",
    "small-streaming": "SMALL_STREAMING",
}
_TRANSCRIBER_OPTIONS = {
    "return_audio_data": "false",
    "identify_speakers": "false",
    "word_timestamps": "false",
    "ort_providers": "CPU",
}
_T = TypeVar("_T")


class MoonshineAdapterError(RuntimeError):
    """A detail-free candidate load or streaming failure."""


@dataclass(frozen=True)
class MoonshineReady:
    model_load_ms: float
    resources: ResourceUsage


@dataclass(frozen=True)
class MoonshinePartial:
    """One online hypothesis; ``decode_ms`` is candidate-call wall time."""

    after_samples: int
    text: str = field(repr=False)
    elapsed_ms: float
    decode_ms: float


@dataclass(frozen=True)
class MoonshineCase:
    """One case result with candidate-call, not end-to-end worker, compute time."""

    partials: tuple[MoonshinePartial, ...] = field(repr=False)
    final: str = field(repr=False)
    elapsed_ms: float
    finalization_ms: float
    compute_ms: float
    deadline_misses: int
    max_backlog_ms: float
    resources: ResourceUsage


@dataclass(frozen=True)
class _LineState:
    line_id: int
    start_time: float
    text: str = field(repr=False)


class _HypothesisAssembler:
    """Retain revised lines without duplicating repeated transcript snapshots."""

    def __init__(self) -> None:
        self._lines: dict[tuple[int, float], _LineState] = {}

    def accept(self, transcript: object) -> str:
        raw_lines = getattr(transcript, "lines", None)
        if (
            not isinstance(raw_lines, (list, tuple))
            or len(raw_lines) > _MAX_TRANSCRIPT_LINES
        ):
            raise MoonshineAdapterError()
        seen: set[tuple[int, float]] = set()
        for raw_line in raw_lines:
            line_id = getattr(raw_line, "line_id", None)
            start_time = getattr(raw_line, "start_time", None)
            text = getattr(raw_line, "text", None)
            if (
                isinstance(line_id, bool)
                or not isinstance(line_id, int)
                or not 0 <= line_id <= _MAX_LINE_ID
                or isinstance(start_time, bool)
                or not isinstance(start_time, (int, float))
                or not isinstance(text, str)
                or len(text) > MAX_HYPOTHESIS_CHARS
            ):
                raise MoonshineAdapterError()
            start = float(start_time)
            if not math.isfinite(start) or start < 0.0:
                raise MoonshineAdapterError()
            key = (line_id, start)
            if key in seen:
                raise MoonshineAdapterError()
            seen.add(key)
            self._lines[key] = _LineState(
                line_id=line_id,
                start_time=start,
                text=text.strip(),
            )
        return self.text()

    def text(self) -> str:
        ordered = sorted(
            self._lines.values(),
            key=lambda line: (line.start_time, line.line_id),
        )
        text = "\n".join(line.text for line in ordered if line.text)
        if len(text) > MAX_HYPOTHESIS_CHARS:
            raise MoonshineAdapterError()
        return text


def _candidate_api(expected_version: str) -> object:
    """Load the exact disposable candidate only inside its worker process."""

    try:
        installed = metadata.version(_EXPECTED_DISTRIBUTION)
        if installed != expected_version:
            raise MoonshineAdapterError()
        candidate = importlib.import_module("moonshine_voice")
    except (ImportError, metadata.PackageNotFoundError, MoonshineAdapterError):
        raise MoonshineAdapterError() from None
    if getattr(candidate, "__version__", None) != expected_version:
        raise MoonshineAdapterError()
    return candidate


def _resource_usage() -> ResourceUsage:
    rss_mb: float | None = None
    threads: int | None = None
    try:
        status = Path("/proc/self/status").read_text(
            encoding="utf-8",
            errors="strict",
        )
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    rss_mb = int(fields[1]) / 1024.0
            elif line.startswith("Threads:"):
                fields = line.split()
                if len(fields) == 2:
                    threads = int(fields[1])
    except (OSError, UnicodeError, ValueError):
        pass
    if rss_mb is None:
        try:
            import resource

            maximum_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            rss_mb = maximum_rss / (1024.0 if sys.platform != "darwin" else 1024**2)
        except (ImportError, OSError, ValueError):
            rss_mb = 0.0
    if threads is None:
        threads = max(1, threading.active_count())
    if (
        not math.isfinite(rss_mb)
        or not 0.0 <= rss_mb <= MAX_RESOURCE_MB
        or not 1 <= threads <= 4096
    ):
        raise MoonshineAdapterError()
    return ResourceUsage(rss_mb=rss_mb, threads=threads, vram_mb=None)


def _read_pcm(request: TranscribeRequest) -> array[float]:
    try:
        snapshot = read_regular_bounded(
            request.pcm.path,
            maximum_bytes=MAX_PCM_BYTES,
            expected_bytes=request.pcm.size_bytes,
        )
    except BoundedReadError:
        raise MoonshineAdapterError() from None
    raw = snapshot.data
    if (
        len(raw) != request.pcm.size_bytes
        or hashlib.sha256(raw).hexdigest() != request.pcm.sha256
    ):
        raise MoonshineAdapterError()
    samples = array("f")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    if len(samples) != request.pcm.samples or any(
        not math.isfinite(sample) for sample in samples
    ):
        raise MoonshineAdapterError()
    return samples


def _audio_chunk(
    source: array[float],
    *,
    offset: int,
    count: int,
) -> list[float]:
    source_start = min(offset, len(source))
    source_end = min(offset + count, len(source))
    chunk = list(source[source_start:source_end])
    chunk.extend([0.0] * (count - len(chunk)))
    if len(chunk) != count:
        raise MoonshineAdapterError()
    return chunk


class MoonshineAdapter:
    """One persistent CPU transcriber with a fresh bounded stream per case."""

    def __init__(
        self,
        model_root: Path,
        *,
        config: MoonshineConfig,
        api: object | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(config, MoonshineConfig):
            raise MoonshineAdapterError()
        try:
            resolved_model_root = Path(model_root).resolve(strict=True)
        except (OSError, RuntimeError):
            raise MoonshineAdapterError() from None
        if not resolved_model_root.is_dir():
            raise MoonshineAdapterError()
        self.config = config
        self.model_root = resolved_model_root
        self._clock = clock
        self._sleeper = sleeper
        self._closed = False
        self._transcriber: object | None = None

        load_started = self._now()
        candidate = api if api is not None else _candidate_api(config.package_version)
        if getattr(candidate, "__version__", None) != config.package_version:
            raise MoonshineAdapterError()
        transcriber_type = getattr(candidate, "Transcriber", None)
        model_arches = getattr(candidate, "ModelArch", None)
        header_version = getattr(
            transcriber_type,
            "MOONSHINE_HEADER_VERSION",
            None,
        )
        force_update_flag = getattr(
            transcriber_type,
            "MOONSHINE_FLAG_FORCE_UPDATE",
            None,
        )
        if (
            type(header_version) is not int
            or header_version != config.api_version
            or type(force_update_flag) is not int
            or force_update_flag != _FORCE_UPDATE_FLAG
            or model_arches is None
        ):
            raise MoonshineAdapterError()
        try:
            model_arch = getattr(
                model_arches,
                _ARCH_ATTRIBUTE[config.model_arch],
            )
        except (AttributeError, KeyError):
            raise MoonshineAdapterError() from None

        try:
            self._transcriber = transcriber_type(
                model_path=str(self.model_root),
                model_arch=model_arch,
                update_interval=_DISABLED_AUTO_UPDATE_SECONDS,
                options=dict(_TRANSCRIBER_OPTIONS),
            )
            library_version = self._transcriber.get_version()
            if (
                type(library_version) is not int
                or library_version != config.api_version
            ):
                raise MoonshineAdapterError()
            load_finished = self._now()
            self.ready = MoonshineReady(
                model_load_ms=self._duration_ms(load_started, load_finished),
                resources=_resource_usage(),
            )
        except Exception as error:
            transcriber = self._transcriber
            self._transcriber = None
            if transcriber is not None:
                try:
                    transcriber.close()
                except Exception:
                    pass
            if isinstance(error, MoonshineAdapterError):
                raise
            raise MoonshineAdapterError() from None

    def __enter__(self) -> MoonshineAdapter:
        if self._closed:
            raise MoonshineAdapterError()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        transcriber = self._transcriber
        self._transcriber = None
        if transcriber is None:
            return
        try:
            transcriber.close()
        except Exception:
            raise MoonshineAdapterError() from None

    def _now(self) -> float:
        try:
            value = float(self._clock())
        except (TypeError, ValueError, OverflowError):
            raise MoonshineAdapterError() from None
        if not math.isfinite(value):
            raise MoonshineAdapterError()
        return value

    @staticmethod
    def _duration_ms(start: float, end: float) -> float:
        duration = end - start
        if not math.isfinite(duration) or duration < 0.0:
            raise MoonshineAdapterError()
        return duration * 1000.0

    def _timed(self, function: Callable[[], _T]) -> tuple[_T, float]:
        started = self._now()
        result = function()
        finished = self._now()
        return result, self._duration_ms(started, finished)

    def transcribe(
        self,
        request: TranscribeRequest,
        *,
        emit_partial: Callable[[int, MoonshinePartial], None],
    ) -> MoonshineCase:
        if self._closed or self._transcriber is None:
            raise MoonshineAdapterError()
        source = _read_pcm(request)
        total_samples = request.pcm.samples + request.stream.tail_padding_samples
        if total_samples > MAX_STREAM_SAMPLES:
            raise MoonshineAdapterError()

        case_started = self._now()
        stream: object | None = None
        try:
            stream, create_ms = self._timed(
                lambda: self._transcriber.create_stream(
                    update_interval=_DISABLED_AUTO_UPDATE_SECONDS,
                )
            )
            error_events: list[object] = []

            def listener(event: object) -> None:
                if (
                    type(event).__name__ == "Error"
                    and getattr(event, "error", None) is not None
                ):
                    error_events.append(event)

            stream.add_listener(listener)
            _, start_ms = self._timed(stream.start)
            # This intentionally sums only wall time inside candidate API
            # calls. Pacing sleeps and controller/worker overhead are not RTF.
            compute_ms = create_ms + start_ms
            stream_started = self._now()
            assembler = _HypothesisAssembler()
            partials: list[MoonshinePartial] = []
            last_online_hypothesis = ""
            partial_interval_samples = (
                request.stream.partial_interval_ms * _SAMPLE_RATE + 999
            ) // 1000
            next_partial_samples = partial_interval_samples
            samples_seen = 0
            deadline_misses = 0
            max_backlog_ms = 0.0

            def raise_on_error_event() -> None:
                if error_events:
                    raise MoonshineAdapterError()

            def accept_online(
                transcript: object,
                *,
                decode_ms: float,
            ) -> None:
                nonlocal last_online_hypothesis
                text = assembler.accept(transcript)
                if not text or text == last_online_hypothesis:
                    last_online_hypothesis = text
                    return
                last_online_hypothesis = text
                if len(partials) >= MAX_PARTIALS_PER_CASE:
                    return
                partial = MoonshinePartial(
                    after_samples=samples_seen,
                    text=text,
                    elapsed_ms=self._duration_ms(case_started, self._now()),
                    decode_ms=decode_ms,
                )
                emit_partial(len(partials), partial)
                partials.append(partial)

            for offset in range(0, total_samples, request.stream.chunk_samples):
                count = min(
                    request.stream.chunk_samples,
                    total_samples - offset,
                )
                if request.stream.pace == "realtime":
                    target = stream_started + offset / _SAMPLE_RATE
                    before_wait = self._now()
                    if before_wait < target:
                        self._sleeper(target - before_wait)
                    if self._now() < target:
                        raise MoonshineAdapterError()
                chunk = _audio_chunk(source, offset=offset, count=count)
                _, add_ms = self._timed(
                    lambda chunk=chunk: stream.add_audio(
                        chunk,
                        _SAMPLE_RATE,
                    )
                )
                compute_ms += add_ms
                samples_seen += count
                raise_on_error_event()

                if samples_seen >= next_partial_samples:
                    transcript, update_ms = self._timed(
                        lambda: stream.update_transcription(_FORCE_UPDATE_FLAG)
                    )
                    compute_ms += update_ms
                    raise_on_error_event()
                    accept_online(transcript, decode_ms=update_ms)
                    while next_partial_samples <= samples_seen:
                        next_partial_samples += partial_interval_samples

                if request.stream.pace == "realtime":
                    deadline = stream_started + samples_seen / _SAMPLE_RATE
                    backlog_ms = max(
                        0.0,
                        self._duration_ms(deadline, max(deadline, self._now())),
                    )
                    max_backlog_ms = max(max_backlog_ms, backlog_ms)
                    if backlog_ms > 0.0:
                        deadline_misses += 1

            if samples_seen != total_samples:
                raise MoonshineAdapterError()

            finalization_started = self._now()
            forced, force_ms = self._timed(
                lambda: stream.update_transcription(_FORCE_UPDATE_FLAG)
            )
            compute_ms += force_ms
            raise_on_error_event()
            assembler.accept(forced)

            stopped, stop_ms = self._timed(stream.stop)
            compute_ms += stop_ms
            raise_on_error_event()
            if stopped is None:
                raise MoonshineAdapterError()
            assembler.accept(stopped)
            finished = self._now()
            return MoonshineCase(
                partials=tuple(partials),
                final=assembler.text(),
                elapsed_ms=self._duration_ms(case_started, finished),
                finalization_ms=self._duration_ms(
                    finalization_started,
                    finished,
                ),
                compute_ms=compute_ms,
                deadline_misses=deadline_misses,
                max_backlog_ms=max_backlog_ms,
                resources=_resource_usage(),
            )
        except MoonshineAdapterError:
            raise
        except Exception:
            raise MoonshineAdapterError() from None
        finally:
            if stream is not None:
                active_exception = sys.exc_info()[0] is not None
                try:
                    stream.close()
                except Exception:
                    if not active_exception:
                        raise MoonshineAdapterError() from None


__all__ = [
    "MoonshineAdapter",
    "MoonshineAdapterError",
    "MoonshineCase",
    "MoonshineConfig",
    "MoonshinePartial",
    "MoonshineReady",
]
