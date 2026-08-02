"""Receipt-bound parakeet.cpp adapter for the isolated streaming-STT worker.

The native bridge is intentionally process-local and detail-free.  This module
owns strict PCM, pacing, JSON, event, artifact, and transcript-privacy checks;
it does not make parakeet.cpp a runtime default or a command authority.
"""

from __future__ import annotations

from array import array
from collections.abc import Callable, Iterator
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass, field
import hashlib
import math
import os
from pathlib import Path
import re
import stat
import sys
import threading
import time
from typing import TypeVar

from ..bounded_io import BoundedReadError, hash_regular_bounded, read_regular_bounded
from ..manifest import (
    MAX_PARAKEET_CPP_LIBRARY_BYTES,
    MAX_PARAKEET_CPP_MODEL_BYTES,
    PARAKEET_CPP_ADAPTER,
    ParakeetCppConfig,
)
from ..protocol import (
    MAX_CASE_EVENT_MS,
    MAX_HYPOTHESIS_CHARS,
    MAX_LINE_BYTES,
    MAX_PARAKEET_CPP_OBSERVED_EVENTS,
    MAX_PARTIALS_PER_CASE,
    MAX_PCM_BYTES,
    MAX_RESOURCE_MB,
    MAX_STREAM_SAMPLES,
    PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION,
    ParakeetCppObservedEvent,
    ResourceUsage,
    TranscribeRequest,
    strict_json_bytes,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DEVICE_RE = re.compile(r"[A-Za-z0-9_.-]{1,63}\Z")
_ROOT_FIELDS = {"text", "eou", "eob", "frame_sec", "events", "words"}
_EVENT_FIELDS = {"type", "frame", "t"}
_WORD_FIELDS = {"w", "start", "end", "conf"}
_MAX_EVENTS_PER_DOCUMENT = 32
_MAX_WORDS_PER_DOCUMENT = 512
_MAX_WORDS_PER_CASE = 4096
_MAX_DECODED_TEXT_BYTES = MAX_HYPOTHESIS_CHARS
_MAX_DEVICE_BYTES = 64
_LIBPARAKEET_BASENAME = "libparakeet.so"
_BRIDGE_BASENAME = "libspeaker_parakeet_bridge.so"
_T = TypeVar("_T")


class ParakeetCppAdapterError(RuntimeError):
    """A detail-free artifact, native lifecycle, PCM, or event failure."""


@dataclass(frozen=True)
class ParakeetCppReady:
    model_load_ms: float
    resources: ResourceUsage
    requested_device: str
    actual_device: str
    num_threads: int
    bridge_abi_version: int
    c_api_version: int


@dataclass(frozen=True)
class ParakeetCppPartial:
    after_samples: int
    text: str = field(repr=False)
    elapsed_ms: float
    decode_ms: float


@dataclass(frozen=True)
class ParakeetCppCase:
    partials: tuple[ParakeetCppPartial, ...] = field(repr=False)
    final: str = field(repr=False)
    elapsed_ms: float
    finalization_ms: float
    compute_ms: float
    deadline_misses: int
    max_backlog_ms: float
    resources: ResourceUsage
    model_padding_samples: int
    chunks_yielded: int
    source_samples_offered: int
    source_samples_consumed: int
    tail_samples_offered: int
    tail_samples_consumed: int
    endpoint_reason: str
    native_endpoint: bool
    encoder_frame: int | None
    encoder_time_seconds: float | None
    event_origin: str
    endpoint_sample: int | None
    endpoint_latency_ms: float | None
    authoritative: bool
    observed_events: tuple[ParakeetCppObservedEvent, ...]


@dataclass(frozen=True)
class _ArtifactIdentity:
    path: Path = field(repr=False)
    sha256: str
    size_bytes: int
    filesystem_identity: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class _NativeEvent:
    event_type: str
    encoder_frame: int
    encoder_time_seconds: float


@dataclass(frozen=True)
class _NativeDocument:
    text: str = field(repr=False)
    events: tuple[_NativeEvent, ...]


@dataclass
class _JsonState:
    last_event_frame: int | None = None
    last_event_time: float | None = None
    seen_events: set[tuple[str, int]] = field(default_factory=set)
    last_word_start: float | None = None
    words_seen: int = 0


def _filesystem_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _verify_artifact(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    maximum_size_bytes: int,
) -> _ArtifactIdentity:
    if (
        not isinstance(expected_sha256, str)
        or _SHA256_RE.fullmatch(expected_sha256) is None
        or isinstance(expected_size_bytes, bool)
        or not isinstance(expected_size_bytes, int)
        or expected_size_bytes <= 0
        or isinstance(maximum_size_bytes, bool)
        or not isinstance(maximum_size_bytes, int)
        or expected_size_bytes > maximum_size_bytes
    ):
        raise ParakeetCppAdapterError()
    candidate = Path(path)
    try:
        before = candidate.lstat()
        resolved = candidate.resolve(strict=True)
        if (
            not candidate.is_absolute()
            or candidate != resolved
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            raise ParakeetCppAdapterError()
        digest = hash_regular_bounded(
            candidate,
            maximum_bytes=maximum_size_bytes,
            expected_bytes=expected_size_bytes,
        )
        after = candidate.lstat()
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        raise ParakeetCppAdapterError() from None
    identity = _filesystem_identity(before)
    if (
        digest.path != candidate
        or digest.sha256 != expected_sha256
        or digest.size_bytes != expected_size_bytes
        or _filesystem_identity(after) != identity
    ):
        raise ParakeetCppAdapterError()
    return _ArtifactIdentity(
        path=candidate,
        sha256=digest.sha256,
        size_bytes=digest.size_bytes,
        filesystem_identity=identity,
    )


def _flush_standard_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        flush = getattr(stream, "flush", None)
        if not callable(flush):
            raise ParakeetCppAdapterError()
        try:
            flush()
        except (OSError, RuntimeError, ValueError):
            raise ParakeetCppAdapterError() from None


def _restore_output_descriptors(
    saved: tuple[tuple[int, int], ...], sink_descriptor: int
) -> None:
    failed = False
    for target, descriptor in saved:
        try:
            os.dup2(descriptor, target, inheritable=False)
        except OSError:
            failed = True
    for _target, descriptor in saved:
        try:
            os.close(descriptor)
        except OSError:
            failed = True
    if sink_descriptor >= 0:
        try:
            os.close(sink_descriptor)
        except OSError:
            failed = True
    try:
        _flush_standard_streams()
    except ParakeetCppAdapterError:
        failed = True
    if failed:
        raise ParakeetCppAdapterError()


@contextmanager
def _suppressed_process_output() -> Iterator[None]:
    """Keep native paths, errors, and transcript fragments out of JSONL."""

    saved: list[tuple[int, int]] = []
    sink_descriptor = -1
    try:
        _flush_standard_streams()
        for target in (1, 2):
            descriptor = os.dup(target)
            os.set_inheritable(descriptor, False)
            saved.append((target, descriptor))
        sink_descriptor = os.open(
            os.devnull,
            os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
        )
        for target, _descriptor in saved:
            os.dup2(sink_descriptor, target, inheritable=False)
    except (OSError, RuntimeError, ValueError, ParakeetCppAdapterError):
        try:
            _restore_output_descriptors(tuple(saved), sink_descriptor)
        except ParakeetCppAdapterError:
            pass
        raise ParakeetCppAdapterError() from None
    try:
        yield
    finally:
        flush_failed = False
        try:
            _flush_standard_streams()
        except ParakeetCppAdapterError:
            flush_failed = True
        _restore_output_descriptors(tuple(saved), sink_descriptor)
        if flush_failed:
            raise ParakeetCppAdapterError()


def _suppressed_call(function: Callable[[], _T]) -> _T:
    try:
        with _suppressed_process_output():
            return function()
    except ParakeetCppAdapterError:
        raise
    except Exception:
        raise ParakeetCppAdapterError() from None


class _ParakeetCppApi:
    """Small ctypes surface over Speaker bridge ABI v1."""

    def __init__(self, bridge_path: Path) -> None:
        try:
            mode = getattr(os, "RTLD_NOW", 0) | getattr(
                os, "RTLD_LOCAL", getattr(ctypes, "RTLD_LOCAL", 0)
            )
            with _suppressed_process_output():
                library = ctypes.CDLL(str(bridge_path), mode=mode)
            self._bridge_abi = library.speaker_parakeet_bridge_abi_version
            self._bridge_abi.argtypes = []
            self._bridge_abi.restype = ctypes.c_int
            self._upstream_abi = library.speaker_parakeet_bridge_upstream_abi_version
            self._upstream_abi.argtypes = []
            self._upstream_abi.restype = ctypes.c_int
            self._open = library.speaker_parakeet_bridge_open
            self._open.argtypes = [
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            self._open.restype = ctypes.c_int
            self._actual_device = library.speaker_parakeet_bridge_actual_device
            self._actual_device.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t,
            ]
            self._actual_device.restype = ctypes.c_int
            self._begin = library.speaker_parakeet_bridge_begin
            self._begin.argtypes = [ctypes.c_void_p]
            self._begin.restype = ctypes.c_int
            self._feed = library.speaker_parakeet_bridge_feed_json
            self._feed.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            self._feed.restype = ctypes.c_int
            self._finalize = library.speaker_parakeet_bridge_finalize_json
            self._finalize.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            self._finalize.restype = ctypes.c_int
            self._free_json = library.speaker_parakeet_bridge_free_json
            self._free_json.argtypes = [ctypes.c_void_p]
            self._free_json.restype = None
            self._close = library.speaker_parakeet_bridge_close
            self._close.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            self._close.restype = ctypes.c_int
            self._library = library
        except (AttributeError, OSError, TypeError, ValueError, ParakeetCppAdapterError):
            raise ParakeetCppAdapterError() from None

    def bridge_abi_version(self) -> int:
        return int(self._bridge_abi())

    def upstream_abi_version(self) -> int:
        return int(self._upstream_abi())

    def open(self, model_path: Path, requested_device: str, num_threads: int) -> object:
        try:
            model_bytes = os.fsencode(model_path)
            device_bytes = requested_device.encode("ascii", errors="strict")
        except (UnicodeError, ValueError):
            raise ParakeetCppAdapterError() from None
        if b"\0" in model_bytes or b"\0" in device_bytes:
            raise ParakeetCppAdapterError()
        handle = ctypes.c_void_p()
        status = self._open(
            model_bytes,
            device_bytes,
            num_threads,
            ctypes.byref(handle),
        )
        if status != 0 or not handle.value:
            if handle.value:
                try:
                    self._close(ctypes.byref(handle))
                except Exception:
                    pass
            raise ParakeetCppAdapterError()
        return handle

    def actual_device(self, handle: object) -> str:
        if not isinstance(handle, ctypes.c_void_p) or not handle.value:
            raise ParakeetCppAdapterError()
        output = ctypes.create_string_buffer(_MAX_DEVICE_BYTES)
        if self._actual_device(handle, output, len(output)) != 0:
            raise ParakeetCppAdapterError()
        raw = bytes(output)
        terminator = raw.find(b"\0")
        if terminator <= 0 or any(raw[terminator + 1 :]):
            raise ParakeetCppAdapterError()
        try:
            return raw[:terminator].decode("ascii", errors="strict")
        except UnicodeError:
            raise ParakeetCppAdapterError() from None

    def begin(self, handle: object) -> None:
        if self._begin(handle) != 0:
            raise ParakeetCppAdapterError()

    def _json_result(self, function: object, *arguments: object) -> bytes:
        output = ctypes.c_void_p()
        output_bytes = ctypes.c_size_t()
        try:
            status = function(
                *arguments,
                ctypes.byref(output),
                ctypes.byref(output_bytes),
            )
            if (
                status != 0
                or not output.value
                or output_bytes.value <= 0
                or output_bytes.value > MAX_LINE_BYTES
            ):
                raise ParakeetCppAdapterError()
            return ctypes.string_at(output, output_bytes.value)
        except ParakeetCppAdapterError:
            raise
        except Exception:
            raise ParakeetCppAdapterError() from None
        finally:
            if output.value:
                try:
                    self._free_json(output)
                except Exception:
                    pass

    def feed_json(self, handle: object, samples: array[float]) -> bytes:
        if samples.typecode != "f" or samples.itemsize != 4 or not samples:
            raise ParakeetCppAdapterError()
        try:
            buffer = (ctypes.c_float * len(samples)).from_buffer(samples)
        except (BufferError, TypeError, ValueError):
            raise ParakeetCppAdapterError() from None
        return self._json_result(self._feed, handle, buffer, len(samples))

    def finalize_json(self, handle: object) -> bytes:
        return self._json_result(self._finalize, handle)

    def close(self, handle: object) -> None:
        if not isinstance(handle, ctypes.c_void_p):
            raise ParakeetCppAdapterError()
        if self._close(ctypes.byref(handle)) != 0 or handle.value:
            raise ParakeetCppAdapterError()


def _host_resources() -> ResourceUsage:
    rss_mb: float | None = None
    threads: int | None = None
    try:
        status_text = Path("/proc/self/status").read_text(
            encoding="utf-8", errors="strict"
        )
        for line in status_text.splitlines():
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
        raise ParakeetCppAdapterError()
    return ResourceUsage(rss_mb=rss_mb, threads=threads, vram_mb=None)


def _number(value: object, *, minimum: float = 0.0, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParakeetCppAdapterError()
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ParakeetCppAdapterError() from None
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise ParakeetCppAdapterError()
    return number


def _integer(value: object, *, minimum: int = 0, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise ParakeetCppAdapterError()
    return value


def _text(value: object, *, maximum: int = MAX_HYPOTHESIS_CHARS) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise ParakeetCppAdapterError()
    return value


def _parse_document(
    raw: bytes,
    *,
    config: ParakeetCppConfig,
    state: _JsonState,
) -> _NativeDocument:
    try:
        value = strict_json_bytes(raw)
    except Exception:
        raise ParakeetCppAdapterError() from None
    if not isinstance(value, dict) or set(value) != _ROOT_FIELDS:
        raise ParakeetCppAdapterError()
    text = _text(value.get("text"))
    frame_sec = _number(
        value.get("frame_sec"), minimum=1e-9, maximum=MAX_CASE_EVENT_MS / 1000.0
    )
    if not math.isclose(frame_sec, config.frame_sec, rel_tol=0.0, abs_tol=1e-9):
        raise ParakeetCppAdapterError()
    eou = _integer(value.get("eou"), maximum=1)
    eob = _integer(value.get("eob"), maximum=1)
    raw_events = value.get("events")
    if (
        not isinstance(raw_events, list)
        or len(raw_events) > _MAX_EVENTS_PER_DOCUMENT
    ):
        raise ParakeetCppAdapterError()
    events: list[_NativeEvent] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, dict) or set(raw_event) != _EVENT_FIELDS:
            raise ParakeetCppAdapterError()
        event_type = raw_event.get("type")
        if event_type not in {"eou", "eob"}:
            raise ParakeetCppAdapterError()
        frame = _integer(raw_event.get("frame"), maximum=MAX_STREAM_SAMPLES)
        event_time = _number(
            raw_event.get("t"), maximum=MAX_CASE_EVENT_MS / 1000.0
        )
        expected_time = frame * config.frame_sec
        if not math.isclose(event_time, expected_time, rel_tol=0.0, abs_tol=0.000501):
            raise ParakeetCppAdapterError()
        canonical_time = expected_time
        if (
            state.last_event_frame is not None
            and (
                frame < state.last_event_frame
                or canonical_time + 1e-9 < float(state.last_event_time)
            )
        ):
            raise ParakeetCppAdapterError()
        key = (str(event_type), frame)
        if key in state.seen_events:
            raise ParakeetCppAdapterError()
        state.seen_events.add(key)
        state.last_event_frame = frame
        state.last_event_time = canonical_time
        events.append(
            _NativeEvent(
                event_type=str(event_type),
                encoder_frame=frame,
                encoder_time_seconds=canonical_time,
            )
        )
    if eou != int(any(event.event_type == "eou" for event in events)) or eob != int(
        any(event.event_type == "eob" for event in events)
    ):
        raise ParakeetCppAdapterError()

    raw_words = value.get("words")
    if not isinstance(raw_words, list) or len(raw_words) > _MAX_WORDS_PER_DOCUMENT:
        raise ParakeetCppAdapterError()
    state.words_seen += len(raw_words)
    if state.words_seen > _MAX_WORDS_PER_CASE:
        raise ParakeetCppAdapterError()
    for raw_word in raw_words:
        if not isinstance(raw_word, dict) or set(raw_word) != _WORD_FIELDS:
            raise ParakeetCppAdapterError()
        _text(raw_word.get("w"), maximum=256)
        start = _number(
            raw_word.get("start"), maximum=MAX_CASE_EVENT_MS / 1000.0
        )
        end = _number(raw_word.get("end"), maximum=MAX_CASE_EVENT_MS / 1000.0)
        confidence = _number(raw_word.get("conf"), minimum=0.0, maximum=1.0)
        if (
            start > end
            or (
                state.last_word_start is not None
                and start + 1e-9 < state.last_word_start
            )
        ):
            raise ParakeetCppAdapterError()
        state.last_word_start = start
    return _NativeDocument(text=text, events=tuple(events))


def _read_pcm(request: TranscribeRequest, config: ParakeetCppConfig) -> array[float]:
    try:
        snapshot = read_regular_bounded(
            request.pcm.path,
            maximum_bytes=MAX_PCM_BYTES,
            expected_bytes=request.pcm.size_bytes,
        )
    except BoundedReadError:
        raise ParakeetCppAdapterError() from None
    raw = snapshot.data
    if (
        len(raw) != request.pcm.size_bytes
        or hashlib.sha256(raw).hexdigest() != request.pcm.sha256
        or request.pcm.sample_rate != config.sample_rate
        or request.pcm.channels != 1
        or request.pcm.encoding != "f32le"
    ):
        raise ParakeetCppAdapterError()
    samples = array("f")
    try:
        samples.frombytes(raw)
        if sys.byteorder != "little":
            samples.byteswap()
    except (BufferError, MemoryError, ValueError):
        raise ParakeetCppAdapterError() from None
    if (
        samples.itemsize != 4
        or len(samples) != request.pcm.samples
        or any(not math.isfinite(sample) for sample in samples)
    ):
        raise ParakeetCppAdapterError()
    return samples


class ParakeetCppAdapter:
    """One receipt-bound model with one utterance-bounded stream at a time."""

    def __init__(
        self,
        model_path: Path,
        library_path: Path,
        bridge_path: Path,
        *,
        config: ParakeetCppConfig,
        library_sha256: str,
        library_size_bytes: int,
        bridge_sha256: str,
        bridge_size_bytes: int,
        api: object | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
        resource_reader: Callable[[], ResourceUsage] = _host_resources,
        artifact_verifier: Callable[..., object] | None = None,
    ) -> None:
        if type(config) is not ParakeetCppConfig:
            raise ParakeetCppAdapterError()
        self.config = config
        self._clock = clock
        self._sleeper = sleeper
        self._resource_reader = resource_reader
        self._artifact_verifier = artifact_verifier or _verify_artifact
        self._closed = False
        self._poisoned = False
        self._stream_active = False
        self._handle: object | None = None
        self._api: object | None = None
        selected_model = Path(model_path)
        selected_library = Path(library_path)
        selected_bridge = Path(bridge_path)
        model_parent = selected_model.parent
        native_parent = selected_library.parent
        if (
            not selected_model.is_absolute()
            or not selected_library.is_absolute()
            or not selected_bridge.is_absolute()
            or selected_model.name != config.model_filename
            or selected_library.name != _LIBPARAKEET_BASENAME
            or selected_bridge.name != _BRIDGE_BASENAME
            or len({selected_model, selected_library, selected_bridge}) != 3
            or selected_library.parent != selected_bridge.parent
            or model_parent == native_parent
            or model_parent in native_parent.parents
            or native_parent in model_parent.parents
            or _DEVICE_RE.fullmatch(config.requested_device) is None
            or _DEVICE_RE.fullmatch(config.actual_device) is None
        ):
            raise ParakeetCppAdapterError()
        self.model_path = selected_model
        self.library_path = selected_library
        self.bridge_path = selected_bridge
        self._artifact_specs = (
            (
                self.library_path,
                library_sha256,
                library_size_bytes,
                MAX_PARAKEET_CPP_LIBRARY_BYTES,
            ),
            (
                self.bridge_path,
                bridge_sha256,
                bridge_size_bytes,
                MAX_PARAKEET_CPP_LIBRARY_BYTES,
            ),
            (
                self.model_path,
                config.model_sha256,
                config.model_size_bytes,
                MAX_PARAKEET_CPP_MODEL_BYTES,
            ),
        )
        self._artifact_identities = self._verify_artifacts()
        load_started = self._now()
        try:
            candidate = api if api is not None else _ParakeetCppApi(self.bridge_path)
            for name in (
                "bridge_abi_version",
                "upstream_abi_version",
                "open",
                "actual_device",
                "begin",
                "feed_json",
                "finalize_json",
                "close",
            ):
                if not callable(getattr(candidate, name, None)):
                    raise ParakeetCppAdapterError()
            bridge_abi = self._native_call(candidate.bridge_abi_version)
            upstream_abi = self._native_call(candidate.upstream_abi_version)
            if (
                type(bridge_abi) is not int
                or bridge_abi != config.bridge_abi_version
                or type(upstream_abi) is not int
                or upstream_abi != config.c_api_version
            ):
                raise ParakeetCppAdapterError()
            handle = self._native_call(
                lambda: candidate.open(
                    self.model_path,
                    config.requested_device,
                    config.num_threads,
                )
            )
            if handle is None:
                raise ParakeetCppAdapterError()
            self._api = candidate
            self._handle = handle
            actual_device = self._native_call(lambda: candidate.actual_device(handle))
            if (
                not isinstance(actual_device, str)
                or actual_device != config.actual_device
                or actual_device.casefold() != config.requested_device.casefold()
            ):
                raise ParakeetCppAdapterError()
            self._require_same_artifacts()
            self.ready = ParakeetCppReady(
                model_load_ms=self._duration_ms(load_started, self._now()),
                resources=self._resources(),
                requested_device=config.requested_device,
                actual_device=actual_device,
                num_threads=config.num_threads,
                bridge_abi_version=bridge_abi,
                c_api_version=upstream_abi,
            )
        except Exception:
            self._abort()
            raise ParakeetCppAdapterError() from None

    def __enter__(self) -> ParakeetCppAdapter:
        if self._closed:
            raise ParakeetCppAdapterError()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def _verify_artifacts(self) -> tuple[object, ...]:
        identities: list[object] = []
        for path, sha256, size_bytes, maximum_bytes in self._artifact_specs:
            try:
                identity = self._artifact_verifier(
                    path,
                    expected_sha256=sha256,
                    expected_size_bytes=size_bytes,
                    maximum_size_bytes=maximum_bytes,
                )
            except Exception:
                raise ParakeetCppAdapterError() from None
            identities.append(identity)
        return tuple(identities)

    def _require_same_artifacts(self) -> None:
        if self._verify_artifacts() != self._artifact_identities:
            raise ParakeetCppAdapterError()

    def _now(self) -> float:
        try:
            value = float(self._clock())
        except (OverflowError, TypeError, ValueError):
            raise ParakeetCppAdapterError() from None
        if not math.isfinite(value):
            raise ParakeetCppAdapterError()
        return value

    @staticmethod
    def _duration_ms(start: float, end: float) -> float:
        duration = end - start
        if not math.isfinite(duration) or duration < 0.0:
            raise ParakeetCppAdapterError()
        return duration * 1000.0

    def _native_call(self, function: Callable[[], _T]) -> _T:
        return _suppressed_call(function)

    def _timed_native(self, function: Callable[[], _T]) -> tuple[_T, float]:
        started = self._now()
        result = self._native_call(function)
        finished = self._now()
        return result, self._duration_ms(started, finished)

    def _resources(self) -> ResourceUsage:
        try:
            resources = self._resource_reader()
        except Exception:
            raise ParakeetCppAdapterError() from None
        if (
            not isinstance(resources, ResourceUsage)
            or not math.isfinite(resources.rss_mb)
            or not 0.0 <= resources.rss_mb <= MAX_RESOURCE_MB
            or not 1 <= resources.threads <= 4096
            or resources.vram_mb is not None
        ):
            raise ParakeetCppAdapterError()
        return resources

    def _abort(self) -> None:
        self._poisoned = True
        self._closed = True
        api, handle = self._api, self._handle
        self._handle = None
        self._stream_active = False
        if api is not None and handle is not None:
            try:
                self._native_call(lambda: api.close(handle))
            except Exception:
                pass
        try:
            if hasattr(self, "_artifact_identities"):
                self._require_same_artifacts()
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        failed = self._stream_active
        api, handle = self._api, self._handle
        self._handle = None
        self._stream_active = False
        if api is not None and handle is not None:
            try:
                self._native_call(lambda: api.close(handle))
            except Exception:
                failed = True
        try:
            self._require_same_artifacts()
        except ParakeetCppAdapterError:
            failed = True
        if failed:
            self._poisoned = True
            raise ParakeetCppAdapterError()

    def transcribe(
        self,
        request: TranscribeRequest,
        *,
        emit_partial: Callable[[int, ParakeetCppPartial], None],
    ) -> ParakeetCppCase:
        if (
            self._closed
            or self._poisoned
            or self._stream_active
            or self._api is None
            or self._handle is None
            or request.protocol_version != PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION
            or request.stream.chunk_samples != self.config.native_chunk_samples
            or request.stream.tail_padding_samples
            > self.config.maximum_tail_padding_samples
            or request.stream.pace not in {"burst", "realtime"}
            or type(request.stream.partial_interval_ms) is not int
            or request.stream.partial_interval_ms <= 0
        ):
            raise ParakeetCppAdapterError()
        source = _read_pcm(request, self.config)
        source_offered = request.pcm.samples
        tail_offered = request.stream.tail_padding_samples
        total_offered = source_offered + tail_offered
        if source_offered <= 0 or total_offered > MAX_STREAM_SAMPLES:
            raise ParakeetCppAdapterError()

        case_started = self._now()
        stream_started = case_started
        partials: list[ParakeetCppPartial] = []
        json_state = _JsonState()
        visible_text = ""
        latest_text = ""
        decoded_text_bytes = 0
        source_consumed = 0
        tail_consumed = 0
        samples_seen = 0
        chunks = 0
        compute_ms = 0.0
        finalization_ms = 0.0
        deadline_misses = 0
        max_backlog_ms = 0.0
        first_eou_event: ParakeetCppObservedEvent | None = None
        accepted_event: ParakeetCppObservedEvent | None = None
        observed_events: list[ParakeetCppObservedEvent] = []
        partial_interval_samples = (
            request.stream.partial_interval_ms * self.config.sample_rate + 999
        ) // 1000
        next_partial_samples = partial_interval_samples

        try:
            self._native_call(lambda: self._api.begin(self._handle))
            self._stream_active = True
            for offset in range(0, total_offered, self.config.native_chunk_samples):
                actual_count = min(
                    self.config.native_chunk_samples,
                    total_offered - offset,
                )
                if request.stream.pace == "realtime":
                    target = stream_started + offset / self.config.sample_rate
                    before_wait = self._now()
                    if before_wait < target:
                        self._sleeper(target - before_wait)
                    if self._now() < target:
                        raise ParakeetCppAdapterError()

                source_start = min(offset, source_offered)
                source_end = min(offset + actual_count, source_offered)
                source_count = max(0, source_end - source_start)
                tail_count = actual_count - source_count
                chunk = array("f", source[source_start:source_end])
                if tail_count:
                    chunk.extend([0.0] * tail_count)
                if len(chunk) != actual_count:
                    raise ParakeetCppAdapterError()
                payload, decode_ms = self._timed_native(
                    lambda chunk=chunk: self._api.feed_json(self._handle, chunk)
                )
                if not isinstance(payload, bytes):
                    raise ParakeetCppAdapterError()
                document = _parse_document(
                    payload,
                    config=self.config,
                    state=json_state,
                )
                text_was_frozen = first_eou_event is not None
                document_text_bytes = len(document.text.encode("utf-8"))
                if (
                    decoded_text_bytes + document_text_bytes
                    > _MAX_DECODED_TEXT_BYTES
                ):
                    raise ParakeetCppAdapterError()
                decoded_text_bytes += document_text_bytes
                if not text_was_frozen:
                    visible_text += document.text
                    latest_text = visible_text.strip()
                source_consumed += source_count
                tail_consumed += tail_count
                samples_seen += actual_count
                chunks += 1
                compute_ms += decode_ms

                document_observations: list[ParakeetCppObservedEvent] = []
                for event in document.events:
                    if (
                        event.encoder_time_seconds
                        > samples_seen / self.config.sample_rate + 1e-9
                    ):
                        raise ParakeetCppAdapterError()
                    if len(observed_events) >= MAX_PARAKEET_CPP_OBSERVED_EVENTS:
                        raise ParakeetCppAdapterError()
                    observation = ParakeetCppObservedEvent(
                        event_type=event.event_type,
                        event_origin="feed",
                        observed_sample=samples_seen,
                        encoder_frame=event.encoder_frame,
                        encoder_time_seconds=event.encoder_time_seconds,
                    )
                    observed_events.append(observation)
                    document_observations.append(observation)

                if first_eou_event is None:
                    first_eou_event = next(
                        (
                            event
                            for event in document_observations
                            if event.event_type == "eou"
                        ),
                        None,
                    )
                    if (
                        first_eou_event is not None
                        and samples_seen >= source_offered
                    ):
                        accepted_event = first_eou_event

                if request.stream.pace == "realtime":
                    deadline = stream_started + samples_seen / self.config.sample_rate
                    now = self._now()
                    backlog_ms = self._duration_ms(deadline, max(deadline, now))
                    max_backlog_ms = max(max_backlog_ms, backlog_ms)
                    if backlog_ms > 0.0:
                        deadline_misses += 1

                if (
                    not text_was_frozen
                    and samples_seen >= next_partial_samples
                    and latest_text
                ):
                    if len(partials) >= MAX_PARTIALS_PER_CASE:
                        raise ParakeetCppAdapterError()
                    if not partials or partials[-1].text != latest_text:
                        partial = ParakeetCppPartial(
                            after_samples=samples_seen,
                            text=latest_text,
                            elapsed_ms=self._duration_ms(case_started, self._now()),
                            decode_ms=decode_ms,
                        )
                        emit_partial(len(partials), partial)
                        partials.append(partial)
                    while next_partial_samples <= samples_seen:
                        next_partial_samples += partial_interval_samples

                if accepted_event is not None:
                    break

            finalize_payload, finalization_ms = self._timed_native(
                lambda: self._api.finalize_json(self._handle)
            )
            self._stream_active = False
            if not isinstance(finalize_payload, bytes):
                raise ParakeetCppAdapterError()
            final_document = _parse_document(
                finalize_payload,
                config=self.config,
                state=json_state,
            )
            compute_ms += finalization_ms
            text_was_frozen = first_eou_event is not None
            final_text_bytes = len(final_document.text.encode("utf-8"))
            if decoded_text_bytes + final_text_bytes > _MAX_DECODED_TEXT_BYTES:
                raise ParakeetCppAdapterError()
            decoded_text_bytes += final_text_bytes
            if not text_was_frozen:
                visible_text += final_document.text
                latest_text = visible_text.strip()
            for event in final_document.events:
                if len(observed_events) >= MAX_PARAKEET_CPP_OBSERVED_EVENTS:
                    raise ParakeetCppAdapterError()
                observation = ParakeetCppObservedEvent(
                    event_type=event.event_type,
                    event_origin="finalize",
                    observed_sample=samples_seen,
                    encoder_frame=event.encoder_frame,
                    encoder_time_seconds=event.encoder_time_seconds,
                )
                observed_events.append(observation)
                if first_eou_event is None and event.event_type == "eou":
                    first_eou_event = observation
        except Exception:
            self._abort()
            raise ParakeetCppAdapterError() from None

        finished = self._now()
        native_endpoint = accepted_event is not None
        endpoint_reason = "eou" if native_endpoint else "tail_exhausted"
        if not native_endpoint and (
            source_consumed != source_offered or tail_consumed != tail_offered
        ):
            self._abort()
            raise ParakeetCppAdapterError()
        endpoint_sample = accepted_event.observed_sample if accepted_event else None
        encoder_frame = (
            accepted_event.encoder_frame if accepted_event is not None else None
        )
        encoder_time_seconds = (
            accepted_event.encoder_time_seconds if accepted_event is not None else None
        )
        event_origin = "feed" if native_endpoint else "none"
        authoritative = native_endpoint
        endpoint_latency_ms = (
            tail_consumed * 1000.0 / self.config.sample_rate
            if authoritative
            else None
        )
        try:
            return ParakeetCppCase(
                partials=tuple(partials),
                final=latest_text,
                elapsed_ms=self._duration_ms(case_started, finished),
                finalization_ms=finalization_ms,
                compute_ms=compute_ms,
                deadline_misses=deadline_misses,
                max_backlog_ms=max_backlog_ms,
                resources=self._resources(),
                model_padding_samples=0,
                chunks_yielded=chunks,
                source_samples_offered=source_offered,
                source_samples_consumed=source_consumed,
                tail_samples_offered=tail_offered,
                tail_samples_consumed=tail_consumed,
                endpoint_reason=endpoint_reason,
                native_endpoint=native_endpoint,
                encoder_frame=encoder_frame,
                encoder_time_seconds=encoder_time_seconds,
                event_origin=event_origin,
                endpoint_sample=endpoint_sample,
                endpoint_latency_ms=endpoint_latency_ms,
                authoritative=authoritative,
                observed_events=tuple(observed_events),
            )
        except Exception:
            self._abort()
            raise ParakeetCppAdapterError() from None


__all__ = [
    "PARAKEET_CPP_ADAPTER",
    "ParakeetCppAdapter",
    "ParakeetCppAdapterError",
    "ParakeetCppCase",
    "ParakeetCppConfig",
    "ParakeetCppPartial",
    "ParakeetCppReady",
]
