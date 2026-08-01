"""Private, capture-coordinate-correlated diagnostics for one voice session.

The normal run log is intentionally best-effort and text-oriented.  This module
provides the smaller evidence primitive needed for acoustic diagnosis: one
bounded admission queue owns every model-rate track and a typed, transcript-free
timeline. Stateful DSP can change per-call lengths, so each track has its own
packed cursor. A capture frame is admitted for every track or none. Any loss,
shape error, I/O error, or unclean shutdown makes the final manifest incomplete.

This schema is the Sherpa live-capture diagnostic profile. Callers copy scalar
lineage from ``CapturedBlock`` into :class:`CaptureCoordinate` and never pass a
device name, transcript, prompt, or arbitrary metadata mapping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import stat
import struct
import threading
import time
from types import MappingProxyType
from typing import BinaryIO, Mapping, Optional
import wave

import numpy as np


SCHEMA_VERSION = 1
_MANIFEST_MAX_BYTES = 1 << 20
_TIMELINE_LINE_MAX_BYTES = 1 << 16
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SYMBOL_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_STREAM_ID_RE = re.compile(r"^sherpa-[0-9a-f]{32}$")
_UTTERANCE_ID_RE = re.compile(r"^u[1-9][0-9]*$")
_CORRELATION_ID_RE = re.compile(r"^playback-[1-9][0-9]*$")
_AGENT_SESSION_ID_RE = re.compile(r"^session-[0-9a-f]{32}$")
_AGENT_TURN_ID_RE = re.compile(r"^turn-[1-9][0-9]*$")
_GAP_REASONS = frozenset(
    {"device_overflow", "media_backpressure", "capture_recovery"}
)
_EXTERNAL_FAILURE_CODES = frozenset(
    {
        "capture_integration_error",
        "capture_shutdown_timeout",
        "finalizer_timeout",
        "observation_error",
        "playback_timeout",
        "receipt_timeout",
    }
)
_FIELD_SYMBOLS = {
    "reason": frozenset(
        {
            "unknown",
            "asr",
            "vad_silence",
            "semantic",
            "max_wait",
            "command",
            "aborted",
            "abandoned",
            "backpressure",
            "capture_recovery",
            "decode_error",
            "echo_rejected",
            "empty_final",
            "input_rejected",
            "internal_error",
            "shutdown",
            "speaker_rejected",
            "stale_fenced",
            "callback_error",
        }
    ),
    "basis": frozenset(
        {"acoustic", "semantic_early", "semantic_hold", "semantic_wait", "max_wait"}
    ),
    "completion_state": frozenset(
        {
            "hard_boundary",
            "unavailable",
            "no_partial",
            "early_guarded",
            "audio_window_pending",
            "detector_error",
            "scored",
        }
    ),
    "selected_source": frozenset(
        {"none", "streaming", "offline", "verifier_consensus", "established_override"}
    ),
    "outcome": frozenset(
        {
            "unavailable",
            "skipped",
            "error",
            "decoded",
            "empty",
            "consensus",
            "no_quorum",
            "tie",
            "control_guard",
            "empty_veto",
            "attested_control",
            "empty_streaming_guard",
            "nonempty",
            "callback_returned",
            "completed",
            "interrupted",
            "dropped",
            "failed",
        }
    ),
    "playback_kind": frozenset(
        {"reply", "auxiliary", "latency_ack", "reminder"}
    ),
}


class DiagnosticTrack(str, Enum):
    """Closed model-rate audio roles accepted by a diagnostic bundle."""

    MODEL_PRE_GAIN_TAP = "model_pre_gain_tap"
    MODEL_GATE_PCM = "model_gate_pcm"
    MODEL_ASR_TAP = "model_asr_tap"
    PLAYBACK_REFERENCE_READER_SNAPSHOT = "playback_reference_reader_snapshot"


class DiagnosticStage(str, Enum):
    """Closed semantic stages that may be correlated with recorded audio."""

    VAD_TRANSITION = "vad_transition"
    ASR_PARTIAL = "asr_partial"
    ASR_STREAMING_FINAL = "asr_streaming_final"
    ENDPOINT_PAUSE_OBSERVED = "endpoint_pause_observed"
    ENDPOINT_EVALUATED = "endpoint_evaluated"
    ENDPOINT_COMMITTED = "endpoint_committed"
    FINALIZER_QUEUED = "finalizer_queued"
    FINALIZER_STARTED = "finalizer_started"
    FINAL_SELECTION = "final_selection"
    FINAL_DISPATCHED = "final_dispatched"
    FINAL_ABORTED = "final_aborted"
    PLAYBACK_ADMITTED = "playback_admitted"
    PLAYBACK_STARTED = "playback_started"
    PLAYBACK_TERMINAL = "playback_terminal"
    BARGE_DETECTED = "barge_detected"


class DiagnosticManifestStatus(str, Enum):
    """Closed publication lifecycle for a bundle manifest."""

    OPEN = "open"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    PUBLISH_FAILED = "publish_failed"


@dataclass(frozen=True, slots=True)
class EndpointReplayConfig:
    """Exact transcript-free policy inputs needed to replay turn decisions.

    ``initial_pause_samples_sec`` snapshots the bounded adaptive-pause learner
    when recording opens. Later accepted pauses are ordered timeline events.
    """

    algorithm: str = "adaptive_endpoint_v1"
    enabled: bool = False
    min_silence_sec: float = 0.5
    max_silence_sec: float = 1.6
    complete_threshold: float = 0.6
    incomplete_threshold: float = 0.3
    high_confidence_floor: float = 0.0
    high_confidence_score: float = 0.75
    adaptive_floor: bool = False
    pause_window: int = 64
    pause_quantile: float = 0.85
    pause_margin: float = 0.15
    pause_min_samples: int = 8
    initial_pause_samples_sec: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.algorithm != "adaptive_endpoint_v1":
            raise ValueError("unknown endpoint replay algorithm")
        for name in ("enabled", "adaptive_floor"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        for name in (
            "min_silence_sec",
            "max_silence_sec",
            "complete_threshold",
            "incomplete_threshold",
            "high_confidence_floor",
            "high_confidence_score",
            "pause_quantile",
            "pause_margin",
        ):
            value = _optional_number(name, getattr(self, name))
            assert value is not None
        for name in ("pause_window", "pause_min_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if not isinstance(self.initial_pause_samples_sec, tuple):
            raise TypeError("initial_pause_samples_sec must be a tuple")
        if len(self.initial_pause_samples_sec) > max(1, self.pause_window):
            raise ValueError("initial pause state exceeds pause_window")
        for sample in self.initial_pause_samples_sec:
            _optional_number("initial_pause_samples_sec", sample)

    def to_payload(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "enabled": self.enabled,
            "min_silence_sec": self.min_silence_sec,
            "max_silence_sec": self.max_silence_sec,
            "complete_threshold": self.complete_threshold,
            "incomplete_threshold": self.incomplete_threshold,
            "high_confidence_floor": self.high_confidence_floor,
            "high_confidence_score": self.high_confidence_score,
            "adaptive_floor": self.adaptive_floor,
            "pause_window": self.pause_window,
            "pause_quantile": self.pause_quantile,
            "pause_margin": self.pause_margin,
            "pause_min_samples": self.pause_min_samples,
            "initial_pause_samples_sec": list(self.initial_pause_samples_sec),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "EndpointReplayConfig":
        if not isinstance(payload, dict):
            raise TypeError("endpoint replay config must be an object")
        expected = {
            "algorithm",
            "enabled",
            "min_silence_sec",
            "max_silence_sec",
            "complete_threshold",
            "incomplete_threshold",
            "high_confidence_floor",
            "high_confidence_score",
            "adaptive_floor",
            "pause_window",
            "pause_quantile",
            "pause_margin",
            "pause_min_samples",
            "initial_pause_samples_sec",
        }
        if set(payload) != expected:
            raise ValueError("endpoint replay config has an invalid shape")
        values = dict(payload)
        samples = values["initial_pause_samples_sec"]
        if not isinstance(samples, list):
            raise TypeError("initial_pause_samples_sec must be an array")
        values["initial_pause_samples_sec"] = tuple(samples)
        return cls(**values)  # type: ignore[arg-type]


_ACOUSTIC_FIELDS = frozenset({"span", "stream_id", "utterance_id", "revision"})
_AGENT_FIELDS = frozenset(
    {"correlation_id", "agent_session_id", "agent_turn_id", "agent_revision"}
)
_STAGE_FIELDS: dict[DiagnosticStage, tuple[frozenset[str], frozenset[str]]] = {
    DiagnosticStage.VAD_TRANSITION: (
        frozenset({"span", "boolean_value"}),
        _ACOUSTIC_FIELDS | {"boolean_value"},
    ),
    DiagnosticStage.ASR_PARTIAL: (
        frozenset({"span", "stream_id", "utterance_id", "revision"}),
        _ACOUSTIC_FIELDS,
    ),
    DiagnosticStage.ASR_STREAMING_FINAL: (
        frozenset({"span", "stream_id", "utterance_id", "revision", "outcome"}),
        _ACOUSTIC_FIELDS | {"outcome"},
    ),
    DiagnosticStage.ENDPOINT_PAUSE_OBSERVED: (
        frozenset({"span", "numeric_value"}),
        _ACOUSTIC_FIELDS | {"numeric_value"},
    ),
    DiagnosticStage.ENDPOINT_EVALUATED: (
        frozenset(
            {
                "span",
                "reason",
                "basis",
                "completion_state",
                "acoustic_endpoint",
                "vad_active",
                "early_endpoint_allowed",
                "trailing_silence_sec",
                "boolean_value",
            }
        ),
        _ACOUSTIC_FIELDS
        | {
            "reason",
            "basis",
            "completion_state",
            "numeric_value",
            "acoustic_endpoint",
            "vad_active",
            "early_endpoint_allowed",
            "trailing_silence_sec",
            "boolean_value",
        },
    ),
    DiagnosticStage.ENDPOINT_COMMITTED: (
        frozenset(
            {
                "span",
                "stream_id",
                "utterance_id",
                "revision",
                "reason",
                "basis",
                "completion_state",
            }
        ),
        _ACOUSTIC_FIELDS
        | {"reason", "basis", "completion_state", "numeric_value"},
    ),
    DiagnosticStage.FINALIZER_QUEUED: (_ACOUSTIC_FIELDS, _ACOUSTIC_FIELDS),
    DiagnosticStage.FINALIZER_STARTED: (_ACOUSTIC_FIELDS, _ACOUSTIC_FIELDS),
    DiagnosticStage.FINAL_SELECTION: (
        _ACOUSTIC_FIELDS | {"selected_source", "outcome", "support", "boolean_value"},
        _ACOUSTIC_FIELDS | {"selected_source", "outcome", "support", "boolean_value"},
    ),
    DiagnosticStage.FINAL_DISPATCHED: (
        _ACOUSTIC_FIELDS | {"outcome"},
        _ACOUSTIC_FIELDS | {"outcome"},
    ),
    DiagnosticStage.FINAL_ABORTED: (
        _ACOUSTIC_FIELDS | {"reason"},
        _ACOUSTIC_FIELDS | {"reason"},
    ),
    DiagnosticStage.PLAYBACK_ADMITTED: (
        _AGENT_FIELDS | {"playback_kind"},
        _AGENT_FIELDS
        | {"stream_id", "utterance_id", "revision", "playback_kind"},
    ),
    DiagnosticStage.PLAYBACK_STARTED: (
        _AGENT_FIELDS,
        _AGENT_FIELDS,
    ),
    DiagnosticStage.PLAYBACK_TERMINAL: (
        _AGENT_FIELDS | {"outcome"},
        _AGENT_FIELDS | {"outcome", "numeric_value"},
    ),
    DiagnosticStage.BARGE_DETECTED: (
        frozenset(),
        _ACOUSTIC_FIELDS,
    ),
}


def _validate_stage_shape(stage: DiagnosticStage, present: set[str]) -> None:
    required, allowed = _STAGE_FIELDS[stage]
    missing = required - present
    unexpected = present - allowed
    if missing:
        raise ValueError(f"{stage.value} is missing required diagnostic fields")
    if unexpected:
        raise ValueError(f"{stage.value} contains forbidden diagnostic fields")


def _non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _optional_non_negative_int(name: str, value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return _non_negative_int(name, value)


def _finite_clock(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative clock")
    return result


def _optional_number(name: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _symbol(name: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a symbolic string")
    if not _SYMBOL_RE.fullmatch(value):
        raise ValueError(
            f"{name} must be a closed symbolic token, not free-form text"
        )
    return value


def _closed_symbol(name: str, value: Optional[str]) -> Optional[str]:
    result = _symbol(name, value)
    if result is not None and result not in _FIELD_SYMBOLS[name]:
        raise ValueError(f"{name} is not a recognized closed symbol")
    return result


def _opaque_id(name: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be an opaque identifier")
    if not _OPAQUE_ID_RE.fullmatch(value):
        raise ValueError(f"{name} must be an opaque token without whitespace")
    expected = {
        "stream_id": _STREAM_ID_RE,
        "utterance_id": _UTTERANCE_ID_RE,
        "correlation_id": _CORRELATION_ID_RE,
        "agent_session_id": _AGENT_SESSION_ID_RE,
        "agent_turn_id": _AGENT_TURN_ID_RE,
    }[name]
    if expected.fullmatch(value) is None:
        raise ValueError(f"{name} does not match its generated identifier form")
    return value


@dataclass(frozen=True, slots=True)
class CaptureCoordinate:
    """Scalar capture facts copied from one ``CapturedBlock``.

    Device identity is deliberately absent.  A gap is flattened into scalar
    fields so this type never retains an engine object or arbitrary payload.
    """

    sequence: int
    sample_rate_hz: int
    captured_started_at: float
    captured_at: float
    capture_epoch: int
    source_generation: int
    capture_generation: int
    source_sample_start: int
    source_sample_end: int
    gap_reason: Optional[str] = None
    gap_prior_generation: Optional[int] = None
    gap_generation: Optional[int] = None
    gap_first_dropped_sequence: Optional[int] = None
    gap_last_dropped_sequence: Optional[int] = None
    gap_dropped_frames: int = 0
    gap_dropped_samples: int = 0

    def __post_init__(self) -> None:
        _non_negative_int("sequence", self.sequence)
        _non_negative_int("capture_epoch", self.capture_epoch)
        _non_negative_int("source_generation", self.source_generation)
        _non_negative_int("capture_generation", self.capture_generation)
        _non_negative_int("source_sample_start", self.source_sample_start)
        _non_negative_int("source_sample_end", self.source_sample_end)
        if self.source_sample_end <= self.source_sample_start:
            raise ValueError("source_sample_end must follow source_sample_start")
        if (
            isinstance(self.sample_rate_hz, bool)
            or not isinstance(self.sample_rate_hz, int)
            or self.sample_rate_hz <= 0
        ):
            raise ValueError("sample_rate_hz must be a positive integer")
        started = _finite_clock("captured_started_at", self.captured_started_at)
        ended = _finite_clock("captured_at", self.captured_at)
        if ended < started:
            raise ValueError("captured_at cannot precede captured_started_at")

        gap_reason = self.gap_reason
        if isinstance(gap_reason, Enum):
            gap_reason = str(gap_reason.value)
            object.__setattr__(self, "gap_reason", gap_reason)
        if gap_reason is not None and gap_reason not in _GAP_REASONS:
            raise ValueError("gap_reason is not a closed capture-gap symbol")
        optional_gap = (
            ("gap_prior_generation", self.gap_prior_generation),
            ("gap_generation", self.gap_generation),
            ("gap_first_dropped_sequence", self.gap_first_dropped_sequence),
            ("gap_last_dropped_sequence", self.gap_last_dropped_sequence),
        )
        for name, value in optional_gap:
            _optional_non_negative_int(name, value)
        _non_negative_int("gap_dropped_frames", self.gap_dropped_frames)
        _non_negative_int("gap_dropped_samples", self.gap_dropped_samples)
        if gap_reason is None:
            if any(value is not None for _name, value in optional_gap) or (
                self.gap_dropped_frames or self.gap_dropped_samples
            ):
                raise ValueError("gap fields require gap_reason")
        else:
            if self.gap_prior_generation is None or self.gap_generation is None:
                raise ValueError("a gap requires prior and current generations")
            if self.gap_generation <= self.gap_prior_generation:
                raise ValueError("gap_generation must advance")
            first = self.gap_first_dropped_sequence
            last = self.gap_last_dropped_sequence
            if (first is None) != (last is None):
                raise ValueError("dropped sequence bounds must be paired")
            if first is not None and last is not None and last < first:
                raise ValueError("last dropped sequence cannot precede first")

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        return {name: value for name, value in payload.items() if value is not None}


@dataclass(frozen=True, slots=True)
class BundleAudioSpan:
    """Receipt for one frame on the packed model-gate coordinate."""

    frame_index: int
    bundle_sample_start: int
    bundle_sample_end: int
    coordinate: CaptureCoordinate
    _bundle_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _non_negative_int("frame_index", self.frame_index)
        _non_negative_int("bundle_sample_start", self.bundle_sample_start)
        _non_negative_int("bundle_sample_end", self.bundle_sample_end)
        if self.bundle_sample_end <= self.bundle_sample_start:
            raise ValueError("bundle_sample_end must follow bundle_sample_start")
        if not isinstance(self.coordinate, CaptureCoordinate):
            raise TypeError("coordinate must be a CaptureCoordinate")


@dataclass(frozen=True, slots=True)
class DiagnosticObservation:
    """One typed, transcript-free semantic observation.

    The fixed schema intentionally offers no text, hash, byte-count, token-count,
    or arbitrary metadata field.  ``symbol``/``secondary_symbol`` accept only
    short machine tokens; ``opaque_id`` accepts only identifier syntax.
    """

    stage: DiagnosticStage
    monotonic_ns: int
    span: Optional[BundleAudioSpan] = None
    stream_id: Optional[str] = None
    utterance_id: Optional[str] = None
    correlation_id: Optional[str] = None
    agent_session_id: Optional[str] = None
    agent_turn_id: Optional[str] = None
    agent_revision: Optional[int] = None
    revision: Optional[int] = None
    reason: Optional[str] = None
    basis: Optional[str] = None
    completion_state: Optional[str] = None
    selected_source: Optional[str] = None
    outcome: Optional[str] = None
    playback_kind: Optional[str] = None
    numeric_value: Optional[float] = None
    support: Optional[int] = None
    boolean_value: Optional[bool] = None
    acoustic_endpoint: Optional[bool] = None
    vad_active: Optional[bool] = None
    early_endpoint_allowed: Optional[bool] = None
    trailing_silence_sec: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, DiagnosticStage):
            raise TypeError("stage must be a DiagnosticStage")
        _non_negative_int("monotonic_ns", self.monotonic_ns)
        if self.span is not None and not isinstance(self.span, BundleAudioSpan):
            raise TypeError("span must be a BundleAudioSpan or None")
        _opaque_id("stream_id", self.stream_id)
        _opaque_id("utterance_id", self.utterance_id)
        _opaque_id("correlation_id", self.correlation_id)
        _opaque_id("agent_session_id", self.agent_session_id)
        _opaque_id("agent_turn_id", self.agent_turn_id)
        _optional_non_negative_int("agent_revision", self.agent_revision)
        _optional_non_negative_int("revision", self.revision)
        for name in (
            "reason",
            "basis",
            "completion_state",
            "selected_source",
            "outcome",
            "playback_kind",
        ):
            _closed_symbol(name, getattr(self, name))
        _optional_number("numeric_value", self.numeric_value)
        _optional_non_negative_int("support", self.support)
        if self.boolean_value is not None and not isinstance(
            self.boolean_value, bool
        ):
            raise TypeError("boolean_value must be a boolean")
        for name in (
            "acoustic_endpoint",
            "vad_active",
            "early_endpoint_allowed",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")
        if self.trailing_silence_sec is not None:
            _finite_clock("trailing_silence_sec", self.trailing_silence_sec)
        present = {
            name
            for name in (
                "span",
                "stream_id",
                "utterance_id",
                "correlation_id",
                "agent_session_id",
                "agent_turn_id",
                "agent_revision",
                "revision",
                "reason",
                "basis",
                "completion_state",
                "selected_source",
                "outcome",
                "playback_kind",
                "numeric_value",
                "support",
                "boolean_value",
                "acoustic_endpoint",
                "vad_active",
                "early_endpoint_allowed",
                "trailing_silence_sec",
            )
            if getattr(self, name) is not None
        }
        _validate_stage_shape(self.stage, present)

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": "observation",
            "stage": self.stage.value,
            "monotonic_ns": self.monotonic_ns,
        }
        if self.span is not None:
            payload.update(
                frame_index=self.span.frame_index,
                bundle_sample_start=self.span.bundle_sample_start,
                bundle_sample_end=self.span.bundle_sample_end,
            )
        for name in (
            "stream_id",
            "utterance_id",
            "correlation_id",
            "agent_session_id",
            "agent_turn_id",
            "agent_revision",
            "revision",
            "reason",
            "basis",
            "completion_state",
            "selected_source",
            "outcome",
            "playback_kind",
            "numeric_value",
            "support",
            "boolean_value",
            "acoustic_endpoint",
            "vad_active",
            "early_endpoint_allowed",
            "trailing_silence_sec",
        ):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload


def _wav_header(sample_rate: int, data_bytes: int) -> bytes:
    return b"".join(
        (
            b"RIFF",
            struct.pack("<I", 36 + data_bytes),
            b"WAVE",
            b"fmt ",
            struct.pack(
                "<IHHIIHH",
                16,
                1,
                1,
                sample_rate,
                sample_rate * 2,
                2,
                16,
            ),
            b"data",
            struct.pack("<I", data_bytes),
        )
    )


def _exclusive_descriptor(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
    except BaseException:
        os.close(fd)
        raise
    return fd


class _SecureWavFile:
    def __init__(self, path: Path, sample_rate: int) -> None:
        self.path = path
        fd = _exclusive_descriptor(path)
        try:
            self.handle: Optional[BinaryIO] = os.fdopen(fd, "w+b")
        except BaseException:
            os.close(fd)
            raise
        self.sample_rate = sample_rate
        self.samples = 0
        self.data_bytes = 0
        self.handle.write(_wav_header(sample_rate, 0))

    def write(self, samples: np.ndarray) -> None:
        if self.handle is None:
            raise RuntimeError("WAV is closed")
        pcm = np.clip(samples, -1.0, 1.0)
        payload = (pcm * 32767.0).astype("<i2", copy=False).tobytes()
        self.handle.write(payload)
        self.samples += int(samples.size)
        self.data_bytes += len(payload)

    def flush(self, *, sync: bool) -> None:
        if self.handle is None:
            return
        self.handle.seek(4)
        self.handle.write(struct.pack("<I", 36 + self.data_bytes))
        self.handle.seek(40)
        self.handle.write(struct.pack("<I", self.data_bytes))
        self.handle.seek(0, 2)
        self.handle.flush()
        if sync:
            os.fsync(self.handle.fileno())

    def close(self) -> None:
        if self.handle is None:
            return
        try:
            self.flush(sync=True)
        finally:
            self.handle.close()
            self.handle = None


@dataclass(frozen=True, slots=True)
class _FrameItem:
    span: BundleAudioSpan
    tracks: Mapping[DiagnosticTrack, np.ndarray]
    track_ranges: Mapping[DiagnosticTrack, tuple[int, int]]


@dataclass(frozen=True, slots=True)
class _ObservationItem:
    observation: DiagnosticObservation


_STOP = object()


class SynchronizedDiagnosticBundle:
    """One bounded writer for capture-correlated PCM and typed observations."""

    def __init__(
        self,
        path_by_role: Mapping[DiagnosticTrack, os.PathLike[str] | str],
        timeline_path: os.PathLike[str] | str,
        manifest_path: os.PathLike[str] | str,
        sample_rate: int,
        endpoint_replay_config: Optional[EndpointReplayConfig] = None,
        queue_max: int = 4096,
        flush_sec: float = 2.0,
        close_timeout_sec: float = 5.0,
    ) -> None:
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate <= 0
        ):
            raise ValueError("sample_rate must be a positive integer")
        if (
            isinstance(queue_max, bool)
            or not isinstance(queue_max, int)
            or queue_max <= 0
        ):
            raise ValueError("queue_max must be a positive integer")
        if (
            isinstance(flush_sec, bool)
            or not isinstance(flush_sec, (int, float))
            or not math.isfinite(float(flush_sec))
            or float(flush_sec) <= 0.0
        ):
            raise ValueError("flush_sec must be a finite positive number")
        if (
            isinstance(close_timeout_sec, bool)
            or not isinstance(close_timeout_sec, (int, float))
            or not math.isfinite(float(close_timeout_sec))
            or float(close_timeout_sec) <= 0.0
        ):
            raise ValueError("close_timeout_sec must be a finite positive number")
        if endpoint_replay_config is None:
            endpoint_replay_config = EndpointReplayConfig()
        if not isinstance(endpoint_replay_config, EndpointReplayConfig):
            raise TypeError("endpoint_replay_config must be EndpointReplayConfig")

        normalized: dict[DiagnosticTrack, Path] = {}
        for role, raw_path in path_by_role.items():
            if not isinstance(role, DiagnosticTrack):
                raise TypeError("path_by_role keys must be DiagnosticTrack values")
            if role in normalized:
                raise ValueError("duplicate diagnostic track role")
            normalized[role] = _absolute_path(raw_path)
        if set(normalized) != set(DiagnosticTrack):
            raise ValueError("path_by_role must contain every diagnostic track")

        self.timeline_path = _absolute_path(timeline_path)
        self.manifest_path = _absolute_path(manifest_path)
        parent = self.manifest_path.parent
        all_paths = [*normalized.values(), self.timeline_path, self.manifest_path]
        if any(path.parent != parent for path in all_paths):
            raise ValueError("all bundle artifacts must share the manifest directory")
        if len({path.name for path in all_paths}) != len(all_paths):
            raise ValueError("bundle artifact paths must be unique")
        for path in all_paths:
            if os.path.lexists(path):
                raise FileExistsError(path)

        self.path_by_role = dict(normalized)
        self.sample_rate = sample_rate
        self.endpoint_replay_config = endpoint_replay_config
        self._flush_sec = float(flush_sec)
        self._close_timeout_sec = float(close_timeout_sec)
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_max)
        self._state_lock = threading.Lock()
        self._writer_stop_requested = threading.Event()
        self._close_done = threading.Event()
        self._close_started = False
        self._close_clean_shutdown = True
        self._close_error: Optional[BaseException] = None
        self._failure_codes: set[str] = set()
        self._accepting = True
        self._closed = False
        self._next_frame_index = 0
        self._next_sample_by_role = {role: 0 for role in DiagnosticTrack}
        self._receipt_token = object()
        self._written_frames = 0
        self._written_samples_by_role = {role: 0 for role in DiagnosticTrack}
        self._manifest_payload: Optional[dict[str, object]] = None
        self._manifest_status = DiagnosticManifestStatus.OPEN
        self._last_flush = time.monotonic()

        self._wav_files: dict[DiagnosticTrack, _SecureWavFile] = {}
        self._timeline: Optional[BinaryIO] = None
        try:
            for role in DiagnosticTrack:
                self._wav_files[role] = _SecureWavFile(
                    self.path_by_role[role], sample_rate
                )
            fd = _exclusive_descriptor(self.timeline_path)
            try:
                self._timeline = os.fdopen(fd, "w+b")
            except BaseException:
                os.close(fd)
                raise
            self._write_timeline_payload(
                {
                    "kind": "header",
                    "schema_version": SCHEMA_VERSION,
                    "sample_rate_hz": sample_rate,
                    "tracks": [role.value for role in DiagnosticTrack],
                    "endpoint_replay_config": endpoint_replay_config.to_payload(),
                }
            )
            self._flush_outputs(sync=True)
        except BaseException:
            self._close_open_outputs()
            raise

        self._thread = threading.Thread(
            target=self._writer_loop,
            name="synchronized-diagnostic-bundle",
            daemon=True,
        )
        self._thread.start()

    @property
    def complete(self) -> bool:
        with self._state_lock:
            return self._manifest_status is DiagnosticManifestStatus.COMPLETE

    @property
    def paths(self) -> Mapping[str, Path]:
        """Read-only basename-role view of every artifact path."""

        values = {role.value: self.path_by_role[role] for role in DiagnosticTrack}
        values["timeline"] = self.timeline_path
        values["manifest"] = self.manifest_path
        return MappingProxyType(values)

    @property
    def manifest_status(self) -> DiagnosticManifestStatus:
        with self._state_lock:
            return self._manifest_status

    @property
    def failure_codes(self) -> tuple[str, ...]:
        with self._state_lock:
            return tuple(sorted(self._failure_codes))

    def _mark_incomplete(self, code: str) -> None:
        with self._state_lock:
            self._failure_codes.add(code)

    def invalidate(self, code: str) -> None:
        """Fail the bundle closed for one controlled integration condition.

        The public seam intentionally accepts only a closed vocabulary.  It is
        safe to call from error handlers because exception strings, paths, and
        device labels can never become persisted diagnostic metadata.
        """

        if code not in _EXTERNAL_FAILURE_CODES:
            raise ValueError("unknown diagnostic invalidation code")
        with self._state_lock:
            if not self._accepting:
                raise RuntimeError("diagnostic bundle is finalizing or finalized")
            self._failure_codes.add(code)

    def write_frame(
        self,
        tracks: Mapping[DiagnosticTrack, np.ndarray],
        coordinate: CaptureCoordinate,
    ) -> Optional[BundleAudioSpan]:
        """Admit one capture-coordinate frame for every track, or none of it.

        Stateful resamplers and frame-based DSP may emit different sample
        counts for the same native block. Each role advances on its own packed
        cursor; the returned observation span uses the model-gate track.
        """

        if not isinstance(coordinate, CaptureCoordinate):
            raise TypeError("coordinate must be a CaptureCoordinate")
        if not isinstance(tracks, Mapping):
            raise TypeError("tracks must be a mapping")

        normalized: dict[DiagnosticTrack, np.ndarray] = {}
        try:
            for role, samples in tracks.items():
                if not isinstance(role, DiagnosticTrack):
                    raise TypeError
                if role in normalized:
                    raise ValueError
                array = np.asarray(samples)
                if array.dtype != np.dtype("float32") or array.ndim != 1:
                    raise ValueError
                if array.size <= 0 or not bool(np.all(np.isfinite(array))):
                    raise ValueError
                normalized[role] = np.array(array, dtype="float32", copy=True)
        except (TypeError, ValueError):
            self._mark_incomplete("invalid_track")
            return None
        if set(normalized) != set(DiagnosticTrack):
            self._mark_incomplete("missing_track")
            return None
        with self._state_lock:
            if not self._accepting:
                raise RuntimeError("diagnostic bundle is closed")
            track_ranges = {
                role: (
                    self._next_sample_by_role[role],
                    self._next_sample_by_role[role] + int(normalized[role].size),
                )
                for role in DiagnosticTrack
            }
            gate_start, gate_end = track_ranges[DiagnosticTrack.MODEL_GATE_PCM]
            span = BundleAudioSpan(
                frame_index=self._next_frame_index,
                bundle_sample_start=gate_start,
                bundle_sample_end=gate_end,
                coordinate=coordinate,
                _bundle_token=self._receipt_token,
            )
            item = _FrameItem(
                span=span,
                tracks=normalized,
                track_ranges=track_ranges,
            )
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                self._failure_codes.add("queue_drop")
                return None
            self._next_frame_index += 1
            for role, (_start, end) in track_ranges.items():
                self._next_sample_by_role[role] = end
            return span

    def observe(self, observation: DiagnosticObservation) -> bool:
        """Admit a typed semantic observation to the same ordered writer."""

        if not isinstance(observation, DiagnosticObservation):
            raise TypeError("observation must be a DiagnosticObservation")
        with self._state_lock:
            if not self._accepting:
                raise RuntimeError("diagnostic bundle is closed")
            if observation.span is not None:
                span = observation.span
                if (
                    span._bundle_token is not self._receipt_token
                    or span.frame_index >= self._next_frame_index
                    or span.bundle_sample_end
                    > self._next_sample_by_role[DiagnosticTrack.MODEL_GATE_PCM]
                ):
                    self._failure_codes.add("foreign_span")
                    return False
            try:
                self._queue.put_nowait(_ObservationItem(observation))
            except queue.Full:
                self._failure_codes.add("observation_drop")
                return False
            return True

    def _writer_loop(self) -> None:
        try:
            while True:
                if self._writer_stop_requested.is_set() and self._queue.empty():
                    break
                try:
                    item = self._queue.get(timeout=min(0.1, self._flush_sec))
                except queue.Empty:
                    self._periodic_flush()
                    continue
                try:
                    if item is _STOP:
                        break
                    if isinstance(item, _FrameItem):
                        self._write_frame_item(item)
                    elif isinstance(item, _ObservationItem):
                        self._write_timeline_payload(item.observation.to_payload())
                    else:
                        self._mark_incomplete("write_error")
                    self._periodic_flush()
                except BaseException:  # an evidence failure must not kill capture
                    self._mark_incomplete("write_error")
                finally:
                    self._queue.task_done()
            with self._state_lock:
                clean_shutdown = self._close_clean_shutdown
            self._finalize_outputs(clean_shutdown=clean_shutdown)
        except BaseException as exc:
            with self._state_lock:
                if "manifest_publish_error" not in self._failure_codes:
                    self._failure_codes.add("finalization_error")
                self._accepting = False
                self._close_started = True
                self._closed = True
                self._close_error = exc
                self._manifest_status = DiagnosticManifestStatus.PUBLISH_FAILED
        finally:
            self._close_open_outputs()
            self._close_done.set()

    def _write_frame_item(self, item: _FrameItem) -> None:
        for role in DiagnosticTrack:
            self._wav_files[role].write(item.tracks[role])
        self._write_timeline_payload(
            {
                "kind": "frame",
                "frame_index": item.span.frame_index,
                "bundle_sample_start": item.span.bundle_sample_start,
                "bundle_sample_end": item.span.bundle_sample_end,
                "track_ranges": {
                    role.value: {
                        "sample_start": item.track_ranges[role][0],
                        "sample_end": item.track_ranges[role][1],
                    }
                    for role in DiagnosticTrack
                },
                "coordinate": item.span.coordinate.to_payload(),
            }
        )
        self._written_frames += 1
        for role in DiagnosticTrack:
            start, end = item.track_ranges[role]
            self._written_samples_by_role[role] += end - start

    def _write_timeline_payload(self, payload: Mapping[str, object]) -> None:
        if self._timeline is None:
            raise RuntimeError("timeline is closed")
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        if len(encoded) > _TIMELINE_LINE_MAX_BYTES:
            raise ValueError("timeline record is too large")
        self._timeline.write(encoded)

    def _periodic_flush(self) -> None:
        if time.monotonic() - self._last_flush < self._flush_sec:
            return
        try:
            self._flush_outputs(sync=True)
        except BaseException:
            self._mark_incomplete("flush_error")
        finally:
            # A persistent device/filesystem error must fail the evidence closed,
            # but it must not turn the single writer into a tight fsync retry loop.
            self._last_flush = time.monotonic()

    def _flush_outputs(self, *, sync: bool) -> None:
        for wav_file in self._wav_files.values():
            wav_file.flush(sync=sync)
        if self._timeline is not None:
            self._timeline.flush()
            if sync:
                os.fsync(self._timeline.fileno())
        self._last_flush = time.monotonic()

    def _close_open_outputs(self) -> None:
        for wav_file in self._wav_files.values():
            try:
                wav_file.close()
            except BaseException:
                pass
        if self._timeline is not None:
            try:
                self._timeline.flush()
                os.fsync(self._timeline.fileno())
            except BaseException:
                pass
            try:
                self._timeline.close()
            except BaseException:
                pass
            self._timeline = None

    def close(self, clean_shutdown: bool = True) -> Path:
        """Request single-owner finalization and wait only to this call's budget.

        The writer thread owns file handles through publication. A slow fsync,
        hash, or filesystem therefore leaves the honest state ``FINALIZING``;
        it can never block capture/runtime shutdown or race a caller closing the
        same handle.
        """

        deadline = time.monotonic() + self._close_timeout_sec
        with self._state_lock:
            if not self._close_started:
                self._close_started = True
                self._close_clean_shutdown = bool(clean_shutdown)
                self._accepting = False
                if not clean_shutdown:
                    self._failure_codes.add("unclean_shutdown")
                self._manifest_status = DiagnosticManifestStatus.FINALIZING
                self._writer_stop_requested.set()

        remaining = max(0.0, deadline - time.monotonic())
        finished = self._close_done.wait(remaining)
        if finished:
            with self._state_lock:
                error = self._close_error
            if error is not None:
                raise error
        return self.manifest_path

    def _finalize_outputs(self, *, clean_shutdown: bool) -> None:
        """Finalize and publish on the sole thread that owns output handles."""

        for wav_file in self._wav_files.values():
            try:
                wav_file.close()
            except BaseException:
                self._mark_incomplete("close_error")

        if self._written_frames == 0:
            with self._state_lock:
                if not self._failure_codes:
                    self._failure_codes.add("no_frames")
        actual_counts = {
            role: wav_file.samples for role, wav_file in self._wav_files.items()
        }
        if actual_counts != self._written_samples_by_role:
            self._mark_incomplete("track_mismatch")
        with self._state_lock:
            provisional_complete = clean_shutdown and not self._failure_codes
            footer_failures = sorted(self._failure_codes)
        try:
            self._write_timeline_payload(
                {
                    "kind": "footer",
                    "complete": provisional_complete,
                    "clean_shutdown": bool(clean_shutdown),
                    "frame_count": self._written_frames,
                    "sample_count": self._written_samples_by_role[
                        DiagnosticTrack.MODEL_GATE_PCM
                    ],
                    "sample_count_by_track": {
                        role.value: self._written_samples_by_role[role]
                        for role in DiagnosticTrack
                    },
                    "failure_codes": footer_failures,
                }
            )
            assert self._timeline is not None
            self._timeline.flush()
            os.fsync(self._timeline.fileno())
            self._timeline.close()
            self._timeline = None
        except BaseException:
            self._mark_incomplete("close_error")
            if self._timeline is not None:
                try:
                    self._timeline.close()
                except BaseException:
                    pass
                self._timeline = None

        if provisional_complete and not _validate_timeline(
            self.timeline_path,
            frame_count=self._written_frames,
            sample_count=self._written_samples_by_role[
                DiagnosticTrack.MODEL_GATE_PCM
            ],
            track_sample_counts={
                role.value: self._written_samples_by_role[role]
                for role in DiagnosticTrack
            },
            sample_rate_hz=self.sample_rate,
            endpoint_replay_config=self.endpoint_replay_config,
        ):
            self._mark_incomplete("timeline_validation_error")

        artifacts: dict[str, dict[str, object]] = {}
        timeline_artifact: dict[str, object] = {}
        try:
            timeline_digest, timeline_bytes = _hash_private_regular(
                self.timeline_path
            )
            timeline_artifact = {
                "file": self.timeline_path.name,
                "sha256": timeline_digest,
                "bytes": timeline_bytes,
            }
            for role in DiagnosticTrack:
                path = self.path_by_role[role]
                digest, size = _hash_private_regular(path)
                artifacts[role.value] = {
                    "file": path.name,
                    "sha256": digest,
                    "bytes": size,
                    "samples": self._wav_files[role].samples,
                }
        except BaseException:
            self._mark_incomplete("hash_error")

        with self._state_lock:
            complete = clean_shutdown and not self._failure_codes
            manifest: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "complete": complete,
                "clean_shutdown": bool(clean_shutdown),
                "provenance": {
                    "native_host_pcm_present": False,
                    "physical_raw_mic_present": False,
                    "model_pre_gain_tap_may_include_host_processing": True,
                    "model_pre_gain_tap_includes_application_resampling": True,
                    "audio_storage_is_pcm16_quantized": True,
                    "model_asr_track_is_continuous_selected_tap": True,
                    "recognizer_reset_replay_receipts_present": False,
                    "turn_selection_ranges_present": False,
                    "frame_track_lengths_may_differ": True,
                    "observation_spans_use_model_gate_track": True,
                    "playback_reference_preserves_reader_snapshot": True,
                    "playback_timestamps_are_receipt_dispatch_time": True,
                    "physical_playback_audibility_present": False,
                },
                "sample_rate_hz": self.sample_rate,
                "endpoint_replay_config": self.endpoint_replay_config.to_payload(),
                "frame_count": self._written_frames,
                "sample_count": self._written_samples_by_role[
                    DiagnosticTrack.MODEL_GATE_PCM
                ],
                "failure_codes": sorted(self._failure_codes),
                "timeline": timeline_artifact,
                "tracks": artifacts,
            }
            self._manifest_payload = manifest

        try:
            _publish_manifest(self.manifest_path, manifest)
        except BaseException as exc:
            # A publisher may report a late cleanup error after its atomic
            # no-clobber commit. The exact private payload is authoritative in
            # that narrow case; a collision remains a hard failure.
            committed = (
                not isinstance(exc, FileExistsError)
                and _read_manifest(self.manifest_path) == manifest
            )
            if not committed:
                self._mark_incomplete("manifest_publish_error")
                raise
        with self._state_lock:
            self._closed = True
            self._manifest_status = (
                DiagnosticManifestStatus.COMPLETE
                if complete
                else DiagnosticManifestStatus.INCOMPLETE
            )

    def __enter__(self) -> "SynchronizedDiagnosticBundle":
        return self

    def __exit__(self, exc_type, _exc, _tb) -> None:
        if exc_type is None:
            self.close(clean_shutdown=True)
            return
        try:
            self.close(clean_shutdown=False)
        except BaseException:
            # Preserve the application exception already unwinding. The bundle
            # retains PUBLISH_FAILED plus the stored close error for inspection.
            pass


def _absolute_path(path: os.PathLike[str] | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _publish_manifest(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary: Optional[Path] = None
    fd = -1
    for _attempt in range(16):
        candidate = path.with_name(
            f".{path.name}.{secrets.token_hex(16)}.tmp"
        )
        try:
            fd = _exclusive_descriptor(candidate)
        except FileExistsError:
            continue
        temporary = candidate
        break
    if temporary is None or fd < 0:
        raise FileExistsError("could not reserve a private manifest temporary")
    committed = False
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == "nt":  # Windows rename is no-clobber when dst exists.
            os.rename(temporary, path)
        else:
            os.link(temporary, path, follow_symlinks=False)
        committed = True
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                # A failed fdopen already supplies the publication error. Keep
                # attempting independent cleanup so the private temp is not
                # stranded merely because descriptor cleanup also failed.
                pass
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
        if committed:
            # Publication already committed a complete inode. Directory fsync
            # and temporary cleanup are durability hygiene, never grounds for
            # contradicting the now-visible exact manifest.
            directory_fd = -1
            try:
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_DIRECTORY", 0)
                directory_fd = os.open(path.parent, flags)
                os.fsync(directory_fd)
            except OSError:
                pass
            finally:
                if directory_fd >= 0:
                    os.close(directory_fd)


def _hash_private_regular(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    digest = hashlib.sha256()
    size = 0
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("artifact must be a private regular file")
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    finally:
        os.close(fd)
    return digest.hexdigest(), size


def _duplicate_safe_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _read_manifest(path: Path) -> Optional[dict[str, object]]:
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            return None
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(65536, _MANIFEST_MAX_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MANIFEST_MAX_BYTES:
                return None
        parsed = json.loads(
            b"".join(chunks).decode("utf-8"),
            object_pairs_hook=_duplicate_safe_object,
        )
        return parsed if isinstance(parsed, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    finally:
        os.close(fd)


def _valid_artifact_entry(value: object, *, track: bool) -> bool:
    if not isinstance(value, dict):
        return False
    expected = {"file", "sha256", "bytes"} | ({"samples"} if track else set())
    if set(value) != expected:
        return False
    filename = value.get("file")
    digest = value.get("sha256")
    size = value.get("bytes")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or filename in {".", ".."}
        or not isinstance(digest, str)
        or _HASH_RE.fullmatch(digest) is None
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
    ):
        return False
    if track:
        samples = value.get("samples")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0:
            return False
    return True


def _validate_timeline(
    path: Path,
    *,
    frame_count: int,
    sample_count: int,
    track_sample_counts: Mapping[str, int],
    sample_rate_hz: int,
    endpoint_replay_config: EndpointReplayConfig,
) -> bool:
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
    except OSError:
        return False
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            return False
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            lines = handle.readlines()
        if len(lines) < 2 or any(
            not line.endswith(b"\n") or len(line) > _TIMELINE_LINE_MAX_BYTES
            for line in lines
        ):
            return False
        records: list[dict[str, object]] = []
        for line in lines:
            record = json.loads(
                line.decode("utf-8"), object_pairs_hook=_duplicate_safe_object
            )
            if not isinstance(record, dict):
                return False
            records.append(record)
        header = records[0]
        if header != {
            "kind": "header",
            "schema_version": SCHEMA_VERSION,
            "sample_rate_hz": sample_rate_hz,
            "tracks": [role.value for role in DiagnosticTrack],
            "endpoint_replay_config": endpoint_replay_config.to_payload(),
        }:
            return False
        footer = records[-1]
        if set(footer) != {
            "kind",
            "complete",
            "clean_shutdown",
            "frame_count",
            "sample_count",
            "sample_count_by_track",
            "failure_codes",
        }:
            return False
        if (
            footer.get("kind") != "footer"
            or footer.get("complete") is not True
            or footer.get("clean_shutdown") is not True
            or footer.get("frame_count") != frame_count
            or footer.get("sample_count") != sample_count
            or footer.get("sample_count_by_track") != dict(track_sample_counts)
            or footer.get("failure_codes") != []
        ):
            return False

        seen_frames = 0
        next_sample_by_track = {role.value: 0 for role in DiagnosticTrack}
        gate_spans: dict[int, tuple[int, int]] = {}
        previous_coordinate: Optional[CaptureCoordinate] = None
        for record in records[1:-1]:
            kind = record.get("kind")
            if kind == "frame":
                if set(record) != {
                    "kind",
                    "frame_index",
                    "bundle_sample_start",
                    "bundle_sample_end",
                    "track_ranges",
                    "coordinate",
                }:
                    return False
                if record.get("frame_index") != seen_frames:
                    return False
                start = record.get("bundle_sample_start")
                end = record.get("bundle_sample_end")
                if (
                    isinstance(start, bool)
                    or not isinstance(start, int)
                    or isinstance(end, bool)
                    or not isinstance(end, int)
                    or end <= start
                ):
                    return False
                track_ranges = record.get("track_ranges")
                if not isinstance(track_ranges, dict) or set(track_ranges) != {
                    role.value for role in DiagnosticTrack
                }:
                    return False
                for role in DiagnosticTrack:
                    value = track_ranges.get(role.value)
                    if not isinstance(value, dict) or set(value) != {
                        "sample_start",
                        "sample_end",
                    }:
                        return False
                    role_start = value.get("sample_start")
                    role_end = value.get("sample_end")
                    if (
                        isinstance(role_start, bool)
                        or not isinstance(role_start, int)
                        or isinstance(role_end, bool)
                        or not isinstance(role_end, int)
                        or role_start != next_sample_by_track[role.value]
                        or role_end <= role_start
                    ):
                        return False
                    next_sample_by_track[role.value] = role_end
                gate_range = track_ranges[DiagnosticTrack.MODEL_GATE_PCM.value]
                if (
                    start != gate_range["sample_start"]
                    or end != gate_range["sample_end"]
                ):
                    return False
                coordinate = record.get("coordinate")
                if not isinstance(coordinate, dict):
                    return False
                try:
                    parsed_coordinate = CaptureCoordinate(**coordinate)
                except (TypeError, ValueError):
                    return False
                if previous_coordinate is not None:
                    same_generation = (
                        parsed_coordinate.capture_generation
                        == previous_coordinate.capture_generation
                    )
                    if parsed_coordinate.capture_epoch != previous_coordinate.capture_epoch:
                        return False
                    if same_generation:
                        if (
                            parsed_coordinate.source_generation
                            != previous_coordinate.source_generation
                            or parsed_coordinate.sequence
                            <= previous_coordinate.sequence
                            or parsed_coordinate.source_sample_start
                            != previous_coordinate.source_sample_end
                            or parsed_coordinate.gap_reason is not None
                        ):
                            return False
                    else:
                        if (
                            parsed_coordinate.gap_reason is None
                            or parsed_coordinate.gap_prior_generation
                            != previous_coordinate.capture_generation
                            or parsed_coordinate.gap_generation
                            != parsed_coordinate.capture_generation
                        ):
                            return False
                        if (
                            parsed_coordinate.source_generation
                            == previous_coordinate.source_generation
                            and parsed_coordinate.source_sample_start
                            < previous_coordinate.source_sample_end
                        ):
                            return False
                previous_coordinate = parsed_coordinate
                gate_spans[seen_frames] = (start, end)
                seen_frames += 1
            elif kind == "observation":
                if not _validate_observation_payload(record, gate_spans):
                    return False
            else:
                return False
        return bool(
            seen_frames == frame_count
            and next_sample_by_track == dict(track_sample_counts)
            and next_sample_by_track[DiagnosticTrack.MODEL_GATE_PCM.value]
            == sample_count
            and _validate_endpoint_replay(records, endpoint_replay_config)
            and _validate_lifecycle(records)
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ):
        return False
    finally:
        if fd >= 0:
            os.close(fd)


def _validate_observation_payload(
    record: Mapping[str, object], gate_spans: Mapping[int, tuple[int, int]]
) -> bool:
    allowed = {
        "kind",
        "stage",
        "monotonic_ns",
        "frame_index",
        "bundle_sample_start",
        "bundle_sample_end",
        "stream_id",
        "utterance_id",
        "correlation_id",
        "agent_session_id",
        "agent_turn_id",
        "agent_revision",
        "revision",
        "reason",
        "basis",
        "completion_state",
        "selected_source",
        "outcome",
        "playback_kind",
        "numeric_value",
        "support",
        "boolean_value",
        "acoustic_endpoint",
        "vad_active",
        "early_endpoint_allowed",
        "trailing_silence_sec",
    }
    if not set(record).issubset(allowed):
        return False
    try:
        stage = DiagnosticStage(record["stage"])
        monotonic_ns = record["monotonic_ns"]
        span = None
        span_fields = (
            "frame_index",
            "bundle_sample_start",
            "bundle_sample_end",
        )
        present = [name in record for name in span_fields]
        if any(present) and not all(present):
            return False
        if all(present):
            frame_index = record["frame_index"]
            start = record["bundle_sample_start"]
            end = record["bundle_sample_end"]
            if (
                isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or frame_index < 0
                or frame_index not in gate_spans
                or isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or start < 0
                or end <= start
                or (start, end) != gate_spans[frame_index]
            ):
                return False
            # The validator needs only the recorded range; a coordinate is not
            # duplicated on semantic events.
            span = (frame_index, start, end)
        _non_negative_int("monotonic_ns", monotonic_ns)  # type: ignore[arg-type]
        _opaque_id("stream_id", record.get("stream_id"))  # type: ignore[arg-type]
        _opaque_id("utterance_id", record.get("utterance_id"))  # type: ignore[arg-type]
        _opaque_id("correlation_id", record.get("correlation_id"))  # type: ignore[arg-type]
        _opaque_id("agent_session_id", record.get("agent_session_id"))  # type: ignore[arg-type]
        _opaque_id("agent_turn_id", record.get("agent_turn_id"))  # type: ignore[arg-type]
        _optional_non_negative_int("agent_revision", record.get("agent_revision"))  # type: ignore[arg-type]
        _optional_non_negative_int("revision", record.get("revision"))  # type: ignore[arg-type]
        for name in (
            "reason",
            "basis",
            "completion_state",
            "selected_source",
            "outcome",
            "playback_kind",
        ):
            _closed_symbol(name, record.get(name))  # type: ignore[arg-type]
        _optional_number("numeric_value", record.get("numeric_value"))  # type: ignore[arg-type]
        _optional_non_negative_int("support", record.get("support"))  # type: ignore[arg-type]
        boolean = record.get("boolean_value")
        if boolean is not None and not isinstance(boolean, bool):
            return False
        for name in (
            "acoustic_endpoint",
            "vad_active",
            "early_endpoint_allowed",
        ):
            value = record.get(name)
            if value is not None and not isinstance(value, bool):
                return False
        trailing = record.get("trailing_silence_sec")
        if trailing is not None:
            _finite_clock("trailing_silence_sec", trailing)  # type: ignore[arg-type]
        present_fields = {
            name
            for name in (
                "stream_id",
                "utterance_id",
                "correlation_id",
                "agent_session_id",
                "agent_turn_id",
                "agent_revision",
                "revision",
                "reason",
                "basis",
                "completion_state",
                "selected_source",
                "outcome",
                "playback_kind",
                "numeric_value",
                "support",
                "boolean_value",
                "acoustic_endpoint",
                "vad_active",
                "early_endpoint_allowed",
                "trailing_silence_sec",
            )
            if record.get(name) is not None
        }
        if span is not None:
            present_fields.add("span")
        _validate_stage_shape(stage, present_fields)
        return stage is not None and (span is None or span[0] in gate_spans)
    except (KeyError, TypeError, ValueError):
        return False


def _endpoint_commit_key(record: Mapping[str, object]) -> tuple[object, ...]:
    return (
        record.get("monotonic_ns"),
        record.get("frame_index"),
        record.get("bundle_sample_start"),
        record.get("bundle_sample_end"),
        record.get("reason"),
        record.get("basis"),
        record.get("completion_state"),
        record.get("numeric_value"),
    )


def _validate_endpoint_replay(
    records: list[dict[str, object]], config: EndpointReplayConfig
) -> bool:
    """Replay every persisted turn decision and bind true decisions to commits."""

    from .endpointing import (
        AdaptiveEndpointPolicy,
        CompletionScoreState,
        EndpointConfig,
        TurnObservation,
        decide_turn,
    )

    policy_config = EndpointConfig(
        enabled=config.enabled,
        min_silence_sec=config.min_silence_sec,
        max_silence_sec=config.max_silence_sec,
        complete_threshold=config.complete_threshold,
        incomplete_threshold=config.incomplete_threshold,
        high_confidence_floor=config.high_confidence_floor,
        high_confidence_score=config.high_confidence_score,
        adaptive_floor=config.adaptive_floor,
        pause_window=config.pause_window,
        pause_quantile=config.pause_quantile,
        pause_margin=config.pause_margin,
        pause_min_samples=config.pause_min_samples,
    )
    try:
        policy = AdaptiveEndpointPolicy(policy_config)
        policy.restore_replay_pause_samples(config.initial_pause_samples_sec)
        if policy.replay_pause_samples() != config.initial_pause_samples_sec:
            return False
    except (TypeError, ValueError):
        return False

    pending_commit: Optional[dict[str, object]] = None
    try:
        for record in records:
            if record.get("kind") != "observation":
                continue
            stage = DiagnosticStage(record["stage"])
            if stage is DiagnosticStage.ENDPOINT_PAUSE_OBSERVED:
                if not config.enabled or pending_commit is not None:
                    return False
                policy.observe_pause(float(record["numeric_value"]))
                continue
            if stage is DiagnosticStage.ENDPOINT_EVALUATED:
                if pending_commit is not None:
                    return False
                state = CompletionScoreState(record["completion_state"])
                score = record.get("numeric_value")
                if (state is CompletionScoreState.SCORED) != (score is not None):
                    return False
                observation = TurnObservation(
                    acoustic_endpoint=record["acoustic_endpoint"],  # type: ignore[arg-type]
                    vad_active=record["vad_active"],  # type: ignore[arg-type]
                    early_endpoint_allowed=record["early_endpoint_allowed"],  # type: ignore[arg-type]
                    trailing_silence_sec=float(record["trailing_silence_sec"]),
                    completion_state=state,
                    completion_score=None if score is None else float(score),
                )
                decision = decide_turn(
                    observation,
                    policy=policy if config.enabled else None,
                )
                if (
                    record.get("boolean_value") is not decision.commit
                    or record.get("reason") != decision.endpoint_reason.value
                    or record.get("basis") != decision.basis.value
                ):
                    return False
                if decision.commit:
                    pending_commit = record
                continue
            if stage is DiagnosticStage.ENDPOINT_COMMITTED:
                if (
                    pending_commit is None
                    or _endpoint_commit_key(record)
                    != _endpoint_commit_key(pending_commit)
                ):
                    return False
                for name in ("stream_id", "utterance_id"):
                    identity = pending_commit.get(name)
                    if identity is not None and record.get(name) != identity:
                        return False
                pending_commit = None
        return pending_commit is None
    except (KeyError, TypeError, ValueError):
        return False


def _validate_lifecycle(records: list[dict[str, object]]) -> bool:
    """Require paired terminals and immutable causal identity per lifecycle."""

    turns: dict[tuple[object, object, object], dict[str, bool]] = {}
    dispatched_turns: set[tuple[object, object, object]] = set()
    playbacks: dict[object, dict[str, object]] = {}
    for record in records:
        if record.get("kind") != "observation":
            continue
        try:
            stage = DiagnosticStage(record["stage"])
        except (KeyError, ValueError):
            return False
        turn_key = (
            record.get("stream_id"),
            record.get("utterance_id"),
            record.get("revision"),
        )
        if stage is DiagnosticStage.ENDPOINT_COMMITTED:
            if None in turn_key or turn_key in turns:
                return False
            turns[turn_key] = {
                "committed": True,
                "streaming_final": False,
                "queued": False,
                "started": False,
                "selected": False,
                "terminal": False,
            }
        elif stage is DiagnosticStage.ASR_STREAMING_FINAL:
            state = turns.get(turn_key)
            if state is None or state["streaming_final"] or state["terminal"]:
                return False
            state["streaming_final"] = True
        elif stage is DiagnosticStage.FINALIZER_QUEUED:
            state = turns.get(turn_key)
            if state is None or state["queued"] or state["terminal"]:
                return False
            state["queued"] = True
        elif stage is DiagnosticStage.FINALIZER_STARTED:
            state = turns.get(turn_key)
            if state is None or state["started"] or state["terminal"]:
                return False
            state["started"] = True
        elif stage is DiagnosticStage.FINAL_SELECTION:
            state = turns.get(turn_key)
            if (
                state is None
                or not state["started"]
                or state["selected"]
                or state["terminal"]
            ):
                return False
            state["selected"] = True
        elif stage is DiagnosticStage.FINAL_DISPATCHED:
            state = turns.get(turn_key)
            if (
                state is None
                or not state["selected"]
                or state["terminal"]
            ):
                return False
            state["terminal"] = True
            dispatched_turns.add(turn_key)
        elif stage is DiagnosticStage.FINAL_ABORTED:
            state = turns.get(turn_key)
            if state is None:
                # An acoustic turn may be abandoned before endpoint commit.
                turns[turn_key] = {
                    "committed": False,
                    "streaming_final": False,
                    "queued": False,
                    "started": False,
                    "selected": False,
                    "terminal": True,
                }
            elif state["terminal"]:
                return False
            else:
                state["terminal"] = True

        correlation = record.get("correlation_id")
        identity = (
            record.get("agent_session_id"),
            record.get("agent_turn_id"),
            record.get("agent_revision"),
        )
        acoustic_values = (
            record.get("stream_id"),
            record.get("utterance_id"),
            record.get("revision"),
        )
        acoustic_present = tuple(value is not None for value in acoustic_values)
        if stage is DiagnosticStage.PLAYBACK_ADMITTED:
            if correlation is None or correlation in playbacks:
                return False
            kind = record.get("playback_kind")
            if kind == "reply":
                if not all(acoustic_present):
                    return False
                acoustic_link: Optional[tuple[object, object, object]] = acoustic_values
            else:
                if any(acoustic_present):
                    return False
                acoustic_link = None
            playbacks[correlation] = {
                "started": False,
                "terminal": False,
                "identity": identity,
                "last_monotonic_ns": record.get("monotonic_ns"),
                "acoustic_link": acoustic_link,
            }
        elif stage is DiagnosticStage.PLAYBACK_STARTED:
            state = playbacks.get(correlation)
            if (
                state is None
                or state["started"]
                or state["terminal"]
                or state["identity"] != identity
                or record.get("monotonic_ns") < state["last_monotonic_ns"]  # type: ignore[operator]
            ):
                return False
            state["started"] = True
            state["last_monotonic_ns"] = record.get("monotonic_ns")
        elif stage is DiagnosticStage.PLAYBACK_TERMINAL:
            state = playbacks.get(correlation)
            if (
                state is None
                or state["terminal"]
                or state["identity"] != identity
                or record.get("monotonic_ns") < state["last_monotonic_ns"]  # type: ignore[operator]
            ):
                return False
            state["terminal"] = True
            state["last_monotonic_ns"] = record.get("monotonic_ns")

    return bool(
        all(state["terminal"] for state in turns.values())
        and all(state["terminal"] for state in playbacks.values())
        and all(
            state["acoustic_link"] is None
            or state["acoustic_link"] in dispatched_turns
            for state in playbacks.values()
        )
    )


def validate_manifest(manifest_path: os.PathLike[str] | str) -> bool:
    """Return ``True`` only for a complete, private, internally bound bundle."""

    path = _absolute_path(manifest_path)
    manifest = _read_manifest(path)
    if manifest is None:
        return False
    if set(manifest) != {
        "schema_version",
        "complete",
        "clean_shutdown",
        "provenance",
        "sample_rate_hz",
        "endpoint_replay_config",
        "frame_count",
        "sample_count",
        "failure_codes",
        "timeline",
        "tracks",
    }:
        return False
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("complete") is not True
        or manifest.get("clean_shutdown") is not True
        or manifest.get("failure_codes") != []
        or manifest.get("provenance")
        != {
            "native_host_pcm_present": False,
            "physical_raw_mic_present": False,
            "model_pre_gain_tap_may_include_host_processing": True,
            "model_pre_gain_tap_includes_application_resampling": True,
            "audio_storage_is_pcm16_quantized": True,
            "model_asr_track_is_continuous_selected_tap": True,
            "recognizer_reset_replay_receipts_present": False,
            "turn_selection_ranges_present": False,
            "frame_track_lengths_may_differ": True,
            "observation_spans_use_model_gate_track": True,
            "playback_reference_preserves_reader_snapshot": True,
            "playback_timestamps_are_receipt_dispatch_time": True,
            "physical_playback_audibility_present": False,
        }
    ):
        return False
    sample_rate = manifest.get("sample_rate_hz")
    try:
        endpoint_replay_config = EndpointReplayConfig.from_payload(
            manifest.get("endpoint_replay_config")
        )
    except (TypeError, ValueError):
        return False
    frame_count = manifest.get("frame_count")
    sample_count = manifest.get("sample_count")
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, int)
        or sample_rate <= 0
        or isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
    ):
        return False

    timeline = manifest.get("timeline")
    tracks = manifest.get("tracks")
    if not _valid_artifact_entry(timeline, track=False) or not isinstance(
        tracks, dict
    ):
        return False
    if set(tracks) != {role.value for role in DiagnosticTrack}:
        return False
    if any(not _valid_artifact_entry(value, track=True) for value in tracks.values()):
        return False
    track_sample_counts = {
        role.value: tracks[role.value]["samples"] for role in DiagnosticTrack
    }
    if sample_count != track_sample_counts[DiagnosticTrack.MODEL_GATE_PCM.value]:
        return False
    entries = [timeline, *tracks.values()]
    names = [entry["file"] for entry in entries]  # type: ignore[index]
    if len(names) != len(set(names)) or path.name in names:
        return False

    parent = path.parent
    for entry in entries:
        artifact = parent / entry["file"]  # type: ignore[index,operator]
        try:
            digest, size = _hash_private_regular(artifact)
        except (OSError, ValueError):
            return False
        if digest != entry["sha256"] or size != entry["bytes"]:  # type: ignore[index]
            return False

    timeline_path = parent / timeline["file"]  # type: ignore[index,operator]
    if not _validate_timeline(
        timeline_path,
        frame_count=frame_count,
        sample_count=sample_count,
        track_sample_counts=track_sample_counts,
        sample_rate_hz=sample_rate,
        endpoint_replay_config=endpoint_replay_config,
    ):
        return False

    for role in DiagnosticTrack:
        entry = tracks[role.value]
        wav_path = parent / entry["file"]
        try:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(wav_path, flags)
            try:
                info = os.fstat(fd)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or stat.S_IMODE(info.st_mode) != 0o600
                ):
                    return False
                with os.fdopen(fd, "rb") as handle:
                    fd = -1
                    with wave.open(handle, "rb") as recording:
                        if (
                            recording.getnchannels() != 1
                            or recording.getsampwidth() != 2
                            or recording.getframerate() != sample_rate
                            or recording.getnframes() != entry["samples"]
                        ):
                            return False
            finally:
                if fd >= 0:
                    os.close(fd)
        except (OSError, EOFError, wave.Error):
            return False
    return True


__all__ = [
    "BundleAudioSpan",
    "CaptureCoordinate",
    "DiagnosticManifestStatus",
    "DiagnosticObservation",
    "DiagnosticStage",
    "DiagnosticTrack",
    "EndpointReplayConfig",
    "SynchronizedDiagnosticBundle",
    "validate_manifest",
]
