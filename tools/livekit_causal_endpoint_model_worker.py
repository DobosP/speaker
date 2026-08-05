#!/usr/bin/env python3
"""Bounded, descriptor-bound Smart Turn causal evaluation worker.

The supervisor passes already-open descriptors for this worker, the production
endpointing module and its two small local dependencies, the exact ONNX model,
the private materialization directory, a private request, and a private result.
Project code is executed only from those verified descriptor bytes.  The worker
has no network client and emits aggregate JSON through the supplied descriptor.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
import re
import stat
import sys
from types import ModuleType
from typing import Callable, Mapping, Optional, Sequence

PROTOCOL_VERSION = 1
SAMPLE_RATE_HZ = 16_000
GRID_SAMPLES = 1_600
PARTIAL_ASSUMPTION = "assumed_nonempty_current_epoch_at_every_scored_span"
NO_PARTIAL_PROFILE = "no_current_epoch_partial_acoustic_fallback"
_OPAQUE_PARTIAL_SENTINEL = "__current_epoch_partial_available__"
_MAX_REQUEST_BYTES = 2 * 1024 * 1024
_MAX_RESULT_BYTES = 128 * 1024
_MAX_CODE_BYTES = 1024 * 1024
_MAX_PCM_BYTES = 32 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_REQUEST_FIELDS = {
    "schema_version",
    "worker_descriptor",
    "endpointing_descriptor",
    "acoustic_descriptor",
    "text_descriptor",
    "model_descriptor",
    "materialization_descriptor",
    "result_descriptor",
    "code_sha256",
    "config_sha256",
    "model_size_bytes",
    "model_sha256",
    "materialization_manifest_sha256",
    "materialization_sha256",
    "pcm_set_sha256",
    "materialization_directory_identity",
    "materialization_leaf_identities",
    "endpoint_config",
    "acoustic_rule2_samples",
    "prosody_min_samples",
    "max_wait_samples",
    "rule3_samples",
    "expected_hold_denominator",
    "expected_eot_denominator",
    "rows",
}
_CODE_NAMES = ("worker", "endpointing", "acoustic", "text")
_ENDPOINT_CONFIG_FIELDS = {
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
}


class ModelWorkerError(RuntimeError):
    """A detail-free worker contract failure."""


@dataclass(frozen=True, slots=True)
class CausalSpan:
    start_sample: int
    end_sample: int

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample


@dataclass(frozen=True, slots=True)
class MaterializedRow:
    ordinal: int
    pcm_filename: str
    pcm_bytes: int
    pcm_samples: int
    pcm_sha256: str
    silence_spans: tuple[CausalSpan, ...]


@dataclass(frozen=True, slots=True)
class PolicyContract:
    endpoint_config: object
    acoustic_rule2_samples: int
    prosody_min_samples: int
    max_wait_samples: int
    rule3_samples: int


def _canonical_json_bytes(value: object) -> bytes:
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
    except (TypeError, UnicodeError, ValueError, OverflowError):
        raise ModelWorkerError() from None


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, OverflowError):
        raise ModelWorkerError() from None


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _stable_file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        stat.S_IMODE(info.st_mode),
        info.st_uid,
        info.st_nlink,
    )


def _stable_directory_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        stat.S_IMODE(info.st_mode),
        info.st_uid,
    )


def _pread_exact(descriptor: int, size: int, maximum: int) -> bytes:
    if size < 0 or size > maximum:
        raise ModelWorkerError()
    chunks = bytearray()
    try:
        while len(chunks) < size:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, size - len(chunks)),
                len(chunks),
            )
            if not chunk:
                break
            chunks.extend(chunk)
    except OSError:
        raise ModelWorkerError() from None
    if len(chunks) != size:
        raise ModelWorkerError()
    return bytes(chunks)


def _bound_payload(
    descriptor: int,
    *,
    maximum: int,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    private: bool,
) -> tuple[bytes, tuple[int, ...]]:
    try:
        before = os.fstat(descriptor)
    except OSError:
        raise ModelWorkerError() from None
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size < 0
        or before.st_size > maximum
        or (expected_size is not None and before.st_size != expected_size)
        or (private and stat.S_IMODE(before.st_mode) & 0o077)
        or (private and hasattr(os, "geteuid") and before.st_uid != os.geteuid())
    ):
        raise ModelWorkerError()
    payload = _pread_exact(descriptor, before.st_size, maximum)
    try:
        after = os.fstat(descriptor)
    except OSError:
        raise ModelWorkerError() from None
    if _identity(after) != _identity(before):
        raise ModelWorkerError()
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ModelWorkerError()
    return payload, _identity(before)


def _descriptor(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 3:
        raise ModelWorkerError()
    return value


def _load_request(descriptor: int) -> tuple[Mapping[str, object], str]:
    payload, _ = _bound_payload(
        descriptor,
        maximum=_MAX_REQUEST_BYTES,
        private=True,
    )
    value = _strict_json(payload)
    if not isinstance(value, dict) or set(value) != _REQUEST_FIELDS:
        raise ModelWorkerError()
    schema_version = value.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != PROTOCOL_VERSION
    ):
        raise ModelWorkerError()
    for name in (
        "worker_descriptor",
        "endpointing_descriptor",
        "acoustic_descriptor",
        "text_descriptor",
        "model_descriptor",
        "materialization_descriptor",
        "result_descriptor",
    ):
        _descriptor(value.get(name))
    code_sha256 = value.get("code_sha256")
    if not isinstance(code_sha256, dict) or set(code_sha256) != set(_CODE_NAMES):
        raise ModelWorkerError()
    for digest in (
        *code_sha256.values(),
        value.get("model_sha256"),
        value.get("config_sha256"),
        value.get("materialization_manifest_sha256"),
        value.get("materialization_sha256"),
        value.get("pcm_set_sha256"),
    ):
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ModelWorkerError()
    return value, hashlib.sha256(payload).hexdigest()


def _exec_project_modules(
    request: Mapping[str, object],
) -> tuple[object, Mapping[str, tuple[int, ...]]]:
    digests = request["code_sha256"]
    if not isinstance(digests, dict):
        raise ModelWorkerError()
    payloads: dict[str, bytes] = {}
    identities: dict[str, tuple[int, ...]] = {}
    for name in _CODE_NAMES:
        descriptor = _descriptor(request[f"{name}_descriptor"])
        payloads[name], identities[name] = _bound_payload(
            descriptor,
            maximum=_MAX_CODE_BYTES,
            expected_sha256=str(digests[name]),
            private=True,
        )

    always_on_agent = ModuleType("always_on_agent")
    always_on_agent.__path__ = []  # type: ignore[attr-defined]
    sys.modules["always_on_agent"] = always_on_agent
    for short_name in ("acoustic", "text"):
        full_name = f"always_on_agent.{short_name}"
        module = ModuleType(full_name)
        module.__file__ = f"<verified-fd:{short_name}>"
        module.__package__ = "always_on_agent"
        sys.modules[full_name] = module
        try:
            exec(  # noqa: S102 - exact verified descriptor bytes are the contract
                compile(payloads[short_name], module.__file__, "exec"),
                module.__dict__,
            )
        except Exception:
            raise ModelWorkerError() from None

    core = ModuleType("core")
    core.__path__ = []  # type: ignore[attr-defined]
    sys.modules["core"] = core
    endpointing = ModuleType("core.endpointing")
    endpointing.__file__ = "<verified-fd:endpointing>"
    endpointing.__package__ = "core"
    sys.modules["core.endpointing"] = endpointing
    try:
        exec(  # noqa: S102 - exact verified descriptor bytes are the contract
            compile(payloads["endpointing"], endpointing.__file__, "exec"),
            endpointing.__dict__,
        )
    except Exception:
        raise ModelWorkerError() from None
    return endpointing, identities


def _rebind_project_modules(
    request: Mapping[str, object],
    identities: Mapping[str, tuple[int, ...]],
) -> None:
    digests = request.get("code_sha256")
    if (
        not isinstance(digests, dict)
        or set(identities) != set(_CODE_NAMES)
    ):
        raise ModelWorkerError()
    for name in _CODE_NAMES:
        _payload, identity = _bound_payload(
            _descriptor(request.get(f"{name}_descriptor")),
            maximum=_MAX_CODE_BYTES,
            expected_sha256=str(digests.get(name)),
            private=True,
        )
        if identity != identities[name]:
            raise ModelWorkerError()


def _parse_rows(
    value: object,
    *,
    expected_holds: int,
    expected_eot: int,
) -> tuple[MaterializedRow, ...]:
    if not isinstance(value, list) or not value:
        raise ModelWorkerError()
    rows: list[MaterializedRow] = []
    for expected_ordinal, raw in enumerate(value):
        if not isinstance(raw, dict) or set(raw) != {
            "ordinal",
            "pcm_filename",
            "pcm_bytes",
            "pcm_samples",
            "pcm_sha256",
            "silence_spans",
        }:
            raise ModelWorkerError()
        filename = raw.get("pcm_filename")
        pcm_bytes = raw.get("pcm_bytes")
        pcm_samples = raw.get("pcm_samples")
        digest = raw.get("pcm_sha256")
        raw_spans = raw.get("silence_spans")
        raw_ordinal = raw.get("ordinal")
        if (
            isinstance(raw_ordinal, bool)
            or not isinstance(raw_ordinal, int)
            or raw_ordinal != expected_ordinal
            or filename != f"row-{expected_ordinal:04d}.pcm"
            or not isinstance(filename, str)
            or isinstance(pcm_bytes, bool)
            or not isinstance(pcm_bytes, int)
            or not 0 < pcm_bytes <= _MAX_PCM_BYTES
            or pcm_bytes % 2
            or isinstance(pcm_samples, bool)
            or not isinstance(pcm_samples, int)
            or pcm_samples != pcm_bytes // 2
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            or not isinstance(raw_spans, list)
            or not raw_spans
        ):
            raise ModelWorkerError()
        spans: list[CausalSpan] = []
        prior_end = -1
        for raw_span in raw_spans:
            if (
                not isinstance(raw_span, list)
                or len(raw_span) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in raw_span
                )
            ):
                raise ModelWorkerError()
            start, end = raw_span
            if (
                start < 0
                or end <= start
                or start < prior_end
                or end > pcm_samples
                or end - start < GRID_SAMPLES
                or end - start > 80_000
            ):
                raise ModelWorkerError()
            spans.append(CausalSpan(start, end))
            prior_end = end
        if (
            len(spans) > 64
            or spans[-1].duration_samples < 3_200
            or pcm_samples - spans[-1].end_sample not in {0, 1}
        ):
            raise ModelWorkerError()
        rows.append(
            MaterializedRow(
                ordinal=expected_ordinal,
                pcm_filename=filename,
                pcm_bytes=pcm_bytes,
                pcm_samples=pcm_samples,
                pcm_sha256=digest,
                silence_spans=tuple(spans),
            )
        )
    if (
        len(rows) != expected_eot
        or sum(len(row.silence_spans) - 1 for row in rows) != expected_holds
    ):
        raise ModelWorkerError()
    return tuple(rows)


def _parse_materialization_leaf_identities(
    value: object,
    rows: Sequence[MaterializedRow],
) -> Mapping[str, tuple[int, ...]]:
    expected_names = {
        "manifest.json",
        *(row.pcm_filename for row in rows),
    }
    if not isinstance(value, dict) or set(value) != expected_names:
        raise ModelWorkerError()
    identities: dict[str, tuple[int, ...]] = {}
    for name, raw_identity in value.items():
        if (
            not isinstance(raw_identity, list)
            or len(raw_identity) != 6
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in raw_identity
            )
        ):
            raise ModelWorkerError()
        identities[name] = tuple(raw_identity)
    return identities


def _read_pcm(
    directory_descriptor: int,
    row: MaterializedRow,
    *,
    expected_stable_identity: tuple[int, ...],
) -> bytes:
    descriptor = -1
    try:
        before = os.stat(
            row.pcm_filename,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size != row.pcm_bytes
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
            or _stable_file_identity(before) != expected_stable_identity
        ):
            raise ModelWorkerError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(row.pcm_filename, flags, dir_fd=directory_descriptor)
        opened = os.fstat(descriptor)
        if (
            _identity(opened) != _identity(before)
            or _stable_file_identity(opened) != expected_stable_identity
        ):
            raise ModelWorkerError()
        payload = _pread_exact(descriptor, row.pcm_bytes, _MAX_PCM_BYTES)
        after = os.stat(
            row.pcm_filename,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            _identity(os.fstat(descriptor)) != _identity(opened)
            or _identity(after) != _identity(opened)
            or hashlib.sha256(payload).hexdigest() != row.pcm_sha256
            or _stable_file_identity(after) != expected_stable_identity
        ):
            raise ModelWorkerError()
        return payload
    except ModelWorkerError:
        raise
    except OSError:
        raise ModelWorkerError() from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_private_at(
    directory_descriptor: int,
    name: str,
    *,
    maximum: int,
    expected_sha256: str,
    expected_stable_identity: tuple[int, ...],
) -> bytes:
    descriptor = -1
    try:
        before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum
            or stat.S_IMODE(before.st_mode) & 0o077
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
            or _stable_file_identity(before) != expected_stable_identity
        ):
            raise ModelWorkerError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
        opened = os.fstat(descriptor)
        payload = _pread_exact(descriptor, opened.st_size, maximum)
        after = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            _identity(opened) != _identity(before)
            or _stable_file_identity(opened) != expected_stable_identity
            or _identity(os.fstat(descriptor)) != _identity(opened)
            or _identity(after) != _identity(opened)
            or _stable_file_identity(after) != expected_stable_identity
            or hashlib.sha256(payload).hexdigest() != expected_sha256
        ):
            raise ModelWorkerError()
        return payload
    except ModelWorkerError:
        raise
    except OSError:
        raise ModelWorkerError() from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _decision_ticks(
    span_samples: int,
    *,
    is_eot: bool,
    max_wait_samples: int,
) -> tuple[int, ...]:
    """Return span-relative ticks; HOLD excludes resume, EOT includes its end."""

    if span_samples <= 0 or max_wait_samples <= 0:
        raise ModelWorkerError()
    limit = min(span_samples, max_wait_samples)
    ticks = list(range(GRID_SAMPLES, limit + 1, GRID_SAMPLES))
    if not is_eot:
        ticks = [tick for tick in ticks if tick < span_samples]
    elif span_samples <= max_wait_samples and (not ticks or ticks[-1] != span_samples):
        ticks.append(span_samples)
    return tuple(ticks)


def _nearest_rank(values: Sequence[int], quantile: float) -> Optional[int]:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return int(ordered[rank - 1])


def _sample_summary(values: Sequence[int]) -> Mapping[str, object]:
    checked: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ModelWorkerError()
        checked.append(value)
    p50 = _nearest_rank(checked, 0.50)
    p95 = _nearest_rank(checked, 0.95)
    minimum = min(checked) if checked else None
    maximum = max(checked) if checked else None
    total = sum(checked)

    def milliseconds(value: Optional[int]) -> Optional[float]:
        return None if value is None else value * 1000.0 / SAMPLE_RATE_HZ

    return {
        "samples": {
            "count": len(checked),
            "sum": total,
            "min": minimum,
            "p50_nearest_rank": p50,
            "p95_nearest_rank": p95,
            "max": maximum,
        },
        "milliseconds": {
            "sum": milliseconds(total),
            "min": milliseconds(minimum),
            "p50_nearest_rank": milliseconds(p50),
            "p95_nearest_rank": milliseconds(p95),
            "max": milliseconds(maximum),
        },
    }


def _policy_with_prior_holds(
    endpointing: object,
    config: object,
    prior_holds: Sequence[CausalSpan],
) -> object:
    policy = endpointing.AdaptiveEndpointPolicy(config)
    for span in prior_holds:
        policy.observe_pause(span.duration_samples / SAMPLE_RATE_HZ)
    return policy


def _assert_max_wait_backstop(endpointing: object, contract: PolicyContract) -> None:
    for score in (0.0, 0.3, 0.5, 1.0):
        policy = endpointing.AdaptiveEndpointPolicy(contract.endpoint_config)
        if not policy.decide(
            acoustic_endpoint=True,
            completion_score=score,
            silence_sec=contract.max_wait_samples / SAMPLE_RATE_HZ,
        ):
            raise ModelWorkerError()


@dataclass(slots=True)
class _MetricAccumulator:
    state_counts: Counter[str]
    basis_counts: Counter[str]
    hold_denominator: int
    early_cut_count: int
    rows_with_cut: set[int]
    eot_denominator: int
    observed_eot_delays: list[int]
    censored_eot_lower_bounds: list[int]
    conservative_eot_delays: list[int]
    recorded_decisions: int

    @classmethod
    def empty(cls) -> "_MetricAccumulator":
        return cls(Counter(), Counter(), 0, 0, set(), 0, [], [], [], 0)

    def begin_span(self, *, is_eot: bool) -> None:
        if is_eot:
            self.eot_denominator += 1
        else:
            self.hold_denominator += 1

    def decision(self, observation: object, decision: object) -> None:
        self.recorded_decisions += 1
        self.state_counts[observation.completion_state.value] += 1
        self.basis_counts[decision.basis.value] += 1

    def finish_span(
        self,
        *,
        is_eot: bool,
        committed_delay: Optional[int],
        span: CausalSpan,
        row_ordinal: int,
        max_wait_samples: int,
    ) -> None:
        if is_eot:
            if committed_delay is None:
                if span.duration_samples >= max_wait_samples:
                    raise ModelWorkerError()
                self.censored_eot_lower_bounds.append(span.duration_samples)
                self.conservative_eot_delays.append(max_wait_samples)
            else:
                self.observed_eot_delays.append(committed_delay)
                self.conservative_eot_delays.append(committed_delay)
        elif committed_delay is not None:
            if committed_delay >= span.duration_samples:
                raise ModelWorkerError()
            self.early_cut_count += 1
            self.rows_with_cut.add(row_ordinal)

    def report(
        self,
        *,
        assumed_partial: bool,
        expected_holds: int,
        expected_eot: int,
    ) -> Mapping[str, object]:
        if (
            self.hold_denominator != expected_holds
            or self.eot_denominator != expected_eot
            or len(self.observed_eot_delays) + len(self.censored_eot_lower_bounds)
            != self.eot_denominator
            or len(self.conservative_eot_delays) != self.eot_denominator
        ):
            raise ModelWorkerError()
        return {
            "partial_state": (
                PARTIAL_ASSUMPTION if assumed_partial else NO_PARTIAL_PROFILE
            ),
            "hold": {
                "denominator": self.hold_denominator,
                "early_cut_count": self.early_cut_count,
                "early_cut_rate": (
                    self.early_cut_count / self.hold_denominator
                    if self.hold_denominator
                    else None
                ),
                "rows_with_any_early_cut": len(self.rows_with_cut),
            },
            "eot": {
                "denominator": self.eot_denominator,
                "publisher_labelled_commit_count": len(self.observed_eot_delays),
                "right_censored_at_publisher_labelled_silence_end_count": len(
                    self.censored_eot_lower_bounds
                ),
                "observed_publisher_labelled_silence_delay": _sample_summary(
                    self.observed_eot_delays
                ),
                "publisher_labelled_censored_delay_lower_bound": _sample_summary(
                    self.censored_eot_lower_bounds
                ),
                "conservative_all_row_delay_with_censored_at_max_wait": (
                    _sample_summary(self.conservative_eot_delays)
                ),
            },
            "recorded_prefix_decision_count": self.recorded_decisions,
            "completion_state_counts": dict(sorted(self.state_counts.items())),
            "decision_basis_counts": dict(sorted(self.basis_counts.items())),
        }


def _evaluate_profile(
    *,
    rows: Sequence[MaterializedRow],
    pcm_loader: Callable[[MaterializedRow], bytes],
    detector: object,
    endpointing: object,
    contract: PolicyContract,
    assumed_partial: bool,
    expected_hold_denominator: int,
    expected_eot_denominator: int,
) -> Mapping[str, Mapping[str, object]]:
    """Evaluate the full counterfactual and conservative whole-label subset.

    A label qualifies for the subset only when its publisher-labelled silence
    end is strictly before rule 3.  This is deliberately not a tick-level
    simulation: a crossing label may contain earlier decisions that would be
    reachable in the live engine, but the entire label is excluded here.
    """

    if bool(getattr(detector, "needs_audio", False)) is not True:
        raise ModelWorkerError()
    _assert_max_wait_backstop(endpointing, contract)
    full = _MetricAccumulator.empty()
    fully_before_rule3 = _MetricAccumulator.empty()
    fully_before_rule3_holds = 0
    fully_before_rule3_eot = 0
    for row in rows:
        samples = None
        if assumed_partial:
            pcm = pcm_loader(row)
            if len(pcm) != row.pcm_bytes:
                raise ModelWorkerError()
            samples = np.frombuffer(pcm, dtype="<i2").astype("float32")
            samples *= np.float32(1.0 / 32768.0)
            if samples.size != row.pcm_samples:
                raise ModelWorkerError()
        prior_holds: list[CausalSpan] = []
        for span_index, span in enumerate(row.silence_spans):
            is_eot = span_index == len(row.silence_spans) - 1
            span_is_fully_before_rule3 = span.end_sample < contract.rule3_samples
            scopes = (
                (full, fully_before_rule3)
                if span_is_fully_before_rule3
                else (full,)
            )
            for accumulator in scopes:
                accumulator.begin_span(is_eot=is_eot)
            if span_is_fully_before_rule3:
                if is_eot:
                    fully_before_rule3_eot += 1
                else:
                    fully_before_rule3_holds += 1
            policy = _policy_with_prior_holds(
                endpointing,
                contract.endpoint_config,
                prior_holds,
            )
            committed_delay: Optional[int] = None
            for delay in _decision_ticks(
                span.duration_samples,
                is_eot=is_eot,
                max_wait_samples=contract.max_wait_samples,
            ):
                prefix_end = span.start_sample + delay
                if prefix_end > span.end_sample or prefix_end > row.pcm_samples:
                    raise ModelWorkerError()
                causal_prefix = None
                if assumed_partial and delay >= contract.prosody_min_samples:
                    if samples is None:
                        raise ModelWorkerError()
                    # A view would retain ``.base`` access to future samples.
                    # Only scored ticks allocate an owning immutable prefix.
                    causal_prefix = np.array(
                        samples[:prefix_end],
                        dtype="float32",
                        order="C",
                        copy=True,
                    )
                    causal_prefix.setflags(write=False)
                partial = _OPAQUE_PARTIAL_SENTINEL if assumed_partial else ""
                observation, decision = endpointing.evaluate_turn_completion(
                    detector=detector,
                    policy=policy,
                    acoustic_endpoint=delay >= contract.acoustic_rule2_samples,
                    partial=partial,
                    silence_sec=delay / SAMPLE_RATE_HZ,
                    samples=causal_prefix,
                    sample_rate=SAMPLE_RATE_HZ,
                    allow_early=True,
                    vad_active=False,
                    audio_min_silence_sec=(
                        contract.prosody_min_samples / SAMPLE_RATE_HZ
                    ),
                    detector_needs_audio=True,
                )
                for accumulator in scopes:
                    accumulator.decision(observation, decision)
                if assumed_partial:
                    expected_state = (
                        endpointing.CompletionScoreState.AUDIO_WINDOW_PENDING
                        if delay < contract.prosody_min_samples
                        else endpointing.CompletionScoreState.SCORED
                    )
                    if observation.completion_state is not expected_state:
                        raise ModelWorkerError()
                    if expected_state is endpointing.CompletionScoreState.SCORED:
                        probability = observation.completion_score
                        if (
                            probability is None
                            or not math.isfinite(probability)
                            or not 0.0 <= probability <= 1.0
                        ):
                            raise ModelWorkerError()
                elif (
                    observation.completion_state
                    is not endpointing.CompletionScoreState.NO_PARTIAL
                ):
                    raise ModelWorkerError()
                if decision.commit:
                    committed_delay = delay
                    break
            for accumulator in scopes:
                accumulator.finish_span(
                    is_eot=is_eot,
                    committed_delay=committed_delay,
                    span=span,
                    row_ordinal=row.ordinal,
                    max_wait_samples=contract.max_wait_samples,
                )
            if not is_eot:
                prior_holds.append(span)
    return {
        "publisher_labels_fully_before_rule3": fully_before_rule3.report(
            assumed_partial=assumed_partial,
            expected_holds=fully_before_rule3_holds,
            expected_eot=fully_before_rule3_eot,
        ),
        "semantic_counterfactual_full_source": full.report(
            assumed_partial=assumed_partial,
            expected_holds=expected_hold_denominator,
            expected_eot=expected_eot_denominator,
        ),
    }


def _positive_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelWorkerError()
    return value


def _write_result(descriptor: int, payload: bytes) -> None:
    if not payload or len(payload) > _MAX_RESULT_BYTES:
        raise ModelWorkerError()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != 0
            or stat.S_IMODE(before.st_mode) != 0o600
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
        ):
            raise ModelWorkerError()
        stable_identity = _stable_file_identity(before)
        offset = 0
        while offset < len(payload):
            written = os.pwrite(descriptor, payload[offset:], offset)
            if written <= 0:
                raise ModelWorkerError()
            offset += written
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if (
            after.st_size != len(payload)
            or _stable_file_identity(after) != stable_identity
            or _pread_exact(descriptor, len(payload), _MAX_RESULT_BYTES) != payload
        ):
            raise ModelWorkerError()
    except ModelWorkerError:
        raise
    except OSError:
        raise ModelWorkerError() from None


def _set_resource_limits() -> None:
    try:
        import resource
    except ImportError:  # pragma: no cover - Linux is the production target
        raise ModelWorkerError() from None
    limits = (
        ("RLIMIT_CORE", 0),
        ("RLIMIT_CPU", 1_800),
        ("RLIMIT_FSIZE", _MAX_RESULT_BYTES),
        ("RLIMIT_NOFILE", 64),
        ("RLIMIT_AS", 4 * 1024 * 1024 * 1024),
    )
    try:
        for name, desired in limits:
            resource_id = getattr(resource, name)
            _soft, hard = resource.getrlimit(resource_id)
            capped = desired if hard == resource.RLIM_INFINITY else min(desired, hard)
            resource.setrlimit(resource_id, (capped, hard))
    except (AttributeError, OSError, ValueError):
        raise ModelWorkerError() from None


def _run(
    request: Mapping[str, object],
    *,
    request_sha256: str,
) -> Mapping[str, object]:
    global np  # noqa: PLW0603 - imported only after hard limits are installed
    import numpy as np

    endpointing, code_identities = _exec_project_modules(request)
    expected_holds = _positive_integer(request.get("expected_hold_denominator"))
    expected_eot = _positive_integer(request.get("expected_eot_denominator"))
    rows = _parse_rows(
        request.get("rows"),
        expected_holds=expected_holds,
        expected_eot=expected_eot,
    )
    leaf_identities = _parse_materialization_leaf_identities(
        request.get("materialization_leaf_identities"),
        rows,
    )
    endpoint_config_value = request.get("endpoint_config")
    if (
        not isinstance(endpoint_config_value, dict)
        or set(endpoint_config_value) != _ENDPOINT_CONFIG_FIELDS
    ):
        raise ModelWorkerError()
    try:
        endpoint_config = endpointing.EndpointConfig(**endpoint_config_value)
    except Exception:
        raise ModelWorkerError() from None
    contract = PolicyContract(
        endpoint_config=endpoint_config,
        acoustic_rule2_samples=_positive_integer(
            request.get("acoustic_rule2_samples")
        ),
        prosody_min_samples=_positive_integer(request.get("prosody_min_samples")),
        max_wait_samples=_positive_integer(request.get("max_wait_samples")),
        rule3_samples=_positive_integer(request.get("rule3_samples")),
    )
    model_descriptor = _descriptor(request.get("model_descriptor"))
    model_size = _positive_integer(request.get("model_size_bytes"))
    model_sha256 = str(request.get("model_sha256"))
    _model_payload, model_identity = _bound_payload(
        model_descriptor,
        maximum=model_size,
        expected_size=model_size,
        expected_sha256=model_sha256,
        private=True,
    )
    materialization_descriptor = _descriptor(
        request.get("materialization_descriptor")
    )
    try:
        materialization_info = os.fstat(materialization_descriptor)
    except OSError:
        raise ModelWorkerError() from None
    if (
        not stat.S_ISDIR(materialization_info.st_mode)
        or stat.S_IMODE(materialization_info.st_mode) != 0o700
        or (
            hasattr(os, "geteuid")
            and materialization_info.st_uid != os.geteuid()
        )
    ):
        raise ModelWorkerError()
    directory_identity = _stable_directory_identity(materialization_info)
    declared_directory_identity = request.get("materialization_directory_identity")
    if (
        not isinstance(declared_directory_identity, list)
        or len(declared_directory_identity) != 5
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in declared_directory_identity
        )
        or declared_directory_identity != list(directory_identity)
    ):
        raise ModelWorkerError()
    manifest_sha256 = str(request.get("materialization_manifest_sha256"))
    manifest_payload = _read_private_at(
        materialization_descriptor,
        "manifest.json",
        maximum=_MAX_REQUEST_BYTES,
        expected_sha256=manifest_sha256,
        expected_stable_identity=leaf_identities["manifest.json"],
    )
    manifest = _strict_json(manifest_payload)
    if (
        not isinstance(manifest, dict)
        or manifest.get("rows") != request.get("rows")
        or manifest.get("materialization_sha256")
        != request.get("materialization_sha256")
        or manifest.get("pcm_set_sha256") != request.get("pcm_set_sha256")
        or manifest.get("row_count") != expected_eot
        or manifest.get("hold_label_count") != expected_holds
    ):
        raise ModelWorkerError()
    try:
        if set(os.listdir(materialization_descriptor)) != {
            "manifest.json",
            *(row.pcm_filename for row in rows),
        }:
            raise ModelWorkerError()
    except OSError:
        raise ModelWorkerError() from None
    try:
        detector = endpointing.ProsodyTurnCompletionDetector(
            f"/proc/self/fd/{model_descriptor}",
            num_threads=1,
        )
        detector.load()
    except Exception:
        raise ModelWorkerError() from None
    if detector.needs_audio is not True:
        raise ModelWorkerError()
    try:
        if detector._session.get_providers() != ["CPUExecutionProvider"]:
            raise ModelWorkerError()
    except (AttributeError, TypeError):
        raise ModelWorkerError() from None

    def loader(row: MaterializedRow) -> bytes:
        return _read_pcm(
            materialization_descriptor,
            row,
            expected_stable_identity=leaf_identities[row.pcm_filename],
        )

    candidate = _evaluate_profile(
        rows=rows,
        pcm_loader=loader,
        detector=detector,
        endpointing=endpointing,
        contract=contract,
        assumed_partial=True,
        expected_hold_denominator=expected_holds,
        expected_eot_denominator=expected_eot,
    )
    no_partial = _evaluate_profile(
        rows=rows,
        pcm_loader=loader,
        detector=detector,
        endpointing=endpointing,
        contract=contract,
        assumed_partial=False,
        expected_hold_denominator=expected_holds,
        expected_eot_denominator=expected_eot,
    )
    _payload_after, model_identity_after = _bound_payload(
        model_descriptor,
        maximum=model_size,
        expected_size=model_size,
        expected_sha256=model_sha256,
        private=True,
    )
    if model_identity_after != model_identity:
        raise ModelWorkerError()
    _rebind_project_modules(request, code_identities)
    manifest_after = _read_private_at(
        materialization_descriptor,
        "manifest.json",
        maximum=_MAX_REQUEST_BYTES,
        expected_sha256=manifest_sha256,
        expected_stable_identity=leaf_identities["manifest.json"],
    )
    try:
        if (
            manifest_after != manifest_payload
            or _stable_directory_identity(os.fstat(materialization_descriptor))
            != directory_identity
            or set(os.listdir(materialization_descriptor))
            != {"manifest.json", *(row.pcm_filename for row in rows)}
        ):
            raise ModelWorkerError()
    except OSError:
        raise ModelWorkerError() from None
    code_sha256 = request.get("code_sha256")
    if not isinstance(code_sha256, dict):
        raise ModelWorkerError()
    return {
        "schema_version": PROTOCOL_VERSION,
        "evaluation_contract": {
            "request_sha256": request_sha256,
            "worker_sha256": code_sha256["worker"],
            "endpointing_sha256": code_sha256["endpointing"],
            "acoustic_sha256": code_sha256["acoustic"],
            "text_sha256": code_sha256["text"],
            "model_sha256": model_sha256,
            "config_sha256": request["config_sha256"],
            "materialization_manifest_sha256": manifest_sha256,
            "materialization_sha256": request["materialization_sha256"],
            "pcm_set_sha256": request["pcm_set_sha256"],
        },
        "candidate": candidate,
        "acoustic_no_partial_fallback": no_partial,
        "runtime_receipt": {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "onnxruntime_version": importlib.metadata.version("onnxruntime"),
            "provider": "CPUExecutionProvider",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        os.umask(0o077)
        _set_resource_limits()
        if argv is None:
            argv = sys.argv[1:]
        if len(argv) != 2 or argv[0] != "--request-fd":
            raise ModelWorkerError()
        request_descriptor = _descriptor(int(argv[1]))
        request, request_sha256 = _load_request(request_descriptor)
        result = _run(request, request_sha256=request_sha256)
        _write_result(
            _descriptor(request.get("result_descriptor")),
            _canonical_json_bytes(result),
        )
        return 0
    except BaseException:  # noqa: BLE001 - never expose private detail
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
