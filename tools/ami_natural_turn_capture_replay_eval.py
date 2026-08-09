"""Aggregate-only AMI natural-turn capture-replay diagnostic.

The four-case private fixture is replayed through the unchanged generic
capture evaluator.  Private references and callback text are retained only
long enough to compute the fixed two-utterance minimum-order overlap metric.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import signal
import stat
import subprocess
import sys
import threading
import time
from typing import Final

from core.config import apply_device_profile, load_config
from core.engine import TranscriptAbortReason
from core.engines.sherpa import SherpaConfig
from core.wer import normalize
from tools import capture_replay_eval
from tools.capture_replay import ReplayAssertion
from tools.capture_replay.ami_natural_turn import (
    AmiNaturalTurnFixtureError,
    LoadedAmiNaturalTurnFixture,
    load_ami_natural_turn_fixture,
    verify_ami_natural_turn_fixture_snapshot,
)
from tools.capture_replay.metrics import (
    ReplayAcousticEvent,
    ReplayRunRecord,
    _validate_record_shape,
    aggregate_metrics,
)
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.overlap_metrics import (
    PROTOCOL_VERSION as OVERLAP_PROTOCOL,
    aggregate_two_utterance_min_order_wer,
    score_two_utterance_min_order_wer,
)


SCHEMA_VERSION: Final = 1
KIND: Final = "ami-natural-turn-capture-replay-eval-v1"
REPEATS: Final = 3
EXPECTED_CASES: Final = 4
EXPECTED_WINDOWS: Final = 2
EXPECTED_EVALUATIONS: Final = EXPECTED_CASES * REPEATS
EXPECTED_OVERLAP_EVALUATIONS: Final = 2 * REPEATS
EXPECTED_DIALOGUE_ACT_EXPOSURES: Final = 24
_MAX_REPORT_BYTES: Final = 16 * 1024 * 1024
_MAX_RECEIPT_BYTES: Final = 4096
_MAX_SOURCE_BYTES: Final = 4 * 1024 * 1024
_MAX_CONFIG_BYTES: Final = 2 * 1024 * 1024
_MAX_WATCHDOG_SECONDS: Final = 7200
_PROCESS_GROUP_TERM_SECONDS: Final = 2.0
_PROCESS_GROUP_KILL_SECONDS: Final = 2.0
_PROCESS_GROUP_POLL_SECONDS: Final = 0.01
_WORKER_ENV: Final = "SPEAKER_AMI_NATURAL_TURN_WORKER"
_SAFE_PROVIDERS: Final = frozenset({"cpu", "cuda"})
_SAFE_ERROR: Final = {
    "error": "ami_natural_turn_capture_replay_unavailable",
    "ok": False,
}
_ROOT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "kind",
        "evidence",
        "fixture",
        "capture_replay",
        "natural_turn",
        "execution",
        "binding_sha256",
    }
)
_GENERIC_ROOT_FIELDS: Final = frozenset(
    {
        "execution_complete",
        "quality_verdict",
        "schema_version",
        "corpus",
        "engine",
        "metrics",
    }
)
_WRAPPER_SOURCE_FILES: Final = (
    "always_on_agent/__init__.py",
    "always_on_agent/capabilities.py",
    "always_on_agent/continuation.py",
    "always_on_agent/diagnostics.py",
    "always_on_agent/event_bus.py",
    "always_on_agent/events.py",
    "always_on_agent/followups.py",
    "always_on_agent/memory.py",
    "always_on_agent/models.py",
    "always_on_agent/origin.py",
    "always_on_agent/planner.py",
    "always_on_agent/recall.py",
    "always_on_agent/runtime.py",
    "always_on_agent/session_actor.py",
    "always_on_agent/speech_analyzer.py",
    "always_on_agent/supervisor.py",
    "always_on_agent/tasks.py",
    "always_on_agent/text.py",
    "core/__init__.py",
    "core/asr_text.py",
    "core/config.py",
    "core/contract.py",
    "core/engines/__init__.py",
    "core/metrics.py",
    "core/realtime_media_stage.py",
    "core/tts_markup.py",
    "core/wer.py",
    "tools/__init__.py",
    "tools/ami_natural_turn_capture_replay_eval.py",
    "tools/capture_replay/ami_natural_turn.py",
    "tools/capture_replay/metrics.py",
    "tools/capture_replay_eval.py",
    "tools/streaming_stt/__init__.py",
    "tools/streaming_stt/bounded_io.py",
    "tools/streaming_stt/overlap_metrics.py",
)
_NON_EAGER_GENERIC_SOURCE_FILES: Final = frozenset(
    {"core/endpointing.py", "core/engines/_faster_whisper.py"}
)
_SOURCE_FILES: Final = tuple(
    sorted(
        (set(_WRAPPER_SOURCE_FILES) | set(capture_replay_eval._EVALUATOR_SOURCE_FILES))
        - _NON_EAGER_GENERIC_SOURCE_FILES
    )
)


class AmiNaturalTurnCaptureReplayEvalError(RuntimeError):
    """A detail-free fixture, execution, privacy, or publication failure."""


class _LifecycleSignal(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__()
        self.signum = signum


@dataclass(frozen=True, slots=True)
class _ConfigBinding:
    configured: SherpaConfig = field(repr=False, compare=False)
    configured_sha256: str
    executed_sha256: str
    artifact_sha256: str
    generic_source_sha256: str
    generic_source_files: int
    provider: str
    final_backend: str
    verifier_backend: str
    base_sha256: str
    base_bytes: int
    local_present: bool
    local_sha256: str | None
    local_bytes: int
    skip_local: bool


@dataclass(frozen=True, slots=True)
class _InputBinding:
    fixture: LoadedAmiNaturalTurnFixture = field(repr=False, compare=False)
    fixture_identity: tuple[object, ...] = field(repr=False)
    config: _ConfigBinding = field(repr=False)
    closure_sha256: str


@dataclass(slots=True)
class _ReportCommitState:
    committed: bool = False
    digest: str = ""
    pending_report: Mapping[str, object] | None = field(default=None, repr=False)


@dataclass(slots=True)
class _WorkerOwnership:
    spawned: bool = False
    cleanup_verified: bool = False


OutcomeFactory = Callable[..., capture_replay_eval._CaptureReplayEvaluationOutcome]


def _canonical_json(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (MemoryError, OverflowError, TypeError, UnicodeError, ValueError):
        raise AmiNaturalTurnCaptureReplayEvalError() from None
    return raw + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise AmiNaturalTurnCaptureReplayEvalError()
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                AmiNaturalTurnCaptureReplayEvalError()
            ),
        )
    except AmiNaturalTurnCaptureReplayEvalError:
        raise
    except (OverflowError, RecursionError, UnicodeError, ValueError):
        raise AmiNaturalTurnCaptureReplayEvalError() from None


def _plain_json(value: object) -> object:
    return _strict_json(_canonical_json(value))


def _absolute_path(value: str) -> Path:
    selected = Path(value).expanduser()
    if not selected.is_absolute() or Path(os.path.abspath(selected)) != selected:
        raise argparse.ArgumentTypeError("expected an absolute path")
    return selected


def _positive_int(value: str) -> int:
    try:
        selected = int(value, 10)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a positive integer") from None
    if selected <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return selected


def _has_git_ancestor(path: Path) -> bool:
    """Return whether one lexical output parent is inside a Git worktree."""

    try:
        current = path.resolve(strict=True)
        while True:
            marker = current / ".git"
            try:
                marker.lstat()
            except FileNotFoundError:
                pass
            else:
                return True
            if current.parent == current:
                return False
            current = current.parent
    except (OSError, RuntimeError, ValueError):
        raise AmiNaturalTurnCaptureReplayEvalError() from None


def _closure_sha256() -> str:
    root = Path(__file__).resolve(strict=True).parents[1]
    rows: list[dict[str, object]] = []
    try:
        for relative in _SOURCE_FILES:
            path = (root / relative).resolve(strict=True)
            if path.relative_to(root).as_posix() != relative:
                raise AmiNaturalTurnCaptureReplayEvalError()
            snapshot = read_regular_bounded(path, maximum_bytes=_MAX_SOURCE_BYTES)
            rows.append(
                {
                    "name": relative,
                    "size_bytes": len(snapshot.data),
                    "sha256": hashlib.sha256(snapshot.data).hexdigest(),
                }
            )
        return _canonical_sha256(rows)
    except AmiNaturalTurnCaptureReplayEvalError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        raise AmiNaturalTurnCaptureReplayEvalError() from None


def _source_snapshot(path: Path, *, required: bool) -> tuple[bool, str | None, int]:
    try:
        snapshot = read_regular_bounded(path, maximum_bytes=_MAX_CONFIG_BYTES)
    except BoundedReadError:
        try:
            path.lstat()
        except FileNotFoundError:
            if not required:
                return False, None, 0
        raise AmiNaturalTurnCaptureReplayEvalError() from None
    return True, hashlib.sha256(snapshot.data).hexdigest(), len(snapshot.data)


def _load_config_binding(
    config_path: Path,
    local_config_path: Path,
    *,
    device: str | None,
    provider: str | None,
    asr_threads: int | None,
) -> _ConfigBinding:
    try:
        base_present, base_sha256, base_bytes = _source_snapshot(
            config_path,
            required=True,
        )
        if not base_present or base_sha256 is None:
            raise AmiNaturalTurnCaptureReplayEvalError()
        local_present, local_sha256, local_bytes = _source_snapshot(
            local_config_path,
            required=False,
        )
        raw = load_config(str(config_path), local=str(local_config_path))
        selected_device = device if device is not None else raw.get("device", "auto")
        effective = apply_device_profile(
            raw, selected_device, strict=device is not None
        )
        configured = SherpaConfig.from_dict(effective.get("sherpa", {}))
        executed = capture_replay_eval._replay_config(
            configured,
            provider=provider,
            asr_threads=asr_threads,
        )
        return _ConfigBinding(
            configured=configured,
            configured_sha256=capture_replay_eval._json_digest(asdict(configured)),
            executed_sha256=capture_replay_eval._json_digest(asdict(executed)),
            artifact_sha256=capture_replay_eval._artifact_metadata_digest(executed),
            generic_source_sha256=capture_replay_eval._evaluator_source_digest(),
            generic_source_files=len(capture_replay_eval._evaluator_source_files()),
            provider=capture_replay_eval._safe_label(
                executed.provider,
                capture_replay_eval._SAFE_PROVIDERS,
            ),
            final_backend=capture_replay_eval._safe_label(
                executed.asr_final_backend,
                capture_replay_eval._SAFE_FINAL_BACKENDS,
            ),
            verifier_backend=capture_replay_eval._safe_label(
                executed.asr_final_verifier_backend,
                capture_replay_eval._SAFE_VERIFIER_BACKENDS,
            ),
            base_sha256=base_sha256,
            base_bytes=base_bytes,
            local_present=local_present,
            local_sha256=local_sha256,
            local_bytes=local_bytes,
            skip_local=os.environ.get("SPEAKER_NO_LOCAL_CONFIG", "").strip().lower()
            not in ("", "0", "false", "no"),
        )
    except AmiNaturalTurnCaptureReplayEvalError:
        raise
    except Exception:
        raise AmiNaturalTurnCaptureReplayEvalError() from None


def _fixture_identity(value: LoadedAmiNaturalTurnFixture) -> tuple[object, ...]:
    try:
        return (
            value.root,
            value.fixture_id,
            value.production_evidence,
            value.lock_sha256,
            value.lock_recipe_sha256,
            value.labels_sha256,
            value.metadata_sha256,
            value.manifest_sha256,
            value.receipt_sha256,
            value.source_contract_sha256,
            value.preparer_closure_sha256,
            value.sample_rate_hz,
            value.frame_samples,
            value.case_samples,
            value.directory_identity,
            value.file_identities,
            value.replay_corpus.digest,
            value.replay_corpus.audio_bytes,
            tuple(
                (
                    case.case_id,
                    case.window_id,
                    case.channel,
                    case.replay_case_index,
                    case.samples,
                    case.frame_samples,
                    case.source_start_sample,
                    case.source_end_sample,
                    case.source_offset_sample,
                    case.source_end_output_sample,
                    case.reference_sha256,
                    tuple(act.reference_sha256 for act in case.dialogue_acts),
                    case.adjacency_pair_id,
                    case.overlap_interval,
                    case.gap_interval,
                )
                for case in value.cases
            ),
        )
    except Exception:
        raise AmiNaturalTurnCaptureReplayEvalError() from None


def _load_input_binding(
    fixture_dir: Path,
    config_path: Path,
    local_config_path: Path,
    *,
    device: str | None,
    provider: str | None,
    asr_threads: int | None,
) -> _InputBinding:
    try:
        loaded = load_ami_natural_turn_fixture(fixture_dir)
        verify_ami_natural_turn_fixture_snapshot(loaded)
        if (
            len(loaded.cases) != EXPECTED_CASES
            or len(loaded.replay_corpus.cases) != EXPECTED_CASES
            or loaded.sample_rate_hz != 16_000
            or loaded.frame_samples != 1_600
            or loaded.case_samples != 128_000
            or tuple(case.replay_case_index for case in loaded.cases)
            != tuple(range(EXPECTED_CASES))
            or tuple(case.case_id for case in loaded.cases)
            != tuple(case.case_id for case in loaded.replay_corpus.cases)
            or tuple(case.channel for case in loaded.cases)
            != ("close", "far", "close", "far")
            or len({case.window_id for case in loaded.cases}) != EXPECTED_WINDOWS
            or any(
                case.samples != loaded.case_samples
                or case.frame_samples != loaded.frame_samples
                or case.source_complete_sample != loaded.case_samples
                for case in loaded.cases
            )
            or sum(case.overlap_interval is not None for case in loaded.cases) != 2
            or sum(case.gap_interval is not None for case in loaded.cases) != 2
            or any(len(case.dialogue_acts) != 2 for case in loaded.cases)
            or sum(len(case.dialogue_acts) for case in loaded.cases) * REPEATS
            != EXPECTED_DIALOGUE_ACT_EXPOSURES
        ):
            raise AmiNaturalTurnCaptureReplayEvalError()
        return _InputBinding(
            fixture=loaded,
            fixture_identity=_fixture_identity(loaded),
            config=_load_config_binding(
                config_path,
                local_config_path,
                device=device,
                provider=provider,
                asr_threads=asr_threads,
            ),
            closure_sha256=_closure_sha256(),
        )
    except AmiNaturalTurnCaptureReplayEvalError:
        raise
    except (AmiNaturalTurnFixtureError, OSError, RuntimeError, TypeError, ValueError):
        raise AmiNaturalTurnCaptureReplayEvalError() from None


def _assert_rebound(
    initial: _InputBinding,
    fixture_dir: Path,
    config_path: Path,
    local_config_path: Path,
    *,
    device: str | None,
    provider: str | None,
    asr_threads: int | None,
) -> None:
    try:
        verify_ami_natural_turn_fixture_snapshot(initial.fixture)
        current = _load_input_binding(
            fixture_dir,
            config_path,
            local_config_path,
            device=device,
            provider=provider,
            asr_threads=asr_threads,
        )
        if (
            current.fixture_identity != initial.fixture_identity
            or current.config != initial.config
            or current.closure_sha256 != initial.closure_sha256
        ):
            raise AmiNaturalTurnCaptureReplayEvalError()
    except AmiNaturalTurnCaptureReplayEvalError:
        raise
    except Exception:
        raise AmiNaturalTurnCaptureReplayEvalError() from None


def _mapping(value: object, fields: set[str] | frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise AmiNaturalTurnCaptureReplayEvalError()
    return value


def _count(value: object) -> int:
    if type(value) is not int or value < 0:
        raise AmiNaturalTurnCaptureReplayEvalError()
    return value


def _finite(value: object, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AmiNaturalTurnCaptureReplayEvalError()
    selected = float(value)
    if not math.isfinite(selected) or selected < 0:
        raise AmiNaturalTurnCaptureReplayEvalError()
    return selected


def _validate_records(
    initial: _InputBinding,
    records: Sequence[ReplayRunRecord],
) -> tuple[int, int, list[object]]:
    expected_order = tuple(
        (repeat, case_index)
        for repeat in range(REPEATS)
        for case_index in range(EXPECTED_CASES)
    )
    if (
        isinstance(records, (str, bytes))
        or tuple((record.repeat, record.case_index) for record in records)
        != expected_order
    ):
        raise AmiNaturalTurnCaptureReplayEvalError()
    final_callbacks = 0
    abort_callbacks = 0
    overlap_results: list[object] = []
    for record in records:
        try:
            _validate_record_shape(record)
        except (TypeError, ValueError):
            raise AmiNaturalTurnCaptureReplayEvalError() from None
        emitted: list[float] = []
        ids: set[str] = set()
        for final in record.finals:
            timestamp = _finite(final.emitted_at)
            assert timestamp is not None
            if (
                not final.utterance_id
                or final.utterance_id in ids
                or (emitted and timestamp < emitted[-1])
            ):
                raise AmiNaturalTurnCaptureReplayEvalError()
            ids.add(final.utterance_id)
            emitted.append(timestamp)
        final_callbacks += len(record.finals)
        abort_callbacks += len(record.abort_reasons)
        case = initial.fixture.cases[record.case_index]
        if case.overlap_interval is not None:
            references = tuple(act.reference for act in case.dialogue_acts)
            if len(references) != 2 or any(
                type(item) is not str for item in references
            ):
                raise AmiNaturalTurnCaptureReplayEvalError()
            hypothesis = " ".join(
                final.text.strip() for final in record.finals if final.text.strip()
            )
            try:
                overlap_results.append(
                    score_two_utterance_min_order_wer(
                        references[0],
                        references[1],
                        hypothesis,
                    )
                )
            except (TypeError, ValueError):
                raise AmiNaturalTurnCaptureReplayEvalError() from None
    if len(overlap_results) != EXPECTED_OVERLAP_EVALUATIONS:
        raise AmiNaturalTurnCaptureReplayEvalError()
    return final_callbacks, abort_callbacks, overlap_results


_ACCURACY_COUNT_FIELDS = {
    "transcript_runs",
    "word_errors",
    "reference_words",
    "hypothesis_words",
    "substitutions",
    "insertions",
    "deletions",
    "character_errors",
    "reference_characters",
    "hypothesis_characters",
}


def _expected_rate(
    *,
    observations: int,
    errors: int,
    reference: int,
    hypothesis: int,
) -> float | None:
    if observations and reference:
        return round(errors / reference, 4)
    if observations and hypothesis:
        return 1.0
    return None


def _validate_accuracy_counts(
    value: object,
    *,
    overlap: bool,
) -> Mapping[str, object]:
    fields = set(_ACCURACY_COUNT_FIELDS) | {
        "wer",
        "cer",
        "reference_order_comparable",
    }
    if overlap:
        fields |= {"linearized_wer", "linearized_cer"}
    row = _mapping(value, fields)
    counts = {key: _count(row.get(key)) for key in _ACCURACY_COUNT_FIELDS}
    if (
        counts["word_errors"]
        != counts["substitutions"] + counts["insertions"] + counts["deletions"]
        or counts["deletions"] > counts["reference_words"]
        or counts["insertions"] > counts["hypothesis_words"]
        or counts["substitutions"] + counts["deletions"] > counts["reference_words"]
        or counts["substitutions"] + counts["insertions"] > counts["hypothesis_words"]
        or counts["hypothesis_words"]
        != counts["reference_words"] - counts["deletions"] + counts["insertions"]
    ):
        raise AmiNaturalTurnCaptureReplayEvalError()
    expected_wer = _expected_rate(
        observations=counts["transcript_runs"],
        errors=counts["word_errors"],
        reference=counts["reference_words"],
        hypothesis=counts["hypothesis_words"],
    )
    expected_cer = _expected_rate(
        observations=counts["transcript_runs"],
        errors=counts["character_errors"],
        reference=counts["reference_characters"],
        hypothesis=counts["hypothesis_characters"],
    )
    if overlap:
        if (
            row.get("reference_order_comparable") is not False
            or row.get("wer") is not None
            or row.get("cer") is not None
            or row.get("linearized_wer") != expected_wer
            or row.get("linearized_cer") != expected_cer
        ):
            raise AmiNaturalTurnCaptureReplayEvalError()
    elif (
        row.get("reference_order_comparable") is not True
        or row.get("wer") != expected_wer
        or row.get("cer") != expected_cer
    ):
        raise AmiNaturalTurnCaptureReplayEvalError()
    return row


def _validate_generic_report(
    value: object,
    initial: _InputBinding,
    *,
    final_callbacks: int,
    abort_callbacks: int,
) -> dict[str, object]:
    report = _mapping(value, _GENERIC_ROOT_FIELDS)
    try:
        capture_replay_eval._validate_report_mode(
            report,
            logical_turn_shadow=False,
            logical_turn_terminal_lineage=False,
        )
    except Exception:
        raise AmiNaturalTurnCaptureReplayEvalError() from None
    corpus = _mapping(
        report.get("corpus"),
        {"manifest_sha256", "cases", "audio_bytes", "audio_seconds"},
    )
    engine = _mapping(
        report.get("engine"),
        {
            "configured_config_sha256",
            "executed_config_sha256",
            "artifact_metadata_sha256",
            "evaluator_source_sha256",
            "evaluator_source_files",
            "provider",
            "asr_final_backend_configured",
            "asr_final_verifier_backend_configured",
            "asr_final_verifier_execution",
            "final_execution",
            "streaming_decode_execution",
            "public_identity_models_loaded",
            "output_models_loaded",
            "model_load_ms",
        },
    )
    metrics = _mapping(
        report.get("metrics"),
        {
            "schema_version",
            "evaluations",
            "coverage",
            "accuracy",
            "latency",
            "streaming",
            "acoustic",
            "throughput",
            "resources",
        },
    )
    coverage = _mapping(
        metrics.get("coverage"), {"expected", "complete", "by_assertion"}
    )
    accuracy = _mapping(
        metrics.get("accuracy"),
        {
            "transcript_runs",
            "word_errors",
            "reference_words",
            "hypothesis_words",
            "wer",
            "substitutions",
            "insertions",
            "deletions",
            "character_errors",
            "reference_characters",
            "hypothesis_characters",
            "cer",
            "command_attempts",
            "command_hits",
            "lexical_command_hits",
            "command_recall",
            "contains_order_ambiguous_reference",
            "by_condition",
            "silence_runs",
            "silence_nonempty_partials",
            "silence_nonempty_finals",
        },
    )
    by_condition = _mapping(
        accuracy.get("by_condition"),
        {"human_overlap", "single_speaker", "turn_transition"},
    )
    overlap = _mapping(
        by_condition.get("human_overlap"),
        {
            "transcript_runs",
            "word_errors",
            "reference_words",
            "hypothesis_words",
            "wer",
            "substitutions",
            "insertions",
            "deletions",
            "character_errors",
            "reference_characters",
            "hypothesis_characters",
            "cer",
            "reference_order_comparable",
            "linearized_wer",
            "linearized_cer",
        },
    )
    _validate_accuracy_counts(overlap, overlap=True)
    single = _validate_accuracy_counts(
        by_condition.get("single_speaker"),
        overlap=False,
    )
    transition = _validate_accuracy_counts(
        by_condition.get("turn_transition"),
        overlap=False,
    )
    streaming = _mapping(
        metrics.get("streaming"),
        {
            "partial_events",
            "final_events",
            "churn_token_edits",
            "retracted_words",
            "stable_partial_missing",
            "text_runs_without_final",
            "abort_reasons",
        },
    )
    abort_reasons = streaming.get("abort_reasons")
    if not isinstance(abort_reasons, Mapping) or not set(abort_reasons).issubset(
        reason.value for reason in TranscriptAbortReason
    ):
        raise AmiNaturalTurnCaptureReplayEvalError()
    by_assertion = coverage.get("by_assertion")
    if not isinstance(by_assertion, Mapping) or set(by_assertion) != {
        assertion.value for assertion in ReplayAssertion
    }:
        raise AmiNaturalTurnCaptureReplayEvalError()
    assertion_counts = {key: _count(item) for key, item in by_assertion.items()}
    latency = _mapping(
        metrics.get("latency"),
        {
            "first_partial_observations",
            "first_partial_from_speech_start_p50_ms",
            "first_partial_from_speech_start_p95_ms",
            "stable_partial_from_speech_start_p50_ms",
            "stable_partial_from_speech_start_p95_ms",
            "stable_partial_observations",
            "speech_end_to_endpoint_observations",
            "speech_end_to_endpoint_p50_ms",
            "speech_end_to_endpoint_p95_ms",
            "endpoint_to_final_p50_ms",
            "endpoint_to_final_p95_ms",
            "endpoint_to_final_observations",
            "speech_end_to_final_p50_ms",
            "speech_end_to_final_p95_ms",
            "speech_end_to_final_observations",
        },
    )
    for key, item in latency.items():
        if key.endswith("_observations"):
            _count(item)
        else:
            _finite(item, optional=True)
    latency_families = (
        (
            "first_partial_observations",
            "first_partial_from_speech_start_p50_ms",
            "first_partial_from_speech_start_p95_ms",
        ),
        (
            "stable_partial_observations",
            "stable_partial_from_speech_start_p50_ms",
            "stable_partial_from_speech_start_p95_ms",
        ),
        (
            "speech_end_to_endpoint_observations",
            "speech_end_to_endpoint_p50_ms",
            "speech_end_to_endpoint_p95_ms",
        ),
        (
            "endpoint_to_final_observations",
            "endpoint_to_final_p50_ms",
            "endpoint_to_final_p95_ms",
        ),
        (
            "speech_end_to_final_observations",
            "speech_end_to_final_p50_ms",
            "speech_end_to_final_p95_ms",
        ),
    )
    for observations_key, p50_key, p95_key in latency_families:
        observations = _count(latency.get(observations_key))
        p50 = _finite(latency.get(p50_key), optional=True)
        p95 = _finite(latency.get(p95_key), optional=True)
        if (
            (observations == 0 and (p50 is not None or p95 is not None))
            or (observations > 0 and (p50 is None or p95 is None))
            or (p50 is not None and p95 is not None and p50 > p95)
            or (p50 is not None and p50 != round(p50, 3))
            or (p95 is not None and p95 != round(p95, 3))
        ):
            raise AmiNaturalTurnCaptureReplayEvalError()
    acoustic = _mapping(
        metrics.get("acoustic"),
        {
            "events",
            "double_talk_runs",
            "double_talk_with_event",
            "event_only_runs",
            "event_only_hits",
            "lineage_coverage",
        },
    )
    events = acoustic.get("events")
    if not isinstance(events, Mapping) or not set(events).issubset(
        event.value for event in ReplayAcousticEvent
    ):
        raise AmiNaturalTurnCaptureReplayEvalError()
    event_counts = {key: _count(item) for key, item in events.items()}
    lineage = _mapping(
        acoustic.get("lineage_coverage"),
        {
            "finals",
            "finals_complete",
            "finals_missing_speech_start",
            "finals_missing_speech_end",
            "finals_missing_endpoint",
            "partials_with_speech_start",
            "partials_missing_speech_start",
            "speech_end_to_endpoint_observations",
            "endpoint_to_final_observations",
            "speech_end_to_final_observations",
        },
    )
    lineage_counts = {key: _count(item) for key, item in lineage.items()}
    throughput = _mapping(
        metrics.get("throughput"),
        {"audio_total_sec", "wall_total_sec", "wall_to_audio_ratio"},
    )
    resources = _mapping(
        metrics.get("resources"),
        {"peak_rss_mb", "peak_vram_mb"},
    )
    audio_total = _finite(throughput.get("audio_total_sec"))
    wall_total = _finite(throughput.get("wall_total_sec"))
    wall_ratio = _finite(throughput.get("wall_to_audio_ratio"))
    assert audio_total is not None and wall_total is not None and wall_ratio is not None
    for item in resources.values():
        measured = _finite(item, optional=True)
        if measured is not None and measured != round(measured, 3):
            raise AmiNaturalTurnCaptureReplayEvalError()
    overall_counts = {key: _count(accuracy.get(key)) for key in _ACCURACY_COUNT_FIELDS}
    condition_counts = {
        key: sum(_count(row.get(key)) for row in (overlap, single, transition))
        for key in _ACCURACY_COUNT_FIELDS
    }
    expected_overall_wer = _expected_rate(
        observations=overall_counts["transcript_runs"],
        errors=overall_counts["word_errors"],
        reference=overall_counts["reference_words"],
        hypothesis=overall_counts["hypothesis_words"],
    )
    expected_overall_cer = _expected_rate(
        observations=overall_counts["transcript_runs"],
        errors=overall_counts["character_errors"],
        reference=overall_counts["reference_characters"],
        hypothesis=overall_counts["hypothesis_characters"],
    )
    command_attempts = _count(accuracy.get("command_attempts"))
    command_hits = _count(accuracy.get("command_hits"))
    lexical_hits = _count(accuracy.get("lexical_command_hits"))
    silence_counts = {
        key: _count(accuracy.get(key))
        for key in (
            "silence_runs",
            "silence_nonempty_partials",
            "silence_nonempty_finals",
        )
    }
    verifier = _mapping(
        engine.get("asr_final_verifier_execution"),
        {
            "requested",
            "available_after_run",
            "attempted_decodes",
            "completed_decodes",
            "decode_errors",
            "nonempty_decodes",
        },
    )
    verifier_counts = {
        key: _count(verifier.get(key))
        for key in (
            "attempted_decodes",
            "completed_decodes",
            "decode_errors",
            "nonempty_decodes",
        )
    }
    streaming_counts = {
        key: _count(streaming.get(key))
        for key in (
            "partial_events",
            "final_events",
            "churn_token_edits",
            "retracted_words",
            "stable_partial_missing",
            "text_runs_without_final",
        )
    }
    expected_audio_seconds = round(
        sum(case.duration_seconds for case in initial.fixture.replay_corpus.cases),
        3,
    )
    expected_total_audio_seconds = round(expected_audio_seconds * REPEATS, 3)
    expected_overlap_reference_words = REPEATS * sum(
        sum(len(normalize(reference)) for reference in case.overlap_references)
        for case in initial.fixture.cases
        if case.overlap_interval is not None
    )
    expected_transition_reference_words = REPEATS * sum(
        len(normalize(case.reference))
        for case in initial.fixture.cases
        if case.gap_interval is not None
    )
    expected_total_reference_words = REPEATS * sum(
        len(normalize(case.reference)) for case in initial.fixture.cases
    )
    expected_transition_reference_characters = REPEATS * sum(
        len(" ".join(normalize(case.reference)))
        for case in initial.fixture.cases
        if case.gap_interval is not None
    )
    expected_total_reference_characters = REPEATS * sum(
        len(" ".join(normalize(case.reference))) for case in initial.fixture.cases
    )
    if (
        report.get("execution_complete") is not True
        or report.get("quality_verdict") != "diagnostic_only"
        or report.get("schema_version") != 1
        or corpus.get("manifest_sha256") != initial.fixture.replay_corpus.digest
        or corpus.get("cases") != EXPECTED_CASES
        or corpus.get("audio_bytes") != initial.fixture.replay_corpus.audio_bytes
        or corpus.get("audio_seconds") != expected_audio_seconds
        or metrics.get("schema_version") != 1
        or metrics.get("evaluations") != EXPECTED_EVALUATIONS
        or coverage.get("expected") != EXPECTED_EVALUATIONS
        or coverage.get("complete") is not True
        or sum(assertion_counts.values()) != EXPECTED_EVALUATIONS
        or assertion_counts[ReplayAssertion.TRANSCRIPT.value] != EXPECTED_EVALUATIONS
        or overall_counts["transcript_runs"] != EXPECTED_EVALUATIONS
        or overall_counts != condition_counts
        or overall_counts["word_errors"]
        != overall_counts["substitutions"]
        + overall_counts["insertions"]
        + overall_counts["deletions"]
        or overall_counts["hypothesis_words"]
        != overall_counts["reference_words"]
        - overall_counts["deletions"]
        + overall_counts["insertions"]
        or overall_counts["substitutions"] + overall_counts["deletions"]
        > overall_counts["reference_words"]
        or overall_counts["substitutions"] + overall_counts["insertions"]
        > overall_counts["hypothesis_words"]
        or overall_counts["reference_words"] != expected_total_reference_words
        or overall_counts["reference_characters"] != expected_total_reference_characters
        or accuracy.get("wer") != expected_overall_wer
        or accuracy.get("cer") != expected_overall_cer
        or command_attempts != 0
        or command_hits != 0
        or lexical_hits != 0
        or accuracy.get("command_recall") is not None
        or any(silence_counts.values())
        or accuracy.get("contains_order_ambiguous_reference") is not True
        or overlap.get("transcript_runs") != EXPECTED_OVERLAP_EVALUATIONS
        or overlap.get("reference_words") != expected_overlap_reference_words
        or any(_count(single.get(key)) != 0 for key in _ACCURACY_COUNT_FIELDS)
        or transition.get("transcript_runs") != EXPECTED_OVERLAP_EVALUATIONS
        or transition.get("reference_words") != expected_transition_reference_words
        or transition.get("reference_characters")
        != expected_transition_reference_characters
        or overlap.get("wer") is not None
        or overlap.get("cer") is not None
        or overlap.get("reference_order_comparable") is not False
        or streaming_counts["final_events"] != final_callbacks
        or streaming_counts["text_runs_without_final"] > EXPECTED_EVALUATIONS
        or sum(_count(item) for item in abort_reasons.values()) != abort_callbacks
        or lineage_counts["finals"] != final_callbacks
        or lineage_counts["partials_with_speech_start"]
        + lineage_counts["partials_missing_speech_start"]
        != streaming_counts["partial_events"]
        or lineage_counts["endpoint_to_final_observations"]
        != final_callbacks - lineage_counts["finals_missing_endpoint"]
        or lineage_counts["speech_end_to_final_observations"]
        != final_callbacks - lineage_counts["finals_missing_speech_end"]
        or lineage_counts["finals_complete"]
        > lineage_counts["endpoint_to_final_observations"]
        or lineage_counts["finals_complete"]
        > lineage_counts["speech_end_to_final_observations"]
        or any(
            lineage_counts[key] > final_callbacks
            for key in (
                "finals_complete",
                "finals_missing_speech_start",
                "finals_missing_speech_end",
                "finals_missing_endpoint",
                "speech_end_to_endpoint_observations",
                "endpoint_to_final_observations",
                "speech_end_to_final_observations",
            )
        )
        or lineage_counts["speech_end_to_endpoint_observations"]
        != _count(latency["speech_end_to_endpoint_observations"])
        or lineage_counts["endpoint_to_final_observations"]
        != _count(latency["endpoint_to_final_observations"])
        or lineage_counts["speech_end_to_final_observations"]
        != _count(latency["speech_end_to_final_observations"])
        or _count(acoustic.get("double_talk_with_event"))
        > _count(acoustic.get("double_talk_runs"))
        or _count(acoustic.get("event_only_hits"))
        > _count(acoustic.get("event_only_runs"))
        or _count(acoustic.get("double_talk_runs")) != 0
        or _count(acoustic.get("double_talk_with_event")) != 0
        or _count(acoustic.get("event_only_runs")) != 0
        or _count(acoustic.get("event_only_hits")) != 0
        or any(item > EXPECTED_EVALUATIONS * 128 for item in event_counts.values())
        or engine.get("configured_config_sha256") != initial.config.configured_sha256
        or engine.get("executed_config_sha256") != initial.config.executed_sha256
        or engine.get("artifact_metadata_sha256") != initial.config.artifact_sha256
        or engine.get("evaluator_source_sha256") != initial.config.generic_source_sha256
        or engine.get("evaluator_source_files") != initial.config.generic_source_files
        or engine.get("final_execution") != "inline-deterministic-replay"
        or engine.get("streaming_decode_execution")
        != "single-session-synchronous-replay"
        or engine.get("public_identity_models_loaded") is not False
        or engine.get("output_models_loaded") is not False
        or engine.get("provider") != initial.config.provider
        or engine.get("asr_final_backend_configured") != initial.config.final_backend
        or engine.get("asr_final_verifier_backend_configured")
        != initial.config.verifier_backend
        or type(verifier.get("requested")) is not bool
        or type(verifier.get("available_after_run")) is not bool
        or verifier["requested"] is not bool(initial.config.verifier_backend)
        or verifier_counts["decode_errors"] != 0
        or verifier_counts["completed_decodes"] != verifier_counts["attempted_decodes"]
        or verifier_counts["nonempty_decodes"] > verifier_counts["completed_decodes"]
        or (not verifier["requested"] and any(verifier_counts.values()))
        or (verifier["requested"] and verifier_counts["completed_decodes"] <= 0)
        or verifier["available_after_run"] is not verifier["requested"]
        or audio_total != expected_total_audio_seconds
        or wall_total != round(wall_total, 3)
        or wall_ratio != round(wall_ratio, 4)
        or not (
            round(max(0.0, wall_total - 0.0005) / audio_total, 4)
            <= wall_ratio
            <= round((wall_total + 0.0005) / audio_total, 4)
        )
    ):
        raise AmiNaturalTurnCaptureReplayEvalError()
    _finite(corpus.get("audio_seconds"))
    _finite(engine.get("model_load_ms"))
    plain = _plain_json(report)
    if not isinstance(plain, dict) or plain != dict(report):
        raise AmiNaturalTurnCaptureReplayEvalError()
    return plain


def _aggregate_report(
    initial: _InputBinding,
    outcome: capture_replay_eval._CaptureReplayEvaluationOutcome,
    *,
    model_executed: bool,
) -> dict[str, object]:
    if type(outcome) is not capture_replay_eval._CaptureReplayEvaluationOutcome:
        raise AmiNaturalTurnCaptureReplayEvalError()
    final_callbacks, abort_callbacks, overlap_results = _validate_records(
        initial,
        outcome.records,
    )
    generic = _validate_generic_report(
        outcome.report,
        initial,
        final_callbacks=final_callbacks,
        abort_callbacks=abort_callbacks,
    )
    try:
        recomputed_metrics = aggregate_metrics(
            initial.fixture.replay_corpus,
            outcome.records,
            repeats=REPEATS,
        )
    except (TypeError, ValueError):
        raise AmiNaturalTurnCaptureReplayEvalError() from None
    if _canonical_json(generic["metrics"]) != _canonical_json(recomputed_metrics):
        raise AmiNaturalTurnCaptureReplayEvalError()
    try:
        raw_diagnostic = aggregate_two_utterance_min_order_wer(
            overlap_results
        ).as_dict()
    except (TypeError, ValueError):
        raise AmiNaturalTurnCaptureReplayEvalError() from None
    if raw_diagnostic.pop("cases", None) != EXPECTED_OVERLAP_EVALUATIONS:
        raise AmiNaturalTurnCaptureReplayEvalError()
    fixture = initial.fixture
    total_samples = sum(case.samples for case in fixture.cases)
    total_frames = sum(case.samples // case.frame_samples for case in fixture.cases)
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "evidence": {
            "diagnostic_only": True,
            "manual_dialogue_acts_descriptive_only": True,
            "ordinary_overlap_wer": False,
            "orc_wer": False,
            "endpoint_ground_truth": False,
            "complete_acoustic_turn": False,
            "capture_device": False,
            "live_latency": False,
            "qualification_authority": False,
            "promotion_authority": False,
            "default_change": False,
            "model_executed": model_executed,
        },
        "fixture": {
            "manifest_sha256": fixture.manifest_sha256,
            "receipt_sha256": fixture.receipt_sha256,
            "labels_sha256": fixture.labels_sha256,
            "lock_recipe_sha256": fixture.lock_recipe_sha256,
            "cases": EXPECTED_CASES,
            "windows": EXPECTED_WINDOWS,
            "audio_bytes": fixture.replay_corpus.audio_bytes,
            "samples": total_samples,
            "frames": total_frames,
            "production_evidence": fixture.production_evidence,
        },
        "capture_replay": generic,
        "natural_turn": {
            "coverage": {
                "cases": EXPECTED_CASES,
                "repeats": REPEATS,
                "evaluations": EXPECTED_EVALUATIONS,
                "source_complete_evaluations": EXPECTED_EVALUATIONS,
                "overlap_evaluations": EXPECTED_OVERLAP_EVALUATIONS,
            },
            "overlap": {
                "ordinary_wer": None,
                "reference_order_comparable": False,
                "diagnostic": {
                    "protocol": raw_diagnostic["protocol"],
                    "evaluations": EXPECTED_OVERLAP_EVALUATIONS,
                    "substitutions": raw_diagnostic["substitutions"],
                    "insertions": raw_diagnostic["insertions"],
                    "deletions": raw_diagnostic["deletions"],
                    "word_errors": raw_diagnostic["word_errors"],
                    "reference_words": raw_diagnostic["reference_words"],
                    "hypothesis_words": raw_diagnostic["hypothesis_words"],
                    "wer": raw_diagnostic["wer"],
                },
            },
            "terminal_counts": {
                "protocol": "manual-dialogue-act-exposure-v1",
                "manual_dialogue_act_exposures": EXPECTED_DIALOGUE_ACT_EXPOSURES,
                "typed_final_callbacks": final_callbacks,
                "typed_abort_callbacks": abort_callbacks,
                "descriptive_only": True,
            },
        },
        "execution": {
            "closure_sha256": initial.closure_sha256,
            "source_files": len(_SOURCE_FILES),
        },
        "binding_sha256": "",
    }
    report["binding_sha256"] = _canonical_sha256(
        {key: value for key, value in report.items() if key != "binding_sha256"}
    )
    return _validate_report(report, initial=initial)


def _assert_aggregate_private(value: object, initial: _InputBinding) -> None:
    strings: list[str] = []

    def inspect(item: object) -> None:
        if item is None or isinstance(item, (bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise AmiNaturalTurnCaptureReplayEvalError()
            return
        if isinstance(item, str):
            if (
                len(item) > 4096
                or item.startswith(("/", "\\", "file:"))
                or PureWindowsPath(item).is_absolute()
            ):
                raise AmiNaturalTurnCaptureReplayEvalError()
            strings.append(item)
            return
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise AmiNaturalTurnCaptureReplayEvalError()
                inspect(nested)
            return
        raise AmiNaturalTurnCaptureReplayEvalError()

    inspect(value)
    encoded = _canonical_json(value)
    if len(encoded) > _MAX_REPORT_BYTES:
        raise AmiNaturalTurnCaptureReplayEvalError()
    forbidden = {
        str(initial.fixture.root),
        initial.fixture.root.name,
        *(case.case_id for case in initial.fixture.cases),
        *(case.window_id for case in initial.fixture.cases),
        *(case.reference for case in initial.fixture.cases),
        *(
            act.reference
            for case in initial.fixture.cases
            for act in case.dialogue_acts
        ),
    }
    if any(
        text and any(text.casefold() in candidate.casefold() for candidate in strings)
        for text in forbidden
    ):
        raise AmiNaturalTurnCaptureReplayEvalError()


def _validate_report(
    value: object,
    *,
    initial: _InputBinding,
    expected_model_executed: bool | None = None,
) -> dict[str, object]:
    report = _mapping(value, _ROOT_FIELDS)
    evidence = _mapping(
        report.get("evidence"),
        {
            "diagnostic_only",
            "manual_dialogue_acts_descriptive_only",
            "ordinary_overlap_wer",
            "orc_wer",
            "endpoint_ground_truth",
            "complete_acoustic_turn",
            "capture_device",
            "live_latency",
            "qualification_authority",
            "promotion_authority",
            "default_change",
            "model_executed",
        },
    )
    fixture = _mapping(
        report.get("fixture"),
        {
            "manifest_sha256",
            "receipt_sha256",
            "labels_sha256",
            "lock_recipe_sha256",
            "cases",
            "windows",
            "audio_bytes",
            "samples",
            "frames",
            "production_evidence",
        },
    )
    natural = _mapping(
        report.get("natural_turn"), {"coverage", "overlap", "terminal_counts"}
    )
    coverage = _mapping(
        natural.get("coverage"),
        {
            "cases",
            "repeats",
            "evaluations",
            "source_complete_evaluations",
            "overlap_evaluations",
        },
    )
    overlap = _mapping(
        natural.get("overlap"),
        {"ordinary_wer", "reference_order_comparable", "diagnostic"},
    )
    diagnostic = _mapping(
        overlap.get("diagnostic"),
        {
            "protocol",
            "evaluations",
            "substitutions",
            "insertions",
            "deletions",
            "word_errors",
            "reference_words",
            "hypothesis_words",
            "wer",
        },
    )
    terminals = _mapping(
        natural.get("terminal_counts"),
        {
            "protocol",
            "manual_dialogue_act_exposures",
            "typed_final_callbacks",
            "typed_abort_callbacks",
            "descriptive_only",
        },
    )
    execution = _mapping(
        report.get("execution"),
        {"closure_sha256", "source_files"},
    )
    expected_fixture = {
        "manifest_sha256": initial.fixture.manifest_sha256,
        "receipt_sha256": initial.fixture.receipt_sha256,
        "labels_sha256": initial.fixture.labels_sha256,
        "lock_recipe_sha256": initial.fixture.lock_recipe_sha256,
        "cases": EXPECTED_CASES,
        "windows": EXPECTED_WINDOWS,
        "audio_bytes": initial.fixture.replay_corpus.audio_bytes,
        "samples": sum(case.samples for case in initial.fixture.cases),
        "frames": sum(
            case.samples // case.frame_samples for case in initial.fixture.cases
        ),
        "production_evidence": initial.fixture.production_evidence,
    }
    expected_coverage = {
        "cases": EXPECTED_CASES,
        "repeats": REPEATS,
        "evaluations": EXPECTED_EVALUATIONS,
        "source_complete_evaluations": EXPECTED_EVALUATIONS,
        "overlap_evaluations": EXPECTED_OVERLAP_EVALUATIONS,
    }
    model_executed = evidence.get("model_executed")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("kind") != KIND
        or fixture != expected_fixture
        or coverage != expected_coverage
        or evidence.get("diagnostic_only") is not True
        or evidence.get("manual_dialogue_acts_descriptive_only") is not True
        or any(
            evidence.get(key) is not False
            for key in (
                "ordinary_overlap_wer",
                "orc_wer",
                "endpoint_ground_truth",
                "complete_acoustic_turn",
                "capture_device",
                "live_latency",
                "qualification_authority",
                "promotion_authority",
                "default_change",
            )
        )
        or type(model_executed) is not bool
        or (
            expected_model_executed is not None
            and model_executed is not expected_model_executed
        )
        or (fixture["production_evidence"] and model_executed is not True)
        or overlap.get("ordinary_wer") is not None
        or overlap.get("reference_order_comparable") is not False
        or diagnostic.get("protocol") != OVERLAP_PROTOCOL
        or diagnostic.get("evaluations") != EXPECTED_OVERLAP_EVALUATIONS
        or terminals.get("protocol") != "manual-dialogue-act-exposure-v1"
        or terminals.get("manual_dialogue_act_exposures")
        != EXPECTED_DIALOGUE_ACT_EXPOSURES
        or terminals.get("descriptive_only") is not True
        or execution.get("closure_sha256") != initial.closure_sha256
        or execution.get("source_files") != len(_SOURCE_FILES)
    ):
        raise AmiNaturalTurnCaptureReplayEvalError()
    counts = {
        key: _count(diagnostic.get(key))
        for key in (
            "substitutions",
            "insertions",
            "deletions",
            "word_errors",
            "reference_words",
            "hypothesis_words",
        )
    }
    wer = _finite(diagnostic.get("wer"))
    expected_wer = (
        counts["word_errors"] / counts["reference_words"]
        if counts["reference_words"]
        else (0.0 if counts["hypothesis_words"] == 0 else 1.0)
    )
    expected_reference_words = REPEATS * sum(
        sum(len(normalize(reference)) for reference in case.overlap_references)
        for case in initial.fixture.cases
        if case.overlap_interval is not None
    )
    if (
        counts["word_errors"]
        != counts["substitutions"] + counts["insertions"] + counts["deletions"]
        or counts["deletions"] > counts["reference_words"]
        or counts["insertions"] > counts["hypothesis_words"]
        or counts["substitutions"] + counts["deletions"] > counts["reference_words"]
        or counts["substitutions"] + counts["insertions"] > counts["hypothesis_words"]
        or counts["reference_words"] != expected_reference_words
        or counts["hypothesis_words"]
        != counts["reference_words"] - counts["deletions"] + counts["insertions"]
        or wer != expected_wer
        or _count(terminals.get("typed_final_callbacks")) < 0
        or _count(terminals.get("typed_abort_callbacks")) < 0
    ):
        raise AmiNaturalTurnCaptureReplayEvalError()
    generic = _validate_generic_report(
        report.get("capture_replay"),
        initial,
        final_callbacks=_count(terminals.get("typed_final_callbacks")),
        abort_callbacks=_count(terminals.get("typed_abort_callbacks")),
    )
    if generic != report.get("capture_replay"):
        raise AmiNaturalTurnCaptureReplayEvalError()
    expected_binding = _canonical_sha256(
        {key: item for key, item in report.items() if key != "binding_sha256"}
    )
    if report.get("binding_sha256") != expected_binding:
        raise AmiNaturalTurnCaptureReplayEvalError()
    _assert_aggregate_private(report, initial)
    plain = _plain_json(report)
    if not isinstance(plain, dict) or plain != dict(report):
        raise AmiNaturalTurnCaptureReplayEvalError()
    return plain


def _parent_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        stat.S_IMODE(value.st_mode),
        value.st_uid,
        value.st_gid,
    )


def _entry_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        stat.S_IMODE(value.st_mode),
        value.st_nlink,
        value.st_size,
        value.st_uid,
        value.st_gid,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _private_parent(value: os.stat_result) -> bool:
    return bool(
        stat.S_ISDIR(value.st_mode)
        and not stat.S_IMODE(value.st_mode) & 0o077
        and (not hasattr(os, "geteuid") or value.st_uid == os.geteuid())
    )


def _report_path(path: Path | str) -> Path:
    selected = Path(path).expanduser()
    if (
        not selected.is_absolute()
        or Path(os.path.abspath(selected)) != selected
        or not selected.name
    ):
        raise AmiNaturalTurnCaptureReplayEvalError()
    return selected


def _preflight_report(path: Path | str) -> Path:
    selected = _report_path(path)
    descriptor = -1
    try:
        if _has_git_ancestor(selected.parent):
            raise AmiNaturalTurnCaptureReplayEvalError()
        with opened_directory_nofollow(selected.parent, require_private=True) as (
            stable,
            directory_fd,
        ):
            if stable != selected.parent or not _private_parent(os.fstat(directory_fd)):
                raise AmiNaturalTurnCaptureReplayEvalError()
            try:
                os.stat(selected.name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise AmiNaturalTurnCaptureReplayEvalError()
            temporary = getattr(os, "O_TMPFILE", 0)
            if not temporary:
                raise AmiNaturalTurnCaptureReplayEvalError()
            descriptor = os.open(
                ".",
                os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | temporary,
                0o600,
                dir_fd=directory_fd,
            )
            created = os.fstat(descriptor)
            if not stat.S_ISREG(created.st_mode) or created.st_nlink != 0:
                raise AmiNaturalTurnCaptureReplayEvalError()
        return selected
    except AmiNaturalTurnCaptureReplayEvalError:
        raise
    except Exception:
        raise AmiNaturalTurnCaptureReplayEvalError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_output_scope(
    output: Path | None,
    initial: _InputBinding,
    *,
    config_path: Path,
    local_config_path: Path,
    provider: str | None,
    asr_threads: int | None,
) -> None:
    """Keep the terminal report outside every immutable input/artifact tree."""

    if output is None:
        return
    try:
        executed = capture_replay_eval._replay_config(
            initial.config.configured,
            provider=provider,
            asr_threads=asr_threads,
        )
        protected = {
            initial.fixture.root.resolve(strict=True),
            config_path.resolve(strict=True).parent,
            local_config_path.parent.resolve(strict=True),
        }
        for field_name in capture_replay_eval._MODEL_PATH_FIELDS:
            raw = getattr(executed, field_name, "")
            if not isinstance(raw, str) or not raw:
                continue
            artifact = Path(raw).expanduser().resolve(strict=True)
            protected.add(artifact if artifact.is_dir() else artifact.parent)
        if any(_paths_overlap(output.parent, root) for root in protected):
            raise AmiNaturalTurnCaptureReplayEvalError()
    except AmiNaturalTurnCaptureReplayEvalError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise AmiNaturalTurnCaptureReplayEvalError() from None


def _commit_signals() -> tuple[int, ...]:
    return tuple(
        signum
        for signum in (
            getattr(signal, "SIGINT", None),
            getattr(signal, "SIGHUP", None),
            getattr(signal, "SIGTERM", None),
        )
        if isinstance(signum, int)
    )


def _publish_report(
    path: Path,
    report: Mapping[str, object],
    *,
    commit_guard: Callable[[], bool],
    state: _ReportCommitState,
) -> str:
    descriptor = -1
    encoded = _canonical_json(report, newline=True)
    digest = hashlib.sha256(encoded).hexdigest()
    try:
        if state.committed or state.digest or state.pending_report is not report:
            raise AmiNaturalTurnCaptureReplayEvalError()
        with opened_directory_nofollow(path.parent, require_private=True) as (
            stable,
            directory_fd,
        ):
            if stable != path.parent:
                raise AmiNaturalTurnCaptureReplayEvalError()
            parent_before = _parent_identity(os.fstat(directory_fd))
            try:
                os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise AmiNaturalTurnCaptureReplayEvalError()
            temporary = getattr(os, "O_TMPFILE", 0)
            if not temporary:
                raise AmiNaturalTurnCaptureReplayEvalError()
            descriptor = os.open(
                ".",
                os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | temporary,
                0o600,
                dir_fd=directory_fd,
            )
            os.fchmod(descriptor, 0o600)
            created = os.fstat(descriptor)
            staged = (created.st_dev, created.st_ino)
            if not stat.S_ISREG(created.st_mode) or created.st_nlink != 0:
                raise AmiNaturalTurnCaptureReplayEvalError()
            view = memoryview(encoded)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if type(count) is not int or count <= 0:
                    raise AmiNaturalTurnCaptureReplayEvalError()
                written += count
            os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            observed = os.read(descriptor, len(encoded) + 1)
            current = os.fstat(descriptor)
            if (
                observed != encoded
                or hashlib.sha256(observed).hexdigest() != digest
                or (current.st_dev, current.st_ino) != staged
                or current.st_nlink != 0
                or current.st_size != len(encoded)
                or stat.S_IMODE(current.st_mode) != 0o600
                or (hasattr(os, "geteuid") and current.st_uid != os.geteuid())
            ):
                raise AmiNaturalTurnCaptureReplayEvalError()
            if commit_guard() is not True:
                raise AmiNaturalTurnCaptureReplayEvalError()
            os.lseek(descriptor, 0, os.SEEK_SET)
            final_observed = os.read(descriptor, len(encoded) + 1)
            final_staged = os.fstat(descriptor)
            if (
                final_observed != encoded
                or hashlib.sha256(final_observed).hexdigest() != digest
                or (final_staged.st_dev, final_staged.st_ino) != staged
                or final_staged.st_nlink != 0
                or final_staged.st_size != len(encoded)
                or stat.S_IMODE(final_staged.st_mode) != 0o600
                or (hasattr(os, "geteuid") and final_staged.st_uid != os.geteuid())
                or _parent_identity(os.fstat(directory_fd)) != parent_before
                or _parent_identity(path.parent.lstat()) != parent_before
                or path.parent.resolve(strict=True) != path.parent
            ):
                raise AmiNaturalTurnCaptureReplayEvalError()
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _commit_signals())
            returned = False
            try:
                try:
                    os.link(
                        f"/proc/self/fd/{descriptor}",
                        path.name,
                        dst_dir_fd=directory_fd,
                        follow_symlinks=True,
                    )
                    returned = True
                finally:
                    if returned:
                        state.committed = True
                        state.digest = digest
                    else:
                        try:
                            linked = os.stat(
                                path.name, dir_fd=directory_fd, follow_symlinks=False
                            )
                            opened = os.fstat(descriptor)
                        except OSError:
                            pass
                        else:
                            if (
                                (linked.st_dev, linked.st_ino) == staged
                                and (opened.st_dev, opened.st_ino) == staged
                                and linked.st_nlink == opened.st_nlink == 1
                                and linked.st_size == opened.st_size == len(encoded)
                                and stat.S_IMODE(linked.st_mode)
                                == stat.S_IMODE(opened.st_mode)
                                == 0o600
                                and (
                                    not hasattr(os, "geteuid")
                                    or linked.st_uid == opened.st_uid == os.geteuid()
                                )
                            ):
                                state.committed = True
                                state.digest = digest
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            if not state.committed:
                raise AmiNaturalTurnCaptureReplayEvalError()
            try:
                os.fsync(directory_fd)
            except BaseException:
                pass
        return state.digest
    except BaseException as error:
        if state.committed:
            return state.digest
        if isinstance(error, (KeyboardInterrupt, _LifecycleSignal)):
            raise
        raise AmiNaturalTurnCaptureReplayEvalError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def run_ami_natural_turn_capture_replay_eval(
    fixture_dir: Path | str,
    *,
    config_path: Path | str,
    local_config_path: Path | str,
    device: str | None = None,
    provider: str | None = None,
    asr_threads: int | None = None,
    output_path: Path | str | None = None,
    _outcome_factory: OutcomeFactory | None = None,
    _commit_state: _ReportCommitState | None = None,
) -> dict[str, object]:
    """Run the fixed diagnostic and optionally publish its terminal report."""

    if provider is not None and provider not in _SAFE_PROVIDERS:
        raise AmiNaturalTurnCaptureReplayEvalError()
    if asr_threads is not None and (type(asr_threads) is not int or asr_threads <= 0):
        raise AmiNaturalTurnCaptureReplayEvalError()
    fixture_path = Path(os.path.abspath(Path(fixture_dir).expanduser()))
    config = Path(os.path.abspath(Path(config_path).expanduser()))
    local = Path(os.path.abspath(Path(local_config_path).expanduser()))
    output = _preflight_report(output_path) if output_path is not None else None
    if _commit_state is not None and type(_commit_state) is not _ReportCommitState:
        raise AmiNaturalTurnCaptureReplayEvalError()
    state = _ReportCommitState() if _commit_state is None else _commit_state
    if state.committed or state.digest or state.pending_report is not None:
        raise AmiNaturalTurnCaptureReplayEvalError()
    initial = _load_input_binding(
        fixture_path,
        config,
        local,
        device=device,
        provider=provider,
        asr_threads=asr_threads,
    )
    _validate_output_scope(
        output,
        initial,
        config_path=config,
        local_config_path=local,
        provider=provider,
        asr_threads=asr_threads,
    )
    injected = _outcome_factory is not None
    if injected and initial.fixture.production_evidence is not False:
        raise AmiNaturalTurnCaptureReplayEvalError()
    _assert_rebound(
        initial,
        fixture_path,
        config,
        local,
        device=device,
        provider=provider,
        asr_threads=asr_threads,
    )
    factory = (
        capture_replay_eval._evaluate_capture_replay_outcome
        if _outcome_factory is None
        else _outcome_factory
    )
    try:
        outcome = factory(
            initial.fixture.replay_corpus,
            initial.config.configured,
            repeats=REPEATS,
            provider=provider,
            asr_threads=asr_threads,
        )
    except Exception:
        raise AmiNaturalTurnCaptureReplayEvalError() from None
    _assert_rebound(
        initial,
        fixture_path,
        config,
        local,
        device=device,
        provider=provider,
        asr_threads=asr_threads,
    )
    report = _aggregate_report(initial, outcome, model_executed=not injected)
    _assert_rebound(
        initial,
        fixture_path,
        config,
        local,
        device=device,
        provider=provider,
        asr_threads=asr_threads,
    )

    def guard() -> bool:
        _assert_rebound(
            initial,
            fixture_path,
            config,
            local,
            device=device,
            provider=provider,
            asr_threads=asr_threads,
        )
        return True

    if output is None:
        if guard() is not True:
            raise AmiNaturalTurnCaptureReplayEvalError()
        return report
    state.pending_report = report
    try:
        observed = _publish_report(output, report, commit_guard=guard, state=state)
        if (
            observed
            != hashlib.sha256(_canonical_json(report, newline=True)).hexdigest()
        ):
            raise AmiNaturalTurnCaptureReplayEvalError()
        return report
    except BaseException:
        if state.committed:
            return report
        raise


def _read_descriptor(descriptor: int, maximum: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while total <= maximum:
        chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    if total > maximum:
        raise AmiNaturalTurnCaptureReplayEvalError()
    return b"".join(chunks)


def _verify_published_report(
    path: Path,
    *,
    fixture_dir: Path,
    config_path: Path,
    local_config_path: Path,
    device: str | None,
    provider: str | None,
    asr_threads: int | None,
    expected_sha256: str | None = None,
    expected_model_executed: bool | None = None,
) -> tuple[dict[str, object], str]:
    directory_descriptor = -1
    report_descriptor = -1
    try:
        initial = _load_input_binding(
            fixture_dir,
            config_path,
            local_config_path,
            device=device,
            provider=provider,
            asr_threads=asr_threads,
        )
        with opened_directory_nofollow(path.parent, require_private=True) as (
            stable,
            opened,
        ):
            if stable != path.parent:
                raise AmiNaturalTurnCaptureReplayEvalError()
            directory_descriptor = os.dup(opened)
        parent_before = os.fstat(directory_descriptor)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        report_descriptor = os.open(path.name, flags, dir_fd=directory_descriptor)
        before = os.fstat(report_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size <= 0
            or before.st_size > _MAX_REPORT_BYTES
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
        ):
            raise AmiNaturalTurnCaptureReplayEvalError()
        raw = _read_descriptor(report_descriptor, _MAX_REPORT_BYTES)
        after = os.fstat(report_descriptor)
        current = os.stat(path.name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            len(raw) != before.st_size
            or _entry_identity(after) != _entry_identity(before)
            or _entry_identity(current) != _entry_identity(before)
            or _parent_identity(os.fstat(directory_descriptor))
            != _parent_identity(parent_before)
            or _parent_identity(path.parent.lstat()) != _parent_identity(parent_before)
            or path.parent.resolve(strict=True) != path.parent
        ):
            raise AmiNaturalTurnCaptureReplayEvalError()
        os.close(report_descriptor)
        report_descriptor = -1
        current_closed = os.stat(
            path.name, dir_fd=directory_descriptor, follow_symlinks=False
        )
        if _entry_identity(current_closed) != _entry_identity(before):
            raise AmiNaturalTurnCaptureReplayEvalError()
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise AmiNaturalTurnCaptureReplayEvalError()
        value = _strict_json(raw)
        report = _validate_report(
            value,
            initial=initial,
            expected_model_executed=expected_model_executed,
        )
        if _canonical_json(report, newline=True) != raw:
            raise AmiNaturalTurnCaptureReplayEvalError()
        _assert_rebound(
            initial,
            fixture_dir,
            config_path,
            local_config_path,
            device=device,
            provider=provider,
            asr_threads=asr_threads,
        )
        return report, digest
    except AmiNaturalTurnCaptureReplayEvalError:
        raise
    except Exception:
        raise AmiNaturalTurnCaptureReplayEvalError() from None
    finally:
        for descriptor in (report_descriptor, directory_descriptor):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise AmiNaturalTurnCaptureReplayEvalError()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Run the fixed AMI natural-turn capture-replay diagnostic."
    )
    parser.add_argument("--fixture-dir", type=_absolute_path, required=True)
    parser.add_argument("--config", type=_absolute_path, required=True)
    parser.add_argument("--local-config", type=_absolute_path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--provider", choices=sorted(_SAFE_PROVIDERS))
    parser.add_argument("--asr-threads", type=_positive_int)
    parser.add_argument("--watchdog-seconds", type=_positive_int, default=1800)
    parser.add_argument("--report", type=_absolute_path, required=True)
    return parser


def _receipt(report: Mapping[str, object], digest: str) -> dict[str, object]:
    return {
        "ok": True,
        "kind": KIND,
        "report_sha256": digest,
        "evaluations": report["natural_turn"]["coverage"]["evaluations"],
        "model_executed": report["evidence"]["model_executed"],
    }


def _worker_main(argv: Sequence[str] | None = None) -> int:
    state = _ReportCommitState()
    previous: dict[int, object] = {}
    try:
        for signum in _commit_signals():
            if signum == getattr(signal, "SIGINT", None):
                continue
            previous[signum] = signal.signal(signum, _lifecycle_handler)
        args = _parser().parse_args(argv)
        report = run_ami_natural_turn_capture_replay_eval(
            args.fixture_dir,
            config_path=args.config,
            local_config_path=args.local_config,
            device=args.device,
            provider=args.provider,
            asr_threads=args.asr_threads,
            output_path=args.report,
            _commit_state=state,
        )
        result = _receipt(report, state.digest)
        code = 0
    except KeyboardInterrupt:
        if state.committed and state.pending_report is not None:
            result = _receipt(state.pending_report, state.digest)
            code = 0
        else:
            result = _SAFE_ERROR
            code = 130
    except _LifecycleSignal as interrupted:
        if state.committed and state.pending_report is not None:
            result = _receipt(state.pending_report, state.digest)
            code = 0
        else:
            result = _SAFE_ERROR
            code = 128 + interrupted.signum
    except BaseException:
        if state.committed and state.pending_report is not None:
            result = _receipt(state.pending_report, state.digest)
            code = 0
        else:
            result = _SAFE_ERROR
            code = 2
    try:
        os.write(1, _canonical_json(result, newline=True))
    except KeyboardInterrupt:
        if not state.committed or state.pending_report is None:
            result = _SAFE_ERROR
            code = 130
    except _LifecycleSignal as interrupted:
        if not state.committed or state.pending_report is None:
            result = _SAFE_ERROR
            code = 128 + interrupted.signum
    except BaseException:
        if not state.committed or state.pending_report is None:
            result = _SAFE_ERROR
            code = 2
    for signum, handler in reversed(tuple(previous.items())):
        try:
            signal.signal(signum, handler)
        except KeyboardInterrupt:
            if not state.committed or state.pending_report is None:
                result = _SAFE_ERROR
                code = 130
        except _LifecycleSignal as interrupted:
            if not state.committed or state.pending_report is None:
                result = _SAFE_ERROR
                code = 128 + interrupted.signum
        except BaseException:
            if not state.committed or state.pending_report is None:
                result = _SAFE_ERROR
                code = 2
    return code


def _validate_receipt(raw: bytes, returncode: int) -> Mapping[str, object]:
    value = _strict_json(raw)
    if returncode == 0:
        receipt = _mapping(
            value, {"ok", "kind", "report_sha256", "evaluations", "model_executed"}
        )
        if (
            receipt.get("ok") is not True
            or receipt.get("kind") != KIND
            or not _is_sha256(receipt.get("report_sha256"))
            or receipt.get("evaluations") != EXPECTED_EVALUATIONS
            or receipt.get("model_executed") is not True
        ):
            raise AmiNaturalTurnCaptureReplayEvalError()
        return receipt
    if value != _SAFE_ERROR or returncode not in {
        2,
        130,
        *[
            128 + item
            for item in _commit_signals()
            if item != getattr(signal, "SIGINT", None)
        ],
    }:
        raise AmiNaturalTurnCaptureReplayEvalError()
    return value


def _signal_process_group(process_group: int, signum: int) -> bool | None:
    """Signal one owned process group: false means the group is already gone."""

    try:
        os.killpg(process_group, signum)
        return True
    except ProcessLookupError:
        return False
    except OSError as error:
        if error.errno == errno.ESRCH:
            return False
        return None


def _process_group_exists(process_group: int) -> bool | None:
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError as error:
        if error.errno == errno.ESRCH:
            return False
        return None


def _wait_for_process_group_exit(process_group: int, deadline: float) -> bool:
    while True:
        exists = _process_group_exists(process_group)
        if exists is False:
            return True
        if exists is None or time.monotonic() >= deadline:
            return False
        time.sleep(_PROCESS_GROUP_POLL_SECONDS)


def _wait_for_process(process: subprocess.Popen[bytes], deadline: float) -> bool:
    try:
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
        return True
    except (OSError, subprocess.TimeoutExpired):
        return process.poll() is not None


def _terminate(process: subprocess.Popen[bytes]) -> bool:
    """Stop the whole owned session and reap its leader, even after leader exit."""

    if os.name != "posix":
        try:
            if process.poll() is None:
                process.terminate()
        except OSError:
            pass
        deadline = time.monotonic() + _PROCESS_GROUP_TERM_SECONDS
        if _wait_for_process(process, deadline):
            return True
        try:
            process.kill()
        except OSError:
            pass
        return _wait_for_process(
            process,
            time.monotonic() + _PROCESS_GROUP_KILL_SECONDS,
        )

    process_group = process.pid
    term_result = _signal_process_group(process_group, signal.SIGTERM)
    term_deadline = time.monotonic() + _PROCESS_GROUP_TERM_SECONDS
    leader_reaped = _wait_for_process(process, term_deadline)
    group_gone = term_result is False or _wait_for_process_group_exit(
        process_group,
        term_deadline,
    )
    if not group_gone:
        kill_result = _signal_process_group(process_group, signal.SIGKILL)
        kill_deadline = time.monotonic() + _PROCESS_GROUP_KILL_SECONDS
        leader_reaped = _wait_for_process(process, kill_deadline) or leader_reaped
        group_gone = kill_result is False or (
            kill_result is True
            and _wait_for_process_group_exit(process_group, kill_deadline)
        )
    return term_result is not None and leader_reaped and group_gone


def _unblock_child_lifecycle_signals() -> None:
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, _commit_signals())


def _checked_worker_cleanup(
    process: subprocess.Popen[bytes],
    ownership: _WorkerOwnership,
) -> tuple[bool, BaseException | None]:
    recorded: BaseException | None = None
    previous_mask: set[signal.Signals] | None = None
    mask_established = False
    for _attempt in range(2):
        try:
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK,
                _commit_signals(),
            )
            mask_established = True
            break
        except (KeyboardInterrupt, _LifecycleSignal) as interrupted:
            if recorded is None:
                recorded = interrupted
        except BaseException:
            break
    clean = False
    if mask_established:
        for _attempt in range(2):
            try:
                clean = _terminate(process) is True
                break
            except (KeyboardInterrupt, _LifecycleSignal) as interrupted:
                if recorded is None:
                    recorded = interrupted
            except BaseException:
                break
        restored = False
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            restored = True
        except (KeyboardInterrupt, _LifecycleSignal) as interrupted:
            if recorded is None:
                recorded = interrupted
            restored = True
        except BaseException:
            pass
        clean = clean and restored
    ownership.cleanup_verified = clean
    return clean, recorded


def _run_guarded_worker(
    raw_argv: Sequence[str],
    *,
    timeout: int,
    _ownership: _WorkerOwnership | None = None,
) -> tuple[int, bytes]:
    if _ownership is not None and type(_ownership) is not _WorkerOwnership:
        raise AmiNaturalTurnCaptureReplayEvalError()
    ownership = _WorkerOwnership() if _ownership is None else _ownership
    if ownership.spawned or ownership.cleanup_verified:
        raise AmiNaturalTurnCaptureReplayEvalError()
    environment = dict(os.environ)
    environment[_WORKER_ENV] = "1"
    process: subprocess.Popen[bytes] | None = None
    previous_mask: set[signal.Signals] | None = None
    try:
        if not hasattr(signal, "pthread_sigmask"):
            raise AmiNaturalTurnCaptureReplayEvalError()
        previous_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK,
            _commit_signals(),
        )
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "tools.ami_natural_turn_capture_replay_eval",
                    *raw_argv,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=environment,
                close_fds=True,
                start_new_session=True,
                preexec_fn=_unblock_child_lifecycle_signals,
            )
            ownership.spawned = True
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            previous_mask = None
    except BaseException as spawn_error:
        if previous_mask is not None:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            except BaseException:
                pass
        if process is not None:
            cleanup_ok, cleanup_signal = _checked_worker_cleanup(process, ownership)
            if not cleanup_ok:
                raise AmiNaturalTurnCaptureReplayEvalError() from None
            if not isinstance(spawn_error, (KeyboardInterrupt, _LifecycleSignal)):
                if cleanup_signal is not None:
                    raise cleanup_signal
        raise spawn_error
    assert process is not None
    if process.stdout is None:
        cleanup_ok, cleanup_signal = _checked_worker_cleanup(process, ownership)
        if not cleanup_ok:
            raise AmiNaturalTurnCaptureReplayEvalError() from None
        if cleanup_signal is not None:
            raise cleanup_signal
        raise AmiNaturalTurnCaptureReplayEvalError()
    captured = bytearray()
    overflow = threading.Event()

    def drain() -> None:
        try:
            while True:
                chunk = process.stdout.read(1024)
                if not chunk:
                    return
                room = _MAX_RECEIPT_BYTES + 1 - len(captured)
                if room > 0:
                    captured.extend(chunk[:room])
                if len(chunk) > room or len(captured) > _MAX_RECEIPT_BYTES:
                    overflow.set()
        except Exception:
            overflow.set()

    reader = threading.Thread(
        target=drain, name="ami-natural-turn-receipt", daemon=True
    )
    reader_started = False
    pending: BaseException | None = None
    returncode: int | None = None
    try:
        reader.start()
        reader_started = True
        returncode = process.wait(timeout=timeout)
    except BaseException as error:
        pending = error
    cleanup_ok, cleanup_signal = _checked_worker_cleanup(process, ownership)
    if cleanup_signal is not None and not isinstance(
        pending,
        (KeyboardInterrupt, _LifecycleSignal),
    ):
        pending = cleanup_signal
    try:
        if reader_started:
            reader.join(timeout=5)
    except BaseException as error:
        if pending is None:
            pending = error
    try:
        process.stdout.close()
    except BaseException as error:
        if pending is None:
            pending = error
    if not cleanup_ok:
        raise AmiNaturalTurnCaptureReplayEvalError()
    if isinstance(pending, subprocess.TimeoutExpired):
        raise AmiNaturalTurnCaptureReplayEvalError() from None
    if pending is not None:
        raise pending
    if (
        returncode is None
        or not reader_started
        or reader.is_alive()
        or overflow.is_set()
    ):
        raise AmiNaturalTurnCaptureReplayEvalError()
    return returncode, bytes(captured)


def _lifecycle_handler(signum: int, _frame: object) -> None:
    raise _LifecycleSignal(signum)


def guarded_main(argv: Sequence[str] | None = None) -> int:
    if os.environ.get(_WORKER_ENV) == "1":
        return _worker_main(argv)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    previous: dict[int, object] = {}
    args: argparse.Namespace | None = None
    ownership = _WorkerOwnership()
    interrupted_code = 2
    verified_report = False
    try:
        for signum in _commit_signals():
            if signum == getattr(signal, "SIGINT", None):
                continue
            previous[signum] = signal.signal(signum, _lifecycle_handler)
        args = _parser().parse_args(raw_argv)
        if args.watchdog_seconds > _MAX_WATCHDOG_SECONDS:
            raise AmiNaturalTurnCaptureReplayEvalError()
        _preflight_report(args.report)
        returncode, raw = _run_guarded_worker(
            raw_argv,
            timeout=args.watchdog_seconds,
            _ownership=ownership,
        )
        receipt = _validate_receipt(raw, returncode)
        if returncode != 0:
            interrupted_code = returncode
            result = receipt
            code = returncode
        else:
            retained, digest = _verify_published_report(
                args.report,
                fixture_dir=args.fixture_dir,
                config_path=args.config,
                local_config_path=args.local_config,
                device=args.device,
                provider=args.provider,
                asr_threads=args.asr_threads,
                expected_sha256=receipt["report_sha256"],
                expected_model_executed=True,
            )
            if _receipt(retained, digest) != receipt:
                raise AmiNaturalTurnCaptureReplayEvalError()
            result = receipt
            code = 0
            verified_report = True
    except KeyboardInterrupt:
        interrupted_code = 130
        result = _SAFE_ERROR
        code = interrupted_code
    except _LifecycleSignal as interrupted:
        interrupted_code = 128 + interrupted.signum
        result = _SAFE_ERROR
        code = interrupted_code
    except BaseException:
        result = _SAFE_ERROR
        code = 2
    if (
        code != 0
        and args is not None
        and ownership.spawned
        and ownership.cleanup_verified
    ):
        for _attempt in range(2):
            try:
                retained, digest = _verify_published_report(
                    args.report,
                    fixture_dir=args.fixture_dir,
                    config_path=args.config,
                    local_config_path=args.local_config,
                    device=args.device,
                    provider=args.provider,
                    asr_threads=args.asr_threads,
                    expected_model_executed=True,
                )
                result = _receipt(retained, digest)
                code = 0
                verified_report = True
                break
            except KeyboardInterrupt:
                interrupted_code = 130
                result = _SAFE_ERROR
                code = interrupted_code
            except _LifecycleSignal as interrupted:
                interrupted_code = 128 + interrupted.signum
                result = _SAFE_ERROR
                code = interrupted_code
            except BaseException:
                code = interrupted_code if interrupted_code != 2 else code
                break
    try:
        os.write(1, _canonical_json(result, newline=True))
    except KeyboardInterrupt:
        if not verified_report:
            interrupted_code = 130
            result = _SAFE_ERROR
            code = interrupted_code
    except _LifecycleSignal as interrupted:
        if not verified_report:
            interrupted_code = 128 + interrupted.signum
            result = _SAFE_ERROR
            code = interrupted_code
    except BaseException:
        if not verified_report:
            result = _SAFE_ERROR
            code = 2
    for signum, handler in reversed(tuple(previous.items())):
        try:
            signal.signal(signum, handler)
        except KeyboardInterrupt:
            if not verified_report:
                result = _SAFE_ERROR
                code = 130
        except _LifecycleSignal as interrupted:
            if not verified_report:
                result = _SAFE_ERROR
                code = 128 + interrupted.signum
        except BaseException:
            if not verified_report:
                result = _SAFE_ERROR
                code = 2
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(guarded_main())


__all__ = [
    "AmiNaturalTurnCaptureReplayEvalError",
    "KIND",
    "SCHEMA_VERSION",
    "guarded_main",
    "run_ami_natural_turn_capture_replay_eval",
]
