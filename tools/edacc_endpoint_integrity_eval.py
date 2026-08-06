"""Aggregate-only EdAcc endpoint-integrity counterfactual.

This diagnostic compares the configured native acoustic endpoint with two exact
Smart Turn v3.2 cells using Speaker's production Sherpa capture loop.  Both
semantic cells force ``allow_early=False`` at the typed turn-evaluation seam.
One preserves the original HOLD-only behavior; the other opts into native-stream
reset-and-accumulate after a typed HOLD.  The run cannot promote early commits
or change a live default.

The retained EdAcc reference, hypotheses, case identifiers, PCM, and paths stay
in memory.  Only closed counters, distributions, and cryptographic bindings may
be serialized.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import signal
import stat
import subprocess
import sys
import threading
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from always_on_agent.logical_turn_shadow import (
        LogicalTurnCompositionShadow,
        LogicalTurnShadowSnapshot,
        LogicalTurnTerminalLineageSnapshot,
    )

from core.endpointing import (
    AdaptiveEndpointPolicy,
    CompletionScoreState,
    EndpointConfig,
    ProsodyTurnCompletionDetector,
    TurnDecision,
    TurnDecisionBasis,
    TurnObservation,
)
from core.engine import TranscriptAbortReason
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from core.wer import normalize
from tools.bench.production import (
    PreparedProduction,
    ProductionInputError,
    load_production_corpus,
    prepare_production,
    verify_production_inputs,
)
from tools.capture_replay import (
    ReplayAssertion,
    ReplayCase,
    ReplayTrack,
)
from tools.capture_replay.metrics import (
    ReplayRunRecord,
    _AccuracyTotals,
    _validate_record_shape,
)
from tools.capture_replay_eval import (
    CaptureReplayEvaluationError,
    _execute_case,
    _quiet_model_output,
    _replay_config,
)
from tools.prepare_public_command_noise_corpus import _has_git_ancestor
from tools.setup_models import (
    SMART_TURN_MODEL_BYTES,
    SMART_TURN_MODEL_SHA256,
)
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.corpus import CorpusCase, LoadedCorpus


SCHEMA_VERSION = 2
KIND = "edacc-endpoint-integrity-v2"
SHADOW_SCHEMA_VERSION = 3
SHADOW_KIND = "edacc-endpoint-integrity-v3"
TERMINAL_LINEAGE_SCHEMA_VERSION = 4
TERMINAL_LINEAGE_KIND = "edacc-endpoint-integrity-v4"
EXPECTED_CORPUS_SHA256 = (
    "4da392c39a0b6bd18057f63c96b4f67c0dfdaf4c14e1d2d76176cdbee0920772"
)
EXPECTED_CASES = 24
EXPECTED_SOURCE_BYTES = 4_790_400
STRATA = ("clean", "disfluent", "overlap_adjacent")
EXPECTED_STRATUM_COUNTS = (8, 8, 8)
DEVICE_PROFILE = "desktop"
FINAL_STT_PROFILE = "sense-voice"
SAMPLE_RATE_HZ = 16_000
FRAME_SAMPLES = 1_600
ZERO_TAIL_SAMPLES = 38_400
ZERO_TAIL_MS = 2_400
ACOUSTIC_RULE2_SEC = 0.8
PROSODY_MIN_SILENCE_SEC = 0.15
ENDPOINT_MIN_SILENCE_SEC = 0.5
ENDPOINT_MAX_SILENCE_SEC = 1.6
LOGICAL_RULE3_SEC = 20.0
COMPLETE_THRESHOLD = 0.6
INCOMPLETE_THRESHOLD = 0.3
ASR_PROVIDER = "cpu"
ASR_THREADS = 1
_MAX_REPORT_BYTES = 16 * 1024 * 1024
_MAX_RECEIPT_BYTES = 4_096
_MAX_WATCHDOG_SECONDS = 7_200
_WORKER_ENV = "SPEAKER_EDACC_ENDPOINT_WORKER"
_MAX_SOURCE_FILE_BYTES = 4 * 1024 * 1024
_SAFE_ERROR: Mapping[str, object] = {
    "error": "edacc_endpoint_integrity_unavailable",
    "ok": False,
}
_SOURCE_FILES = (
    "core/endpointing.py",
    "core/engine.py",
    "core/engines/_asr_segment.py",
    "core/engines/_semantic_hold.py",
    "core/engines/sherpa.py",
    "tools/bench/production.py",
    "tools/capture_replay/corpus.py",
    "tools/capture_replay/metrics.py",
    "tools/capture_replay/replay.py",
    "tools/capture_replay_eval.py",
    "tools/edacc_endpoint_integrity_eval.py",
    "tools/prepare_public_command_noise_corpus.py",
    "tools/setup_models.py",
    "tools/streaming_stt/bounded_io.py",
)
_SHADOW_SOURCE_FILES = (
    "always_on_agent/logical_turn.py",
    "always_on_agent/logical_turn_shadow.py",
    "core/contract.py",
)
_TERMINAL_LINEAGE_SOURCE_FILES = (
    "always_on_agent/acoustic.py",
    "core/engines/_acoustic_turn.py",
)
_CELL_NAMES = (
    "acoustic",
    "prosody_hold_only",
    "prosody_reset_accumulate",
)
_SEMANTIC_CELL_NAMES = frozenset(_CELL_NAMES[1:])
_COMPARISON_NAMES = (
    "prosody_hold_only_vs_acoustic",
    "prosody_reset_accumulate_vs_acoustic",
    "prosody_reset_accumulate_vs_prosody_hold_only",
)
_FINAL_BUCKETS = ("zero", "single", "multi")
_FORBIDDEN_REPORT_KEYS = frozenset(
    {
        "case_id",
        "case_ids",
        "hypothesis",
        "hypotheses",
        "path",
        "paths",
        "pcm",
        "reference",
        "references",
        "text",
        "texts",
        "transcript",
        "transcripts",
        "utterance_id",
        "utterance_ids",
    }
)
_SAFE_STATIC_STRINGS = frozenset(
    {
        KIND,
        SHADOW_KIND,
        TERMINAL_LINEAGE_KIND,
        "sherpa_raw_composition_v1",
        "sherpa_selected_terminal_lineage_v1",
        "diagnostic_only",
        "capture-loop-counterfactual",
        "production-sherpa-capture-loop",
        "in-memory-f32le",
        "fresh-per-row",
        "CPUExecutionProvider",
        ASR_PROVIDER,
        DEVICE_PROFILE,
        FINAL_STT_PROFILE,
        *_CELL_NAMES,
        *_COMPARISON_NAMES,
        *_FINAL_BUCKETS,
        *STRATA,
        *(state.value for state in CompletionScoreState),
        *(basis.value for basis in TurnDecisionBasis),
        *(reason.value for reason in TranscriptAbortReason),
    }
)


class EdaccEndpointIntegrityError(RuntimeError):
    """A detail-free corpus, model, execution, or publication failure."""


@dataclass(frozen=True)
class _ModelBinding:
    path: Path = field(repr=False)
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class _CellResult:
    report: Mapping[str, object]
    row_final_buckets: tuple[str, ...]


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError):
        raise EdaccEndpointIntegrityError() from None


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_source_revision(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and 40 <= len(value) <= 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_files(
    *,
    logical_turn_shadow: bool = False,
    logical_turn_terminal_lineage: bool = False,
) -> tuple[str, ...]:
    if (
        type(logical_turn_shadow) is not bool
        or type(logical_turn_terminal_lineage) is not bool
        or (logical_turn_terminal_lineage and not logical_turn_shadow)
    ):
        raise EdaccEndpointIntegrityError()
    files = _SOURCE_FILES
    if logical_turn_shadow:
        files = (*files, *_SHADOW_SOURCE_FILES)
    if logical_turn_terminal_lineage:
        files = (*files, *_TERMINAL_LINEAGE_SOURCE_FILES)
    return files


def _source_binding(
    *,
    logical_turn_shadow: bool = False,
    logical_turn_terminal_lineage: bool = False,
) -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    try:
        for relative in _source_files(
            logical_turn_shadow=logical_turn_shadow,
            logical_turn_terminal_lineage=logical_turn_terminal_lineage,
        ):
            snapshot = read_regular_bounded(
                root / relative,
                maximum_bytes=_MAX_SOURCE_FILE_BYTES,
            )
            name = relative.encode("ascii")
            digest.update(len(name).to_bytes(4, "big"))
            digest.update(name)
            digest.update(len(snapshot.data).to_bytes(8, "big"))
            digest.update(hashlib.sha256(snapshot.data).digest())
    except (BoundedReadError, OSError, ValueError, OverflowError):
        raise EdaccEndpointIntegrityError() from None
    return digest.hexdigest()


def _snapshot_model(path: Path | str) -> _ModelBinding:
    try:
        snapshot = read_regular_bounded(
            Path(path).expanduser(),
            maximum_bytes=SMART_TURN_MODEL_BYTES,
            expected_bytes=SMART_TURN_MODEL_BYTES,
        )
        digest = hashlib.sha256(snapshot.data).hexdigest()
        if digest != SMART_TURN_MODEL_SHA256:
            raise EdaccEndpointIntegrityError()
        return _ModelBinding(
            path=snapshot.path,
            sha256=digest,
            size_bytes=len(snapshot.data),
        )
    except (BoundedReadError, OSError, ValueError, EdaccEndpointIntegrityError):
        raise EdaccEndpointIntegrityError() from None


def _validate_exact_corpus(corpus: LoadedCorpus) -> None:
    counts: Counter[str] = Counter()
    canonical = frozenset(STRATA)
    try:
        if (
            corpus.schema_version != 2
            or corpus.digest != EXPECTED_CORPUS_SHA256
            or len(corpus.cases) != EXPECTED_CASES
            or corpus.audio_bytes != EXPECTED_SOURCE_BYTES
        ):
            raise EdaccEndpointIntegrityError()
        for case in corpus.cases:
            membership = canonical.intersection(case.tags)
            if (
                case.assertion != "transcript"
                or len(case.audio_bytes) != case.samples * 4
                or len(membership) != 1
            ):
                raise EdaccEndpointIntegrityError()
            counts[next(iter(membership))] += 1
        if tuple(counts[tag] for tag in STRATA) != EXPECTED_STRATUM_COUNTS:
            raise EdaccEndpointIntegrityError()
    except (AttributeError, TypeError, ValueError, EdaccEndpointIntegrityError):
        raise EdaccEndpointIntegrityError() from None


def _prepared_binding(prepared: PreparedProduction) -> str:
    return _canonical_sha256(
        {
            "device_profile": prepared.device_profile,
            "final_stt_profile": prepared.final_stt_profile,
            "final_stt_profile_sha256": prepared.final_stt_profile_sha256,
            "final_stt_profile_schema_version": (
                prepared.final_stt_profile_schema_version
            ),
            "production_config_sha256": prepared.production_config_sha256,
            "execution_config_sha256": prepared.execution_config_sha256,
            "replay_safety_schema_version": prepared.replay_safety_schema_version,
            "replay_safety_sha256": prepared.replay_safety_sha256,
            "active_artifacts_sha256": prepared.active_artifacts_sha256,
            "active_artifact_roots": prepared.active_artifact_roots,
            "active_artifact_files": prepared.active_artifact_files,
            "active_artifact_bytes": prepared.active_artifact_bytes,
            "source_revision": prepared.source_revision,
            "runtime_identities": prepared.runtime_identities,
        }
    )


def _base_config(prepared: PreparedProduction) -> SherpaConfig:
    try:
        config = SherpaConfig.from_dict(prepared.config.get("sherpa", {}))
        if (
            prepared.device_profile != DEVICE_PROFILE
            or prepared.final_stt_profile != FINAL_STT_PROFILE
            or config.sample_rate != SAMPLE_RATE_HZ
            or not math.isclose(float(config.block_sec), 0.1, abs_tol=1e-12)
            or not math.isclose(
                float(config.asr_rule2_min_trailing_silence),
                ACOUSTIC_RULE2_SEC,
                abs_tol=1e-12,
            )
            or config.asr_final_backend != "sense_voice"
            or not str(config.asr_final_model or "").strip()
            or not str(config.asr_final_tokens or "").strip()
            or not str(config.vad_model or "").strip()
        ):
            raise EdaccEndpointIntegrityError()
        return config
    except (AttributeError, TypeError, ValueError, EdaccEndpointIntegrityError):
        raise EdaccEndpointIntegrityError() from None


def _cell_config(
    base: SherpaConfig,
    *,
    cell: str,
    model: _ModelBinding,
) -> SherpaConfig:
    common = replace(
        base,
        input_calibrate=False,
        input_calibrate_sec=0.0,
        asr_rule2_min_trailing_silence=ACOUSTIC_RULE2_SEC,
        endpoint_adaptive_floor=False,
        endpoint_min_silence_sec=ENDPOINT_MIN_SILENCE_SEC,
        endpoint_max_silence_sec=ENDPOINT_MAX_SILENCE_SEC,
        asr_rule3_min_utterance_length=LOGICAL_RULE3_SEC,
        endpoint_complete_threshold=COMPLETE_THRESHOLD,
        endpoint_incomplete_threshold=INCOMPLETE_THRESHOLD,
        endpoint_high_confidence_floor=0.0,
        endpoint_prosody_min_silence=PROSODY_MIN_SILENCE_SEC,
        endpoint_prosody_threads=1,
        endpoint_reset_on_semantic_hold=False,
        provider=ASR_PROVIDER,
        asr_num_threads=ASR_THREADS,
    )
    if cell == "acoustic":
        return replace(
            common,
            endpoint_enabled=False,
            endpoint_detector="lexical",
            endpoint_prosody_model="",
        )
    if cell == "prosody_hold_only":
        return replace(
            common,
            endpoint_enabled=True,
            endpoint_detector="prosody",
            endpoint_prosody_model=str(model.path),
        )
    if cell == "prosody_reset_accumulate":
        return replace(
            common,
            endpoint_enabled=True,
            endpoint_detector="prosody",
            endpoint_prosody_model=str(model.path),
            endpoint_reset_on_semantic_hold=True,
        )
    raise EdaccEndpointIntegrityError()


def _adapt_case(case: CorpusCase) -> ReplayCase:
    """Create a capture ReplayCase without inventing speech/word timestamps."""

    try:
        alignment = (-case.samples) % FRAME_SAMPLES
        zeros = b"\0" * ((alignment + ZERO_TAIL_SAMPLES) * 4)
        raw = case.audio_bytes + zeros
        samples = case.samples + alignment + ZERO_TAIL_SAMPLES
        if samples <= 0 or samples % FRAME_SAMPLES or len(raw) != samples * 4:
            raise EdaccEndpointIntegrityError()
        track = ReplayTrack(
            role="mic",
            path=case.source_path,
            sha256=hashlib.sha256(raw).hexdigest(),
            samples=samples,
            pcm_bytes=raw,
        )
        return ReplayCase(
            case_id=case.case_id,
            sample_rate_hz=SAMPLE_RATE_HZ,
            frame_samples=FRAME_SAMPLES,
            tracks=MappingProxyType({"mic": track}),
            speech_intervals=(),
            speaker_intervals=(),
            word_intervals=(),
            assertion=ReplayAssertion.TRANSCRIPT,
            expected_text=case.expected_text,
            commands=case.commands,
            tags=case.tags,
            aec_delay_samples=0,
        )
    except (AttributeError, TypeError, ValueError, EdaccEndpointIntegrityError):
        raise EdaccEndpointIntegrityError() from None


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if any(not math.isfinite(value) or value < 0.0 for value in ordered):
        raise EdaccEndpointIntegrityError()
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return round(ordered[index], 3)


def _full_counts(enum_type) -> dict[str, int]:
    return {item.value: 0 for item in enum_type}


def _full_transitions(enum_type) -> dict[str, int]:
    return {
        f"{left.value}->{right.value}": 0
        for left in enum_type
        for right in enum_type
    }


class _TraceAccumulator:
    """Aggregate typed endpoint observations without retaining text or PCM."""

    def __init__(self, *, candidate: bool) -> None:
        self._candidate = candidate
        self.state_counts = _full_counts(CompletionScoreState)
        self.basis_counts = _full_counts(TurnDecisionBasis)
        self.state_transitions = _full_transitions(CompletionScoreState)
        self.basis_transitions = _full_transitions(TurnDecisionBasis)
        self.evaluations = 0
        self.hold_ticks = 0
        self.hold_rows = 0
        self._row_evaluations = 0
        self._row_held = False
        self._prior_state: CompletionScoreState | None = None
        self._prior_basis: TurnDecisionBasis | None = None

    def begin_row(self) -> None:
        self._row_evaluations = 0
        self._row_held = False
        self._prior_state = None
        self._prior_basis = None

    def observe(self, observation: TurnObservation, decision: TurnDecision) -> None:
        if (
            not isinstance(observation, TurnObservation)
            or not isinstance(decision, TurnDecision)
            or observation.early_endpoint_allowed is not False
        ):
            raise EdaccEndpointIntegrityError()
        state = observation.completion_state
        basis = decision.basis
        if state not in CompletionScoreState or basis not in TurnDecisionBasis:
            raise EdaccEndpointIntegrityError()
        if state is CompletionScoreState.DETECTOR_ERROR:
            raise EdaccEndpointIntegrityError()
        if self._candidate and state is CompletionScoreState.UNAVAILABLE:
            raise EdaccEndpointIntegrityError()
        if basis is TurnDecisionBasis.SEMANTIC_EARLY:
            raise EdaccEndpointIntegrityError()
        self.evaluations += 1
        self._row_evaluations += 1
        self.state_counts[state.value] += 1
        self.basis_counts[basis.value] += 1
        if self._prior_state is not None:
            self.state_transitions[
                f"{self._prior_state.value}->{state.value}"
            ] += 1
        if self._prior_basis is not None:
            self.basis_transitions[
                f"{self._prior_basis.value}->{basis.value}"
            ] += 1
        self._prior_state = state
        self._prior_basis = basis
        if basis is TurnDecisionBasis.SEMANTIC_HOLD:
            self.hold_ticks += 1
            self._row_held = True

    def finish_row(self) -> None:
        if self._row_evaluations <= 0:
            raise EdaccEndpointIntegrityError()
        self.hold_rows += int(self._row_held)

    def report(self) -> dict[str, object]:
        if self._candidate and self.state_counts[CompletionScoreState.SCORED.value] <= 0:
            raise EdaccEndpointIntegrityError()
        if sum(self.state_counts.values()) != self.evaluations:
            raise EdaccEndpointIntegrityError()
        if sum(self.basis_counts.values()) != self.evaluations:
            raise EdaccEndpointIntegrityError()
        return {
            "evaluations": self.evaluations,
            "state_counts": dict(self.state_counts),
            "basis_counts": dict(self.basis_counts),
            "state_transitions": dict(self.state_transitions),
            "basis_transitions": dict(self.basis_transitions),
            "hold_rows": self.hold_rows,
            "hold_ticks": self.hold_ticks,
            "hard_boundary_count": self.state_counts[
                CompletionScoreState.HARD_BOUNDARY.value
            ],
            "max_wait_count": self.basis_counts[TurnDecisionBasis.MAX_WAIT.value],
        }


def _install_hold_only_trace(
    engine: SherpaOnnxEngine,
    trace: _TraceAccumulator,
) -> None:
    original = engine._turn_evaluation

    def wrapped(
        *,
        acoustic_endpoint: bool,
        partial: str,
        silence_sec: float,
        samples=None,
        allow_early: bool = True,
        vad_active: bool = False,
    ):
        del allow_early
        observation, decision = original(
            acoustic_endpoint=acoustic_endpoint,
            partial=partial,
            silence_sec=silence_sec,
            samples=samples,
            allow_early=False,
            vad_active=vad_active,
        )
        trace.observe(observation, decision)
        return observation, decision

    engine._turn_evaluation = wrapped  # type: ignore[method-assign]


def _fresh_policy(engine: SherpaOnnxEngine, *, candidate: bool) -> None:
    if candidate:
        engine._endpoint_policy = AdaptiveEndpointPolicy(
            EndpointConfig.from_sherpa(engine.config)
        )
    else:
        engine._endpoint_policy = None


def _attest_engine(engine: SherpaOnnxEngine, *, cell: str) -> None:
    if cell not in _CELL_NAMES:
        raise EdaccEndpointIntegrityError()
    candidate = cell in _SEMANTIC_CELL_NAMES
    reset_accumulate = cell == "prosody_reset_accumulate"
    if (
        engine._recognizer is None
        or engine._vad is None
        or engine._final_recognizer is None
        or engine.config.asr_final_backend != "sense_voice"
        or engine.config.provider != ASR_PROVIDER
        or engine.config.asr_num_threads != ASR_THREADS
        or engine.config.endpoint_reset_on_semantic_hold is not reset_accumulate
    ):
        raise EdaccEndpointIntegrityError()
    if not candidate:
        if engine._endpoint_policy is not None or engine.config.endpoint_enabled:
            raise EdaccEndpointIntegrityError()
        return
    detector = engine._turn_detector
    if (
        type(detector) is not ProsodyTurnCompletionDetector
        or detector.needs_audio is not True
        or engine._endpoint_policy is None
        or engine._endpoint_wants_audio is not True
        or engine.config.endpoint_detector != "prosody"
        or engine.config.endpoint_prosody_threads != 1
        or engine.config.endpoint_adaptive_floor is not False
    ):
        raise EdaccEndpointIntegrityError()
    try:
        if detector._session.get_providers() != ["CPUExecutionProvider"]:
            raise EdaccEndpointIntegrityError()
    except (AttributeError, TypeError, EdaccEndpointIntegrityError):
        raise EdaccEndpointIntegrityError() from None


class _CellAccumulator:
    """Reduce each private row immediately into aggregate-safe values."""

    def __init__(self) -> None:
        self.rows = 0
        self.finals = 0
        self.zero_rows = 0
        self.single_rows = 0
        self.multi_rows = 0
        self._accuracy = _AccuracyTotals()
        self.exact_rows = 0
        self.endpoint_delay_ms: list[float] = []
        self.row_wall_ms: list[float] = []
        self.peak_rss_mb: list[float] = []
        self.row_final_buckets: list[str] = []
        self.abort_counts = _full_counts(TranscriptAbortReason)
        self.abort_total = 0

    def add(self, case: CorpusCase, record: ReplayRunRecord, *, ordinal: int) -> None:
        try:
            _validate_record_shape(record)
            if (
                record.case_index != ordinal
                or record.repeat != 0
                or record.commands
                or record.acoustic_events
                or not math.isfinite(float(record.wall_seconds))
                or record.wall_seconds < 0.0
            ):
                raise EdaccEndpointIntegrityError()
            final_count = len(record.finals)
            if final_count == 0:
                bucket = "zero"
                self.zero_rows += 1
            elif final_count == 1:
                bucket = "single"
                self.single_rows += 1
            else:
                bucket = "multi"
                self.multi_rows += 1
            joined = " ".join(final.text.strip() for final in record.finals).strip()
            self._accuracy.add(case.expected_text, joined)
            self.exact_rows += int(
                tuple(normalize(case.expected_text)) == tuple(normalize(joined))
            )
            for final in record.finals:
                if (
                    final.speech_end_at is None
                    or final.endpoint_committed_at is None
                ):
                    raise EdaccEndpointIntegrityError()
                delay = 1_000.0 * (
                    float(final.endpoint_committed_at) - float(final.speech_end_at)
                )
                if not math.isfinite(delay) or delay < -1e-6:
                    raise EdaccEndpointIntegrityError()
                self.endpoint_delay_ms.append(max(0.0, delay))
            self.rows += 1
            self.finals += final_count
            self.row_final_buckets.append(bucket)
            self.row_wall_ms.append(float(record.wall_seconds) * 1_000.0)
            for reason in record.abort_reasons:
                self.abort_counts[reason.value] += 1
                self.abort_total += 1
            if record.peak_rss_mb is not None:
                value = float(record.peak_rss_mb)
                if not math.isfinite(value) or value < 0.0:
                    raise EdaccEndpointIntegrityError()
                self.peak_rss_mb.append(value)
        except (AttributeError, TypeError, ValueError, EdaccEndpointIntegrityError):
            raise EdaccEndpointIntegrityError() from None

    def report(self) -> dict[str, object]:
        if (
            self.rows != EXPECTED_CASES
            or self.zero_rows + self.single_rows + self.multi_rows != self.rows
            or len(self.row_final_buckets) != self.rows
            or len(self.endpoint_delay_ms) != self.finals
        ):
            raise EdaccEndpointIntegrityError()
        accuracy = dict(self._accuracy.as_report())
        accuracy["exact_rows"] = self.exact_rows
        accuracy["exact_rate"] = round(self.exact_rows / self.rows, 4)
        return {
            "coverage": {"rows": self.rows, "complete": True},
            "turn_integrity": {
                "finals": self.finals,
                "zero_final_rows": self.zero_rows,
                "single_final_rows": self.single_rows,
                "multi_final_rows": self.multi_rows,
            },
            "joined_selected_final_accuracy": accuracy,
            "vad_speech_end_to_endpoint_ms": {
                "samples": len(self.endpoint_delay_ms),
                "p50": _percentile(self.endpoint_delay_ms, 0.50),
                "p95": _percentile(self.endpoint_delay_ms, 0.95),
                "max": (
                    round(max(self.endpoint_delay_ms), 3)
                    if self.endpoint_delay_ms
                    else None
                ),
            },
            "transcript_aborts": {
                "total": self.abort_total,
                "terminal_events": self.finals + self.abort_total,
                "reason_counts": dict(self.abort_counts),
            },
            "resources": {
                "row_wall_ms_total": round(sum(self.row_wall_ms), 3),
                "row_wall_ms_p50": _percentile(self.row_wall_ms, 0.50),
                "row_wall_ms_p95": _percentile(self.row_wall_ms, 0.95),
                "process_peak_rss_mb": (
                    round(max(self.peak_rss_mb), 3)
                    if self.peak_rss_mb
                    else None
                ),
            },
        }


def _run_cell(
    corpus: LoadedCorpus,
    configured: SherpaConfig,
    *,
    cell: str,
    logical_turn_shadow: bool = False,
    logical_turn_terminal_lineage: bool = False,
) -> _CellResult:
    if (
        cell not in _CELL_NAMES
        or type(logical_turn_shadow) is not bool
        or type(logical_turn_terminal_lineage) is not bool
        or (logical_turn_terminal_lineage and not logical_turn_shadow)
    ):
        raise EdaccEndpointIntegrityError()
    candidate = cell in _SEMANTIC_CELL_NAMES
    try:
        executed = _replay_config(
            configured,
            provider=ASR_PROVIDER,
            asr_threads=ASR_THREADS,
        )
        trace = _TraceAccumulator(candidate=candidate)
        aggregate = _CellAccumulator()
        shadow_snapshots: list[LogicalTurnShadowSnapshot] = []
        terminal_lineage_snapshots: list[
            LogicalTurnTerminalLineageSnapshot
        ] = []
        build_started = time.perf_counter()
        with _quiet_model_output():
            engine = SherpaOnnxEngine(executed)
            engine._build()
            engine.warm()
        model_load_ms = (time.perf_counter() - build_started) * 1_000.0
        _attest_engine(engine, cell=cell)
        _install_hold_only_trace(engine, trace)
        try:
            with _quiet_model_output():
                for ordinal, source_case in enumerate(corpus.cases):
                    _fresh_policy(engine, candidate=candidate)
                    trace.begin_row()
                    replay_case = _adapt_case(source_case)
                    shadow: LogicalTurnCompositionShadow | None = None
                    case_options: dict[str, object] = {
                        "case_index": ordinal,
                        "repeat": 0,
                    }
                    if logical_turn_shadow:
                        from always_on_agent.logical_turn_shadow import (
                            LogicalTurnCompositionShadow,
                        )

                        shadow = LogicalTurnCompositionShadow(
                            terminal_lineage=logical_turn_terminal_lineage,
                        )
                        case_options["logical_turn_shadow"] = shadow
                    record = _execute_case(engine, replay_case, **case_options)
                    if shadow is not None:
                        shadow.close()
                        shadow_snapshots.append(shadow.snapshot())
                        if logical_turn_terminal_lineage:
                            terminal_lineage_snapshots.append(
                                shadow.terminal_lineage_snapshot()
                            )
                    trace.finish_row()
                    aggregate.add(source_case, record, ordinal=ordinal)
                    del replay_case
                    del record
        finally:
            session = engine._streaming_decode_session
            if session is not None and not session.closed:
                session.close()
        report = aggregate.report()
        report["config_sha256"] = _canonical_sha256(asdict(executed))
        report["provider"] = executed.provider
        report["asr_threads"] = executed.asr_num_threads
        report["semantic_hold_reset_enabled"] = (
            executed.endpoint_reset_on_semantic_hold
        )
        report["model_load_ms"] = round(model_load_ms, 3)
        report["endpoint_trace"] = trace.report()
        if logical_turn_shadow:
            from always_on_agent.logical_turn_shadow import (
                aggregate_logical_turn_shadows,
            )

            if len(shadow_snapshots) != EXPECTED_CASES:
                raise EdaccEndpointIntegrityError()
            report["logical_turn_shadow"] = aggregate_logical_turn_shadows(
                shadow_snapshots
            )
        if logical_turn_terminal_lineage:
            from always_on_agent.logical_turn_shadow import (
                aggregate_logical_turn_terminal_lineage,
            )

            if len(terminal_lineage_snapshots) != EXPECTED_CASES:
                raise EdaccEndpointIntegrityError()
            report["logical_turn_terminal_lineage"] = (
                aggregate_logical_turn_terminal_lineage(
                    terminal_lineage_snapshots
                )
            )
        buckets = tuple(aggregate.row_final_buckets)
        del engine
        gc.collect()
        return _CellResult(report=report, row_final_buckets=buckets)
    except (
        CaptureReplayEvaluationError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        EdaccEndpointIntegrityError,
    ):
        raise EdaccEndpointIntegrityError() from None


def _row_transitions(
    control: Sequence[str],
    candidate: Sequence[str],
) -> dict[str, int]:
    if (
        len(control) != EXPECTED_CASES
        or len(candidate) != EXPECTED_CASES
        or any(value not in _FINAL_BUCKETS for value in (*control, *candidate))
    ):
        raise EdaccEndpointIntegrityError()
    counts = {
        f"{left}->{right}": 0 for left in _FINAL_BUCKETS for right in _FINAL_BUCKETS
    }
    for left, right in zip(control, candidate, strict=True):
        counts[f"{left}->{right}"] += 1
    return counts


def _comparison_report(
    control: Sequence[str],
    candidate: Sequence[str],
) -> dict[str, object]:
    transitions = _row_transitions(control, candidate)
    return {
        "row_final_count_transitions": transitions,
        "control_multi_to_candidate_single_rows": transitions["multi->single"],
        "control_single_to_candidate_multi_rows": transitions["single->multi"],
        "candidate_reduced_multi_rows": transitions["multi->single"] > 0,
    }


def _report(
    *,
    prepared: PreparedProduction,
    model: _ModelBinding,
    source_sha256: str,
    cells: Mapping[str, _CellResult],
    logical_turn_shadow: bool = False,
    logical_turn_terminal_lineage: bool = False,
) -> dict[str, object]:
    if (
        type(logical_turn_shadow) is not bool
        or type(logical_turn_terminal_lineage) is not bool
        or (logical_turn_terminal_lineage and not logical_turn_shadow)
    ):
        raise EdaccEndpointIntegrityError()
    control_buckets = cells["acoustic"].row_final_buckets
    comparisons = {
        "prosody_hold_only_vs_acoustic": _comparison_report(
            control_buckets,
            cells["prosody_hold_only"].row_final_buckets,
        ),
        "prosody_reset_accumulate_vs_acoustic": _comparison_report(
            control_buckets,
            cells["prosody_reset_accumulate"].row_final_buckets,
        ),
        "prosody_reset_accumulate_vs_prosody_hold_only": _comparison_report(
            cells["prosody_hold_only"].row_final_buckets,
            cells["prosody_reset_accumulate"].row_final_buckets,
        ),
    }
    protocol: dict[str, object] = {
        "capture_loop": "production-sherpa-capture-loop",
        "transport": "in-memory-f32le",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "frame_samples": FRAME_SAMPLES,
        "zero_tail_samples": ZERO_TAIL_SAMPLES,
        "zero_tail_ms": ZERO_TAIL_MS,
        "source_alignment_padding_max_samples": FRAME_SAMPLES - 1,
        "input_calibration_enabled": False,
        "word_timing_supplied": False,
        "ground_truth_speech_timing_supplied": False,
        "allow_early": False,
        "adaptive_floor": False,
        "policy_state": "fresh-per-row",
        "acoustic_rule2_ms": round(ACOUSTIC_RULE2_SEC * 1_000),
        "semantic_max_wait_ms": round(ENDPOINT_MAX_SILENCE_SEC * 1_000),
        "logical_rule3_ms": round(LOGICAL_RULE3_SEC * 1_000),
        "complete_threshold": COMPLETE_THRESHOLD,
        "incomplete_threshold": INCOMPLETE_THRESHOLD,
        "asr_provider": ASR_PROVIDER,
        "asr_threads": ASR_THREADS,
    }
    limitations: dict[str, object] = {
        "diagnostic_only": True,
        "capture_loop_counterfactual": True,
        "hold_only_control_included": True,
        "native_reset_accumulate_evaluated": True,
        "early_commit_evaluated": False,
        "posthoc_turn_join_evaluated": False,
        "live_device_evidence": False,
        "model_quality_promotion": False,
        "runtime_default_authority": False,
        "wall_time_includes_paced_replay": True,
        "peak_rss_is_process_high_water": True,
    }
    if logical_turn_shadow:
        protocol["raw_composition_shadow_enabled"] = True
        limitations["raw_composition_shadow_evaluated"] = True
    if logical_turn_terminal_lineage:
        protocol["selected_terminal_lineage_enabled"] = True
        limitations["selected_terminal_lineage_evaluated"] = True
        limitations["async_terminal_delivery_evaluated"] = False
    if logical_turn_terminal_lineage:
        schema_version = TERMINAL_LINEAGE_SCHEMA_VERSION
        kind = TERMINAL_LINEAGE_KIND
    elif logical_turn_shadow:
        schema_version = SHADOW_SCHEMA_VERSION
        kind = SHADOW_KIND
    else:
        schema_version = SCHEMA_VERSION
        kind = KIND
    report: dict[str, object] = {
        "schema_version": schema_version,
        "kind": kind,
        "execution_complete": True,
        "quality_verdict": "diagnostic_only",
        "evidence_scope": "capture-loop-counterfactual",
        "bindings": {
            "corpus": {
                "schema_version": 2,
                "sha256": EXPECTED_CORPUS_SHA256,
                "cases": EXPECTED_CASES,
                "source_bytes": EXPECTED_SOURCE_BYTES,
                "strata": [
                    {"tag": tag, "cases": count}
                    for tag, count in zip(
                        STRATA, EXPECTED_STRATUM_COUNTS, strict=True
                    )
                ],
            },
            "production": {
                "device_profile": prepared.device_profile,
                "final_stt_profile": prepared.final_stt_profile,
                "final_stt_profile_sha256": prepared.final_stt_profile_sha256,
                "production_config_sha256": prepared.production_config_sha256,
                "execution_config_sha256": prepared.execution_config_sha256,
                "replay_safety_sha256": prepared.replay_safety_sha256,
                "active_artifacts_sha256": prepared.active_artifacts_sha256,
                "active_artifact_roots": prepared.active_artifact_roots,
                "active_artifact_files": prepared.active_artifact_files,
                "active_artifact_bytes": prepared.active_artifact_bytes,
                "source_revision": prepared.source_revision,
                "runtime_identities_sha256": _canonical_sha256(
                    prepared.runtime_identities
                ),
                "binding_sha256": _prepared_binding(prepared),
            },
            "smart_turn": {
                "sha256": model.sha256,
                "size_bytes": model.size_bytes,
                "provider": "CPUExecutionProvider",
                "threads": 1,
            },
            "source": {
                "sha256": source_sha256,
                "files": len(
                    _source_files(
                        logical_turn_shadow=logical_turn_shadow,
                        logical_turn_terminal_lineage=(
                            logical_turn_terminal_lineage
                        ),
                    )
                ),
            },
        },
        "protocol": protocol,
        "cells": {name: dict(cells[name].report) for name in _CELL_NAMES},
        "comparison": comparisons,
        "limitations": limitations,
    }
    _validate_report(report)
    return report


def _require_keys(value: object, expected: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise EdaccEndpointIntegrityError()
    return value


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EdaccEndpointIntegrityError()
    return value


def _nonnegative_number(value: object, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise EdaccEndpointIntegrityError()
    return float(value)


def _rounded_ratio(numerator: int, denominator: int) -> float:
    if numerator < 0 or denominator <= 0:
        raise EdaccEndpointIntegrityError()
    return round(numerator / denominator, 4)


def _safe_leaf_walk(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or key in _FORBIDDEN_REPORT_KEYS:
                raise EdaccEndpointIntegrityError()
            _safe_leaf_walk(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _safe_leaf_walk(child)
        return
    if isinstance(value, str):
        if (
            value not in _SAFE_STATIC_STRINGS
            and not _is_sha256(value)
            and not _is_source_revision(value)
        ):
            # Closed transition labels are safe enum pairs, never source values.
            if "->" not in value:
                raise EdaccEndpointIntegrityError()
            left, separator, right = value.partition("->")
            allowed = {
                *_FINAL_BUCKETS,
                *(state.value for state in CompletionScoreState),
                *(basis.value for basis in TurnDecisionBasis),
            }
            if separator != "->" or left not in allowed or right not in allowed:
                raise EdaccEndpointIntegrityError()
        return
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise EdaccEndpointIntegrityError()


def _validate_cell_report(
    value: object,
    *,
    candidate: bool,
    reset_accumulate: bool,
    logical_turn_shadow: bool = False,
    logical_turn_terminal_lineage: bool = False,
) -> None:
    if (
        type(logical_turn_shadow) is not bool
        or type(logical_turn_terminal_lineage) is not bool
        or (logical_turn_terminal_lineage and not logical_turn_shadow)
    ):
        raise EdaccEndpointIntegrityError()
    expected_keys = {
        "coverage",
        "turn_integrity",
        "joined_selected_final_accuracy",
        "vad_speech_end_to_endpoint_ms",
        "transcript_aborts",
        "resources",
        "config_sha256",
        "provider",
        "asr_threads",
        "semantic_hold_reset_enabled",
        "model_load_ms",
        "endpoint_trace",
    }
    if logical_turn_shadow:
        expected_keys.add("logical_turn_shadow")
    if logical_turn_terminal_lineage:
        expected_keys.add("logical_turn_terminal_lineage")
    cell = _require_keys(
        value,
        expected_keys,
    )
    coverage = _require_keys(cell["coverage"], {"rows", "complete"})
    integrity = _require_keys(
        cell["turn_integrity"],
        {"finals", "zero_final_rows", "single_final_rows", "multi_final_rows"},
    )
    accuracy = _require_keys(
        cell["joined_selected_final_accuracy"],
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
            "exact_rows",
            "exact_rate",
        },
    )
    delay = _require_keys(
        cell["vad_speech_end_to_endpoint_ms"],
        {"samples", "p50", "p95", "max"},
    )
    aborts = _require_keys(
        cell["transcript_aborts"],
        {"total", "terminal_events", "reason_counts"},
    )
    abort_counts = _require_keys(
        aborts["reason_counts"],
        {reason.value for reason in TranscriptAbortReason},
    )
    resources = _require_keys(
        cell["resources"],
        {
            "row_wall_ms_total",
            "row_wall_ms_p50",
            "row_wall_ms_p95",
            "process_peak_rss_mb",
        },
    )
    trace = _require_keys(
        cell["endpoint_trace"],
        {
            "evaluations",
            "state_counts",
            "basis_counts",
            "state_transitions",
            "basis_transitions",
            "hold_rows",
            "hold_ticks",
            "hard_boundary_count",
            "max_wait_count",
        },
    )
    state_counts = _require_keys(
        trace["state_counts"], {state.value for state in CompletionScoreState}
    )
    basis_counts = _require_keys(
        trace["basis_counts"], {basis.value for basis in TurnDecisionBasis}
    )
    state_transitions = _require_keys(
        trace["state_transitions"],
        set(_full_transitions(CompletionScoreState)),
    )
    basis_transitions = _require_keys(
        trace["basis_transitions"],
        set(_full_transitions(TurnDecisionBasis)),
    )
    state_count_values = {
        key: _nonnegative_int(item) for key, item in state_counts.items()
    }
    basis_count_values = {
        key: _nonnegative_int(item) for key, item in basis_counts.items()
    }
    state_transition_values = {
        key: _nonnegative_int(item) for key, item in state_transitions.items()
    }
    basis_transition_values = {
        key: _nonnegative_int(item) for key, item in basis_transitions.items()
    }
    state_outgoing = {
        state.value: sum(
            state_transition_values[f"{state.value}->{right.value}"]
            for right in CompletionScoreState
        )
        for state in CompletionScoreState
    }
    state_incoming = {
        state.value: sum(
            state_transition_values[f"{left.value}->{state.value}"]
            for left in CompletionScoreState
        )
        for state in CompletionScoreState
    }
    basis_outgoing = {
        basis.value: sum(
            basis_transition_values[f"{basis.value}->{right.value}"]
            for right in TurnDecisionBasis
        )
        for basis in TurnDecisionBasis
    }
    basis_incoming = {
        basis.value: sum(
            basis_transition_values[f"{left.value}->{basis.value}"]
            for left in TurnDecisionBasis
        )
        for basis in TurnDecisionBasis
    }
    rows = _nonnegative_int(coverage["rows"])
    finals = _nonnegative_int(integrity["finals"])
    zero_rows = _nonnegative_int(integrity["zero_final_rows"])
    single_rows = _nonnegative_int(integrity["single_final_rows"])
    multi_rows = _nonnegative_int(integrity["multi_final_rows"])
    evaluations = _nonnegative_int(trace["evaluations"])
    hold_rows = _nonnegative_int(trace["hold_rows"])
    hold_ticks = _nonnegative_int(trace["hold_ticks"])
    word_errors = _nonnegative_int(accuracy["word_errors"])
    substitutions = _nonnegative_int(accuracy["substitutions"])
    insertions = _nonnegative_int(accuracy["insertions"])
    deletions = _nonnegative_int(accuracy["deletions"])
    reference_words = _nonnegative_int(accuracy["reference_words"])
    hypothesis_words = _nonnegative_int(accuracy["hypothesis_words"])
    character_errors = _nonnegative_int(accuracy["character_errors"])
    reference_characters = _nonnegative_int(accuracy["reference_characters"])
    hypothesis_characters = _nonnegative_int(accuracy["hypothesis_characters"])
    exact_rows = _nonnegative_int(accuracy["exact_rows"])
    wer = _nonnegative_number(accuracy["wer"])
    cer = _nonnegative_number(accuracy["cer"])
    exact_rate = _nonnegative_number(accuracy["exact_rate"])
    delay_samples = _nonnegative_int(delay["samples"])
    delay_p50 = _nonnegative_number(delay["p50"], optional=True)
    delay_p95 = _nonnegative_number(delay["p95"], optional=True)
    delay_max = _nonnegative_number(delay["max"], optional=True)
    wall_total = _nonnegative_number(resources["row_wall_ms_total"])
    wall_p50 = _nonnegative_number(resources["row_wall_ms_p50"])
    wall_p95 = _nonnegative_number(resources["row_wall_ms_p95"])
    _nonnegative_number(resources["process_peak_rss_mb"], optional=True)
    _nonnegative_number(cell["model_load_ms"])
    abort_total = _nonnegative_int(aborts["total"])
    terminal_events = _nonnegative_int(aborts["terminal_events"])
    if (
        coverage["complete"] is not True
        or rows != EXPECTED_CASES
        or zero_rows + single_rows + multi_rows != rows
        or finals < single_rows + 2 * multi_rows
        or _nonnegative_int(accuracy["transcript_runs"]) != rows
        or word_errors != substitutions + insertions + deletions
        or reference_words <= 0
        or reference_characters <= 0
        or hypothesis_words != reference_words - deletions + insertions
        or substitutions + deletions > reference_words
        or rows - exact_rows > word_errors
        or rows - exact_rows > character_errors
        or (
            exact_rows == rows
            and (word_errors != 0 or character_errors != 0)
        )
        or abs(reference_characters - hypothesis_characters) > character_errors
        or wer != _rounded_ratio(word_errors, reference_words)
        or cer != _rounded_ratio(character_errors, reference_characters)
        or exact_rows > rows
        or exact_rate != _rounded_ratio(exact_rows, rows)
        or exact_rate > 1.0
        or hypothesis_words < 0
        or hypothesis_characters < 0
        or delay_samples != finals
        or (
            delay_samples == 0
            and any(value is not None for value in (delay_p50, delay_p95, delay_max))
        )
        or (
            delay_samples > 0
            and (
                delay_p50 is None
                or delay_p95 is None
                or delay_max is None
                or not delay_p50 <= delay_p95 <= delay_max
            )
        )
        or wall_p50 is None
        or wall_p95 is None
        or not wall_p50 <= wall_p95 <= wall_total
        or sum(_nonnegative_int(item) for item in abort_counts.values())
        != abort_total
        or terminal_events != finals + abort_total
        or not _is_sha256(cell["config_sha256"])
        or cell["provider"] != ASR_PROVIDER
        or cell["asr_threads"] != ASR_THREADS
        or cell["semantic_hold_reset_enabled"] is not reset_accumulate
        or evaluations < rows
        or hold_rows > rows
        or hold_ticks > evaluations
        or hold_rows > hold_ticks
        or sum(state_count_values.values()) != evaluations
        or sum(basis_count_values.values()) != evaluations
        or sum(state_transition_values.values()) != evaluations - rows
        or sum(basis_transition_values.values()) != evaluations - rows
        or any(
            state_outgoing[name] > count or state_incoming[name] > count
            for name, count in state_count_values.items()
        )
        or any(
            basis_outgoing[name] > count or basis_incoming[name] > count
            for name, count in basis_count_values.items()
        )
        or basis_count_values[TurnDecisionBasis.SEMANTIC_HOLD.value]
        != hold_ticks
        or hold_ticks
        > state_count_values[CompletionScoreState.SCORED.value]
        or (hold_ticks == 0) != (hold_rows == 0)
        or _nonnegative_int(trace["hard_boundary_count"])
        != state_count_values[CompletionScoreState.HARD_BOUNDARY.value]
        or _nonnegative_int(trace["max_wait_count"])
        != basis_count_values[TurnDecisionBasis.MAX_WAIT.value]
        or state_count_values[CompletionScoreState.HARD_BOUNDARY.value]
        != basis_count_values[TurnDecisionBasis.MAX_WAIT.value]
        or basis_count_values[TurnDecisionBasis.SEMANTIC_EARLY.value] != 0
        or state_count_values[CompletionScoreState.DETECTOR_ERROR.value] != 0
        or (
            not candidate
            and (
                hold_ticks != 0
                or hold_rows != 0
                or any(
                    state_count_values[state.value] != 0
                    for state in CompletionScoreState
                    if state
                    not in {
                        CompletionScoreState.UNAVAILABLE,
                        CompletionScoreState.HARD_BOUNDARY,
                    }
                )
                or any(
                    basis_count_values[basis.value] != 0
                    for basis in TurnDecisionBasis
                    if basis
                    not in {
                        TurnDecisionBasis.ACOUSTIC,
                        TurnDecisionBasis.MAX_WAIT,
                    }
                )
            )
        )
        or (
            candidate
            and (
                hold_ticks <= 0
                or state_count_values[CompletionScoreState.UNAVAILABLE.value] != 0
                or state_count_values[CompletionScoreState.SCORED.value] <= 0
                or basis_count_values[TurnDecisionBasis.SEMANTIC_WAIT.value]
                != 0
            )
        )
    ):
        raise EdaccEndpointIntegrityError()
    if logical_turn_shadow:
        _validate_shadow_cell_report(
            cell["logical_turn_shadow"],
            reset_accumulate=reset_accumulate,
            minimum_paired_terminals=finals,
        )
    if logical_turn_terminal_lineage:
        _validate_terminal_lineage_cell_report(
            cell["logical_turn_terminal_lineage"],
            raw_shadow=cell["logical_turn_shadow"],
            finals=finals,
            abort_total=abort_total,
            abort_counts=abort_counts,
        )


def _validate_shadow_cell_report(
    value: object,
    *,
    reset_accumulate: bool,
    minimum_paired_terminals: int,
) -> None:
    try:
        from always_on_agent.logical_turn_shadow import (
            validate_logical_turn_shadow_report,
        )

        validate_logical_turn_shadow_report(value)
        if not isinstance(value, Mapping):
            raise EdaccEndpointIntegrityError()
        coverage = value["coverage"]
        lifecycle = value["lifecycle"]
        parity = value["parity"]
        health = value["health"]
        minimum_paired = _nonnegative_int(minimum_paired_terminals)
        if not all(
            isinstance(section, Mapping)
            for section in (coverage, lifecycle, parity, health)
        ):
            raise EdaccEndpointIntegrityError()
        multi_epoch_commits = _nonnegative_int(parity["multi_epoch_commits"])
        if (
            coverage["runs"] != EXPECTED_CASES
            or coverage["closed_runs"] != EXPECTED_CASES
            or coverage["active_at_report"] != 0
            or coverage["complete"] is not True
            or health["ok"] is not True
            or any(
                _nonnegative_int(health[name]) != 0
                for name in (
                    "coordinator_rejections",
                    "bounds_overflows",
                    "internal_errors",
                    "late_observations",
                )
            )
            or any(
                _nonnegative_int(parity[name]) != 0
                for name in (
                    "composition_mismatches",
                    "extra_legacy_terminals",
                    "missing_legacy_terminals",
                )
            )
            or parity["legacy_terminals"] != parity["paired_terminals"]
            or parity["paired_terminals"] != parity["composition_matches"]
            or _nonnegative_int(parity["paired_terminals"])
            < minimum_paired
            or (
                reset_accumulate
                and (
                    multi_epoch_commits <= 0
                    or _nonnegative_int(lifecycle["held_native_finals"]) <= 0
                    or parity["evidence_sufficient"] is not True
                    or parity["exact"] is not True
                )
            )
            or (
                not reset_accumulate
                and (
                    multi_epoch_commits != 0
                    or parity["evidence_sufficient"] is not False
                    or parity["exact"] is not False
                )
            )
        ):
            raise EdaccEndpointIntegrityError()
    except (KeyError, TypeError, ValueError, EdaccEndpointIntegrityError):
        raise EdaccEndpointIntegrityError() from None


def _validate_terminal_lineage_cell_report(
    value: object,
    *,
    raw_shadow: object,
    finals: int,
    abort_total: int,
    abort_counts: Mapping[str, object],
) -> None:
    try:
        from always_on_agent.logical_turn_shadow import (
            validate_logical_turn_terminal_lineage_report,
        )

        validate_logical_turn_terminal_lineage_report(value)
        if not isinstance(value, Mapping) or not isinstance(raw_shadow, Mapping):
            raise EdaccEndpointIntegrityError()
        coverage = value["coverage"]
        outcomes = value["outcomes"]
        lineage = value["lineage"]
        health = value["health"]
        raw_parity = raw_shadow["parity"]
        if not all(
            isinstance(section, Mapping)
            for section in (coverage, outcomes, lineage, health, raw_parity)
        ):
            raise EdaccEndpointIntegrityError()
        selected = _nonnegative_int(finals)
        aborted = _nonnegative_int(abort_total)
        raw_commits = _nonnegative_int(lineage["raw_commits"])
        outer_reasons = {
            reason.value: _nonnegative_int(abort_counts[reason.value])
            for reason in TranscriptAbortReason
        }
        if (
            coverage["runs"] != EXPECTED_CASES
            or coverage["closed_runs"] != EXPECTED_CASES
            or coverage["complete"] is not True
            or coverage["terminal_observations"] != selected + aborted
            or outcomes["selected_finals"] != selected
            or outcomes["typed_aborts"] != aborted
            or outcomes["abort_reason_counts"] != outer_reasons
            or raw_parity["paired_terminals"] != raw_commits
            or raw_commits != selected + aborted
            or lineage["claimed_raw_commits"] != raw_commits
            or lineage["bound_raw_commits"] != raw_commits
            or lineage["matched_selected_finals"] != selected
            or lineage["matched_typed_aborts"] != aborted
            or lineage["matched_abort_reason_counts"] != outer_reasons
            or selected <= 0
            or any(
                _nonnegative_int(lineage[name]) != 0
                for name in (
                    "unscoped_selected_finals",
                    "unscoped_typed_aborts",
                    "duplicate_selected_finals",
                    "duplicate_typed_aborts",
                    "missing_downstream_terminals",
                    "unbound_raw_commits",
                )
            )
            or lineage["evidence_sufficient"] is not True
            or lineage["exact"] is not True
            or health["ok"] is not True
            or any(
                _nonnegative_int(health[name]) != 0
                for name in (
                    "binding_rejections",
                    "terminal_rejections",
                    "internal_errors",
                    "late_observations",
                )
            )
        ):
            raise EdaccEndpointIntegrityError()
    except (KeyError, TypeError, ValueError, EdaccEndpointIntegrityError):
        raise EdaccEndpointIntegrityError() from None


def _validate_comparison_report(
    value: object,
    *,
    control_cell: Mapping[str, object],
    candidate_cell: Mapping[str, object],
) -> None:
    comparison = _require_keys(
        value,
        {
            "row_final_count_transitions",
            "control_multi_to_candidate_single_rows",
            "control_single_to_candidate_multi_rows",
            "candidate_reduced_multi_rows",
        },
    )
    transitions = _require_keys(
        comparison["row_final_count_transitions"],
        {f"{left}->{right}" for left in _FINAL_BUCKETS for right in _FINAL_BUCKETS},
    )
    control_integrity = _require_keys(
        control_cell["turn_integrity"],
        {"finals", "zero_final_rows", "single_final_rows", "multi_final_rows"},
    )
    candidate_integrity = _require_keys(
        candidate_cell["turn_integrity"],
        {"finals", "zero_final_rows", "single_final_rows", "multi_final_rows"},
    )
    transition_counts = {
        key: _nonnegative_int(item) for key, item in transitions.items()
    }
    control_marginals = {
        bucket: sum(
            transition_counts[f"{bucket}->{candidate_bucket}"]
            for candidate_bucket in _FINAL_BUCKETS
        )
        for bucket in _FINAL_BUCKETS
    }
    candidate_marginals = {
        bucket: sum(
            transition_counts[f"{control_bucket}->{bucket}"]
            for control_bucket in _FINAL_BUCKETS
        )
        for bucket in _FINAL_BUCKETS
    }
    expected_control_marginals = {
        "zero": _nonnegative_int(control_integrity["zero_final_rows"]),
        "single": _nonnegative_int(control_integrity["single_final_rows"]),
        "multi": _nonnegative_int(control_integrity["multi_final_rows"]),
    }
    expected_candidate_marginals = {
        "zero": _nonnegative_int(candidate_integrity["zero_final_rows"]),
        "single": _nonnegative_int(candidate_integrity["single_final_rows"]),
        "multi": _nonnegative_int(candidate_integrity["multi_final_rows"]),
    }
    control_accuracy = control_cell["joined_selected_final_accuracy"]
    candidate_accuracy = candidate_cell["joined_selected_final_accuracy"]
    if (
        sum(transition_counts.values()) != EXPECTED_CASES
        or control_marginals != expected_control_marginals
        or candidate_marginals != expected_candidate_marginals
        or control_accuracy["reference_words"]
        != candidate_accuracy["reference_words"]
        or control_accuracy["reference_characters"]
        != candidate_accuracy["reference_characters"]
        or comparison["control_multi_to_candidate_single_rows"]
        != transition_counts["multi->single"]
        or comparison["control_single_to_candidate_multi_rows"]
        != transition_counts["single->multi"]
        or comparison["candidate_reduced_multi_rows"]
        is not (transition_counts["multi->single"] > 0)
    ):
        raise EdaccEndpointIntegrityError()


def _validate_report(value: object) -> None:
    report = _require_keys(
        value,
        {
            "schema_version",
            "kind",
            "execution_complete",
            "quality_verdict",
            "evidence_scope",
            "bindings",
            "protocol",
            "cells",
            "comparison",
            "limitations",
        },
    )
    identity = (report["schema_version"], report["kind"])
    if identity == (SCHEMA_VERSION, KIND):
        logical_turn_shadow = False
        logical_turn_terminal_lineage = False
    elif identity == (SHADOW_SCHEMA_VERSION, SHADOW_KIND):
        logical_turn_shadow = True
        logical_turn_terminal_lineage = False
    elif identity == (
        TERMINAL_LINEAGE_SCHEMA_VERSION,
        TERMINAL_LINEAGE_KIND,
    ):
        logical_turn_shadow = True
        logical_turn_terminal_lineage = True
    else:
        raise EdaccEndpointIntegrityError()
    bindings = _require_keys(
        report["bindings"], {"corpus", "production", "smart_turn", "source"}
    )
    corpus = _require_keys(
        bindings["corpus"],
        {"schema_version", "sha256", "cases", "source_bytes", "strata"},
    )
    production = _require_keys(
        bindings["production"],
        {
            "device_profile",
            "final_stt_profile",
            "final_stt_profile_sha256",
            "production_config_sha256",
            "execution_config_sha256",
            "replay_safety_sha256",
            "active_artifacts_sha256",
            "active_artifact_roots",
            "active_artifact_files",
            "active_artifact_bytes",
            "source_revision",
            "runtime_identities_sha256",
            "binding_sha256",
        },
    )
    smart_turn = _require_keys(
        bindings["smart_turn"], {"sha256", "size_bytes", "provider", "threads"}
    )
    source = _require_keys(bindings["source"], {"sha256", "files"})
    protocol_keys = {
        "capture_loop",
        "transport",
        "sample_rate_hz",
        "frame_samples",
        "zero_tail_samples",
        "zero_tail_ms",
        "source_alignment_padding_max_samples",
        "input_calibration_enabled",
        "word_timing_supplied",
        "ground_truth_speech_timing_supplied",
        "allow_early",
        "adaptive_floor",
        "policy_state",
        "acoustic_rule2_ms",
        "semantic_max_wait_ms",
        "logical_rule3_ms",
        "complete_threshold",
        "incomplete_threshold",
        "asr_provider",
        "asr_threads",
    }
    if logical_turn_shadow:
        protocol_keys.add("raw_composition_shadow_enabled")
    if logical_turn_terminal_lineage:
        protocol_keys.add("selected_terminal_lineage_enabled")
    protocol = _require_keys(report["protocol"], protocol_keys)
    cells = _require_keys(report["cells"], set(_CELL_NAMES))
    comparisons = _require_keys(report["comparison"], set(_COMPARISON_NAMES))
    limitation_keys = {
        "diagnostic_only",
        "capture_loop_counterfactual",
        "hold_only_control_included",
        "native_reset_accumulate_evaluated",
        "early_commit_evaluated",
        "posthoc_turn_join_evaluated",
        "live_device_evidence",
        "model_quality_promotion",
        "runtime_default_authority",
        "wall_time_includes_paced_replay",
        "peak_rss_is_process_high_water",
    }
    if logical_turn_shadow:
        limitation_keys.add("raw_composition_shadow_evaluated")
    if logical_turn_terminal_lineage:
        limitation_keys.update(
            {
                "selected_terminal_lineage_evaluated",
                "async_terminal_delivery_evaluated",
            }
        )
    limitations = _require_keys(report["limitations"], limitation_keys)
    _validate_cell_report(
        cells["acoustic"],
        candidate=False,
        reset_accumulate=False,
        logical_turn_shadow=logical_turn_shadow,
        logical_turn_terminal_lineage=logical_turn_terminal_lineage,
    )
    _validate_cell_report(
        cells["prosody_hold_only"],
        candidate=True,
        reset_accumulate=False,
        logical_turn_shadow=logical_turn_shadow,
        logical_turn_terminal_lineage=logical_turn_terminal_lineage,
    )
    _validate_cell_report(
        cells["prosody_reset_accumulate"],
        candidate=True,
        reset_accumulate=True,
        logical_turn_shadow=logical_turn_shadow,
        logical_turn_terminal_lineage=logical_turn_terminal_lineage,
    )
    _validate_comparison_report(
        comparisons["prosody_hold_only_vs_acoustic"],
        control_cell=cells["acoustic"],
        candidate_cell=cells["prosody_hold_only"],
    )
    _validate_comparison_report(
        comparisons["prosody_reset_accumulate_vs_acoustic"],
        control_cell=cells["acoustic"],
        candidate_cell=cells["prosody_reset_accumulate"],
    )
    _validate_comparison_report(
        comparisons["prosody_reset_accumulate_vs_prosody_hold_only"],
        control_cell=cells["prosody_hold_only"],
        candidate_cell=cells["prosody_reset_accumulate"],
    )
    if (
        report["execution_complete"] is not True
        or report["quality_verdict"] != "diagnostic_only"
        or report["evidence_scope"] != "capture-loop-counterfactual"
        or corpus["schema_version"] != 2
        or corpus["sha256"] != EXPECTED_CORPUS_SHA256
        or corpus["cases"] != EXPECTED_CASES
        or corpus["source_bytes"] != EXPECTED_SOURCE_BYTES
        or production["device_profile"] != DEVICE_PROFILE
        or production["final_stt_profile"] != FINAL_STT_PROFILE
        or any(
            not _is_sha256(production[name])
            for name in (
                "final_stt_profile_sha256",
                "production_config_sha256",
                "execution_config_sha256",
                "replay_safety_sha256",
                "active_artifacts_sha256",
                "runtime_identities_sha256",
                "binding_sha256",
            )
        )
        or not _is_source_revision(production["source_revision"])
        or not all(
            _nonnegative_int(production[name]) > 0
            for name in (
                "active_artifact_roots",
                "active_artifact_files",
                "active_artifact_bytes",
            )
        )
        or smart_turn["sha256"] != SMART_TURN_MODEL_SHA256
        or smart_turn["size_bytes"] != SMART_TURN_MODEL_BYTES
        or smart_turn["provider"] != "CPUExecutionProvider"
        or smart_turn["threads"] != 1
        or not _is_sha256(source["sha256"])
        or source["files"]
        != len(
            _source_files(
                logical_turn_shadow=logical_turn_shadow,
                logical_turn_terminal_lineage=logical_turn_terminal_lineage,
            )
        )
        or protocol
        != {
            "capture_loop": "production-sherpa-capture-loop",
            "transport": "in-memory-f32le",
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "frame_samples": FRAME_SAMPLES,
            "zero_tail_samples": ZERO_TAIL_SAMPLES,
            "zero_tail_ms": ZERO_TAIL_MS,
            "source_alignment_padding_max_samples": FRAME_SAMPLES - 1,
            "input_calibration_enabled": False,
            "word_timing_supplied": False,
            "ground_truth_speech_timing_supplied": False,
            "allow_early": False,
            "adaptive_floor": False,
            "policy_state": "fresh-per-row",
            "acoustic_rule2_ms": 800,
            "semantic_max_wait_ms": 1_600,
            "logical_rule3_ms": 20_000,
            "complete_threshold": COMPLETE_THRESHOLD,
            "incomplete_threshold": INCOMPLETE_THRESHOLD,
            "asr_provider": ASR_PROVIDER,
            "asr_threads": ASR_THREADS,
            **(
                {"raw_composition_shadow_enabled": True}
                if logical_turn_shadow
                else {}
            ),
            **(
                {"selected_terminal_lineage_enabled": True}
                if logical_turn_terminal_lineage
                else {}
            ),
        }
        or limitations
        != {
            "diagnostic_only": True,
            "capture_loop_counterfactual": True,
            "hold_only_control_included": True,
            "native_reset_accumulate_evaluated": True,
            "early_commit_evaluated": False,
            "posthoc_turn_join_evaluated": False,
            "live_device_evidence": False,
            "model_quality_promotion": False,
            "runtime_default_authority": False,
            "wall_time_includes_paced_replay": True,
            "peak_rss_is_process_high_water": True,
            **(
                {"raw_composition_shadow_evaluated": True}
                if logical_turn_shadow
                else {}
            ),
            **(
                {
                    "selected_terminal_lineage_evaluated": True,
                    "async_terminal_delivery_evaluated": False,
                }
                if logical_turn_terminal_lineage
                else {}
            ),
        }
    ):
        raise EdaccEndpointIntegrityError()
    if not isinstance(corpus["strata"], list) or corpus["strata"] != [
        {"tag": tag, "cases": count}
        for tag, count in zip(STRATA, EXPECTED_STRATUM_COUNTS, strict=True)
    ]:
        raise EdaccEndpointIntegrityError()
    if logical_turn_terminal_lineage and sum(
        _nonnegative_int(
            cells[name]["logical_turn_terminal_lineage"]["lineage"][
                "matched_abort_reason_counts"
            ][TranscriptAbortReason.INPUT_REJECTED.value]
        )
        for name in _CELL_NAMES
    ) <= 0:
        raise EdaccEndpointIntegrityError()
    _safe_leaf_walk(report)
    _canonical_bytes(report)


def evaluate_edacc_endpoint_integrity(
    *,
    corpus_path: Path | str,
    config_path: Path | str,
    local_config_path: Path | str,
    smart_turn_model: Path | str,
    logical_turn_shadow: bool = False,
    logical_turn_terminal_lineage: bool = False,
) -> Mapping[str, object]:
    """Execute all three fixed cells and return one validated aggregate report."""

    try:
        if (
            type(logical_turn_shadow) is not bool
            or type(logical_turn_terminal_lineage) is not bool
            or (logical_turn_terminal_lineage and not logical_turn_shadow)
        ):
            raise EdaccEndpointIntegrityError()
        prepared = prepare_production(
            config_path=config_path,
            local_config_path=local_config_path,
            device_profile=DEVICE_PROFILE,
            final_stt_profile=FINAL_STT_PROFILE,
        )
        corpus = load_production_corpus(corpus_path)
        _validate_exact_corpus(corpus)
        model = _snapshot_model(smart_turn_model)
        source_sha256 = _source_binding(
            logical_turn_shadow=logical_turn_shadow,
            logical_turn_terminal_lineage=logical_turn_terminal_lineage,
        )
        prepared_sha256 = _prepared_binding(prepared)
        base = _base_config(prepared)
        cells: dict[str, _CellResult] = {}
        for name in _CELL_NAMES:
            cell_options: dict[str, object] = {"cell": name}
            if logical_turn_shadow:
                cell_options["logical_turn_shadow"] = True
            if logical_turn_terminal_lineage:
                cell_options["logical_turn_terminal_lineage"] = True
            cells[name] = _run_cell(
                corpus,
                _cell_config(base, cell=name, model=model),
                **cell_options,
            )

        verify_production_inputs(
            prepared,
            corpus,
            config_path=config_path,
            local_config_path=local_config_path,
        )
        _validate_exact_corpus(corpus)
        rebound_model = _snapshot_model(model.path)
        if rebound_model != model:
            raise EdaccEndpointIntegrityError()
        rebound_source_sha256 = _source_binding(
            logical_turn_shadow=logical_turn_shadow,
            logical_turn_terminal_lineage=logical_turn_terminal_lineage,
        )
        if (
            rebound_source_sha256 != source_sha256
            or _prepared_binding(prepared) != prepared_sha256
        ):
            raise EdaccEndpointIntegrityError()
        return _report(
            prepared=prepared,
            model=model,
            source_sha256=source_sha256,
            cells=cells,
            logical_turn_shadow=logical_turn_shadow,
            logical_turn_terminal_lineage=logical_turn_terminal_lineage,
        )
    except (
        BoundedReadError,
        CaptureReplayEvaluationError,
        ProductionInputError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        EdaccEndpointIntegrityError,
    ):
        raise EdaccEndpointIntegrityError() from None


def _parent_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        stat.S_IMODE(value.st_mode),
        value.st_uid,
        value.st_gid,
    )


def _private_parent(value: os.stat_result) -> bool:
    return bool(
        stat.S_ISDIR(value.st_mode)
        and not stat.S_IMODE(value.st_mode) & 0o077
        and (not hasattr(os, "geteuid") or value.st_uid == os.geteuid())
    )


def _valid_published_file(
    value: os.stat_result,
    *,
    expected_bytes: int,
) -> bool:
    return bool(
        stat.S_ISREG(value.st_mode)
        and value.st_nlink == 1
        and stat.S_IMODE(value.st_mode) == 0o600
        and value.st_size == expected_bytes
        and (not hasattr(os, "geteuid") or value.st_uid == os.geteuid())
    )


def _published_file_identity(value: os.stat_result) -> tuple[int, ...]:
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


def _read_descriptor_exact(descriptor: int, expected_bytes: int) -> bytes:
    if expected_bytes <= 0 or expected_bytes > _MAX_REPORT_BYTES:
        raise EdaccEndpointIntegrityError()
    chunks: list[bytes] = []
    offset = 0
    while offset <= expected_bytes:
        chunk = os.pread(
            descriptor,
            min(1024 * 1024, expected_bytes - offset + 1),
            offset,
        )
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
        if offset > expected_bytes:
            raise EdaccEndpointIntegrityError()
    payload = b"".join(chunks)
    if len(payload) != expected_bytes:
        raise EdaccEndpointIntegrityError()
    return payload


def _close_checked(descriptor: int) -> None:
    os.close(descriptor)


def _report_destination(path: Path | str) -> Path:
    supplied = Path(path).expanduser()
    if (
        not supplied.is_absolute()
        or Path(os.path.abspath(supplied)) != supplied
        or not supplied.name
    ):
        raise EdaccEndpointIntegrityError()
    return supplied


def _preflight_report_destination(path: Path | str) -> Path:
    """Reject an unusable report target before either native cell starts."""

    candidate = _report_destination(path)
    try:
        if _has_git_ancestor(candidate.parent):
            raise EdaccEndpointIntegrityError()
        with opened_directory_nofollow(
            candidate.parent,
            require_private=True,
        ) as (stable_parent, parent_descriptor):
            parent = os.fstat(parent_descriptor)
            if (
                stable_parent != candidate.parent
                or not _private_parent(parent)
                or _parent_identity(candidate.parent.lstat())
                != _parent_identity(parent)
                or candidate.parent.resolve(strict=True) != candidate.parent
            ):
                raise EdaccEndpointIntegrityError()
            try:
                os.stat(
                    candidate.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise EdaccEndpointIntegrityError()
        return candidate
    except EdaccEndpointIntegrityError:
        raise
    except Exception:
        raise EdaccEndpointIntegrityError() from None


def _publish_private_report(path: Path | str, payload: bytes) -> str:
    """Atomically publish exact aggregate bytes or leave no owned output."""

    candidate = _report_destination(path)
    if (
        not isinstance(payload, bytes)
        or not payload
        or len(payload) > _MAX_REPORT_BYTES
    ):
        raise EdaccEndpointIntegrityError()
    parent_descriptor = -1
    writer_descriptor = -1
    reader_descriptor = -1
    temporary_name = f".edacc-endpoint-{secrets.token_hex(16)}.tmp"
    temporary_inode: tuple[int, int] | None = None
    temporary_exists = False
    published_inode: tuple[int, int] | None = None
    published_created = False
    published_succeeded = False
    expected_digest = hashlib.sha256(payload).hexdigest()
    try:
        if _has_git_ancestor(candidate.parent):
            raise EdaccEndpointIntegrityError()
        with opened_directory_nofollow(
            candidate.parent,
            require_private=True,
        ) as (stable_parent, opened_parent):
            if stable_parent != candidate.parent:
                raise EdaccEndpointIntegrityError()
            parent_descriptor = os.dup(opened_parent)
        parent_before = os.fstat(parent_descriptor)
        if not _private_parent(parent_before):
            raise EdaccEndpointIntegrityError()
        try:
            os.stat(
                candidate.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise EdaccEndpointIntegrityError()

        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        writer_descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        temporary_exists = True
        opened_temporary = os.fstat(writer_descriptor)
        temporary_inode = (opened_temporary.st_dev, opened_temporary.st_ino)
        os.fchmod(writer_descriptor, 0o600)
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(writer_descriptor, view[written:])
            if not isinstance(count, int) or count <= 0:
                raise EdaccEndpointIntegrityError()
            written += count
        os.fsync(writer_descriptor)
        temporary = os.fstat(writer_descriptor)
        current_temporary = os.stat(
            temporary_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (temporary.st_dev, temporary.st_ino) != temporary_inode:
            raise EdaccEndpointIntegrityError()
        if (
            not _valid_published_file(temporary, expected_bytes=len(payload))
            or (current_temporary.st_dev, current_temporary.st_ino)
            != temporary_inode
            or _read_descriptor_exact(writer_descriptor, len(payload)) != payload
        ):
            raise EdaccEndpointIntegrityError()
        parent_current = os.fstat(parent_descriptor)
        if (
            _parent_identity(parent_current) != _parent_identity(parent_before)
            or _parent_identity(candidate.parent.lstat())
            != _parent_identity(parent_before)
            or not _private_parent(parent_current)
            or candidate.parent.resolve(strict=True) != candidate.parent
        ):
            raise EdaccEndpointIntegrityError()

        os.link(
            temporary_name,
            candidate.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        published_created = True
        published_inode = temporary_inode
        linked = os.stat(
            candidate.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (linked.st_dev, linked.st_ino) != published_inode:
            raise EdaccEndpointIntegrityError()
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        temporary_exists = False
        os.fsync(parent_descriptor)
        published = os.fstat(writer_descriptor)
        if not _valid_published_file(published, expected_bytes=len(payload)):
            raise EdaccEndpointIntegrityError()
        _close_checked(writer_descriptor)
        writer_descriptor = -1

        read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        read_flags |= getattr(os, "O_NOFOLLOW", 0)
        reader_descriptor = os.open(
            candidate.name,
            read_flags,
            dir_fd=parent_descriptor,
        )
        reread_before = os.fstat(reader_descriptor)
        reread_payload = _read_descriptor_exact(reader_descriptor, len(payload))
        reread_after = os.fstat(reader_descriptor)
        current = os.stat(
            candidate.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not _valid_published_file(reread_before, expected_bytes=len(payload))
            or (reread_before.st_dev, reread_before.st_ino) != published_inode
            or (reread_after.st_dev, reread_after.st_ino) != published_inode
            or (current.st_dev, current.st_ino) != published_inode
            or reread_payload != payload
        ):
            raise EdaccEndpointIntegrityError()
        _close_checked(reader_descriptor)
        reader_descriptor = -1
        current_after_close = os.stat(
            candidate.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        parent_after = os.fstat(parent_descriptor)
        if (
            (current_after_close.st_dev, current_after_close.st_ino)
            != published_inode
            or not _valid_published_file(
                current_after_close,
                expected_bytes=len(payload),
            )
            or _parent_identity(parent_after) != _parent_identity(parent_before)
            or _parent_identity(candidate.parent.lstat())
            != _parent_identity(parent_before)
            or not _private_parent(parent_after)
            or candidate.parent.resolve(strict=True) != candidate.parent
        ):
            raise EdaccEndpointIntegrityError()
        observed_digest = hashlib.sha256(reread_payload).hexdigest()
        if observed_digest != expected_digest:
            raise EdaccEndpointIntegrityError()
        published_succeeded = True
        return observed_digest
    except EdaccEndpointIntegrityError:
        raise
    except Exception:
        raise EdaccEndpointIntegrityError() from None
    finally:
        cleanup_failed = False
        if parent_descriptor >= 0 and not published_succeeded:
            if published_created and published_inode is not None:
                try:
                    current = os.stat(
                        candidate.name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                    if (current.st_dev, current.st_ino) == published_inode:
                        os.unlink(candidate.name, dir_fd=parent_descriptor)
                except FileNotFoundError:
                    pass
                except OSError:
                    cleanup_failed = True
            if temporary_exists:
                try:
                    current = os.stat(
                        temporary_name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                    if temporary_inode is None or (
                        current.st_dev,
                        current.st_ino,
                    ) == temporary_inode:
                        os.unlink(temporary_name, dir_fd=parent_descriptor)
                except FileNotFoundError:
                    pass
                except OSError:
                    cleanup_failed = True
            try:
                os.fsync(parent_descriptor)
            except OSError:
                cleanup_failed = True
        for descriptor in (reader_descriptor, writer_descriptor, parent_descriptor):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    if not published_succeeded:
                        cleanup_failed = True
        if cleanup_failed:
            raise EdaccEndpointIntegrityError() from None


def _verify_published_report(
    path: Path | str,
    *,
    expected_sha256: str,
) -> Mapping[str, object]:
    """Reopen and validate the retained bytes from the guarded parent."""

    candidate = _report_destination(path)
    parent_descriptor = -1
    report_descriptor = -1
    try:
        if not _is_sha256(expected_sha256) or _has_git_ancestor(candidate.parent):
            raise EdaccEndpointIntegrityError()
        with opened_directory_nofollow(
            candidate.parent,
            require_private=True,
        ) as (stable_parent, opened_parent):
            if stable_parent != candidate.parent:
                raise EdaccEndpointIntegrityError()
            parent_descriptor = os.dup(opened_parent)
        parent_before = os.fstat(parent_descriptor)
        if not _private_parent(parent_before):
            raise EdaccEndpointIntegrityError()

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        report_descriptor = os.open(
            candidate.name,
            flags,
            dir_fd=parent_descriptor,
        )
        before = os.fstat(report_descriptor)
        if (
            before.st_size <= 0
            or before.st_size > _MAX_REPORT_BYTES
            or not _valid_published_file(before, expected_bytes=before.st_size)
        ):
            raise EdaccEndpointIntegrityError()
        payload = _read_descriptor_exact(report_descriptor, before.st_size)
        after = os.fstat(report_descriptor)
        current = os.stat(
            candidate.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        parent_after_read = os.fstat(parent_descriptor)
        if (
            _published_file_identity(after) != _published_file_identity(before)
            or _published_file_identity(current) != _published_file_identity(before)
            or _parent_identity(parent_after_read) != _parent_identity(parent_before)
            or _parent_identity(candidate.parent.lstat())
            != _parent_identity(parent_before)
            or not _private_parent(parent_after_read)
            or candidate.parent.resolve(strict=True) != candidate.parent
        ):
            raise EdaccEndpointIntegrityError()
        _close_checked(report_descriptor)
        report_descriptor = -1

        current_after_close = os.stat(
            candidate.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        parent_after_close = os.fstat(parent_descriptor)
        if (
            _published_file_identity(current_after_close)
            != _published_file_identity(before)
            or _parent_identity(parent_after_close) != _parent_identity(parent_before)
            or _parent_identity(candidate.parent.lstat())
            != _parent_identity(parent_before)
            or not _private_parent(parent_after_close)
            or candidate.parent.resolve(strict=True) != candidate.parent
            or hashlib.sha256(payload).hexdigest() != expected_sha256
            or not payload.endswith(b"\n")
        ):
            raise EdaccEndpointIntegrityError()
        try:
            value = json.loads(payload.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise EdaccEndpointIntegrityError() from None
        _validate_report(value)
        if _canonical_bytes(value) + b"\n" != payload:
            raise EdaccEndpointIntegrityError()
        return value
    except EdaccEndpointIntegrityError:
        raise
    except Exception:
        raise EdaccEndpointIntegrityError() from None
    finally:
        for descriptor in (report_descriptor, parent_descriptor):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise EdaccEndpointIntegrityError()


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


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description=(
            "Run the fixed three-cell EdAcc acoustic/Smart-Turn capture-loop "
            "counterfactual and publish aggregate evidence."
        )
    )
    parser.add_argument("--corpus", type=_absolute_path, required=True)
    parser.add_argument("--config", type=_absolute_path, required=True)
    parser.add_argument("--local-config", type=_absolute_path, required=True)
    parser.add_argument("--smart-turn-model", type=_absolute_path, required=True)
    parser.add_argument(
        "--logical-turn-shadow",
        action="store_true",
        help=(
            "add transcript-free raw native-composition shadow evidence; "
            "does not transfer runtime authority"
        ),
    )
    parser.add_argument(
        "--logical-turn-terminal-lineage",
        action="store_true",
        help=(
            "add transcript-free inline selected-final/typed-abort lineage; "
            "requires --logical-turn-shadow and does not cover async delivery"
        ),
    )
    parser.add_argument("--report", type=_absolute_path, required=True)
    parser.add_argument(
        "--watchdog-seconds",
        type=_positive_int,
        default=3_600,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if (
            args.logical_turn_terminal_lineage
            and not args.logical_turn_shadow
        ):
            raise EdaccEndpointIntegrityError()
        _preflight_report_destination(args.report)
        report = evaluate_edacc_endpoint_integrity(
            corpus_path=args.corpus,
            config_path=args.config,
            local_config_path=args.local_config,
            smart_turn_model=args.smart_turn_model,
            logical_turn_shadow=args.logical_turn_shadow,
            logical_turn_terminal_lineage=(
                args.logical_turn_terminal_lineage
            ),
        )
        payload = _canonical_bytes(report) + b"\n"
        published_sha256 = _publish_private_report(args.report, payload)
        cells = report["cells"]
        comparisons = report["comparison"]
        hold_only = cells["prosody_hold_only"]
        reset_accumulate = cells["prosody_reset_accumulate"]
        receipt: Mapping[str, object] = {
            "ok": True,
            "quality_verdict": "diagnostic_only",
            "report_sha256": published_sha256,
            "rows": EXPECTED_CASES,
            "hold_only_hold_rows": hold_only["endpoint_trace"]["hold_rows"],
            "reset_accumulate_hold_rows": reset_accumulate["endpoint_trace"][
                "hold_rows"
            ],
            "reset_multi_row_repairs_vs_acoustic": comparisons[
                "prosody_reset_accumulate_vs_acoustic"
            ]["control_multi_to_candidate_single_rows"],
            "reset_multi_row_repairs_vs_hold_only": comparisons[
                "prosody_reset_accumulate_vs_prosody_hold_only"
            ][
                "control_multi_to_candidate_single_rows"
            ],
        }
        code = 0
    except Exception:  # noqa: BLE001 - private details never reach the terminal
        receipt = _SAFE_ERROR
        code = 2
    print(
        json.dumps(
            receipt,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return code


def _validated_worker_receipt(
    payload: bytes,
    *,
    returncode: int,
) -> Mapping[str, object]:
    if not payload or len(payload) > _MAX_RECEIPT_BYTES:
        raise EdaccEndpointIntegrityError()
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise EdaccEndpointIntegrityError() from None
    if returncode == 0:
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {
                "ok",
                "quality_verdict",
                "report_sha256",
                "rows",
                "hold_only_hold_rows",
                "reset_accumulate_hold_rows",
                "reset_multi_row_repairs_vs_acoustic",
                "reset_multi_row_repairs_vs_hold_only",
            }
            or value.get("ok") is not True
            or value.get("quality_verdict") != "diagnostic_only"
            or not _is_sha256(value.get("report_sha256"))
            or value.get("rows") != EXPECTED_CASES
            or _nonnegative_int(value.get("hold_only_hold_rows"))
            > EXPECTED_CASES
            or _nonnegative_int(value.get("reset_accumulate_hold_rows"))
            > EXPECTED_CASES
            or _nonnegative_int(value.get("reset_multi_row_repairs_vs_acoustic"))
            > EXPECTED_CASES
            or _nonnegative_int(value.get("reset_multi_row_repairs_vs_hold_only"))
            > EXPECTED_CASES
        ):
            raise EdaccEndpointIntegrityError()
        return dict(value)
    if returncode != 2 or value != dict(_SAFE_ERROR):
        raise EdaccEndpointIntegrityError()
    return dict(value)


def _terminate_worker_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=2.0)
        return
    except (subprocess.TimeoutExpired, OSError):
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=2.0)
    except (subprocess.TimeoutExpired, OSError):
        pass


def _run_guarded_worker(
    raw_argv: Sequence[str],
    *,
    timeout_seconds: int,
) -> tuple[int, bytes]:
    environment = dict(os.environ)
    environment[_WORKER_ENV] = "1"
    process = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-m",
            "tools.edacc_endpoint_integrity_eval",
            *raw_argv,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=environment,
        close_fds=True,
        start_new_session=True,
    )
    if process.stdout is None:
        _terminate_worker_process(process)
        raise EdaccEndpointIntegrityError()
    captured = bytearray()
    overflow = threading.Event()
    reader_error = threading.Event()

    def drain() -> None:
        try:
            while True:
                chunk = process.stdout.read(1024)
                if not chunk:
                    return
                remaining = _MAX_RECEIPT_BYTES + 1 - len(captured)
                if remaining > 0:
                    captured.extend(chunk[:remaining])
                if len(chunk) > remaining or len(captured) > _MAX_RECEIPT_BYTES:
                    overflow.set()
        except Exception:  # noqa: BLE001 - parent exposes no worker detail
            reader_error.set()

    reader = threading.Thread(
        target=drain,
        name="edacc-endpoint-receipt",
        daemon=True,
    )
    reader.start()
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _terminate_worker_process(process)
        raise EdaccEndpointIntegrityError() from None
    finally:
        if process.poll() is None:
            _terminate_worker_process(process)
        reader.join(timeout=5.0)
        try:
            process.stdout.close()
        except OSError:
            pass
    if reader.is_alive() or reader_error.is_set() or overflow.is_set():
        raise EdaccEndpointIntegrityError()
    return returncode, bytes(captured)


def guarded_main(argv: Sequence[str] | None = None) -> int:
    """Run exact model work in a killable child with bounded IPC."""

    if os.environ.get(_WORKER_ENV) == "1":
        return main(argv)
    try:
        raw_argv = list(sys.argv[1:] if argv is None else argv)
        args = _parser().parse_args(raw_argv)
        if (
            args.logical_turn_terminal_lineage
            and not args.logical_turn_shadow
        ):
            raise EdaccEndpointIntegrityError()
        if args.watchdog_seconds > _MAX_WATCHDOG_SECONDS:
            raise EdaccEndpointIntegrityError()
        _preflight_report_destination(args.report)
        returncode, payload = _run_guarded_worker(
            raw_argv,
            timeout_seconds=args.watchdog_seconds,
        )
        receipt = _validated_worker_receipt(payload, returncode=returncode)
        if returncode == 0:
            retained_report = _verify_published_report(
                args.report,
                expected_sha256=receipt["report_sha256"],
            )
            retained_cells = retained_report["cells"]
            retained_control = retained_cells["acoustic"]
            retained_hold_only = retained_cells["prosody_hold_only"]
            retained_reset = retained_cells["prosody_reset_accumulate"]
            retained_comparisons = retained_report["comparison"]
            retained_shadow_enabled = (
                retained_report["schema_version"]
                in {
                    SHADOW_SCHEMA_VERSION,
                    TERMINAL_LINEAGE_SCHEMA_VERSION,
                }
            )
            retained_terminal_lineage_enabled = (
                retained_report["schema_version"]
                == TERMINAL_LINEAGE_SCHEMA_VERSION
            )
            if (
                retained_shadow_enabled is not args.logical_turn_shadow
                or retained_terminal_lineage_enabled
                is not args.logical_turn_terminal_lineage
                or receipt["rows"] != retained_control["coverage"]["rows"]
                or receipt["rows"] != retained_hold_only["coverage"]["rows"]
                or receipt["rows"] != retained_reset["coverage"]["rows"]
                or receipt["hold_only_hold_rows"]
                != retained_hold_only["endpoint_trace"]["hold_rows"]
                or receipt["reset_accumulate_hold_rows"]
                != retained_reset["endpoint_trace"]["hold_rows"]
                or receipt["reset_multi_row_repairs_vs_acoustic"]
                != retained_comparisons[
                    "prosody_reset_accumulate_vs_acoustic"
                ]["control_multi_to_candidate_single_rows"]
                or receipt["reset_multi_row_repairs_vs_hold_only"]
                != retained_comparisons[
                    "prosody_reset_accumulate_vs_prosody_hold_only"
                ]["control_multi_to_candidate_single_rows"]
            ):
                raise EdaccEndpointIntegrityError()
        code = returncode
    except Exception:  # noqa: BLE001 - parent exposes no path/model detail
        receipt = _SAFE_ERROR
        code = 2
    print(
        json.dumps(
            receipt,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(guarded_main())
