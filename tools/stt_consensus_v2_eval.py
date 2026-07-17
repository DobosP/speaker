"""Aggregate-only diagnostic for stronger local acoustic final consensus.

The current production-shaped baseline is the exact Small verifier consensus.
This tool compares it with cached local Faster-Whisper Small.en/Base and an
independent Parakeet FastConformer-RNNT.  Whole-utterance votes return an
existing acoustic rendering.  The experimental span repair copies only exact
token spans supported by at least two model families and is fenced against any
controller-control change.

No chatbot, TTS, tool, recorder, network service, or audio device is started.
References, hypotheses, clip identifiers, and paths stay private in memory;
stdout and the optional mode-600 report contain aggregate metrics and digests.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from time import perf_counter_ns
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from always_on_agent.speech_analyzer import exact_control_class
from core.asr_verifier import _exact_tokens, verify_asr_consensus
from core.audio_frontend import AudioResampler
from core.contract import is_stop_command
from tools.local_gpu_stt_eval import (
    _DEFAULT_SMALL_MODEL,
    _DEFAULT_SMALL_REVISION,
    _decode_model,
    _load_machine_config,
    _local_snapshot,
    _production_model_digest,
    _production_rows,
    _tree_digest,
)
import tools.recorded_stt_eval as recorded_stt_eval
from tools.recorded_stt_eval import (
    AccuracyTotals,
    EvaluationPrerequisiteError,
    EvaluationTotals,
    _config_digest,
    _load_corpus,
    _measure,
    _pair_errors,
    _sum_accuracy,
    _write_report,
    compare_candidate,
)
from tools.stt_selector_eval import SourceHypothesis, _unsupported_token_edits


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MANIFEST = _REPO_ROOT / "tests" / "fixtures" / "recorded_voice_manifest.json"
_DEFAULT_SMALL_EN_MODEL = "Systran/faster-whisper-small.en"
_DEFAULT_SMALL_EN_REVISION = "d1d751a5f8271d482d14ca55d9e2deeebbae577f"
_DEFAULT_BASE_MODEL = "Systran/faster-whisper-base"
_DEFAULT_BASE_REVISION = "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
_DEFAULT_PARAKEET_DIR = (
    "pretrained_models/sherpa/"
    "sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming"
)
_PARAKEET_SAMPLE_RATE = 16_000
_PARAKEET_FEATURE_DIM = 80
_PARAKEET_MAX_ACTIVE_PATHS = 4
_PARAKEET_FILES = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
)
_SAFE_ID = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
_SAFE_ERROR = {"ok": False, "error": "stt_consensus_v2_prerequisites_unavailable"}


@dataclass(frozen=True)
class AcousticHypothesis:
    """One private acoustic rendering with an explicit model-family label."""

    source_id: str
    family_id: str
    text: str = field(repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_id, str)
            or _SAFE_ID.fullmatch(self.source_id) is None
            or not isinstance(self.family_id, str)
            or _SAFE_ID.fullmatch(self.family_id) is None
            or not isinstance(self.text, str)
        ):
            raise ValueError("invalid acoustic hypothesis")


class RepairReason(str, Enum):
    EXACT_VOTE = "exact_vote"
    SPAN_REPAIR = "span_repair"
    NO_SUPPORT = "no_support"
    TIE = "tie"
    CONFLICT = "conflict"
    EMPTY = "empty"
    CONTROL_GUARD = "control_guard"
    ERROR = "error"


@dataclass(frozen=True)
class RepairDecision:
    """A private candidate plus aggregate-safe vote/repair provenance."""

    chosen: str = field(repr=False)
    reason: RepairReason
    accepted: bool
    changed: bool
    source_support: int = 0
    family_support: int = 0
    repaired_spans: int = 0
    constructed: bool = False


@dataclass(frozen=True)
class _PrivateCase:
    reference: str = field(repr=False)
    baseline: str = field(repr=False)
    hypotheses: tuple[AcousticHypothesis, ...] = field(repr=False)


@dataclass(frozen=True)
class _ParakeetBatch:
    texts: tuple[str, ...] = field(repr=False)
    load_ms: float
    first_decode_ms: float
    warm_p50_ms: float
    warm_p95_ms: float
    warm_max_ms: float
    repeat_disagreements: int
    empty_outputs: int

    def as_dict(self) -> dict[str, object]:
        return {
            "load_ms": self.load_ms,
            "first_decode_ms": self.first_decode_ms,
            "warm_decode_ms": {
                "p50": self.warm_p50_ms,
                "p95": self.warm_p95_ms,
                "max": self.warm_max_ms,
            },
            "repeat_disagreements": self.repeat_disagreements,
            "empty_outputs": self.empty_outputs,
        }


def _fallback(baseline: str, reason: RepairReason) -> RepairDecision:
    return RepairDecision(baseline, reason, False, False)


def _selected_hypotheses(
    hypotheses: Sequence[AcousticHypothesis], source_ids: Sequence[str]
) -> tuple[AcousticHypothesis, ...]:
    if isinstance(source_ids, (str, bytes)) or not source_ids:
        raise ValueError("invalid source selection")
    requested = tuple(source_ids)
    if len(set(requested)) != len(requested):
        raise ValueError("invalid source selection")
    by_id = {hypothesis.source_id: hypothesis for hypothesis in hypotheses}
    if len(by_id) != len(hypotheses) or any(source_id not in by_id for source_id in requested):
        raise ValueError("invalid source selection")
    return tuple(by_id[source_id] for source_id in requested)


def exact_acoustic_vote(
    *,
    baseline: str,
    hypotheses: Sequence[AcousticHypothesis],
    source_ids: Sequence[str],
    min_source_support: int = 2,
    min_family_support: int = 2,
) -> RepairDecision:
    """Return an existing rendering only after one unique exact acoustic vote."""

    if not isinstance(baseline, str):
        return _fallback("", RepairReason.ERROR)
    try:
        selected = _selected_hypotheses(hypotheses, source_ids)
        if (
            isinstance(min_source_support, bool)
            or not isinstance(min_source_support, int)
            or min_source_support < 2
            or isinstance(min_family_support, bool)
            or not isinstance(min_family_support, int)
            or min_family_support < 1
        ):
            raise ValueError
        groups: dict[tuple[str, ...], list[AcousticHypothesis]] = defaultdict(list)
        for hypothesis in selected:
            tokens = _exact_tokens(hypothesis.text)
            if tokens:
                groups[tokens].append(hypothesis)
        qualified = [
            (tokens, members)
            for tokens, members in groups.items()
            if len(members) >= min_source_support
            and len({member.family_id for member in members}) >= min_family_support
        ]
        if not qualified:
            return _fallback(baseline, RepairReason.NO_SUPPORT)
        best_score = max(
            (len({member.family_id for member in members}), len(members))
            for _tokens, members in qualified
        )
        leaders = [
            (tokens, members)
            for tokens, members in qualified
            if (len({member.family_id for member in members}), len(members)) == best_score
        ]
        if len(leaders) != 1:
            return _fallback(baseline, RepairReason.TIE)
        _tokens, members = leaders[0]
        chosen = next(
            hypothesis.text
            for hypothesis in selected
            if hypothesis.source_id in {member.source_id for member in members}
        )
        changed = _exact_tokens(chosen) != _exact_tokens(baseline)
        if changed and exact_control_class(chosen) != exact_control_class(baseline):
            return _fallback(baseline, RepairReason.CONTROL_GUARD)
        return RepairDecision(
            chosen,
            RepairReason.EXACT_VOTE,
            True,
            changed,
            source_support=len(members),
            family_support=len({member.family_id for member in members}),
        )
    except Exception:  # noqa: BLE001 - never expose private/native detail
        return _fallback(baseline, RepairReason.ERROR)


@dataclass(frozen=True)
class _SpanProposal:
    start: int
    end: int
    replacement: tuple[str, ...] = field(repr=False)
    sources: frozenset[str]
    families: frozenset[str]


def _span_proposals(
    baseline_tokens: tuple[str, ...],
    hypotheses: Sequence[AcousticHypothesis],
) -> tuple[_SpanProposal, ...]:
    supporters: dict[
        tuple[int, int, tuple[str, ...]], set[tuple[str, str]]
    ] = defaultdict(set)
    for hypothesis in hypotheses:
        tokens = _exact_tokens(hypothesis.text)
        if not tokens:
            continue
        matcher = SequenceMatcher(None, baseline_tokens, tokens, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            supporters[(i1, i2, tokens[j1:j2])].add(
                (hypothesis.source_id, hypothesis.family_id)
            )
    return tuple(
        _SpanProposal(
            start,
            end,
            replacement,
            frozenset(source for source, _family in members),
            frozenset(family for _source, family in members),
        )
        for (start, end, replacement), members in supporters.items()
    )


def _spans_conflict(left: _SpanProposal, right: _SpanProposal) -> bool:
    if left.start == left.end and right.start == right.end:
        return left.start == right.start
    if left.start == left.end:
        return right.start <= left.start < right.end
    if right.start == right.end:
        return left.start <= right.start < left.end
    return max(left.start, right.start) < min(left.end, right.end)


def conservative_span_repair(
    *,
    baseline: str,
    hypotheses: Sequence[AcousticHypothesis],
    source_ids: Sequence[str],
    min_source_support: int = 2,
    min_family_support: int = 2,
) -> RepairDecision:
    """Copy non-conflicting edit spans supported by independent model families.

    The baseline token sequence is the scaffold. Every inserted or substituted
    token is copied from an exact source span; deletions likewise require the
    same aligned proposal from the configured support threshold. Ambiguous
    overlapping supported proposals cause a full fallback.
    """

    if not isinstance(baseline, str):
        return _fallback("", RepairReason.ERROR)
    try:
        selected = _selected_hypotheses(hypotheses, source_ids)
        if (
            isinstance(min_source_support, bool)
            or not isinstance(min_source_support, int)
            or min_source_support < 2
            or isinstance(min_family_support, bool)
            or not isinstance(min_family_support, int)
            or min_family_support < 1
        ):
            raise ValueError
        baseline_tokens = _exact_tokens(baseline)
        if not baseline_tokens:
            return _fallback(baseline, RepairReason.EMPTY)
        qualified = [
            proposal
            for proposal in _span_proposals(baseline_tokens, selected)
            if len(proposal.sources) >= min_source_support
            and len(proposal.families) >= min_family_support
        ]
        if not qualified:
            return _fallback(baseline, RepairReason.NO_SUPPORT)
        for index, left in enumerate(qualified):
            if any(_spans_conflict(left, right) for right in qualified[index + 1 :]):
                return _fallback(baseline, RepairReason.CONFLICT)
        qualified.sort(key=lambda proposal: (proposal.start, proposal.end))
        candidate_tokens: list[str] = []
        cursor = 0
        for proposal in qualified:
            candidate_tokens.extend(baseline_tokens[cursor : proposal.start])
            candidate_tokens.extend(proposal.replacement)
            cursor = proposal.end
        candidate_tokens.extend(baseline_tokens[cursor:])
        candidate_tuple = tuple(candidate_tokens)
        if not candidate_tuple or candidate_tuple == baseline_tokens:
            return _fallback(baseline, RepairReason.NO_SUPPORT)

        # Prefer an existing full rendering. Only genuinely composite repairs
        # use the normalized token renderer; all of its tokens retain exact
        # span provenance from the baseline or a supported acoustic proposal.
        renderer = next(
            (hypothesis.text for hypothesis in selected if _exact_tokens(hypothesis.text) == candidate_tuple),
            None,
        )
        constructed = renderer is None
        chosen = renderer if renderer is not None else " ".join(candidate_tuple)
        if exact_control_class(chosen) != exact_control_class(baseline):
            return _fallback(baseline, RepairReason.CONTROL_GUARD)
        return RepairDecision(
            chosen,
            RepairReason.SPAN_REPAIR,
            True,
            True,
            source_support=min(len(proposal.sources) for proposal in qualified),
            family_support=min(len(proposal.families) for proposal in qualified),
            repaired_spans=len(qualified),
            constructed=constructed,
        )
    except Exception:  # noqa: BLE001 - never expose private/native detail
        return _fallback(baseline, RepairReason.ERROR)


def _percentile_ms(values_ns: Sequence[int], percentile: float) -> float:
    if not values_ns:
        return 0.0
    ordered = sorted(values_ns)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index] / 1_000_000, 3)


def _parakeet_directory(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except Exception as exc:
        raise EvaluationPrerequisiteError() from exc
    if not resolved.is_dir() or any(not (resolved / name).is_file() for name in _PARAKEET_FILES):
        raise EvaluationPrerequisiteError()
    return resolved


def _default_parakeet_factory(model_dir: Path, *, threads: int, provider: str):
    import sherpa_onnx

    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"),
        joiner=str(model_dir / "joiner.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=threads,
        sample_rate=_PARAKEET_SAMPLE_RATE,
        feature_dim=_PARAKEET_FEATURE_DIM,
        decoding_method="greedy_search",
        max_active_paths=_PARAKEET_MAX_ACTIVE_PATHS,
        provider=provider,
        model_type="nemo_transducer",
    )


def _decode_parakeet(
    model_dir: Path,
    corpus,
    *,
    repeats: int,
    threads: int = 4,
    provider: str = "cpu",
    recognizer_factory: Callable[..., object] = _default_parakeet_factory,
) -> _ParakeetBatch:
    if (
        isinstance(repeats, bool)
        or not isinstance(repeats, int)
        or repeats < 2
        or isinstance(threads, bool)
        or not isinstance(threads, int)
        or threads < 1
        or provider not in {"cpu", "cuda"}
    ):
        raise EvaluationPrerequisiteError()
    model_dir = _parakeet_directory(model_dir)
    load_started = perf_counter_ns()
    recognizer = recognizer_factory(model_dir, threads=threads, provider=provider)
    load_ns = perf_counter_ns() - load_started
    outputs: list[str] = []
    warm_latencies: list[int] = []
    first_decode_ns = 0
    disagreements = 0
    for clip_index, item in enumerate(corpus):
        source = np.asarray(item.samples, dtype=np.float32)
        sample_rate = int(item.sample_rate)
        if sample_rate != 16_000:
            source = AudioResampler(sample_rate, 16_000, quality="VHQ").process(
                source, last=True
            )
            source = np.asarray(source, dtype=np.float32)
            sample_rate = 16_000
        normalized_outputs: set[tuple[str, ...]] = set()
        first_text = ""
        for repeat_index in range(repeats):
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, source)
            started = perf_counter_ns()
            recognizer.decode_stream(stream)
            elapsed = perf_counter_ns() - started
            text = getattr(getattr(stream, "result", None), "text", None)
            if not isinstance(text, str):
                raise EvaluationPrerequisiteError()
            if clip_index == 0 and repeat_index == 0:
                first_decode_ns = elapsed
            else:
                warm_latencies.append(elapsed)
            if repeat_index == 0:
                first_text = text
            normalized_outputs.add(_exact_tokens(text))
        disagreements += len(normalized_outputs) > 1
        outputs.append(first_text)
    return _ParakeetBatch(
        texts=tuple(outputs),
        load_ms=round(load_ns / 1_000_000, 3),
        first_decode_ms=round(first_decode_ns / 1_000_000, 3),
        warm_p50_ms=_percentile_ms(warm_latencies, 0.50),
        warm_p95_ms=_percentile_ms(warm_latencies, 0.95),
        warm_max_ms=_percentile_ms(warm_latencies, 1.0),
        repeat_disagreements=disagreements,
        empty_outputs=sum(not _exact_tokens(text) for text in outputs),
    )


def _make_cases(
    rows,
    small: Sequence[str],
    small_en: Sequence[str],
    base: Sequence[str],
    parakeet: Sequence[str],
) -> tuple[_PrivateCase, ...]:
    if not (len(rows) == len(small) == len(small_en) == len(base) == len(parakeet)):
        raise EvaluationPrerequisiteError()
    cases: list[_PrivateCase] = []
    for row, small_text, small_en_text, base_text, parakeet_text in zip(
        rows, small, small_en, base, parakeet
    ):
        current = verify_asr_consensus(
            baseline_selected=row.production,
            streaming=row.streaming,
            offline=row.offline,
            verifier=small_text,
        ).chosen
        hypotheses = (
            AcousticHypothesis("streaming", "zipformer", row.streaming),
            AcousticHypothesis("offline", "sensevoice", row.offline),
            AcousticHypothesis("small", "whisper", small_text),
            AcousticHypothesis("small_en", "whisper", small_en_text),
            AcousticHypothesis("base", "whisper", base_text),
            AcousticHypothesis("parakeet", "parakeet", parakeet_text),
        )
        cases.append(_PrivateCase(row.reference, current, hypotheses))
    return tuple(cases)


def _totals(accuracy: AccuracyTotals, clips: int) -> EvaluationTotals:
    return EvaluationTotals(accuracy, accuracy, accuracy, clips, clips, {})


def _evaluate_variants(
    cases: Iterable[_PrivateCase],
    variants: Mapping[str, Callable[[_PrivateCase], RepairDecision]],
    *,
    keywords: Sequence[str] = (),
) -> dict[str, object]:
    cases = tuple(cases)
    if not cases or not variants:
        raise EvaluationPrerequisiteError()
    baseline_accuracy = AccuracyTotals()
    baseline_errors: list[tuple[int, int]] = []
    for case in cases:
        baseline_accuracy = _sum_accuracy(
            baseline_accuracy,
            _measure(((case.reference, case.baseline),), keywords=keywords),
        )
        baseline_errors.append(_pair_errors(case.reference, case.baseline))
    baseline_totals = _totals(baseline_accuracy, len(cases))

    reports: dict[str, object] = {}
    for name, variant in variants.items():
        accuracy = AccuracyTotals()
        errors: list[tuple[int, int]] = []
        reasons: Counter[str] = Counter()
        accepted = changes = repaired_spans = constructed = 0
        unsupported = control_changes = generated_tokens = 0
        stop_attempts = stop_hits = false_stops = stop_regressions = 0
        for case in cases:
            try:
                decision = variant(case)
            except Exception:  # noqa: BLE001 - private detail cannot escape
                decision = _fallback(case.baseline, RepairReason.ERROR)
            if not isinstance(decision, RepairDecision):
                decision = _fallback(case.baseline, RepairReason.ERROR)
            candidate = decision.chosen if decision.accepted else case.baseline
            reasons[decision.reason.value] += 1
            accepted += decision.accepted
            changes += decision.changed
            repaired_spans += decision.repaired_spans
            constructed += decision.constructed
            accuracy = _sum_accuracy(
                accuracy,
                _measure(((case.reference, candidate),), keywords=keywords),
            )
            errors.append(_pair_errors(case.reference, candidate))
            sources = tuple(
                SourceHypothesis(hypothesis.source_id, hypothesis.text)
                for hypothesis in case.hypotheses
            )
            unsupported += _unsupported_token_edits(
                case.baseline, candidate, sources, 2
            )
            source_inventory = {
                token for source in case.hypotheses for token in _exact_tokens(source.text)
            }
            source_inventory.update(_exact_tokens(case.baseline))
            generated_tokens += sum(
                token not in source_inventory for token in _exact_tokens(candidate)
            )
            baseline_control = exact_control_class(case.baseline)
            candidate_control = exact_control_class(candidate)
            control_changes += candidate_control != baseline_control
            reference_stop = is_stop_command(case.reference)
            baseline_stop = is_stop_command(case.baseline)
            candidate_stop = is_stop_command(candidate)
            stop_attempts += reference_stop
            stop_hits += reference_stop and candidate_stop
            false_stops += not reference_stop and candidate_stop
            stop_regressions += baseline_stop == reference_stop and candidate_stop != reference_stop

        totals = _totals(accuracy, len(cases))
        comparison = compare_candidate(
            baseline_totals, totals, baseline_errors, errors
        )
        beats_current = accuracy.word_errors < baseline_accuracy.word_errors
        strictly_safe = (
            beats_current
            and comparison.losses == 0
            and accuracy.char_edits <= baseline_accuracy.char_edits
            and unsupported == 0
            and generated_tokens == 0
            and control_changes == 0
            and false_stops == 0
            and stop_regressions == 0
            and reasons[RepairReason.ERROR.value] == 0
        )
        reports[name] = {
            "accuracy": accuracy.as_dict(),
            "comparison": {
                **comparison.as_dict(),
                "beats_current": beats_current,
                "strictly_safe": strictly_safe,
            },
            "accepted": accepted,
            "changed": changes,
            "repaired_spans": repaired_spans,
            "constructed_repairs": constructed,
            "decision_reasons": dict(sorted(reasons.items())),
            "unsupported_token_edits": unsupported,
            "generated_tokens": generated_tokens,
            "control_class_changes": control_changes,
            "stop_commands": {
                "attempts": stop_attempts,
                "hits": stop_hits,
                "false_activations": false_stops,
                "regressions": stop_regressions,
            },
        }
    return {
        "aggregate_only": True,
        "clips": len(cases),
        "current_exact_small_consensus": {"accuracy": baseline_accuracy.as_dict()},
        "variants": reports,
    }


def _direct_accuracy(cases: Sequence[_PrivateCase], source_id: str) -> dict[str, object]:
    accuracy = AccuracyTotals()
    control_changes = stop_attempts = stop_hits = false_stops = stop_regressions = 0
    for case in cases:
        hypothesis = next(item for item in case.hypotheses if item.source_id == source_id)
        accuracy = _sum_accuracy(accuracy, _measure(((case.reference, hypothesis.text),)))
        control_changes += exact_control_class(hypothesis.text) != exact_control_class(case.baseline)
        reference_stop = is_stop_command(case.reference)
        baseline_stop = is_stop_command(case.baseline)
        candidate_stop = is_stop_command(hypothesis.text)
        stop_attempts += reference_stop
        stop_hits += reference_stop and candidate_stop
        false_stops += not reference_stop and candidate_stop
        stop_regressions += baseline_stop == reference_stop and candidate_stop != reference_stop
    return {
        "accuracy": accuracy.as_dict(),
        "control_class_changes": control_changes,
        "stop_commands": {
            "attempts": stop_attempts,
            "hits": stop_hits,
            "false_activations": false_stops,
            "regressions": stop_regressions,
        },
    }


def _variants() -> dict[str, Callable[[_PrivateCase], RepairDecision]]:
    independent_ids = ("parakeet", "small", "small_en", "base")
    return {
        "exact_parakeet_small": lambda case: exact_acoustic_vote(
            baseline=case.baseline,
            hypotheses=case.hypotheses,
            source_ids=("parakeet", "small"),
            min_source_support=2,
            min_family_support=2,
        ),
        "exact_independent_ensemble": lambda case: exact_acoustic_vote(
            baseline=case.baseline,
            hypotheses=case.hypotheses,
            source_ids=independent_ids,
            min_source_support=2,
            min_family_support=2,
        ),
        "exact_whisper_trio_correlated": lambda case: exact_acoustic_vote(
            baseline=case.baseline,
            hypotheses=case.hypotheses,
            source_ids=("small", "small_en", "base"),
            min_source_support=2,
            min_family_support=1,
        ),
        "span_independent_ensemble": lambda case: conservative_span_repair(
            baseline=case.baseline,
            hypotheses=case.hypotheses,
            source_ids=independent_ids,
            min_source_support=2,
            min_family_support=2,
        ),
    }


def _contract_digest(
    repeats: int,
    parakeet_provider: str,
    parakeet_threads: int = 4,
) -> str:
    payload = {
        "baseline": "production_exact_small_consensus",
        "whisper": {
            "device": "cuda:0",
            "compute_type": "float16",
            "language": "en",
            "task": "transcribe",
            "beam_size": 5,
            "patience": 1.0,
            "temperature": 0.0,
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "vad_filter": False,
            "vad_parameters": None,
            "condition_on_previous_text": False,
            "initial_prompt": None,
            "prefix": None,
            "hotwords": None,
            "suppress_blank": True,
            "suppress_tokens": [-1],
            "without_timestamps": True,
            "word_timestamps": False,
            "model_workers": 1,
            "local_files_only": True,
            "repeats": repeats,
        },
        "parakeet": {
            "model_type": "nemo_transducer",
            "decoding_method": "greedy_search",
            "sample_rate": _PARAKEET_SAMPLE_RATE,
            "feature_dim": _PARAKEET_FEATURE_DIM,
            "max_active_paths": _PARAKEET_MAX_ACTIVE_PATHS,
            "threads": parakeet_threads,
            "provider": parakeet_provider,
            "repeats": repeats,
        },
        "votes": "unicode_exact_existing_rendering",
        "span_repair": "sequence_matcher_exact_span_two_sources_two_families",
        "control_guard": "exact_control_class_unchanged",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate-only local acoustic consensus-v2 diagnostic."
    )
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--corpus-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--config-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--small-model", default=_DEFAULT_SMALL_MODEL)
    parser.add_argument("--small-revision", default=_DEFAULT_SMALL_REVISION)
    parser.add_argument("--small-en-model", default=_DEFAULT_SMALL_EN_MODEL)
    parser.add_argument("--small-en-revision", default=_DEFAULT_SMALL_EN_REVISION)
    parser.add_argument("--base-model", default=_DEFAULT_BASE_MODEL)
    parser.add_argument("--base-revision", default=_DEFAULT_BASE_REVISION)
    parser.add_argument("--parakeet-model", type=Path)
    parser.add_argument("--parakeet-provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--parakeet-threads", type=int, default=4)
    parser.add_argument("--decode-repeats", type=int, default=3)
    parser.add_argument("--keyword", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if not 2 <= args.decode_repeats <= 8 or not 1 <= args.parakeet_threads <= 32:
            raise EvaluationPrerequisiteError()
        corpus_root = args.corpus_root.expanduser().resolve(strict=True)
        recorded_stt_eval._REPO_ROOT = corpus_root
        corpus, corpus_digest = _load_corpus(args.manifest)
        config, effective_config, config_root = _load_machine_config(args.config_root)
        rows = _production_rows(config, config_root, corpus)

        small_path = _local_snapshot(args.small_model, revision=args.small_revision)
        small_en_path = _local_snapshot(args.small_en_model, revision=args.small_en_revision)
        base_path = _local_snapshot(args.base_model, revision=args.base_revision)
        parakeet_path = _parakeet_directory(
            args.parakeet_model
            if args.parakeet_model is not None
            else config_root / _DEFAULT_PARAKEET_DIR
        )

        small = _decode_model(small_path, corpus, repeats=args.decode_repeats)
        gc.collect()
        small_en = _decode_model(small_en_path, corpus, repeats=args.decode_repeats)
        gc.collect()
        base = _decode_model(base_path, corpus, repeats=args.decode_repeats)
        gc.collect()
        parakeet = _decode_parakeet(
            parakeet_path,
            corpus,
            repeats=args.decode_repeats,
            threads=args.parakeet_threads,
            provider=args.parakeet_provider,
        )
        cases = _make_cases(
            rows, small.texts, small_en.texts, base.texts, parakeet.texts
        )
        report = _evaluate_variants(cases, _variants(), keywords=tuple(args.keyword))
        deterministic = all(
            batch.repeat_disagreements == 0
            for batch in (small, small_en, base, parakeet)
        )
        winner_found = deterministic and any(
            item["comparison"]["strictly_safe"]
            for item in report["variants"].values()
        )
        report.update(
            ok=deterministic,
            winner_found=winner_found,
            corpus_digest=corpus_digest,
            production_config_digest=_config_digest(config),
            production_model_digest=_production_model_digest(config, config_root),
            effective_machine_config_digest=_config_digest(effective_config),
            effective_machine_model_digest=_production_model_digest(
                effective_config, config_root
            ),
            acoustic_models={
                "small": {
                    **_direct_accuracy(cases, "small"),
                    "artifact_digest": _tree_digest(small_path),
                    "revision": args.small_revision,
                    **small.as_dict(),
                },
                "small_en": {
                    **_direct_accuracy(cases, "small_en"),
                    "artifact_digest": _tree_digest(small_en_path),
                    "revision": args.small_en_revision,
                    **small_en.as_dict(),
                },
                "base": {
                    **_direct_accuracy(cases, "base"),
                    "artifact_digest": _tree_digest(base_path),
                    "revision": args.base_revision,
                    **base.as_dict(),
                },
                "parakeet_candidate": {
                    **_direct_accuracy(cases, "parakeet"),
                    "artifact_digest": _tree_digest(parakeet_path),
                    "provider": args.parakeet_provider,
                    "threads": args.parakeet_threads,
                    **parakeet.as_dict(),
                },
            },
            runtime={
                "python": platform.python_version(),
                "faster_whisper": __import__("faster_whisper").__version__,
                "ctranslate2": __import__("ctranslate2").__version__,
                "sherpa_onnx": __import__("sherpa_onnx").__version__,
                "whisper_device": "cuda:0",
                "whisper_compute_type": "float16",
                "parakeet_provider": args.parakeet_provider,
                "parakeet_threads": args.parakeet_threads,
            },
            contract_digest=_contract_digest(
                args.decode_repeats,
                args.parakeet_provider,
                args.parakeet_threads,
            ),
        )
        if args.output is not None:
            _write_report(args.output, report)
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0 if winner_found else 3
    except Exception:  # noqa: BLE001 - never expose private/native detail
        print(json.dumps(_SAFE_ERROR, sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
