"""Receipt-bound, development-only EdAcc STT baseline/candidate comparison.

This controller deliberately stays separate from the fixed four-source public
conversation qualification.  It accepts exactly the retained production EdAcc
slice, then runs the current Zipformer worker before the current
Faster-Whisper Small worker.  Successful execution is coverage evidence only;
the aggregate report has no promotion or runtime-default authority.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import sys

from tools import public_conversation_fixture as fixture
from tools import public_conversation_qualification as qualification
from tools import streaming_stt_eval, streaming_stt_suite
from tools.streaming_stt.bounded_io import (
    hash_regular_bounded,
    read_regular_bounded,
)
from tools.streaming_stt.corpus import load_corpus, verify_corpus_snapshot
from tools.streaming_stt.manifest import (
    FASTER_WHISPER_ENDPOINT_ADAPTER,
    SHERPA_ZIPFORMER_ADAPTER,
    FasterWhisperEndpointConfig,
    SherpaZipformerConfig,
    WorkerManifest,
    load_worker_manifest,
)


SCHEMA_VERSION = 1
KIND = "edacc-receipt-bound-stt-comparison-v1"
SAFE_SUITE_KIND = "edacc-safe-aggregate-stt-suite-v1"
SAFE_SUITE_SCHEMA_VERSION = 1

EXPECTED_SOURCE_ID = "edacc-test"
EXPECTED_LOCK_RECIPE_SHA256 = (
    "7a35945d20bb1e91f251edac155d126ca6cb8abe44880e359be5725f48ca212e"
)
EXPECTED_LOCK_RAW_SHA256 = (
    "68ac2c85b8825fabda10b578646780fc16a37a1b2ab0ca3285fa9f8c93a3168b"
)
EXPECTED_CORPUS_SHA256 = (
    "4da392c39a0b6bd18057f63c96b4f67c0dfdaf4c14e1d2d76176cdbee0920772"
)
EXPECTED_RECEIPT_SHA256 = (
    "3c516b6fbb8ce139498cd2e01d83a4faf825e58bf2b9abcaa52896bdab4c853f"
)
EXPECTED_PCM_BYTES = 4_790_400
STRATUM_TAGS = ("clean", "disfluent", "overlap_adjacent")
STRATUM_COUNTS = (8, 8, 8)

FIXED_REPEATS = 3
FIXED_CHUNK_SAMPLES = 1_600
FIXED_PACE = "burst"
FIXED_PARTIAL_INTERVAL_MS = 200
FIXED_TAIL_PADDING_SAMPLES = 0

WORKER_ROLES = ("baseline", "candidate")
BASELINE_SCHEMA_VERSION = 4
BASELINE_MODEL_ID = "production-gigaspeech-zipformer-fp32-benchmark-1t"
CANDIDATE_SCHEMA_VERSION = 6
CANDIDATE_MODEL_ID = "faster-whisper-small-local"

_MAX_IMPLEMENTATION_BYTES = 512 * 1024
_MAX_RECEIPT_BYTES = 256 * 1024
_MAX_REPORT_BYTES = 64 * 1024 * 1024
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXED_WORKER = _REPO_ROOT / "tools" / "streaming_stt" / "worker.py"
_PRODUCTION_RUNNER = streaming_stt_suite.run_suite
_SAFE_ERROR = {"error": "edacc STT comparison failed", "ok": False}


class ComparisonError(RuntimeError):
    """A detail-free source, model, execution, or report contract failure."""


@dataclass(frozen=True)
class _SourceState:
    lock_raw_sha256: str
    lock_recipe_sha256: str
    corpus_path: Path = field(repr=False)
    receipt_path: Path = field(repr=False)
    corpus_manifest_sha256: str
    receipt_sha256: str
    cases: int
    pcm_bytes: int
    stratum_counts: tuple[int, ...]


@dataclass(frozen=True)
class _WorkerState:
    paths: tuple[Path, Path] = field(repr=False)
    manifests: tuple[WorkerManifest, WorkerManifest] = field(repr=False)
    manifest_digests: tuple[str, str]
    adapters: tuple[str, str]
    stream_configs: tuple[dict[str, object], dict[str, object]]
    evidence_roots: tuple[Path, ...] = field(repr=False)
    model_bindings: tuple[dict[str, object], dict[str, object]]
    worker_source_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ComparisonError() from None


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_field(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ComparisonError()
    return value


def _exact_int(value: object, expected: int) -> bool:
    return type(value) is int and value == expected


def _implementation_sha256() -> str:
    try:
        path = Path(__file__).resolve(strict=True)
        size = path.lstat().st_size
        return hash_regular_bounded(
            path,
            maximum_bytes=_MAX_IMPLEMENTATION_BYTES,
            expected_bytes=size,
        ).sha256
    except Exception:
        raise ComparisonError() from None


def _worker_source_sha256() -> str:
    try:
        path = _FIXED_WORKER.resolve(strict=True)
        size = path.lstat().st_size
        return hash_regular_bounded(
            path,
            maximum_bytes=4 * 1024 * 1024,
            expected_bytes=size,
        ).sha256
    except Exception:
        raise ComparisonError() from None


def _assert_path_free(value: object) -> None:
    try:
        qualification._assert_path_free(value)
    except Exception:
        raise ComparisonError() from None


def _source_state(
    corpus_path: Path | str,
    *,
    lock_path: Path | str,
) -> _SourceState:
    """Revalidate the exact production source and its lock-derived strata."""

    try:
        lock = fixture.load_fixture_lock(lock_path)
        binding = fixture.validate_private_source(
            corpus_path,
            lock=lock,
            expected_source_id=EXPECTED_SOURCE_ID,
        )
        if (
            lock.recipe_sha256 != EXPECTED_LOCK_RECIPE_SHA256
            or lock.raw_sha256 != EXPECTED_LOCK_RAW_SHA256
            or binding.source_id != EXPECTED_SOURCE_ID
            or binding.corpus_manifest_sha256 != EXPECTED_CORPUS_SHA256
            or binding.receipt_sha256 != EXPECTED_RECEIPT_SHA256
            or binding.cases != fixture.CASES_PER_SOURCE
            or binding.pcm_bytes != EXPECTED_PCM_BYTES
        ):
            raise ComparisonError()

        corpus = load_corpus(binding.corpus_path)
        slots = lock.slots_for(EXPECTED_SOURCE_ID)
        canonical_tags = frozenset(STRATUM_TAGS)
        counts: Counter[str] = Counter()
        if len(corpus.cases) != fixture.CASES_PER_SOURCE or len(slots) != len(
            corpus.cases
        ):
            raise ComparisonError()
        for ordinal, (case, slot) in enumerate(
            zip(corpus.cases, slots, strict=True)
        ):
            membership = canonical_tags.intersection(case.tags)
            if (
                slot.source_id != EXPECTED_SOURCE_ID
                or slot.ordinal != ordinal
                or slot.channel != "single"
                or slot.pair_id is not None
                or slot.stratum not in canonical_tags
                or membership != {slot.stratum}
            ):
                raise ComparisonError()
            counts[slot.stratum] += 1
        expected_counts = tuple(
            counts[tag] for tag in STRATUM_TAGS
        )
        if expected_counts != STRATUM_COUNTS:
            raise ComparisonError()

        receipt_snapshot = read_regular_bounded(
            binding.receipt_path,
            maximum_bytes=_MAX_RECEIPT_BYTES,
        )
        if hashlib.sha256(receipt_snapshot.data).hexdigest() != binding.receipt_sha256:
            raise ComparisonError()
        receipt = fixture._strict_json(receipt_snapshot.data)
        if (
            not isinstance(receipt, Mapping)
            or not isinstance(receipt.get("decoder"), Mapping)
            or receipt["decoder"].get("production_evidence") is not True
        ):
            raise ComparisonError()
        verify_corpus_snapshot(corpus)
        return _SourceState(
            lock_raw_sha256=lock.raw_sha256,
            lock_recipe_sha256=lock.recipe_sha256,
            corpus_path=binding.corpus_path,
            receipt_path=binding.receipt_path,
            corpus_manifest_sha256=binding.corpus_manifest_sha256,
            receipt_sha256=binding.receipt_sha256,
            cases=binding.cases,
            pcm_bytes=binding.pcm_bytes,
            stratum_counts=expected_counts,
        )
    except ComparisonError:
        raise
    except Exception:
        raise ComparisonError() from None


def _absolute_distinct_worker_paths(
    baseline: Path | str,
    candidate: Path | str,
) -> tuple[Path, Path]:
    try:
        paths = tuple(
            Path(os.path.abspath(Path(value).expanduser())).resolve(
                strict=False
            )
            for value in (baseline, candidate)
        )
    except Exception:
        raise ComparisonError() from None
    if os.path.normcase(str(paths[0])) == os.path.normcase(str(paths[1])):
        raise ComparisonError()
    return paths  # type: ignore[return-value]


def _artifact_set_sha256(manifest: WorkerManifest) -> str:
    return _canonical_sha256(
        [
            {
                "name": artifact.name,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
            }
            for artifact in manifest.artifacts
        ]
    )


def _model_binding(
    manifest: WorkerManifest,
    *,
    role: str,
    worker_index: int,
    worker_sha256: str,
) -> dict[str, object]:
    config = manifest.adapter_config
    if config is None:
        raise ComparisonError()
    by_name = manifest.artifact_by_name
    return {
        "role": role,
        "worker_index": worker_index,
        "schema_version": manifest.schema_version,
        "model_id": manifest.model_id,
        "adapter": manifest.adapter,
        "language": getattr(config, "language", None),
        "manifest_sha256": manifest.digest,
        "python_sha256": manifest.python.sha256,
        "worker_sha256": worker_sha256,
        "artifact_set_sha256": _artifact_set_sha256(manifest),
        "adapter_config_sha256": _canonical_sha256(config.as_dict()),
        "runtime_receipt_sha256": (
            by_name["runtime-receipt"].sha256
            if "runtime-receipt" in by_name
            else None
        ),
        "model_receipt_sha256": (
            by_name["model-receipt"].sha256
            if "model-receipt" in by_name
            else None
        ),
    }


def _worker_state(
    baseline_worker_manifest: Path | str,
    candidate_worker_manifest: Path | str,
) -> _WorkerState:
    try:
        paths = _absolute_distinct_worker_paths(
            baseline_worker_manifest,
            candidate_worker_manifest,
        )
        manifests = tuple(load_worker_manifest(path) for path in paths)
        baseline, candidate = manifests
        fixed_worker = _FIXED_WORKER.resolve(strict=True)
        worker_sha256 = _worker_source_sha256()
        if (
            baseline.schema_version != BASELINE_SCHEMA_VERSION
            or baseline.model_id != BASELINE_MODEL_ID
            or baseline.adapter != SHERPA_ZIPFORMER_ADAPTER
            or not isinstance(baseline.adapter_config, SherpaZipformerConfig)
            or baseline.adapter_config != SherpaZipformerConfig()
            or candidate.schema_version != CANDIDATE_SCHEMA_VERSION
            or candidate.model_id != CANDIDATE_MODEL_ID
            or candidate.adapter != FASTER_WHISPER_ENDPOINT_ADAPTER
            or not isinstance(
                candidate.adapter_config,
                FasterWhisperEndpointConfig,
            )
            or candidate.adapter_config.language != "en"
            or baseline.worker.path != fixed_worker
            or candidate.worker.path != fixed_worker
            or baseline.worker.sha256 != worker_sha256
            or candidate.worker.sha256 != worker_sha256
            or baseline.digest == candidate.digest
        ):
            raise ComparisonError()

        streams = qualification._resolved_stream_configs(
            paths,
            chunk_samples=FIXED_CHUNK_SAMPLES,
            pace=FIXED_PACE,
            partial_interval_ms=FIXED_PARTIAL_INTERVAL_MS,
            tail_padding_samples=FIXED_TAIL_PADDING_SAMPLES,
        )
        expected_stream = {
            "chunk_samples": FIXED_CHUNK_SAMPLES,
            "pace": FIXED_PACE,
            "partial_interval_ms": FIXED_PARTIAL_INTERVAL_MS,
            "tail_padding_samples": FIXED_TAIL_PADDING_SAMPLES,
        }
        if tuple(streams) != (expected_stream, expected_stream):
            raise ComparisonError()
        evidence_roots = tuple(
            sorted(
                {
                    root
                    for manifest in manifests
                    for root in qualification._worker_evidence_roots(
                        manifest
                    )
                },
                key=str,
            )
        )
        bindings = tuple(
            _model_binding(
                manifest,
                role=WORKER_ROLES[index],
                worker_index=index,
                worker_sha256=worker_sha256,
            )
            for index, manifest in enumerate(manifests)
        )
        return _WorkerState(
            paths=paths,
            manifests=manifests,
            manifest_digests=(baseline.digest, candidate.digest),
            adapters=(baseline.adapter, candidate.adapter),
            stream_configs=streams,
            evidence_roots=evidence_roots,
            model_bindings=bindings,
            worker_source_sha256=worker_sha256,
        )
    except ComparisonError:
        raise
    except Exception:
        raise ComparisonError() from None


def _overlaps(left: Path, right: Path) -> bool:
    return qualification._is_within(left, right) or qualification._is_within(
        right,
        left,
    )


def _validate_private_paths(
    *,
    final_output: Path,
    scratch: Path,
    checkpoint: Path,
    source: _SourceState,
    worker_evidence_roots: Sequence[Path],
) -> None:
    try:
        protected = (source.corpus_path.parent, *worker_evidence_roots)
        selected = (final_output, scratch, checkpoint)
        if (
            final_output == checkpoint
            or qualification._is_within(final_output, scratch)
            or qualification._is_within(scratch, final_output)
            or qualification._is_within(final_output, _REPO_ROOT)
            or qualification._is_within(scratch, _REPO_ROOT)
            or any(qualification._has_git_ancestor(path) for path in selected)
            or any(
                _overlaps(path, root)
                for path in selected
                for root in protected
            )
        ):
            raise ComparisonError()
    except ComparisonError:
        raise
    except Exception:
        raise ComparisonError() from None


def _resolved_worker_rows(workers: _WorkerState) -> list[dict[str, object]]:
    try:
        return qualification._worker_stream_rows(
            workers.manifest_digests,
            workers.adapters,
            workers.stream_configs,
        )
    except Exception:
        raise ComparisonError() from None


def _expected_suite_config() -> dict[str, object]:
    row = {
        "corpus_index": 0,
        "tags": list(STRATUM_TAGS),
        "selection_sha256": _canonical_sha256(list(STRATUM_TAGS)),
    }
    strata = {
        "mode": "per_corpus",
        "corpora": [row],
        "selection_sha256": _canonical_sha256([row]),
    }
    value: dict[str, object] = {
        "repeats": FIXED_REPEATS,
        "strata": strata,
        "overrides": {
            "chunk_samples": FIXED_CHUNK_SAMPLES,
            "pace": FIXED_PACE,
            "partial_interval_ms": FIXED_PARTIAL_INTERVAL_MS,
            "tail_padding_samples": FIXED_TAIL_PADDING_SAMPLES,
        },
    }
    value["contract_sha256"] = _canonical_sha256(value)
    return value


def _evaluator_state(
    workers: _WorkerState,
) -> tuple[tuple[Mapping[str, object], Mapping[str, object]], str, str]:
    try:
        bindings = tuple(
            streaming_stt_eval._evaluator_binding(adapter)
            for adapter in workers.adapters
        )
        digests = tuple(
            streaming_stt_suite._evaluator_implementation_sha256(binding)
            for binding in bindings
        )
        controller_sha256 = streaming_stt_suite._controller_source_sha256()
        if len(set(digests)) != 1:
            raise ComparisonError()
        return bindings, digests[0], controller_sha256  # type: ignore[return-value]
    except ComparisonError:
        raise
    except Exception:
        raise ComparisonError() from None


def _reduce_source_suite(
    value: object,
    *,
    source: _SourceState,
    workers: _WorkerState,
    implementation_sha256: str,
) -> dict[str, object]:
    """Validate the complete generic suite and retain only allowlisted aggregates."""

    try:
        if not isinstance(value, Mapping) or set(value) != {
            "ok",
            "complete",
            "controller",
            "config",
            "runs",
        }:
            raise ComparisonError()
        controller = value.get("controller")
        config = value.get("config")
        runs = value.get("runs")
        controller_fields = {
            "kind",
            "schema_version",
            "execution",
            "state",
            "planned_worker_manifests",
            "planned_corpora",
            "planned_runs",
            "completed_runs",
            "controller_sha256",
            "completed_worker_set_sha256",
            "completed_corpus_set_sha256",
            "completed_matrix_sha256",
            "stratum_selection_sha256",
            "evaluator_implementation_sha256",
            "worker_manifests",
            "corpora",
            "runs",
            "worker_set_sha256",
            "corpus_set_sha256",
            "matrix_sha256",
        }
        expected_config = _expected_suite_config()
        worker_rows = _resolved_worker_rows(workers)
        worker_set_sha256 = _canonical_sha256(list(workers.manifest_digests))
        corpus_set_sha256 = _canonical_sha256(
            [source.corpus_manifest_sha256]
        )
        evaluator_bindings, evaluator_sha256, source_controller_sha256 = (
            _evaluator_state(workers)
        )
        if (
            value.get("complete") is not True
            or type(value.get("ok")) is not bool
            or not isinstance(controller, Mapping)
            or set(controller) != controller_fields
            or not isinstance(runs, list)
            or len(runs) != len(WORKER_ROLES)
            or _canonical_json_bytes(config)
            != _canonical_json_bytes(expected_config)
            or controller.get("kind")
            != "bounded_sequential_streaming_stt_suite"
            or not _exact_int(controller.get("schema_version"), 1)
            or controller.get("execution")
            != "strictly_sequential_cross_product"
            or controller.get("state") != "complete"
            or not _exact_int(
                controller.get("planned_worker_manifests"), 2
            )
            or not _exact_int(controller.get("worker_manifests"), 2)
            or not _exact_int(controller.get("planned_corpora"), 1)
            or not _exact_int(controller.get("corpora"), 1)
            or not _exact_int(controller.get("planned_runs"), 2)
            or not _exact_int(controller.get("completed_runs"), 2)
            or not _exact_int(controller.get("runs"), 2)
            or controller.get("controller_sha256")
            != source_controller_sha256
            or controller.get("stratum_selection_sha256")
            != expected_config["strata"]["selection_sha256"]
            or controller.get("evaluator_implementation_sha256")
            != evaluator_sha256
            or controller.get("worker_set_sha256") != worker_set_sha256
            or controller.get("completed_worker_set_sha256")
            != worker_set_sha256
            or controller.get("corpus_set_sha256") != corpus_set_sha256
            or controller.get("completed_corpus_set_sha256")
            != corpus_set_sha256
        ):
            raise ComparisonError()

        raw_bindings: list[Mapping[str, object]] = []
        safe_bindings: list[dict[str, object]] = []
        safe_runs: list[dict[str, object]] = []
        run_success: list[bool] = []
        for worker_index, run in enumerate(runs):
            if not isinstance(run, Mapping) or set(run) != {
                "binding",
                "report",
            }:
                raise ComparisonError()
            binding = run.get("binding")
            report = run.get("report")
            binding_fields = {
                "worker_index",
                "corpus_index",
                "worker_manifest_sha256",
                "corpus_manifest_sha256",
                "stratum_selection_sha256",
                "evaluator_implementation_sha256",
                "report_sha256",
                "binding_sha256",
            }
            report_fields = {
                "ok",
                "evidence",
                "worker",
                "corpus",
                "config",
                "metrics",
                "worker_stderr",
                "evaluator",
            }
            if (
                not isinstance(binding, Mapping)
                or set(binding) != binding_fields
                or not isinstance(report, Mapping)
                or set(report) != report_fields
                or not _exact_int(
                    binding.get("worker_index"), worker_index
                )
                or not _exact_int(binding.get("corpus_index"), 0)
                or binding.get("worker_manifest_sha256")
                != workers.manifest_digests[worker_index]
                or binding.get("corpus_manifest_sha256")
                != source.corpus_manifest_sha256
                or binding.get("stratum_selection_sha256")
                != _canonical_sha256(list(STRATUM_TAGS))
                or binding.get("evaluator_implementation_sha256")
                != evaluator_sha256
            ):
                raise ComparisonError()
            source_report_sha256 = _sha256_field(
                binding.get("report_sha256")
            )
            raw_contract = dict(binding)
            source_binding_sha256 = _sha256_field(
                raw_contract.pop("binding_sha256")
            )
            if (
                source_report_sha256 != _canonical_sha256(report)
                or source_binding_sha256 != _canonical_sha256(raw_contract)
            ):
                raise ComparisonError()
            try:
                safe_report = qualification._safe_report_snapshot(
                    report,
                    adapter=workers.adapters[worker_index],
                    expected_stream=workers.stream_configs[worker_index],
                    source_report_sha256=source_report_sha256,
                    worker_manifest_sha256=workers.manifest_digests[
                        worker_index
                    ],
                    corpus_manifest_sha256=source.corpus_manifest_sha256,
                    evaluator_binding=evaluator_bindings[worker_index],
                    evaluator_implementation_sha256=evaluator_sha256,
                    tags=STRATUM_TAGS,
                    stratum_counts=tuple(
                        zip(STRATUM_TAGS, STRATUM_COUNTS, strict=True)
                    ),
                    repeats=FIXED_REPEATS,
                )
            except Exception:
                raise ComparisonError() from None
            safe_binding: dict[str, object] = {
                "worker_index": worker_index,
                "corpus_index": 0,
                "worker_manifest_sha256": workers.manifest_digests[
                    worker_index
                ],
                "corpus_manifest_sha256": source.corpus_manifest_sha256,
                "stratum_selection_sha256": _canonical_sha256(
                    list(STRATUM_TAGS)
                ),
                "evaluator_implementation_sha256": evaluator_sha256,
                "worker_adapter": workers.adapters[worker_index],
                "stream_config_sha256": worker_rows[worker_index][
                    "stream_sha256"
                ],
                "source_report_sha256": source_report_sha256,
                "source_binding_sha256": source_binding_sha256,
                "report_sha256": _canonical_sha256(safe_report),
            }
            safe_binding["binding_sha256"] = _canonical_sha256(safe_binding)
            raw_bindings.append(binding)
            safe_bindings.append(safe_binding)
            safe_runs.append({"binding": safe_binding, "report": safe_report})
            run_success.append(safe_report["ok"] is True)

        source_matrix_sha256 = _canonical_sha256(raw_bindings)
        safe_matrix_sha256 = _canonical_sha256(safe_bindings)
        expected_ok = all(run_success)
        if (
            value.get("ok") is not expected_ok
            or controller.get("matrix_sha256") != source_matrix_sha256
            or controller.get("completed_matrix_sha256")
            != source_matrix_sha256
        ):
            raise ComparisonError()
        _assert_path_free(value)
        safe_controller = {
            "kind": SAFE_SUITE_KIND,
            "schema_version": SAFE_SUITE_SCHEMA_VERSION,
            "execution": "strictly_sequential_cross_product",
            "state": "complete",
            "planned_worker_manifests": 2,
            "planned_corpora": 1,
            "planned_runs": 2,
            "completed_runs": 2,
            "comparison_implementation_sha256": implementation_sha256,
            "source_controller_sha256": source_controller_sha256,
            "completed_worker_set_sha256": worker_set_sha256,
            "completed_corpus_set_sha256": corpus_set_sha256,
            "completed_matrix_sha256": safe_matrix_sha256,
            "stratum_selection_sha256": expected_config["strata"][
                "selection_sha256"
            ],
            "evaluator_implementation_sha256": evaluator_sha256,
            "worker_stream_set_sha256": _canonical_sha256(worker_rows),
            "worker_manifests": 2,
            "corpora": 1,
            "runs": 2,
            "worker_set_sha256": worker_set_sha256,
            "corpus_set_sha256": corpus_set_sha256,
            "matrix_sha256": safe_matrix_sha256,
            "source_matrix_sha256": source_matrix_sha256,
        }
        result = {
            "ok": expected_ok,
            "complete": True,
            "controller": safe_controller,
            "config": expected_config,
            "runs": safe_runs,
        }
        _assert_path_free(result)
        return result
    except ComparisonError:
        raise
    except Exception:
        raise ComparisonError() from None


def _source_report(source: _SourceState) -> dict[str, object]:
    return {
        "source_id": EXPECTED_SOURCE_ID,
        "fixture_id": fixture.FIXTURE_ID,
        "lock_raw_sha256": source.lock_raw_sha256,
        "lock_recipe_sha256": source.lock_recipe_sha256,
        "corpus_manifest_sha256": source.corpus_manifest_sha256,
        "receipt_sha256": source.receipt_sha256,
        "cases": source.cases,
        "pcm_bytes": source.pcm_bytes,
        "strata": [
            {"tag": tag, "cases": count}
            for tag, count in zip(
                STRATUM_TAGS,
                STRATUM_COUNTS,
                strict=True,
            )
        ],
        "production_evidence": True,
        "validation": "before_and_after_suite",
    }


def _contract(
    source: _SourceState,
    workers: _WorkerState,
    *,
    implementation_sha256: str,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "implementation_sha256": implementation_sha256,
        "worker_source_sha256": workers.worker_source_sha256,
        "fixture_id": fixture.FIXTURE_ID,
        "lock_recipe_sha256": source.lock_recipe_sha256,
        "source_id": EXPECTED_SOURCE_ID,
        "expected_lock_raw_sha256": EXPECTED_LOCK_RAW_SHA256,
        "expected_corpus_sha256": EXPECTED_CORPUS_SHA256,
        "expected_receipt_sha256": EXPECTED_RECEIPT_SHA256,
        "expected_pcm_bytes": EXPECTED_PCM_BYTES,
        "worker_roles": list(WORKER_ROLES),
        "resolved_worker_streams": _resolved_worker_rows(workers),
        "model_binding_set_sha256": _canonical_sha256(
            list(workers.model_bindings)
        ),
        "source_strata": [
            {"tag": tag, "cases": count}
            for tag, count in zip(
                STRATUM_TAGS,
                STRATUM_COUNTS,
                strict=True,
            )
        ],
        "repeats": FIXED_REPEATS,
        "execution": "strictly_sequential_two_by_one",
        "quality_semantics": "coverage_only_no_thresholds",
        "development_only": True,
        "promotional": False,
    }
    value["contract_sha256"] = _canonical_sha256(value)
    return value


def _validated_model_bindings(
    value: object,
    *,
    worker_source_sha256: str,
) -> tuple[list[dict[str, object]], tuple[str, str], tuple[str, str]]:
    fields = {
        "role",
        "worker_index",
        "schema_version",
        "model_id",
        "adapter",
        "language",
        "manifest_sha256",
        "python_sha256",
        "worker_sha256",
        "artifact_set_sha256",
        "adapter_config_sha256",
        "runtime_receipt_sha256",
        "model_receipt_sha256",
    }
    expected = (
        (
            WORKER_ROLES[0],
            BASELINE_SCHEMA_VERSION,
            BASELINE_MODEL_ID,
            SHERPA_ZIPFORMER_ADAPTER,
        ),
        (
            WORKER_ROLES[1],
            CANDIDATE_SCHEMA_VERSION,
            CANDIDATE_MODEL_ID,
            FASTER_WHISPER_ENDPOINT_ADAPTER,
        ),
    )
    if not isinstance(value, list) or len(value) != 2:
        raise ComparisonError()
    rows: list[dict[str, object]] = []
    digests: list[str] = []
    adapters: list[str] = []
    for index, (row, contract) in enumerate(zip(value, expected, strict=True)):
        role, schema_version, model_id, adapter = contract
        if (
            not isinstance(row, Mapping)
            or set(row) != fields
            or row.get("role") != role
            or not _exact_int(row.get("worker_index"), index)
            or not _exact_int(row.get("schema_version"), schema_version)
            or row.get("model_id") != model_id
            or row.get("adapter") != adapter
            or row.get("language") != "en"
            or row.get("worker_sha256") != worker_source_sha256
        ):
            raise ComparisonError()
        for key in (
            "manifest_sha256",
            "python_sha256",
            "worker_sha256",
            "artifact_set_sha256",
            "adapter_config_sha256",
        ):
            _sha256_field(row.get(key))
        if index == 0:
            if (
                row.get("runtime_receipt_sha256") is not None
                or row.get("model_receipt_sha256") is not None
            ):
                raise ComparisonError()
        else:
            _sha256_field(row.get("runtime_receipt_sha256"))
            _sha256_field(row.get("model_receipt_sha256"))
        rows.append(dict(row))
        digests.append(str(row["manifest_sha256"]))
        adapters.append(adapter)
    if len(set(digests)) != 2:
        raise ComparisonError()
    return rows, tuple(digests), tuple(adapters)  # type: ignore[return-value]


def _validated_contract(
    value: object,
    *,
    model_bindings: Sequence[Mapping[str, object]],
    worker_digests: Sequence[str],
    worker_adapters: Sequence[str],
) -> tuple[str, list[dict[str, object]]]:
    fields = {
        "schema_version",
        "kind",
        "implementation_sha256",
        "worker_source_sha256",
        "fixture_id",
        "lock_recipe_sha256",
        "source_id",
        "expected_lock_raw_sha256",
        "expected_corpus_sha256",
        "expected_receipt_sha256",
        "expected_pcm_bytes",
        "worker_roles",
        "resolved_worker_streams",
        "model_binding_set_sha256",
        "source_strata",
        "repeats",
        "execution",
        "quality_semantics",
        "development_only",
        "promotional",
        "contract_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ComparisonError()
    body = dict(value)
    contract_sha256 = _sha256_field(body.pop("contract_sha256"))
    implementation_sha256 = _sha256_field(
        value.get("implementation_sha256")
    )
    _sha256_field(value.get("worker_source_sha256"))
    expected_strata = [
        {"tag": tag, "cases": count}
        for tag, count in zip(STRATUM_TAGS, STRATUM_COUNTS, strict=True)
    ]
    if (
        not _exact_int(value.get("schema_version"), SCHEMA_VERSION)
        or value.get("kind") != KIND
        or value.get("fixture_id") != fixture.FIXTURE_ID
        or value.get("lock_recipe_sha256") != EXPECTED_LOCK_RECIPE_SHA256
        or value.get("source_id") != EXPECTED_SOURCE_ID
        or value.get("expected_lock_raw_sha256")
        != EXPECTED_LOCK_RAW_SHA256
        or value.get("expected_corpus_sha256") != EXPECTED_CORPUS_SHA256
        or value.get("expected_receipt_sha256") != EXPECTED_RECEIPT_SHA256
        or not _exact_int(
            value.get("expected_pcm_bytes"), EXPECTED_PCM_BYTES
        )
        or value.get("worker_roles") != list(WORKER_ROLES)
        or value.get("model_binding_set_sha256")
        != _canonical_sha256(list(model_bindings))
        or value.get("source_strata") != expected_strata
        or not _exact_int(value.get("repeats"), FIXED_REPEATS)
        or value.get("execution") != "strictly_sequential_two_by_one"
        or value.get("quality_semantics")
        != "coverage_only_no_thresholds"
        or value.get("development_only") is not True
        or value.get("promotional") is not False
        or contract_sha256 != _canonical_sha256(body)
    ):
        raise ComparisonError()
    stream = {
        "chunk_samples": FIXED_CHUNK_SAMPLES,
        "pace": FIXED_PACE,
        "partial_interval_ms": FIXED_PARTIAL_INTERVAL_MS,
        "tail_padding_samples": FIXED_TAIL_PADDING_SAMPLES,
    }
    expected_rows = qualification._worker_stream_rows(
        worker_digests,
        worker_adapters,
        (stream, stream),
    )
    if _canonical_json_bytes(
        value.get("resolved_worker_streams")
    ) != _canonical_json_bytes(expected_rows):
        raise ComparisonError()
    return implementation_sha256, expected_rows


def _validated_source_report(value: object) -> dict[str, object]:
    fields = {
        "source_id",
        "fixture_id",
        "lock_raw_sha256",
        "lock_recipe_sha256",
        "corpus_manifest_sha256",
        "receipt_sha256",
        "cases",
        "pcm_bytes",
        "strata",
        "production_evidence",
        "validation",
    }
    expected_strata = [
        {"tag": tag, "cases": count}
        for tag, count in zip(STRATUM_TAGS, STRATUM_COUNTS, strict=True)
    ]
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("source_id") != EXPECTED_SOURCE_ID
        or value.get("fixture_id") != fixture.FIXTURE_ID
        or value.get("lock_recipe_sha256") != EXPECTED_LOCK_RECIPE_SHA256
        or value.get("corpus_manifest_sha256") != EXPECTED_CORPUS_SHA256
        or value.get("receipt_sha256") != EXPECTED_RECEIPT_SHA256
        or not _exact_int(value.get("cases"), fixture.CASES_PER_SOURCE)
        or not _exact_int(value.get("pcm_bytes"), EXPECTED_PCM_BYTES)
        or value.get("strata") != expected_strata
        or value.get("production_evidence") is not True
        or value.get("validation") != "before_and_after_suite"
    ):
        raise ComparisonError()
    if value.get("lock_raw_sha256") != EXPECTED_LOCK_RAW_SHA256:
        raise ComparisonError()
    return dict(value)


def _validate_retained_suite(
    value: object,
    *,
    implementation_sha256: str,
    worker_digests: Sequence[str],
    worker_adapters: Sequence[str],
    worker_rows: Sequence[Mapping[str, object]],
    corpus_sha256: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "ok",
        "complete",
        "controller",
        "config",
        "runs",
    }:
        raise ComparisonError()
    expected_config = _expected_suite_config()
    controller = value.get("controller")
    runs = value.get("runs")
    controller_fields = {
        "kind",
        "schema_version",
        "execution",
        "state",
        "planned_worker_manifests",
        "planned_corpora",
        "planned_runs",
        "completed_runs",
        "comparison_implementation_sha256",
        "source_controller_sha256",
        "completed_worker_set_sha256",
        "completed_corpus_set_sha256",
        "completed_matrix_sha256",
        "stratum_selection_sha256",
        "evaluator_implementation_sha256",
        "worker_stream_set_sha256",
        "worker_manifests",
        "corpora",
        "runs",
        "worker_set_sha256",
        "corpus_set_sha256",
        "matrix_sha256",
        "source_matrix_sha256",
    }
    worker_set_sha256 = _canonical_sha256(list(worker_digests))
    corpus_set_sha256 = _canonical_sha256([corpus_sha256])
    if (
        value.get("complete") is not True
        or type(value.get("ok")) is not bool
        or _canonical_json_bytes(value.get("config"))
        != _canonical_json_bytes(expected_config)
        or not isinstance(controller, Mapping)
        or set(controller) != controller_fields
        or controller.get("kind") != SAFE_SUITE_KIND
        or not _exact_int(
            controller.get("schema_version"), SAFE_SUITE_SCHEMA_VERSION
        )
        or controller.get("execution") != "strictly_sequential_cross_product"
        or controller.get("state") != "complete"
        or not _exact_int(
            controller.get("planned_worker_manifests"), 2
        )
        or not _exact_int(controller.get("worker_manifests"), 2)
        or not _exact_int(controller.get("planned_corpora"), 1)
        or not _exact_int(controller.get("corpora"), 1)
        or not _exact_int(controller.get("planned_runs"), 2)
        or not _exact_int(controller.get("completed_runs"), 2)
        or not _exact_int(controller.get("runs"), 2)
        or controller.get("comparison_implementation_sha256")
        != implementation_sha256
        or controller.get("worker_set_sha256") != worker_set_sha256
        or controller.get("completed_worker_set_sha256")
        != worker_set_sha256
        or controller.get("corpus_set_sha256") != corpus_set_sha256
        or controller.get("completed_corpus_set_sha256")
        != corpus_set_sha256
        or controller.get("stratum_selection_sha256")
        != expected_config["strata"]["selection_sha256"]
        or controller.get("worker_stream_set_sha256")
        != _canonical_sha256(list(worker_rows))
        or not isinstance(runs, list)
        or len(runs) != 2
    ):
        raise ComparisonError()
    source_controller_sha256 = _sha256_field(
        controller.get("source_controller_sha256")
    )
    evaluator_sha256 = _sha256_field(
        controller.get("evaluator_implementation_sha256")
    )
    if not source_controller_sha256:
        raise ComparisonError()

    safe_bindings: list[Mapping[str, object]] = []
    source_bindings: list[dict[str, object]] = []
    run_success: list[bool] = []
    stream = {
        "chunk_samples": FIXED_CHUNK_SAMPLES,
        "pace": FIXED_PACE,
        "partial_interval_ms": FIXED_PARTIAL_INTERVAL_MS,
        "tail_padding_samples": FIXED_TAIL_PADDING_SAMPLES,
    }
    overrides = expected_config["overrides"]
    binding_fields = {
        "worker_index",
        "corpus_index",
        "worker_manifest_sha256",
        "corpus_manifest_sha256",
        "stratum_selection_sha256",
        "evaluator_implementation_sha256",
        "worker_adapter",
        "stream_config_sha256",
        "source_report_sha256",
        "source_binding_sha256",
        "report_sha256",
        "binding_sha256",
    }
    for worker_index, run in enumerate(runs):
        if not isinstance(run, Mapping) or set(run) != {
            "binding",
            "report",
        }:
            raise ComparisonError()
        binding = run.get("binding")
        report = run.get("report")
        if (
            not isinstance(binding, Mapping)
            or set(binding) != binding_fields
            or not _exact_int(
                binding.get("worker_index"), worker_index
            )
            or not _exact_int(binding.get("corpus_index"), 0)
            or binding.get("worker_manifest_sha256")
            != worker_digests[worker_index]
            or binding.get("corpus_manifest_sha256") != corpus_sha256
            or binding.get("stratum_selection_sha256")
            != _canonical_sha256(list(STRATUM_TAGS))
            or binding.get("evaluator_implementation_sha256")
            != evaluator_sha256
            or binding.get("worker_adapter")
            != worker_adapters[worker_index]
            or binding.get("stream_config_sha256")
            != worker_rows[worker_index]["stream_sha256"]
        ):
            raise ComparisonError()
        safe_body = dict(binding)
        safe_binding_sha256 = _sha256_field(
            safe_body.pop("binding_sha256")
        )
        if safe_binding_sha256 != _canonical_sha256(safe_body):
            raise ComparisonError()
        source_report_sha256 = _sha256_field(
            binding.get("source_report_sha256")
        )
        source_binding_sha256 = _sha256_field(
            binding.get("source_binding_sha256")
        )
        try:
            safe_report = qualification._retained_run_report(
                report,
                adapter=worker_adapters[worker_index],
                worker_manifest_sha256=worker_digests[worker_index],
                corpus_manifest_sha256=corpus_sha256,
                evaluator_implementation_sha256=evaluator_sha256,
                tags=STRATUM_TAGS,
                stratum_counts=tuple(
                    zip(STRATUM_TAGS, STRATUM_COUNTS, strict=True)
                ),
                repeats=FIXED_REPEATS,
                overrides=overrides,
                expected_stream=stream,
            )
        except Exception:
            raise ComparisonError() from None
        if (
            safe_report.get("source_report_sha256")
            != source_report_sha256
            or binding.get("report_sha256")
            != _canonical_sha256(safe_report)
        ):
            raise ComparisonError()
        source_binding = {
            "worker_index": worker_index,
            "corpus_index": 0,
            "worker_manifest_sha256": worker_digests[worker_index],
            "corpus_manifest_sha256": corpus_sha256,
            "stratum_selection_sha256": _canonical_sha256(
                list(STRATUM_TAGS)
            ),
            "evaluator_implementation_sha256": evaluator_sha256,
            "report_sha256": source_report_sha256,
        }
        if source_binding_sha256 != _canonical_sha256(source_binding):
            raise ComparisonError()
        source_binding["binding_sha256"] = source_binding_sha256
        safe_bindings.append(binding)
        source_bindings.append(source_binding)
        run_success.append(safe_report["ok"] is True)

    safe_matrix_sha256 = _canonical_sha256(safe_bindings)
    source_matrix_sha256 = _canonical_sha256(source_bindings)
    expected_ok = all(run_success)
    if (
        value.get("ok") is not expected_ok
        or controller.get("matrix_sha256") != safe_matrix_sha256
        or controller.get("completed_matrix_sha256") != safe_matrix_sha256
        or controller.get("source_matrix_sha256") != source_matrix_sha256
    ):
        raise ComparisonError()
    return dict(value)


def validate_comparison_report(value: object) -> dict[str, object]:
    """Strictly validate and copy one serialized aggregate comparison report."""

    try:
        encoded = _canonical_json_bytes(value)
        if len(encoded) + 1 > _MAX_REPORT_BYTES:
            raise ComparisonError()
        snapshot = json.loads(encoded)
        outer_fields = {
            "ok",
            "schema_version",
            "kind",
            "execution_complete",
            "execution_semantics",
            "quality_decision",
            "development_only",
            "promotional",
            "contract",
            "model_bindings",
            "source",
            "suite",
        }
        if (
            not isinstance(snapshot, dict)
            or set(snapshot) != outer_fields
            or type(snapshot.get("ok")) is not bool
            or not _exact_int(
                snapshot.get("schema_version"), SCHEMA_VERSION
            )
            or snapshot.get("kind") != KIND
            or snapshot.get("execution_complete") is not True
            or snapshot.get("execution_semantics") != "coverage_only"
            or snapshot.get("quality_decision") != "not_evaluated"
            or snapshot.get("development_only") is not True
            or snapshot.get("promotional") is not False
        ):
            raise ComparisonError()
        raw_contract = snapshot.get("contract")
        if not isinstance(raw_contract, Mapping):
            raise ComparisonError()
        worker_source_sha256 = _sha256_field(
            raw_contract.get("worker_source_sha256")
        )
        model_bindings, worker_digests, worker_adapters = (
            _validated_model_bindings(
                snapshot.get("model_bindings"),
                worker_source_sha256=worker_source_sha256,
            )
        )
        implementation_sha256, worker_rows = _validated_contract(
            raw_contract,
            model_bindings=model_bindings,
            worker_digests=worker_digests,
            worker_adapters=worker_adapters,
        )
        source = _validated_source_report(snapshot.get("source"))
        suite = _validate_retained_suite(
            snapshot.get("suite"),
            implementation_sha256=implementation_sha256,
            worker_digests=worker_digests,
            worker_adapters=worker_adapters,
            worker_rows=worker_rows,
            corpus_sha256=str(source["corpus_manifest_sha256"]),
        )
        if snapshot.get("ok") is not suite["ok"]:
            raise ComparisonError()
        _assert_path_free(snapshot)
        return snapshot
    except ComparisonError:
        raise
    except Exception:
        raise ComparisonError() from None


def run_comparison(
    *,
    baseline_worker_manifest: Path | str,
    candidate_worker_manifest: Path | str,
    corpus_path: Path | str,
    scratch_parent: Path | str,
    output_path: Path | str,
    lock_path: Path | str = fixture.DEFAULT_LOCK,
) -> dict[str, object]:
    """Run the fixed Zipformer/Faster-Whisper comparison exactly once."""

    try:
        implementation_sha256 = _implementation_sha256()
        final_output = qualification._new_private_output_path(output_path)
        scratch = qualification._scratch_candidate(scratch_parent)
        try:
            scratch.lstat()
        except FileNotFoundError:
            pass
        else:
            raise ComparisonError()
        source = _source_state(corpus_path, lock_path=lock_path)
        workers = _worker_state(
            baseline_worker_manifest,
            candidate_worker_manifest,
        )
        evaluator_state = _evaluator_state(workers)
        checkpoint = scratch / qualification._INNER_CHECKPOINT_NAME
        _validate_private_paths(
            final_output=final_output,
            scratch=scratch,
            checkpoint=checkpoint,
            source=source,
            worker_evidence_roots=workers.evidence_roots,
        )
        scratch, checkpoint, scratch_identity = (
            qualification._prepare_private_scratch(scratch)
        )
        if not callable(_PRODUCTION_RUNNER):
            raise ComparisonError()
        raw_result: object | None = None
        runner_failed = False
        try:
            raw_result = _PRODUCTION_RUNNER(
                workers.paths,
                (source.corpus_path,),
                scratch_parent=scratch,
                repeats=FIXED_REPEATS,
                chunk_samples=FIXED_CHUNK_SAMPLES,
                pace=FIXED_PACE,
                partial_interval_ms=FIXED_PARTIAL_INTERVAL_MS,
                tail_padding_samples=FIXED_TAIL_PADDING_SAMPLES,
                stratum_tags=(),
                corpus_stratum_tags={source.corpus_path: STRATUM_TAGS},
                checkpoint_path=checkpoint,
            )
        except BaseException:
            runner_failed = True

        post_validation_failed = False
        try:
            current_source = _source_state(corpus_path, lock_path=lock_path)
            current_workers = _worker_state(
                baseline_worker_manifest,
                candidate_worker_manifest,
            )
            if (
                _implementation_sha256() != implementation_sha256
                or current_source != source
                or current_workers != workers
                or _evaluator_state(current_workers) != evaluator_state
            ):
                raise ComparisonError()
            _validate_private_paths(
                final_output=final_output,
                scratch=scratch,
                checkpoint=checkpoint,
                source=source,
                worker_evidence_roots=workers.evidence_roots,
            )
            checkpoint_value = qualification._checkpoint_snapshot(
                scratch=scratch,
                checkpoint=checkpoint,
                scratch_identity=scratch_identity,
                required=not runner_failed,
            )
            if not runner_failed and checkpoint_value != raw_result:
                raise ComparisonError()
            if (
                qualification._new_private_output_path(final_output)
                != final_output
            ):
                raise ComparisonError()
        except BaseException:
            post_validation_failed = True
        if runner_failed or post_validation_failed or raw_result is None:
            raise ComparisonError()

        safe_suite = _reduce_source_suite(
            raw_result,
            source=source,
            workers=workers,
            implementation_sha256=implementation_sha256,
        )
        result = {
            "ok": safe_suite["ok"],
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "execution_complete": True,
            "execution_semantics": "coverage_only",
            "quality_decision": "not_evaluated",
            "development_only": True,
            "promotional": False,
            "contract": _contract(
                source,
                workers,
                implementation_sha256=implementation_sha256,
            ),
            "model_bindings": list(workers.model_bindings),
            "source": _source_report(source),
            "suite": safe_suite,
        }
        validated = validate_comparison_report(result)

        try:
            current_source = _source_state(corpus_path, lock_path=lock_path)
            current_workers = _worker_state(
                baseline_worker_manifest,
                candidate_worker_manifest,
            )
            if (
                _implementation_sha256() != implementation_sha256
                or current_source != source
                or current_workers != workers
                or _evaluator_state(current_workers) != evaluator_state
            ):
                raise ComparisonError()
            _validate_private_paths(
                final_output=final_output,
                scratch=scratch,
                checkpoint=checkpoint,
                source=source,
                worker_evidence_roots=workers.evidence_roots,
            )
            final_checkpoint = qualification._checkpoint_snapshot(
                scratch=scratch,
                checkpoint=checkpoint,
                scratch_identity=scratch_identity,
                required=True,
            )
            if (
                final_checkpoint != raw_result
                or qualification._new_private_output_path(final_output)
                != final_output
            ):
                raise ComparisonError()
        except BaseException:
            raise ComparisonError() from None
        return validated
    except ComparisonError:
        raise
    except Exception:
        raise ComparisonError() from None


def write_new_report(
    path: Path | str,
    value: Mapping[str, object],
) -> str:
    """Validate and descriptor-safely publish a new aggregate-only report."""

    try:
        validated = validate_comparison_report(value)
        selected = qualification._new_private_output_path(path)
        if (
            qualification._is_within(selected, _REPO_ROOT)
            or qualification._has_git_ancestor(selected)
        ):
            raise ComparisonError()
        return qualification.write_new_report(selected, validated)
    except ComparisonError:
        raise
    except Exception:
        raise ComparisonError() from None


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise ComparisonError()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description=(
            "Run the fixed development-only production-EdAcc Zipformer then "
            "Faster-Whisper Small comparison."
        )
    )
    parser.add_argument("--lock", type=Path, default=fixture.DEFAULT_LOCK)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--baseline-worker-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--candidate-worker-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        result = run_comparison(
            baseline_worker_manifest=args.baseline_worker_manifest,
            candidate_worker_manifest=args.candidate_worker_manifest,
            corpus_path=args.corpus,
            scratch_parent=args.scratch_root,
            output_path=args.output,
            lock_path=args.lock,
        )
        report_sha256 = write_new_report(args.output, result)
        summary = {
            "ok": result["ok"],
            "kind": KIND,
            "execution_complete": True,
            "quality_decision": "not_evaluated",
            "development_only": True,
            "promotional": False,
            "report_written": True,
            "report_sha256": report_sha256,
        }
        print(_canonical_json_bytes(summary).decode("ascii"))
        return 0 if result["ok"] is True else 1
    except Exception:
        print(_canonical_json_bytes(_SAFE_ERROR).decode("ascii"), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
