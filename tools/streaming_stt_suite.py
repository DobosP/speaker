"""Bounded sequential cross-product controller for streaming-STT benchmarks.

This module composes :mod:`tools.streaming_stt_eval` without changing its
single-worker/single-corpus default.  It deliberately starts one benchmark at
a time so model runtimes cannot overlap.  Reports remain aggregate-only: input
and scratch paths are never copied into the suite result.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import tempfile

from tools import streaming_stt_eval
from tools.streaming_stt.bounded_io import BoundedReadError, hash_regular_bounded
from tools.streaming_stt.metrics import normalize_stratum_tags
from tools.streaming_stt.protocol import (
    MAX_NATIVE_TAIL_PADDING_SAMPLES,
    MAX_STREAM_CHUNK_SAMPLES,
    StreamConfig,
)


SUITE_SCHEMA_VERSION = 1
MAX_WORKER_MANIFESTS = 8
MAX_CORPORA = 8
MAX_CELLS = 64

_MAX_DECLARED_PATHS = 64
_MAX_PATH_CHARS = 4096
_MAX_CONTROLLER_BYTES = 1024 * 1024
_MAX_NESTED_REPORT_BYTES = 2 * 1024 * 1024
_MAX_SUITE_REPORT_BYTES = 32 * 1024 * 1024
_EXPECTED_REPORT_FIELDS = frozenset(
    {
        "ok",
        "evidence",
        "worker",
        "corpus",
        "config",
        "metrics",
        "worker_stderr",
        "evaluator",
    }
)
_SAFE_ERROR = {"ok": False, "error": "streaming_stt_suite_failed"}
_DEFAULT_SCRATCH = Path(__file__).resolve().parents[1] / ".cache" / "streaming-stt"

BenchmarkRun = Callable[..., Mapping[str, object]]


class SuiteError(RuntimeError):
    """A detail-free suite validation, benchmark, or output failure."""

    def __init__(self) -> None:
        super().__init__("streaming STT suite failed")


def _strict_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise SuiteError() from None


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_strict_json(value).encode("utf-8")).hexdigest()


def _normalize_path(value: Path | str) -> Path:
    if isinstance(value, bytes) or not isinstance(value, (str, os.PathLike)):
        raise SuiteError()
    try:
        raw = os.fspath(value)
        if not isinstance(raw, str) or not 0 < len(raw) <= _MAX_PATH_CHARS:
            raise SuiteError()
        selected = Path(os.path.abspath(Path(raw).expanduser())).resolve(strict=False)
    except SuiteError:
        raise
    except Exception:
        raise SuiteError() from None
    if not selected.is_absolute() or len(str(selected)) > _MAX_PATH_CHARS:
        raise SuiteError()
    return selected


def _normalize_output_path(value: Path | str) -> Path:
    """Resolve parent components while retaining the final lexical filename."""

    if isinstance(value, bytes) or not isinstance(value, (str, os.PathLike)):
        raise SuiteError()
    try:
        raw = os.fspath(value)
        if not isinstance(raw, str) or not 0 < len(raw) <= _MAX_PATH_CHARS:
            raise SuiteError()
        lexical = Path(os.path.abspath(Path(raw).expanduser()))
        if not lexical.name:
            raise SuiteError()
        selected = lexical.parent.resolve(strict=False) / lexical.name
    except SuiteError:
        raise
    except Exception:
        raise SuiteError() from None
    if not selected.is_absolute() or len(str(selected)) > _MAX_PATH_CHARS:
        raise SuiteError()
    return selected


def _bounded_unique_paths(
    values: Iterable[Path | str],
    *,
    maximum: int,
) -> tuple[Path, ...]:
    if isinstance(values, (str, bytes, os.PathLike)):
        raise SuiteError()
    selected: list[Path] = []
    seen: set[str] = set()
    try:
        iterator = iter(values)
        for index in range(_MAX_DECLARED_PATHS + 1):
            try:
                raw = next(iterator)
            except StopIteration:
                break
            if index >= _MAX_DECLARED_PATHS:
                raise SuiteError()
            path = _normalize_path(raw)
            key = os.path.normcase(str(path))
            if key in seen:
                continue
            seen.add(key)
            selected.append(path)
            if len(selected) > maximum:
                raise SuiteError()
    except SuiteError:
        raise
    except Exception:
        raise SuiteError() from None
    if not selected:
        raise SuiteError()
    return tuple(selected)


def _bounded_integer(
    value: int | None,
    *,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SuiteError()
    if not minimum <= value <= maximum:
        raise SuiteError()
    return value


def _validated_stream_options(
    *,
    stream: StreamConfig | None,
    chunk_samples: int | None,
    pace: str | None,
    partial_interval_ms: int | None,
    tail_padding_samples: int | None,
) -> None:
    overrides = (
        chunk_samples,
        pace,
        partial_interval_ms,
        tail_padding_samples,
    )
    if stream is not None:
        if type(stream) is not StreamConfig or any(
            value is not None for value in overrides
        ):
            raise SuiteError()
        chunk_samples = stream.chunk_samples
        pace = stream.pace
        partial_interval_ms = stream.partial_interval_ms
        tail_padding_samples = stream.tail_padding_samples
    _bounded_integer(
        chunk_samples,
        minimum=1,
        maximum=MAX_STREAM_CHUNK_SAMPLES,
    )
    if pace is not None and pace not in {"burst", "realtime"}:
        raise SuiteError()
    _bounded_integer(partial_interval_ms, minimum=1, maximum=5000)
    _bounded_integer(
        tail_padding_samples,
        minimum=0,
        maximum=MAX_NATIVE_TAIL_PADDING_SAMPLES,
    )


def _validated_stratum_plan(
    corpora: Sequence[Path],
    *,
    stratum_tags: Sequence[str],
    corpus_stratum_tags: Mapping[Path | str, Sequence[str]] | None,
) -> tuple[str, tuple[tuple[str, ...], ...]]:
    try:
        global_tags = normalize_stratum_tags(stratum_tags)
    except Exception:
        raise SuiteError() from None
    if corpus_stratum_tags is None:
        raw_items: tuple[tuple[Path | str, Sequence[str]], ...] = ()
    else:
        if not isinstance(corpus_stratum_tags, Mapping):
            raise SuiteError()
        try:
            iterator = iter(corpus_stratum_tags.items())
            bounded_items: list[tuple[Path | str, Sequence[str]]] = []
            for index in range(MAX_CORPORA + 1):
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                if index >= MAX_CORPORA or not isinstance(item, tuple) or len(item) != 2:
                    raise SuiteError()
                bounded_items.append(item)
            raw_items = tuple(bounded_items)
        except Exception:
            raise SuiteError() from None
    if global_tags and raw_items:
        raise SuiteError()

    corpus_index = {
        os.path.normcase(str(corpus)): index for index, corpus in enumerate(corpora)
    }
    selected: list[tuple[str, ...]] = [()] * len(corpora)
    seen: set[str] = set()
    for raw_path, raw_tags in raw_items:
        path = _normalize_path(raw_path)
        key = os.path.normcase(str(path))
        if key in seen or key not in corpus_index:
            raise SuiteError()
        seen.add(key)
        try:
            tags = normalize_stratum_tags(raw_tags)
        except Exception:
            raise SuiteError() from None
        if not tags:
            raise SuiteError()
        selected[corpus_index[key]] = tags

    if raw_items:
        return "per_corpus", tuple(selected)
    if global_tags:
        return "global", tuple(global_tags for _corpus in corpora)
    return "none", tuple(selected)


def _is_absolute_path_text(value: str) -> bool:
    if value.startswith(("/", "file://", "~/", "\\\\")):
        return True
    try:
        return PureWindowsPath(value).is_absolute()
    except Exception:
        return True


def _assert_path_free(value: object) -> None:
    if isinstance(value, str):
        if _is_absolute_path_text(value):
            raise SuiteError()
        return
    if value is None or isinstance(value, (bool, int, float)):
        return
    if isinstance(value, list):
        for item in value:
            _assert_path_free(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SuiteError()
            _assert_path_free(key)
            _assert_path_free(item)
        return
    raise SuiteError()


def _sha256_field(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SuiteError()
    return value


def _evaluator_implementation_sha256(evaluator: object) -> str:
    if not isinstance(evaluator, dict):
        raise SuiteError()
    schema_version = evaluator.get("schema_version")
    files = evaluator.get("files")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 1
        or not isinstance(files, dict)
        or not files
        or len(files) > 64
    ):
        raise SuiteError()
    for relative_path, digest in files.items():
        relative = PurePosixPath(relative_path) if isinstance(relative_path, str) else None
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or len(relative_path) > 256
            or _is_absolute_path_text(relative_path)
            or "\\" in relative_path
            or relative is None
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or str(relative) != relative_path
        ):
            raise SuiteError()
        _sha256_field(digest)
    return _canonical_digest(
        {
            "schema_version": schema_version,
            "files": files,
        }
    )


def _snapshot_nested_report(
    value: Mapping[str, object],
    *,
    expected_stratum_tags: Sequence[str],
) -> tuple[dict[str, object], str]:
    if not isinstance(value, Mapping) or set(value) != _EXPECTED_REPORT_FIELDS:
        raise SuiteError()
    encoded = _strict_json(value).encode("utf-8")
    if len(encoded) > _MAX_NESTED_REPORT_BYTES:
        raise SuiteError()
    try:
        snapshot = json.loads(encoded)
    except (json.JSONDecodeError, UnicodeError, RecursionError):
        raise SuiteError() from None
    if not isinstance(snapshot, dict) or type(snapshot.get("ok")) is not bool:
        raise SuiteError()
    worker = snapshot.get("worker")
    corpus = snapshot.get("corpus")
    config = snapshot.get("config")
    metrics = snapshot.get("metrics")
    if (
        not isinstance(worker, dict)
        or not isinstance(corpus, dict)
        or not isinstance(config, dict)
        or not isinstance(metrics, dict)
        or type(metrics.get("coverage_complete")) is not bool
    ):
        raise SuiteError()
    if snapshot["ok"] is True and metrics["coverage_complete"] is not True:
        raise SuiteError()

    contract_sha256 = _sha256_field(config.get("contract_sha256"))
    contract = dict(config)
    del contract["contract_sha256"]
    if contract_sha256 != _canonical_digest(contract):
        raise SuiteError()

    try:
        selected_tags = tuple(expected_stratum_tags)
        if normalize_stratum_tags(selected_tags) != selected_tags:
            raise SuiteError()
    except SuiteError:
        raise
    except Exception:
        raise SuiteError() from None
    evidence = snapshot.get("evidence")
    if not isinstance(evidence, dict):
        raise SuiteError()
    if selected_tags:
        if config.get("stratum_tags") != list(selected_tags):
            raise SuiteError()
        selection_sha256 = _canonical_digest(list(selected_tags))
        if evidence.get("stratification") != {
            "kind": "explicit_corpus_case_tags",
            "aggregate_only": True,
            "overlapping": True,
            "tags": len(selected_tags),
            "selection_sha256": selection_sha256,
        }:
            raise SuiteError()
        strata = metrics.get("strata")
        if not isinstance(strata, list) or len(strata) != len(selected_tags):
            raise SuiteError()
        for expected_tag, row in zip(selected_tags, strata, strict=True):
            if not isinstance(row, dict) or set(row) != {"tag", "cases", "metrics"}:
                raise SuiteError()
            cases = row.get("cases")
            nested_metrics = row.get("metrics")
            if (
                row.get("tag") != expected_tag
                or type(cases) is not int
                or cases <= 0
                or not isinstance(nested_metrics, dict)
                or nested_metrics.get("coverage_complete") is not True
                or type(nested_metrics.get("evaluations")) is not int
                or nested_metrics["evaluations"] <= 0
                or nested_metrics["evaluations"] % cases != 0
                or "strata" in nested_metrics
            ):
                raise SuiteError()
    elif (
        "stratum_tags" in config
        or "stratification" in evidence
        or "strata" in metrics
    ):
        raise SuiteError()
    _sha256_field(worker.get("manifest_sha256"))
    _sha256_field(corpus.get("manifest_sha256"))
    evaluator_implementation_sha256 = _evaluator_implementation_sha256(
        snapshot.get("evaluator")
    )
    _assert_path_free(snapshot)
    return snapshot, evaluator_implementation_sha256


def _controller_source_sha256() -> str:
    try:
        path = Path(__file__).resolve(strict=True)
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or not 0 < metadata.st_size <= _MAX_CONTROLLER_BYTES
        ):
            raise SuiteError()
        source = hash_regular_bounded(
            path,
            maximum_bytes=_MAX_CONTROLLER_BYTES,
            expected_bytes=metadata.st_size,
        )
    except SuiteError:
        raise
    except (OSError, RuntimeError, ValueError, BoundedReadError):
        raise SuiteError() from None
    return source.sha256


def _stream_contract(
    *,
    repeats: int,
    stream: StreamConfig | None,
    chunk_samples: int | None,
    pace: str | None,
    partial_interval_ms: int | None,
    tail_padding_samples: int | None,
    stratum_mode: str,
    stratum_plan: Sequence[Sequence[str]],
) -> dict[str, object]:
    strata = [
        {
            "corpus_index": index,
            "tags": list(tags),
            "selection_sha256": _canonical_digest(list(tags)),
        }
        for index, tags in enumerate(stratum_plan)
    ]
    selected: dict[str, object] = {
        "repeats": repeats,
        "strata": {
            "mode": stratum_mode,
            "corpora": strata,
            "selection_sha256": _canonical_digest(strata),
        },
    }
    if stratum_mode == "global":
        selected["stratum_tags"] = list(stratum_plan[0])
    if stream is not None:
        selected["stream"] = stream.as_dict()
    else:
        selected["overrides"] = {
            "chunk_samples": chunk_samples,
            "pace": pace,
            "partial_interval_ms": partial_interval_ms,
            "tail_padding_samples": tail_padding_samples,
        }
    selected["contract_sha256"] = _canonical_digest(selected)
    return selected


def _suite_report(
    *,
    state: str,
    planned_workers: int,
    planned_corpora: int,
    worker_digests: Sequence[str | None],
    corpus_digests: Sequence[str | None],
    evaluator_implementation_sha256: str | None,
    controller_sha256: str,
    config: Mapping[str, object],
    runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if state not in {"in_progress", "failed", "complete"}:
        raise SuiteError()
    complete = state == "complete"
    planned_runs = planned_workers * planned_corpora
    if (
        len(runs) > planned_runs
        or (complete and len(runs) != planned_runs)
        or (runs and evaluator_implementation_sha256 is None)
    ):
        raise SuiteError()
    run_bindings = [item["binding"] for item in runs]
    completed_worker_digests = [
        value for value in worker_digests if value is not None
    ]
    completed_corpus_digests = [
        value for value in corpus_digests if value is not None
    ]
    controller: dict[str, object] = {
        "kind": "bounded_sequential_streaming_stt_suite",
        "schema_version": SUITE_SCHEMA_VERSION,
        "execution": "strictly_sequential_cross_product",
        "state": state,
        "planned_worker_manifests": planned_workers,
        "planned_corpora": planned_corpora,
        "planned_runs": planned_runs,
        "completed_runs": len(runs),
        "controller_sha256": controller_sha256,
        "completed_worker_set_sha256": _canonical_digest(
            completed_worker_digests
        ),
        "completed_corpus_set_sha256": _canonical_digest(
            completed_corpus_digests
        ),
        "completed_matrix_sha256": _canonical_digest(run_bindings),
        "stratum_selection_sha256": config["strata"][  # type: ignore[index]
            "selection_sha256"
        ],
    }
    if evaluator_implementation_sha256 is not None:
        controller["evaluator_implementation_sha256"] = (
            evaluator_implementation_sha256
        )
    if complete:
        if any(value is None for value in (*worker_digests, *corpus_digests)):
            raise SuiteError()
        controller.update(
            {
                "worker_manifests": planned_workers,
                "corpora": planned_corpora,
                "runs": planned_runs,
                "worker_set_sha256": _canonical_digest(worker_digests),
                "corpus_set_sha256": _canonical_digest(corpus_digests),
                "matrix_sha256": _canonical_digest(run_bindings),
            }
        )
    result: dict[str, object] = {
        "ok": complete
        and all(item["report"]["ok"] is True for item in runs),  # type: ignore[index]
        "complete": complete,
        "controller": controller,
        "config": dict(config),
        "runs": list(runs),
    }
    _assert_path_free(result)
    if len(_strict_json(result).encode("utf-8")) > _MAX_SUITE_REPORT_BYTES:
        raise SuiteError()
    return result


def run_suite(
    worker_manifest_paths: Iterable[Path | str],
    corpus_paths: Iterable[Path | str],
    *,
    scratch_parent: Path | str,
    repeats: int = 3,
    stream: StreamConfig | None = None,
    chunk_samples: int | None = None,
    pace: str | None = None,
    partial_interval_ms: int | None = None,
    tail_padding_samples: int | None = None,
    stratum_tags: Sequence[str] = (),
    corpus_stratum_tags: Mapping[Path | str, Sequence[str]] | None = None,
    checkpoint_path: Path | str | None = None,
    run_benchmark_fn: BenchmarkRun | None = None,
) -> dict[str, object]:
    """Run a deduplicated cross-product and checkpoint every completed cell.

    When ``checkpoint_path`` is supplied it always contains this invocation's
    current red/incomplete state, then atomically becomes the final report.
    """

    try:
        workers = _bounded_unique_paths(
            worker_manifest_paths,
            maximum=MAX_WORKER_MANIFESTS,
        )
        corpora = _bounded_unique_paths(corpus_paths, maximum=MAX_CORPORA)
        if len(workers) * len(corpora) > MAX_CELLS:
            raise SuiteError()
        if isinstance(repeats, bool) or not isinstance(repeats, int):
            raise SuiteError()
        if not 1 <= repeats <= 8:
            raise SuiteError()
        _validated_stream_options(
            stream=stream,
            chunk_samples=chunk_samples,
            pace=pace,
            partial_interval_ms=partial_interval_ms,
            tail_padding_samples=tail_padding_samples,
        )
        stratum_mode, stratum_plan = _validated_stratum_plan(
            corpora,
            stratum_tags=stratum_tags,
            corpus_stratum_tags=corpus_stratum_tags,
        )
        config = _stream_contract(
            repeats=repeats,
            stream=stream,
            chunk_samples=chunk_samples,
            pace=pace,
            partial_interval_ms=partial_interval_ms,
            tail_padding_samples=tail_padding_samples,
            stratum_mode=stratum_mode,
            stratum_plan=stratum_plan,
        )
        scratch = _normalize_path(scratch_parent)
        checkpoint = (
            None
            if checkpoint_path is None
            else _normalize_output_path(checkpoint_path)
        )
        if checkpoint is not None and os.path.normcase(str(checkpoint)) in {
            os.path.normcase(str(path)) for path in (*workers, *corpora)
        }:
            raise SuiteError()
        runner = (
            streaming_stt_eval.run_benchmark
            if run_benchmark_fn is None
            else run_benchmark_fn
        )
        if not callable(runner):
            raise SuiteError()
        controller_sha256 = _controller_source_sha256()

        runs: list[dict[str, object]] = []
        worker_digests: list[str | None] = [None] * len(workers)
        corpus_digests: list[str | None] = [None] * len(corpora)
        evaluator_implementation_sha256: str | None = None
        total_report_bytes = 0
        if checkpoint is not None:
            write_report(
                checkpoint,
                _suite_report(
                    state="in_progress",
                    planned_workers=len(workers),
                    planned_corpora=len(corpora),
                    worker_digests=worker_digests,
                    corpus_digests=corpus_digests,
                    evaluator_implementation_sha256=None,
                    controller_sha256=controller_sha256,
                    config=config,
                    runs=runs,
                ),
            )
        try:
            for worker_index, worker_manifest in enumerate(workers):
                for corpus_index, corpus in enumerate(corpora):
                    selected_tags = stratum_plan[corpus_index]
                    nested, nested_evaluator_sha256 = _snapshot_nested_report(
                        runner(
                            worker_manifest,
                            corpus,
                            scratch_parent=scratch,
                            repeats=repeats,
                            stream=stream,
                            chunk_samples=chunk_samples,
                            pace=pace,
                            partial_interval_ms=partial_interval_ms,
                            tail_padding_samples=tail_padding_samples,
                            stratum_tags=selected_tags,
                        ),
                        expected_stratum_tags=selected_tags,
                    )
                    if evaluator_implementation_sha256 is None:
                        evaluator_implementation_sha256 = nested_evaluator_sha256
                    elif (
                        nested_evaluator_sha256
                        != evaluator_implementation_sha256
                    ):
                        raise SuiteError()
                    nested_bytes = len(_strict_json(nested).encode("utf-8"))
                    total_report_bytes += nested_bytes
                    if total_report_bytes > _MAX_SUITE_REPORT_BYTES:
                        raise SuiteError()
                    worker = nested["worker"]
                    corpus_report = nested["corpus"]
                    if not isinstance(worker, dict) or not isinstance(
                        corpus_report, dict
                    ):
                        raise SuiteError()
                    worker_digest = _sha256_field(
                        worker.get("manifest_sha256")
                    )
                    corpus_digest = _sha256_field(
                        corpus_report.get("manifest_sha256")
                    )
                    previous_worker = worker_digests[worker_index]
                    previous_corpus = corpus_digests[corpus_index]
                    if (
                        previous_worker is not None
                        and previous_worker != worker_digest
                    ):
                        raise SuiteError()
                    if (
                        previous_corpus is not None
                        and previous_corpus != corpus_digest
                    ):
                        raise SuiteError()
                    worker_digests[worker_index] = worker_digest
                    corpus_digests[corpus_index] = corpus_digest
                    stratum_selection_sha256 = _canonical_digest(
                        list(selected_tags)
                    )
                    binding = {
                        "worker_index": worker_index,
                        "corpus_index": corpus_index,
                        "worker_manifest_sha256": worker_digest,
                        "corpus_manifest_sha256": corpus_digest,
                        "stratum_selection_sha256": stratum_selection_sha256,
                        "evaluator_implementation_sha256": (
                            evaluator_implementation_sha256
                        ),
                        "report_sha256": _canonical_digest(nested),
                    }
                    binding["binding_sha256"] = _canonical_digest(binding)
                    runs.append({"binding": binding, "report": nested})
                    if checkpoint is not None:
                        write_report(
                            checkpoint,
                            _suite_report(
                                state="in_progress",
                                planned_workers=len(workers),
                                planned_corpora=len(corpora),
                                worker_digests=worker_digests,
                                corpus_digests=corpus_digests,
                                evaluator_implementation_sha256=(
                                    evaluator_implementation_sha256
                                ),
                                controller_sha256=controller_sha256,
                                config=config,
                                runs=runs,
                            ),
                        )

            if _controller_source_sha256() != controller_sha256:
                raise SuiteError()
            result = _suite_report(
                state="complete",
                planned_workers=len(workers),
                planned_corpora=len(corpora),
                worker_digests=worker_digests,
                corpus_digests=corpus_digests,
                evaluator_implementation_sha256=(
                    evaluator_implementation_sha256
                ),
                controller_sha256=controller_sha256,
                config=config,
                runs=runs,
            )
            if checkpoint is not None:
                write_report(checkpoint, result)
            return result
        except Exception:
            if checkpoint is not None:
                try:
                    write_report(
                        checkpoint,
                        _suite_report(
                            state="failed",
                            planned_workers=len(workers),
                            planned_corpora=len(corpora),
                            worker_digests=worker_digests,
                            corpus_digests=corpus_digests,
                            evaluator_implementation_sha256=(
                                evaluator_implementation_sha256
                            ),
                            controller_sha256=controller_sha256,
                            config=config,
                            runs=runs,
                        ),
                    )
                except Exception:
                    pass
            raise SuiteError() from None
    except SuiteError:
        raise
    except Exception:
        raise SuiteError() from None


def write_report(path: Path | str, payload: Mapping[str, object]) -> None:
    """Atomically write one standards-compliant aggregate report at mode 600."""

    selected = _normalize_output_path(path)
    _assert_path_free(payload)
    encoded = (_strict_json(payload) + "\n").encode("utf-8")
    if len(encoded) > _MAX_SUITE_REPORT_BYTES:
        raise SuiteError()
    descriptor = -1
    temporary = ""
    try:
        selected.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".streaming-stt-suite-",
            dir=selected.parent,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, selected)
        temporary = ""
        os.chmod(selected, 0o600, follow_symlinks=False)
        metadata = selected.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise SuiteError()
    except SuiteError:
        raise
    except Exception:
        raise SuiteError() from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded aggregate-only worker/corpus matrix strictly one "
            "benchmark at a time."
        )
    )
    parser.add_argument(
        "--worker-manifest",
        action="append",
        required=True,
        type=Path,
    )
    parser.add_argument("--corpus", action="append", required=True, type=Path)
    parser.add_argument("--scratch-root", type=Path, default=_DEFAULT_SCRATCH)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk-samples", type=int)
    parser.add_argument("--pace", choices=("burst", "realtime"))
    parser.add_argument("--partial-interval-ms", type=int)
    parser.add_argument("--tail-padding-samples", type=int)
    parser.add_argument(
        "--stratum-tag",
        action="append",
        default=[],
        help="bounded aggregate stratum required in every selected corpus",
    )
    parser.add_argument(
        "--corpus-stratum-tag",
        action="append",
        nargs=2,
        default=[],
        metavar=("CORPUS", "TAG"),
        help=(
            "corpus path and tag for a corpus-specific aggregate stratum; "
            "repeat for additional tags"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="atomic mode-600 final report and per-cell red checkpoint",
    )
    return parser


def _cli_corpus_stratum_tags(
    values: Sequence[Sequence[str]],
) -> Mapping[str, Sequence[str]] | None:
    if len(values) > MAX_CORPORA * 32:
        raise SuiteError()
    selected: dict[str, list[str]] = {}
    for value in values:
        if len(value) != 2:
            raise SuiteError()
        corpus, tag = value
        selected.setdefault(corpus, []).append(tag)
    return selected or None


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = run_suite(
            args.worker_manifest,
            args.corpus,
            scratch_parent=args.scratch_root,
            repeats=args.repeats,
            chunk_samples=args.chunk_samples,
            pace=args.pace,
            partial_interval_ms=args.partial_interval_ms,
            tail_padding_samples=args.tail_padding_samples,
            stratum_tags=args.stratum_tag,
            corpus_stratum_tags=_cli_corpus_stratum_tags(
                args.corpus_stratum_tag
            ),
            checkpoint_path=args.output,
        )
        print(_strict_json(payload))
        return 0 if payload["ok"] else 1
    except Exception:  # noqa: BLE001 - paths and native/model details stay private
        print(_strict_json(_SAFE_ERROR))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
