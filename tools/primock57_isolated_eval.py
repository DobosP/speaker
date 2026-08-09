"""Receipt-bound PriMock57 isolated streaming-STT evaluation wrapper.

The generic streaming evaluator deliberately accepts many corpus families.
This narrow wrapper admits only the exact production PriMock57 isolated bundle,
revalidates it after execution, and adds an aggregate-only binding that cannot
be confused with the generic evaluator's raw report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re
from typing import Final

from tools import prepare_primock57_conversation_fixture as primock57_fixture
from tools import streaming_stt_eval
from tools.streaming_stt.bounded_io import hash_regular_bounded
from tools.streaming_stt.protocol import StreamConfig


SCHEMA_VERSION: Final = 1
KIND: Final = "primock57-isolated-production-stt-eval-binding-v1"
BINDING_FIELD: Final = "primock57_binding"

_MAX_SOURCE_BYTES: Final = 4 * 1024 * 1024
_REFERENCE_WORD_RE: Final = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_WINDOWS_LOCAL_PATH_RE: Final = re.compile(r"(?i)(?<![a-z0-9])[a-z]:(?!//)\S+")
_WINDOWS_ROOTED_PATH_RE: Final = re.compile(r"(?i)(?<![a-z0-9_.-])\\(?![\\\s])[^\\\s]+")
_REPORT_FIELDS: Final = frozenset(
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
_PRODUCTION_RUNNER = streaming_stt_eval.run_benchmark


class Primock57IsolatedEvalError(RuntimeError):
    """A detail-free source, execution, or aggregate-report contract failure."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise Primock57IsolatedEvalError() from None


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Primock57IsolatedEvalError()
    return value


def _closure_sha256() -> str:
    """Bind the wrapper and the two production verifier/executor front doors."""

    try:
        repo_root = Path(__file__).resolve(strict=True).parents[1]
        selected = (
            ("tools/primock57_isolated_eval.py", Path(__file__)),
            (
                "tools/prepare_primock57_conversation_fixture.py",
                Path(primock57_fixture.__file__),
            ),
            ("tools/streaming_stt_eval.py", Path(streaming_stt_eval.__file__)),
        )
        rows: list[dict[str, object]] = []
        for expected_name, candidate in selected:
            path = candidate.resolve(strict=True)
            if path.relative_to(repo_root).as_posix() != expected_name:
                raise Primock57IsolatedEvalError()
            metadata = path.lstat()
            if metadata.st_size <= 0 or metadata.st_size > _MAX_SOURCE_BYTES:
                raise Primock57IsolatedEvalError()
            snapshot = hash_regular_bounded(
                path,
                maximum_bytes=_MAX_SOURCE_BYTES,
                expected_bytes=metadata.st_size,
            )
            rows.append(
                {
                    "name": expected_name,
                    "sha256": snapshot.sha256,
                    "size_bytes": snapshot.size_bytes,
                }
            )
        return _canonical_sha256(rows)
    except Primock57IsolatedEvalError:
        raise
    except Exception:
        raise Primock57IsolatedEvalError() from None


def _is_absolute_path_text(value: str) -> bool:
    lowered = value.casefold()
    if (
        value.startswith(("/", "~/", "\\\\"))
        or "file://" in lowered
        or _WINDOWS_LOCAL_PATH_RE.search(value) is not None
        or _WINDOWS_ROOTED_PATH_RE.search(value) is not None
    ):
        return True
    try:
        if PureWindowsPath(value).is_absolute():
            return True
    except Exception:
        return True
    for index, character in enumerate(value):
        if character != "/":
            continue
        if index > 0 and (value[index - 1].isalnum() or value[index - 1] in "_./-"):
            continue
        if index > 0 and value[index - 1] == "~":
            return True
        prefix = value[:index]
        if (
            index + 1 < len(value)
            and value[index + 1] == "/"
            and re.search(r"[a-z][a-z0-9+.-]*:$", prefix, re.IGNORECASE) is not None
        ):
            continue
        return True
    return False


def _normalized_words(value: str) -> tuple[str, ...]:
    return tuple(_REFERENCE_WORD_RE.findall(value.casefold()))


def _forbidden_reference_fragments(values: tuple[str, ...]) -> frozenset[str]:
    fragments: set[str] = set()
    for value in values:
        words = _normalized_words(value)
        if len(words) == 1:
            fragments.add(words[0])
            continue
        for start in range(len(words)):
            for stop in range(start + 2, len(words) + 1):
                fragments.add(" ".join(words[start:stop]))
    return frozenset(fragments)


def _contains_forbidden_reference(
    value: str,
    *,
    fragments: frozenset[str],
) -> bool:
    normalized = " ".join(_normalized_words(value))
    if not normalized:
        return False
    bounded = f" {normalized} "
    return any(f" {fragment} " in bounded for fragment in fragments)


def _assert_aggregate_private(
    value: object,
    *,
    forbidden_text: tuple[str, ...],
    forbidden_paths: tuple[str, ...] = (),
    _forbidden_fragments: frozenset[str] | None = None,
) -> None:
    """Reject non-JSON values, embedded local paths, and reference fragments."""

    fragments = (
        _forbidden_reference_fragments(forbidden_text)
        if _forbidden_fragments is None
        else _forbidden_fragments
    )

    if isinstance(value, str):
        if (
            _is_absolute_path_text(value)
            or any(
                path and path.casefold() in value.casefold() for path in forbidden_paths
            )
            or _contains_forbidden_reference(
                value,
                fragments=fragments,
            )
        ):
            raise Primock57IsolatedEvalError()
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Primock57IsolatedEvalError()
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_aggregate_private(
                item,
                forbidden_text=forbidden_text,
                forbidden_paths=forbidden_paths,
                _forbidden_fragments=fragments,
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise Primock57IsolatedEvalError()
            _assert_aggregate_private(
                key,
                forbidden_text=forbidden_text,
                forbidden_paths=forbidden_paths,
                _forbidden_fragments=fragments,
            )
            _assert_aggregate_private(
                item,
                forbidden_text=forbidden_text,
                forbidden_paths=forbidden_paths,
                _forbidden_fragments=fragments,
            )
        return
    raise Primock57IsolatedEvalError()


def _validated_worker_stderr(value: object) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"bytes", "sha256", "truncated"}
        or type(value.get("bytes")) is not int
        or value["bytes"] < 0
        or value.get("sha256") != _sha256(value.get("sha256"))
        or type(value.get("truncated")) is not bool
    ):
        raise Primock57IsolatedEvalError()


def _validated_evaluator(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "kind",
        "production_model",
        "real_candidate_model",
        "model_executed",
        "files",
    }:
        raise Primock57IsolatedEvalError()
    files = value.get("files")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("kind")
        not in {
            "isolated_streaming_stt_fake_harness",
            "isolated_streaming_stt_production_harness",
            "isolated_streaming_stt_candidate_harness",
        }
        or any(
            type(value.get(field)) is not bool
            for field in (
                "production_model",
                "real_candidate_model",
                "model_executed",
            )
        )
        or not isinstance(files, Mapping)
        or not files
    ):
        raise Primock57IsolatedEvalError()
    for name, digest in files.items():
        if (
            type(name) is not str
            or not name
            or _is_absolute_path_text(name)
            or digest != _sha256(digest)
        ):
            raise Primock57IsolatedEvalError()


def _resolve_corpus_path(corpus_path: Path | str) -> Path:
    try:
        path = Path(corpus_path).expanduser().resolve(strict=True)
        if path.name != "corpus.json" or not path.is_file():
            raise Primock57IsolatedEvalError()
        return path
    except Primock57IsolatedEvalError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise Primock57IsolatedEvalError() from None


def _validated_bundle(
    corpus_path: Path,
) -> primock57_fixture.LoadedPrimock57IsolatedBundle:
    try:
        bundle = primock57_fixture.load_primock57_isolated_bundle(corpus_path.parent)
        corpus = bundle.corpus
        provenance = corpus.provenance
        if (
            bundle.production_evidence is not True
            or corpus.path != corpus_path
            or corpus.schema_version != 2
            or corpus.digest != _sha256(corpus.digest)
            or len(corpus.cases) != 3
            or provenance is None
            or provenance.suite != primock57_fixture.FIXTURE_ID
            or bundle.receipt_sha256 != _sha256(bundle.receipt_sha256)
            or bundle.source_contract_sha256 != _sha256(bundle.source_contract_sha256)
        ):
            raise Primock57IsolatedEvalError()
        return bundle
    except Primock57IsolatedEvalError:
        raise
    except BaseException:
        raise Primock57IsolatedEvalError() from None


def _corpus_binding(
    bundle: primock57_fixture.LoadedPrimock57IsolatedBundle,
) -> dict[str, object]:
    try:
        value = streaming_stt_eval._corpus_binding(bundle.corpus)
        if (
            not isinstance(value, dict)
            or value.get("manifest_sha256") != bundle.corpus.digest
            or type(value.get("cases")) is not int
            or value.get("cases") != len(bundle.corpus.cases)
        ):
            raise Primock57IsolatedEvalError()
        return value
    except Primock57IsolatedEvalError:
        raise
    except Exception:
        raise Primock57IsolatedEvalError() from None


def _validated_raw_report(
    value: object,
    *,
    bundle: primock57_fixture.LoadedPrimock57IsolatedBundle,
    corpus_binding: Mapping[str, object],
) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _REPORT_FIELDS
        or BINDING_FIELD in value
        or type(value.get("ok")) is not bool
        or value.get("corpus") != corpus_binding
    ):
        raise Primock57IsolatedEvalError()
    for field in ("evidence", "worker", "config", "metrics"):
        if not isinstance(value.get(field), Mapping):
            raise Primock57IsolatedEvalError()
    _validated_worker_stderr(value.get("worker_stderr"))
    _validated_evaluator(value.get("evaluator"))
    forbidden_text = tuple(
        case.expected_text for case in bundle.corpus.cases if case.expected_text
    )
    forbidden_paths = tuple(
        text
        for case in bundle.corpus.cases
        for text in (str(case.source_path), case.source_path.name)
    )
    _assert_aggregate_private(
        value,
        forbidden_text=forbidden_text,
        forbidden_paths=forbidden_paths,
    )
    result = dict(value)
    _canonical_json(result)
    return result


def _same_snapshot(
    expected: primock57_fixture.LoadedPrimock57IsolatedBundle,
    current: primock57_fixture.LoadedPrimock57IsolatedBundle,
) -> bool:
    """Use the loader's complete corpus bytes and private inode snapshots."""

    return current == expected


def run_primock57_isolated_eval(
    worker_manifest_path: Path | str,
    corpus_path: Path | str,
    *,
    scratch_parent: Path | str,
    repeats: int = 3,
    stream: StreamConfig | None = None,
    chunk_samples: int | None = None,
    pace: str | None = None,
    partial_interval_ms: int | None = None,
    tail_padding_samples: int | None = None,
    stratum_tags: Sequence[str] = (),
) -> dict[str, object]:
    """Evaluate one exact production PriMock57 isolated corpus aggregate."""

    try:
        resolved_corpus = _resolve_corpus_path(corpus_path)
        closure_sha256 = _closure_sha256()
        initial = _validated_bundle(resolved_corpus)
        expected_corpus = _corpus_binding(initial)
        if not callable(_PRODUCTION_RUNNER):
            raise Primock57IsolatedEvalError()

        raw_result: object | None = None
        runner_failed = False
        try:
            raw_result = _PRODUCTION_RUNNER(
                worker_manifest_path,
                resolved_corpus,
                scratch_parent=scratch_parent,
                repeats=repeats,
                stream=stream,
                chunk_samples=chunk_samples,
                pace=pace,
                partial_interval_ms=partial_interval_ms,
                tail_padding_samples=tail_padding_samples,
                stratum_tags=stratum_tags,
            )
        except BaseException:
            runner_failed = True

        post_validation_failed = False
        try:
            current = _validated_bundle(resolved_corpus)
            if (
                not _same_snapshot(initial, current)
                or _closure_sha256() != closure_sha256
                or _corpus_binding(current) != expected_corpus
            ):
                raise Primock57IsolatedEvalError()
        except BaseException:
            post_validation_failed = True
        if runner_failed or post_validation_failed or raw_result is None:
            raise Primock57IsolatedEvalError()

        report = _validated_raw_report(
            raw_result,
            bundle=initial,
            corpus_binding=expected_corpus,
        )
        binding: dict[str, object] = {
            "kind": KIND,
            "schema_version": SCHEMA_VERSION,
            "fixture_id": primock57_fixture.FIXTURE_ID,
            "production_evidence": True,
            "aggregate_only": True,
            "validation": "before_and_after_runner",
            "corpus_manifest_sha256": initial.corpus.digest,
            "corpus_case_set_sha256": _sha256(expected_corpus.get("case_set_sha256")),
            "corpus_cases": len(initial.corpus.cases),
            "receipt_sha256": initial.receipt_sha256,
            "source_contract_sha256": initial.source_contract_sha256,
            "wrapper_closure_sha256": closure_sha256,
            "evaluator_report_sha256": _canonical_sha256(report),
            "privacy": {
                "local_paths": False,
                "per_case_rows": False,
                "verbatim_transcripts": False,
            },
        }
        binding["binding_sha256"] = _canonical_sha256(binding)
        report[BINDING_FIELD] = binding
        _assert_aggregate_private(
            report,
            forbidden_text=tuple(
                case.expected_text
                for case in initial.corpus.cases
                if case.expected_text
            ),
            forbidden_paths=tuple(
                text
                for case in initial.corpus.cases
                for text in (str(case.source_path), case.source_path.name)
            ),
        )
        _canonical_json(report)

        final = _validated_bundle(resolved_corpus)
        if (
            not _same_snapshot(initial, final)
            or _closure_sha256() != closure_sha256
            or _corpus_binding(final) != expected_corpus
        ):
            raise Primock57IsolatedEvalError()
        return report
    except Primock57IsolatedEvalError:
        raise
    except BaseException:
        raise Primock57IsolatedEvalError() from None


__all__ = [
    "BINDING_FIELD",
    "KIND",
    "Primock57IsolatedEvalError",
    "SCHEMA_VERSION",
    "run_primock57_isolated_eval",
]
