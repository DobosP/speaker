"""Receipt-bound aggregate evaluation for the PriMock57 overlap bundle.

The generic streaming evaluator has one reference per input.  This controller
keeps the existing worker/sandbox/resource boundary but privately presents each
PriMock case as role-A stem, role-B stem, then deterministic mix.  It publishes
only integer edit aggregates; hypotheses and references never cross the report
boundary.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
from typing import Final, Literal

from core.wer import normalize, word_error_rate
from tools import prepare_primock57_conversation_fixture as fixture
from tools import primock57_isolated_eval as privacy
from tools import streaming_stt_eval
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.corpus import CorpusCase, LoadedCorpus
from tools.streaming_stt.metrics import RunRecord, aggregate_metrics
from tools.streaming_stt.overlap_metrics import (
    PROTOCOL_VERSION as OVERLAP_METRIC,
    OverlapMetricResult,
    aggregate_two_utterance_min_order_wer,
    score_two_utterance_min_order_wer,
)
from tools.streaming_stt.protocol import (
    FinalEvent,
    NativeFinalEvent,
    ParakeetCppFinalEvent,
    PcmInput,
    TranscribeRequest,
)
from tools.streaming_stt.source_bundle import (
    WORKER_SOURCE_FILES,
    SourceBundle,
    stage_worker_source_bundle,
    verify_source_bundle,
)
from tools.streaming_stt.supervisor import StreamingWorker


SCHEMA_VERSION: Final = 1
KIND: Final = "primock57-overlap-stt-eval-v1"
LOCK_KIND: Final = "primock57-overlap-evaluator-lock-v1"
LOCK_RECIPE_SHA256: Final = (
    "0697956f0671fb77e101f40b67f4a46bfb5a6da9cd0962d2cde4eec4577f261e"
)
LOCK_FILE_SHA256: Final = (
    "b8fdc497548b8d51fd5381513427cfffb715f1b6757481410900e45c337d2748"
)
LOCK_FILE_BYTES: Final = 986
DEFAULT_LOCK: Final = (
    Path(__file__).resolve().parent
    / "streaming_stt"
    / "primock57-overlap-evaluator-v1.lock.json"
)
_MAX_LOCK_BYTES: Final = 16 * 1024
_MAX_SOURCE_BYTES: Final = 4 * 1024 * 1024
_MAX_REPORT_BYTES: Final = 1024 * 1024
_REPEATS: Final = 3
_INPUT_ORDER: Final = ("role_a", "role_b", "mix")
_SAFE_ERROR: Final = {
    "error": "primock57_overlap_eval_prerequisites_unavailable",
    "ok": False,
}
_REPORT_ROOT_FIELDS: Final = frozenset(
    {
        "binding_sha256",
        "bundle",
        "config",
        "evaluator",
        "evidence",
        "kind",
        "metrics",
        "ok",
        "schema_version",
        "worker",
        "worker_stderr",
    }
)

InputKind = Literal["role_a", "role_b", "mix"]


class Primock57OverlapEvalError(RuntimeError):
    """A detail-free admission, execution, privacy, or publication failure."""


class _LifecycleSignal(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__()
        self.signum = signum


@dataclass(frozen=True, slots=True)
class _EvaluatorLock:
    raw_sha256: str
    recipe_sha256: str
    manifest_sha256: str
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class _PrivateInput:
    case_index: int
    kind: InputKind
    path: Path = field(repr=False)
    sha256: str
    samples: int
    audio: bytes = field(repr=False)
    reference_a: str = field(repr=False)
    reference_b: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class _BundleSnapshot:
    bundle: fixture.LoadedPrimock57OverlapBundle
    directory_snapshot: tuple[int, ...] = field(repr=False)
    file_snapshots: tuple[object, ...] = field(repr=False)
    inputs: tuple[_PrivateInput, ...] = field(repr=False)


@dataclass(slots=True)
class _ReportCommitState:
    committed: bool = False
    digest: str = ""


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
        raise Primock57OverlapEvalError() from None
    return raw + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Primock57OverlapEvalError()
    return value


def _strict_json_bytes(raw: bytes, *, maximum_bytes: int) -> object:
    if not raw or len(raw) > maximum_bytes:
        raise Primock57OverlapEvalError()

    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise Primock57OverlapEvalError()
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                Primock57OverlapEvalError()
            ),
        )
    except Primock57OverlapEvalError:
        raise
    except (OverflowError, UnicodeError, ValueError):
        raise Primock57OverlapEvalError() from None


def _load_lock(path: Path | str = DEFAULT_LOCK) -> _EvaluatorLock:
    try:
        selected = Path(os.path.abspath(Path(path).expanduser()))
        if (
            selected != DEFAULT_LOCK
            or selected.resolve(strict=True) != selected
            or selected.name != DEFAULT_LOCK.name
        ):
            raise Primock57OverlapEvalError()
        snapshot = read_regular_bounded(
            selected,
            maximum_bytes=_MAX_LOCK_BYTES,
            expected_bytes=LOCK_FILE_BYTES,
        )
        raw = snapshot.data
        if hashlib.sha256(raw).hexdigest() != LOCK_FILE_SHA256:
            raise Primock57OverlapEvalError()
        value = _strict_json_bytes(raw, maximum_bytes=_MAX_LOCK_BYTES)
        if not isinstance(value, dict) or set(value) != {
            "bundle",
            "evidence_scope",
            "execution",
            "fixture_id",
            "kind",
            "recipe_digest_rule",
            "recipe_sha256",
            "schema_version",
        }:
            raise Primock57OverlapEvalError()
        bundle = value.get("bundle")
        execution = value.get("execution")
        scope = value.get("evidence_scope")
        if (
            value.get("schema_version") != 1
            or value.get("kind") != LOCK_KIND
            or value.get("fixture_id") != fixture.FIXTURE_ID
            or value.get("recipe_digest_rule")
            != "sha256-canonical-json-without-recipe_sha256-v1"
            or value.get("recipe_sha256") != LOCK_RECIPE_SHA256
            or not isinstance(bundle, dict)
            or set(bundle)
            != {
                "cases",
                "manifest_file",
                "manifest_sha256",
                "pcm_inputs",
                "receipt_file",
                "receipt_sha256",
            }
            or bundle.get("cases") != 3
            or bundle.get("pcm_inputs") != 9
            or bundle.get("manifest_file") != fixture.OVERLAP_MANIFEST_FILENAME
            or bundle.get("receipt_file") != fixture.PREPARATION_RECEIPT_FILENAME
            or not isinstance(execution, dict)
            or execution
            != {
                "input_order": list(_INPUT_ORDER),
                "metric": OVERLAP_METRIC,
                "repeats": _REPEATS,
            }
            or scope
            != {
                "diagnostic_only": True,
                "endpoint_evidence": False,
                "latency_evidence": False,
                "promotion_authority": False,
                "qualification_authority": False,
            }
        ):
            raise Primock57OverlapEvalError()
        digest_value = dict(value)
        digest_value.pop("recipe_sha256")
        if _canonical_sha256(digest_value) != LOCK_RECIPE_SHA256:
            raise Primock57OverlapEvalError()
        return _EvaluatorLock(
            raw_sha256=LOCK_FILE_SHA256,
            recipe_sha256=LOCK_RECIPE_SHA256,
            manifest_sha256=_sha256(bundle.get("manifest_sha256")),
            receipt_sha256=_sha256(bundle.get("receipt_sha256")),
        )
    except Primock57OverlapEvalError:
        raise
    except (BoundedReadError, OSError, RuntimeError, TypeError, ValueError):
        raise Primock57OverlapEvalError() from None


def _closure_sha256() -> str:
    try:
        repo_root = Path(__file__).resolve(strict=True).parents[1]
        selected = (
            ("tools/primock57_overlap_eval.py", Path(__file__)),
            (
                "tools/prepare_primock57_conversation_fixture.py",
                Path(fixture.__file__),
            ),
            ("tools/primock57_isolated_eval.py", Path(privacy.__file__)),
            ("tools/streaming_stt_eval.py", Path(streaming_stt_eval.__file__)),
            (
                "tools/streaming_stt/overlap_metrics.py",
                Path(score_two_utterance_min_order_wer.__code__.co_filename),
            ),
        )
        rows: list[dict[str, object]] = []
        for expected, candidate in selected:
            resolved = candidate.resolve(strict=True)
            if resolved.relative_to(repo_root).as_posix() != expected:
                raise Primock57OverlapEvalError()
            metadata = resolved.lstat()
            digest = hash_regular_bounded(
                resolved,
                maximum_bytes=_MAX_SOURCE_BYTES,
                expected_bytes=metadata.st_size,
            )
            rows.append(
                {
                    "name": expected,
                    "sha256": digest.sha256,
                    "size_bytes": digest.size_bytes,
                }
            )
        return _canonical_sha256(rows)
    except Primock57OverlapEvalError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        raise Primock57OverlapEvalError() from None


def _mapping(value: object, expected: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise Primock57OverlapEvalError()
    return value


def _snapshot_bundle(
    bundle_dir: Path | str,
    evaluator_lock: _EvaluatorLock,
) -> _BundleSnapshot:
    try:
        bundle = fixture.load_primock57_overlap_bundle(bundle_dir)
        if (
            bundle.production_evidence is not True
            or bundle.digest != evaluator_lock.manifest_sha256
            or bundle.receipt_sha256 != evaluator_lock.receipt_sha256
            or len(bundle.cases) != 3
            or bundle.path.name != fixture.OVERLAP_MANIFEST_FILENAME
        ):
            raise Primock57OverlapEvalError()
        root = bundle.path.parent
        with fixture._open_private_directory(root) as (
            opened_root,
            directory_fd,
            directory_snapshot,
        ):
            if opened_root != root:
                raise Primock57OverlapEvalError()
            receipt_raw, receipt_file = fixture._read_private_at(
                directory_fd,
                fixture.PREPARATION_RECEIPT_FILENAME,
                maximum_bytes=fixture._MAX_RECEIPT_BYTES,
            )
            manifest_raw, manifest_file = fixture._read_private_at(
                directory_fd,
                fixture.OVERLAP_MANIFEST_FILENAME,
                maximum_bytes=fixture._MAX_OVERLAP_MANIFEST_BYTES,
            )
            if (
                receipt_file.sha256 != bundle.receipt_sha256
                or manifest_file.sha256 != bundle.digest
                or hashlib.sha256(receipt_raw).hexdigest() != bundle.receipt_sha256
                or hashlib.sha256(manifest_raw).hexdigest() != bundle.digest
            ):
                raise Primock57OverlapEvalError()
            file_snapshots: list[object] = [receipt_file, manifest_file]
            inputs: list[_PrivateInput] = []
            expected_names = {
                fixture.PREPARATION_RECEIPT_FILENAME,
                fixture.OVERLAP_MANIFEST_FILENAME,
            }
            for case_index, raw_case in enumerate(bundle.cases):
                case = _mapping(
                    raw_case,
                    {"case_id", "envelope", "mix", "overlap", "role_a", "role_b"},
                )
                envelope = _mapping(
                    case.get("envelope"),
                    {"samples", "source_end_sample", "source_start_sample"},
                )
                samples = envelope.get("samples")
                if type(samples) is not int or samples <= 0:
                    raise Primock57OverlapEvalError()
                role_a = _mapping(
                    case.get("role_a"),
                    {
                        "activity_end_sample",
                        "activity_start_sample",
                        "bytes",
                        "file",
                        "interval_index",
                        "reference",
                        "reference_sha256",
                        "sha256",
                    },
                )
                role_b = _mapping(case.get("role_b"), set(role_a))
                mix = _mapping(
                    case.get("mix"), {"arithmetic", "bytes", "file", "sha256"}
                )
                references = (role_a.get("reference"), role_b.get("reference"))
                if any(
                    type(reference) is not str or not reference
                    for reference in references
                ):
                    raise Primock57OverlapEvalError()
                for kind, row in (("role_a", role_a), ("role_b", role_b), ("mix", mix)):
                    name = row.get("file")
                    digest = row.get("sha256")
                    declared_bytes = row.get("bytes")
                    if (
                        type(name) is not str
                        or type(declared_bytes) is not int
                        or declared_bytes != samples * 4
                    ):
                        raise Primock57OverlapEvalError()
                    raw, private_file = fixture._read_private_at(
                        directory_fd,
                        name,
                        maximum_bytes=fixture._MAX_WAV_BYTES,
                    )
                    if (
                        private_file.sha256 != _sha256(digest)
                        or private_file.size_bytes != declared_bytes
                    ):
                        raise Primock57OverlapEvalError()
                    expected_names.add(name)
                    file_snapshots.append(private_file)
                    inputs.append(
                        _PrivateInput(
                            case_index=case_index,
                            kind=kind,  # type: ignore[arg-type]
                            path=root / name,
                            sha256=private_file.sha256,
                            samples=samples,
                            audio=raw,
                            reference_a=str(references[0]),
                            reference_b=str(references[1]),
                        )
                    )
            if (
                tuple(item.kind for item in inputs) != _INPUT_ORDER * 3
                or len(inputs) != 9
                or set(os.listdir(directory_fd)) != expected_names
            ):
                raise Primock57OverlapEvalError()
            return _BundleSnapshot(
                bundle=bundle,
                directory_snapshot=directory_snapshot,
                file_snapshots=tuple(file_snapshots),
                inputs=tuple(inputs),
            )
    except Primock57OverlapEvalError:
        raise
    except Exception:
        raise Primock57OverlapEvalError() from None


def _same_bundle(left: _BundleSnapshot, right: _BundleSnapshot) -> bool:
    return left == right


def preflight_primock57_overlap_bundle(
    bundle_dir: Path | str,
) -> dict[str, object]:
    """Validate only the evaluator lock and exact retained production bundle."""

    evaluator_lock = _load_lock()
    closure = _closure_sha256()
    loaded = _snapshot_bundle(bundle_dir, evaluator_lock)
    current = _snapshot_bundle(bundle_dir, evaluator_lock)
    if not _same_bundle(loaded, current) or _closure_sha256() != closure:
        raise Primock57OverlapEvalError()
    return {
        "fixture_id": fixture.FIXTURE_ID,
        "cases": 3,
        "pcm_inputs": 9,
        "manifest_sha256": loaded.bundle.digest,
        "receipt_sha256": loaded.bundle.receipt_sha256,
        "source_contract_sha256": loaded.bundle.source_contract_sha256,
        "evaluator_lock_recipe_sha256": evaluator_lock.recipe_sha256,
        "wrapper_closure_sha256": closure,
    }


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _has_git_ancestor(path: Path) -> bool:
    try:
        return fixture._has_git_ancestor(path)
    except Exception:
        raise Primock57OverlapEvalError() from None


def _stable_existing_directory(value: Path | str) -> Path:
    try:
        candidate = Path(os.path.abspath(Path(value).expanduser()))
        if candidate.resolve(strict=True) != candidate:
            raise Primock57OverlapEvalError()
        with opened_directory_nofollow(candidate, require_private=True) as (
            stable,
            _fd,
        ):
            if stable != candidate:
                raise Primock57OverlapEvalError()
        return candidate
    except Primock57OverlapEvalError:
        raise
    except (BoundedReadError, OSError, RuntimeError, TypeError, ValueError):
        raise Primock57OverlapEvalError() from None


def _validate_paths(
    *,
    initial: _BundleSnapshot,
    manifest: object,
    scratch_parent: Path | str,
    output_path: Path | str | None,
) -> tuple[Path, Path | None]:
    scratch = _stable_existing_directory(scratch_parent)
    output: Path | None = None
    protected = {
        initial.bundle.path.parent,
        Path(manifest.path).resolve(strict=True).parent,
        Path(manifest.python.path).resolve(strict=True).parent,
        Path(manifest.worker.path).resolve(strict=True).parent,
        *(Path(item.path).resolve(strict=True).parent for item in manifest.artifacts),
    }
    if output_path is not None:
        candidate = Path(os.path.abspath(Path(output_path).expanduser()))
        if candidate.exists() or candidate.is_symlink() or not candidate.is_absolute():
            raise Primock57OverlapEvalError()
        output_parent = _stable_existing_directory(candidate.parent)
        output = output_parent / candidate.name
        if _has_git_ancestor(output_parent):
            raise Primock57OverlapEvalError()
        if _paths_overlap(scratch, output_parent):
            raise Primock57OverlapEvalError()
    if _has_git_ancestor(scratch):
        raise Primock57OverlapEvalError()
    for root in protected:
        if _paths_overlap(scratch, root) or (
            output is not None and _paths_overlap(output.parent, root)
        ):
            raise Primock57OverlapEvalError()
    return scratch, output


def _private_corpus(initial: _BundleSnapshot) -> LoadedCorpus:
    cases: list[CorpusCase] = []
    for index, item in enumerate(initial.inputs):
        expected = (
            item.reference_a
            if item.kind == "role_a"
            else item.reference_b
            if item.kind == "role_b"
            else f"{item.reference_a} {item.reference_b}"
        )
        cases.append(
            CorpusCase(
                case_id=f"primock57-overlap-input-{index:02d}",
                source_path=item.path,
                sha256=item.sha256,
                samples=item.samples,
                expected_text=expected,
                assertion="transcript",
                commands=(),
                forbidden_commands=(),
                tags=("primock57", "overlap"),
                audio_bytes=item.audio,
            )
        )
    return LoadedCorpus(
        path=initial.bundle.path,
        digest=initial.bundle.digest,
        schema_version=1,
        purpose="primock57-overlap-diagnostic",
        provenance=None,
        cases=tuple(cases),
        audio_bytes=sum(len(item.audio) for item in initial.inputs),
    )


def _scratch_binding(
    scratch: Path,
    *,
    expected_identity: tuple[int, int],
    source_bundle: SourceBundle,
    pcm_paths: Sequence[Path],
    initial: _BundleSnapshot,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Rebind the private execution tree without retaining its local path."""

    if len(pcm_paths) != len(initial.inputs):
        raise Primock57OverlapEvalError()
    try:
        with opened_directory_nofollow(scratch, require_private=True) as (
            stable,
            directory_fd,
        ):
            opened = os.fstat(directory_fd)
            lexical = scratch.lstat()
            opened_snapshot = fixture._file_snapshot(opened)
            if (
                stable != scratch
                or (opened.st_dev, opened.st_ino) != expected_identity
                or fixture._file_snapshot(lexical) != opened_snapshot
                or stat.S_IMODE(opened.st_mode) != 0o700
                or opened.st_uid != os.getuid()
            ):
                raise Primock57OverlapEvalError()
            names = tuple(sorted(os.listdir(directory_fd)))
        verify_source_bundle(source_bundle, required_files=WORKER_SOURCE_FILES)
        for path, item in zip(pcm_paths, initial.inputs, strict=True):
            metadata = path.lstat()
            if (
                path.parent != scratch
                or path.resolve(strict=True) != path
                or not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
                or metadata.st_uid != os.getuid()
                or metadata.st_size != len(item.audio)
            ):
                raise Primock57OverlapEvalError()
            digest = hash_regular_bounded(
                path,
                maximum_bytes=streaming_stt_eval.MAX_PCM_BYTES,
                expected_bytes=len(item.audio),
            )
            if digest.sha256 != item.sha256:
                raise Primock57OverlapEvalError()
        return opened_snapshot, names
    except Primock57OverlapEvalError:
        raise
    except Exception:
        raise Primock57OverlapEvalError() from None


def _source_complete(item: _PrivateInput, final: object) -> bool:
    if type(final) is NativeFinalEvent:
        return (
            final.source_samples == item.samples
            and final.source_samples_consumed == item.samples
        )
    if type(final) is ParakeetCppFinalEvent:
        return (
            final.source_samples_offered == item.samples
            and final.source_samples_consumed == item.samples
        )
    if type(final) is FinalEvent:
        return final.samples_seen >= item.samples and math.isclose(
            final.audio_seconds,
            item.samples / 16_000,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    return False


def _accuracy_dict(results: Sequence[object]) -> dict[str, object]:
    substitutions = sum(int(item.substitutions) for item in results)
    insertions = sum(int(item.insertions) for item in results)
    deletions = sum(int(item.deletions) for item in results)
    reference_words = sum(int(item.ref_words) for item in results)
    hypothesis_words = sum(int(item.hyp_words) for item in results)
    word_errors = substitutions + insertions + deletions
    wer = 0.0 if reference_words == 0 else word_errors / reference_words
    if not math.isfinite(wer):
        raise Primock57OverlapEvalError()
    return {
        "evaluations": len(results),
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "word_errors": word_errors,
        "reference_words": reference_words,
        "hypothesis_words": hypothesis_words,
        "wer": wer,
    }


def _overlap_metrics(
    initial: _BundleSnapshot,
    records: Sequence[RunRecord],
) -> dict[str, object]:
    if len(records) != len(initial.inputs) * _REPEATS:
        raise Primock57OverlapEvalError()
    stems: list[object] = []
    mixes: list[OverlapMetricResult] = []
    outputs: dict[int, set[tuple[str, ...]]] = {}
    for record in records:
        item = initial.inputs[record.case_index]
        final = record.trace.final
        if not _source_complete(item, final):
            raise Primock57OverlapEvalError()
        hypothesis = final.text
        outputs.setdefault(record.case_index, set()).add(tuple(normalize(hypothesis)))
        if item.kind == "mix":
            mixes.append(
                score_two_utterance_min_order_wer(
                    item.reference_a,
                    item.reference_b,
                    hypothesis,
                )
            )
        else:
            reference = item.reference_a if item.kind == "role_a" else item.reference_b
            stems.append(word_error_rate(reference, hypothesis))
    if len(stems) != 6 * _REPEATS or len(mixes) != 3 * _REPEATS:
        raise Primock57OverlapEvalError()
    stem_accuracy = _accuracy_dict(stems)
    raw_mix_accuracy = aggregate_two_utterance_min_order_wer(mixes).as_dict()
    if raw_mix_accuracy.pop("cases", None) != len(mixes):
        raise Primock57OverlapEvalError()
    mix_accuracy = {"evaluations": len(mixes), **raw_mix_accuracy}
    degradation = float(mix_accuracy["wer"]) - float(stem_accuracy["wer"])
    if not math.isfinite(degradation):
        raise Primock57OverlapEvalError()
    return {
        "coverage": {
            "cases": 3,
            "pcm_inputs": 9,
            "repeats": _REPEATS,
            "evaluations": len(records),
            "source_complete_evaluations": len(records),
        },
        "stem_accuracy": stem_accuracy,
        "mix_accuracy": mix_accuracy,
        "mix_minus_stem_wer": degradation,
        "repeat_disagreement_inputs": sum(len(value) > 1 for value in outputs.values()),
    }


def _adapter_runtime_matches(manifest: object, ready: object) -> bool:
    if manifest.adapter not in {
        streaming_stt_eval.FASTER_WHISPER_ENDPOINT_ADAPTER,
        streaming_stt_eval.KYUTAI_ADAPTER,
        streaming_stt_eval.NEMOTRON_ADAPTER,
        streaming_stt_eval.PARAKEET_REALTIME_EOU_ADAPTER,
    }:
        return True
    config = manifest.adapter_config
    return (
        isinstance(
            config,
            (
                streaming_stt_eval.FasterWhisperEndpointConfig,
                streaming_stt_eval.KyutaiConfig,
                streaming_stt_eval.NemotronConfig,
                streaming_stt_eval.ParakeetRealtimeEouConfig,
            ),
        )
        and ready.runtime.get("python") == config.python_version
    )


def _raise_lifecycle(signum: int) -> None:
    if signum == getattr(signal, "SIGINT", None):
        raise KeyboardInterrupt
    raise _LifecycleSignal(signum)


def _enter_worker_owned(worker: StreamingWorker) -> StreamingWorker:
    """Defer lifecycle delivery only until a newly spawned child is owned."""

    signal_numbers = _commit_signal_numbers()
    pending: list[int] = []
    previous_handlers: dict[int, object] = {}
    original_factory = getattr(worker, "_popen_factory", None)
    factory_replaced = False
    restored = False

    def handler(signum: int, _frame: object) -> None:
        if not pending:
            pending.append(signum)

    def restore() -> BaseException | None:
        nonlocal restored
        if restored:
            return None
        restore_error: BaseException | None = None
        previous_mask: set[signal.Signals] | None = None
        try:
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, signal_numbers)
        except BaseException as error:
            restore_error = error
        if factory_replaced:
            try:
                setattr(worker, "_popen_factory", original_factory)
            except BaseException as error:
                restore_error = error
        for signum, previous in reversed(tuple(previous_handlers.items())):
            try:
                signal.signal(signum, previous)
            except BaseException as error:
                restore_error = error
        restored = True
        if previous_mask is not None:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            except BaseException as error:
                restore_error = error
        return restore_error

    def owned_factory(*args: object, **kwargs: object) -> object:
        if pending:
            _raise_lifecycle(pending[0])
        process = original_factory(*args, **kwargs)
        setattr(worker, "_process", process)
        restore_error = restore()
        if pending:
            _raise_lifecycle(pending[0])
        if restore_error is not None:
            raise restore_error
        return process

    enter_error: BaseException | None = None
    entered: object | None = None
    try:
        for signum in signal_numbers:
            previous_handlers[signum] = signal.signal(signum, handler)
        if callable(original_factory):
            setattr(worker, "_popen_factory", owned_factory)
            factory_replaced = True
        entered = worker.__enter__()
    except BaseException as error:
        enter_error = error
    finally:
        restore_error = restore()
        if restore_error is not None and (
            enter_error is None
            or isinstance(restore_error, (KeyboardInterrupt, _LifecycleSignal))
        ):
            enter_error = restore_error
    if pending:
        _raise_lifecycle(pending[0])
    if enter_error is not None:
        raise enter_error
    if entered is not worker:
        raise Primock57OverlapEvalError()
    return worker


def _run_locked(
    worker_manifest_path: Path | str,
    bundle_dir: Path | str,
    *,
    scratch_parent: Path | str,
    output_path: Path | str | None,
    run_guard: Callable[[], None],
    commit_state: _ReportCommitState,
) -> dict[str, object]:
    evaluator_lock = _load_lock()
    closure = _closure_sha256()
    initial = _snapshot_bundle(bundle_dir, evaluator_lock)
    manifest = streaming_stt_eval.load_worker_manifest(worker_manifest_path)
    scratch_parent_path, output = _validate_paths(
        initial=initial,
        manifest=manifest,
        scratch_parent=scratch_parent,
        output_path=output_path,
    )
    selected_stream = streaming_stt_eval._selected_stream(
        manifest,
        None,
        chunk_samples=None,
        pace=None,
        partial_interval_ms=None,
        tail_padding_samples=None,
    )
    try:
        if (
            manifest.worker.path.resolve(strict=True)
            != streaming_stt_eval._FIXED_WORKER
        ):
            raise Primock57OverlapEvalError()
    except (OSError, RuntimeError):
        raise Primock57OverlapEvalError() from None
    runtime_receipt = streaming_stt_eval._verified_runtime_receipt_digest(manifest)
    model_receipt = streaming_stt_eval._verified_model_receipt_digest(manifest)
    evaluator_binding = streaming_stt_eval._evaluator_binding(manifest.adapter)
    scratch: Path | None = None
    scratch_identity: tuple[int, int] | None = None
    source_bundle: SourceBundle | None = None
    worker: StreamingWorker | None = None
    worker_context_exited = False
    records: list[RunRecord] = []
    pcm_paths: list[Path] = []
    report: dict[str, object] | None = None
    final_guard: Callable[[], bool] | None = None
    execution_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        previous_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK,
            _commit_signal_numbers(),
        )
        try:
            scratch = streaming_stt_eval._prepare_scratch(scratch_parent_path)
            scratch_meta = scratch.lstat()
            scratch_identity = (scratch_meta.st_dev, scratch_meta.st_ino)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        if scratch is None or scratch_identity is None:
            raise Primock57OverlapEvalError()
        source_bundle = stage_worker_source_bundle(
            streaming_stt_eval._REPO_ROOT,
            scratch / "source-bundle",
        )
        worker_binding = streaming_stt_eval._worker_binding(
            manifest,
            source_bundle,
            runtime_receipt_sha256=runtime_receipt,
            model_receipt_sha256=model_receipt,
        )
        for index, item in enumerate(initial.inputs):
            path = scratch / f"input-{index:02d}.f32le"
            streaming_stt_eval._write_pcm_snapshot(path, item.audio)
            digest = hash_regular_bounded(
                path,
                maximum_bytes=streaming_stt_eval.MAX_PCM_BYTES,
                expected_bytes=len(item.audio),
            )
            if digest.sha256 != item.sha256:
                raise Primock57OverlapEvalError()
            pcm_paths.append(path)

        worker = StreamingWorker(manifest, scratch, source_bundle)
        enter_error: BaseException | None = None
        try:
            _enter_worker_owned(worker)
        except BaseException as error:
            enter_error = error
        if enter_error is not None:
            raise enter_error

        session_error: BaseException | None = None
        try:
            if worker.ready is None:
                raise Primock57OverlapEvalError()
            for repeat in range(_REPEATS):
                for index, (item, pcm_path) in enumerate(
                    zip(initial.inputs, pcm_paths, strict=True)
                ):
                    request = TranscribeRequest(
                        request_id=f"overlap-{index:02d}-r{repeat}",
                        pcm=PcmInput(
                            path=pcm_path.resolve(strict=True),
                            sha256=item.sha256,
                            samples=item.samples,
                        ),
                        stream=selected_stream,
                        protocol_version=streaming_stt_eval._wire_protocol_version(
                            manifest
                        ),
                    )
                    records.append(
                        RunRecord(
                            case_index=index,
                            repeat=repeat,
                            trace=worker.transcribe(request),
                        )
                    )
        except BaseException as error:
            session_error = error

        exit_error: BaseException | None = None
        previous_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK,
            _commit_signal_numbers(),
        )
        try:
            worker.__exit__(
                None if session_error is None else type(session_error),
                session_error,
                None if session_error is None else session_error.__traceback__,
            )
            worker_context_exited = True
        except BaseException as error:
            exit_error = error
        finally:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            except BaseException as error:
                if exit_error is None or isinstance(
                    error,
                    (KeyboardInterrupt, _LifecycleSignal),
                ):
                    exit_error = error
        if isinstance(exit_error, (KeyboardInterrupt, _LifecycleSignal)):
            raise exit_error
        if session_error is not None:
            raise session_error
        if exit_error is not None:
            raise exit_error
        ready = worker.ready
        if (
            ready is None
            or not worker.fully_stopped
            or not _adapter_runtime_matches(manifest, ready)
        ):
            raise Primock57OverlapEvalError()
        post = _snapshot_bundle(bundle_dir, evaluator_lock)
        post_scratch = _scratch_binding(
            scratch,
            expected_identity=scratch_identity,
            source_bundle=source_bundle,
            pcm_paths=pcm_paths,
            initial=initial,
        )
        current_manifest = streaming_stt_eval.load_worker_manifest(manifest.path)
        current_runtime = streaming_stt_eval._verified_runtime_receipt_digest(
            current_manifest
        )
        current_model = streaming_stt_eval._verified_model_receipt_digest(
            current_manifest
        )
        current_worker_binding = streaming_stt_eval._worker_binding(
            current_manifest,
            source_bundle,
            runtime_receipt_sha256=current_runtime,
            model_receipt_sha256=current_model,
        )
        if (
            not _same_bundle(initial, post)
            or _load_lock() != evaluator_lock
            or _closure_sha256() != closure
            or current_manifest.digest != manifest.digest
            or current_runtime != runtime_receipt
            or current_model != model_receipt
            or current_worker_binding != worker_binding
            or streaming_stt_eval._evaluator_binding(current_manifest.adapter)
            != evaluator_binding
        ):
            raise Primock57OverlapEvalError()
        private_corpus = _private_corpus(initial)
        generic = aggregate_metrics(
            private_corpus,
            records,
            ready,
            repeats=_REPEATS,
        )
        if generic.get("coverage_complete") is not True:
            raise Primock57OverlapEvalError()
        metrics = _overlap_metrics(initial, records)
        worker_report = streaming_stt_eval._worker_report(
            manifest,
            worker_binding,
            ready.runtime,
            worker.cgroup_evidence,
            worker.resource_observations,
            expected_completed_cases=len(records),
        )
        config: dict[str, object] = {
            **selected_stream.as_dict(),
            "repeats": _REPEATS,
            "input_order_sha256": _canonical_sha256(list(_INPUT_ORDER)),
            "contract_sha256": "",
        }
        config["contract_sha256"] = _canonical_sha256(
            {key: value for key, value in config.items() if key != "contract_sha256"}
        )
        model_executed = evaluator_binding.get("model_executed")
        if type(model_executed) is not bool:
            raise Primock57OverlapEvalError()
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "ok": True,
            "evidence": {
                "diagnostic_only": True,
                "endpoint_evidence": False,
                "latency_evidence": False,
                "model_executed": model_executed,
                "promotion_authority": False,
                "qualification_authority": False,
            },
            "bundle": {
                "fixture_id": fixture.FIXTURE_ID,
                "manifest_sha256": initial.bundle.digest,
                "receipt_sha256": initial.bundle.receipt_sha256,
                "source_contract_sha256": initial.bundle.source_contract_sha256,
                "evaluator_lock_file_sha256": evaluator_lock.raw_sha256,
                "evaluator_lock_recipe_sha256": evaluator_lock.recipe_sha256,
                "cases": 3,
                "pcm_inputs": 9,
            },
            "config": config,
            "metrics": metrics,
            "worker": worker_report,
            "worker_stderr": dict(worker.stderr_summary),
            "evaluator": {
                "generic": evaluator_binding,
                "wrapper_closure_sha256": closure,
            },
            "binding_sha256": "",
        }
        report["binding_sha256"] = _canonical_sha256(
            {key: value for key, value in report.items() if key != "binding_sha256"}
        )
        _validate_report(
            report,
            initial=initial,
            expected_worker=worker_report,
            expected_evaluator=report["evaluator"],
            expected_config=config,
        )

        def commit_guard() -> bool:
            try:
                run_guard()
                final = _snapshot_bundle(bundle_dir, evaluator_lock)
                final_manifest = streaming_stt_eval.load_worker_manifest(manifest.path)
                return (
                    worker is not None
                    and worker.fully_stopped
                    and _same_bundle(initial, final)
                    and _load_lock() == evaluator_lock
                    and _closure_sha256() == closure
                    and final_manifest.digest == manifest.digest
                    and streaming_stt_eval._verified_runtime_receipt_digest(
                        final_manifest
                    )
                    == runtime_receipt
                    and streaming_stt_eval._verified_model_receipt_digest(
                        final_manifest
                    )
                    == model_receipt
                    and streaming_stt_eval._worker_binding(
                        final_manifest,
                        source_bundle,
                        runtime_receipt_sha256=runtime_receipt,
                        model_receipt_sha256=model_receipt,
                    )
                    == worker_binding
                    and streaming_stt_eval._evaluator_binding(final_manifest.adapter)
                    == evaluator_binding
                )
            except (KeyboardInterrupt, _LifecycleSignal):
                raise
            except Exception:
                return False

        final_guard = commit_guard
        if (
            commit_guard() is not True
            or _scratch_binding(
                scratch,
                expected_identity=scratch_identity,
                source_bundle=source_bundle,
                pcm_paths=pcm_paths,
                initial=initial,
            )
            != post_scratch
        ):
            raise Primock57OverlapEvalError()
    except BaseException as error:
        execution_error = error

    if worker is not None and (not worker_context_exited or not worker.fully_stopped):
        stop_error: BaseException | None = None
        try:
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK,
                _commit_signal_numbers(),
            )
        except BaseException as error:
            stop_error = error
        else:
            try:
                worker.close()
            except BaseException as error:
                stop_error = error
            finally:
                try:
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                except BaseException as error:
                    stop_error = error
        if stop_error is not None and (
            execution_error is None
            or isinstance(stop_error, (KeyboardInterrupt, _LifecycleSignal))
        ):
            execution_error = stop_error

    if worker is not None and not worker.fully_stopped:
        cleanup_error = Primock57OverlapEvalError()
    elif scratch is not None:
        if scratch_identity is None:
            try:
                metadata = scratch.lstat()
                if (
                    not stat.S_ISDIR(metadata.st_mode)
                    or stat.S_IMODE(metadata.st_mode) != 0o700
                    or metadata.st_uid != os.getuid()
                ):
                    raise Primock57OverlapEvalError()
                scratch_identity = (metadata.st_dev, metadata.st_ino)
            except BaseException as error:
                cleanup_error = error
        if scratch_identity is not None and cleanup_error is None:
            try:
                streaming_stt_eval._remove_private_scratch(
                    scratch,
                    expected_identity=scratch_identity,
                )
            except BaseException as error:
                cleanup_error = error

    if execution_error is not None:
        raise execution_error
    if cleanup_error is not None:
        if isinstance(cleanup_error, (KeyboardInterrupt, _LifecycleSignal)):
            raise cleanup_error
        raise Primock57OverlapEvalError() from None
    if report is None or final_guard is None:
        raise Primock57OverlapEvalError()
    if output is None:
        if final_guard() is not True:
            raise Primock57OverlapEvalError()
        return report
    try:
        _publish_report(
            output,
            report,
            commit_guard=final_guard,
            state=commit_state,
        )
        return report
    except BaseException:
        if commit_state.committed:
            return report
        raise


def _validate_accuracy(
    value: object,
    *,
    evaluations: int,
    protocol: str | None = None,
) -> None:
    fields = {
        "deletions",
        "evaluations",
        "hypothesis_words",
        "insertions",
        "reference_words",
        "substitutions",
        "wer",
        "word_errors",
    }
    if protocol is not None:
        fields.add("protocol")
    checked = _mapping(value, fields)
    counts = tuple(
        checked.get(field)
        for field in (
            "substitutions",
            "insertions",
            "deletions",
            "word_errors",
            "reference_words",
            "hypothesis_words",
        )
    )
    if any(type(item) is not int or item < 0 for item in counts):
        raise Primock57OverlapEvalError()
    substitutions, insertions, deletions, errors, reference, hypothesis = counts
    wer = checked.get("wer")
    if (
        checked.get("evaluations") != evaluations
        or (protocol is not None and checked.get("protocol") != protocol)
        or errors != substitutions + insertions + deletions
        or reference <= 0
        or deletions > reference
        or substitutions > reference - deletions
        or insertions > hypothesis
        or substitutions > hypothesis - insertions
        or hypothesis != reference - deletions + insertions
        or type(wer) is not float
        or not math.isfinite(wer)
        or wer != errors / reference
    ):
        raise Primock57OverlapEvalError()


def _validate_report(
    value: object,
    *,
    initial: _BundleSnapshot,
    expected_worker: Mapping[str, object] | None = None,
    expected_evaluator: Mapping[str, object] | None = None,
    expected_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _REPORT_ROOT_FIELDS
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != KIND
        or value.get("ok") is not True
        or not isinstance(value.get("metrics"), Mapping)
        or not isinstance(value.get("bundle"), Mapping)
        or not isinstance(value.get("worker"), Mapping)
        or not isinstance(value.get("evaluator"), Mapping)
        or not isinstance(value.get("evidence"), Mapping)
        or not isinstance(value.get("config"), Mapping)
        or not isinstance(value.get("worker_stderr"), Mapping)
    ):
        raise Primock57OverlapEvalError()
    expected_binding = _canonical_sha256(
        {key: item for key, item in value.items() if key != "binding_sha256"}
    )
    if value.get("binding_sha256") != expected_binding:
        raise Primock57OverlapEvalError()
    bundle = _mapping(
        value.get("bundle"),
        {
            "cases",
            "evaluator_lock_file_sha256",
            "evaluator_lock_recipe_sha256",
            "fixture_id",
            "manifest_sha256",
            "pcm_inputs",
            "receipt_sha256",
            "source_contract_sha256",
        },
    )
    if (
        bundle.get("fixture_id") != fixture.FIXTURE_ID
        or bundle.get("cases") != 3
        or bundle.get("pcm_inputs") != 9
        or bundle.get("manifest_sha256") != initial.bundle.digest
        or bundle.get("receipt_sha256") != initial.bundle.receipt_sha256
        or bundle.get("source_contract_sha256") != initial.bundle.source_contract_sha256
        or any(
            _sha256(bundle.get(field)) != bundle.get(field)
            for field in (
                "evaluator_lock_file_sha256",
                "evaluator_lock_recipe_sha256",
                "manifest_sha256",
                "receipt_sha256",
                "source_contract_sha256",
            )
        )
    ):
        raise Primock57OverlapEvalError()
    evidence = _mapping(
        value.get("evidence"),
        {
            "diagnostic_only",
            "endpoint_evidence",
            "latency_evidence",
            "model_executed",
            "promotion_authority",
            "qualification_authority",
        },
    )
    if (
        evidence.get("diagnostic_only") is not True
        or evidence.get("endpoint_evidence") is not False
        or evidence.get("latency_evidence") is not False
        or type(evidence.get("model_executed")) is not bool
        or evidence.get("promotion_authority") is not False
        or evidence.get("qualification_authority") is not False
    ):
        raise Primock57OverlapEvalError()
    config = _mapping(
        value.get("config"),
        {
            "chunk_samples",
            "contract_sha256",
            "input_order_sha256",
            "pace",
            "partial_interval_ms",
            "repeats",
            "tail_padding_samples",
        },
    )
    if (
        type(config.get("chunk_samples")) is not int
        or config["chunk_samples"] <= 0
        or config.get("pace") not in {"burst", "realtime"}
        or type(config.get("partial_interval_ms")) is not int
        or config["partial_interval_ms"] <= 0
        or type(config.get("tail_padding_samples")) is not int
        or config["tail_padding_samples"] < 0
        or config.get("repeats") != _REPEATS
        or config.get("input_order_sha256") != _canonical_sha256(list(_INPUT_ORDER))
        or config.get("contract_sha256")
        != _canonical_sha256(
            {key: item for key, item in config.items() if key != "contract_sha256"}
        )
        or (expected_config is not None and config != expected_config)
    ):
        raise Primock57OverlapEvalError()
    metrics = _mapping(
        value.get("metrics"),
        {
            "coverage",
            "mix_accuracy",
            "mix_minus_stem_wer",
            "repeat_disagreement_inputs",
            "stem_accuracy",
        },
    )
    coverage = _mapping(
        metrics.get("coverage"),
        {
            "cases",
            "evaluations",
            "pcm_inputs",
            "repeats",
            "source_complete_evaluations",
        },
    )
    if coverage != {
        "cases": 3,
        "pcm_inputs": 9,
        "repeats": _REPEATS,
        "evaluations": 27,
        "source_complete_evaluations": 27,
    }:
        raise Primock57OverlapEvalError()
    stem_accuracy = metrics.get("stem_accuracy")
    mix_accuracy = metrics.get("mix_accuracy")
    _validate_accuracy(stem_accuracy, evaluations=18)
    _validate_accuracy(
        mix_accuracy,
        evaluations=9,
        protocol=OVERLAP_METRIC,
    )
    if not isinstance(stem_accuracy, Mapping) or not isinstance(
        mix_accuracy,
        Mapping,
    ):
        raise Primock57OverlapEvalError()
    expected_reference_words = _REPEATS * sum(
        len(normalize(item.reference_a)) + len(normalize(item.reference_b))
        for item in initial.inputs
        if item.kind == "mix"
    )
    if (
        expected_reference_words <= 0
        or stem_accuracy.get("reference_words") != expected_reference_words
        or mix_accuracy.get("reference_words") != expected_reference_words
        or stem_accuracy.get("reference_words") != mix_accuracy.get("reference_words")
        or type(metrics.get("mix_minus_stem_wer")) is not float
        or not math.isfinite(metrics["mix_minus_stem_wer"])
        or metrics["mix_minus_stem_wer"]
        != float(mix_accuracy["wer"]) - float(stem_accuracy["wer"])
        or type(metrics.get("repeat_disagreement_inputs")) is not int
        or not 0 <= metrics["repeat_disagreement_inputs"] <= 9
    ):
        raise Primock57OverlapEvalError()
    evaluator = _mapping(
        value.get("evaluator"),
        {"generic", "wrapper_closure_sha256"},
    )
    privacy._validated_evaluator(evaluator.get("generic"))
    generic = evaluator.get("generic")
    if (
        not isinstance(generic, Mapping)
        or evidence.get("model_executed") != generic.get("model_executed")
        or _sha256(evaluator.get("wrapper_closure_sha256"))
        != evaluator.get("wrapper_closure_sha256")
        or (expected_evaluator is not None and evaluator != expected_evaluator)
        or (expected_worker is not None and value.get("worker") != expected_worker)
    ):
        raise Primock57OverlapEvalError()
    references = tuple(
        reference
        for item in initial.inputs
        for reference in (item.reference_a, item.reference_b)
    )
    forbidden_paths = tuple(
        text for item in initial.inputs for text in (str(item.path), item.path.name)
    )
    forbidden_labels = {
        "role_a",
        "role_b",
        "role-a",
        "role-b",
        *(str(case.get("case_id")) for case in initial.bundle.cases),
    }
    try:
        privacy._validated_worker_stderr(value.get("worker_stderr"))
        privacy._assert_aggregate_private(
            value,
            forbidden_text=references,
            forbidden_paths=forbidden_paths,
        )
        encoded = _canonical_json(value)
        lowered = encoded.decode("ascii").casefold()
        if len(encoded) > _MAX_REPORT_BYTES or any(
            label and label.casefold() in lowered for label in forbidden_labels
        ):
            raise Primock57OverlapEvalError()
    except Primock57OverlapEvalError:
        raise
    except Exception:
        raise Primock57OverlapEvalError() from None
    return dict(value)


def _signal_numbers() -> tuple[int, ...]:
    return tuple(
        signum
        for signum in (
            getattr(signal, "SIGHUP", None),
            getattr(signal, "SIGTERM", None),
        )
        if isinstance(signum, int)
    )


def _commit_signal_numbers() -> tuple[int, ...]:
    return tuple(
        signum
        for signum in dict.fromkeys(
            (getattr(signal, "SIGINT", None), *_signal_numbers())
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
    try:
        if state.committed or state.digest:
            raise Primock57OverlapEvalError()
        parent = _stable_existing_directory(path.parent)
        if path.parent != parent or path.exists() or path.is_symlink():
            raise Primock57OverlapEvalError()
        encoded = _canonical_json(report, newline=True)
        digest = hashlib.sha256(encoded).hexdigest()
        with opened_directory_nofollow(parent, require_private=True) as (
            stable_parent,
            directory_fd,
        ):
            if stable_parent != parent:
                raise Primock57OverlapEvalError()
            parent_before = fixture._file_snapshot(os.fstat(directory_fd))
            try:
                os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise Primock57OverlapEvalError()
            temporary_flag = getattr(os, "O_TMPFILE", 0)
            if not temporary_flag:
                raise Primock57OverlapEvalError()
            descriptor = os.open(
                ".",
                os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | temporary_flag,
                0o600,
                dir_fd=directory_fd,
            )
            os.fchmod(descriptor, 0o600)
            created = os.fstat(descriptor)
            if (
                not stat.S_ISREG(created.st_mode)
                or created.st_nlink != 0
                or created.st_uid != os.getuid()
            ):
                raise Primock57OverlapEvalError()
            written = 0
            view = memoryview(encoded)
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if type(count) is not int or count <= 0:
                    raise Primock57OverlapEvalError()
                written += count
            os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            observed = bytearray()
            while len(observed) <= len(encoded):
                chunk = os.read(descriptor, len(encoded) + 1 - len(observed))
                if not chunk:
                    break
                observed.extend(chunk)
            before_link = os.fstat(descriptor)
            if (
                bytes(observed) != encoded
                or hashlib.sha256(observed).hexdigest() != digest
                or before_link.st_nlink != 0
                or before_link.st_size != len(encoded)
                or stat.S_IMODE(before_link.st_mode) != 0o600
            ):
                raise Primock57OverlapEvalError()
            if commit_guard() is not True:
                raise Primock57OverlapEvalError()
            os.lseek(descriptor, 0, os.SEEK_SET)
            final_observed = bytearray()
            while len(final_observed) <= len(encoded):
                chunk = os.read(
                    descriptor,
                    len(encoded) + 1 - len(final_observed),
                )
                if not chunk:
                    break
                final_observed.extend(chunk)
            current = os.fstat(descriptor)
            parent_path = parent.lstat()
            if (
                bytes(final_observed) != encoded
                or hashlib.sha256(final_observed).hexdigest() != digest
                or not stat.S_ISREG(current.st_mode)
                or stat.S_IMODE(current.st_mode) != 0o600
                or current.st_nlink != 0
                or current.st_uid != os.getuid()
                or current.st_size != len(encoded)
                or fixture._file_snapshot(os.fstat(directory_fd)) != parent_before
                or fixture._file_snapshot(parent_path) != parent_before
                or parent.resolve(strict=True) != parent
            ):
                raise Primock57OverlapEvalError()
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK,
                _commit_signal_numbers(),
            )
            link_returned = False
            try:
                try:
                    os.link(
                        f"/proc/self/fd/{descriptor}",
                        path.name,
                        dst_dir_fd=directory_fd,
                        follow_symlinks=True,
                    )
                    link_returned = True
                finally:
                    if link_returned:
                        state.committed = True
                        state.digest = digest
                    else:
                        try:
                            published = os.stat(
                                path.name,
                                dir_fd=directory_fd,
                                follow_symlinks=False,
                            )
                            current = os.fstat(descriptor)
                        except OSError:
                            pass
                        else:
                            if (
                                (current.st_dev, current.st_ino)
                                == (published.st_dev, published.st_ino)
                                and stat.S_ISREG(current.st_mode)
                                and stat.S_ISREG(published.st_mode)
                                and current.st_size == len(encoded)
                                and published.st_size == len(encoded)
                                and stat.S_IMODE(current.st_mode) == 0o600
                                and stat.S_IMODE(published.st_mode) == 0o600
                                and current.st_uid == os.getuid()
                                and published.st_uid == os.getuid()
                                and current.st_nlink == 1
                                and published.st_nlink == 1
                            ):
                                state.committed = True
                                state.digest = digest
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            if not state.committed:
                raise Primock57OverlapEvalError()
            try:
                os.fsync(directory_fd)
            except BaseException:
                pass
        return digest
    except BaseException as error:
        if state.committed:
            return state.digest
        if isinstance(error, (KeyboardInterrupt, _LifecycleSignal)):
            raise
        raise Primock57OverlapEvalError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def run_primock57_overlap_eval(
    worker_manifest_path: Path | str,
    bundle_dir: Path | str,
    *,
    scratch_parent: Path | str,
    output_path: Path | str | None = None,
    _commit_state: _ReportCommitState | None = None,
) -> dict[str, object]:
    """Run the fixed three-repeat diagnostic and optionally publish it."""

    state = _commit_state or _ReportCommitState()
    if state.committed or state.digest:
        raise Primock57OverlapEvalError()
    run_lock: object | None = None
    report: dict[str, object] | None = None
    pending: BaseException | None = None
    try:
        run_lock = streaming_stt_eval._BenchmarkRunLock.acquire(
            streaming_stt_eval._HOST_LOCK_PATH
        )
        run_lock.assert_bound()
        report = _run_locked(
            worker_manifest_path,
            bundle_dir,
            scratch_parent=scratch_parent,
            output_path=output_path,
            run_guard=run_lock.assert_bound,
            commit_state=state,
        )
    except BaseException as error:
        pending = error
    finally:
        if run_lock is not None:
            try:
                run_lock.close()
            except BaseException as error:
                if not state.committed and pending is None:
                    pending = error
    if pending is not None:
        if state.committed and report is not None:
            return report
        if isinstance(pending, (KeyboardInterrupt, _LifecycleSignal)):
            raise pending
        if isinstance(pending, Primock57OverlapEvalError):
            raise pending
        raise Primock57OverlapEvalError() from None
    if report is None:
        raise Primock57OverlapEvalError()
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed receipt-bound PriMock57 overlap diagnostic."
    )
    parser.add_argument("--worker-manifest", type=Path, required=True)
    parser.add_argument("--overlap-bundle", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _lifecycle_handler(signum: int, _frame: object) -> None:
    raise _LifecycleSignal(signum)


def main(argv: Sequence[str] | None = None) -> int:
    state = _ReportCommitState()
    previous: dict[int, object] = {}
    try:
        for signum in _signal_numbers():
            previous[signum] = signal.signal(signum, _lifecycle_handler)
        args = _parser().parse_args(argv)
        report = run_primock57_overlap_eval(
            args.worker_manifest,
            args.overlap_bundle,
            scratch_parent=args.scratch_root,
            output_path=args.output,
            _commit_state=state,
        )
        summary = _canonical_json(
            {
                "kind": KIND,
                "ok": True,
                "report_sha256": state.digest,
                "evaluations": report["metrics"]["coverage"]["evaluations"],
            },
            newline=True,
        )
        try:
            os.write(1, summary)
        except BaseException:
            pass
        return 0
    except KeyboardInterrupt:
        result = 0 if state.committed else 130
    except _LifecycleSignal as interrupted:
        result = 0 if state.committed else 128 + interrupted.signum
    except BaseException:
        result = 0 if state.committed else 2
    finally:
        for signum, handler in reversed(tuple(previous.items())):
            try:
                signal.signal(signum, handler)
            except BaseException:
                pass
    if not state.committed:
        try:
            os.write(1, _canonical_json(_SAFE_ERROR, newline=True))
        except BaseException:
            pass
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_LOCK",
    "KIND",
    "LOCK_RECIPE_SHA256",
    "Primock57OverlapEvalError",
    "SCHEMA_VERSION",
    "main",
    "preflight_primock57_overlap_bundle",
    "run_primock57_overlap_eval",
]
