"""Receipt-bound aggregate evaluation for retained NOTSOFAR-1 overlap audio.

The fixed controller sends every retained device channel as one complete-source
input to the existing isolated streaming worker.  It scores each private
hypothesis against both private utterance references and publishes only pooled
integer edit counts, anonymous device-channel aggregates, and bounded paired
agreement/difference counters.  It is deliberately not a diarization or
ordinary-WER evaluator.
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
from pathlib import PurePosixPath
import signal
import stat
from typing import Final, Literal

from core.wer import normalize
from tools import prepare_notsofar1_natural_overlap_fixture as fixture
from tools import primock57_isolated_eval as privacy
from tools import streaming_stt_eval
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.metrics import RunRecord
from tools.streaming_stt.overlap_metrics import (
    PROTOCOL_VERSION as OVERLAP_METRIC,
    OverlapMetricResult,
    aggregate_two_utterance_min_order_wer,
    score_two_utterance_min_order_wer,
)
from tools.streaming_stt.protocol import (
    CaseTrace,
    FinalEvent,
    NativeFinalEvent,
    ParakeetCppFinalEvent,
    PartialEvent,
    PcmInput,
    TranscribeRequest,
)
from tools.streaming_stt.source_bundle import (
    BoundSource,
    SOURCE_BUNDLE_SCHEMA_VERSION,
    WORKER_RELATIVE_PATH,
    WORKER_SOURCE_FILES,
    SourceBundle,
    stage_worker_source_bundle,
    verify_source_bundle,
)
from tools.streaming_stt import source_bundle as source_bundle_support
from tools.streaming_stt.supervisor import StreamingWorker


SCHEMA_VERSION: Final = 1
KIND: Final = "notsofar1-natural-overlap-stt-eval-v1"
LOCK_KIND: Final = "notsofar1-natural-overlap-evaluator-lock-v1"
LOCK_RECIPE_SHA256: Final = (
    "e65f70c42e2568f914ffb4cc4e29c37fcba6fdf9ff291c89aebb0471ffa66ec1"
)
LOCK_FILE_SHA256: Final = (
    "1e036ba048eedcdbfb2196083c4268c478c34af560cff8e0a10a54ec15076c62"
)
LOCK_FILE_BYTES: Final = 1620
FIXTURE_MANIFEST_SHA256: Final = (
    "898e291c713154e3cbd78ba9c38fca6fed08284edace98ac615ebdd37265da7b"
)
FIXTURE_RECEIPT_SHA256: Final = (
    "29b5d0f1f11f2ac56dee92a7d5ab6ef02a18c4d695a0b9b981b67dde3a24e2ba"
)
DEFAULT_LOCK: Final = (
    Path(__file__).resolve().parent
    / "streaming_stt"
    / "notsofar1-natural-overlap-evaluator-v1.lock.json"
)
_MAX_LOCK_BYTES: Final = 32 * 1024
_MAX_SOURCE_BYTES: Final = 4 * 1024 * 1024
_MAX_REPORT_BYTES: Final = 1024 * 1024
_REPEATS: Final = 1
_WINDOWS: Final = 7
_CHANNELS: Final = ("channel-a", "channel-b")
_INPUT_ORDER: Final = "window-index-ascending/channel-a-then-channel-b-v1"
_SAFE_ERROR: Final = {
    "error": "notsofar1_natural_overlap_eval_prerequisites_unavailable",
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

Channel = Literal["channel-a", "channel-b"]


class Notsofar1NaturalOverlapEvalError(RuntimeError):
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
    window_index: int
    channel: Channel
    path: Path = field(repr=False)
    sha256: str
    samples: int
    audio: bytes = field(repr=False)
    reference_a: str = field(repr=False)
    reference_b: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class _BundleSnapshot:
    bundle: fixture.LoadedNotsofarNaturalOverlapBundle
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
        raise Notsofar1NaturalOverlapEvalError() from None
    return raw + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Notsofar1NaturalOverlapEvalError()
    return value


def _strict_json_bytes(raw: bytes, *, maximum_bytes: int) -> object:
    if type(raw) is not bytes or not raw or len(raw) > maximum_bytes:
        raise Notsofar1NaturalOverlapEvalError()

    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise Notsofar1NaturalOverlapEvalError()
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                Notsofar1NaturalOverlapEvalError()
            ),
        )
    except Notsofar1NaturalOverlapEvalError:
        raise
    except (OverflowError, UnicodeError, ValueError):
        raise Notsofar1NaturalOverlapEvalError() from None


def _exact_json_tree(value: object) -> None:
    """Reject subclasses, hooks, non-finite values, and oversized report trees."""

    stack: list[tuple[object, int]] = [(value, 0)]
    items = 0
    while stack:
        current, depth = stack.pop()
        if depth > 16:
            raise Notsofar1NaturalOverlapEvalError()
        current_type = type(current)
        if current_type is dict:
            items += len(current)
            for key, item in current.items():
                if type(key) is not str or not key or len(key) > 128:
                    raise Notsofar1NaturalOverlapEvalError()
                stack.append((item, depth + 1))
        elif current_type is list:
            items += len(current)
            stack.extend((item, depth + 1) for item in current)
        elif current_type is str:
            if len(current) > 16_384:
                raise Notsofar1NaturalOverlapEvalError()
        elif current_type is int:
            if not -(10**18) <= current <= 10**18:
                raise Notsofar1NaturalOverlapEvalError()
        elif current_type is float:
            if not math.isfinite(current):
                raise Notsofar1NaturalOverlapEvalError()
        elif current_type not in {bool, type(None)}:
            raise Notsofar1NaturalOverlapEvalError()
        if items > 100_000:
            raise Notsofar1NaturalOverlapEvalError()


def _load_lock(path: Path | str = DEFAULT_LOCK) -> _EvaluatorLock:
    try:
        selected = Path(os.path.abspath(Path(path).expanduser()))
        if (
            selected != DEFAULT_LOCK
            or selected.resolve(strict=True) != selected
            or selected.name != DEFAULT_LOCK.name
        ):
            raise Notsofar1NaturalOverlapEvalError()
        snapshot = read_regular_bounded(
            selected,
            maximum_bytes=_MAX_LOCK_BYTES,
            expected_bytes=LOCK_FILE_BYTES,
        )
        raw = snapshot.data
        if hashlib.sha256(raw).hexdigest() != LOCK_FILE_SHA256:
            raise Notsofar1NaturalOverlapEvalError()
        value = _strict_json_bytes(raw, maximum_bytes=_MAX_LOCK_BYTES)
        _exact_json_tree(value)
        if type(value) is not dict or set(value) != {
            "bundle",
            "evidence_scope",
            "execution",
            "fixture_id",
            "kind",
            "privacy",
            "recipe_digest_rule",
            "recipe_sha256",
            "schema_version",
        }:
            raise Notsofar1NaturalOverlapEvalError()
        bundle = value.get("bundle")
        execution = value.get("execution")
        scope = value.get("evidence_scope")
        privacy_contract = value.get("privacy")
        if (
            type(bundle) is not dict
            or type(execution) is not dict
            or type(scope) is not dict
            or type(privacy_contract) is not dict
            or type(value.get("schema_version")) is not int
            or value.get("schema_version") != 1
            or value.get("kind") != LOCK_KIND
            or value.get("fixture_id") != fixture.FIXTURE_ID
            or value.get("recipe_digest_rule")
            != "sha256-canonical-json-without-recipe_sha256-v1"
            or value.get("recipe_sha256") != LOCK_RECIPE_SHA256
            or bundle
            != {
                "manifest_file": fixture.MANIFEST_FILENAME,
                "manifest_sha256": FIXTURE_MANIFEST_SHA256,
                "pcm_inputs": 14,
                "receipt_file": fixture.PREPARATION_RECEIPT_FILENAME,
                "receipt_sha256": FIXTURE_RECEIPT_SHA256,
                "windows": _WINDOWS,
            }
            or type(bundle.get("pcm_inputs")) is not int
            or type(bundle.get("windows")) is not int
            or type(execution.get("repeats")) is not int
            or type(execution.get("workers")) is not int
            or any(type(item) is not bool for item in scope.values())
            or type(privacy_contract.get("forbidden_report_content")) is not list
            or any(
                type(item) is not str
                for item in privacy_contract.get("forbidden_report_content", [])
            )
            or execution
            != {
                "input_order": _INPUT_ORDER,
                "metric": OVERLAP_METRIC,
                "repeats": _REPEATS,
                "workers": 1,
            }
            or scope
            != {
                "device_identity_authority": False,
                "diagnostic_only": True,
                "diarization_evidence": False,
                "endpoint_evidence": False,
                "latency_evidence": False,
                "live_evidence": False,
                "promotion_authority": False,
                "qualification_authority": False,
                "runtime_default_authority": False,
            }
            or privacy_contract
            != {
                "aggregate_schema": "aggregate-channel-and-paired-device-counts-only-v1",
                "channel_accuracy": "anonymous-channel-a-channel-b-aggregate-only-v1",
                "device_hypothesis_scoring": "each-device-hypothesis-against-both-window-references-v1",
                "forbidden_report_content": [
                    "references",
                    "reference_fragments",
                    "hypotheses",
                    "window_rows",
                    "speaker_roles",
                    "raw_device_ids",
                    "local_paths",
                ],
            }
        ):
            raise Notsofar1NaturalOverlapEvalError()
        digest_value = dict(value)
        digest_value.pop("recipe_sha256")
        if _canonical_sha256(digest_value) != LOCK_RECIPE_SHA256:
            raise Notsofar1NaturalOverlapEvalError()
        return _EvaluatorLock(
            raw_sha256=LOCK_FILE_SHA256,
            recipe_sha256=LOCK_RECIPE_SHA256,
            manifest_sha256=FIXTURE_MANIFEST_SHA256,
            receipt_sha256=FIXTURE_RECEIPT_SHA256,
        )
    except Notsofar1NaturalOverlapEvalError:
        raise
    except (BoundedReadError, OSError, RuntimeError, TypeError, ValueError):
        raise Notsofar1NaturalOverlapEvalError() from None


def _closure_sha256() -> str:
    try:
        repo_root = Path(__file__).resolve(strict=True).parents[1]
        selected = (
            ("tools/notsofar1_natural_overlap_eval.py", Path(__file__)),
            (
                "tools/prepare_notsofar1_natural_overlap_fixture.py",
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
                raise Notsofar1NaturalOverlapEvalError()
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
    except Notsofar1NaturalOverlapEvalError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        raise Notsofar1NaturalOverlapEvalError() from None


def _current_worker_source_bundle() -> SourceBundle:
    """Reconstruct the exact source-bundle receipt without staging a tree."""

    try:
        repo_root = streaming_stt_eval._REPO_ROOT.resolve(strict=True)
        descriptors: list[int] = []
        payloads: dict[str, bytes] = {}
        with opened_directory_nofollow(repo_root) as (stable_root, root_fd):
            try:
                tools_fd = source_bundle_support._open_child_directory(root_fd, "tools")
                descriptors.append(tools_fd)
                package_fd = source_bundle_support._open_child_directory(
                    tools_fd, "streaming_stt"
                )
                descriptors.append(package_fd)
                adapters_fd = source_bundle_support._open_child_directory(
                    package_fd, "adapters"
                )
                descriptors.append(adapters_fd)
                parents = {
                    "tools/__init__.py": (tools_fd, "__init__.py"),
                    "tools/streaming_stt/__init__.py": (package_fd, "__init__.py"),
                    "tools/streaming_stt/adapters/__init__.py": (
                        adapters_fd,
                        "__init__.py",
                    ),
                }
                for relative in WORKER_SOURCE_FILES:
                    if relative.startswith("tools/streaming_stt/adapters/"):
                        parent, name = adapters_fd, PurePosixPath(relative).name
                    else:
                        parent, name = parents.get(
                            relative,
                            (package_fd, PurePosixPath(relative).name),
                        )
                    payloads[relative] = source_bundle_support._read_source_at(
                        parent, name
                    )
                if (
                    stable_root.lstat().st_dev != os.fstat(root_fd).st_dev
                    or stable_root.lstat().st_ino != os.fstat(root_fd).st_ino
                    or source_bundle_support._directory_identity(
                        os.stat("tools", dir_fd=root_fd, follow_symlinks=False)
                    )
                    != source_bundle_support._directory_identity(os.fstat(tools_fd))
                    or source_bundle_support._directory_identity(
                        os.stat(
                            "streaming_stt",
                            dir_fd=tools_fd,
                            follow_symlinks=False,
                        )
                    )
                    != source_bundle_support._directory_identity(os.fstat(package_fd))
                    or source_bundle_support._directory_identity(
                        os.stat(
                            "adapters",
                            dir_fd=package_fd,
                            follow_symlinks=False,
                        )
                    )
                    != source_bundle_support._directory_identity(os.fstat(adapters_fd))
                ):
                    raise Notsofar1NaturalOverlapEvalError()
            finally:
                for descriptor in reversed(descriptors):
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
        rows = [
            BoundSource(
                relative,
                hashlib.sha256(payloads[relative]).hexdigest(),
                len(payloads[relative]),
            )
            for relative in sorted(WORKER_SOURCE_FILES)
        ]
        manifest_value = {
            "schema_version": SOURCE_BUNDLE_SCHEMA_VERSION,
            "worker": WORKER_RELATIVE_PATH,
            "files": [
                {
                    "path": item.relative_path,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in rows
            ],
        }
        encoded = _canonical_json(manifest_value, newline=True)
        return SourceBundle(
            root=repo_root,
            manifest_path=repo_root / "source-bundle.json",
            tree_sha256=hashlib.sha256(encoded).hexdigest(),
            worker_relative_path=WORKER_RELATIVE_PATH,
            files=tuple(rows),
        )
    except Notsofar1NaturalOverlapEvalError:
        raise
    except Exception:
        raise Notsofar1NaturalOverlapEvalError() from None


def _mapping(value: object, expected: set[str]) -> Mapping[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise Notsofar1NaturalOverlapEvalError()
    return value


def _snapshot_bundle(
    bundle_dir: Path | str,
    evaluator_lock: _EvaluatorLock,
) -> _BundleSnapshot:
    try:
        bundle = fixture.load_notsofar1_natural_overlap_bundle(bundle_dir)
        if (
            bundle.production_evidence is not True
            or bundle.digest != evaluator_lock.manifest_sha256
            or bundle.receipt_sha256 != evaluator_lock.receipt_sha256
            or len(bundle.windows) != _WINDOWS
            or bundle.path.name != fixture.MANIFEST_FILENAME
        ):
            raise Notsofar1NaturalOverlapEvalError()
        fixture.verify_notsofar1_natural_overlap_bundle(bundle)
        root = bundle.path.parent
        inputs: list[_PrivateInput] = []
        with opened_directory_nofollow(root, require_private=True) as (
            opened_root,
            directory_fd,
        ):
            if opened_root != root:
                raise Notsofar1NaturalOverlapEvalError()
            for window_index, raw_window in enumerate(bundle.windows):
                window = _mapping(
                    raw_window,
                    {
                        "channels",
                        "envelope",
                        "overlap",
                        "role_a",
                        "role_b",
                        "window_id",
                    },
                )
                envelope = _mapping(
                    window.get("envelope"),
                    {"samples", "source_end_sample", "source_start_sample"},
                )
                samples = envelope.get("samples")
                role_a = _mapping(
                    window.get("role_a"),
                    {
                        "activity_relative_end_sample",
                        "activity_relative_start_sample",
                        "reference",
                        "reference_sha256",
                    },
                )
                role_b = _mapping(window.get("role_b"), set(role_a))
                references = (role_a.get("reference"), role_b.get("reference"))
                channels = window.get("channels")
                if (
                    type(samples) is not int
                    or samples <= 0
                    or any(
                        type(reference) is not str or not reference
                        for reference in references
                    )
                    or type(channels) is not list
                    or len(channels) != 2
                ):
                    raise Notsofar1NaturalOverlapEvalError()
                for channel_index, raw_channel in enumerate(channels):
                    channel = _mapping(
                        raw_channel,
                        {
                            "case_id",
                            "channel",
                            "file",
                            "samples",
                            "sha256",
                            "size_bytes",
                        },
                    )
                    expected_channel = _CHANNELS[channel_index]
                    name = channel.get("file")
                    if (
                        channel.get("channel") != expected_channel
                        or type(name) is not str
                        or channel.get("samples") != samples
                        or channel.get("size_bytes") != samples * 4
                    ):
                        raise Notsofar1NaturalOverlapEvalError()
                    raw, private_file = fixture._read_private(
                        directory_fd,
                        name,
                        maximum=fixture._MAX_AUDIO_BYTES,
                    )
                    if (
                        private_file.sha256 != _sha256(channel.get("sha256"))
                        or private_file.size_bytes != samples * 4
                    ):
                        raise Notsofar1NaturalOverlapEvalError()
                    inputs.append(
                        _PrivateInput(
                            window_index=window_index,
                            channel=expected_channel,  # type: ignore[arg-type]
                            path=root / name,
                            sha256=private_file.sha256,
                            samples=samples,
                            audio=raw,
                            reference_a=str(references[0]),
                            reference_b=str(references[1]),
                        )
                    )
        expected = tuple(
            (window, channel) for window in range(_WINDOWS) for channel in _CHANNELS
        )
        if tuple((item.window_index, item.channel) for item in inputs) != expected:
            raise Notsofar1NaturalOverlapEvalError()
        final = fixture.load_notsofar1_natural_overlap_bundle(bundle_dir)
        if final != bundle:
            raise Notsofar1NaturalOverlapEvalError()
        return _BundleSnapshot(bundle=bundle, inputs=tuple(inputs))
    except Notsofar1NaturalOverlapEvalError:
        raise
    except Exception:
        raise Notsofar1NaturalOverlapEvalError() from None


def _same_bundle(left: _BundleSnapshot, right: _BundleSnapshot) -> bool:
    return left == right


def preflight_notsofar1_natural_overlap_bundle(
    bundle_dir: Path | str,
) -> dict[str, object]:
    """Validate only the closed evaluator and exact retained production bundle."""

    evaluator_lock = _load_lock()
    closure = _closure_sha256()
    loaded = _snapshot_bundle(bundle_dir, evaluator_lock)
    current = _snapshot_bundle(bundle_dir, evaluator_lock)
    if not _same_bundle(loaded, current) or _closure_sha256() != closure:
        raise Notsofar1NaturalOverlapEvalError()
    return {
        "fixture_id": fixture.FIXTURE_ID,
        "windows": _WINDOWS,
        "pcm_inputs": 14,
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
        raise Notsofar1NaturalOverlapEvalError() from None


def _stable_existing_directory(value: Path | str) -> Path:
    try:
        candidate = Path(os.path.abspath(Path(value).expanduser()))
        if candidate.resolve(strict=True) != candidate:
            raise Notsofar1NaturalOverlapEvalError()
        with opened_directory_nofollow(candidate, require_private=True) as (
            stable,
            _fd,
        ):
            if stable != candidate:
                raise Notsofar1NaturalOverlapEvalError()
        return candidate
    except Notsofar1NaturalOverlapEvalError:
        raise
    except (BoundedReadError, OSError, RuntimeError, TypeError, ValueError):
        raise Notsofar1NaturalOverlapEvalError() from None


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
            raise Notsofar1NaturalOverlapEvalError()
        output_parent = _stable_existing_directory(candidate.parent)
        output = output_parent / candidate.name
        if _has_git_ancestor(output_parent) or _paths_overlap(scratch, output_parent):
            raise Notsofar1NaturalOverlapEvalError()
    if _has_git_ancestor(scratch):
        raise Notsofar1NaturalOverlapEvalError()
    for root in protected:
        if _paths_overlap(scratch, root) or (
            output is not None and _paths_overlap(output.parent, root)
        ):
            raise Notsofar1NaturalOverlapEvalError()
    return scratch, output


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
        raise Notsofar1NaturalOverlapEvalError()
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
                raise Notsofar1NaturalOverlapEvalError()
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
                raise Notsofar1NaturalOverlapEvalError()
            digest = hash_regular_bounded(
                path,
                maximum_bytes=streaming_stt_eval.MAX_PCM_BYTES,
                expected_bytes=len(item.audio),
            )
            if digest.sha256 != item.sha256:
                raise Notsofar1NaturalOverlapEvalError()
        return opened_snapshot, names
    except Notsofar1NaturalOverlapEvalError:
        raise
    except Exception:
        raise Notsofar1NaturalOverlapEvalError() from None


def _source_complete(
    item: _PrivateInput,
    final: object,
    *,
    tail_padding_samples: int = 0,
) -> bool:
    if type(tail_padding_samples) is not int or tail_padding_samples < 0:
        return False
    if type(final) is NativeFinalEvent:
        return (
            final.source_samples == item.samples
            and final.source_samples_consumed == item.samples
            and final.declared_tail_samples == tail_padding_samples
            and 0 <= final.tail_samples_consumed <= tail_padding_samples
            and math.isclose(
                final.audio_seconds,
                item.samples / fixture.SAMPLE_RATE_HZ,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and final.samples_seen
            == final.source_samples_consumed + final.tail_samples_consumed
            and (
                (
                    final.native_endpoint is True
                    and final.endpoint_reason in {"eou", "eob"}
                )
                or (
                    final.native_endpoint is False
                    and final.endpoint_reason == "tail_exhausted"
                    and final.tail_samples_consumed == tail_padding_samples
                )
            )
        )
    if type(final) is ParakeetCppFinalEvent:
        return (
            final.source_samples_offered == item.samples
            and final.source_samples_consumed == item.samples
            and final.tail_samples_offered == tail_padding_samples
            and 0 <= final.tail_samples_consumed <= tail_padding_samples
            and math.isclose(
                final.audio_seconds,
                item.samples / fixture.SAMPLE_RATE_HZ,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and final.samples_seen
            == final.source_samples_consumed + final.tail_samples_consumed
            and (
                (final.native_endpoint is True and final.endpoint_reason == "eou")
                or (
                    final.native_endpoint is False
                    and final.endpoint_reason == "tail_exhausted"
                    and final.tail_samples_consumed == tail_padding_samples
                )
            )
        )
    if type(final) is FinalEvent:
        return (
            final.samples_seen == item.samples + tail_padding_samples
            and math.isclose(
                final.audio_seconds,
                item.samples / fixture.SAMPLE_RATE_HZ,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
    return False


def _aggregate_accuracy(results: Sequence[OverlapMetricResult]) -> dict[str, object]:
    aggregate = aggregate_two_utterance_min_order_wer(results).as_dict()
    cases = aggregate.pop("cases", None)
    if type(cases) is not int or cases != len(results):
        raise Notsofar1NaturalOverlapEvalError()
    return {"evaluations": cases, **aggregate}


def _overlap_metrics(
    initial: _BundleSnapshot,
    records: Sequence[RunRecord],
    *,
    tail_padding_samples: int = 0,
) -> dict[str, object]:
    if len(records) != len(initial.inputs) or len(records) != 14:
        raise Notsofar1NaturalOverlapEvalError()
    aggregate_results: list[OverlapMetricResult] = []
    channel_results: dict[str, list[OverlapMetricResult]] = {
        "channel-a": [],
        "channel-b": [],
    }
    pairs: dict[int, dict[str, tuple[OverlapMetricResult, tuple[str, ...]]]] = {}
    for expected_index, record in enumerate(records):
        if (
            type(record) is not RunRecord
            or record.case_index != expected_index
            or record.repeat != 0
            or type(record.trace) is not CaseTrace
            or type(record.trace.partials) is not tuple
            or any(
                type(partial) is not PartialEvent for partial in record.trace.partials
            )
            or type(record.trace.final)
            not in {FinalEvent, NativeFinalEvent, ParakeetCppFinalEvent}
        ):
            raise Notsofar1NaturalOverlapEvalError()
        item = initial.inputs[record.case_index]
        final = record.trace.final
        if not _source_complete(
            item,
            final,
            tail_padding_samples=tail_padding_samples,
        ):
            raise Notsofar1NaturalOverlapEvalError()
        hypothesis = final.text
        measured = score_two_utterance_min_order_wer(
            item.reference_a,
            item.reference_b,
            hypothesis,
        )
        aggregate_results.append(measured)
        channel_results[item.channel].append(measured)
        pairs.setdefault(item.window_index, {})[item.channel] = (
            measured,
            tuple(normalize(hypothesis)),
        )
    if len(pairs) != _WINDOWS or any(
        set(value) != set(_CHANNELS) for value in pairs.values()
    ):
        raise Notsofar1NaturalOverlapEvalError()
    exact_agreement = 0
    order_agreement = 0
    absolute_word_error_difference = 0
    absolute_hypothesis_word_difference = 0
    for value in pairs.values():
        channel_a, hypothesis_a = value["channel-a"]
        channel_b, hypothesis_b = value["channel-b"]
        exact_agreement += int(hypothesis_a == hypothesis_b)
        order_agreement += int(channel_a.selected_order == channel_b.selected_order)
        absolute_word_error_difference += abs(
            channel_a.word_errors - channel_b.word_errors
        )
        absolute_hypothesis_word_difference += abs(
            channel_a.hypothesis_words - channel_b.hypothesis_words
        )
    return {
        "coverage": {
            "windows": _WINDOWS,
            "pcm_inputs": 14,
            "repeats": _REPEATS,
            "evaluations": 14,
            "source_complete_evaluations": 14,
        },
        "aggregate_accuracy": _aggregate_accuracy(aggregate_results),
        "channel_accuracy": {
            "channel_a": _aggregate_accuracy(channel_results["channel-a"]),
            "channel_b": _aggregate_accuracy(channel_results["channel-b"]),
        },
        "paired_device": {
            "windows": _WINDOWS,
            "comparable_windows": _WINDOWS,
            "exact_match_agreement_windows": exact_agreement,
            "selected_order_agreement_windows": order_agreement,
            "absolute_word_error_difference": absolute_word_error_difference,
            "absolute_hypothesis_word_difference": absolute_hypothesis_word_difference,
        },
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
        raise Notsofar1NaturalOverlapEvalError()
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
            raise Notsofar1NaturalOverlapEvalError()
    except (OSError, RuntimeError):
        raise Notsofar1NaturalOverlapEvalError() from None
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
            raise Notsofar1NaturalOverlapEvalError()
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
                raise Notsofar1NaturalOverlapEvalError()
            pcm_paths.append(path)

        worker = StreamingWorker(manifest, scratch, source_bundle)
        _enter_worker_owned(worker)
        session_error: BaseException | None = None
        try:
            if worker.ready is None:
                raise Notsofar1NaturalOverlapEvalError()
            for index, (item, pcm_path) in enumerate(
                zip(initial.inputs, pcm_paths, strict=True)
            ):
                request = TranscribeRequest(
                    request_id=f"notsofar1-natural-overlap-{index:02d}",
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
                        repeat=0,
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
            raise Notsofar1NaturalOverlapEvalError()

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
            raise Notsofar1NaturalOverlapEvalError()
        metrics = _overlap_metrics(
            initial,
            records,
            tail_padding_samples=selected_stream.tail_padding_samples,
        )
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
            "workers": 1,
            "repeats": _REPEATS,
            "input_order": _INPUT_ORDER,
            "contract_sha256": "",
        }
        config["contract_sha256"] = _canonical_sha256(
            {key: value for key, value in config.items() if key != "contract_sha256"}
        )
        model_executed = evaluator_binding.get("model_executed")
        if type(model_executed) is not bool:
            raise Notsofar1NaturalOverlapEvalError()
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "ok": True,
            "evidence": {
                "device_identity_authority": False,
                "diagnostic_only": True,
                "diarization_evidence": False,
                "endpoint_evidence": False,
                "latency_evidence": False,
                "live_evidence": False,
                "model_executed": model_executed,
                "promotion_authority": False,
                "qualification_authority": False,
                "runtime_default_authority": False,
            },
            "bundle": {
                "fixture_id": fixture.FIXTURE_ID,
                "manifest_sha256": initial.bundle.digest,
                "receipt_sha256": initial.bundle.receipt_sha256,
                "source_contract_sha256": initial.bundle.source_contract_sha256,
                "evaluator_lock_file_sha256": evaluator_lock.raw_sha256,
                "evaluator_lock_recipe_sha256": evaluator_lock.recipe_sha256,
                "windows": _WINDOWS,
                "pcm_inputs": 14,
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
            raise Notsofar1NaturalOverlapEvalError()
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
        cleanup_error = Notsofar1NaturalOverlapEvalError()
    elif scratch is not None:
        if scratch_identity is None:
            try:
                metadata = scratch.lstat()
                if (
                    not stat.S_ISDIR(metadata.st_mode)
                    or stat.S_IMODE(metadata.st_mode) != 0o700
                    or metadata.st_uid != os.getuid()
                ):
                    raise Notsofar1NaturalOverlapEvalError()
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
        raise Notsofar1NaturalOverlapEvalError() from None
    if report is None or final_guard is None:
        raise Notsofar1NaturalOverlapEvalError()
    if output is None:
        if final_guard() is not True:
            raise Notsofar1NaturalOverlapEvalError()
        return report
    try:
        _publish_report(output, report, commit_guard=final_guard, state=commit_state)
        return report
    except BaseException:
        if commit_state.committed:
            return report
        raise


def _validate_accuracy(
    value: object,
    *,
    evaluations: int,
) -> dict[str, object]:
    checked = _mapping(
        value,
        {
            "deletions",
            "evaluations",
            "hypothesis_words",
            "insertions",
            "protocol",
            "reference_words",
            "substitutions",
            "wer",
            "word_errors",
        },
    )
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
        raise Notsofar1NaturalOverlapEvalError()
    substitutions, insertions, deletions, errors, reference, hypothesis = counts
    wer = checked.get("wer")
    if (
        checked.get("evaluations") != evaluations
        or checked.get("protocol") != OVERLAP_METRIC
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
        raise Notsofar1NaturalOverlapEvalError()
    return dict(checked)


def _validate_report(
    value: object,
    *,
    initial: _BundleSnapshot,
    expected_worker: Mapping[str, object] | None = None,
    expected_evaluator: Mapping[str, object] | None = None,
    expected_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    _exact_json_tree(value)
    if (
        type(value) is not dict
        or set(value) != _REPORT_ROOT_FIELDS
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != KIND
        or value.get("ok") is not True
        or any(
            type(value.get(field)) is not dict
            for field in (
                "metrics",
                "bundle",
                "worker",
                "evaluator",
                "evidence",
                "config",
                "worker_stderr",
            )
        )
    ):
        raise Notsofar1NaturalOverlapEvalError()
    expected_binding = _canonical_sha256(
        {key: item for key, item in value.items() if key != "binding_sha256"}
    )
    if value.get("binding_sha256") != expected_binding:
        raise Notsofar1NaturalOverlapEvalError()
    bundle = _mapping(
        value.get("bundle"),
        {
            "evaluator_lock_file_sha256",
            "evaluator_lock_recipe_sha256",
            "fixture_id",
            "manifest_sha256",
            "pcm_inputs",
            "receipt_sha256",
            "source_contract_sha256",
            "windows",
        },
    )
    if (
        bundle.get("fixture_id") != fixture.FIXTURE_ID
        or type(bundle.get("windows")) is not int
        or bundle.get("windows") != _WINDOWS
        or type(bundle.get("pcm_inputs")) is not int
        or bundle.get("pcm_inputs") != 14
        or bundle.get("manifest_sha256") != initial.bundle.digest
        or bundle.get("receipt_sha256") != initial.bundle.receipt_sha256
        or bundle.get("source_contract_sha256") != initial.bundle.source_contract_sha256
        or bundle.get("evaluator_lock_file_sha256") != LOCK_FILE_SHA256
        or bundle.get("evaluator_lock_recipe_sha256") != LOCK_RECIPE_SHA256
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
        raise Notsofar1NaturalOverlapEvalError()
    evidence = _mapping(
        value.get("evidence"),
        {
            "device_identity_authority",
            "diagnostic_only",
            "diarization_evidence",
            "endpoint_evidence",
            "latency_evidence",
            "live_evidence",
            "model_executed",
            "promotion_authority",
            "qualification_authority",
            "runtime_default_authority",
        },
    )
    if (
        evidence
        != {
            "device_identity_authority": False,
            "diagnostic_only": True,
            "diarization_evidence": False,
            "endpoint_evidence": False,
            "latency_evidence": False,
            "live_evidence": False,
            "model_executed": evidence.get("model_executed"),
            "promotion_authority": False,
            "qualification_authority": False,
            "runtime_default_authority": False,
        }
        or type(evidence.get("model_executed")) is not bool
    ):
        raise Notsofar1NaturalOverlapEvalError()
    config = _mapping(
        value.get("config"),
        {
            "chunk_samples",
            "contract_sha256",
            "input_order",
            "pace",
            "partial_interval_ms",
            "repeats",
            "tail_padding_samples",
            "workers",
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
        or type(config.get("repeats")) is not int
        or config.get("workers") != 1
        or type(config.get("workers")) is not int
        or config.get("input_order") != _INPUT_ORDER
        or config.get("contract_sha256")
        != _canonical_sha256(
            {key: item for key, item in config.items() if key != "contract_sha256"}
        )
        or (expected_config is not None and config != expected_config)
    ):
        raise Notsofar1NaturalOverlapEvalError()
    metrics = _mapping(
        value.get("metrics"),
        {
            "aggregate_accuracy",
            "channel_accuracy",
            "coverage",
            "paired_device",
        },
    )
    coverage = _mapping(
        metrics.get("coverage"),
        {
            "evaluations",
            "pcm_inputs",
            "repeats",
            "source_complete_evaluations",
            "windows",
        },
    )
    if any(type(coverage.get(field)) is not int for field in coverage) or coverage != {
        "windows": _WINDOWS,
        "pcm_inputs": 14,
        "repeats": _REPEATS,
        "evaluations": 14,
        "source_complete_evaluations": 14,
    }:
        raise Notsofar1NaturalOverlapEvalError()
    aggregate_accuracy = _validate_accuracy(
        metrics.get("aggregate_accuracy"), evaluations=14
    )
    channels = _mapping(
        metrics.get("channel_accuracy"),
        {"channel_a", "channel_b"},
    )
    channel_values = tuple(
        _validate_accuracy(channels.get(channel), evaluations=7)
        for channel in ("channel_a", "channel_b")
    )
    count_fields = (
        "substitutions",
        "insertions",
        "deletions",
        "word_errors",
        "reference_words",
        "hypothesis_words",
    )
    if any(
        aggregate_accuracy[field]
        != sum(int(channel[field]) for channel in channel_values)
        for field in count_fields
    ):
        raise Notsofar1NaturalOverlapEvalError()
    expected_reference_words = sum(
        len(normalize(item.reference_a)) + len(normalize(item.reference_b))
        for item in initial.inputs
    )
    paired = _mapping(
        metrics.get("paired_device"),
        {
            "absolute_hypothesis_word_difference",
            "absolute_word_error_difference",
            "comparable_windows",
            "exact_match_agreement_windows",
            "selected_order_agreement_windows",
            "windows",
        },
    )
    if (
        aggregate_accuracy.get("reference_words") != expected_reference_words
        or any(
            channel.get("reference_words") * 2 != expected_reference_words
            for channel in channel_values
        )
        or any(type(item) is not int or item < 0 for item in paired.values())
        or paired.get("windows") != _WINDOWS
        or paired.get("comparable_windows") != _WINDOWS
        or not 0 <= paired["exact_match_agreement_windows"] <= _WINDOWS
        or not 0 <= paired["selected_order_agreement_windows"] <= _WINDOWS
        or paired["absolute_word_error_difference"]
        > sum(int(channel["word_errors"]) for channel in channel_values)
        or paired["absolute_hypothesis_word_difference"]
        > sum(int(channel["hypothesis_words"]) for channel in channel_values)
    ):
        raise Notsofar1NaturalOverlapEvalError()
    evaluator = _mapping(
        value.get("evaluator"),
        {"generic", "wrapper_closure_sha256"},
    )
    try:
        privacy._validated_evaluator(evaluator.get("generic"))
    except Exception:
        raise Notsofar1NaturalOverlapEvalError() from None
    generic = evaluator.get("generic")
    if (
        type(generic) is not dict
        or evidence.get("model_executed") != generic.get("model_executed")
        or _sha256(evaluator.get("wrapper_closure_sha256"))
        != evaluator.get("wrapper_closure_sha256")
        or (expected_evaluator is not None and evaluator != expected_evaluator)
        or (expected_worker is not None and value.get("worker") != expected_worker)
    ):
        raise Notsofar1NaturalOverlapEvalError()
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
        "sc_meetup_0",
        "sc_rockfall_2",
        "sc_meetup_0/ch0.wav",
        "sc_rockfall_2/ch0.wav",
        *(str(window.get("window_id")) for window in initial.bundle.windows),
        *(
            str(channel.get("case_id"))
            for window in initial.bundle.windows
            for channel in window["channels"]
        ),
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
        forbidden_schema_terms = (
            '"hypothesis"',
            '"reference"',
            '"window_rows"',
            '"raw_device_id"',
            '"speaker_role"',
            '"winner"',
            '"delta"',
            '"ordinary_wer"',
        )
        if (
            len(encoded) > _MAX_REPORT_BYTES
            or any(label and label.casefold() in lowered for label in forbidden_labels)
            or any(term in lowered for term in forbidden_schema_terms)
        ):
            raise Notsofar1NaturalOverlapEvalError()
    except Notsofar1NaturalOverlapEvalError:
        raise
    except Exception:
        raise Notsofar1NaturalOverlapEvalError() from None
    return dict(value)


def validate_notsofar1_natural_overlap_report(
    value: object,
    *,
    bundle_dir: Path | str,
    worker_manifest_path: Path | str,
) -> dict[str, object]:
    """Rebind one closed report to the bundle and current worker authority."""

    _exact_json_tree(value)
    evaluator_lock = _load_lock()
    closure = _closure_sha256()
    initial = _snapshot_bundle(bundle_dir, evaluator_lock)
    try:
        manifest = streaming_stt_eval.load_worker_manifest(worker_manifest_path)
        if (
            manifest.worker.path.resolve(strict=True)
            != streaming_stt_eval._FIXED_WORKER
        ):
            raise Notsofar1NaturalOverlapEvalError()
        runtime_receipt = streaming_stt_eval._verified_runtime_receipt_digest(manifest)
        model_receipt = streaming_stt_eval._verified_model_receipt_digest(manifest)
        evaluator_binding = streaming_stt_eval._evaluator_binding(manifest.adapter)
        selected_stream = streaming_stt_eval._selected_stream(
            manifest,
            None,
            chunk_samples=None,
            pace=None,
            partial_interval_ms=None,
            tail_padding_samples=None,
        )
        config: dict[str, object] = {
            **selected_stream.as_dict(),
            "workers": 1,
            "repeats": _REPEATS,
            "input_order": _INPUT_ORDER,
            "contract_sha256": "",
        }
        config["contract_sha256"] = _canonical_sha256(
            {key: item for key, item in config.items() if key != "contract_sha256"}
        )
        report = _mapping(value, set(_REPORT_ROOT_FIELDS))
        reported_worker = _mapping(report.get("worker"), set(report["worker"]))
        reported_runtime = _mapping(
            reported_worker.get("runtime"),
            {"platform", "python"},
        )
        if any(
            type(reported_runtime.get(field)) is not str
            or not reported_runtime[field]
            or len(reported_runtime[field]) > 256
            for field in ("platform", "python")
        ):
            raise Notsofar1NaturalOverlapEvalError()
        source_tree_sha256 = _sha256(reported_worker.get("source_bundle_sha256"))
        source_bundle = _current_worker_source_bundle()
        rebound_source_bundle = _current_worker_source_bundle()
        if (
            source_bundle.tree_sha256 != source_tree_sha256
            or rebound_source_bundle != source_bundle
        ):
            raise Notsofar1NaturalOverlapEvalError()
        expected_worker = streaming_stt_eval._worker_report(
            manifest,
            streaming_stt_eval._worker_binding(
                manifest,
                source_bundle,
                runtime_receipt_sha256=runtime_receipt,
                model_receipt_sha256=model_receipt,
            ),
            reported_runtime,
            reported_worker.get("cgroup_evidence"),
            reported_worker.get("resource_observations"),
            expected_completed_cases=14,
        )
        validated = _validate_report(
            value,
            initial=initial,
            expected_worker=expected_worker,
            expected_evaluator={
                "generic": evaluator_binding,
                "wrapper_closure_sha256": closure,
            },
            expected_config=config,
        )
    except Notsofar1NaturalOverlapEvalError:
        raise
    except Exception:
        raise Notsofar1NaturalOverlapEvalError() from None
    current = _snapshot_bundle(bundle_dir, evaluator_lock)
    current_manifest = streaming_stt_eval.load_worker_manifest(manifest.path)
    current_source_bundle = _current_worker_source_bundle()
    if (
        not _same_bundle(initial, current)
        or _load_lock() != evaluator_lock
        or _closure_sha256() != closure
        or current_manifest.digest != manifest.digest
        or streaming_stt_eval._verified_runtime_receipt_digest(current_manifest)
        != runtime_receipt
        or streaming_stt_eval._verified_model_receipt_digest(current_manifest)
        != model_receipt
        or streaming_stt_eval._evaluator_binding(current_manifest.adapter)
        != evaluator_binding
        or current_source_bundle != source_bundle
        or streaming_stt_eval._worker_binding(
            current_manifest,
            source_bundle,
            runtime_receipt_sha256=runtime_receipt,
            model_receipt_sha256=model_receipt,
        )
        != streaming_stt_eval._worker_binding(
            manifest,
            source_bundle,
            runtime_receipt_sha256=runtime_receipt,
            model_receipt_sha256=model_receipt,
        )
    ):
        raise Notsofar1NaturalOverlapEvalError()
    return validated


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
            raise Notsofar1NaturalOverlapEvalError()
        parent = _stable_existing_directory(path.parent)
        if path.parent != parent or path.exists() or path.is_symlink():
            raise Notsofar1NaturalOverlapEvalError()
        encoded = _canonical_json(report, newline=True)
        digest = hashlib.sha256(encoded).hexdigest()
        with opened_directory_nofollow(parent, require_private=True) as (
            stable_parent,
            directory_fd,
        ):
            if stable_parent != parent:
                raise Notsofar1NaturalOverlapEvalError()
            parent_before = fixture._file_snapshot(os.fstat(directory_fd))
            try:
                os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise Notsofar1NaturalOverlapEvalError()
            temporary_flag = getattr(os, "O_TMPFILE", 0)
            if not temporary_flag:
                raise Notsofar1NaturalOverlapEvalError()
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
                raise Notsofar1NaturalOverlapEvalError()
            written = 0
            view = memoryview(encoded)
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if type(count) is not int or count <= 0:
                    raise Notsofar1NaturalOverlapEvalError()
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
                raise Notsofar1NaturalOverlapEvalError()
            if commit_guard() is not True:
                raise Notsofar1NaturalOverlapEvalError()
            os.lseek(descriptor, 0, os.SEEK_SET)
            final_observed = bytearray()
            while len(final_observed) <= len(encoded):
                chunk = os.read(descriptor, len(encoded) + 1 - len(final_observed))
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
                raise Notsofar1NaturalOverlapEvalError()
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
                raise Notsofar1NaturalOverlapEvalError()
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
        raise Notsofar1NaturalOverlapEvalError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def run_notsofar1_natural_overlap_eval(
    worker_manifest_path: Path | str,
    bundle_dir: Path | str,
    *,
    scratch_parent: Path | str,
    output_path: Path | str | None = None,
    _commit_state: _ReportCommitState | None = None,
) -> dict[str, object]:
    """Run the fixed one-worker, 14-input diagnostic and optionally publish it."""

    state = _commit_state or _ReportCommitState()
    if state.committed or state.digest:
        raise Notsofar1NaturalOverlapEvalError()
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
        if isinstance(pending, Notsofar1NaturalOverlapEvalError):
            raise pending
        raise Notsofar1NaturalOverlapEvalError() from None
    if report is None:
        raise Notsofar1NaturalOverlapEvalError()
    return report


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise Notsofar1NaturalOverlapEvalError()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Run the fixed retained NOTSOFAR-1 natural-overlap diagnostic."
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
        report = run_notsofar1_natural_overlap_eval(
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
    "Notsofar1NaturalOverlapEvalError",
    "SCHEMA_VERSION",
    "main",
    "preflight_notsofar1_natural_overlap_bundle",
    "run_notsofar1_natural_overlap_eval",
    "validate_notsofar1_natural_overlap_report",
]
