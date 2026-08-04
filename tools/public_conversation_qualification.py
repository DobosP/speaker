"""Exact development-only public-conversation STT qualification wrapper.

The four source receipts remain owned by :mod:`tools.public_conversation_fixture`.
This additive controller closes the orchestration gap: every two-worker run
must expose the source-specific strata already declared by the locked fixture.
Execution success is coverage evidence only; this module has no promotion or
runtime authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PureWindowsPath

from tools import public_conversation_fixture as fixture
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.corpus import load_corpus, verify_corpus_snapshot
from tools.streaming_stt.metrics import (
    normalize_stratum_tags,
    validate_stratum_tags,
)
from tools.streaming_stt.manifest import load_worker_manifest

SCHEMA_VERSION = 1
KIND = "public-conversation-canonical-strata-qualification-v1"
WORKER_ROLES = ("baseline", "candidate")

_SAFE_ERROR = {"error": "public conversation qualification failed", "ok": False}
_MAX_IMPLEMENTATION_BYTES = 256 * 1024
_MAX_REPORT_BYTES = 64 * 1024 * 1024
_MAX_PATH_CHARS = 4096
_INNER_CHECKPOINT_NAME = ".canonical-strata-inner-suite-checkpoint.json"
_REPORT_TEMP_PREFIX = ".public-conversation-report-"
_AMI_SOURCE = "ami-es2004a-close-far"
_EXPECTED_SELECTION_LAYOUT = {
    "harper-valley": tuple((f"task_{index:02d}", 3) for index in range(8)),
    "edacc-test": (("clean", 8), ("disfluent", 8), ("overlap_adjacent", 8)),
    _AMI_SOURCE: (
        ("isolated_close", 8),
        ("isolated_far", 8),
        ("turn_transition_close", 4),
        ("turn_transition_far", 4),
    ),
    "cvss4-en": (
        ("d0_2_6s", 6),
        ("d1_6_10s", 6),
        ("d2_10_15s", 6),
        ("d3_15_25s", 6),
    ),
}
_EXPECTED_SLOT_STRATA = {
    "harper-valley": tuple(f"task_{index:02d}" for index in range(8)),
    "edacc-test": ("clean", "disfluent", "overlap_adjacent"),
    _AMI_SOURCE: ("isolated", "turn_transition"),
    "cvss4-en": ("d0_2_6s", "d1_6_10s", "d2_10_15s", "d3_15_25s"),
}
_EXPECTED_SLOT_CHANNELS = {
    "harper-valley": ("caller",),
    "edacc-test": ("single",),
    _AMI_SOURCE: ("close", "far"),
    "cvss4-en": ("original",),
}
_EXPECTED_SLOT_LAYOUT = {
    "harper-valley": tuple(
        (f"task_{index:02d}", "caller")
        for index in range(8)
        for _case in range(3)
    ),
    "edacc-test": tuple(
        (stratum, "single")
        for stratum in ("clean", "disfluent", "overlap_adjacent")
        for _case in range(8)
    ),
    _AMI_SOURCE: tuple(
        (stratum, channel)
        for stratum, pairs in (("isolated", 8), ("turn_transition", 4))
        for _pair in range(pairs)
        for channel in ("close", "far")
    ),
    "cvss4-en": tuple(
        (bucket, "original")
        for bucket in ("d0_2_6s", "d1_6_10s", "d2_10_15s", "d3_15_25s")
        for _case in range(6)
    ),
}
_EXPECTED_REPORT_STRATA = {
    source_id: normalize_stratum_tags(
        (
            *_EXPECTED_SLOT_STRATA[source_id],
            *(
                _EXPECTED_SLOT_CHANNELS[source_id]
                if len(_EXPECTED_SLOT_CHANNELS[source_id]) > 1
                else ()
            ),
        )
    )
    for source_id in fixture.SOURCE_IDS
}

RunValidatedSuite = Callable[..., Mapping[str, object]]


class QualificationError(RuntimeError):
    """A detail-free qualification-contract failure."""


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
        raise QualificationError() from None


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _implementation_sha256() -> str:
    try:
        path = Path(__file__).resolve(strict=True)
        metadata = path.lstat()
        return hash_regular_bounded(
            path,
            maximum_bytes=_MAX_IMPLEMENTATION_BYTES,
            expected_bytes=metadata.st_size,
        ).sha256
    except Exception:
        raise QualificationError() from None


def _private_parent(path: Path) -> Path:
    try:
        parent = path.parent.resolve(strict=True)
        metadata = parent.lstat()
    except Exception:
        raise QualificationError() from None
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise QualificationError()
    return parent


def _new_private_output_path(value: Path | str) -> Path:
    if isinstance(value, bytes) or not isinstance(value, (str, os.PathLike)):
        raise QualificationError()
    try:
        supplied = Path(value).expanduser()
        if not supplied.is_absolute() or not supplied.name:
            raise QualificationError()
        candidate = _private_parent(supplied) / supplied.name
        if len(str(candidate)) > _MAX_PATH_CHARS:
            raise QualificationError()
        candidate.lstat()
    except FileNotFoundError:
        return candidate
    except QualificationError:
        raise
    except Exception:
        raise QualificationError() from None
    raise QualificationError()


def _scratch_candidate(value: Path | str) -> Path:
    if isinstance(value, bytes) or not isinstance(value, (str, os.PathLike)):
        raise QualificationError()
    try:
        supplied = Path(value).expanduser()
        if not supplied.is_absolute():
            raise QualificationError()
        scratch = supplied.resolve(strict=False)
        if len(str(scratch)) > _MAX_PATH_CHARS:
            raise QualificationError()
        return scratch
    except QualificationError:
        raise
    except Exception:
        raise QualificationError() from None


def _prepare_private_scratch(scratch: Path) -> tuple[Path, Path]:
    try:
        scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
        scratch = scratch.resolve(strict=True)
        metadata = scratch.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise QualificationError()
        checkpoint = scratch / _INNER_CHECKPOINT_NAME
    except QualificationError:
        raise
    except Exception:
        raise QualificationError() from None
    try:
        checkpoint.lstat()
    except FileNotFoundError:
        return scratch, checkpoint
    except Exception:
        raise QualificationError() from None
    raise QualificationError()


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _has_git_ancestor(path: Path) -> bool:
    for ancestor in (path, *path.parents):
        marker = ancestor / ".git"
        try:
            metadata = marker.lstat()
        except (FileNotFoundError, NotADirectoryError):
            continue
        except Exception:
            raise QualificationError() from None
        if not stat.S_ISDIR(metadata.st_mode):
            return True
        try:
            (marker / "HEAD").lstat()
        except (FileNotFoundError, NotADirectoryError):
            continue
        except Exception:
            raise QualificationError() from None
        return True
    return False


def _validate_private_paths(
    *,
    final_output: Path,
    scratch: Path,
    suite_checkpoint: Path,
    fixture_set: fixture.FixtureSet,
    worker_evidence_roots: Sequence[Path],
) -> None:
    repo_root = Path(__file__).resolve(strict=True).parents[1]
    protected_roots = (
        *(source.corpus_path.parent for source in fixture_set.sources),
        *worker_evidence_roots,
    )
    selected_paths = (final_output, scratch, suite_checkpoint)
    if (
        final_output == suite_checkpoint
        or _is_within(scratch, final_output)
        or _is_within(final_output, repo_root)
        or _is_within(scratch, repo_root)
        or any(_has_git_ancestor(path) for path in selected_paths)
        or any(
            _is_within(selected, root)
            for selected in selected_paths
            for root in protected_roots
        )
    ):
        raise QualificationError()


def _entry_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_uid,
    )


def _unlink_same_entry(
    directory_descriptor: int,
    name: str,
    identity: tuple[int, int, int, int],
) -> None:
    try:
        metadata = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if _entry_identity(metadata) == identity:
            os.unlink(name, dir_fd=directory_descriptor)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def write_new_report(path: Path | str, payload: Mapping[str, object]) -> None:
    """Failure-atomically publish one report without replacing an existing file."""

    selected = _new_private_output_path(path)
    _assert_path_free(payload)
    encoded = _canonical_json_bytes(payload) + b"\n"
    if len(encoded) > _MAX_REPORT_BYTES:
        raise QualificationError()

    descriptor = -1
    temporary_name: str | None = None
    temporary_identity: tuple[int, int, int, int] | None = None
    final_linked = False
    try:
        with opened_directory_nofollow(
            selected.parent,
            require_private=True,
        ) as (stable_parent, directory_descriptor):
            if stable_parent != selected.parent:
                raise QualificationError()
            try:
                os.stat(
                    selected.name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise QualificationError()

            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            temporary_name = f"{_REPORT_TEMP_PREFIX}{secrets.token_hex(16)}.tmp"
            descriptor = os.open(
                temporary_name,
                flags,
                0o600,
                dir_fd=directory_descriptor,
            )
            created = os.fstat(descriptor)
            if (
                not stat.S_ISREG(created.st_mode)
                or created.st_nlink != 1
                or (hasattr(os, "geteuid") and created.st_uid != os.geteuid())
            ):
                raise QualificationError()
            temporary_identity = _entry_identity(created)
            os.fchmod(descriptor, 0o600)
            created = os.fstat(descriptor)
            if (
                _entry_identity(created) != temporary_identity
                or stat.S_IMODE(created.st_mode) != 0o600
            ):
                raise QualificationError()
            view = memoryview(encoded)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if type(count) is not int or count <= 0:
                    raise QualificationError()
                written += count
            os.fsync(descriptor)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size != len(encoded)
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            ):
                raise QualificationError()
            if _entry_identity(metadata) != temporary_identity:
                raise QualificationError()

            os.link(
                temporary_name,
                selected.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            final_linked = True
            linked = os.stat(
                selected.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            after_link = os.fstat(descriptor)
            if (
                _entry_identity(linked) != temporary_identity
                or _entry_identity(after_link) != temporary_identity
                or linked.st_nlink != 2
                or after_link.st_nlink != 2
            ):
                raise QualificationError()

            os.unlink(temporary_name, dir_fd=directory_descriptor)
            temporary_name = None
            published = os.stat(
                selected.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if (
                _entry_identity(published) != temporary_identity
                or published.st_nlink != 1
                or published.st_size != len(encoded)
                or stat.S_IMODE(published.st_mode) != 0o600
            ):
                raise QualificationError()
            try:
                os.fsync(directory_descriptor)
            except OSError:
                # The complete, file-fsynced report is already published. Do
                # not report a non-mutating failure after that commit point.
                pass
            final_linked = False
    except Exception:
        if (
            temporary_identity is None
            and temporary_name is not None
            and descriptor >= 0
        ):
            try:
                opened = os.fstat(descriptor)
                if stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1:
                    temporary_identity = _entry_identity(opened)
            except OSError:
                pass
        try:
            with opened_directory_nofollow(
                selected.parent,
                require_private=True,
            ) as (_stable_parent, directory_descriptor):
                if final_linked and temporary_identity is not None:
                    _unlink_same_entry(
                        directory_descriptor,
                        selected.name,
                        temporary_identity,
                    )
                if temporary_name is not None and temporary_identity is not None:
                    _unlink_same_entry(
                        directory_descriptor,
                        temporary_name,
                        temporary_identity,
                    )
                try:
                    os.fsync(directory_descriptor)
                except OSError:
                    pass
        except BoundedReadError:
            pass
        raise QualificationError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


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
            raise QualificationError()
        return
    if value is None or isinstance(value, (bool, int, float)):
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_path_free(item)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise QualificationError()
            _assert_path_free(key)
            _assert_path_free(item)
        return
    raise QualificationError()


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if type(value) is not str or not value:
            raise QualificationError()
        if value not in result:
            result.append(value)
    return tuple(result)


def _selection_layout(source: fixture.FixtureSource) -> tuple[tuple[str, int], ...]:
    rows = source.selection_recipe.get("strata")
    if not isinstance(rows, list):
        raise QualificationError()
    result: list[tuple[str, int]] = []
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"id", "cases"}
            or type(row.get("id")) is not str
            or type(row.get("cases")) is not int
            or int(row["cases"]) <= 0
        ):
            raise QualificationError()
        result.append((str(row["id"]), int(row["cases"])))
    return tuple(result)


def canonical_stratum_plan(
    fixture_set: fixture.FixtureSet,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the one accepted source/tag plan after checking lock ordering."""

    lock = fixture_set.lock
    if (
        tuple(source.source_id for source in lock.sources) != fixture.SOURCE_IDS
        or tuple(source.source_id for source in fixture_set.sources)
        != fixture.SOURCE_IDS
        or set(_EXPECTED_SELECTION_LAYOUT) != set(fixture.SOURCE_IDS)
    ):
        raise QualificationError()

    plan: list[tuple[str, tuple[str, ...]]] = []
    for source_id in fixture.SOURCE_IDS:
        source = lock.source_by_id(source_id)
        slots = lock.slots_for(source_id)
        if (
            _selection_layout(source) != _EXPECTED_SELECTION_LAYOUT[source_id]
            or tuple(slot.ordinal for slot in slots)
            != tuple(range(fixture.CASES_PER_SOURCE))
            or tuple((slot.stratum, slot.channel) for slot in slots)
            != _EXPECTED_SLOT_LAYOUT[source_id]
        ):
            raise QualificationError()
        slot_strata = _ordered_unique(slot.stratum for slot in slots)
        slot_channels = _ordered_unique(
            slot.channel for slot in slots if slot.channel is not None
        )
        if (
            slot_strata != _EXPECTED_SLOT_STRATA[source_id]
            or slot_channels != _EXPECTED_SLOT_CHANNELS[source_id]
        ):
            raise QualificationError()
        selected = normalize_stratum_tags(
            (
                *slot_strata,
                *(slot_channels if len(slot_channels) > 1 else ()),
            )
        )
        if selected != _EXPECTED_REPORT_STRATA[source_id]:
            raise QualificationError()
        plan.append((source_id, selected))
    return tuple(plan)


def _preflight_corpus_strata(
    fixture_set: fixture.FixtureSet,
    plan: Sequence[tuple[str, tuple[str, ...]]],
) -> tuple[
    Mapping[Path, tuple[str, ...]],
    tuple[tuple[str, tuple[tuple[str, int], ...]], ...],
]:
    bindings = {source.source_id: source for source in fixture_set.sources}
    if tuple(bindings) != fixture.SOURCE_IDS:
        raise QualificationError()
    result: dict[Path, tuple[str, ...]] = {}
    expected_counts: list[tuple[str, tuple[tuple[str, int], ...]]] = []
    for source_id, tags in plan:
        binding = bindings[source_id]
        corpus = load_corpus(binding.corpus_path)
        slots = fixture_set.lock.slots_for(source_id)
        if len(corpus.cases) != len(slots):
            raise QualificationError()
        canonical_tags = frozenset(tags)
        counts = {tag: 0 for tag in tags}
        for case, slot in zip(corpus.cases, slots, strict=True):
            expected_membership = {slot.stratum}
            if source_id == _AMI_SOURCE:
                expected_membership.add(str(slot.channel))
            actual_membership = canonical_tags.intersection(case.tags)
            if actual_membership != expected_membership:
                raise QualificationError()
            for tag in expected_membership:
                counts[tag] += 1
        if validate_stratum_tags(corpus, tags) != tags:
            raise QualificationError()
        slot_counts = {
            tag: sum(
                slot.stratum == tag
                or (source_id == _AMI_SOURCE and slot.channel == tag)
                for slot in slots
            )
            for tag in tags
        }
        if counts != slot_counts or any(count <= 0 for count in counts.values()):
            raise QualificationError()
        verify_corpus_snapshot(corpus)
        result[binding.corpus_path] = tags
        expected_counts.append(
            (source_id, tuple((tag, counts[tag]) for tag in tags))
        )
    return result, tuple(expected_counts)


def _worker_paths(
    baseline_worker_manifest: Path | str,
    candidate_worker_manifest: Path | str,
) -> tuple[Path, Path]:
    result = tuple(
        Path(os.path.abspath(Path(value).expanduser())).resolve(strict=False)
        for value in (baseline_worker_manifest, candidate_worker_manifest)
    )
    if os.path.normcase(str(result[0])) == os.path.normcase(str(result[1])):
        raise QualificationError()
    return result  # type: ignore[return-value]


def _bound_parent(value: object) -> Path:
    try:
        path = Path(getattr(value, "path"))
        parent = path.parent.resolve(strict=True)
        metadata = parent.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise QualificationError()
        return parent
    except Exception:
        raise QualificationError() from None


def _receipt_tree_root(artifact: object) -> Path | None:
    if getattr(artifact, "name", None) not in {
        "runtime-receipt",
        "model-receipt",
    }:
        return None
    try:
        size_bytes = getattr(artifact, "size_bytes")
        digest = getattr(artifact, "sha256")
        if type(size_bytes) is not int or size_bytes <= 0 or type(digest) is not str:
            raise QualificationError()
        snapshot = read_regular_bounded(
            Path(getattr(artifact, "path")),
            maximum_bytes=size_bytes,
            expected_bytes=size_bytes,
        )
        if hashlib.sha256(snapshot.data).hexdigest() != digest:
            raise QualificationError()
        value = json.loads(snapshot.data)
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "root",
            "files",
        }:
            return None
        root_value = value.get("root")
        if type(root_value) is not str:
            raise QualificationError()
        root = Path(root_value)
        resolved = root.resolve(strict=True)
        metadata = resolved.lstat()
        if (
            not root.is_absolute()
            or root != resolved
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            raise QualificationError()
        return resolved
    except QualificationError:
        raise
    except Exception:
        raise QualificationError() from None


def _worker_evidence_roots(manifest: object) -> tuple[Path, ...]:
    # Small generated test doubles carry only a digest. Production manifests
    # expose the complete, already-validated bound-file topology.
    if not hasattr(manifest, "path"):
        return ()
    try:
        roots = {
            Path(getattr(manifest, "path")).parent.resolve(strict=True),
            _bound_parent(getattr(manifest, "worker")),
        }
        python = Path(getattr(getattr(manifest, "python"), "path"))
        venv_root = python.parent.parent.resolve(strict=True)
        if not stat.S_ISDIR(venv_root.lstat().st_mode):
            raise QualificationError()
        roots.add(venv_root)
        artifacts = tuple(getattr(manifest, "artifacts"))
        for artifact in artifacts:
            roots.add(_bound_parent(artifact))
            receipt_root = _receipt_tree_root(artifact)
            if receipt_root is not None:
                roots.add(receipt_root)
        if any(not root.is_absolute() for root in roots):
            raise QualificationError()
        return tuple(sorted(roots, key=str))
    except QualificationError:
        raise
    except Exception:
        raise QualificationError() from None


def _worker_manifest_state(
    workers: Sequence[Path],
) -> tuple[tuple[str, ...], tuple[Path, ...]]:
    try:
        manifests = tuple(load_worker_manifest(path) for path in workers)
    except Exception:
        raise QualificationError() from None
    digests = tuple(manifest.digest for manifest in manifests)
    if (
        len(digests) != len(WORKER_ROLES)
        or len(set(digests)) != len(WORKER_ROLES)
        or any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in digests
        )
    ):
        raise QualificationError()
    roots = tuple(
        sorted(
            {
                root
                for manifest in manifests
                for root in _worker_evidence_roots(manifest)
            },
            key=str,
        )
    )
    return digests, roots


def _validate_two_by_four_result(
    value: object,
    *,
    fixture_set: fixture.FixtureSet,
    worker_manifest_digests: Sequence[str],
    stratum_plan: Sequence[tuple[str, tuple[str, ...]]],
    expected_stratum_counts: Sequence[
        tuple[str, tuple[tuple[str, int], ...]]
    ],
    repeats: int,
    chunk_samples: int | None,
    pace: str | None,
    partial_interval_ms: int | None,
    tail_padding_samples: int | None,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"ok", "fixture", "suite"}:
        raise QualificationError()
    suite = value.get("suite")
    if not isinstance(suite, Mapping):
        raise QualificationError()
    try:
        fixture._validate_suite_binding(suite, fixture_set=fixture_set)
    except Exception:
        raise QualificationError() from None
    controller = suite.get("controller")
    config = suite.get("config")
    runs = suite.get("runs")
    expected_fixture = {
        "fixture_id": fixture.FIXTURE_ID,
        "lock_recipe_sha256": fixture_set.lock.recipe_sha256,
        "fixture_set_sha256": fixture_set.fixture_set_sha256,
        "sources": len(fixture_set.sources),
        "cases": fixture_set.cases,
        "pcm_bytes": fixture_set.pcm_bytes,
        "validation": "before_and_after_suite",
    }
    if (
        type(value.get("ok")) is not bool
        or value.get("fixture") != expected_fixture
        or type(suite.get("ok")) is not bool
        or suite.get("complete") is not True
        or not isinstance(controller, Mapping)
        or not isinstance(config, Mapping)
        or not isinstance(runs, list)
        or controller.get("execution")
        != "strictly_sequential_cross_product"
        or controller.get("planned_worker_manifests") != len(WORKER_ROLES)
        or controller.get("worker_manifests") != len(WORKER_ROLES)
        or controller.get("planned_corpora") != len(fixture.SOURCE_IDS)
        or controller.get("corpora") != len(fixture.SOURCE_IDS)
        or len(runs) != len(WORKER_ROLES) * len(fixture.SOURCE_IDS)
    ):
        raise QualificationError()
    config_rows = [
        {
            "corpus_index": index,
            "tags": list(tags),
            "selection_sha256": _canonical_sha256(list(tags)),
        }
        for index, (_source_id, tags) in enumerate(stratum_plan)
    ]
    expected_strata_config = {
        "mode": "per_corpus",
        "corpora": config_rows,
        "selection_sha256": _canonical_sha256(config_rows),
    }
    expected_config: dict[str, object] = {
        "repeats": repeats,
        "strata": expected_strata_config,
        "overrides": {
            "chunk_samples": chunk_samples,
            "pace": pace,
            "partial_interval_ms": partial_interval_ms,
            "tail_padding_samples": tail_padding_samples,
        },
    }
    expected_config["contract_sha256"] = _canonical_sha256(expected_config)
    corpus_digests = tuple(
        source.corpus_manifest_sha256 for source in fixture_set.sources
    )
    expected_worker_set = _canonical_sha256(list(worker_manifest_digests))
    expected_corpus_set = _canonical_sha256(list(corpus_digests))
    evaluator_digest = controller.get("evaluator_implementation_sha256")
    if (
        config != expected_config
        or controller.get("stratum_selection_sha256")
        != expected_strata_config["selection_sha256"]
        or controller.get("worker_set_sha256") != expected_worker_set
        or controller.get("completed_worker_set_sha256") != expected_worker_set
        or controller.get("corpus_set_sha256") != expected_corpus_set
        or controller.get("completed_corpus_set_sha256") != expected_corpus_set
        or type(evaluator_digest) is not str
        or len(evaluator_digest) != 64
        or any(character not in "0123456789abcdef" for character in evaluator_digest)
        or tuple(source_id for source_id, _counts in expected_stratum_counts)
        != fixture.SOURCE_IDS
    ):
        raise QualificationError()
    count_plan = {
        source_id: dict(counts)
        for source_id, counts in expected_stratum_counts
    }
    expected_order = [
        (worker_index, corpus_index)
        for worker_index in range(len(WORKER_ROLES))
        for corpus_index in range(len(fixture.SOURCE_IDS))
    ]
    observed_order: list[tuple[object, object]] = []
    observed_bindings: list[Mapping[str, object]] = []
    run_success: list[bool] = []
    for run in runs:
        if not isinstance(run, Mapping) or set(run) != {"binding", "report"}:
            raise QualificationError()
        binding = run.get("binding")
        report = run.get("report")
        worker = report.get("worker") if isinstance(report, Mapping) else None
        corpus = report.get("corpus") if isinstance(report, Mapping) else None
        metrics = report.get("metrics") if isinstance(report, Mapping) else None
        if (
            not isinstance(binding, Mapping)
            or set(binding)
            != {
                "worker_index",
                "corpus_index",
                "worker_manifest_sha256",
                "corpus_manifest_sha256",
                "stratum_selection_sha256",
                "evaluator_implementation_sha256",
                "report_sha256",
                "binding_sha256",
            }
            or not isinstance(report, Mapping)
            or type(report.get("ok")) is not bool
            or not isinstance(worker, Mapping)
            or not isinstance(corpus, Mapping)
            or not isinstance(metrics, Mapping)
        ):
            raise QualificationError()
        worker_index = binding.get("worker_index")
        if (
            type(worker_index) is not int
            or not 0 <= worker_index < len(worker_manifest_digests)
            or worker.get("manifest_sha256")
            != worker_manifest_digests[worker_index]
        ):
            raise QualificationError()
        corpus_index = binding.get("corpus_index")
        if type(corpus_index) is not int or not 0 <= corpus_index < len(stratum_plan):
            raise QualificationError()
        source_id, tags = stratum_plan[corpus_index]
        binding_contract = dict(binding)
        binding_sha256 = binding_contract.pop("binding_sha256")
        if (
            binding.get("worker_manifest_sha256")
            != worker_manifest_digests[worker_index]
            or binding.get("corpus_manifest_sha256")
            != corpus_digests[corpus_index]
            or binding.get("stratum_selection_sha256")
            != _canonical_sha256(list(tags))
            or binding.get("evaluator_implementation_sha256") != evaluator_digest
            or binding.get("report_sha256") != _canonical_sha256(report)
            or binding_sha256 != _canonical_sha256(binding_contract)
            or corpus.get("manifest_sha256") != corpus_digests[corpus_index]
        ):
            raise QualificationError()
        strata = metrics.get("strata")
        if not isinstance(strata, list) or len(strata) != len(tags):
            raise QualificationError()
        for tag, row in zip(tags, strata, strict=True):
            if (
                not isinstance(row, Mapping)
                or row.get("tag") != tag
                or row.get("cases") != count_plan[source_id][tag]
                or not isinstance(row.get("metrics"), Mapping)
            ):
                raise QualificationError()
        observed_order.append((worker_index, corpus_index))
        observed_bindings.append(binding)
        run_success.append(report["ok"] is True)
    expected_matrix = _canonical_sha256(observed_bindings)
    if observed_order != expected_order:
        raise QualificationError()
    expected_ok = all(run_success)
    if (
        suite.get("ok") is not expected_ok
        or value.get("ok") is not expected_ok
        or controller.get("matrix_sha256") != expected_matrix
        or controller.get("completed_matrix_sha256") != expected_matrix
    ):
        raise QualificationError()
    _assert_path_free(value)
    return value


def _contract(
    fixture_set: fixture.FixtureSet,
    plan: Sequence[tuple[str, tuple[str, ...]]],
    *,
    implementation_sha256: str,
) -> dict[str, object]:
    source_strata = [
        {"source_id": source_id, "tags": list(tags)}
        for source_id, tags in plan
    ]
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "implementation_sha256": implementation_sha256,
        "fixture_id": fixture.FIXTURE_ID,
        "lock_recipe_sha256": fixture_set.lock.recipe_sha256,
        "worker_roles": list(WORKER_ROLES),
        "source_strata": source_strata,
        "execution": "strictly_sequential_two_by_four",
        "quality_semantics": "coverage_only_no_thresholds",
        "development_only": True,
        "promotional": False,
    }
    value["contract_sha256"] = _canonical_sha256(value)
    return value


def run_qualification(
    *,
    baseline_worker_manifest: Path | str,
    candidate_worker_manifest: Path | str,
    corpus_paths: Iterable[Path | str],
    scratch_parent: Path | str,
    output_path: Path | str,
    lock_path: Path | str = fixture.DEFAULT_LOCK,
    repeats: int = 3,
    chunk_samples: int | None = None,
    pace: str | None = None,
    partial_interval_ms: int | None = None,
    tail_padding_samples: int | None = None,
    run_validated_suite_fn: RunValidatedSuite | None = None,
) -> dict[str, object]:
    """Validate, preflight all canonical strata, and run baseline then candidate."""

    try:
        implementation_sha256 = _implementation_sha256()
        final_output = _new_private_output_path(output_path)
        scratch = _scratch_candidate(scratch_parent)
        suite_checkpoint = scratch / _INNER_CHECKPOINT_NAME
        fixture_set = fixture.validate_fixture_set(
            corpus_paths,
            lock_path=lock_path,
        )
        plan = canonical_stratum_plan(fixture_set)
        corpus_stratum_tags, expected_stratum_counts = _preflight_corpus_strata(
            fixture_set,
            plan,
        )
        workers = _worker_paths(
            baseline_worker_manifest,
            candidate_worker_manifest,
        )
        worker_manifest_digests, worker_evidence_roots = _worker_manifest_state(
            workers
        )
        _validate_private_paths(
            final_output=final_output,
            scratch=scratch,
            suite_checkpoint=suite_checkpoint,
            fixture_set=fixture_set,
            worker_evidence_roots=worker_evidence_roots,
        )
        scratch, suite_checkpoint = _prepare_private_scratch(scratch)
        runner = (
            fixture.run_validated_suite
            if run_validated_suite_fn is None
            else run_validated_suite_fn
        )
        if not callable(runner):
            raise QualificationError()
        raw_result = runner(
            workers,
            fixture_set.corpus_paths,
            scratch_parent=scratch,
            lock_path=lock_path,
            repeats=repeats,
            chunk_samples=chunk_samples,
            pace=pace,
            partial_interval_ms=partial_interval_ms,
            tail_padding_samples=tail_padding_samples,
            stratum_tags=(),
            corpus_stratum_tags=corpus_stratum_tags,
            checkpoint_path=suite_checkpoint,
        )
        bound = _validate_two_by_four_result(
            raw_result,
            fixture_set=fixture_set,
            worker_manifest_digests=worker_manifest_digests,
            stratum_plan=plan,
            expected_stratum_counts=expected_stratum_counts,
            repeats=repeats,
            chunk_samples=chunk_samples,
            pace=pace,
            partial_interval_ms=partial_interval_ms,
            tail_padding_samples=tail_padding_samples,
        )
        if (
            _implementation_sha256() != implementation_sha256
            or _worker_manifest_state(workers)
            != (worker_manifest_digests, worker_evidence_roots)
        ):
            raise QualificationError()
        fixture.verify_fixture_set_snapshot(fixture_set)
        _validate_private_paths(
            final_output=final_output,
            scratch=scratch,
            suite_checkpoint=suite_checkpoint,
            fixture_set=fixture_set,
            worker_evidence_roots=worker_evidence_roots,
        )
        if _new_private_output_path(final_output) != final_output:
            raise QualificationError()
        contract = _contract(
            fixture_set,
            plan,
            implementation_sha256=implementation_sha256,
        )
        result = {
            "ok": bound["ok"],
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "execution_complete": True,
            "execution_semantics": "coverage_only",
            "quality_decision": "not_evaluated",
            "development_only": True,
            "promotional": False,
            "contract": contract,
            "worker_roles": [
                {
                    "role": role,
                    "worker_index": index,
                    "manifest_sha256": worker_manifest_digests[index],
                }
                for index, role in enumerate(WORKER_ROLES)
            ],
            "fixture": bound["fixture"],
            "suite": bound["suite"],
        }
        _assert_path_free(result)
        return result
    except QualificationError:
        raise
    except Exception:
        raise QualificationError() from None


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise QualificationError()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description=(
            "Run the exact development-only public-conversation baseline/candidate "
            "matrix with mandatory source-specific strata."
        )
    )
    parser.add_argument("--lock", type=Path, default=fixture.DEFAULT_LOCK)
    parser.add_argument("--corpus", action="append", type=Path, required=True)
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
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk-samples", type=int)
    parser.add_argument("--pace", choices=("burst", "realtime"))
    parser.add_argument("--partial-interval-ms", type=int)
    parser.add_argument("--tail-padding-samples", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        result = run_qualification(
            baseline_worker_manifest=args.baseline_worker_manifest,
            candidate_worker_manifest=args.candidate_worker_manifest,
            corpus_paths=tuple(args.corpus),
            scratch_parent=args.scratch_root,
            output_path=args.output,
            lock_path=args.lock,
            repeats=args.repeats,
            chunk_samples=args.chunk_samples,
            pace=args.pace,
            partial_interval_ms=args.partial_interval_ms,
            tail_padding_samples=args.tail_padding_samples,
        )
        write_new_report(args.output, result)
        summary = {
            "ok": result["ok"],
            "kind": KIND,
            "execution_complete": result["execution_complete"],
            "quality_decision": result["quality_decision"],
            "development_only": True,
            "promotional": False,
            "report_written": True,
        }
        print(_canonical_json_bytes(summary).decode("ascii"))
        return 0 if result["ok"] is True else 1
    except Exception:
        print(_canonical_json_bytes(_SAFE_ERROR).decode("ascii"), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "KIND",
    "QualificationError",
    "SCHEMA_VERSION",
    "WORKER_ROLES",
    "canonical_stratum_plan",
    "main",
    "run_qualification",
    "write_new_report",
]
