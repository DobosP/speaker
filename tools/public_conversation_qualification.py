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
import math
import os
import secrets
import stat
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PureWindowsPath

from tools import public_conversation_fixture as fixture
from tools import streaming_stt_eval, streaming_stt_suite
from tools.streaming_stt.bounded_io import (
    hash_regular_bounded,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.corpus import load_corpus, verify_corpus_snapshot
from tools.streaming_stt.metrics import (
    normalize_stratum_tags,
    validate_stratum_tags,
)
from tools.streaming_stt.manifest import (
    FASTER_WHISPER_ENDPOINT_ADAPTER,
    MOONSHINE_ADAPTER,
    MOONSHINE_EXTERNAL_ENDPOINT_ADAPTER,
    NEMOTRON_ADAPTER,
    PARAKEET_CPP_ADAPTER,
    PARAKEET_REALTIME_EOU_ADAPTER,
    SHERPA_ZIPFORMER_ADAPTER,
    load_worker_manifest,
)
from tools.streaming_stt.protocol import MAX_PARAKEET_CPP_OBSERVED_EVENTS

SCHEMA_VERSION = 2
KIND = "public-conversation-canonical-strata-qualification-v2"
WORKER_ROLES = ("baseline", "candidate")

_SAFE_SUITE_SCHEMA_VERSION = 1
_SAFE_SUITE_KIND = "public-conversation-safe-aggregate-suite-v1"
_ALLOWED_ADAPTERS = {
    "fake-json-v1",
    FASTER_WHISPER_ENDPOINT_ADAPTER,
    MOONSHINE_ADAPTER,
    MOONSHINE_EXTERNAL_ENDPOINT_ADAPTER,
    NEMOTRON_ADAPTER,
    PARAKEET_CPP_ADAPTER,
    PARAKEET_REALTIME_EOU_ADAPTER,
    SHERPA_ZIPFORMER_ADAPTER,
}

_SAFE_ERROR = {"error": "public conversation qualification failed", "ok": False}
_MAX_IMPLEMENTATION_BYTES = 256 * 1024
_MAX_REPORT_BYTES = 64 * 1024 * 1024
# Matches the generic suite's complete JSON-plus-newline publication bound.
_MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024
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
_EXPECTED_REPORT_STRATUM_COUNTS = {
    source_id: tuple(
        (
            tag,
            sum(
                slot_stratum == tag or slot_channel == tag
                for slot_stratum, slot_channel in _EXPECTED_SLOT_LAYOUT[source_id]
            ),
        )
        for tag in _EXPECTED_REPORT_STRATA[source_id]
    )
    for source_id in fixture.SOURCE_IDS
}
_PRODUCTION_RUNNER = fixture.run_validated_suite


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


def _entry_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_uid,
    )


def _private_directory(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(metadata.st_mode)
        and stat.S_IMODE(metadata.st_mode) == 0o700
        and (
            not hasattr(os, "geteuid")
            or metadata.st_uid == os.geteuid()
        )
    )


def _private_regular(
    metadata: os.stat_result,
    *,
    maximum_bytes: int,
) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and stat.S_IMODE(metadata.st_mode) == 0o600
        and metadata.st_nlink == 1
        and 0 < metadata.st_size <= maximum_bytes
        and (
            not hasattr(os, "geteuid")
            or metadata.st_uid == os.geteuid()
        )
    )


def _read_descriptor_bounded(descriptor: int, maximum_bytes: int) -> bytes:
    if type(descriptor) is not int or descriptor < 0 or maximum_bytes <= 0:
        raise QualificationError()
    chunks: list[bytes] = []
    remaining = maximum_bytes
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _prepare_private_scratch(
    scratch: Path,
) -> tuple[Path, Path, tuple[int, int, int, int]]:
    try:
        scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
        scratch = scratch.resolve(strict=True)
        metadata = scratch.lstat()
        if not _private_directory(metadata):
            raise QualificationError()
        checkpoint = scratch / _INNER_CHECKPOINT_NAME
    except QualificationError:
        raise
    except Exception:
        raise QualificationError() from None
    try:
        checkpoint.lstat()
    except FileNotFoundError:
        return scratch, checkpoint, _entry_identity(metadata)
    except Exception:
        raise QualificationError() from None
    raise QualificationError()


def _checkpoint_snapshot(
    *,
    scratch: Path,
    checkpoint: Path,
    scratch_identity: tuple[int, int, int, int],
    required: bool,
) -> object | None:
    descriptor = -1
    try:
        with opened_directory_nofollow(
            scratch,
            require_private=True,
        ) as (stable_scratch, directory_descriptor):
            directory_before = os.fstat(directory_descriptor)
            if (
                stable_scratch != scratch
                or checkpoint.parent != scratch
                or checkpoint.name != _INNER_CHECKPOINT_NAME
                or _entry_identity(directory_before) != scratch_identity
                or not _private_directory(directory_before)
                or _entry_identity(scratch.lstat()) != scratch_identity
                or not _private_directory(scratch.lstat())
            ):
                raise QualificationError()
            try:
                before = os.stat(
                    checkpoint.name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if required:
                    raise QualificationError() from None
                return None
            if not _private_regular(before, maximum_bytes=_MAX_CHECKPOINT_BYTES):
                raise QualificationError()
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(
                checkpoint.name,
                flags,
                dir_fd=directory_descriptor,
            )
            opened = os.fstat(descriptor)
            if (
                _entry_identity(opened) != _entry_identity(before)
                or not _private_regular(
                    opened,
                    maximum_bytes=_MAX_CHECKPOINT_BYTES,
                )
            ):
                raise QualificationError()
            data = _read_descriptor_bounded(descriptor, opened.st_size + 1)
            after = os.fstat(descriptor)
            current = os.stat(
                checkpoint.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            directory_after = os.fstat(directory_descriptor)
            if (
                len(data) != opened.st_size
                or _entry_identity(after) != _entry_identity(opened)
                or _entry_identity(current) != _entry_identity(opened)
                or after.st_size != opened.st_size
                or current.st_size != opened.st_size
                or not _private_regular(
                    after,
                    maximum_bytes=_MAX_CHECKPOINT_BYTES,
                )
                or not _private_regular(
                    current,
                    maximum_bytes=_MAX_CHECKPOINT_BYTES,
                )
                or _entry_identity(directory_after) != scratch_identity
                or not _private_directory(directory_after)
                or _entry_identity(scratch.lstat()) != scratch_identity
                or not _private_directory(scratch.lstat())
            ):
                raise QualificationError()
        try:
            value = json.loads(data.decode("ascii"))
        except (UnicodeError, ValueError, RecursionError):
            raise QualificationError() from None
        if not isinstance(value, Mapping):
            raise QualificationError()
        _assert_path_free(value)
        return value
    except QualificationError:
        raise
    except Exception:
        raise QualificationError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


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


def write_new_report(path: Path | str, payload: Mapping[str, object]) -> str:
    """Failure-atomically publish one report without replacing an existing file."""

    selected = _new_private_output_path(path)
    _assert_path_free(payload)
    encoded = _canonical_json_bytes(payload) + b"\n"
    if len(encoded) > _MAX_REPORT_BYTES:
        raise QualificationError()
    report_sha256 = hashlib.sha256(encoded).hexdigest()

    descriptor = -1
    temporary_name: str | None = None
    temporary_identity: tuple[int, int, int, int] | None = None
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

            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
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
            os.lseek(descriptor, 0, os.SEEK_SET)
            observed = _read_descriptor_bounded(descriptor, len(encoded) + 1)
            readback = os.fstat(descriptor)
            if (
                observed != encoded
                or hashlib.sha256(observed).hexdigest() != report_sha256
                or _entry_identity(readback) != temporary_identity
                or readback.st_size != len(encoded)
                or not _private_regular(
                    readback,
                    maximum_bytes=_MAX_REPORT_BYTES,
                )
            ):
                raise QualificationError()

            os.link(
                temporary_name,
                selected.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
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
        return report_sha256
    except BaseException:
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
                if temporary_identity is not None:
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
        except BaseException:
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


def _sha256_field(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise QualificationError()
    return value


def _count(value: object) -> int:
    if type(value) is not int or value < 0:
        raise QualificationError()
    return value


def _number_or_none(
    value: object,
    *,
    nonnegative: bool = True,
) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError()
    selected = float(value)
    if not math.isfinite(selected) or (nonnegative and selected < 0.0):
        raise QualificationError()
    return value


def _ordered_numbers(
    values: Sequence[int | float | None],
    *,
    allow_all_none: bool,
) -> None:
    present = tuple(value for value in values if value is not None)
    if (not present and allow_all_none) or (
        len(present) == len(values)
        and all(
            float(left) <= float(right)
            for left, right in zip(present, present[1:])
        )
    ):
        return
    raise QualificationError()


def _transcript_snapshot(
    value: object,
    *,
    evaluations: int,
) -> dict[str, int | float]:
    fields = {
        "clips",
        "nonempty",
        "exact",
        "wer",
        "cer",
        "word_errors",
        "substitutions",
        "insertions",
        "deletions",
        "ref_words",
        "hyp_words",
        "char_edits",
        "ref_chars",
        "hyp_chars",
        "keyword_attempts",
        "keyword_hits",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise QualificationError()
    counts = {
        key: _count(value.get(key))
        for key in fields - {"wer", "cer"}
    }
    wer = _number_or_none(value.get("wer"))
    cer = _number_or_none(value.get("cer"))
    expected_wer = (
        0.0
        if counts["ref_words"] == counts["hyp_words"] == 0
        else (
            1.0
            if counts["ref_words"] == 0
            else round(counts["word_errors"] / counts["ref_words"], 4)
        )
    )
    expected_cer = (
        0.0
        if counts["ref_chars"] == counts["hyp_chars"] == 0
        else (
            1.0
            if counts["ref_chars"] == 0
            else round(counts["char_edits"] / counts["ref_chars"], 4)
        )
    )
    if (
        wer is None
        or cer is None
        or counts["clips"] != evaluations
        or counts["exact"] > counts["nonempty"]
        or counts["nonempty"] > evaluations
        or counts["word_errors"]
        != counts["substitutions"]
        + counts["insertions"]
        + counts["deletions"]
        or counts["keyword_hits"] > counts["keyword_attempts"]
        or float(wer) != expected_wer
        or float(cer) != expected_cer
    ):
        raise QualificationError()
    return {
        key: value[key]
        for key in (
            "clips",
            "nonempty",
            "exact",
            "wer",
            "cer",
            "word_errors",
            "substitutions",
            "insertions",
            "deletions",
            "ref_words",
            "hyp_words",
            "char_edits",
            "ref_chars",
            "hyp_chars",
            "keyword_attempts",
            "keyword_hits",
        )
    }


def _accuracy_snapshot(
    value: object,
    *,
    evaluations: int,
) -> dict[str, object]:
    fields = {
        "transcript",
        "command_attempts",
        "command_hits",
        "command_recall",
        "silence_nonempty_partials",
        "silence_nonempty_finals",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise QualificationError()
    attempts = _count(value.get("command_attempts"))
    hits = _count(value.get("command_hits"))
    partials = _count(value.get("silence_nonempty_partials"))
    finals = _count(value.get("silence_nonempty_finals"))
    recall = _number_or_none(value.get("command_recall"))
    expected_recall = round(hits / attempts, 4) if attempts else None
    if (
        hits > attempts
        or recall != expected_recall
        or finals > evaluations
    ):
        raise QualificationError()
    return {
        "transcript": _transcript_snapshot(
            value.get("transcript"),
            evaluations=evaluations,
        ),
        "command_attempts": attempts,
        "command_hits": hits,
        "command_recall": recall,
        "silence_nonempty_partials": partials,
        "silence_nonempty_finals": finals,
    }


def _latency_snapshot(value: object, *, evaluations: int) -> dict[str, object]:
    fields = {
        "model_load_ms",
        "first_nonempty_partial_p50_ms",
        "first_nonempty_partial_p95_ms",
        "stable_partial_p50_ms",
        "stable_partial_p95_ms",
        "stable_partial_missing",
        "finalization_p50_ms",
        "finalization_p95_ms",
        "finalization_max_ms",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise QualificationError()
    selected = {
        key: _number_or_none(value.get(key))
        for key in fields - {"stable_partial_missing"}
    }
    missing = _count(value.get("stable_partial_missing"))
    if selected["model_load_ms"] is None or missing > evaluations:
        raise QualificationError()
    _ordered_numbers(
        (
            selected["first_nonempty_partial_p50_ms"],
            selected["first_nonempty_partial_p95_ms"],
        ),
        allow_all_none=True,
    )
    _ordered_numbers(
        (
            selected["stable_partial_p50_ms"],
            selected["stable_partial_p95_ms"],
        ),
        allow_all_none=True,
    )
    _ordered_numbers(
        (
            selected["finalization_p50_ms"],
            selected["finalization_p95_ms"],
            selected["finalization_max_ms"],
        ),
        allow_all_none=False,
    )
    return {
        "model_load_ms": selected["model_load_ms"],
        "first_nonempty_partial_p50_ms": selected[
            "first_nonempty_partial_p50_ms"
        ],
        "first_nonempty_partial_p95_ms": selected[
            "first_nonempty_partial_p95_ms"
        ],
        "stable_partial_p50_ms": selected["stable_partial_p50_ms"],
        "stable_partial_p95_ms": selected["stable_partial_p95_ms"],
        "stable_partial_missing": missing,
        "finalization_p50_ms": selected["finalization_p50_ms"],
        "finalization_p95_ms": selected["finalization_p95_ms"],
        "finalization_max_ms": selected["finalization_max_ms"],
    }


def _streaming_snapshot(value: object) -> dict[str, object]:
    base_fields = {
        "partial_events",
        "churn_token_edits",
        "retracted_words",
        "deadline_misses",
        "max_backlog_p95_ms",
        "max_backlog_ms",
        "model_padding_total_samples",
        "model_padding_max_samples",
    }
    if not isinstance(value, Mapping) or frozenset(value) not in {
        frozenset(base_fields),
        frozenset((*base_fields, "deadline_metrics_applicable")),
    }:
        raise QualificationError()
    applicable = value.get("deadline_metrics_applicable", True)
    if type(applicable) is not bool:
        raise QualificationError()
    counts = {
        key: _count(value.get(key))
        for key in (
            "partial_events",
            "churn_token_edits",
            "retracted_words",
            "model_padding_total_samples",
            "model_padding_max_samples",
        )
    }
    deadline_misses = value.get("deadline_misses")
    backlog_p95 = _number_or_none(value.get("max_backlog_p95_ms"))
    backlog_max = _number_or_none(value.get("max_backlog_ms"))
    if applicable:
        deadline_misses = _count(deadline_misses)
        _ordered_numbers(
            (backlog_p95, backlog_max),
            allow_all_none=False,
        )
    elif deadline_misses is not None or backlog_p95 is not None or backlog_max is not None:
        raise QualificationError()
    if counts["model_padding_max_samples"] > counts[
        "model_padding_total_samples"
    ]:
        raise QualificationError()
    result: dict[str, object] = {
        "partial_events": counts["partial_events"],
        "churn_token_edits": counts["churn_token_edits"],
        "retracted_words": counts["retracted_words"],
        "deadline_misses": deadline_misses,
        "max_backlog_p95_ms": backlog_p95,
        "max_backlog_ms": backlog_max,
        "model_padding_total_samples": counts["model_padding_total_samples"],
        "model_padding_max_samples": counts["model_padding_max_samples"],
    }
    if "deadline_metrics_applicable" in value:
        result["deadline_metrics_applicable"] = applicable
    return result


def _throughput_snapshot(value: object) -> dict[str, object]:
    base_fields = {"audio_total_sec", "compute_total_ms", "aggregate_rtf"}
    endpoint_fields = {
        "aggregate_rtf_scope",
        "model_input_total_sec",
        "model_input_aggregate_rtf",
    }
    if not isinstance(value, Mapping) or frozenset(value) not in {
        frozenset(base_fields),
        frozenset((*base_fields, *endpoint_fields)),
    }:
        raise QualificationError()
    result: dict[str, object] = {
        key: _number_or_none(value.get(key)) for key in base_fields
    }
    if any(item is None for item in result.values()):
        raise QualificationError()
    if endpoint_fields <= set(value):
        if value.get("aggregate_rtf_scope") != "declared_source_audio":
            raise QualificationError()
        result.update(
            {
                "aggregate_rtf_scope": "declared_source_audio",
                "model_input_total_sec": _number_or_none(
                    value.get("model_input_total_sec")
                ),
                "model_input_aggregate_rtf": _number_or_none(
                    value.get("model_input_aggregate_rtf")
                ),
            }
        )
        if (
            result["model_input_total_sec"] is None
            or result["model_input_aggregate_rtf"] is None
        ):
            raise QualificationError()
    return result


def _determinism_snapshot(
    value: object,
    *,
    cases: int,
    repeats: int,
) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != {
        "repeats",
        "cases_with_final_disagreement",
    }:
        raise QualificationError()
    observed_repeats = _count(value.get("repeats"))
    disagreements = _count(value.get("cases_with_final_disagreement"))
    if observed_repeats != repeats or disagreements > cases:
        raise QualificationError()
    return {
        "repeats": observed_repeats,
        "cases_with_final_disagreement": disagreements,
    }


def _resources_snapshot(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "max_reported_rss_mb",
        "peak_threads",
        "peak_vram_mb",
        "source",
        "rss_scope",
    }:
        raise QualificationError()
    rss = _number_or_none(value.get("max_reported_rss_mb"))
    vram = _number_or_none(value.get("peak_vram_mb"))
    threads = _count(value.get("peak_threads"))
    if (
        rss is None
        or threads < 1
        or value.get("source") != "worker_ready_and_final_resource_reports"
        or value.get("rss_scope")
        != "maximum_of_ready_and_final_samples_not_process_peak"
    ):
        raise QualificationError()
    return {
        "max_reported_rss_mb": rss,
        "peak_threads": threads,
        "peak_vram_mb": vram,
        "source": "worker_ready_and_final_resource_reports",
        "rss_scope": "maximum_of_ready_and_final_samples_not_process_peak",
    }


def _sample_totals(value: object, *, fields: set[str]) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise QualificationError()
    return {key: _count(value.get(key)) for key in fields}


def _endpointing_snapshot(
    value: object,
    *,
    expected_evaluations: int,
    expected_adapter: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise QualificationError()
    native_count_fields = {
        "evaluations",
        "native_endpoints",
        "native_eou_events",
        "native_eob_backchannels",
        "authoritative_endpoints",
        "early_source_endpoints",
        "source_early_native_events",
        "source_early_eou_events",
        "source_early_eob_backchannels",
        "tail_exhaustions",
        "terminal_model_padding_samples_total",
        "model_input_samples_consumed_total",
    }
    native_number_fields = {
        "endpoint_sample_p50",
        "endpoint_sample_p95",
        "authoritative_endpoint_latency_p50_ms",
        "authoritative_endpoint_latency_p95_ms",
        "authoritative_endpoint_latency_max_ms",
        "endpoint_probability_p50",
        "endpoint_probability_p95",
    }
    native_fields = native_count_fields | native_number_fields | {
        "reasons",
        "source_samples",
        "declared_tail_samples",
    }
    cpp_count_fields = {
        "evaluations",
        "accepted_endpoints",
        "tail_exhaustions",
        "terminal_model_padding_samples_total",
        "model_input_samples_consumed_total",
    }
    cpp_number_fields = {
        "accepted_endpoint_sample_p50",
        "accepted_endpoint_sample_p95",
        "observed_encoder_frame_p50",
        "observed_encoder_frame_p95",
        "observed_encoder_time_p50_seconds",
        "observed_encoder_time_p95_seconds",
        "accepted_endpoint_latency_p50_ms",
        "accepted_endpoint_latency_p95_ms",
        "accepted_endpoint_latency_max_ms",
    }
    cpp_fields = cpp_count_fields | cpp_number_fields | {
        "terminal_reasons",
        "terminal_origins",
        "observations",
        "source_samples",
        "tail_samples",
    }
    if expected_adapter == PARAKEET_REALTIME_EOU_ADAPTER and set(value) == native_fields:
        counts = {key: _count(value.get(key)) for key in native_count_fields}
        reasons = _sample_totals(
            value.get("reasons"),
            fields={"eou", "eob", "tail_exhausted"},
        )
        source = _sample_totals(
            value.get("source_samples"),
            fields={"offered_total", "consumed_total", "dropped_total"},
        )
        tail = _sample_totals(
            value.get("declared_tail_samples"),
            fields={"offered_total", "consumed_total", "unconsumed_total"},
        )
        numbers = {
            key: _number_or_none(value.get(key)) for key in native_number_fields
        }
        if (
            counts["evaluations"] != expected_evaluations
            or counts["native_eou_events"] != reasons["eou"]
            or counts["native_eob_backchannels"] != reasons["eob"]
            or counts["native_endpoints"] != reasons["eou"] + reasons["eob"]
            or sum(reasons.values()) != expected_evaluations
            or counts["tail_exhaustions"] != reasons["tail_exhausted"]
            or counts["authoritative_endpoints"] > reasons["eou"]
            or counts["early_source_endpoints"]
            != counts["source_early_native_events"]
            or counts["source_early_native_events"]
            != counts["source_early_eou_events"]
            + counts["source_early_eob_backchannels"]
            or counts["authoritative_endpoints"]
            + counts["source_early_eou_events"]
            != counts["native_eou_events"]
            or counts["source_early_eob_backchannels"]
            > counts["native_eob_backchannels"]
            or source["offered_total"]
            != source["consumed_total"] + source["dropped_total"]
            or tail["offered_total"]
            != tail["consumed_total"] + tail["unconsumed_total"]
            or counts["model_input_samples_consumed_total"]
            != source["consumed_total"]
            + tail["consumed_total"]
            + counts["terminal_model_padding_samples_total"]
        ):
            raise QualificationError()
        endpoint_samples = (
            numbers["endpoint_sample_p50"],
            numbers["endpoint_sample_p95"],
        )
        endpoint_probabilities = (
            numbers["endpoint_probability_p50"],
            numbers["endpoint_probability_p95"],
        )
        authoritative_latencies = (
            numbers["authoritative_endpoint_latency_p50_ms"],
            numbers["authoritative_endpoint_latency_p95_ms"],
            numbers["authoritative_endpoint_latency_max_ms"],
        )
        if counts["native_endpoints"]:
            _ordered_numbers(endpoint_samples, allow_all_none=False)
            _ordered_numbers(endpoint_probabilities, allow_all_none=False)
            if any(
                probability is None or float(probability) > 1.0
                for probability in endpoint_probabilities
            ):
                raise QualificationError()
        elif any(item is not None for item in (*endpoint_samples, *endpoint_probabilities)):
            raise QualificationError()
        if counts["authoritative_endpoints"]:
            _ordered_numbers(authoritative_latencies, allow_all_none=False)
        elif any(item is not None for item in authoritative_latencies):
            raise QualificationError()
    elif expected_adapter == PARAKEET_CPP_ADAPTER and set(value) == cpp_fields:
        counts = {key: _count(value.get(key)) for key in cpp_count_fields}
        reasons = _sample_totals(
            value.get("terminal_reasons"),
            fields={"eou", "tail_exhausted"},
        )
        origins = _sample_totals(
            value.get("terminal_origins"),
            fields={"feed", "none"},
        )
        observations = value.get("observations")
        if not isinstance(observations, Mapping) or set(observations) != {
            "total",
            "eou",
            "eob",
            "origins",
            "source_early",
            "post_source_feed",
        }:
            raise QualificationError()
        observation_counts = {
            key: _count(observations.get(key))
            for key in ("total", "eou", "eob")
        }
        observation_origins = _sample_totals(
            observations.get("origins"),
            fields={"feed", "finalize"},
        )
        source_early = _sample_totals(
            observations.get("source_early"),
            fields={"total", "eou", "eob"},
        )
        post_source = _sample_totals(
            observations.get("post_source_feed"),
            fields={"total", "eou", "eob"},
        )
        source = _sample_totals(
            value.get("source_samples"),
            fields={"offered_total", "consumed_total", "dropped_total"},
        )
        tail = _sample_totals(
            value.get("tail_samples"),
            fields={"offered_total", "consumed_total", "unconsumed_total"},
        )
        numbers = {
            key: _number_or_none(value.get(key)) for key in cpp_number_fields
        }
        if (
            counts["evaluations"] != expected_evaluations
            or sum(reasons.values()) != expected_evaluations
            or sum(origins.values()) != expected_evaluations
            or counts["accepted_endpoints"] != reasons["eou"]
            or counts["accepted_endpoints"] != origins["feed"]
            or counts["tail_exhaustions"] != reasons["tail_exhausted"]
            or counts["tail_exhaustions"] != origins["none"]
            or observation_counts["total"]
            != observation_counts["eou"] + observation_counts["eob"]
            or sum(observation_origins.values()) != observation_counts["total"]
            or source_early["total"]
            != source_early["eou"] + source_early["eob"]
            or post_source["total"]
            != post_source["eou"] + post_source["eob"]
            or source_early["total"] > observation_counts["total"]
            or post_source["total"] > observation_counts["total"]
            or source_early["eou"] > observation_counts["eou"]
            or source_early["eob"] > observation_counts["eob"]
            or post_source["eou"] > observation_counts["eou"]
            or post_source["eob"] > observation_counts["eob"]
            or source_early["eou"] + post_source["eou"]
            > observation_counts["eou"]
            or source_early["eob"] + post_source["eob"]
            > observation_counts["eob"]
            or observation_origins["feed"]
            != source_early["total"] + post_source["total"]
            or counts["accepted_endpoints"] > post_source["eou"]
            or observation_counts["total"]
            > MAX_PARAKEET_CPP_OBSERVED_EVENTS * expected_evaluations
            or source["offered_total"]
            != source["consumed_total"] + source["dropped_total"]
            or source["dropped_total"] != 0
            or tail["offered_total"]
            != tail["consumed_total"] + tail["unconsumed_total"]
            or counts["terminal_model_padding_samples_total"] != 0
            or counts["model_input_samples_consumed_total"]
            != source["consumed_total"]
            + tail["consumed_total"]
            + counts["terminal_model_padding_samples_total"]
        ):
            raise QualificationError()
        accepted_samples = (
            numbers["accepted_endpoint_sample_p50"],
            numbers["accepted_endpoint_sample_p95"],
        )
        accepted_latencies = (
            numbers["accepted_endpoint_latency_p50_ms"],
            numbers["accepted_endpoint_latency_p95_ms"],
            numbers["accepted_endpoint_latency_max_ms"],
        )
        observed_frames = (
            numbers["observed_encoder_frame_p50"],
            numbers["observed_encoder_frame_p95"],
        )
        observed_times = (
            numbers["observed_encoder_time_p50_seconds"],
            numbers["observed_encoder_time_p95_seconds"],
        )
        if counts["accepted_endpoints"]:
            _ordered_numbers(accepted_samples, allow_all_none=False)
            _ordered_numbers(accepted_latencies, allow_all_none=False)
        elif any(item is not None for item in (*accepted_samples, *accepted_latencies)):
            raise QualificationError()
        if observation_counts["total"]:
            _ordered_numbers(observed_frames, allow_all_none=False)
            _ordered_numbers(observed_times, allow_all_none=False)
        elif any(item is not None for item in (*observed_frames, *observed_times)):
            raise QualificationError()
    else:
        raise QualificationError()
    snapshot = json.loads(_canonical_json_bytes(value))
    if not isinstance(snapshot, dict):
        raise QualificationError()
    return snapshot


def _metrics_snapshot(
    value: object,
    *,
    adapter: str,
    cases: int,
    repeats: int,
    expected_strata: Sequence[tuple[str, int]] | None,
) -> dict[str, object]:
    required = {
        "evaluations",
        "coverage_complete",
        "accuracy",
        "latency",
        "streaming",
        "throughput",
        "determinism",
        "resources",
    }
    allowed = required | {"endpointing", "strata"}
    if not isinstance(value, Mapping) or not required <= set(value) <= allowed:
        raise QualificationError()
    evaluations = _count(value.get("evaluations"))
    coverage = value.get("coverage_complete")
    if evaluations != cases * repeats or type(coverage) is not bool:
        raise QualificationError()
    result: dict[str, object] = {
        "evaluations": evaluations,
        "coverage_complete": coverage,
        "accuracy": _accuracy_snapshot(
            value.get("accuracy"),
            evaluations=evaluations,
        ),
        "latency": _latency_snapshot(
            value.get("latency"),
            evaluations=evaluations,
        ),
        "streaming": _streaming_snapshot(value.get("streaming")),
        "throughput": _throughput_snapshot(value.get("throughput")),
        "determinism": _determinism_snapshot(
            value.get("determinism"),
            cases=cases,
            repeats=repeats,
        ),
        "resources": _resources_snapshot(value.get("resources")),
    }
    endpoint_adapter = adapter in {
        PARAKEET_REALTIME_EOU_ADAPTER,
        PARAKEET_CPP_ADAPTER,
    }
    if endpoint_adapter and "endpointing" in value:
        result["endpointing"] = _endpointing_snapshot(
            value.get("endpointing"),
            expected_evaluations=evaluations,
            expected_adapter=adapter,
        )
        if "aggregate_rtf_scope" not in result["throughput"]:
            raise QualificationError()
    elif "endpointing" in value or "aggregate_rtf_scope" in result["throughput"]:
        raise QualificationError()
    elif endpoint_adapter:
        raise QualificationError()
    strata = value.get("strata")
    if expected_strata is None:
        if strata is not None:
            raise QualificationError()
    else:
        if not isinstance(strata, list) or len(strata) != len(expected_strata):
            raise QualificationError()
        selected_rows: list[dict[str, object]] = []
        for row, (tag, stratum_cases) in zip(
            strata,
            expected_strata,
            strict=True,
        ):
            if (
                not isinstance(row, Mapping)
                or set(row) != {"tag", "cases", "metrics"}
                or row.get("tag") != tag
                or row.get("cases") != stratum_cases
            ):
                raise QualificationError()
            nested_metrics = _metrics_snapshot(
                row.get("metrics"),
                adapter=adapter,
                cases=stratum_cases,
                repeats=repeats,
                expected_strata=None,
            )
            if nested_metrics["coverage_complete"] is not True:
                raise QualificationError()
            selected_rows.append(
                {
                    "tag": tag,
                    "cases": stratum_cases,
                    "metrics": nested_metrics,
                }
            )
        result["strata"] = selected_rows
    return result


def _config_snapshot(
    value: object,
    *,
    expected_stream: Mapping[str, object],
    expected_tags: Sequence[str],
    repeats: int,
) -> dict[str, object]:
    stream_fields = {
        "chunk_samples",
        "pace",
        "partial_interval_ms",
        "tail_padding_samples",
    }
    fields = stream_fields | {
        "repeats",
        "stratum_tags",
        "contract_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or set(expected_stream) != stream_fields
        or type(value.get("chunk_samples")) is not int
        or type(value.get("partial_interval_ms")) is not int
        or type(value.get("tail_padding_samples")) is not int
        or type(value.get("repeats")) is not int
        or type(value.get("pace")) is not str
        or not isinstance(value.get("stratum_tags"), list)
        or type(expected_stream.get("chunk_samples")) is not int
        or expected_stream["chunk_samples"] <= 0
        or expected_stream.get("pace") not in {"burst", "realtime"}
        or type(expected_stream.get("partial_interval_ms")) is not int
        or expected_stream["partial_interval_ms"] <= 0
        or type(expected_stream.get("tail_padding_samples")) is not int
        or expected_stream["tail_padding_samples"] < 0
        or type(repeats) is not int
        or repeats <= 0
    ):
        raise QualificationError()
    expected: dict[str, object] = {
        **expected_stream,
        "repeats": repeats,
        "stratum_tags": list(expected_tags),
    }
    expected["contract_sha256"] = _canonical_sha256(expected)
    if value != expected:
        raise QualificationError()
    snapshot = json.loads(_canonical_json_bytes(value))
    if not isinstance(snapshot, dict):
        raise QualificationError()
    return snapshot


def _safe_evidence(
    *,
    adapter: str,
    tags: Sequence[str],
) -> dict[str, object]:
    return {
        "kind": "aggregate_after_pcm_development_evidence",
        "aggregate_only": True,
        "capture": False,
        "vad": False,
        "aec": False,
        "audio_device": False,
        "live_latency": False,
        "quality_thresholds": "none",
        "promotional": False,
        "endpointing": adapter
        in {PARAKEET_REALTIME_EOU_ADAPTER, PARAKEET_CPP_ADAPTER},
        "stratification": {
            "kind": "explicit_corpus_case_tags",
            "aggregate_only": True,
            "overlapping": True,
            "tags": len(tags),
            "selection_sha256": _canonical_sha256(list(tags)),
        },
    }


def _safe_report_snapshot(
    report: object,
    *,
    adapter: str,
    expected_stream: Mapping[str, object],
    source_report_sha256: str,
    worker_manifest_sha256: str,
    corpus_manifest_sha256: str,
    evaluator_binding: Mapping[str, object],
    evaluator_implementation_sha256: str,
    tags: Sequence[str],
    stratum_counts: Sequence[tuple[str, int]],
    repeats: int,
) -> dict[str, object]:
    if not isinstance(report, Mapping):
        raise QualificationError()
    worker = report.get("worker")
    corpus = report.get("corpus")
    evidence = report.get("evidence")
    if (
        not isinstance(worker, Mapping)
        or not isinstance(corpus, Mapping)
        or not isinstance(evidence, Mapping)
        or worker.get("manifest_sha256") != worker_manifest_sha256
        or corpus.get("manifest_sha256") != corpus_manifest_sha256
        or _canonical_json_bytes(report.get("evaluator"))
        != _canonical_json_bytes(evaluator_binding)
    ):
        raise QualificationError()
    expected_stratification = {
        "kind": "explicit_corpus_case_tags",
        "aggregate_only": True,
        "overlapping": True,
        "tags": len(tags),
        "selection_sha256": _canonical_sha256(list(tags)),
    }
    pace = expected_stream.get("pace")
    endpoint_adapter = adapter in {
        PARAKEET_REALTIME_EOU_ADAPTER,
        PARAKEET_CPP_ADAPTER,
    }
    model_executed = evaluator_binding.get("model_executed")
    expected_latency_evidence = (
        "complete_pcm_to_candidate_final_only"
        if adapter == FASTER_WHISPER_ENDPOINT_ADAPTER
        else (
            "paced_replay_only"
            if pace == "realtime"
            else "accelerated_replay_only"
        )
    )
    if (
        type(model_executed) is not bool
        or evidence.get("kind") != "bounded_pcm_streaming_harness"
        or evidence.get("fake_adapter") is not (not model_executed)
        or evidence.get("production_model")
        is not evaluator_binding.get("production_model")
        or evidence.get("real_candidate_model")
        is not evaluator_binding.get("real_candidate_model")
        or evidence.get("model_executed") is not model_executed
        or evidence.get("replay_pace") != pace
        or evidence.get("accelerated_replay") is not (pace == "burst")
        or evidence.get("conversational_latency_valid") is not False
        or evidence.get("latency_evidence") != expected_latency_evidence
        or evidence.get("end_to_end_rtf") is not False
        or evidence.get("pcm_snapshot_io_included") is not False
        or evidence.get("controller_ipc_included") is not False
        or evidence.get("capture_to_text_latency") is not False
        or evidence.get("vad") is not False
        or evidence.get("endpointing") is not endpoint_adapter
        or evidence.get("aec") is not False
        or evidence.get("live_hardware") is not False
        or evidence.get("adoption_authority") is not False
        or _canonical_json_bytes(evidence.get("stratification"))
        != _canonical_json_bytes(expected_stratification)
    ):
        raise QualificationError()
    metrics = _metrics_snapshot(
        report.get("metrics"),
        adapter=adapter,
        cases=fixture.CASES_PER_SOURCE,
        repeats=repeats,
        expected_strata=stratum_counts,
    )
    ok = report.get("ok")
    if type(ok) is not bool or ok is not metrics["coverage_complete"]:
        raise QualificationError()
    return {
        "ok": ok,
        "evidence": _safe_evidence(adapter=adapter, tags=tags),
        "worker": {
            "manifest_sha256": worker_manifest_sha256,
            "adapter": adapter,
        },
        "corpus": {"manifest_sha256": corpus_manifest_sha256},
        "config": _config_snapshot(
            report.get("config"),
            expected_stream=expected_stream,
            expected_tags=tags,
            repeats=repeats,
        ),
        "metrics": metrics,
        "evaluator": {
            "implementation_sha256": evaluator_implementation_sha256,
        },
        "source_report_sha256": _sha256_field(source_report_sha256),
    }


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
) -> tuple[tuple[str, ...], tuple[Path, ...], tuple[str, ...]]:
    try:
        manifests = tuple(load_worker_manifest(path) for path in workers)
    except Exception:
        raise QualificationError() from None
    digests = tuple(manifest.digest for manifest in manifests)
    adapters = tuple(getattr(manifest, "adapter", None) for manifest in manifests)
    if (
        len(digests) != len(WORKER_ROLES)
        or len(set(digests)) != len(WORKER_ROLES)
        or any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in digests
        )
        or any(type(adapter) is not str or not adapter for adapter in adapters)
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
    return digests, roots, adapters  # type: ignore[return-value]


def _stream_snapshot(value: object) -> dict[str, object]:
    fields = {
        "chunk_samples",
        "pace",
        "partial_interval_ms",
        "tail_padding_samples",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or type(value.get("chunk_samples")) is not int
        or value["chunk_samples"] <= 0
        or value.get("pace") not in {"burst", "realtime"}
        or type(value.get("partial_interval_ms")) is not int
        or value["partial_interval_ms"] <= 0
        or type(value.get("tail_padding_samples")) is not int
        or value["tail_padding_samples"] < 0
    ):
        raise QualificationError()
    snapshot = json.loads(_canonical_json_bytes(value))
    if not isinstance(snapshot, dict):
        raise QualificationError()
    return snapshot


def _worker_stream_rows(
    worker_manifest_digests: Sequence[str],
    worker_adapters: Sequence[str],
    worker_stream_configs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if not (
        len(worker_manifest_digests)
        == len(worker_adapters)
        == len(worker_stream_configs)
        == len(WORKER_ROLES)
    ):
        raise QualificationError()
    result: list[dict[str, object]] = []
    for index, role in enumerate(WORKER_ROLES):
        digest = _sha256_field(worker_manifest_digests[index])
        adapter = worker_adapters[index]
        if type(adapter) is not str or adapter not in _ALLOWED_ADAPTERS:
            raise QualificationError()
        stream = _stream_snapshot(worker_stream_configs[index])
        result.append(
            {
                "role": role,
                "worker_index": index,
                "manifest_sha256": digest,
                "adapter": adapter,
                "stream": stream,
                "stream_sha256": _canonical_sha256(stream),
            }
        )
    return result


def _resolved_stream_configs(
    workers: Sequence[Path],
    *,
    chunk_samples: int | None,
    pace: str | None,
    partial_interval_ms: int | None,
    tail_padding_samples: int | None,
) -> tuple[dict[str, object], ...]:
    try:
        manifests = tuple(load_worker_manifest(path) for path in workers)
        result: list[dict[str, object]] = []
        for manifest in manifests:
            selected = streaming_stt_eval._selected_stream(
                manifest,
                None,
                chunk_samples=chunk_samples,
                pace=pace,
                partial_interval_ms=partial_interval_ms,
                tail_padding_samples=tail_padding_samples,
            )
            result.append(_stream_snapshot(selected.as_dict()))
        if len(result) != len(WORKER_ROLES):
            raise QualificationError()
        return tuple(result)
    except QualificationError:
        raise
    except Exception:
        raise QualificationError() from None


def _validate_two_by_four_result(
    value: object,
    *,
    fixture_set: fixture.FixtureSet,
    worker_manifest_digests: Sequence[str],
    worker_adapters: Sequence[str],
    worker_stream_configs: Sequence[Mapping[str, object]],
    qualification_implementation_sha256: str,
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
    if (
        type(value.get("ok")) is not bool
        or _canonical_json_bytes(value.get("fixture"))
        != _canonical_json_bytes(expected_fixture)
        or type(suite.get("ok")) is not bool
        or suite.get("complete") is not True
        or not isinstance(controller, Mapping)
        or set(controller) != controller_fields
        or not isinstance(config, Mapping)
        or not isinstance(runs, list)
        or controller.get("execution")
        != "strictly_sequential_cross_product"
        or controller.get("kind")
        != "bounded_sequential_streaming_stt_suite"
        or type(controller.get("schema_version")) is not int
        or controller.get("schema_version") != 1
        or controller.get("state") != "complete"
        or controller.get("planned_worker_manifests") != len(WORKER_ROLES)
        or controller.get("worker_manifests") != len(WORKER_ROLES)
        or controller.get("planned_corpora") != len(fixture.SOURCE_IDS)
        or controller.get("corpora") != len(fixture.SOURCE_IDS)
        or controller.get("planned_runs")
        != len(WORKER_ROLES) * len(fixture.SOURCE_IDS)
        or controller.get("completed_runs")
        != len(WORKER_ROLES) * len(fixture.SOURCE_IDS)
        or controller.get("runs")
        != len(WORKER_ROLES) * len(fixture.SOURCE_IDS)
        or len(runs) != len(WORKER_ROLES) * len(fixture.SOURCE_IDS)
        or len(worker_manifest_digests) != len(WORKER_ROLES)
        or len(worker_adapters) != len(WORKER_ROLES)
        or len(worker_stream_configs) != len(WORKER_ROLES)
        or _sha256_field(qualification_implementation_sha256)
        != qualification_implementation_sha256
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
    resolved_worker_streams = _worker_stream_rows(
        worker_manifest_digests,
        worker_adapters,
        worker_stream_configs,
    )
    worker_stream_set_sha256 = _canonical_sha256(resolved_worker_streams)
    try:
        evaluator_bindings = tuple(
            streaming_stt_eval._evaluator_binding(adapter)
            for adapter in worker_adapters
        )
        evaluator_digests = tuple(
            streaming_stt_suite._evaluator_implementation_sha256(binding)
            for binding in evaluator_bindings
        )
        expected_controller_sha256 = (
            streaming_stt_suite._controller_source_sha256()
        )
    except Exception:
        raise QualificationError() from None
    if len(set(evaluator_digests)) != 1:
        raise QualificationError()
    evaluator_digest = evaluator_digests[0]
    if (
        _canonical_json_bytes(config) != _canonical_json_bytes(expected_config)
        or controller.get("controller_sha256") != expected_controller_sha256
        or controller.get("stratum_selection_sha256")
        != expected_strata_config["selection_sha256"]
        or controller.get("worker_set_sha256") != expected_worker_set
        or controller.get("completed_worker_set_sha256") != expected_worker_set
        or controller.get("corpus_set_sha256") != expected_corpus_set
        or controller.get("completed_corpus_set_sha256") != expected_corpus_set
        or controller.get("evaluator_implementation_sha256")
        != evaluator_digest
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
    safe_runs: list[dict[str, object]] = []
    safe_bindings: list[dict[str, object]] = []
    run_success: list[bool] = []
    for run in runs:
        if not isinstance(run, Mapping) or set(run) != {"binding", "report"}:
            raise QualificationError()
        binding = run.get("binding")
        report = run.get("report")
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
            or set(report)
            != {
                "ok",
                "evidence",
                "worker",
                "corpus",
                "config",
                "metrics",
                "worker_stderr",
                "evaluator",
            }
            or type(report.get("ok")) is not bool
        ):
            raise QualificationError()
        worker_index = binding.get("worker_index")
        if (
            type(worker_index) is not int
            or not 0 <= worker_index < len(worker_manifest_digests)
        ):
            raise QualificationError()
        corpus_index = binding.get("corpus_index")
        if type(corpus_index) is not int or not 0 <= corpus_index < len(stratum_plan):
            raise QualificationError()
        source_id, tags = stratum_plan[corpus_index]
        source_report_sha256 = _sha256_field(binding.get("report_sha256"))
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
            or source_report_sha256 != _canonical_sha256(report)
            or binding_sha256 != _canonical_sha256(binding_contract)
        ):
            raise QualificationError()
        safe_report = _safe_report_snapshot(
            report,
            adapter=worker_adapters[worker_index],
            expected_stream=worker_stream_configs[worker_index],
            source_report_sha256=source_report_sha256,
            worker_manifest_sha256=worker_manifest_digests[worker_index],
            corpus_manifest_sha256=corpus_digests[corpus_index],
            evaluator_binding=evaluator_bindings[worker_index],
            evaluator_implementation_sha256=evaluator_digest,
            tags=tags,
            stratum_counts=tuple(
                (tag, count_plan[source_id][tag]) for tag in tags
            ),
            repeats=repeats,
        )
        safe_binding: dict[str, object] = {
            "worker_index": worker_index,
            "corpus_index": corpus_index,
            "worker_manifest_sha256": worker_manifest_digests[worker_index],
            "corpus_manifest_sha256": corpus_digests[corpus_index],
            "stratum_selection_sha256": _canonical_sha256(list(tags)),
            "evaluator_implementation_sha256": evaluator_digest,
            "worker_adapter": worker_adapters[worker_index],
            "stream_config_sha256": resolved_worker_streams[worker_index][
                "stream_sha256"
            ],
            "source_report_sha256": source_report_sha256,
            "source_binding_sha256": _sha256_field(binding_sha256),
            "report_sha256": _canonical_sha256(safe_report),
        }
        safe_binding["binding_sha256"] = _canonical_sha256(safe_binding)
        observed_order.append((worker_index, corpus_index))
        observed_bindings.append(binding)
        safe_bindings.append(safe_binding)
        safe_runs.append({"binding": safe_binding, "report": safe_report})
        run_success.append(safe_report["ok"] is True)
    expected_source_matrix = _canonical_sha256(observed_bindings)
    if observed_order != expected_order:
        raise QualificationError()
    expected_ok = all(run_success)
    if (
        suite.get("ok") is not expected_ok
        or value.get("ok") is not expected_ok
        or controller.get("matrix_sha256") != expected_source_matrix
        or controller.get("completed_matrix_sha256") != expected_source_matrix
    ):
        raise QualificationError()
    _assert_path_free(value)
    safe_matrix = _canonical_sha256(safe_bindings)
    safe_controller = {
        "kind": _SAFE_SUITE_KIND,
        "schema_version": _SAFE_SUITE_SCHEMA_VERSION,
        "execution": "strictly_sequential_cross_product",
        "state": "complete",
        "planned_worker_manifests": len(WORKER_ROLES),
        "planned_corpora": len(fixture.SOURCE_IDS),
        "planned_runs": len(safe_runs),
        "completed_runs": len(safe_runs),
        "qualification_implementation_sha256": (
            qualification_implementation_sha256
        ),
        "source_controller_sha256": expected_controller_sha256,
        "completed_worker_set_sha256": expected_worker_set,
        "completed_corpus_set_sha256": expected_corpus_set,
        "completed_matrix_sha256": safe_matrix,
        "stratum_selection_sha256": expected_strata_config[
            "selection_sha256"
        ],
        "evaluator_implementation_sha256": evaluator_digest,
        "worker_stream_set_sha256": worker_stream_set_sha256,
        "worker_manifests": len(WORKER_ROLES),
        "corpora": len(fixture.SOURCE_IDS),
        "runs": len(safe_runs),
        "worker_set_sha256": expected_worker_set,
        "corpus_set_sha256": expected_corpus_set,
        "matrix_sha256": safe_matrix,
        "source_matrix_sha256": expected_source_matrix,
    }
    safe_value = {
        "ok": expected_ok,
        "fixture": expected_fixture,
        "suite": {
            "ok": expected_ok,
            "complete": True,
            "controller": safe_controller,
            "config": expected_config,
            "runs": safe_runs,
        },
    }
    _assert_path_free(safe_value)
    return safe_value


def _contract(
    fixture_set: fixture.FixtureSet,
    plan: Sequence[tuple[str, tuple[str, ...]]],
    *,
    implementation_sha256: str,
    worker_manifest_digests: Sequence[str],
    worker_adapters: Sequence[str],
    worker_stream_configs: Sequence[Mapping[str, object]],
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
        "resolved_worker_streams": _worker_stream_rows(
            worker_manifest_digests,
            worker_adapters,
            worker_stream_configs,
        ),
        "source_strata": source_strata,
        "execution": "strictly_sequential_two_by_four",
        "quality_semantics": "coverage_only_no_thresholds",
        "development_only": True,
        "promotional": False,
    }
    value["contract_sha256"] = _canonical_sha256(value)
    return value


def _retained_suite_config(
    value: object,
) -> tuple[dict[str, object], int, dict[str, object]]:
    if not isinstance(value, Mapping) or set(value) != {
        "repeats",
        "strata",
        "overrides",
        "contract_sha256",
    }:
        raise QualificationError()
    repeats = value.get("repeats")
    overrides = value.get("overrides")
    if type(repeats) is not int or repeats <= 0 or not isinstance(overrides, Mapping):
        raise QualificationError()
    override_fields = {
        "chunk_samples",
        "pace",
        "partial_interval_ms",
        "tail_padding_samples",
    }
    if set(overrides) != override_fields:
        raise QualificationError()
    for key in ("chunk_samples", "partial_interval_ms"):
        selected = overrides.get(key)
        if selected is not None and (type(selected) is not int or selected <= 0):
            raise QualificationError()
    tail = overrides.get("tail_padding_samples")
    if tail is not None and (type(tail) is not int or tail < 0):
        raise QualificationError()
    if overrides.get("pace") not in {None, "burst", "realtime"}:
        raise QualificationError()
    rows = [
        {
            "corpus_index": index,
            "tags": list(_EXPECTED_REPORT_STRATA[source_id]),
            "selection_sha256": _canonical_sha256(
                list(_EXPECTED_REPORT_STRATA[source_id])
            ),
        }
        for index, source_id in enumerate(fixture.SOURCE_IDS)
    ]
    expected_strata = {
        "mode": "per_corpus",
        "corpora": rows,
        "selection_sha256": _canonical_sha256(rows),
    }
    contract = dict(value)
    digest = _sha256_field(contract.pop("contract_sha256"))
    if (
        _canonical_json_bytes(value.get("strata"))
        != _canonical_json_bytes(expected_strata)
        or digest != _canonical_sha256(contract)
    ):
        raise QualificationError()
    snapshot = json.loads(_canonical_json_bytes(value))
    if not isinstance(snapshot, dict):
        raise QualificationError()
    return snapshot, repeats, dict(overrides)


def _retained_run_report(
    value: object,
    *,
    adapter: str,
    worker_manifest_sha256: str,
    corpus_manifest_sha256: str,
    evaluator_implementation_sha256: str,
    tags: Sequence[str],
    stratum_counts: Sequence[tuple[str, int]],
    repeats: int,
    overrides: Mapping[str, object],
    expected_stream: Mapping[str, object],
) -> dict[str, object]:
    fields = {
        "ok",
        "evidence",
        "worker",
        "corpus",
        "config",
        "metrics",
        "evaluator",
        "source_report_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise QualificationError()
    worker = value.get("worker")
    corpus = value.get("corpus")
    evaluator = value.get("evaluator")
    config = value.get("config")
    if (
        not isinstance(worker, Mapping)
        or set(worker) != {"manifest_sha256", "adapter"}
        or worker.get("manifest_sha256") != worker_manifest_sha256
        or worker.get("adapter") != adapter
        or not isinstance(corpus, Mapping)
        or set(corpus) != {"manifest_sha256"}
        or corpus.get("manifest_sha256") != corpus_manifest_sha256
        or evaluator
        != {"implementation_sha256": evaluator_implementation_sha256}
        or _canonical_json_bytes(value.get("evidence"))
        != _canonical_json_bytes(_safe_evidence(adapter=adapter, tags=tags))
        or not isinstance(config, Mapping)
    ):
        raise QualificationError()
    selected_config = _config_snapshot(
        config,
        expected_stream=expected_stream,
        expected_tags=tags,
        repeats=repeats,
    )
    if any(
        override is not None and selected_config.get(key) != override
        for key, override in overrides.items()
    ):
        raise QualificationError()
    metrics = _metrics_snapshot(
        value.get("metrics"),
        adapter=adapter,
        cases=fixture.CASES_PER_SOURCE,
        repeats=repeats,
        expected_strata=stratum_counts,
    )
    ok = value.get("ok")
    if type(ok) is not bool or ok is not metrics["coverage_complete"]:
        raise QualificationError()
    _sha256_field(value.get("source_report_sha256"))
    snapshot = json.loads(_canonical_json_bytes(value))
    if not isinstance(snapshot, dict):
        raise QualificationError()
    return snapshot


def validate_qualification_report(value: object) -> dict[str, object]:
    """Validate and copy one serialized qualification-v2 retained report."""

    try:
        encoded = _canonical_json_bytes(value)
        if len(encoded) + 1 > _MAX_REPORT_BYTES:
            raise QualificationError()
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
            "worker_roles",
            "fixture",
            "suite",
        }
        if (
            not isinstance(snapshot, dict)
            or set(snapshot) != outer_fields
            or type(snapshot.get("ok")) is not bool
            or type(snapshot.get("schema_version")) is not int
            or snapshot.get("schema_version") != SCHEMA_VERSION
            or snapshot.get("kind") != KIND
            or snapshot.get("execution_complete") is not True
            or snapshot.get("execution_semantics") != "coverage_only"
            or snapshot.get("quality_decision") != "not_evaluated"
            or snapshot.get("development_only") is not True
            or snapshot.get("promotional") is not False
        ):
            raise QualificationError()
        contract = snapshot.get("contract")
        contract_fields = {
            "schema_version",
            "kind",
            "implementation_sha256",
            "fixture_id",
            "lock_recipe_sha256",
            "worker_roles",
            "resolved_worker_streams",
            "source_strata",
            "execution",
            "quality_semantics",
            "development_only",
            "promotional",
            "contract_sha256",
        }
        expected_source_strata = [
            {
                "source_id": source_id,
                "tags": list(_EXPECTED_REPORT_STRATA[source_id]),
            }
            for source_id in fixture.SOURCE_IDS
        ]
        if not isinstance(contract, Mapping) or set(contract) != contract_fields:
            raise QualificationError()
        contract_digest = _sha256_field(contract.get("contract_sha256"))
        contract_body = dict(contract)
        contract_body.pop("contract_sha256")
        implementation_sha256 = _sha256_field(
            contract.get("implementation_sha256")
        )
        lock_recipe_sha256 = _sha256_field(contract.get("lock_recipe_sha256"))
        if (
            type(contract.get("schema_version")) is not int
            or contract.get("schema_version") != SCHEMA_VERSION
            or contract.get("kind") != KIND
            or contract.get("fixture_id") != fixture.FIXTURE_ID
            or contract.get("worker_roles") != list(WORKER_ROLES)
            or contract.get("source_strata") != expected_source_strata
            or contract.get("execution") != "strictly_sequential_two_by_four"
            or contract.get("quality_semantics")
            != "coverage_only_no_thresholds"
            or contract.get("development_only") is not True
            or contract.get("promotional") is not False
            or contract_digest != _canonical_sha256(contract_body)
        ):
            raise QualificationError()

        roles = snapshot.get("worker_roles")
        if not isinstance(roles, list) or len(roles) != len(WORKER_ROLES):
            raise QualificationError()
        worker_digests: list[str] = []
        for index, (row, role) in enumerate(zip(roles, WORKER_ROLES, strict=True)):
            if (
                not isinstance(row, Mapping)
                or set(row) != {"role", "worker_index", "manifest_sha256"}
                or row.get("role") != role
                or type(row.get("worker_index")) is not int
                or row.get("worker_index") != index
            ):
                raise QualificationError()
            worker_digests.append(_sha256_field(row.get("manifest_sha256")))
        if len(set(worker_digests)) != len(worker_digests):
            raise QualificationError()
        raw_worker_streams = contract.get("resolved_worker_streams")
        if (
            not isinstance(raw_worker_streams, list)
            or len(raw_worker_streams) != len(WORKER_ROLES)
        ):
            raise QualificationError()
        worker_adapters: list[str] = []
        worker_stream_configs: list[dict[str, object]] = []
        for index, row in enumerate(raw_worker_streams):
            if (
                not isinstance(row, Mapping)
                or set(row)
                != {
                    "role",
                    "worker_index",
                    "manifest_sha256",
                    "adapter",
                    "stream",
                    "stream_sha256",
                }
                or row.get("role") != WORKER_ROLES[index]
                or type(row.get("worker_index")) is not int
                or row.get("worker_index") != index
                or row.get("manifest_sha256") != worker_digests[index]
            ):
                raise QualificationError()
            adapter = row.get("adapter")
            if type(adapter) is not str or adapter not in _ALLOWED_ADAPTERS:
                raise QualificationError()
            stream = _stream_snapshot(row.get("stream"))
            if row.get("stream_sha256") != _canonical_sha256(stream):
                raise QualificationError()
            worker_adapters.append(adapter)
            worker_stream_configs.append(stream)
        expected_worker_streams = _worker_stream_rows(
            worker_digests,
            worker_adapters,
            worker_stream_configs,
        )
        if raw_worker_streams != expected_worker_streams:
            raise QualificationError()
        worker_stream_set_sha256 = _canonical_sha256(expected_worker_streams)

        fixture_value = snapshot.get("fixture")
        if (
            not isinstance(fixture_value, Mapping)
            or set(fixture_value)
            != {
                "fixture_id",
                "lock_recipe_sha256",
                "fixture_set_sha256",
                "sources",
                "cases",
                "pcm_bytes",
                "validation",
            }
            or fixture_value.get("fixture_id") != fixture.FIXTURE_ID
            or fixture_value.get("lock_recipe_sha256") != lock_recipe_sha256
            or fixture_value.get("sources") != len(fixture.SOURCE_IDS)
            or fixture_value.get("cases")
            != len(fixture.SOURCE_IDS) * fixture.CASES_PER_SOURCE
            or type(fixture_value.get("pcm_bytes")) is not int
            or fixture_value["pcm_bytes"] <= 0
            or fixture_value.get("validation") != "before_and_after_suite"
        ):
            raise QualificationError()
        _sha256_field(fixture_value.get("fixture_set_sha256"))

        suite = snapshot.get("suite")
        if not isinstance(suite, Mapping) or set(suite) != {
            "ok",
            "complete",
            "controller",
            "config",
            "runs",
        }:
            raise QualificationError()
        suite_config, repeats, overrides = _retained_suite_config(
            suite.get("config")
        )
        controller = suite.get("controller")
        controller_fields = {
            "kind",
            "schema_version",
            "execution",
            "state",
            "planned_worker_manifests",
            "planned_corpora",
            "planned_runs",
            "completed_runs",
            "qualification_implementation_sha256",
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
        run_count = len(WORKER_ROLES) * len(fixture.SOURCE_IDS)
        if (
            not isinstance(controller, Mapping)
            or set(controller) != controller_fields
            or controller.get("kind") != _SAFE_SUITE_KIND
            or type(controller.get("schema_version")) is not int
            or controller.get("schema_version") != _SAFE_SUITE_SCHEMA_VERSION
            or controller.get("execution")
            != "strictly_sequential_cross_product"
            or controller.get("state") != "complete"
            or controller.get("planned_worker_manifests") != len(WORKER_ROLES)
            or controller.get("worker_manifests") != len(WORKER_ROLES)
            or controller.get("planned_corpora") != len(fixture.SOURCE_IDS)
            or controller.get("corpora") != len(fixture.SOURCE_IDS)
            or controller.get("planned_runs") != run_count
            or controller.get("completed_runs") != run_count
            or controller.get("runs") != run_count
            or controller.get("qualification_implementation_sha256")
            != implementation_sha256
            or controller.get("stratum_selection_sha256")
            != suite_config["strata"]["selection_sha256"]
            or controller.get("worker_stream_set_sha256")
            != worker_stream_set_sha256
        ):
            raise QualificationError()
        source_controller_sha256 = _sha256_field(
            controller.get("source_controller_sha256")
        )
        evaluator_sha256 = _sha256_field(
            controller.get("evaluator_implementation_sha256")
        )
        if not source_controller_sha256:
            raise QualificationError()
        runs = suite.get("runs")
        if not isinstance(runs, list) or len(runs) != run_count:
            raise QualificationError()

        corpus_digests: list[str | None] = [None] * len(fixture.SOURCE_IDS)
        safe_bindings: list[Mapping[str, object]] = []
        source_bindings: list[dict[str, object]] = []
        run_success: list[bool] = []
        expected_order = [
            (worker_index, corpus_index)
            for worker_index in range(len(WORKER_ROLES))
            for corpus_index in range(len(fixture.SOURCE_IDS))
        ]
        for run, (worker_index, corpus_index) in zip(
            runs,
            expected_order,
            strict=True,
        ):
            if not isinstance(run, Mapping) or set(run) != {"binding", "report"}:
                raise QualificationError()
            binding = run.get("binding")
            report = run.get("report")
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
            if (
                not isinstance(binding, Mapping)
                or set(binding) != binding_fields
                or type(binding.get("worker_index")) is not int
                or binding.get("worker_index") != worker_index
                or type(binding.get("corpus_index")) is not int
                or binding.get("corpus_index") != corpus_index
                or binding.get("worker_manifest_sha256")
                != worker_digests[worker_index]
                or binding.get("evaluator_implementation_sha256")
                != evaluator_sha256
                or binding.get("worker_adapter") != worker_adapters[worker_index]
                or binding.get("stream_config_sha256")
                != expected_worker_streams[worker_index]["stream_sha256"]
            ):
                raise QualificationError()
            corpus_digest = _sha256_field(
                binding.get("corpus_manifest_sha256")
            )
            previous_corpus = corpus_digests[corpus_index]
            if previous_corpus is not None and previous_corpus != corpus_digest:
                raise QualificationError()
            corpus_digests[corpus_index] = corpus_digest
            source_id = fixture.SOURCE_IDS[corpus_index]
            tags = _EXPECTED_REPORT_STRATA[source_id]
            if binding.get("stratum_selection_sha256") != _canonical_sha256(
                list(tags)
            ):
                raise QualificationError()
            source_report_sha256 = _sha256_field(
                binding.get("source_report_sha256")
            )
            source_binding_sha256 = _sha256_field(
                binding.get("source_binding_sha256")
            )
            safe_report_sha256 = _sha256_field(binding.get("report_sha256"))
            safe_binding_contract = dict(binding)
            safe_binding_digest = _sha256_field(
                safe_binding_contract.pop("binding_sha256")
            )
            if safe_binding_digest != _canonical_sha256(safe_binding_contract):
                raise QualificationError()
            if not isinstance(report, Mapping):
                raise QualificationError()
            worker = report.get("worker")
            if not isinstance(worker, Mapping):
                raise QualificationError()
            adapter = worker.get("adapter")
            if adapter != worker_adapters[worker_index]:
                raise QualificationError()
            safe_report = _retained_run_report(
                report,
                adapter=adapter,
                worker_manifest_sha256=worker_digests[worker_index],
                corpus_manifest_sha256=corpus_digest,
                evaluator_implementation_sha256=evaluator_sha256,
                tags=tags,
                stratum_counts=_EXPECTED_REPORT_STRATUM_COUNTS[source_id],
                repeats=repeats,
                overrides=overrides,
                expected_stream=worker_stream_configs[worker_index],
            )
            if (
                safe_report.get("source_report_sha256")
                != source_report_sha256
                or safe_report_sha256 != _canonical_sha256(safe_report)
            ):
                raise QualificationError()
            source_binding_contract = {
                "worker_index": worker_index,
                "corpus_index": corpus_index,
                "worker_manifest_sha256": worker_digests[worker_index],
                "corpus_manifest_sha256": corpus_digest,
                "stratum_selection_sha256": _canonical_sha256(list(tags)),
                "evaluator_implementation_sha256": evaluator_sha256,
                "report_sha256": source_report_sha256,
            }
            if source_binding_sha256 != _canonical_sha256(
                source_binding_contract
            ):
                raise QualificationError()
            source_binding_contract["binding_sha256"] = source_binding_sha256
            safe_bindings.append(binding)
            source_bindings.append(source_binding_contract)
            run_success.append(safe_report["ok"] is True)

        if any(item is None for item in corpus_digests):
            raise QualificationError()
        worker_set_sha256 = _canonical_sha256(worker_digests)
        corpus_set_sha256 = _canonical_sha256(corpus_digests)
        safe_matrix_sha256 = _canonical_sha256(safe_bindings)
        source_matrix_sha256 = _canonical_sha256(source_bindings)
        expected_ok = all(run_success)
        if (
            snapshot.get("ok") is not expected_ok
            or suite.get("ok") is not expected_ok
            or suite.get("complete") is not True
            or controller.get("worker_set_sha256") != worker_set_sha256
            or controller.get("completed_worker_set_sha256")
            != worker_set_sha256
            or controller.get("corpus_set_sha256") != corpus_set_sha256
            or controller.get("completed_corpus_set_sha256")
            != corpus_set_sha256
            or controller.get("matrix_sha256") != safe_matrix_sha256
            or controller.get("completed_matrix_sha256")
            != safe_matrix_sha256
            or controller.get("source_matrix_sha256") != source_matrix_sha256
        ):
            raise QualificationError()
        _assert_path_free(snapshot)
        return snapshot
    except QualificationError:
        raise
    except Exception:
        raise QualificationError() from None


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
        (
            worker_manifest_digests,
            worker_evidence_roots,
            worker_adapters,
        ) = _worker_manifest_state(workers)
        worker_stream_configs = _resolved_stream_configs(
            workers,
            chunk_samples=chunk_samples,
            pace=pace,
            partial_interval_ms=partial_interval_ms,
            tail_padding_samples=tail_padding_samples,
        )
        _validate_private_paths(
            final_output=final_output,
            scratch=scratch,
            suite_checkpoint=suite_checkpoint,
            fixture_set=fixture_set,
            worker_evidence_roots=worker_evidence_roots,
        )
        scratch, suite_checkpoint, scratch_identity = _prepare_private_scratch(
            scratch
        )
        if not callable(_PRODUCTION_RUNNER):
            raise QualificationError()
        raw_result: object | None = None
        runner_failed = False
        try:
            raw_result = _PRODUCTION_RUNNER(
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
        except BaseException:  # Revalidate private evidence before translation.
            runner_failed = True
        post_validation_failed = False
        try:
            if (
                _implementation_sha256() != implementation_sha256
                or _worker_manifest_state(workers)
                != (
                    worker_manifest_digests,
                    worker_evidence_roots,
                    worker_adapters,
                )
                or _resolved_stream_configs(
                    workers,
                    chunk_samples=chunk_samples,
                    pace=pace,
                    partial_interval_ms=partial_interval_ms,
                    tail_padding_samples=tail_padding_samples,
                )
                != worker_stream_configs
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
            checkpoint_value = _checkpoint_snapshot(
                scratch=scratch,
                checkpoint=suite_checkpoint,
                scratch_identity=scratch_identity,
                required=not runner_failed,
            )
            if not runner_failed:
                if (
                    not isinstance(raw_result, Mapping)
                    or checkpoint_value != raw_result.get("suite")
                ):
                    raise QualificationError()
            if _new_private_output_path(final_output) != final_output:
                raise QualificationError()
        except BaseException:
            post_validation_failed = True
        if runner_failed or post_validation_failed or raw_result is None:
            raise QualificationError()
        bound = _validate_two_by_four_result(
            raw_result,
            fixture_set=fixture_set,
            worker_manifest_digests=worker_manifest_digests,
            worker_adapters=worker_adapters,
            worker_stream_configs=worker_stream_configs,
            qualification_implementation_sha256=implementation_sha256,
            stratum_plan=plan,
            expected_stratum_counts=expected_stratum_counts,
            repeats=repeats,
            chunk_samples=chunk_samples,
            pace=pace,
            partial_interval_ms=partial_interval_ms,
            tail_padding_samples=tail_padding_samples,
        )
        contract = _contract(
            fixture_set,
            plan,
            implementation_sha256=implementation_sha256,
            worker_manifest_digests=worker_manifest_digests,
            worker_adapters=worker_adapters,
            worker_stream_configs=worker_stream_configs,
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
        validated_result = validate_qualification_report(result)
        try:
            if (
                _implementation_sha256() != implementation_sha256
                or _worker_manifest_state(workers)
                != (
                    worker_manifest_digests,
                    worker_evidence_roots,
                    worker_adapters,
                )
                or _resolved_stream_configs(
                    workers,
                    chunk_samples=chunk_samples,
                    pace=pace,
                    partial_interval_ms=partial_interval_ms,
                    tail_padding_samples=tail_padding_samples,
                )
                != worker_stream_configs
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
            final_checkpoint = _checkpoint_snapshot(
                scratch=scratch,
                checkpoint=suite_checkpoint,
                scratch_identity=scratch_identity,
                required=True,
            )
            if (
                not isinstance(raw_result, Mapping)
                or final_checkpoint != raw_result.get("suite")
                or _new_private_output_path(final_output) != final_output
            ):
                raise QualificationError()
        except BaseException:
            raise QualificationError() from None
        return validated_result
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
        report_sha256 = write_new_report(args.output, result)
        summary = {
            "ok": result["ok"],
            "kind": KIND,
            "execution_complete": result["execution_complete"],
            "quality_decision": result["quality_decision"],
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


__all__ = [
    "KIND",
    "QualificationError",
    "SCHEMA_VERSION",
    "WORKER_ROLES",
    "canonical_stratum_plan",
    "main",
    "run_qualification",
    "validate_qualification_report",
    "write_new_report",
]
