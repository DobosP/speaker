"""Bind exact local Anyreach model/data artifacts without executing them.

The command accepts only already-staged private inputs.  It does not install,
download, import candidate/runtime libraries, execute or load the ONNX model,
or decode the Parquet benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Mapping, Sequence

from tools.semantic_interruption import anyreach as contract
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    opened_directory_nofollow,
    read_regular_bounded,
)


_SAFE_ERROR = {
    "ok": False,
    "error": "anyreach_semantic_turn_prerequisites_unavailable",
}
_PENDING_MANIFEST_FILENAME = ".anyreach-artifact-manifest.pending"


class ProvisionError(RuntimeError):
    """A detail-free unsafe, incomplete, or changed provision input."""


@dataclass(frozen=True)
class _PublishedManifest:
    path: Path
    identity: tuple[int, ...]
    directory_identity: tuple[int, ...]


def _absolute(path: Path | str) -> Path:
    try:
        supplied = Path(path).expanduser()
        candidate = Path(os.path.abspath(supplied))
    except (OSError, RuntimeError, TypeError, ValueError):
        raise ProvisionError() from None
    if not supplied.is_absolute() or candidate != supplied:
        raise ProvisionError()
    return candidate


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _overlaps(left: Path, right: Path) -> bool:
    return _inside(left, right) or _inside(right, left)


def _inode_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
    )


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        *_inode_identity(metadata),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _file_identity_without_links(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _private_regular(metadata: os.stat_result, *, size_bytes: int) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_nlink == 1
        and stat.S_IMODE(metadata.st_mode) == 0o600
        and metadata.st_size == size_bytes
        and (not hasattr(os, "geteuid") or metadata.st_uid == os.geteuid())
    )


def _has_git_ancestor(path: Path) -> bool:
    """Reject private/path-bearing outputs beneath any Git worktree marker."""

    try:
        current = path.resolve(strict=True)
        while True:
            try:
                (current / ".git").lstat()
            except FileNotFoundError:
                pass
            else:
                return True
            if current.parent == current:
                return False
            current = current.parent
    except (OSError, RuntimeError, ValueError):
        raise ProvisionError() from None


def _write_new_private(path: Path, payload: bytes) -> None:
    if not isinstance(payload, bytes) or not payload:
        raise ProvisionError()
    descriptor = -1
    try:
        if (
            not path.is_absolute()
            or not path.name
            or path.name in {".", ".."}
            or "/" in path.name
            or "\x00" in path.name
        ):
            raise ProvisionError()
        with opened_directory_nofollow(
            path.parent,
            require_private=True,
        ) as (stable_parent, parent_descriptor):
            if stable_parent != path.parent:
                raise ProvisionError()
            parent_identity = _inode_identity(os.fstat(parent_descriptor))
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_descriptor)
            os.fchmod(descriptor, 0o600)
            created = os.fstat(descriptor)
            if not _private_regular(created, size_bytes=0):
                raise ProvisionError()
            created_identity = _inode_identity(created)
            with os.fdopen(descriptor, "wb", buffering=0) as handle:
                descriptor = -1
                if handle.write(payload) != len(payload):
                    raise ProvisionError()
                os.fsync(handle.fileno())
                written = os.fstat(handle.fileno())
                named = os.stat(
                    path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not _private_regular(written, size_bytes=len(payload))
                    or not _private_regular(named, size_bytes=len(payload))
                    or _inode_identity(written) != created_identity
                    or _file_identity(named) != _file_identity(written)
                ):
                    raise ProvisionError()
                os.fsync(parent_descriptor)
                final_descriptor = os.fstat(handle.fileno())
                final_named = os.stat(
                    path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    _file_identity(final_descriptor) != _file_identity(written)
                    or _file_identity(final_named) != _file_identity(written)
                    or _inode_identity(os.fstat(parent_descriptor)) != parent_identity
                ):
                    raise ProvisionError()
    except (
        BoundedReadError,
        OSError,
        OverflowError,
        ProvisionError,
        RuntimeError,
        ValueError,
    ):
        raise ProvisionError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _new_private_output(
    path: Path | str,
    *,
    forbidden: tuple[Path, ...],
) -> Path:
    candidate = _absolute(path)
    if (
        not candidate.name
        or candidate.name in {".", ".."}
        or "\x00" in candidate.name
        or any(_overlaps(candidate, item) for item in forbidden)
    ):
        raise ProvisionError()
    try:
        parent, _names, parent_identity = contract._private_directory(candidate.parent)
        if parent != candidate.parent or _has_git_ancestor(parent):
            raise ProvisionError()
        parent_before = parent.lstat()
        if contract._directory_identity(parent_before) != parent_identity:
            raise ProvisionError()
        with opened_directory_nofollow(parent, require_private=True) as (
            stable_parent,
            parent_descriptor,
        ):
            opened_parent = os.fstat(parent_descriptor)
            if stable_parent != parent or _inode_identity(
                opened_parent
            ) != _inode_identity(parent_before):
                raise ProvisionError()
            os.mkdir(candidate.name, mode=0o700, dir_fd=parent_descriptor)
            created = os.stat(
                candidate.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(created.st_mode)
                or stat.S_IMODE(created.st_mode) != 0o700
                or (hasattr(os, "geteuid") and created.st_uid != os.geteuid())
            ):
                raise ProvisionError()
            os.fsync(parent_descriptor)
            created_after = os.stat(
                candidate.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            current_parent = parent.lstat()
            if _file_identity(created_after) != _file_identity(
                created
            ) or _inode_identity(current_parent) != _inode_identity(
                os.fstat(parent_descriptor)
            ):
                raise ProvisionError()
        stable, names, identity = contract._private_directory(candidate)
        current = candidate.lstat()
        if (
            stable != candidate
            or names
            or _file_identity(current) != _file_identity(created_after)
            or contract._directory_identity(current) != identity
        ):
            raise ProvisionError()
        return stable
    except ProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        raise ProvisionError() from None


def _manifest_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return contract._canonical_json(payload) + b"\n"
    except contract.AnyreachContractError:
        raise ProvisionError() from None


def _cleanup_failed_manifest_publication(destination: Path, descriptor: int) -> None:
    """Remove only pending/canonical names still bound to the staged inode."""

    try:
        opened = os.fstat(descriptor)
        expected = _file_identity_without_links(opened)
        with opened_directory_nofollow(
            destination,
            require_private=True,
        ) as (stable, parent_descriptor):
            if stable != destination:
                raise ProvisionError()
            matched: list[str] = []
            for name in (
                _PENDING_MANIFEST_FILENAME,
                contract.MANIFEST_FILENAME,
            ):
                try:
                    metadata = os.stat(
                        name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    continue
                if _file_identity_without_links(metadata) != expected:
                    raise ProvisionError()
                matched.append(name)
            for name in matched:
                os.unlink(name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
            remaining = set(os.listdir(parent_descriptor))
            if remaining.intersection(
                {_PENDING_MANIFEST_FILENAME, contract.MANIFEST_FILENAME}
            ):
                raise ProvisionError()
    except (BoundedReadError, OSError, ProvisionError, RuntimeError, ValueError):
        raise ProvisionError() from None


def _publish_terminal_manifest(
    destination: Path,
    payload: bytes,
) -> _PublishedManifest:
    """Stage complete bytes, then no-clobber publish the canonical marker."""

    pending = destination / _PENDING_MANIFEST_FILENAME
    manifest = destination / contract.MANIFEST_FILENAME
    _write_new_private(pending, payload)
    descriptor = -1
    try:
        with opened_directory_nofollow(
            destination,
            require_private=True,
        ) as (stable, parent_descriptor):
            expected_pending_names = tuple(
                sorted(
                    (
                        _PENDING_MANIFEST_FILENAME,
                        contract.LOCK_COPY_FILENAME,
                        contract.BENCHMARK_COPY_FILENAME,
                    )
                )
            )
            if (
                stable != destination
                or tuple(sorted(os.listdir(parent_descriptor)))
                != expected_pending_names
            ):
                raise ProvisionError()
            before = os.stat(
                pending.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if not _private_regular(before, size_bytes=len(payload)):
                raise ProvisionError()
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(
                pending.name,
                flags,
                dir_fd=parent_descriptor,
            )
            opened = os.fstat(descriptor)
            if _file_identity(opened) != _file_identity(before):
                raise ProvisionError()
            chunks: list[bytes] = []
            consumed = 0
            while consumed <= len(payload):
                chunk = os.read(
                    descriptor,
                    min(64 * 1024, len(payload) + 1 - consumed),
                )
                if not chunk:
                    break
                consumed += len(chunk)
                if consumed > len(payload):
                    raise ProvisionError()
                chunks.append(chunk)
            read_after = os.fstat(descriptor)
            named_before_link = os.stat(
                pending.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                consumed != len(payload)
                or b"".join(chunks) != payload
                or _file_identity(read_after) != _file_identity(opened)
                or _file_identity(named_before_link) != _file_identity(opened)
            ):
                raise ProvisionError()

            os.link(
                pending.name,
                manifest.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            linked_pending = os.stat(
                pending.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            linked_manifest = os.stat(
                manifest.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            linked_descriptor = os.fstat(descriptor)
            linked_identity = _file_identity_without_links(linked_descriptor)
            if (
                linked_descriptor.st_nlink != 2
                or linked_pending.st_nlink != 2
                or linked_manifest.st_nlink != 2
                or _file_identity_without_links(linked_pending) != linked_identity
                or _file_identity_without_links(linked_manifest) != linked_identity
            ):
                raise ProvisionError()
            os.fsync(parent_descriptor)

            os.unlink(pending.name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
            final_descriptor = os.fstat(descriptor)
            final_manifest = os.stat(
                manifest.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            expected_final_names = tuple(
                sorted(
                    (
                        contract.LOCK_COPY_FILENAME,
                        contract.BENCHMARK_COPY_FILENAME,
                        contract.MANIFEST_FILENAME,
                    )
                )
            )
            if (
                not _private_regular(final_descriptor, size_bytes=len(payload))
                or not _private_regular(final_manifest, size_bytes=len(payload))
                or _file_identity(final_manifest) != _file_identity(final_descriptor)
                or tuple(sorted(os.listdir(parent_descriptor))) != expected_final_names
            ):
                raise ProvisionError()
        verified = contract._read_private_regular(
            manifest,
            maximum_bytes=len(payload),
            expected_bytes=len(payload),
            retain_bytes=True,
        )
        if verified.path != manifest or verified.data != payload:
            raise ProvisionError()
        final_directory, final_names, final_directory_identity = (
            contract._private_directory(destination)
        )
        if final_directory != destination or final_names != tuple(
            sorted(
                (
                    contract.LOCK_COPY_FILENAME,
                    contract.BENCHMARK_COPY_FILENAME,
                    contract.MANIFEST_FILENAME,
                )
            )
        ):
            raise ProvisionError()
        return _PublishedManifest(
            path=manifest,
            identity=verified.identity,
            directory_identity=final_directory_identity,
        )
    except BaseException as error:
        if descriptor >= 0:
            _cleanup_failed_manifest_publication(destination, descriptor)
        if isinstance(error, (GeneratorExit, KeyboardInterrupt, SystemExit)):
            raise
        raise ProvisionError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _retract_published_manifest(publication: _PublishedManifest) -> None:
    """Remove only the exact canonical inode from a failed full reopen."""

    descriptor = -1
    try:
        with opened_directory_nofollow(
            publication.path.parent,
            require_private=True,
        ) as (stable, parent_descriptor):
            before_parent = os.fstat(parent_descriptor)
            if (
                stable != publication.path.parent
                or contract._directory_identity(before_parent)[:6]
                != publication.directory_identity[:6]
            ):
                raise ProvisionError()
            before = os.stat(
                publication.path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if _file_identity(before) != publication.identity:
                raise ProvisionError()
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(
                publication.path.name,
                flags,
                dir_fd=parent_descriptor,
            )
            opened = os.fstat(descriptor)
            named = os.stat(
                publication.path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                _file_identity(opened) != publication.identity
                or _file_identity(named) != publication.identity
            ):
                raise ProvisionError()
            os.unlink(publication.path.name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
            after = os.fstat(descriptor)
            parent_after = os.fstat(parent_descriptor)
            if (
                after.st_nlink != 0
                or _file_identity_without_links(after)
                != _file_identity_without_links(opened)
                or contract._directory_identity(parent_after)[:6]
                != publication.directory_identity[:6]
                or publication.path.name in os.listdir(parent_descriptor)
            ):
                raise ProvisionError()
    except (BoundedReadError, OSError, ProvisionError, RuntimeError, ValueError):
        raise ProvisionError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _binding_payload(item: contract.BoundArtifact) -> dict[str, object]:
    return {
        "name": item.name,
        "role": item.role,
        "path": str(item.path),
        "sha256": item.sha256,
        "size_bytes": item.size_bytes,
    }


def provision_candidate(
    *,
    model_root: Path | str,
    benchmark_parquet: Path | str,
    output_dir: Path | str,
    accepted_terms: tuple[str, ...],
) -> contract.AnyreachArtifactManifest:
    """Publish a private manifest without model execution or Parquet decode."""

    publication: _PublishedManifest | None = None
    completed = False
    try:
        lock = contract.load_source_lock()
        if type(accepted_terms) is not tuple or accepted_terms != (lock.acceptance_id,):
            raise ProvisionError()
        model_path = _absolute(model_root)
        model_bound = contract._bind_model_root(model_path, lock)
        benchmark_snapshot = contract._snapshot_benchmark_source(
            benchmark_parquet,
            lock,
        )
        lock_snapshot = read_regular_bounded(
            lock.path,
            maximum_bytes=contract.LOCK_SIZE_BYTES,
            expected_bytes=contract.LOCK_SIZE_BYTES,
        )
        if hashlib.sha256(lock_snapshot.data).hexdigest() != contract.LOCK_RAW_SHA256:
            raise ProvisionError()
        destination = _new_private_output(
            output_dir,
            forbidden=(
                model_path,
                benchmark_snapshot.path.parent,
                lock_snapshot.path,
            ),
        )
        retained_lock = destination / contract.LOCK_COPY_FILENAME
        retained_benchmark = destination / contract.BENCHMARK_COPY_FILENAME
        _write_new_private(retained_lock, lock_snapshot.data)
        _write_new_private(retained_benchmark, benchmark_snapshot.data)
        source_bound = contract._bind_exact_file(
            retained_lock,
            name="source-lock",
            role="source-lock",
            size_bytes=contract.LOCK_SIZE_BYTES,
            sha256=contract.LOCK_RAW_SHA256,
        )
        benchmark_bound = contract._bind_exact_file(
            retained_benchmark,
            name="benchmark-parquet",
            role="benchmark-source",
            size_bytes=lock.benchmark_artifact.size_bytes,
            sha256=lock.benchmark_artifact.sha256,
        )
        artifacts = (*model_bound, benchmark_bound)
        artifact_set = contract.artifact_set_sha256(source_bound, artifacts)
        payload = {
            "schema_version": 1,
            "kind": contract.MANIFEST_KIND,
            "candidate_id": contract.CANDIDATE_ID,
            "source_lock": _binding_payload(source_bound),
            "artifacts": [_binding_payload(item) for item in artifacts],
            "benchmark_selection": {
                "source_rows": lock.source_rows,
                "selected_rows": len(lock.slots),
                "selected_ids_sha256": lock.selected_ids_sha256,
                "parquet_decoded": False,
            },
            "artifact_set_sha256": artifact_set,
            "evidence_scope": dict(lock.evidence_scope),
        }
        payload["manifest_self_sha256"] = contract._canonical_sha256(payload)
        publication = _publish_terminal_manifest(
            destination,
            _manifest_bytes(payload),
        )
        manifest = contract.load_artifact_manifest(publication.path)
        if (
            manifest.candidate_id != contract.CANDIDATE_ID
            or manifest.artifact_set_sha256 != artifact_set
            or manifest.selected_rows != 24
            or any(
                manifest.evidence_scope[field] is not False
                for field in (
                    "model_executed",
                    "tokenizer_executed",
                    "onnx_contract_verified",
                    "parquet_decoded",
                    "production_evidence",
                )
            )
        ):
            raise ProvisionError()
        completed = True
        return manifest
    except ProvisionError:
        raise
    except (
        BoundedReadError,
        contract.AnyreachContractError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        raise ProvisionError() from None
    finally:
        if publication is not None and not completed:
            _retract_published_manifest(publication)


def _safe_result(
    manifest: contract.AnyreachArtifactManifest,
) -> dict[str, object]:
    return {
        "ok": True,
        "kind": contract.CANDIDATE_ID,
        "manifest_sha256": manifest.digest,
        "artifact_set_sha256": manifest.artifact_set_sha256,
        "benchmark_rows": manifest.selected_rows,
        "model_executed": False,
        "parquet_decoded": False,
    }


def _absolute_argument(value: str) -> Path:
    try:
        return _absolute(value)
    except ProvisionError:
        raise argparse.ArgumentTypeError("expected an absolute path") from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind an exact pre-existing private Anyreach Q8 artifact root and "
            "synthetic benchmark into a private artifact manifest. This command "
            "does not install, download, import candidate/runtime libraries, "
            "execute or load a model, or decode Parquet."
        )
    )
    parser.add_argument("--model-root", type=_absolute_argument, required=True)
    parser.add_argument(
        "--benchmark-parquet",
        type=_absolute_argument,
        required=True,
    )
    parser.add_argument("--output-dir", type=_absolute_argument, required=True)
    parser.add_argument(
        "--accept-license",
        action="append",
        default=[],
        metavar="ID",
        help=f"required exact metadata-assertion ID: {contract.ACCEPTANCE_ID}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = provision_candidate(
            model_root=args.model_root,
            benchmark_parquet=args.benchmark_parquet,
            output_dir=args.output_dir,
            accepted_terms=tuple(args.accept_license),
        )
        result: Mapping[str, object] = _safe_result(manifest)
        code = 0
    except BaseException:  # noqa: BLE001 - every lifecycle failure stays private
        result = _SAFE_ERROR
        code = 2
    print(
        json.dumps(
            result,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
