"""Strict bindings for the production-profile whole-turn replay."""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
import os
import re
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from core.config import (
    apply_device_profile,
    apply_final_stt_profile,
    deep_merge,
    load_config,
)
from tools.streaming_stt.corpus import (
    CorpusError,
    LoadedCorpus,
    load_corpus,
    verify_corpus_snapshot,
)
from tools.conversation_eval.provenance import (
    LOCAL_OLLAMA_HOST,
    full_identity_contract,
    identity_snapshot,
    repository_metadata,
)
from tools.streaming_stt.bounded_io import BoundedReadError, opened_directory_nofollow


_REPLAY_SAFETY_POLICY = {
    "schema_version": 1,
    "memory_backend": "session",
    "local_llm": {
        "host": LOCAL_OLLAMA_HOST,
        "cloud_enabled": False,
        "cloud_strategy": "local_only",
        "live_routing": False,
        "router_model": None,
    },
    "disabled_external_providers": [
        "gui_actions",
        "obsidian",
        "reminders",
        "trusted_apps",
        "watch",
        "web_search",
    ],
}
_ARTIFACT_FIELDS = (
    "asr_tokens",
    "asr_encoder",
    "asr_decoder",
    "asr_joiner",
    "asr_bpe_vocab",
    "vad_model",
    "punct_model",
    "tts_model",
    "tts_tokens",
    "tts_data_dir",
    "tts_voices",
    "tts_lexicon",
    "asr_final_model",
    "asr_final_tokens",
    "asr_final_decoder",
    "asr_final_joiner",
    "asr_final_verifier_model",
    "asr_final_hr_dict_dir",
    "asr_final_hr_lexicon",
    "asr_final_hr_rule_fsts",
    "asr_final_rule_fsts",
    "endpoint_prosody_model",
    "denoise_model",
    "aec_model",
    "kws_tokens",
    "kws_encoder",
    "kws_decoder",
    "kws_joiner",
    "kws_keywords_file",
)
_ARTIFACT_LIST_FIELDS = {
    "tts_lexicon",
    "asr_final_hr_rule_fsts",
    "asr_final_rule_fsts",
}
_MAX_ARTIFACT_FILES = 100_000
_MAX_ARTIFACT_DIRECTORIES = 100_000
_MAX_ARTIFACT_ENTRIES = _MAX_ARTIFACT_FILES + _MAX_ARTIFACT_DIRECTORIES
_MAX_ARTIFACT_BYTES = 32 * 1024 * 1024 * 1024
_MAX_ARTIFACT_DEPTH = 32
_MAX_ARTIFACT_RELATIVE_PATH_BYTES = 4096
_SAFE_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,127}\Z")


class ProductionInputError(RuntimeError):
    """A detail-free production-input validation failure."""


@dataclass(frozen=True)
class PreparedProduction:
    config: dict = field(repr=False)
    device_profile: str
    final_stt_profile: str
    final_stt_profile_sha256: str
    final_stt_profile_schema_version: int
    production_config_sha256: str
    execution_config_sha256: str
    replay_safety_schema_version: int
    replay_safety_sha256: str
    active_artifacts_sha256: str
    active_artifact_roots: int
    active_artifact_files: int
    active_artifact_bytes: int
    source_revision: str
    runtime_identities: dict[str, object]


@dataclass(frozen=True)
class BoundOllamaIdentity:
    sha256: str
    role_contracts: dict[str, dict[str, str]]
    snapshot: dict[str, object] = field(repr=False)


def _canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise ProductionInputError() from None
    return hashlib.sha256(encoded).hexdigest()


def _apply_replay_safety(config: dict) -> dict:
    """Keep production control flow while excluding external replay effects."""
    llm = config.get("llm", {}) or {}
    if str(llm.get("backend", "") or "").lower() != "ollama":
        raise ProductionInputError()
    overlay: dict[str, object] = {
        "memory": {"backend": "session"},
        "llm": {
            "host": LOCAL_OLLAMA_HOST,
            "cloud": {"enabled": False, "strategy": "local_only"},
            "live_routing": False,
            # The normal router factory has no evaluation-header injection seam.
            # Nulling this optional third client keeps the bound main/fast closure
            # exact; the capability router still uses the bound fast tier.
            "router_model": None,
        },
    }
    for name in _REPLAY_SAFETY_POLICY["disabled_external_providers"]:
        overlay[str(name)] = {"enabled": False}
    return deep_merge(config, overlay)


def _metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return _metadata_identity(metadata)


def _hash_regular_at(
    parent_descriptor: int,
    name: str,
    *,
    maximum_bytes: int,
) -> tuple[bytes, int]:
    descriptor = -1
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ProductionInputError()
        if before.st_size < 0 or before.st_size > maximum_bytes:
            raise ProductionInputError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        opened_identity = _metadata_identity(opened)
        if opened_identity != _metadata_identity(before):
            raise ProductionInputError()
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise ProductionInputError()
            digest.update(chunk)
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            _metadata_identity(after) != opened_identity
            or _metadata_identity(current) != opened_identity
        ):
            raise ProductionInputError()
        if total != int(after.st_size):
            raise ProductionInputError()
        return digest.digest(), total
    except (OSError, OverflowError, ValueError, ProductionInputError):
        raise ProductionInputError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _hash_regular_file(
    path: Path,
    *,
    maximum_bytes: int = _MAX_ARTIFACT_BYTES,
) -> tuple[bytes, int]:
    try:
        selected = Path(os.path.abspath(path.expanduser()))
        if not selected.name:
            raise ProductionInputError()
        with opened_directory_nofollow(
            selected.parent,
            require_private=False,
        ) as (_stable_parent, parent_descriptor):
            return _hash_regular_at(
                parent_descriptor,
                selected.name,
                maximum_bytes=maximum_bytes,
            )
    except (BoundedReadError, OSError, RuntimeError, ValueError, ProductionInputError):
        raise ProductionInputError() from None


def _artifact_name(name: object) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\x00" in name
    ):
        raise ProductionInputError()
    return name


def _scan_artifact_directory(
    directory_descriptor: int,
    *,
    prefix: str,
    depth: int,
    state: dict[str, int],
    rows: list[tuple[str, bytes, int]],
) -> None:
    if depth > _MAX_ARTIFACT_DEPTH:
        raise ProductionInputError()
    opened_identity = _directory_identity(os.fstat(directory_descriptor))
    names: list[str] = []
    with os.scandir(directory_descriptor) as entries:
        for entry in entries:
            names.append(_artifact_name(entry.name))
            state["entries"] += 1
            if state["entries"] > state["maximum_entries"]:
                raise ProductionInputError()
    names.sort()
    for name in names:
        relative = f"{prefix}/{name}" if prefix else name
        if len(relative.encode("utf-8")) > _MAX_ARTIFACT_RELATIVE_PATH_BYTES:
            raise ProductionInputError()
        before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if stat.S_ISDIR(before.st_mode):
            state["directories"] += 1
            if state["directories"] > state["maximum_directories"]:
                raise ProductionInputError()
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            child_descriptor = os.open(name, flags, dir_fd=directory_descriptor)
            try:
                child_identity = _directory_identity(os.fstat(child_descriptor))
                if child_identity != _directory_identity(before):
                    raise ProductionInputError()
                _scan_artifact_directory(
                    child_descriptor,
                    prefix=relative,
                    depth=depth + 1,
                    state=state,
                    rows=rows,
                )
                current = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (
                    _directory_identity(os.fstat(child_descriptor)) != child_identity
                    or _directory_identity(current) != child_identity
                ):
                    raise ProductionInputError()
            finally:
                os.close(child_descriptor)
        elif stat.S_ISREG(before.st_mode):
            state["files"] += 1
            if state["files"] > state["maximum_files"]:
                raise ProductionInputError()
            remaining = state["maximum_bytes"] - state["bytes"]
            if remaining < 0:
                raise ProductionInputError()
            content, size = _hash_regular_at(
                directory_descriptor,
                name,
                maximum_bytes=remaining,
            )
            state["bytes"] += size
            rows.append((relative, content, size))
        else:
            raise ProductionInputError()
    if _directory_identity(os.fstat(directory_descriptor)) != opened_identity:
        raise ProductionInputError()


def _hash_artifact_tree(
    root: Path,
    *,
    maximum_files: int,
    maximum_directories: int,
    maximum_bytes: int,
) -> tuple[tuple[tuple[str, bytes, int], ...], int]:
    try:
        state = {
            "entries": 0,
            "files": 0,
            "directories": 0,
            "bytes": 0,
            "maximum_entries": min(
                _MAX_ARTIFACT_ENTRIES,
                maximum_files + maximum_directories,
            ),
            "maximum_files": maximum_files,
            "maximum_directories": maximum_directories,
            "maximum_bytes": maximum_bytes,
        }
        rows: list[tuple[str, bytes, int]] = []
        with opened_directory_nofollow(
            root,
            require_private=False,
        ) as (stable_root, root_descriptor):
            root_identity = _directory_identity(os.fstat(root_descriptor))
            _scan_artifact_directory(
                root_descriptor,
                prefix="",
                depth=0,
                state=state,
                rows=rows,
            )
            current = stable_root.lstat()
            if (
                _directory_identity(os.fstat(root_descriptor)) != root_identity
                or _directory_identity(current) != root_identity
            ):
                raise ProductionInputError()
        if not rows:
            raise ProductionInputError()
        return tuple(rows), state["directories"]
    except (
        BoundedReadError,
        OSError,
        RuntimeError,
        UnicodeError,
        ValueError,
        OverflowError,
        ProductionInputError,
    ):
        raise ProductionInputError() from None


def _active_artifact_values(config: dict) -> tuple[tuple[str, str], ...]:
    from core.engines.sherpa import SherpaConfig

    sherpa = SherpaConfig.from_dict(config.get("sherpa", {}))
    rows: list[tuple[str, str]] = []
    for name in _ARTIFACT_FIELDS:
        if name == "asr_bpe_vocab":
            unit = str(sherpa.asr_modeling_unit or "").strip().lower()
            if not (
                str(sherpa.asr_hotwords or "").strip()
                and sherpa.asr_decoding_method == "modified_beam_search"
                and unit in {"bpe", "cjkchar+bpe"}
            ):
                continue
        if name == "asr_final_verifier_model" and not str(
            sherpa.asr_final_verifier_backend or ""
        ).strip():
            continue
        if name == "endpoint_prosody_model" and sherpa.endpoint_detector != "prosody":
            continue
        if name == "denoise_model" and not sherpa.denoise_enabled:
            continue
        if name == "aec_model" and not sherpa.aec_enabled:
            continue
        if name.startswith("kws_") and not str(sherpa.kws_tokens or "").strip():
            continue
        raw = str(getattr(sherpa, name, "") or "").strip()
        if not raw:
            continue
        values = (
            tuple(part.strip() for part in raw.split(",") if part.strip())
            if name in _ARTIFACT_LIST_FIELDS
            else (raw,)
        )
        rows.extend((name, value) for value in values)
    return tuple(rows)


def active_artifact_identity(
    config: dict,
    *,
    config_root: Path,
) -> tuple[str, int, int, int]:
    """Hash every active local audio-model file/tree without exposing paths."""
    digest = hashlib.sha256()
    roots = files = directories = total_bytes = 0
    try:
        for field_name, raw_path in _active_artifact_values(config):
            selected = Path(raw_path).expanduser()
            if not selected.is_absolute():
                selected = config_root / selected
            metadata = selected.lstat()
            digest.update(field_name.encode("ascii"))
            digest.update(b"\0")
            roots += 1
            if stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                content, size = _hash_regular_file(
                    selected,
                    maximum_bytes=_MAX_ARTIFACT_BYTES - total_bytes,
                )
                digest.update(b"file\0")
                digest.update(content)
                files += 1
                total_bytes += size
            elif stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                rows, nested_directories = _hash_artifact_tree(
                    selected,
                    maximum_files=_MAX_ARTIFACT_FILES - files,
                    maximum_directories=_MAX_ARTIFACT_DIRECTORIES - directories,
                    maximum_bytes=_MAX_ARTIFACT_BYTES - total_bytes,
                )
                digest.update(b"tree\0")
                for relative_path, content, size in rows:
                    relative = relative_path.encode("utf-8")
                    digest.update(hashlib.sha256(relative).digest())
                    digest.update(content)
                    files += 1
                    total_bytes += size
                directories += nested_directories
            else:
                raise ProductionInputError()
            if files > _MAX_ARTIFACT_FILES or total_bytes > _MAX_ARTIFACT_BYTES:
                raise ProductionInputError()
        if roots == 0 or files == 0 or total_bytes <= 0:
            raise ProductionInputError()
        return digest.hexdigest(), roots, files, total_bytes
    except (OSError, RuntimeError, UnicodeError, ValueError, ProductionInputError):
        raise ProductionInputError() from None


def runtime_identities(config: dict) -> dict[str, object]:
    """Bind the executing Python plus the active provider distributions."""
    try:
        backend = str((config.get("llm", {}) or {}).get("backend", "")).lower()
        if backend != "ollama":
            raise ProductionInputError()
        sherpa = config.get("sherpa", {}) or {}
        distributions = [
            ("numpy", "numpy"),
            ("sherpa_onnx", "sherpa-onnx"),
            ("onnxruntime", "onnxruntime"),
            ("ollama", "ollama"),
        ]
        if str(sherpa.get("asr_final_verifier_backend", "") or "").strip() == "faster_whisper":
            distributions.extend(
                (("faster_whisper", "faster-whisper"), ("ctranslate2", "ctranslate2"))
            )
        executable = Path(sys.executable).resolve(strict=True)
        executable_digest, _size = _hash_regular_file(executable)
        bound: dict[str, object] = {}
        for label, name in distributions:
            distribution = importlib_metadata.distribution(name)
            version = distribution.version
            record = distribution.read_text("RECORD")
            if (
                not isinstance(version, str)
                or _SAFE_VERSION_RE.fullmatch(version) is None
                or not isinstance(record, str)
                or not record
                or len(record.encode("utf-8")) > 2 * 1024 * 1024
            ):
                raise ProductionInputError()
            bound[label] = {
                "version": version,
                "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
            }
        return {
            "python": {
                "version": (
                    f"{sys.version_info.major}.{sys.version_info.minor}."
                    f"{sys.version_info.micro}"
                ),
                "executable_sha256": executable_digest.hex(),
            },
            "distributions": bound,
        }
    except (OSError, ValueError, importlib_metadata.PackageNotFoundError, ProductionInputError):
        raise ProductionInputError() from None


def _source_revision(reader: Callable[..., dict[str, object]]) -> str:
    try:
        metadata = reader(Path(__file__).resolve().parents[2])
        revision = metadata.get("revision") if isinstance(metadata, dict) else None
        dirty = metadata.get("dirty") if isinstance(metadata, dict) else None
        if (
            not isinstance(revision, str)
            or re.fullmatch(r"[0-9a-f]{40,64}", revision) is None
            or dirty is not False
        ):
            raise ProductionInputError()
        return revision
    except (OSError, ValueError, ProductionInputError):
        raise ProductionInputError() from None


def bind_ollama_identity(
    config: dict,
    *,
    snapshot_reader: Callable[[dict[str, str], dict], dict[str, object]] | None = None,
) -> BoundOllamaIdentity:
    """Bind main/fast aliases to strict immutable blob/config contracts."""
    llm = config.get("llm", {}) or {}
    if str(llm.get("backend", "") or "").lower() != "ollama":
        raise ProductionInputError()
    role_models = {
        "main": str(llm.get("main_model", "") or ""),
        "fast": str(llm.get("fast_model", "") or ""),
    }
    if not all(role_models.values()):
        raise ProductionInputError()
    reader = snapshot_reader or identity_snapshot
    try:
        snapshot = reader(role_models, config)
        if not isinstance(snapshot, dict) or snapshot.get("ok") is not True:
            raise ProductionInputError()
        models = snapshot.get("models")
        if not isinstance(models, Mapping):
            raise ProductionInputError()
        contracts: dict[str, dict[str, str]] = {}
        for role, model in role_models.items():
            contract = full_identity_contract(models.get(model), model)
            if contract is None:
                raise ProductionInputError()
            contracts[role] = {
                "blob_sha256": contract[0],
                "effective_config_sha256": contract[1],
            }
        return BoundOllamaIdentity(
            sha256=_canonical_sha256(contracts),
            role_contracts=contracts,
            snapshot=snapshot,
        )
    except (OSError, TypeError, ValueError, ProductionInputError):
        raise ProductionInputError() from None


def verify_ollama_identity(
    before: BoundOllamaIdentity,
    config: dict,
    *,
    snapshot_reader: Callable[[dict[str, str], dict], dict[str, object]] | None = None,
) -> None:
    after = bind_ollama_identity(config, snapshot_reader=snapshot_reader)
    if after.sha256 != before.sha256 or after.role_contracts != before.role_contracts:
        raise ProductionInputError()


def prepare_production(
    *,
    config_path: str | Path,
    local_config_path: str | Path,
    device_profile: str,
    final_stt_profile: str,
    artifact_reader: Callable[..., tuple[str, int, int, int]] | None = None,
    repository_reader: Callable[..., dict[str, object]] | None = None,
    runtime_reader: Callable[[dict], dict[str, object]] | None = None,
) -> PreparedProduction:
    """Resolve one concrete device + atomic final-STT configuration."""
    if not device_profile or device_profile == "auto" or not final_stt_profile:
        raise ProductionInputError()
    try:
        loaded = load_config(str(config_path), local=str(local_config_path))
        device_config = apply_device_profile(
            loaded,
            device_profile,
            strict=True,
        )
        production_config, metadata = apply_final_stt_profile(
            device_config,
            final_stt_profile,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        raise ProductionInputError() from None
    production_digest = _canonical_sha256(production_config)
    execution_config = _apply_replay_safety(production_config)
    safety_digest = _canonical_sha256(_REPLAY_SAFETY_POLICY)
    config_root = Path(config_path).expanduser().resolve(strict=True).parent
    artifacts = (artifact_reader or active_artifact_identity)(
        execution_config,
        config_root=config_root,
    )
    revision = _source_revision(repository_reader or repository_metadata)
    runtime = (runtime_reader or runtime_identities)(execution_config)
    return PreparedProduction(
        config=execution_config,
        device_profile=device_profile,
        final_stt_profile=metadata.name,
        final_stt_profile_sha256=metadata.sha256,
        final_stt_profile_schema_version=metadata.schema_version,
        production_config_sha256=production_digest,
        execution_config_sha256=_canonical_sha256(execution_config),
        replay_safety_schema_version=int(_REPLAY_SAFETY_POLICY["schema_version"]),
        replay_safety_sha256=safety_digest,
        active_artifacts_sha256=artifacts[0],
        active_artifact_roots=artifacts[1],
        active_artifact_files=artifacts[2],
        active_artifact_bytes=artifacts[3],
        source_revision=revision,
        runtime_identities=runtime,
    )


def load_production_corpus(path: str | Path) -> LoadedCorpus:
    """Load only the receipt-aware schema families used by retained evidence."""
    try:
        corpus = load_corpus(path)
        if corpus.schema_version not in {2, 3, 4}:
            raise ProductionInputError()
        verify_corpus_snapshot(corpus)
        return corpus
    except (CorpusError, OSError, ValueError, ProductionInputError):
        raise ProductionInputError() from None


def verify_production_inputs(
    prepared: PreparedProduction,
    corpus: LoadedCorpus,
    *,
    config_path: str | Path,
    local_config_path: str | Path,
    artifact_reader: Callable[..., tuple[str, int, int, int]] | None = None,
    repository_reader: Callable[..., dict[str, object]] | None = None,
    runtime_reader: Callable[[dict], dict[str, object]] | None = None,
) -> None:
    """Fail closed if the corpus or effective configuration changed in-run."""
    try:
        verify_corpus_snapshot(corpus)
        current = prepare_production(
            config_path=config_path,
            local_config_path=local_config_path,
            device_profile=prepared.device_profile,
            final_stt_profile=prepared.final_stt_profile,
            artifact_reader=artifact_reader,
            repository_reader=repository_reader,
            runtime_reader=runtime_reader,
        )
    except (CorpusError, ProductionInputError, OSError, ValueError):
        raise ProductionInputError() from None
    if (
        current.final_stt_profile_sha256 != prepared.final_stt_profile_sha256
        or current.production_config_sha256 != prepared.production_config_sha256
        or current.execution_config_sha256 != prepared.execution_config_sha256
        or current.replay_safety_sha256 != prepared.replay_safety_sha256
        or current.active_artifacts_sha256 != prepared.active_artifacts_sha256
        or current.active_artifact_roots != prepared.active_artifact_roots
        or current.active_artifact_files != prepared.active_artifact_files
        or current.active_artifact_bytes != prepared.active_artifact_bytes
        or current.source_revision != prepared.source_revision
        or current.runtime_identities != prepared.runtime_identities
    ):
        raise ProductionInputError()


def corpus_identity(corpus: LoadedCorpus) -> dict[str, object]:
    """Return path/text-free source identity for the aggregate report."""
    return {
        "schema_version": corpus.schema_version,
        "sha256": corpus.digest,
        "case_count": len(corpus.cases),
        "audio_bytes": corpus.audio_bytes,
        "provenance": (
            corpus.provenance.as_dict() if corpus.provenance is not None else None
        ),
    }
