"""Explicit session performance policy; model/resources only, no capability policy."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .config import deep_merge

PERFORMANCE_MODE_NAMES = ("current", "responsive", "compact")
_PATH_KEYS = frozenset({
    "asr_encoder", "asr_decoder", "asr_joiner", "asr_tokens",
    "tts_model", "tts_tokens", "tts_data_dir", "tts_voices", "tts_lexicon",
})
_SHERPA_KEYS = _PATH_KEYS | {
    "tts_backend", "tts_speaker_id",
    "asr_num_threads", "tts_num_threads",
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MAX_FILE_BYTES = 512 * 1024**2
_MAX_TREE_BYTES = 64 * 1024**2


class PerformanceModeError(ValueError):
    """A mode cannot be applied atomically with its exact installed assets."""


@dataclass(frozen=True)
class PerformanceModeMetadata:
    name: str
    sha256: str
    schema_version: int = 1


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _fingerprint(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _file_hash(path: Path, maximum: int = _MAX_FILE_BYTES) -> tuple[int, str, tuple[int, ...]]:
    named_before = path.lstat()
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (_fingerprint(named_before) != _fingerprint(before) or
                not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum):
            raise PerformanceModeError("performance asset must be a bounded nonempty regular file")
        digest = hashlib.sha256()
        read = 0
        while chunk := stream.read(1024 * 1024):
            read += len(chunk)
            if read > maximum:
                raise PerformanceModeError("performance asset exceeds its byte budget")
            digest.update(chunk)
        if (read != before.st_size or
                _fingerprint(before) != _fingerprint(os.fstat(stream.fileno())) or
                _fingerprint(before) != _fingerprint(path.lstat())):
            raise PerformanceModeError("performance asset changed while verifying")
        return read, digest.hexdigest(), _fingerprint(before)


def asset_sha256(path: Path) -> str:
    """File SHA or bounded exact directory inventory SHA, without native imports."""
    try:
        info = path.lstat()
        if stat.S_ISREG(info.st_mode):
            return _file_hash(path)[1]
        if not stat.S_ISDIR(info.st_mode):
            raise PerformanceModeError("performance asset type is unsupported")
        inventory = []
        snapshots = [(path, _fingerprint(info))]
        pending = [(path, _fingerprint(info))]
        total = entries = 0
        while pending:
            directory, identity = pending.pop()
            if _fingerprint(directory.lstat()) != identity:
                raise PerformanceModeError("performance asset directory changed while verifying")
            # Snapshot before enumeration and bound entries incrementally. This
            # also propagates permission/traversal errors instead of omitting data.
            with os.scandir(directory) as children:
                for entry in children:
                    entries += 1
                    if entries > 2048:
                        raise PerformanceModeError("performance asset tree is oversized")
                    leaf = directory / entry.name
                    named = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(named.st_mode):
                        child_identity = _fingerprint(named)
                        snapshots.append((leaf, child_identity))
                        pending.append((leaf, child_identity))
                    elif stat.S_ISREG(named.st_mode):
                        size, digest, file_identity = _file_hash(leaf, _MAX_TREE_BYTES)
                        if _fingerprint(named) != file_identity:
                            raise PerformanceModeError("performance asset changed while verifying")
                        snapshots.append((leaf, file_identity))
                        total += size
                        if total > _MAX_TREE_BYTES or len(inventory) >= 1024:
                            raise PerformanceModeError("performance asset tree exceeds its byte budget")
                        inventory.append({"file": leaf.relative_to(path).as_posix(), "bytes": size, "sha256": digest})
                    else:
                        raise PerformanceModeError("performance asset tree contains an unsupported entry")
        if any(_fingerprint(leaf.lstat()) != identity for leaf, identity in snapshots):
            raise PerformanceModeError("performance asset tree changed while verifying")
        # Some filesystems expose coarse/stable directory timestamps. Recheck
        # the bounded membership set, rather than relying on mtime for additions.
        expected_members = {leaf.relative_to(path).as_posix() for leaf, _ in snapshots if leaf != path}
        actual_members = set()
        pending_paths = [path]
        while pending_paths:
            directory = pending_paths.pop()
            with os.scandir(directory) as children:
                for entry in children:
                    leaf = directory / entry.name
                    actual_members.add(leaf.relative_to(path).as_posix())
                    if len(actual_members) > 2048:
                        raise PerformanceModeError("performance asset tree is oversized")
                    member = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(member.st_mode):
                        pending_paths.append(leaf)
                    elif not stat.S_ISREG(member.st_mode):
                        raise PerformanceModeError("performance asset tree contains an unsupported entry")
        if actual_members != expected_members:
            raise PerformanceModeError("performance asset tree changed while verifying")
        if any(_fingerprint(leaf.lstat()) != identity for leaf, identity in snapshots):
            raise PerformanceModeError("performance asset tree changed while verifying")
        if not inventory:
            raise PerformanceModeError("performance asset directory is empty")
        inventory.sort(key=lambda item: item["file"])
        return hashlib.sha256(_canonical(inventory)).hexdigest()
    except (OSError, ValueError, OverflowError) as error:
        if isinstance(error, PerformanceModeError):
            raise
        raise PerformanceModeError("performance asset is unavailable or changed; provision the complete mode") from None


def _validate_profile(mode: str, profile: object) -> dict:
    if type(profile) is not dict or set(profile) != {
        "schema_version", "sherpa", "llm", "warm_start_policy", "asset_sha256"
    }:
        raise PerformanceModeError("performance profile must contain its complete model/resource contract")
    if type(profile["schema_version"]) is not int or profile["schema_version"] != 1:
        raise PerformanceModeError("performance profile schema_version must be 1")
    sherpa = profile["sherpa"]
    if type(sherpa) is not dict or set(sherpa) != _SHERPA_KEYS:
        raise PerformanceModeError("performance profile has incomplete or unsupported sherpa fields")
    for key in _PATH_KEYS:
        value = sherpa[key]
        if type(value) is not str or "\x00" in value or len(value) > 4096:
            raise PerformanceModeError("performance asset path must be a bounded string")
        if key not in {"tts_voices", "tts_lexicon"} and not value:
            raise PerformanceModeError("performance model family is incomplete")
    backend = "kitten" if mode == "compact" else ""
    if type(sherpa["tts_backend"]) is not str or sherpa["tts_backend"] != backend:
        raise PerformanceModeError("performance mode has the wrong TTS family selector")
    if (mode == "compact") != bool(sherpa["tts_voices"]) or sherpa["tts_lexicon"]:
        raise PerformanceModeError("performance mode has inconsistent TTS assets")
    if type(sherpa["tts_speaker_id"]) is not int or sherpa["tts_speaker_id"] != 0:
        raise PerformanceModeError("performance mode requires its measured speaker 0")
    for key in ("asr_num_threads", "tts_num_threads"):
        if type(sherpa[key]) is not int or not 1 <= sherpa[key] <= 8:
            raise PerformanceModeError("performance threads must be between 1 and 8")
    llm = profile["llm"]
    if type(llm) is not dict or set(llm) != {"main_keep_alive", "fast_keep_alive"}:
        raise PerformanceModeError("performance profile may only tune LLM residency")
    for value in llm.values():
        if not ((type(value) is int and -1 <= value <= 86400) or
                (type(value) is str and re.fullmatch(r"[1-9][0-9]{0,4}[smh]", value))):
            raise PerformanceModeError("performance keep_alive is invalid")
    if type(profile["warm_start_policy"]) is not str or profile["warm_start_policy"] != "fast":
        raise PerformanceModeError("performance profile requires fast-only startup warming")
    hashes = profile["asset_sha256"]
    nonempty = {key for key in _PATH_KEYS if sherpa[key]}
    if type(hashes) is not dict or set(hashes) != nonempty:
        raise PerformanceModeError("performance asset checksum contract is incomplete")
    if any(type(value) is not str or not _SHA.fullmatch(value) for value in hashes.values()):
        raise PerformanceModeError("performance asset checksum is invalid")
    return profile


def apply_performance_mode(config: dict, requested: str | None = None, *,
                           root: Path | None = None) -> tuple[dict, PerformanceModeMetadata | None]:
    """Apply one model/resource mode after device selection, before final-STT selection.

    Current is an exact no-op. Every optimized mode verifies all bound assets
    before publishing any override. No endpoint, DSP, permissions, cloud,
    memory, context budget, final recognizer/verifier or model routing changes.
    """
    mode = requested if requested is not None else config.get("performance_mode", "current")
    if type(mode) is not str or mode not in PERFORMANCE_MODE_NAMES:
        raise PerformanceModeError("unknown performance mode; use current, responsive or compact")
    if mode == "current":
        return config, None
    profiles = config.get("performance_profiles")
    if type(profiles) is not dict or set(profiles) != {"responsive", "compact"}:
        raise PerformanceModeError("performance profiles are incomplete; provision both explicit modes")
    # Copy the bounded scalar-only contract before any filesystem work can yield.
    # Revalidate the snapshot so a concurrent edit cannot bypass its allowlist.
    _validate_profile(mode, profiles[mode])
    try:
        selected = {key: dict(value) if type(value) is dict else value
                    for key, value in profiles[mode].items()}
    except RuntimeError:
        raise PerformanceModeError("performance profile changed while selecting") from None
    selected = _validate_profile(mode, selected)
    base = root if root is not None else Path.cwd()
    bound = dict(selected["sherpa"])
    inherited = config.get("sherpa", {})
    if type(inherited) is not dict or type(config.get("llm", {})) is not dict:
        raise PerformanceModeError("performance mode requires typed existing model settings")
    phrases = inherited.get("asr_hotwords", "")
    if type(phrases) is not str:
        raise PerformanceModeError("performance mode requires a typed streaming hotword contract")
    if any(line.strip() for line in phrases.splitlines()):
        expected_vocab = "f191a4935f668fa8cd8e607bcd378404f948321cd3134a5ea13d324ba921673d"
        expected_stream = {
            "asr_encoder": "563fde436d16cf7607cf408cd6b30909819d03162652ef389c2450ced3f45ac1",
            "asr_decoder": "7bf787f90b194b307e5a4ad6a34fadb4e748304c35f78a8d66358a05b13ee6ef",
            "asr_joiner": "210591f72b3c56b8364f85f345dca240bc2b4c00632848f4aa923630d5639d3b",
            "asr_tokens": "49e3c2646595fd907228b3c6787069658f67b17377c60aeb8619c4551b2316fb",
        }
        if (inherited.get("asr_modeling_unit") != "bpe" or
                inherited.get("asr_hotwords_case_policy") != "upper_ascii_words" or
                inherited.get("asr_decoding_method") != "modified_beam_search" or
                inherited.get("asr_bpe_vocab_sha256") != expected_vocab or
                any(selected["asset_sha256"][key] != value for key, value in expected_stream.items()) or
                type(inherited.get("asr_bpe_vocab")) is not str):
            raise PerformanceModeError("performance mode cannot preserve incompatible active English BPE hotwords")
        vocabulary = Path(inherited["asr_bpe_vocab"])
        if not vocabulary.is_absolute():
            vocabulary = base / vocabulary
        if asset_sha256(vocabulary) != expected_vocab:
            raise PerformanceModeError("performance mode hotword vocabulary checksum mismatch")
        bound["asr_bpe_vocab"] = str(vocabulary)
    for key, expected in selected["asset_sha256"].items():
        path = Path(bound[key])
        if not path.is_absolute():
            path = base / path
        if asset_sha256(path) != expected:
            raise PerformanceModeError("performance asset checksum mismatch; mode was not applied")
        bound[key] = str(path)
    # The strict allowlist above prevents a resource preset from widening authority.
    merged = deep_merge(config, {
        "sherpa": bound, "llm": selected["llm"],
        "warm_start_policy": selected["warm_start_policy"], "performance_mode": mode,
    })
    return merged, PerformanceModeMetadata(mode, hashlib.sha256(_canonical(selected)).hexdigest())
