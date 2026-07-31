#!/usr/bin/env python3
"""Prepare small, deterministic public-audio regression fixtures on demand.

The source archives and generated ``.npy`` clips stay in ``.cache/``.  Nothing
downloaded by this tool is a final held-out set, owner-voice evidence, or live
hardware evidence.  Network access is opt-in with ``--accept-download``.

Examples::

    python -m tools.public_voice_fixtures list
    python -m tools.public_voice_fixtures prepare --suite commands --accept-download
    python -m tools.public_voice_fixtures prepare --suite conversation \
        --source ami-es2002a-mix-headset=/path/to/ES2002a.Mix-Headset.wav \
        --source ami-manual-annotations-162=/path/to/annotations.zip --offline
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
import re
import signal
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from time import monotonic, sleep
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlparse
from xml.etree import ElementTree

try:
    import fcntl
except ImportError:  # pragma: no cover - Linux is the reference runtime
    fcntl = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = _REPO_ROOT / "tests" / "fixtures" / "public_voice_manifest.v2.json"
DEFAULT_CACHE = _REPO_ROOT / ".cache" / "public_voice"
SUITES = ("commands", "conversation", "echo")
_V3_SUITES = ("intent-asr",)
_SUITES_BY_SCHEMA = {2: SUITES, 3: _V3_SUITES}
_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,95}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_AMI_WORD_MEMBER_RE = re.compile(
    r"(?P<meeting>[A-Za-z0-9_-]+)\.(?P<speaker>[A-Z])\.words\.xml\Z"
)
_MAX_SOURCE_BYTES = 256 * 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_AUDIO_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_PARQUET_RESULT_BYTES = 128 * 1024
_MAX_PARQUET_WORKER_BYTES = 512 * 1024
_MAX_PARQUET_PROBE_BYTES = 4096
_MAX_VENV_MARKER_BYTES = 4096
_PARQUET_PROBE_TIMEOUT_SEC = 10.0
_PARQUET_WORKER_TIMEOUT_SEC = 60.0
_PARQUET_GROUP_REAP_TIMEOUT_SEC = 2.0
_PARQUET_WORKER = Path(__file__).with_name("public_voice_parquet_worker.py")
_PARQUET_PROTOCOL_VERSION = 1
_PARQUET_PROBE_SCHEMA_VERSION = 1
_PARQUET_PYARROW_VERSION = "25.0.0"
_MAX_PARQUET_AUDIO_BYTES = 8 * 1024 * 1024
_MAX_PARQUET_TOTAL_AUDIO_BYTES = 32 * 1024 * 1024
_MAX_PARQUET_CASE_SAMPLES = 30 * 16_000
_MAX_PARQUET_TOTAL_SAMPLES = 14 * _MAX_PARQUET_CASE_SAMPLES
_DOWNLOAD_CHUNK_BYTES = 64 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_FICLONE = 0x40049409
EVIDENCE_SCOPE = {
    "whole_clip_decoder_smoke": True,
    "frame_replay": False,
    "aec_behavior": False,
    "live_hardware": False,
    "owner_voice": False,
    "held_out": False,
}
_SOURCE_KINDS = frozenset({"canonical", "mirror"})
_LICENSE_IDS = frozenset({"Apache-2.0", "CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0", "MIT"})
_REDISTRIBUTION_POLICIES = frozenset({"cache-only", "redistributable"})
_PERSONAL_DATA_CLASSES = frozenset(
    {"none", "public-pseudonymous-voice", "public-transcript-annotations"}
)
_ASSERTIONS_BY_SUITE = {
    "commands": frozenset({"transcript", "unknown_word", "empty_reference"}),
    "conversation": frozenset({"transcript", "event_only"}),
    "echo": frozenset(
        {
            "farend_reference",
            "echo_only_mic",
            "nearend_only_mic",
            "doubletalk_reference",
            "doubletalk_mic",
        }
    ),
    "intent-asr": frozenset({"transcript"}),
}
_FIXTURE_TAGS = frozenset(
    {"backchannel", "four-speaker-overlap", "laughter", "meeting", "single-speaker"}
)
_PRODUCT_ROLES = frozenset({"target", "bystander"})
_PRODUCT_TAGS = frozenset({"future-taxonomy", "multi-speaker", "near-control"})
_PRODUCT_ACTIONS = frozenset({"stop", "vault.search", "reminder.create", "timer"})
_PRODUCT_RUNTIME_IDENTIFIERS = {
    "stop": (
        "always_on_agent.models.IntentKind.STOP",
        "always_on_agent.speech_analyzer.exact_control_class:stop",
        "always_on_agent.events.EventKind.CONTROL_STOP",
    ),
    "vault.search": (
        "always_on_agent.models.IntentKind.SEARCH",
        "always_on_agent.speech_analyzer.search_scope:vault",
        "always_on_agent.planner.PlanStep:vault.search",
        "core.obsidian.ObsidianVault._terms",
    ),
    "reminder.create": (
        "core.reminders.ReminderParser.parse",
        "core.trusted_apps.DeviceToolCommandDispatcher:reminder.create",
        "always_on_agent.planner.PlanStep:device.command",
    ),
    "timer": ("core.intents.IntentGrammar:timer",),
    None: ("always_on_agent.speech_analyzer.exact_control_class:none",),
}
_GROUP_CONTRACTS = {
    "farend_single_talk": {
        "farend_reference": "farend_reference",
        "microphone": "echo_only_mic",
    },
    "nearend_single_talk": {"microphone": "nearend_only_mic"},
    "double_talk": {
        "farend_reference": "doubletalk_reference",
        "microphone": "doubletalk_mic",
    },
}
_COMMAND_WORDS = (
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
)
_MINDS_INTENTS = (
    "abroad",
    "address",
    "app_error",
    "atm_limit",
    "balance",
    "business_loan",
    "card_issues",
    "cash_deposit",
    "direct_debit",
    "freeze",
    "high_value_payment",
    "joint_account",
    "latest_transactions",
    "pay_bill",
)
_MINDS_SOURCE_SHA256 = (
    "37004471dc896ce20771b3fdda0ee8fb33ec7a030fb4e2fd047c561ec0a1ee30"
)
_MINDS_SOURCE_REVISION = "40ce77cb32a384e4d50a568e1ec39ac804019d33"
_MINDS_SOURCE_SIZE_BYTES = 34_196_221
_MINDS_SOURCE_ROWS = 563
_MINDS_SOURCE_ROW_GROUPS = 6
_MINDS_LANGUAGE_ID = 4
_MINDS_SELECTION_SEED = "speaker-public-minds14-v1"
_PARQUET_RESULT_FIELDS = {
    "schema_version",
    "source_sha256",
    "source_size_bytes",
    "source_row_count",
    "source_row_groups",
    "pyarrow_version",
    "cases",
}
_PARQUET_CASE_FIELDS = {
    "fixture_id",
    "intent_id",
    "intent_label",
    "language_id",
    "language",
    "row_index",
    "source_path",
    "transcription",
    "transcription_sha256",
    "audio_file",
    "audio_sha256",
    "audio_size_bytes",
    "selection_rank_sha256",
}
_PARQUET_PROBE_FIELDS = {
    "schema_version",
    "prefix",
    "base_prefix",
    "enable_user_site",
    "sys_path",
    "pyarrow_version",
    "pyarrow_file",
}
_PARQUET_PROBE_CODE = r"""
import json
import os
import site
import sys

try:
    import pyarrow

    def resolved(value):
        return os.path.realpath(os.path.abspath(os.fspath(value)))

    payload = {
        "schema_version": 1,
        "prefix": resolved(sys.prefix),
        "base_prefix": resolved(sys.base_prefix),
        "enable_user_site": site.ENABLE_USER_SITE,
        "sys_path": [resolved(value or os.getcwd()) for value in sys.path],
        "pyarrow_version": pyarrow.__version__,
        "pyarrow_file": resolved(pyarrow.__file__),
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    os.write(1, encoded)
except BaseException:
    raise SystemExit(73)
"""


class PublicFixtureError(RuntimeError):
    """A fail-closed manifest, source, annotation, or extraction error."""


class DownloadApprovalRequired(PublicFixtureError):
    """Raised before network access when the caller did not opt in."""

    def __init__(self, size_bytes: int):
        self.size_bytes = int(size_bytes)
        super().__init__(
            f"{self.size_bytes} source bytes are missing; rerun with "
            "--accept-download or provide --source ID=PATH"
        )


@dataclass(frozen=True)
class PublicVoiceManifest:
    path: Path
    raw: bytes
    data: Mapping[str, object]
    sources: tuple[Mapping[str, object], ...]
    fixtures: tuple[Mapping[str, object], ...]
    groups: tuple[Mapping[str, object], ...]
    product_scenarios: tuple[Mapping[str, object], ...]

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()

    @property
    def source_by_id(self) -> dict[str, Mapping[str, object]]:
        return {str(item["id"]): item for item in self.sources}


@dataclass(frozen=True)
class _SpeechMember:
    name: str
    label: str
    speaker_id: str | None
    rank: str
    size: int
    offset: int


@dataclass(frozen=True)
class _PreparedAudio:
    samples: object
    expected_text: str
    selection: Mapping[str, object]


def _require_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise PublicFixtureError(f"invalid {field}")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value or "")
    if _SHA256_RE.fullmatch(text) is None:
        raise PublicFixtureError(f"invalid {field}")
    return text


def _require_size(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PublicFixtureError(f"invalid {field}")
    if value <= 0 or value > _MAX_SOURCE_BYTES:
        raise PublicFixtureError(f"invalid {field}")
    return value


def _require_nonempty_text(value: object, *, field: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PublicFixtureError(f"invalid {field}")
    if len(value.encode("utf-8")) > maximum or "\x00" in value:
        raise PublicFixtureError(f"invalid {field}")
    return value


def _require_https_url(value: object, *, field: str) -> str:
    text = _require_nonempty_text(value, field=field, maximum=2048)
    parsed = urlparse(text)
    if parsed.scheme != "https" or not parsed.netloc:
        raise PublicFixtureError(f"invalid {field}")
    if parsed.username is not None or parsed.password is not None:
        raise PublicFixtureError(f"{field} may not contain credentials")
    return text


def _strict_json_loads(raw: bytes | str) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> object:
        raise ValueError("non-finite JSON number")

    return json.loads(
        raw,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


@contextmanager
def _stable_regular_reader(
    path: Path,
    *,
    field: str,
    maximum: int | None,
    expected_size: int | None = None,
    require_single_link: bool = True,
) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    descriptor = -1
    handle: BinaryIO | None = None
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or (
            require_single_link and before.st_nlink != 1
        ):
            raise PublicFixtureError(f"{field} must be a regular single-link file")
        if expected_size is not None and before.st_size != expected_size:
            raise PublicFixtureError(f"{field} size does not match")
        if maximum is not None and before.st_size > maximum:
            raise PublicFixtureError(f"{field} exceeds its size limit")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or (require_single_link and opened.st_nlink != 1)
            or (expected_size is not None and opened.st_size != expected_size)
            or (maximum is not None and opened.st_size > maximum)
            or _file_identity(opened) != _file_identity(before)
        ):
            raise PublicFixtureError(f"{field} changed before reading")
        handle = os.fdopen(descriptor, "rb", buffering=0)
        descriptor = -1
        try:
            yield handle, opened
        finally:
            after = os.fstat(handle.fileno())
            try:
                current = path.lstat()
            except OSError as exc:
                raise PublicFixtureError(f"{field} changed while reading") from exc
            if _file_identity(after) != _file_identity(opened) or _file_identity(
                current
            ) != _file_identity(opened):
                raise PublicFixtureError(f"{field} changed while reading")
    except PublicFixtureError:
        raise
    except OSError as exc:
        raise PublicFixtureError(f"{field} is unavailable") from exc
    finally:
        if handle is not None:
            handle.close()
        elif descriptor >= 0:
            os.close(descriptor)


def _read_stable_regular_bytes(
    path: Path,
    *,
    field: str,
    maximum: int,
    require_single_link: bool = True,
) -> bytes:
    with _stable_regular_reader(
        path,
        field=field,
        maximum=maximum,
        require_single_link=require_single_link,
    ) as (handle, opened):
        payload = handle.read(opened.st_size + 1)
        if len(payload) != opened.st_size:
            raise PublicFixtureError(f"{field} changed while reading")
        return payload


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], *, field: str
) -> None:
    if set(value) != expected:
        raise PublicFixtureError(f"invalid {field} fields")


def _require_tags(
    value: object, *, allowed: frozenset[str], field: str
) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise PublicFixtureError(f"invalid {field}")
    tags = tuple(value)
    if len(set(tags)) != len(tags) or any(tag not in allowed for tag in tags):
        raise PublicFixtureError(f"invalid {field}")
    return tags


def _require_member_path(value: object, *, field: str) -> str:
    text = _require_nonempty_text(value, field=field, maximum=512)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise PublicFixtureError(f"invalid {field}")
    return text


def _ami_speaker_identity(member_name: str) -> tuple[str, str]:
    match = _AMI_WORD_MEMBER_RE.fullmatch(PurePosixPath(member_name).name)
    if match is None:
        raise PublicFixtureError("invalid AMI annotation speaker identity")
    return match.group("meeting").casefold(), match.group("speaker").casefold()


def _require_unique_ami_speakers(member_names: Sequence[str]) -> None:
    identities = tuple(_ami_speaker_identity(name) for name in member_names)
    if (
        len(set(identities)) != len(identities)
        or len({meeting for meeting, _speaker in identities}) != 1
    ):
        raise PublicFixtureError("duplicate or mixed AMI annotation speaker identity")


def _require_positive_number(value: object, *, field: str, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublicFixtureError(f"invalid {field}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0 or number > maximum:
        raise PublicFixtureError(f"invalid {field}")
    return number


def _validate_product_scenario(
    raw_scenario: object, *, scenario_ids: set[str]
) -> Mapping[str, object]:
    if not isinstance(raw_scenario, dict):
        raise PublicFixtureError("invalid product scenario")
    scenario_id = _require_id(raw_scenario.get("id"), field="product scenario id")
    if scenario_id in scenario_ids:
        raise PublicFixtureError("duplicate product scenario id")
    scenario_ids.add(scenario_id)
    _require_exact_keys(
        raw_scenario,
        {
            "id",
            "phrase",
            "evidence",
            "contract_status",
            "role",
            "expected_action",
            "expected_slots",
            "runtime_identifiers",
            "tags",
        },
        field="product scenario",
    )
    _require_nonempty_text(
        raw_scenario.get("phrase"), field="product scenario phrase", maximum=256
    )
    if raw_scenario.get("evidence") != "text-contract-only":
        raise PublicFixtureError("product scenarios must be text-contract-only")
    role = raw_scenario.get("role")
    action = raw_scenario.get("expected_action")
    if role not in _PRODUCT_ROLES:
        raise PublicFixtureError("invalid product scenario role")
    tags = _require_tags(
        raw_scenario.get("tags"), allowed=_PRODUCT_TAGS, field="product scenario tags"
    )
    status = raw_scenario.get("contract_status")
    runtime_identifiers = raw_scenario.get("runtime_identifiers")
    slots = raw_scenario.get("expected_slots")
    if not isinstance(slots, dict):
        raise PublicFixtureError("invalid product scenario slots")
    if not isinstance(runtime_identifiers, list) or any(
        not isinstance(item, str) for item in runtime_identifiers
    ):
        raise PublicFixtureError("invalid product runtime identifiers")
    if status == "future-taxonomy":
        if (
            action is not None
            or slots
            or runtime_identifiers
            or "future-taxonomy" not in tags
        ):
            raise PublicFixtureError("future taxonomy must not claim runtime support")
        return raw_scenario
    if status != "exact" or role != "target":
        raise PublicFixtureError("unsupported product scenario contract")
    if action is not None and action not in _PRODUCT_ACTIONS:
        raise PublicFixtureError("invalid product scenario action")
    if tuple(runtime_identifiers) != _PRODUCT_RUNTIME_IDENTIFIERS[action]:
        raise PublicFixtureError("product runtime identifiers do not match action")
    if action == "stop":
        if slots:
            raise PublicFixtureError("stop does not accept slots")
    elif action == "vault.search":
        if set(slots) != {"search_scope", "terms"} or slots["search_scope"] != "vault":
            raise PublicFixtureError("vault search requires scope and terms")
        terms = slots["terms"]
        if (
            not isinstance(terms, list)
            or not terms
            or any(not isinstance(term, str) or not term for term in terms)
        ):
            raise PublicFixtureError("invalid vault search terms")
    elif action == "reminder.create":
        if set(slots) != {"message", "delay_seconds"}:
            raise PublicFixtureError("reminder requires message and delay")
        _require_nonempty_text(slots["message"], field="reminder message", maximum=256)
        _require_size(slots["delay_seconds"], field="reminder delay")
    elif action == "timer":
        if set(slots) != {"value", "unit", "duration_seconds"}:
            raise PublicFixtureError("timer requires value, unit, and duration")
        _require_nonempty_text(slots["value"], field="timer value", maximum=32)
        _require_nonempty_text(slots["unit"], field="timer unit", maximum=32)
        _require_size(slots["duration_seconds"], field="timer duration")
    elif action is None:
        if slots or "near-control" not in tags:
            raise PublicFixtureError("exact no-control scenario must be a near-control")
    return raw_scenario


def _validate_extractor(
    raw_fixture: Mapping[str, object],
    *,
    required_source_ids: tuple[str, ...],
    schema_version: int,
) -> None:
    extract = raw_fixture.get("extract")
    if not isinstance(extract, dict):
        raise PublicFixtureError("invalid fixture extractor")
    extractor = extract.get("type")
    if extractor == "hf_audio_parquet":
        if schema_version != 3:
            raise PublicFixtureError("invalid fixture extractor")
        _require_exact_keys(
            extract,
            {
                "type",
                "config",
                "split",
                "row_count",
                "row_groups",
                "language_id",
                "intent_id",
                "intent_label",
            },
            field="Hugging Face audio Parquet extractor",
        )
        intent_id = extract.get("intent_id")
        if (
            extract.get("config") != "en-US"
            or extract.get("split") != "train"
            or extract.get("row_count") != _MINDS_SOURCE_ROWS
            or extract.get("row_groups") != _MINDS_SOURCE_ROW_GROUPS
            or extract.get("language_id") != _MINDS_LANGUAGE_ID
            or isinstance(intent_id, bool)
            or not isinstance(intent_id, int)
            or not 0 <= intent_id < len(_MINDS_INTENTS)
            or extract.get("intent_label") != _MINDS_INTENTS[intent_id]
            or required_source_ids
        ):
            raise PublicFixtureError("invalid MInDS-14 Parquet extractor")
        return
    if extractor == "speech_commands_tar":
        allowed = {"type", "target"}
        if "max_duration_sec" in extract:
            allowed.add("max_duration_sec")
        _require_exact_keys(extract, allowed, field="Speech Commands extractor")
        target = extract.get("target")
        if target not in {*_COMMAND_WORDS, "_unknown_", "_silence_"}:
            raise PublicFixtureError("invalid Speech Commands target")
        if target == "_silence_":
            _require_positive_number(
                extract.get("max_duration_sec"), field="silence duration", maximum=10.0
            )
        elif "max_duration_sec" in extract:
            raise PublicFixtureError("unexpected Speech Commands duration")
        return
    if extractor == "tar_member":
        _require_exact_keys(
            extract,
            {"type", "member", "proof_member", "proof_contains"},
            field="tar extractor",
        )
        _require_member_path(extract.get("member"), field="tar member")
        _require_member_path(extract.get("proof_member"), field="tar proof member")
        _require_nonempty_text(
            extract.get("proof_contains"), field="tar proof text", maximum=2048
        )
        return
    if extractor == "direct_wav":
        _require_exact_keys(extract, {"type"}, field="direct WAV extractor")
        return
    if extractor != "wav_segment":
        raise PublicFixtureError("invalid fixture extractor")
    _require_exact_keys(
        extract, {"type", "start_sec", "end_sec", "annotation"}, field="WAV segment"
    )
    start = _require_positive_number(
        extract.get("start_sec"), field="segment start", maximum=86_400.0
    )
    end = _require_positive_number(
        extract.get("end_sec"), field="segment end", maximum=86_400.0
    )
    if end <= start:
        raise PublicFixtureError("invalid WAV segment")
    annotation = extract.get("annotation")
    if not isinstance(annotation, dict):
        raise PublicFixtureError("invalid annotation recipe")
    allowed_annotation = {
        "source_id",
        "members",
        "min_active_speakers",
        "max_active_speakers",
        "min_concurrent_speakers",
    }
    for optional in ("expected_text", "required_vocal_type"):
        if optional in annotation:
            allowed_annotation.add(optional)
    _require_exact_keys(annotation, allowed_annotation, field="annotation recipe")
    annotation_source = _require_id(
        annotation.get("source_id"), field="annotation source"
    )
    if annotation_source not in required_source_ids:
        raise PublicFixtureError("annotation source must be a required source")
    members = annotation.get("members")
    if not isinstance(members, list) or not members:
        raise PublicFixtureError("annotation has no members")
    member_names = tuple(
        _require_member_path(member, field="annotation member") for member in members
    )
    if len(set(member_names)) != len(member_names):
        raise PublicFixtureError("duplicate annotation member")
    _require_unique_ami_speakers(member_names)
    for field in (
        "min_active_speakers",
        "max_active_speakers",
        "min_concurrent_speakers",
    ):
        value = annotation.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PublicFixtureError("invalid annotation speaker requirement")
    if annotation["max_active_speakers"] < annotation["min_active_speakers"]:
        raise PublicFixtureError("invalid annotation speaker range")
    if annotation["min_concurrent_speakers"] > annotation["max_active_speakers"]:
        raise PublicFixtureError("invalid annotation concurrency requirement")
    if "expected_text" in annotation:
        _require_nonempty_text(
            annotation["expected_text"], field="annotation expected text"
        )
    if "required_vocal_type" in annotation:
        _require_id(annotation["required_vocal_type"], field="annotation vocal type")


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> PublicVoiceManifest:
    """Load and strictly validate the committed public-fixture recipe."""

    manifest_path = Path(path)
    try:
        raw = _read_stable_regular_bytes(
            manifest_path,
            field="public voice manifest",
            maximum=_MAX_MANIFEST_BYTES,
        )
        if not raw:
            raise PublicFixtureError("public voice manifest is empty")
        data = _strict_json_loads(raw)
    except PublicFixtureError:
        raise
    except Exception as exc:  # noqa: BLE001 - converted to a stable public error
        raise PublicFixtureError("unable to load public voice manifest") from exc
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int:
        raise PublicFixtureError("unsupported public voice manifest schema")
    schema_version = int(data["schema_version"])
    if schema_version not in _SUITES_BY_SCHEMA:
        raise PublicFixtureError("unsupported public voice manifest schema")
    _require_exact_keys(
        data,
        {
            "schema_version",
            "purpose",
            "selection_seed",
            "output_sample_rate",
            "evidence_scope",
            "product_scenarios",
            "sources",
            "fixtures",
            "groups",
        },
        field="manifest",
    )
    purpose = data.get("purpose")
    if not isinstance(purpose, str) or "never a final held-out" not in purpose:
        raise PublicFixtureError("manifest must state its development-only purpose")
    seed = data.get("selection_seed")
    if not isinstance(seed, str) or not seed or len(seed.encode("utf-8")) > 128:
        raise PublicFixtureError("invalid selection_seed")
    if (
        type(data.get("output_sample_rate")) is not int
        or data.get("output_sample_rate") != 16000
    ):
        raise PublicFixtureError("public replay fixtures must be 16 kHz")
    if json.dumps(
        data.get("evidence_scope"), sort_keys=True, separators=(",", ":")
    ) != json.dumps(EVIDENCE_SCOPE, sort_keys=True, separators=(",", ":")):
        raise PublicFixtureError("invalid public voice evidence scope")

    sources_raw = data.get("sources")
    fixtures_raw = data.get("fixtures")
    groups_raw = data.get("groups", [])
    product_scenarios_raw = data.get("product_scenarios")
    if not isinstance(sources_raw, list) or not sources_raw:
        raise PublicFixtureError("manifest has no sources")
    if not isinstance(fixtures_raw, list) or not fixtures_raw:
        raise PublicFixtureError("manifest has no fixtures")
    if not isinstance(groups_raw, list):
        raise PublicFixtureError("invalid groups")
    if not isinstance(product_scenarios_raw, list):
        raise PublicFixtureError("manifest has no product scenarios")
    if schema_version == 2 and not product_scenarios_raw:
        raise PublicFixtureError("manifest has no product scenarios")
    if schema_version == 3 and product_scenarios_raw:
        raise PublicFixtureError("public-v3 intent audio has no product scenarios")

    source_ids: set[str] = set()
    source_filenames: set[str] = set()
    sources: list[Mapping[str, object]] = []
    for raw_source in sources_raw:
        if not isinstance(raw_source, dict):
            raise PublicFixtureError("invalid source")
        source_id = _require_id(raw_source.get("id"), field="source id")
        if source_id in source_ids:
            raise PublicFixtureError("duplicate source id")
        source_ids.add(source_id)
        source_fields = {
            "id",
            "upstream_dataset",
            "upstream_version",
            "source_kind",
            "canonical_url",
            "download_url",
            "filename",
            "sha256",
            "size_bytes",
            "license_id",
            "license_url",
            "attribution",
            "citation",
            "redistribution",
            "human_voice",
            "personal_data",
        }
        if schema_version == 3:
            source_fields.add("acquisition")
        _require_exact_keys(raw_source, source_fields, field="source")
        filename_value = raw_source.get("filename")
        filename = filename_value if isinstance(filename_value, str) else ""
        folded_filename = filename.casefold()
        if (
            not filename
            or filename.startswith(".")
            or Path(filename).name != filename
            or "\x00" in filename
            or folded_filename in source_filenames
        ):
            raise PublicFixtureError("invalid source filename")
        source_filenames.add(folded_filename)
        for field in (
            "upstream_dataset",
            "upstream_version",
            "citation",
            "attribution",
        ):
            _require_nonempty_text(raw_source.get(field), field=f"source {field}")
        if raw_source.get("source_kind") not in _SOURCE_KINDS:
            raise PublicFixtureError("invalid source kind")
        _require_https_url(raw_source.get("canonical_url"), field="canonical URL")
        _require_https_url(raw_source.get("download_url"), field="download URL")
        _require_https_url(raw_source.get("license_url"), field="license URL")
        _require_sha256(raw_source.get("sha256"), field="source sha256")
        _require_size(raw_source.get("size_bytes"), field="source size")
        if raw_source.get("license_id") not in _LICENSE_IDS:
            raise PublicFixtureError("invalid source license id")
        redistribution = raw_source.get("redistribution")
        if redistribution not in _REDISTRIBUTION_POLICIES:
            raise PublicFixtureError("invalid source redistribution policy")
        if not isinstance(raw_source.get("human_voice"), bool):
            raise PublicFixtureError("invalid source human-voice flag")
        personal_data = raw_source.get("personal_data")
        if personal_data not in _PERSONAL_DATA_CLASSES:
            raise PublicFixtureError("invalid source personal-data class")
        if bool(raw_source["human_voice"]) != (
            personal_data == "public-pseudonymous-voice"
        ):
            raise PublicFixtureError("source voice and personal-data flags disagree")
        if (
            schema_version == 3
            and raw_source.get("acquisition") != "verified-override-only"
        ):
            raise PublicFixtureError("public-v3 source requires verified override")
        sources.append(raw_source)

    fixture_ids: set[str] = set()
    fixtures: list[Mapping[str, object]] = []
    for raw_fixture in fixtures_raw:
        if not isinstance(raw_fixture, dict):
            raise PublicFixtureError("invalid fixture")
        fixture_id = _require_id(raw_fixture.get("id"), field="fixture id")
        if fixture_id in fixture_ids:
            raise PublicFixtureError("duplicate fixture id")
        fixture_ids.add(fixture_id)
        allowed_fixture_fields = {
            "id",
            "suite",
            "source_id",
            "extract",
            "assertion",
        }
        optional_fixture_fields = [
            "required_source_ids",
            "expected_text",
            "expected_text_from_archive_label",
            "tags",
        ]
        if schema_version == 3:
            optional_fixture_fields.append("expected_text_from_source")
        for optional in optional_fixture_fields:
            if optional in raw_fixture:
                allowed_fixture_fields.add(optional)
        _require_exact_keys(raw_fixture, allowed_fixture_fields, field="fixture")
        suite = raw_fixture.get("suite")
        if suite not in _SUITES_BY_SCHEMA[schema_version]:
            raise PublicFixtureError("invalid fixture suite")
        source_id = _require_id(raw_fixture.get("source_id"), field="source ref")
        if source_id not in source_ids:
            raise PublicFixtureError("unknown fixture source")
        required_raw = raw_fixture.get("required_source_ids", [])
        if not isinstance(required_raw, list):
            raise PublicFixtureError("unknown required source")
        required = tuple(
            _require_id(value, field="required source ref") for value in required_raw
        )
        if (
            any(value not in source_ids for value in required)
            or len(set(required)) != len(required)
            or source_id in required
        ):
            raise PublicFixtureError("unknown required source")
        _validate_extractor(
            raw_fixture,
            required_source_ids=required,
            schema_version=schema_version,
        )
        assertion = raw_fixture.get("assertion")
        if assertion not in _ASSERTIONS_BY_SUITE[str(suite)]:
            raise PublicFixtureError("invalid fixture assertion")
        expected_text = raw_fixture.get("expected_text")
        expected_from_source = raw_fixture.get("expected_text_from_source")
        if assertion == "transcript" and expected_from_source is True:
            if "expected_text" in raw_fixture:
                raise PublicFixtureError("source transcript fixture is over-specified")
        elif assertion == "transcript":
            _require_nonempty_text(expected_text, field="fixture expected text")
        elif assertion == "empty_reference" and expected_text != "":
            raise PublicFixtureError("empty-reference fixture needs an empty reference")
        elif (
            assertion == "unknown_word"
            and raw_fixture.get("expected_text_from_archive_label") is not True
        ):
            raise PublicFixtureError("unknown-word fixture needs its archive label")
        elif assertion not in {"transcript", "empty_reference", "unknown_word"} and (
            "expected_text" in raw_fixture
            or "expected_text_from_archive_label" in raw_fixture
        ):
            raise PublicFixtureError("non-transcript fixture has a text reference")
        extract = raw_fixture["extract"]
        assert isinstance(extract, dict)
        if (
            extract.get("type") == "hf_audio_parquet"
            and expected_from_source is not True
        ):
            raise PublicFixtureError("Parquet transcript must come from source")
        if (
            extract.get("type") != "hf_audio_parquet"
            and "expected_text_from_source" in raw_fixture
        ):
            raise PublicFixtureError("unexpected source transcript flag")
        if assertion == "unknown_word" and extract.get("target") != "_unknown_":
            raise PublicFixtureError("unknown-word assertion has the wrong target")
        if assertion == "empty_reference" and extract.get("target") != "_silence_":
            raise PublicFixtureError("empty reference must be pinned silence")
        if assertion == "transcript" and extract.get("type") == "wav_segment":
            annotation = extract["annotation"]
            assert isinstance(annotation, dict)
            if annotation.get("expected_text") != expected_text:
                raise PublicFixtureError("annotation and fixture transcripts disagree")
        _require_tags(
            raw_fixture.get("tags", []),
            allowed=_FIXTURE_TAGS,
            field="fixture tags",
        )
        fixtures.append(raw_fixture)

    if schema_version == 3:
        intent_ids = []
        for fixture in fixtures:
            extract = fixture["extract"]
            assert isinstance(extract, dict)
            intent_ids.append(extract.get("intent_id"))
        if (
            len(sources) != 1
            or seed != _MINDS_SELECTION_SEED
            or sources[0].get("id") != "minds14-en-us-train"
            or sources[0].get("upstream_version") != _MINDS_SOURCE_REVISION
            or sources[0].get("sha256") != _MINDS_SOURCE_SHA256
            or sources[0].get("size_bytes") != _MINDS_SOURCE_SIZE_BYTES
            or sources[0].get("license_id") != "CC-BY-4.0"
            or sources[0].get("redistribution") != "cache-only"
            or sources[0].get("acquisition") != "verified-override-only"
            or len(fixtures) != len(_MINDS_INTENTS)
            or intent_ids != list(range(len(_MINDS_INTENTS)))
            or groups_raw
        ):
            raise PublicFixtureError("invalid public-v3 MInDS-14 coverage")

    groups: list[Mapping[str, object]] = []
    group_ids: set[str] = set()
    for raw_group in groups_raw:
        if not isinstance(raw_group, dict):
            raise PublicFixtureError("invalid group")
        group_id = _require_id(raw_group.get("id"), field="group id")
        if group_id in group_ids:
            raise PublicFixtureError("duplicate group id")
        group_ids.add(group_id)
        _require_exact_keys(
            raw_group, {"id", "suite", "roles", "assertion"}, field="group"
        )
        suite = raw_group.get("suite")
        roles = raw_group.get("roles")
        assertion = raw_group.get("assertion")
        contract = _GROUP_CONTRACTS.get(str(assertion))
        if (
            suite not in _SUITES_BY_SCHEMA[schema_version]
            or not isinstance(roles, dict)
            or contract is None
            or set(roles) != set(contract)
        ):
            raise PublicFixtureError("invalid group")
        fixture_by_id = {str(item["id"]): item for item in fixtures}
        for role, fixture_id in roles.items():
            if not isinstance(fixture_id, str):
                raise PublicFixtureError("group references unknown fixture")
            fixture = fixture_by_id.get(str(fixture_id))
            if (
                fixture is None
                or fixture["suite"] != suite
                or fixture["assertion"] != contract[role]
            ):
                raise PublicFixtureError("group references unknown fixture")
        groups.append(raw_group)

    scenario_ids: set[str] = set()
    product_scenarios = tuple(
        _validate_product_scenario(item, scenario_ids=scenario_ids)
        for item in product_scenarios_raw
    )

    return PublicVoiceManifest(
        path=manifest_path,
        raw=raw,
        data=data,
        sources=tuple(sources),
        fixtures=tuple(fixtures),
        groups=tuple(groups),
        product_scenarios=product_scenarios,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with _stable_regular_reader(
        path,
        field="hashed file",
        maximum=_MAX_SOURCE_BYTES,
    ) as (handle, opened):
        total = 0
        while total < opened.st_size:
            chunk = handle.read(min(1024 * 1024, opened.st_size - total))
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        if total != opened.st_size or handle.read(1):
            raise PublicFixtureError("hashed file changed while reading")
    return digest.hexdigest()


def _verify_source_file(path: Path, source: Mapping[str, object]) -> None:
    expected_size = int(source["size_bytes"])
    digest = hashlib.sha256()
    with _stable_regular_reader(
        path,
        field="source file",
        maximum=expected_size,
        expected_size=expected_size,
    ) as (handle, opened):
        total = 0
        while total < opened.st_size:
            chunk = handle.read(min(1024 * 1024, opened.st_size - total))
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        if total != expected_size or handle.read(1):
            raise PublicFixtureError("source file size does not match manifest")
    if digest.hexdigest() != source["sha256"]:
        raise PublicFixtureError("source file hash does not match manifest")


def _clone_or_copy_file(
    source_path: Path,
    destination: Path,
    *,
    expected_size: int | None = None,
    require_single_link: bool = False,
) -> None:
    """Create an independent on-disk file, preferring a CoW reflink."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _stable_regular_reader(
            source_path,
            field="snapshot source",
            maximum=expected_size,
            expected_size=expected_size,
            require_single_link=require_single_link,
        ) as (source, opened):
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(destination, flags, 0o600)
            with os.fdopen(descriptor, "wb", buffering=0) as output:
                cloned = False
                if fcntl is not None:
                    try:
                        fcntl.ioctl(output.fileno(), _FICLONE, source.fileno())
                        cloned = True
                    except OSError:
                        output.seek(0)
                        output.truncate(0)
                        source.seek(0)
                if not cloned:
                    remaining = opened.st_size
                    while remaining:
                        chunk = source.read(min(_COPY_CHUNK_BYTES, remaining))
                        if not chunk:
                            raise PublicFixtureError(
                                "snapshot source changed while copying"
                            )
                        output.write(chunk)
                        remaining -= len(chunk)
                    if source.read(1):
                        raise PublicFixtureError(
                            "snapshot source changed while copying"
                        )
                else:
                    source.seek(opened.st_size)
                    if source.read(1):
                        raise PublicFixtureError(
                            "snapshot source changed while copying"
                        )
                if os.fstat(output.fileno()).st_size != opened.st_size:
                    raise PublicFixtureError("snapshot output size does not match")
                output.flush()
                os.fsync(output.fileno())
                os.fchmod(output.fileno(), 0o400)
    except PublicFixtureError:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise
    except Exception as exc:  # noqa: BLE001 - stable fail-closed boundary
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise PublicFixtureError("unable to snapshot pinned source") from exc


def _unlock_private_tree(root: Path) -> None:
    """Make only owned directories removable; never chmod files or links."""

    root_descriptor = -1
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_DIRECTORY", 0)
        )
        root_descriptor = os.open(root, flags)
    except OSError:
        return

    def unlock_directory(descriptor: int) -> None:
        try:
            os.fchmod(descriptor, 0o700)
            with os.scandir(descriptor) as entries:
                names = tuple(entry.name for entry in entries)
        except OSError:
            return
        for name in names:
            child_descriptor = -1
            try:
                child_descriptor = os.open(
                    name,
                    flags,
                    dir_fd=descriptor,
                )
                info = os.fstat(child_descriptor)
                if stat.S_ISDIR(info.st_mode):
                    unlock_directory(child_descriptor)
            except OSError:
                continue
            finally:
                if child_descriptor >= 0:
                    os.close(child_descriptor)

    try:
        unlock_directory(root_descriptor)
    finally:
        os.close(root_descriptor)


@contextmanager
def _verified_source_snapshots(
    manifest: PublicVoiceManifest,
    suite: str,
    resolved: Mapping[str, Path],
    *,
    snapshot_parent: Path,
) -> Iterator[dict[str, Path]]:
    """Yield private immutable copies whose bytes match every source record."""

    snapshot_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if snapshot_parent.is_symlink() or not snapshot_parent.is_dir():
        raise PublicFixtureError("invalid source snapshot directory")
    snapshot_root = Path(
        tempfile.mkdtemp(prefix="speaker-public-sources-", dir=snapshot_parent)
    )
    snapshots: dict[str, Path] = {}
    required_sources = _required_sources(manifest, suite)
    try:
        for source in required_sources:
            source_id = str(source["id"])
            source_path = resolved.get(source_id)
            if source_path is None:
                raise PublicFixtureError("resolved source is missing")
            destination = snapshot_root / str(source["filename"])
            _clone_or_copy_file(
                source_path,
                destination,
                expected_size=int(source["size_bytes"]),
                require_single_link=True,
            )
            _verify_source_file(destination, source)
            snapshots[source_id] = destination
        snapshot_root.chmod(0o500)
        yield snapshots
        for source in required_sources:
            _verify_source_file(snapshots[str(source["id"])], source)
    finally:
        _unlock_private_tree(snapshot_root)
        shutil.rmtree(snapshot_root, ignore_errors=True)


def _source_cache_path(cache_dir: Path, source: Mapping[str, object]) -> Path:
    return cache_dir / "sources" / str(source["filename"])


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        return None


def _open_url(request: urllib.request.Request, timeout: float):
    return urllib.request.build_opener(_NoRedirect).open(request, timeout=timeout)


def _download_source(
    source: Mapping[str, object],
    destination: Path,
    *,
    opener: Callable[[urllib.request.Request, float], BinaryIO] = _open_url,
    timeout_sec: float = 30.0,
) -> Path:
    """Stream one pinned source into an atomic cache file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(source["size_bytes"])
    expected_hash = str(source["sha256"])
    request = urllib.request.Request(
        str(source["download_url"]),
        headers={"User-Agent": "speaker-public-fixtures/2"},
    )
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    temp_path = Path(temp_name)
    digest = hashlib.sha256()
    total = 0
    try:
        with os.fdopen(fd, "wb") as output:
            with opener(request, timeout_sec) as response:
                geturl = getattr(response, "geturl", None)
                if not callable(geturl) or geturl() != source["download_url"]:
                    raise PublicFixtureError("source download redirected")
                while True:
                    chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > expected_size:
                        raise PublicFixtureError(
                            "download exceeded the pinned source size"
                        )
                    output.write(chunk)
                    digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if total != expected_size or digest.hexdigest() != expected_hash:
            raise PublicFixtureError("download does not match pinned source")
        os.replace(temp_path, destination)
        return destination
    except PublicFixtureError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize network/filesystem failures
        raise PublicFixtureError("source download failed") from exc
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _required_sources(
    manifest: PublicVoiceManifest, suite: str
) -> tuple[Mapping[str, object], ...]:
    schema_version = int(manifest.data["schema_version"])
    if suite not in _SUITES_BY_SCHEMA[schema_version]:
        raise PublicFixtureError("unknown public fixture suite")
    wanted: set[str] = set()
    for fixture in manifest.fixtures:
        if fixture["suite"] != suite:
            continue
        wanted.add(str(fixture["source_id"]))
        wanted.update(str(value) for value in fixture.get("required_source_ids", []))
    return tuple(source for source in manifest.sources if source["id"] in wanted)


def source_records_for_suite(
    manifest: PublicVoiceManifest, suite: str
) -> tuple[dict[str, object], ...]:
    """Return the exact public source records bound into prepared metadata."""

    records: list[dict[str, object]] = []
    for source in _required_sources(manifest, suite):
        record = {
            "id": source["id"],
            "sha256": source["sha256"],
            "size_bytes": source["size_bytes"],
            "upstream_dataset": source["upstream_dataset"],
            "upstream_version": source["upstream_version"],
            "canonical_url": source["canonical_url"],
            "source_kind": source["source_kind"],
            "license_id": source["license_id"],
            "license_url": source["license_url"],
            "attribution": source["attribution"],
            "citation": source["citation"],
            "redistribution": source["redistribution"],
            "human_voice": source["human_voice"],
            "personal_data": source["personal_data"],
        }
        if "acquisition" in source:
            record["acquisition"] = source["acquisition"]
        records.append(record)
    return tuple(records)


def _resolve_sources(
    manifest: PublicVoiceManifest,
    suite: str,
    cache_dir: Path,
    *,
    source_overrides: Mapping[str, Path] | None = None,
    allow_download: bool,
    offline: bool,
    opener: Callable[[urllib.request.Request, float], BinaryIO] = _open_url,
) -> dict[str, Path]:
    overrides = dict(source_overrides or {})
    known = manifest.source_by_id
    if set(overrides) - set(known):
        raise PublicFixtureError("override references unknown source")
    required_sources = _required_sources(manifest, suite)

    resolved: dict[str, Path] = {}
    missing: list[Mapping[str, object]] = []
    for source in required_sources:
        source_id = str(source["id"])
        override = overrides.get(source_id)
        if source.get("acquisition") == "verified-override-only" and override is None:
            raise PublicFixtureError(
                "source requires an explicit hash-verified --source override"
            )
        if override is not None:
            _verify_source_file(Path(override), source)
            resolved[source_id] = Path(override)
            continue
        cached = _source_cache_path(cache_dir, source)
        try:
            _verify_source_file(cached, source)
        except PublicFixtureError:
            missing.append(source)
        else:
            resolved[source_id] = cached

    if missing and (offline or not allow_download):
        raise DownloadApprovalRequired(
            sum(int(source["size_bytes"]) for source in missing)
        )
    for source in missing:
        destination = _source_cache_path(cache_dir, source)
        resolved[str(source["id"])] = _download_source(
            source, destination, opener=opener
        )
    return resolved


def _normalise_words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _normalised_word_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.lower()))


def _contains_token_subsequence(
    observed: Sequence[str], expected: Sequence[str]
) -> bool:
    if not expected or len(expected) > len(observed):
        return False
    width = len(expected)
    return any(
        tuple(observed[index : index + width]) == tuple(expected)
        for index in range(len(observed) - width + 1)
    )


def _read_audio(
    source: Path | io.BytesIO,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
    output_sample_rate: int = 16000,
    max_duration_sec: float | None = None,
):
    import numpy as np
    import soundfile as sf

    try:
        with sf.SoundFile(source) as audio:
            sample_rate = int(audio.samplerate)
            if sample_rate <= 0 or audio.frames <= 0:
                raise PublicFixtureError("audio source is empty")
            start = (
                0 if start_sec is None else int(round(float(start_sec) * sample_rate))
            )
            end = (
                int(audio.frames)
                if end_sec is None
                else int(round(float(end_sec) * sample_rate))
            )
            if start < 0 or end <= start or end > int(audio.frames):
                raise PublicFixtureError("audio segment is outside its source")
            audio.seek(start)
            samples = audio.read(end - start, dtype="float32", always_2d=True)
    except PublicFixtureError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PublicFixtureError("unable to decode source audio") from exc

    mono = np.asarray(samples, dtype="float32").mean(axis=1, dtype="float32")
    if sample_rate != output_sample_rate:
        from core.audio_frontend import AudioResampler

        mono = AudioResampler(sample_rate, output_sample_rate).process(mono, last=True)
    mono = np.asarray(mono, dtype="float32").reshape(-1)
    if max_duration_sec is not None:
        duration = float(max_duration_sec)
        if not math.isfinite(duration) or duration <= 0.0:
            raise PublicFixtureError("invalid maximum fixture duration")
        mono = mono[: int(round(duration * output_sample_rate))]
    if mono.size == 0 or not np.isfinite(mono).all():
        raise PublicFixtureError("decoded fixture audio is invalid")
    return mono


def _minds_selection_rank(
    seed: str, source_sha256: str, intent_label: str, source_path: str
) -> str:
    """Return the approved deterministic MInDS-14 case-selection rank."""

    return hashlib.sha256(
        seed.encode("utf-8")
        + b"\0"
        + source_sha256.encode("ascii")
        + b"\0"
        + intent_label.encode("utf-8")
        + b"\0"
        + source_path.encode("utf-8")
    ).hexdigest()


def _is_equal_or_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolved_production_venv() -> Path:
    active_root = Path(os.path.abspath(sys.executable)).parent.parent
    candidates = (
        active_root if active_root.name == ".venv" else None,
        _REPO_ROOT / ".venv",
    )
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_dir():
            return resolved
    return (_REPO_ROOT / ".venv").resolve(strict=False)


def _snapshot_venv_path(
    path: Path,
    *,
    directory: bool,
) -> tuple[tuple[int, ...], Path, tuple[int, ...]]:
    lexical_info = path.lstat()
    if directory:
        if not (
            stat.S_ISDIR(lexical_info.st_mode) or stat.S_ISLNK(lexical_info.st_mode)
        ):
            raise PublicFixtureError("invalid isolated --parquet-python")
    elif not (stat.S_ISREG(lexical_info.st_mode) or stat.S_ISLNK(lexical_info.st_mode)):
        raise PublicFixtureError("invalid isolated --parquet-python")
    resolved = path.resolve(strict=True)
    resolved_info = resolved.stat()
    if directory:
        if not stat.S_ISDIR(resolved_info.st_mode):
            raise PublicFixtureError("invalid isolated --parquet-python")
    elif not stat.S_ISREG(resolved_info.st_mode):
        raise PublicFixtureError("invalid isolated --parquet-python")
    return _file_identity(lexical_info), resolved, _file_identity(resolved_info)


def _effective_venv_marker(interpreter: Path, root: Path) -> Path:
    candidates = (interpreter.parent / "pyvenv.cfg", root / "pyvenv.cfg")
    present: list[Path] = []
    for candidate in candidates:
        try:
            candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            raise PublicFixtureError("invalid isolated --parquet-python") from None
        present.append(candidate)
    if len(present) != 1:
        raise PublicFixtureError("invalid isolated --parquet-python")
    return present[0]


def _read_valid_venv_marker(path: Path) -> tuple[bytes, tuple[int, ...]]:
    try:
        with _stable_regular_reader(
            path,
            field="isolated environment marker",
            maximum=_MAX_VENV_MARKER_BYTES,
        ) as (handle, opened):
            marker = handle.read(opened.st_size + 1)
            if len(marker) != opened.st_size:
                raise PublicFixtureError("invalid isolated --parquet-python")
            identity = _file_identity(opened)
    except PublicFixtureError:
        raise PublicFixtureError("invalid isolated --parquet-python") from None
    try:
        text = marker.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise PublicFixtureError("invalid isolated --parquet-python") from None
    if "\x00" in text:
        raise PublicFixtureError("invalid isolated --parquet-python")
    values: list[str] = []
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip().casefold() == "include-system-site-packages":
            values.append(value.strip().casefold())
    if values != ["false"]:
        raise PublicFixtureError("invalid isolated --parquet-python")
    return marker, identity


def _parquet_probe_environment(root: Path) -> dict[str, str]:
    return {
        "HOME": str(root),
        "TMPDIR": str(root),
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "ARROW_NUM_THREADS": "1",
    }


def _run_parquet_python_probe(
    interpreter: Path,
    *,
    candidate_root: Path,
    code: str = _PARQUET_PROBE_CODE,
    timeout_sec: float = _PARQUET_PROBE_TIMEOUT_SEC,
) -> bytes:
    process: subprocess.Popen[bytes] | None = None
    group_reaped = False
    output = bytearray()
    timed_out = False
    overflowed = False
    try:
        process = subprocess.Popen(
            [str(interpreter), "-I", "-B", "-c", code],
            cwd=candidate_root,
            env=_parquet_probe_environment(candidate_root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
            bufsize=0,
        )
        if process.stdout is None:
            raise PublicFixtureError("invalid isolated --parquet-python")
        descriptor = process.stdout.fileno()
        os.set_blocking(descriptor, False)
        deadline = monotonic() + timeout_sec
        while True:
            remaining = _MAX_PARQUET_PROBE_BYTES + 1 - len(output)
            if remaining <= 0:
                overflowed = True
                break
            try:
                chunk = os.read(descriptor, min(4096, remaining))
            except BlockingIOError:
                chunk = None
            if chunk:
                output.extend(chunk)
                if len(output) > _MAX_PARQUET_PROBE_BYTES:
                    overflowed = True
                    break
            if process.poll() is not None:
                break
            if monotonic() >= deadline:
                timed_out = True
                break
            sleep(0.01)

        _terminate_worker_process_group(process)
        group_reaped = True
        # A process that left the private group may still hold this pipe.
        # Consume only bytes already buffered; never wait for a later EOF.
        while len(output) <= _MAX_PARQUET_PROBE_BYTES:
            try:
                chunk = os.read(
                    descriptor,
                    min(4096, _MAX_PARQUET_PROBE_BYTES + 1 - len(output)),
                )
            except BlockingIOError:
                break
            if not chunk:
                break
            output.extend(chunk)
        if len(output) > _MAX_PARQUET_PROBE_BYTES:
            overflowed = True
        return_code = process.returncode
        if timed_out or overflowed or return_code != 0 or not output:
            raise PublicFixtureError("invalid isolated --parquet-python")
        return bytes(output)
    except PublicFixtureError:
        raise PublicFixtureError("invalid isolated --parquet-python") from None
    except (OSError, ValueError):
        raise PublicFixtureError("invalid isolated --parquet-python") from None
    finally:
        try:
            if process is not None and not group_reaped:
                _terminate_worker_process_group(process)
                group_reaped = True
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()


def _strict_probe_path(value: object) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise PublicFixtureError("invalid isolated --parquet-python")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise PublicFixtureError("invalid isolated --parquet-python") from None
    if len(encoded) > 4096:
        raise PublicFixtureError("invalid isolated --parquet-python")
    path = Path(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or Path(os.path.normpath(value)) != path
    ):
        raise PublicFixtureError("invalid isolated --parquet-python")
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError):
        raise PublicFixtureError("invalid isolated --parquet-python") from None
    if resolved != path:
        raise PublicFixtureError("invalid isolated --parquet-python")
    return resolved


def _validate_parquet_probe_result(
    raw: bytes,
    *,
    candidate_root: Path,
    production_venv: Path,
) -> None:
    if not raw or len(raw) > _MAX_PARQUET_PROBE_BYTES:
        raise PublicFixtureError("invalid isolated --parquet-python")
    try:
        result = _strict_json_loads(raw)
    except Exception:
        raise PublicFixtureError("invalid isolated --parquet-python") from None
    if not isinstance(result, dict) or set(result) != _PARQUET_PROBE_FIELDS:
        raise PublicFixtureError("invalid isolated --parquet-python")
    prefix = _strict_probe_path(result.get("prefix"))
    base_prefix = _strict_probe_path(result.get("base_prefix"))
    pyarrow_file = _strict_probe_path(result.get("pyarrow_file"))
    sys_path = result.get("sys_path")
    if (
        type(result.get("schema_version")) is not int
        or result.get("schema_version") != _PARQUET_PROBE_SCHEMA_VERSION
        or prefix != candidate_root
        or prefix == base_prefix
        or result.get("enable_user_site") is not False
        or result.get("pyarrow_version") != _PARQUET_PYARROW_VERSION
        or pyarrow_file == candidate_root
        or not _is_equal_or_within(pyarrow_file, candidate_root)
        or not pyarrow_file.is_file()
        or not isinstance(sys_path, list)
        or not 0 < len(sys_path) <= 128
    ):
        raise PublicFixtureError("invalid isolated --parquet-python")
    for raw_path in sys_path:
        resolved_path = _strict_probe_path(raw_path)
        if _is_equal_or_within(resolved_path, production_venv):
            raise PublicFixtureError("invalid isolated --parquet-python")


def _validate_parquet_python(value: Path | str | None) -> Path:
    if value is None:
        raise PublicFixtureError(
            "Parquet preparation requires an isolated --parquet-python"
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise PublicFixtureError("--parquet-python must be an absolute path")
    lexical = Path(os.path.abspath(path))
    candidate_root = lexical.parent.parent
    try:
        production_venv = _resolved_production_venv()
        root_snapshot = _snapshot_venv_path(candidate_root, directory=True)
        executable_snapshot = _snapshot_venv_path(lexical, directory=False)
        resolved_root = root_snapshot[1]
        if _is_equal_or_within(resolved_root, production_venv):
            raise PublicFixtureError("the production .venv cannot prepare Parquet")
        if not os.access(lexical, os.X_OK):
            raise PublicFixtureError("invalid isolated --parquet-python")
        marker_path = _effective_venv_marker(lexical, candidate_root)
        marker, marker_identity = _read_valid_venv_marker(marker_path)
        probe = _run_parquet_python_probe(
            lexical,
            candidate_root=resolved_root,
        )
        _validate_parquet_probe_result(
            probe,
            candidate_root=resolved_root,
            production_venv=production_venv,
        )
        if (
            _resolved_production_venv() != production_venv
            or _snapshot_venv_path(candidate_root, directory=True) != root_snapshot
            or _snapshot_venv_path(lexical, directory=False) != executable_snapshot
            or _effective_venv_marker(lexical, candidate_root) != marker_path
        ):
            raise PublicFixtureError("invalid isolated --parquet-python") from None
        marker_after, marker_identity_after = _read_valid_venv_marker(marker_path)
        if marker_after != marker or marker_identity_after != marker_identity:
            raise PublicFixtureError("invalid isolated --parquet-python")
    except PublicFixtureError:
        raise
    except OSError:
        raise PublicFixtureError("invalid isolated --parquet-python") from None
    return lexical


def _read_stable_worker_file(path: Path, *, maximum: int) -> bytes:
    with _stable_regular_reader(
        path,
        field="Parquet worker output",
        maximum=maximum,
    ) as (handle, opened):
        if stat.S_IMODE(opened.st_mode) & 0o077:
            raise PublicFixtureError("invalid Parquet worker output")
        payload = handle.read(opened.st_size + 1)
        if len(payload) != opened.st_size:
            raise PublicFixtureError("Parquet worker output changed while reading")
        return payload


def _read_stable_repo_worker(path: Path) -> bytes:
    payload = _read_stable_regular_bytes(
        path,
        field="Parquet worker",
        maximum=_MAX_PARQUET_WORKER_BYTES,
    )
    if not payload:
        raise PublicFixtureError("invalid Parquet worker")
    return payload


def _write_private_worker_snapshot(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        path.chmod(0o400)
    except OSError as exc:
        try:
            path.unlink()
        except OSError:
            pass
        raise PublicFixtureError("unable to snapshot Parquet worker") from exc


def _validate_parquet_worker_output(
    manifest: PublicVoiceManifest,
    fixtures: Sequence[Mapping[str, object]],
    output_dir: Path,
    *,
    worker_sha256: str,
    output_sample_rate: int,
) -> dict[str, _PreparedAudio]:
    if len(fixtures) != len(_MINDS_INTENTS):
        raise PublicFixtureError("invalid Parquet fixture set")
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise PublicFixtureError("invalid Parquet worker output directory")
    expected_names = {"result.json"} | {f"{fixture['id']}.wav" for fixture in fixtures}
    try:
        entries = tuple(output_dir.iterdir())
    except OSError as exc:
        raise PublicFixtureError("unable to enumerate Parquet worker output") from exc
    names = [entry.name for entry in entries]
    if (
        set(names) != expected_names
        or len(names) != len(expected_names)
        or len({name.casefold() for name in names}) != len(names)
    ):
        raise PublicFixtureError("unexpected Parquet worker output")

    raw_result = _read_stable_worker_file(
        output_dir / "result.json", maximum=_MAX_PARQUET_RESULT_BYTES
    )
    try:
        result = _strict_json_loads(raw_result)
    except Exception as exc:  # noqa: BLE001 - normalized trust-boundary error
        raise PublicFixtureError("invalid Parquet worker result") from exc
    if not isinstance(result, dict) or set(result) != _PARQUET_RESULT_FIELDS:
        raise PublicFixtureError("invalid Parquet worker result")
    if (
        result.get("schema_version") != _PARQUET_PROTOCOL_VERSION
        or result.get("source_sha256") != _MINDS_SOURCE_SHA256
        or result.get("source_size_bytes") != _MINDS_SOURCE_SIZE_BYTES
        or result.get("source_row_count") != _MINDS_SOURCE_ROWS
        or result.get("source_row_groups") != _MINDS_SOURCE_ROW_GROUPS
        or result.get("pyarrow_version") != _PARQUET_PYARROW_VERSION
    ):
        raise PublicFixtureError("Parquet worker provenance does not match")
    raw_cases = result.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != len(fixtures):
        raise PublicFixtureError("invalid Parquet worker cases")

    source = manifest.source_by_id.get(str(fixtures[0]["source_id"]))
    if (
        source is None
        or source.get("sha256") != _MINDS_SOURCE_SHA256
        or source.get("size_bytes") != _MINDS_SOURCE_SIZE_BYTES
    ):
        raise PublicFixtureError("Parquet source provenance does not match")
    seed = str(manifest.data["selection_seed"])
    prepared: dict[str, _PreparedAudio] = {}
    seen_rows: set[int] = set()
    total_audio_bytes = 0
    total_samples = 0

    for fixture, raw_case in zip(fixtures, raw_cases):
        if not isinstance(raw_case, dict) or set(raw_case) != _PARQUET_CASE_FIELDS:
            raise PublicFixtureError("invalid Parquet worker case")
        fixture_id = str(fixture["id"])
        extract = fixture["extract"]
        assert isinstance(extract, dict)
        intent_id = extract["intent_id"]
        intent_label = str(extract["intent_label"])
        expected_source_directory = f"en-US~{intent_label.upper()}"
        row_index = raw_case.get("row_index")
        source_path = _require_nonempty_text(
            raw_case.get("source_path"),
            field="Parquet source path",
            maximum=2048,
        )
        source_path_parts = PurePosixPath(source_path).parts
        transcript = _require_nonempty_text(
            raw_case.get("transcription"),
            field="Parquet transcription",
            maximum=4096,
        )
        audio_file = f"{fixture_id}.wav"
        audio_size = raw_case.get("audio_size_bytes")
        if (
            raw_case.get("fixture_id") != fixture_id
            or raw_case.get("intent_id") != intent_id
            or raw_case.get("intent_label") != intent_label
            or raw_case.get("language_id") != _MINDS_LANGUAGE_ID
            or raw_case.get("language") != "en-US"
            or len(source_path_parts) != 2
            or source_path_parts[0] != expected_source_directory
            or source_path_parts[1].startswith(".")
            or not source_path_parts[1].lower().endswith(".wav")
            or source_path != f"{expected_source_directory}/{source_path_parts[1]}"
            or isinstance(row_index, bool)
            or not isinstance(row_index, int)
            or not 0 <= row_index < _MINDS_SOURCE_ROWS
            or row_index in seen_rows
            or raw_case.get("audio_file") != audio_file
            or isinstance(audio_size, bool)
            or not isinstance(audio_size, int)
            or not 0 < audio_size <= _MAX_PARQUET_AUDIO_BYTES
            or raw_case.get("transcription_sha256")
            != hashlib.sha256(transcript.encode("utf-8")).hexdigest()
            or raw_case.get("selection_rank_sha256")
            != _minds_selection_rank(
                seed, _MINDS_SOURCE_SHA256, intent_label, source_path
            )
        ):
            raise PublicFixtureError("Parquet worker case does not match manifest")
        seen_rows.add(row_index)
        audio_path = output_dir / audio_file
        payload = _read_stable_worker_file(audio_path, maximum=_MAX_PARQUET_AUDIO_BYTES)
        if (
            len(payload) != audio_size
            or raw_case.get("audio_sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise PublicFixtureError("Parquet worker audio does not match")
        total_audio_bytes += len(payload)
        if total_audio_bytes > _MAX_PARQUET_TOTAL_AUDIO_BYTES:
            raise PublicFixtureError("Parquet worker audio budget exceeded")

        try:
            import soundfile as sf

            with sf.SoundFile(io.BytesIO(payload)) as audio:
                if (
                    audio.format != "WAV"
                    or audio.subtype != "ULAW"
                    or int(audio.samplerate) != 8000
                    or int(audio.channels) != 1
                    or int(audio.frames) <= 0
                    or int(audio.frames) > 30 * 8000
                ):
                    raise PublicFixtureError("invalid Parquet worker audio")
        except PublicFixtureError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PublicFixtureError("unable to inspect Parquet worker audio") from exc
        samples = _read_audio(
            io.BytesIO(payload), output_sample_rate=output_sample_rate
        )
        if len(samples) > _MAX_PARQUET_CASE_SAMPLES:
            raise PublicFixtureError("Parquet worker decoded duration exceeded")
        total_samples += len(samples)
        if total_samples > _MAX_PARQUET_TOTAL_SAMPLES:
            raise PublicFixtureError("Parquet worker corpus duration exceeded")
        prepared[fixture_id] = _PreparedAudio(
            samples=samples,
            expected_text=transcript,
            selection={
                "intent_id": intent_id,
                "intent_label": intent_label,
                "language_id": _MINDS_LANGUAGE_ID,
                "language": "en-US",
                "row_index": row_index,
                "source_path": source_path,
                "transcription_sha256": raw_case["transcription_sha256"],
                "audio_sha256": raw_case["audio_sha256"],
                "audio_size_bytes": audio_size,
                "selection_rank_sha256": raw_case["selection_rank_sha256"],
                "worker_sha256": worker_sha256,
                "worker_protocol_version": _PARQUET_PROTOCOL_VERSION,
                "pyarrow_version": _PARQUET_PYARROW_VERSION,
                "source_row_count": _MINDS_SOURCE_ROWS,
                "source_row_groups": _MINDS_SOURCE_ROW_GROUPS,
            },
        )
    if len(prepared) != len(fixtures):
        raise PublicFixtureError("Parquet worker did not prepare every case")
    return prepared


def _terminate_worker_process_group(process: subprocess.Popen[bytes]) -> None:
    """Kill and observe disappearance of the worker's entire private group."""

    process_group = int(process.pid)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        raise PublicFixtureError("unable to terminate Parquet worker group") from exc
    try:
        process.wait(timeout=_PARQUET_GROUP_REAP_TIMEOUT_SEC)
    except subprocess.TimeoutExpired as exc:
        raise PublicFixtureError("Parquet worker leader did not terminate") from exc

    deadline = monotonic() + _PARQUET_GROUP_REAP_TIMEOUT_SEC
    while True:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return
        except OSError as exc:
            raise PublicFixtureError(
                "unable to verify Parquet worker group termination"
            ) from exc
        if monotonic() >= deadline:
            raise PublicFixtureError("Parquet worker group did not terminate")
        sleep(0.01)


def _extract_hf_audio_parquet(
    source_path: Path,
    manifest: PublicVoiceManifest,
    fixtures: Sequence[Mapping[str, object]],
    *,
    parquet_python: Path | str | None,
    output_sample_rate: int,
    temporary_parent: Path,
) -> dict[str, _PreparedAudio]:
    interpreter = _validate_parquet_python(parquet_python)
    if len(fixtures) != len(_MINDS_INTENTS):
        raise PublicFixtureError("Parquet preparation requires all intent cases")
    try:
        if _PARQUET_WORKER.is_symlink():
            raise PublicFixtureError("invalid Parquet worker")
        worker_source = _PARQUET_WORKER.resolve(strict=True)
    except OSError as exc:
        raise PublicFixtureError("Parquet worker is unavailable") from exc
    worker_payload = _read_stable_repo_worker(worker_source)
    worker_sha256 = hashlib.sha256(worker_payload).hexdigest()

    temporary_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if temporary_parent.is_symlink() or not temporary_parent.is_dir():
        raise PublicFixtureError("invalid Parquet worker temporary directory")
    temporary_parent.chmod(0o700)
    root = Path(
        tempfile.mkdtemp(prefix="speaker-public-parquet-", dir=temporary_parent)
    )
    root.chmod(0o700)
    worker = root / "public_voice_parquet_worker.py"
    _write_private_worker_snapshot(worker, worker_payload)
    output_dir = root / "output"
    output_dir.mkdir(mode=0o700)
    request_path = root / "request.json"
    process: subprocess.Popen[bytes] | None = None
    group_reaped = False
    reap_attempted = False
    try:
        source = manifest.source_by_id[str(fixtures[0]["source_id"])]
        request = {
            "schema_version": _PARQUET_PROTOCOL_VERSION,
            "source_path": str(source_path.resolve(strict=True)),
            "source_sha256": source["sha256"],
            "source_size_bytes": source["size_bytes"],
            "selection_seed": manifest.data["selection_seed"],
            "source_id": source["id"],
            "cases": [
                {
                    "fixture_id": fixture["id"],
                    "intent_id": fixture["extract"]["intent_id"],
                    "intent_label": fixture["extract"]["intent_label"],
                }
                for fixture in fixtures
            ],
        }
        _atomic_write_json(request_path, request)
        request_path.chmod(0o400)
        environment = {
            "HOME": str(root),
            "TMPDIR": str(root),
            "PATH": os.defpath,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "ARROW_NUM_THREADS": "1",
        }
        try:
            process = subprocess.Popen(
                [
                    str(interpreter),
                    "-I",
                    "-B",
                    str(worker),
                    "--request",
                    str(request_path),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
            return_code = process.wait(timeout=_PARQUET_WORKER_TIMEOUT_SEC)
        except subprocess.TimeoutExpired as exc:
            raise PublicFixtureError("Parquet worker timed out") from exc
        except OSError as exc:
            raise PublicFixtureError("unable to start Parquet worker") from exc
        reap_attempted = True
        _terminate_worker_process_group(process)
        group_reaped = True
        if return_code != 0:
            raise PublicFixtureError("Parquet worker failed")
        if (
            hashlib.sha256(
                _read_stable_worker_file(worker, maximum=_MAX_PARQUET_WORKER_BYTES)
            ).hexdigest()
            != worker_sha256
            or hashlib.sha256(_read_stable_repo_worker(worker_source)).hexdigest()
            != worker_sha256
        ):
            raise PublicFixtureError("Parquet worker changed during preparation")
        return _validate_parquet_worker_output(
            manifest,
            fixtures,
            output_dir,
            worker_sha256=worker_sha256,
            output_sample_rate=output_sample_rate,
        )
    finally:
        if process is not None and not group_reaped and not reap_attempted:
            reap_attempted = True
            _terminate_worker_process_group(process)
            group_reaped = True
        if process is None or group_reaped:
            _unlock_private_tree(root)
            shutil.rmtree(root, ignore_errors=True)


def _safe_tar_member(
    archive: tarfile.TarFile, name: str, *, maximum: int
) -> tarfile.TarInfo:
    if (
        not name
        or PurePosixPath(name).is_absolute()
        or ".." in PurePosixPath(name).parts
    ):
        raise PublicFixtureError("unsafe archive member")
    try:
        member = archive.getmember(name)
    except KeyError as exc:
        raise PublicFixtureError("pinned archive member is missing") from exc
    if not member.isfile() or member.size <= 0 or member.size > maximum:
        raise PublicFixtureError("pinned archive member is invalid")
    return member


def _read_tar_member(
    archive: tarfile.TarFile, member: tarfile.TarInfo, *, maximum: int
) -> bytes:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise PublicFixtureError("unable to read pinned archive member")
    data = extracted.read(maximum + 1)
    if len(data) != member.size or len(data) > maximum:
        raise PublicFixtureError("archive member size mismatch")
    return data


def _extract_tar_fixture(
    source_path: Path,
    fixture: Mapping[str, object],
    *,
    output_sample_rate: int,
) -> _PreparedAudio:
    extract = fixture["extract"]
    assert isinstance(extract, dict)
    try:
        with tarfile.open(source_path, mode="r:*") as archive:
            member = _safe_tar_member(
                archive,
                str(extract.get("member") or ""),
                maximum=_MAX_AUDIO_MEMBER_BYTES,
            )
            proof_member_name = extract.get("proof_member")
            if proof_member_name is not None:
                proof_member = _safe_tar_member(
                    archive, str(proof_member_name), maximum=4 * 1024 * 1024
                )
                proof = _read_tar_member(
                    archive, proof_member, maximum=4 * 1024 * 1024
                ).decode("utf-8")
                if str(extract.get("proof_contains") or "") not in proof:
                    raise PublicFixtureError("archive transcript proof does not match")
            data = _read_tar_member(archive, member, maximum=_MAX_AUDIO_MEMBER_BYTES)
    except PublicFixtureError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PublicFixtureError("unable to read source archive") from exc
    samples = _read_audio(io.BytesIO(data), output_sample_rate=output_sample_rate)
    return _PreparedAudio(
        samples=samples,
        expected_text=str(fixture.get("expected_text") or ""),
        selection={"archive_member": member.name},
    )


def _speech_member_from_tarinfo(
    member: tarfile.TarInfo, *, seed: str
) -> _SpeechMember | None:
    if not member.isfile() or member.size <= 0 or member.size > _MAX_AUDIO_MEMBER_BYTES:
        return None
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 2:
        return None
    label, filename = path.parts
    if not filename.lower().endswith(".wav"):
        return None
    if label == "_silence_":
        speaker_id = None
    else:
        marker = "_nohash_"
        if marker not in filename:
            return None
        speaker_id = filename.split(marker, 1)[0]
        if not speaker_id:
            return None
    rank = hashlib.sha256(
        seed.encode("utf-8") + b"\0" + member.name.encode("utf-8")
    ).hexdigest()
    return _SpeechMember(
        name=member.name,
        label=label,
        speaker_id=speaker_id,
        rank=rank,
        size=int(member.size),
        offset=int(member.offset_data),
    )


def _select_speech_command_members(
    members: Iterable[_SpeechMember],
    targets: Sequence[str],
) -> dict[str, _SpeechMember]:
    """Choose one stable candidate per target while keeping speakers distinct."""

    candidates = tuple(members)
    selected: dict[str, _SpeechMember] = {}
    used_speakers: set[str] = set()
    for target in targets:
        if target in _COMMAND_WORDS or target == "_silence_":
            choices = [item for item in candidates if item.label == target]
        elif target == "_unknown_":
            choices = [
                item
                for item in candidates
                if item.label not in _COMMAND_WORDS
                and item.label not in {"_silence_", "_background_noise_"}
            ]
        else:
            raise PublicFixtureError("unknown Speech Commands target")
        choices.sort(key=lambda item: (item.rank, item.name))
        choice = next(
            (
                item
                for item in choices
                if item.speaker_id is None or item.speaker_id not in used_speakers
            ),
            None,
        )
        if choice is None:
            raise PublicFixtureError(
                "Speech Commands split cannot satisfy distinct-speaker selection"
            )
        selected[target] = choice
        if choice.speaker_id is not None:
            used_speakers.add(choice.speaker_id)
    return selected


def _extract_speech_commands(
    source_path: Path,
    fixtures: Sequence[Mapping[str, object]],
    *,
    seed: str,
    output_sample_rate: int,
) -> dict[str, _PreparedAudio]:
    targets: list[str] = []
    fixture_by_target: dict[str, Mapping[str, object]] = {}
    try:
        with tarfile.open(source_path, mode="r:*") as archive:
            members = [
                parsed
                for info in archive.getmembers()
                if (parsed := _speech_member_from_tarinfo(info, seed=seed)) is not None
            ]
            for fixture in fixtures:
                extract = fixture["extract"]
                assert isinstance(extract, dict)
                target = str(extract.get("target") or "")
                if target in fixture_by_target:
                    raise PublicFixtureError("duplicate Speech Commands target")
                targets.append(target)
                fixture_by_target[target] = fixture
            selected = _select_speech_command_members(members, targets)
            selected_names = {item.name for item in selected.values()}
            info_by_name = {info.name: info for info in archive.getmembers()}
            payloads: dict[str, bytes] = {}
            for chosen in sorted(selected.values(), key=lambda item: item.offset):
                info = info_by_name.get(chosen.name)
                if info is None:
                    raise PublicFixtureError("selected command member disappeared")
                payloads[chosen.name] = _read_tar_member(
                    archive, info, maximum=_MAX_AUDIO_MEMBER_BYTES
                )
            if set(payloads) != selected_names:
                raise PublicFixtureError("selected command member is missing")
    except PublicFixtureError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PublicFixtureError("unable to read Speech Commands archive") from exc

    prepared: dict[str, _PreparedAudio] = {}
    for target in targets:
        fixture = fixture_by_target[target]
        extract = fixture["extract"]
        assert isinstance(extract, dict)
        chosen = selected[target]
        samples = _read_audio(
            io.BytesIO(payloads[chosen.name]),
            output_sample_rate=output_sample_rate,
            max_duration_sec=extract.get("max_duration_sec"),
        )
        expected = (
            chosen.label
            if fixture.get("expected_text_from_archive_label") is True
            else str(fixture.get("expected_text") or "")
        )
        prepared[str(fixture["id"])] = _PreparedAudio(
            samples=samples,
            expected_text=expected,
            selection={
                "archive_member": chosen.name,
                "archive_label": chosen.label,
                "speaker_id": chosen.speaker_id,
                "selection_rank_sha256": chosen.rank,
            },
        )
    return prepared


def _xml_local_name(element: ElementTree.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _annotation_evidence(
    archive_path: Path,
    annotation: Mapping[str, object],
    *,
    start_sec: float,
    end_sec: float,
) -> Mapping[str, object]:
    members = annotation.get("members")
    if not isinstance(members, list) or not members:
        raise PublicFixtureError("annotation has no members")
    member_names = tuple(
        _require_member_path(member, field="annotation member") for member in members
    )
    if len(set(member_names)) != len(member_names):
        raise PublicFixtureError("duplicate annotation member")
    _require_unique_ami_speakers(member_names)
    vocal_types: set[str] = set()
    words: list[tuple[float, str, str]] = []
    speaker_intervals: dict[str, list[tuple[float, float]]] = {}
    identities = tuple(_ami_speaker_identity(name) for name in member_names)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member_name, (meeting, speaker) in zip(member_names, identities):
                speaker_identity = f"{meeting}:{speaker}"
                path = PurePosixPath(member_name)
                if path.is_absolute() or ".." in path.parts:
                    raise PublicFixtureError("unsafe annotation member")
                try:
                    info = archive.getinfo(member_name)
                except KeyError as exc:
                    raise PublicFixtureError("annotation member is missing") from exc
                if info.file_size <= 0 or info.file_size > 4 * 1024 * 1024:
                    raise PublicFixtureError("annotation member is invalid")
                root = ElementTree.fromstring(archive.read(info))
                speaker_active = False
                event_intervals: list[tuple[float, float]] = []
                for element in root.iter():
                    try:
                        event_start = float(element.attrib["starttime"])
                        event_end = float(element.attrib["endtime"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not (math.isfinite(event_start) and math.isfinite(event_end)):
                        # AMI contains a few unrelated unaligned events. They
                        # cannot provide evidence for this selected window.
                        continue
                    if event_end < event_start:
                        if (
                            max(event_start, event_end) > start_sec
                            and min(event_start, event_end) < end_sec
                        ):
                            raise PublicFixtureError(
                                "selected annotation has invalid timing"
                            )
                        continue
                    overlaps = (
                        event_end > start_sec
                        and event_start < end_sec
                        and event_end > event_start
                    )
                    if not overlaps:
                        continue
                    kind = _xml_local_name(element)
                    if kind in {"w", "vocalsound"}:
                        speaker_active = True
                        event_intervals.append(
                            (max(start_sec, event_start), min(end_sec, event_end))
                        )
                    if kind == "w" and not element.attrib.get("punc"):
                        token = (element.text or "").strip()
                        if token:
                            words.append((event_start, speaker_identity, token))
                    if kind == "vocalsound":
                        vocal_type = str(element.attrib.get("type") or "")
                        if vocal_type:
                            vocal_types.add(vocal_type)
                if speaker_active:
                    speaker_intervals[speaker_identity] = event_intervals
    except PublicFixtureError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PublicFixtureError("unable to validate AMI annotation") from exc

    active_speakers = len(speaker_intervals)
    minimum = annotation.get("min_active_speakers", 0)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
        raise PublicFixtureError("invalid annotation speaker requirement")
    if active_speakers < minimum:
        raise PublicFixtureError("annotation speaker requirement is not met")
    maximum = annotation.get("max_active_speakers", len(members))
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
        raise PublicFixtureError("invalid annotation speaker maximum")
    if active_speakers > maximum:
        raise PublicFixtureError("annotation speaker maximum is exceeded")
    boundaries = sorted(
        {
            point
            for intervals in speaker_intervals.values()
            for interval in intervals
            for point in interval
        }
    )
    max_concurrent = 0
    for left, right in zip(boundaries, boundaries[1:]):
        if right <= left:
            continue
        midpoint = (left + right) / 2.0
        max_concurrent = max(
            max_concurrent,
            sum(
                any(start < midpoint < end for start, end in intervals)
                for intervals in speaker_intervals.values()
            ),
        )
    concurrent_minimum = annotation.get("min_concurrent_speakers", 0)
    if (
        isinstance(concurrent_minimum, bool)
        or not isinstance(concurrent_minimum, int)
        or concurrent_minimum < 0
    ):
        raise PublicFixtureError("invalid annotation concurrency requirement")
    if max_concurrent < concurrent_minimum:
        raise PublicFixtureError("annotation concurrency requirement is not met")
    required_vocal = annotation.get("required_vocal_type")
    if required_vocal is not None and str(required_vocal) not in vocal_types:
        raise PublicFixtureError("annotation vocal event requirement is not met")
    expected_text = annotation.get("expected_text")
    annotation_tokens = [token for _start, _speaker, token in sorted(words)]
    transcript = " ".join(annotation_tokens)
    if expected_text is not None:
        observed_tokens = _normalised_word_tokens(transcript)
        expected_tokens = _normalised_word_tokens(str(expected_text))
        if not _contains_token_subsequence(observed_tokens, expected_tokens):
            raise PublicFixtureError("annotation transcript does not match")
    return {
        "active_speakers": active_speakers,
        "max_concurrent_speakers": max_concurrent,
        "vocal_types": sorted(vocal_types),
        "annotation_tokens": annotation_tokens,
        "annotation_token_count": len(annotation_tokens),
        "annotation_transcript": transcript,
    }


def _extract_wav_segment(
    source_path: Path,
    fixture: Mapping[str, object],
    sources: Mapping[str, Path],
    *,
    output_sample_rate: int,
) -> _PreparedAudio:
    extract = fixture["extract"]
    assert isinstance(extract, dict)
    try:
        start_sec = float(extract["start_sec"])
        end_sec = float(extract["end_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PublicFixtureError("invalid WAV segment") from exc
    if (
        not math.isfinite(start_sec)
        or not math.isfinite(end_sec)
        or end_sec <= start_sec
    ):
        raise PublicFixtureError("invalid WAV segment")
    selection: dict[str, object] = {
        "start_sec": start_sec,
        "end_sec": end_sec,
    }
    annotation = extract.get("annotation")
    if annotation is not None:
        if not isinstance(annotation, dict):
            raise PublicFixtureError("invalid annotation recipe")
        annotation_source = sources.get(str(annotation.get("source_id") or ""))
        if annotation_source is None:
            raise PublicFixtureError("annotation source was not resolved")
        selection.update(
            _annotation_evidence(
                annotation_source,
                annotation,
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )
    samples = _read_audio(
        source_path,
        start_sec=start_sec,
        end_sec=end_sec,
        output_sample_rate=output_sample_rate,
    )
    return _PreparedAudio(
        samples=samples,
        expected_text=str(fixture.get("expected_text") or ""),
        selection=selection,
    )


def _extract_direct_wav(
    source_path: Path,
    fixture: Mapping[str, object],
    *,
    output_sample_rate: int,
) -> _PreparedAudio:
    return _PreparedAudio(
        samples=_read_audio(source_path, output_sample_rate=output_sample_rate),
        expected_text=str(fixture.get("expected_text") or ""),
        selection={},
    )


def _atomic_save_npy(path: Path, samples: object) -> str:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, np.asarray(samples, dtype="float32").reshape(-1))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        return _sha256_file(path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _prepare_suite_from_snapshots(
    manifest: PublicVoiceManifest,
    suite: str,
    cache: Path,
    resolved: Mapping[str, Path],
    *,
    parquet_python: Path | str | None = None,
) -> Path:
    fixtures = tuple(item for item in manifest.fixtures if item["suite"] == suite)
    if not fixtures:
        raise PublicFixtureError("public fixture suite has no fixtures")
    sample_rate = int(manifest.data["output_sample_rate"])
    prepared: dict[str, _PreparedAudio] = {}

    speech_groups: dict[str, list[Mapping[str, object]]] = {}
    for fixture in fixtures:
        extract = fixture["extract"]
        assert isinstance(extract, dict)
        if extract["type"] == "speech_commands_tar":
            speech_groups.setdefault(str(fixture["source_id"]), []).append(fixture)
    for source_id, grouped in speech_groups.items():
        prepared.update(
            _extract_speech_commands(
                resolved[source_id],
                grouped,
                seed=str(manifest.data["selection_seed"]),
                output_sample_rate=sample_rate,
            )
        )

    parquet_groups: dict[str, list[Mapping[str, object]]] = {}
    for fixture in fixtures:
        extract = fixture["extract"]
        assert isinstance(extract, dict)
        if extract["type"] == "hf_audio_parquet":
            parquet_groups.setdefault(str(fixture["source_id"]), []).append(fixture)
    for source_id, grouped in parquet_groups.items():
        prepared.update(
            _extract_hf_audio_parquet(
                resolved[source_id],
                manifest,
                grouped,
                parquet_python=parquet_python,
                output_sample_rate=sample_rate,
                temporary_parent=cache / ".parquet-workers",
            )
        )

    for fixture in fixtures:
        fixture_id = str(fixture["id"])
        if fixture_id in prepared:
            continue
        source_path = resolved[str(fixture["source_id"])]
        extract = fixture["extract"]
        assert isinstance(extract, dict)
        extractor = extract["type"]
        if extractor == "tar_member":
            result = _extract_tar_fixture(
                source_path, fixture, output_sample_rate=sample_rate
            )
        elif extractor == "wav_segment":
            result = _extract_wav_segment(
                source_path, fixture, resolved, output_sample_rate=sample_rate
            )
        elif extractor == "direct_wav":
            result = _extract_direct_wav(
                source_path, fixture, output_sample_rate=sample_rate
            )
        else:
            raise PublicFixtureError("unsupported extractor")
        prepared[fixture_id] = result

    output_dir = cache / f"v{manifest.data['schema_version']}" / suite
    cases: list[dict[str, object]] = []
    for fixture in fixtures:
        fixture_id = str(fixture["id"])
        result = prepared.get(fixture_id)
        if result is None:
            raise PublicFixtureError("fixture was not prepared")
        output_path = output_dir / f"{fixture_id}.npy"
        output_hash = _atomic_save_npy(output_path, result.samples)
        source_ids = [str(fixture["source_id"])] + [
            str(value) for value in fixture.get("required_source_ids", [])
        ]
        source_hashes = {
            source_id: str(manifest.source_by_id[source_id]["sha256"])
            for source_id in source_ids
        }
        cases.append(
            {
                "name": fixture_id,
                "file": output_path.name,
                "expectation": result.expected_text
                or str(fixture.get("assertion") or ""),
                "expected_text": result.expected_text,
                "assertion": fixture["assertion"],
                "tags": list(fixture.get("tags", [])),
                "sample_rate": sample_rate,
                "sample_count": int(len(result.samples)),
                "duration_sec": round(len(result.samples) / sample_rate, 6),
                "sha256": output_hash,
                "source_sha256": source_hashes,
                "selection": dict(result.selection),
            }
        )

    source_records = list(source_records_for_suite(manifest, suite))
    metadata = {
        "schema_version": int(manifest.data["schema_version"]),
        "suite": suite,
        "purpose": manifest.data["purpose"],
        "evidence_scope": dict(EVIDENCE_SCOPE),
        "audio_committed": False,
        "manifest_sha256": manifest.digest,
        "selection_seed": manifest.data["selection_seed"],
        "cases": cases,
        "groups": [group for group in manifest.groups if group.get("suite") == suite],
        "sources": source_records,
    }
    _atomic_write_json(output_dir / "metadata.json", metadata)
    return output_dir


def prepare_suite(
    manifest: PublicVoiceManifest,
    suite: str,
    cache_dir: Path | str = DEFAULT_CACHE,
    *,
    source_overrides: Mapping[str, Path] | None = None,
    parquet_python: Path | str | None = None,
    allow_download: bool = False,
    offline: bool = False,
    opener: Callable[[urllib.request.Request, float], BinaryIO] = _open_url,
) -> Path:
    """Prepare one suite entirely from private hash-verified source snapshots."""

    cache = Path(cache_dir)
    suite_fixtures = tuple(
        fixture for fixture in manifest.fixtures if fixture["suite"] == suite
    )
    if any(
        isinstance(fixture.get("extract"), dict)
        and fixture["extract"].get("type") == "hf_audio_parquet"
        for fixture in suite_fixtures
    ):
        parquet_python = _validate_parquet_python(parquet_python)
    resolved = _resolve_sources(
        manifest,
        suite,
        cache,
        source_overrides=source_overrides,
        allow_download=allow_download,
        offline=offline,
        opener=opener,
    )
    with _verified_source_snapshots(
        manifest,
        suite,
        resolved,
        snapshot_parent=cache / ".source-snapshots",
    ) as snapshots:
        return _prepare_suite_from_snapshots(
            manifest,
            suite,
            cache,
            snapshots,
            parquet_python=parquet_python,
        )


def _parse_source_overrides(values: Sequence[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise PublicFixtureError("--source must be ID=PATH")
        source_id, raw_path = value.split("=", 1)
        source_id = _require_id(source_id, field="source override id")
        if source_id in overrides or not raw_path:
            raise PublicFixtureError("invalid source override")
        overrides[source_id] = Path(raw_path).expanduser()
    return overrides


def _format_bytes(value: int) -> str:
    return f"{value / (1024 * 1024):.1f} MiB"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare hash-pinned public development/regression voice fixtures. "
            "Generated audio stays in .cache and is never a final held-out set."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="show suites and maximum source downloads")
    prepare = subparsers.add_parser("prepare", help="fetch/extract one suite")
    prepare.add_argument(
        "--suite",
        choices=tuple(dict.fromkeys(SUITES + _V3_SUITES)),
        required=True,
    )
    prepare.add_argument(
        "--accept-download",
        action="store_true",
        help="allow missing pinned sources to be downloaded",
    )
    prepare.add_argument(
        "--offline",
        action="store_true",
        help="forbid network even if --accept-download is also present",
    )
    prepare.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="reuse an already-downloaded source after size/hash verification",
    )
    prepare.add_argument(
        "--parquet-python",
        type=Path,
        help="absolute interpreter path for the isolated PyArrow preparation worker",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "list":
            for suite in _SUITES_BY_SCHEMA[int(manifest.data["schema_version"])]:
                sources = _required_sources(manifest, suite)
                size = sum(int(source["size_bytes"]) for source in sources)
                fixtures = sum(
                    1 for fixture in manifest.fixtures if fixture["suite"] == suite
                )
                if fixtures == 0:
                    continue
                print(
                    f"{suite}: {fixtures} fixtures, up to {_format_bytes(size)} "
                    "of pinned sources"
                )
            print("development/regression only; not final held-out or live evidence")
            return 0
        overrides = _parse_source_overrides(args.source)
        output_dir = prepare_suite(
            manifest,
            args.suite,
            args.cache_dir,
            source_overrides=overrides,
            parquet_python=args.parquet_python,
            allow_download=bool(args.accept_download),
            offline=bool(args.offline),
        )
        print(f"prepared {args.suite} public regression fixtures in {output_dir}")
        print("not final held-out, owner-voice, room, route, or live-hardware evidence")
        return 0
    except PublicFixtureError as exc:
        print(f"public fixture preparation failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
