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
import shutil
import tarfile
import tempfile
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
_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,95}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_AMI_WORD_MEMBER_RE = re.compile(
    r"(?P<meeting>[A-Za-z0-9_-]+)\.(?P<speaker>[A-Z])\.words\.xml\Z"
)
_MAX_SOURCE_BYTES = 256 * 1024 * 1024
_MAX_AUDIO_MEMBER_BYTES = 64 * 1024 * 1024
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
_LICENSE_IDS = frozenset(
    {"Apache-2.0", "CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0", "MIT"}
)
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
        if not isinstance(terms, list) or not terms or any(
            not isinstance(term, str) or not term for term in terms
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
) -> None:
    extract = raw_fixture.get("extract")
    if not isinstance(extract, dict):
        raise PublicFixtureError("invalid fixture extractor")
    extractor = extract.get("type")
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
        raw = manifest_path.read_bytes()
        data = _strict_json_loads(raw)
    except PublicFixtureError:
        raise
    except Exception as exc:  # noqa: BLE001 - converted to a stable public error
        raise PublicFixtureError("unable to load public voice manifest") from exc
    if (
        not isinstance(data, dict)
        or type(data.get("schema_version")) is not int
        or data.get("schema_version") != 2
    ):
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
    if not isinstance(product_scenarios_raw, list) or not product_scenarios_raw:
        raise PublicFixtureError("manifest has no product scenarios")

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
        _require_exact_keys(
            raw_source,
            {
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
            },
            field="source",
        )
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
        for optional in (
            "required_source_ids",
            "expected_text",
            "expected_text_from_archive_label",
            "tags",
        ):
            if optional in raw_fixture:
                allowed_fixture_fields.add(optional)
        _require_exact_keys(raw_fixture, allowed_fixture_fields, field="fixture")
        suite = raw_fixture.get("suite")
        if suite not in SUITES:
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
        _validate_extractor(raw_fixture, required_source_ids=required)
        assertion = raw_fixture.get("assertion")
        if assertion not in _ASSERTIONS_BY_SUITE[str(suite)]:
            raise PublicFixtureError("invalid fixture assertion")
        expected_text = raw_fixture.get("expected_text")
        if assertion == "transcript":
            _require_nonempty_text(expected_text, field="fixture expected text")
        elif assertion == "empty_reference" and expected_text != "":
            raise PublicFixtureError("empty-reference fixture needs an empty reference")
        elif assertion == "unknown_word" and raw_fixture.get(
            "expected_text_from_archive_label"
        ) is not True:
            raise PublicFixtureError("unknown-word fixture needs its archive label")
        elif assertion not in {"transcript", "empty_reference", "unknown_word"} and (
            "expected_text" in raw_fixture
            or "expected_text_from_archive_label" in raw_fixture
        ):
            raise PublicFixtureError("non-transcript fixture has a text reference")
        extract = raw_fixture["extract"]
        assert isinstance(extract, dict)
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
            suite not in SUITES
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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source_file(path: Path, source: Mapping[str, object]) -> None:
    try:
        stat_size = path.stat().st_size
    except OSError as exc:
        raise PublicFixtureError("source file is unavailable") from exc
    expected_size = int(source["size_bytes"])
    if stat_size != expected_size:
        raise PublicFixtureError("source file size does not match manifest")
    if _sha256_file(path) != source["sha256"]:
        raise PublicFixtureError("source file hash does not match manifest")


def _clone_or_copy_file(source_path: Path, destination: Path) -> None:
    """Create an independent on-disk file, preferring a CoW reflink."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source_path.open("rb", buffering=0) as source:
            with destination.open("xb", buffering=0) as output:
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
                    shutil.copyfileobj(source, output, length=_COPY_CHUNK_BYTES)
                output.flush()
                os.fsync(output.fileno())
        destination.chmod(0o400)
    except Exception as exc:  # noqa: BLE001 - stable fail-closed boundary
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise PublicFixtureError("unable to snapshot pinned source") from exc


def _unlock_private_tree(root: Path) -> None:
    try:
        paths = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    except OSError:
        paths = []
    for path in paths:
        try:
            path.chmod(0o700 if path.is_dir() else 0o600)
        except OSError:
            pass
    try:
        root.chmod(0o700)
    except OSError:
        pass


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
            _clone_or_copy_file(source_path, destination)
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
    if suite not in SUITES:
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

    return tuple(
        {
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
        for source in _required_sources(manifest, suite)
    )


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
        "schema_version": 2,
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
    allow_download: bool = False,
    offline: bool = False,
    opener: Callable[[urllib.request.Request, float], BinaryIO] = _open_url,
) -> Path:
    """Prepare one suite entirely from private hash-verified source snapshots."""

    cache = Path(cache_dir)
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
        return _prepare_suite_from_snapshots(manifest, suite, cache, snapshots)


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
    prepare.add_argument("--suite", choices=SUITES, required=True)
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "list":
            for suite in SUITES:
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
