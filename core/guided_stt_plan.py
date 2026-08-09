"""Fixed, private reference plan for safe owner STT capture.

The plan is deliberately project-owned and immutable.  ``./live.sh`` writes its
canonical bytes into each private run bundle before the microphone is opened;
the capture-only child accepts only those exact bytes.  Recognized text is not
used to choose, reorder, or repair cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Mapping


GUIDED_STT_PLAN_ID = "agent-core-v1"
GUIDED_STT_PLAN_SCHEMA_VERSION = 1
GUIDED_STT_CAPTURE_PROTOCOL = "guided-stt-capture-v1"
GUIDED_STT_SUMMARY_CONTRACT_KIND = "guided-stt-capture-summary-v1"
GUIDED_STT_SUMMARY_CONTRACT_SCHEMA_VERSION = 1
GUIDED_STT_ROUTE_PROFILE = {
    "app_aliases": ["obsidian"],
    "reminders_enabled": True,
    "vault_enabled": True,
}
_ROUTES = frozenset(
    {
        "none",
        "vault.search",
        "web.search",
        "reminder.create",
        "reminder.list",
        "reminder.cancel",
        "app.open",
    }
)
_MAX_PLAN_BYTES = 64 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SUMMARY_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SUMMARY_CONTRACT_DOMAIN = b"speaker-guided-stt-capture-summary-v1\0"
_SUMMARY_CONTRACT_FIELDS = (
    "session",
    "topology",
    "audio_egress",
    "engine",
    "llm",
    "device",
    "final_stt_profile",
    "final_stt_profile_sha256",
    "final_stt_profile_schema_version",
    "capture_only_contract_sha256",
    "capture_only_profile_sha256",
    "capture_only_sherpa_sha256",
    "execution_purpose",
    "llm_constructed",
    "control_plane_constructed",
    "tts_constructed",
    "keyword_spotter_constructed",
    "speaker_gate_constructed",
    "playback_worker_constructed",
    "output_device_queried",
    "owner_authority",
    "speaker_identity_policy",
    "capture_protocol",
    "capture_plan_id",
    "capture_plan_sha256",
    "capture_planned_cases",
    "capture_accepted_finals",
    "capture_state",
    "capture_completed",
    "effects_enabled",
    "diagnostic_manifest_sha256",
    "diagnostic_status",
)


class GuidedSttPlanError(ValueError):
    """A detail-free fixed-plan or private-file contract failure."""

    def __init__(self) -> None:
        super().__init__("guided STT plan is unavailable")


@dataclass(frozen=True, slots=True)
class GuidedSttCase:
    case_id: str
    geometry: str
    expected_route: str
    expected_text: str = field(repr=False)
    tags: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class GuidedSttPlan:
    plan_id: str
    schema_version: int
    sha256: str
    cases: tuple[GuidedSttCase, ...] = field(repr=False)

    @property
    def case_count(self) -> int:
        return len(self.cases)


_CASE_ROWS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "close-01-number",
        "The access code is four zero nine two.",
        ("guided-stt", "geometry-close", "numeral-sequence", "expected-tool.none"),
    ),
    (
        "close-02-vault",
        "Search in my vault for the speaker roadmap.",
        ("guided-stt", "geometry-close", "vault-alias", "expected-tool.vault.search"),
    ),
    (
        "close-03-obsidian-vault",
        "Find in my Obsidian the speaker roadmap.",
        (
            "guided-stt",
            "geometry-close",
            "obsidian-vault-alias",
            "expected-tool.vault.search",
        ),
    ),
    (
        "close-04-web",
        "Search the web for the weather in Bucharest.",
        ("guided-stt", "geometry-close", "expected-tool.web.search"),
    ),
    (
        "close-05-reminder-create",
        "Remind me to stretch in twenty minutes.",
        (
            "guided-stt",
            "geometry-close",
            "numeral-word",
            "expected-tool.reminder.create",
        ),
    ),
    (
        "close-06-app",
        "Open Obsidian.",
        (
            "guided-stt",
            "geometry-close",
            "obsidian-app-alias",
            "expected-tool.app.open",
        ),
    ),
    (
        "close-07-negative-vault",
        "Tell me a short joke about a bank vault.",
        ("guided-stt", "geometry-close", "negative-control", "expected-tool.none"),
    ),
    (
        "close-08-negative-app",
        "Tell me what Obsidian means.",
        ("guided-stt", "geometry-close", "negative-control", "expected-tool.none"),
    ),
    (
        "far-01-number",
        "My backup number is seven three zero five.",
        ("guided-stt", "geometry-far", "numeral-sequence", "expected-tool.none"),
    ),
    (
        "far-02-vault-go",
        "Go in my vault for the speaker roadmap.",
        ("guided-stt", "geometry-far", "vault-alias", "expected-tool.vault.search"),
    ),
    (
        "far-03-dobo-brain",
        "Query dobo brain for the speaker roadmap.",
        (
            "guided-stt",
            "geometry-far",
            "dobo-brain-alias",
            "expected-tool.vault.search",
        ),
    ),
    (
        "far-04-paul-brain",
        "Summarize paul brain about the speaker roadmap.",
        (
            "guided-stt",
            "geometry-far",
            "paul-brain-alias",
            "expected-tool.vault.search",
        ),
    ),
    (
        "far-05-reminder-list",
        "List my reminders.",
        ("guided-stt", "geometry-far", "expected-tool.reminder.list"),
    ),
    (
        "far-06-reminder-cancel",
        "Cancel my next reminder.",
        ("guided-stt", "geometry-far", "expected-tool.reminder.cancel"),
    ),
    (
        "far-07-web-exclusion",
        "Search the web, not my vault, for the weather in Bucharest.",
        ("guided-stt", "geometry-far", "negative-control", "expected-tool.web.search"),
    ),
    (
        "far-08-vault-exclusion",
        "Do not go in my vault for the speaker roadmap.",
        ("guided-stt", "geometry-far", "negative-control", "expected-tool.none"),
    ),
)


def _canonical_json(payload: object) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (MemoryError, TypeError, UnicodeError, ValueError, OverflowError):
        raise GuidedSttPlanError() from None


def canonical_guided_stt_plan_bytes() -> bytes:
    return _canonical_json(
        {
            "schema_version": GUIDED_STT_PLAN_SCHEMA_VERSION,
            "cases": [
                {"id": case_id, "expected_text": text, "tags": list(tags)}
                for case_id, text, tags in _CASE_ROWS
            ],
        }
    )


def guided_stt_route_availability_sha256() -> str:
    payload = _canonical_json(GUIDED_STT_ROUTE_PROFILE)
    return hashlib.sha256(
        b"speaker-guided-stt-route-profile-v1\0" + payload
    ).hexdigest()


def guided_stt_sherpa_config_sha256(config: Mapping[str, object]) -> str:
    """Bind the exact effective Sherpa mapping without exposing model paths."""

    if not isinstance(config, Mapping) or isinstance(config, (str, bytes)):
        raise GuidedSttPlanError()
    payload = _canonical_json(dict(config))
    return hashlib.sha256(
        b"speaker-guided-stt-effective-sherpa-v1\0" + payload
    ).hexdigest()


def guided_stt_capture_config_sha256(config: Mapping[str, object]) -> str:
    """Bind capture/DSP/VAD/endpoint settings while excluding final-STT A/B."""

    if not isinstance(config, Mapping) or isinstance(config, (str, bytes)):
        raise GuidedSttPlanError()
    if any(not isinstance(key, str) for key in config):
        raise GuidedSttPlanError()
    capture_config = {
        key: value for key, value in config.items() if not key.startswith("asr_final_")
    }
    payload = _canonical_json(capture_config)
    return hashlib.sha256(
        b"speaker-guided-stt-capture-config-v1\0" + payload
    ).hexdigest()


def guided_stt_summary_contract_sha256(
    meta: Mapping[str, object],
) -> str:
    """Bind the fixed aggregate success facts written by capture-only runs.

    The full run summary is intentionally not hashed here: it contains volatile
    duration and telemetry fields and is not final until the logging queue has
    drained.  The launcher separately hashes those final bytes in the terminal
    bundle receipt.
    """

    if not isinstance(meta, Mapping) or isinstance(meta, (str, bytes)):
        raise GuidedSttPlanError()
    try:
        facts = {field: meta[field] for field in _SUMMARY_CONTRACT_FIELDS}
    except (KeyError, TypeError):
        raise GuidedSttPlanError() from None

    plan = built_in_guided_stt_plan()
    sha_fields = (
        "final_stt_profile_sha256",
        "capture_only_contract_sha256",
        "capture_only_profile_sha256",
        "capture_only_sherpa_sha256",
        "capture_plan_sha256",
        "diagnostic_manifest_sha256",
    )
    disabled_fields = (
        "llm_constructed",
        "control_plane_constructed",
        "tts_constructed",
        "keyword_spotter_constructed",
        "speaker_gate_constructed",
        "playback_worker_constructed",
        "output_device_queried",
        "owner_authority",
        "effects_enabled",
    )
    diagnostic_status = facts.get("diagnostic_status")
    if (
        facts.get("session") != "local"
        or facts.get("topology") != "local_device"
        or facts.get("audio_egress") != "device_only"
        or facts.get("engine") != "sherpa"
        or facts.get("llm") != "disabled"
        or not isinstance(facts.get("device"), str)
        or _SAFE_SUMMARY_NAME_RE.fullmatch(str(facts["device"])) is None
        or facts.get("final_stt_profile")
        not in {"sense-voice", "parakeet-faster-whisper"}
        or type(facts.get("final_stt_profile_schema_version")) is not int
        or int(facts["final_stt_profile_schema_version"]) <= 0
        or any(
            not isinstance(facts.get(field), str)
            or _SHA256_RE.fullmatch(str(facts[field])) is None
            for field in sha_fields
        )
        or facts.get("final_stt_profile_sha256")
        != facts.get("capture_only_profile_sha256")
        or facts.get("execution_purpose") != "stt_capture_only"
        or any(facts.get(field) is not False for field in disabled_fields)
        or facts.get("speaker_identity_policy") != "off_for_session"
        or facts.get("capture_protocol") != GUIDED_STT_CAPTURE_PROTOCOL
        or facts.get("capture_plan_id") != plan.plan_id
        or facts.get("capture_plan_sha256") != plan.sha256
        or facts.get("capture_planned_cases") != plan.case_count
        or facts.get("capture_accepted_finals") != plan.case_count
        or facts.get("capture_state") != "open"
        or facts.get("capture_completed") is not True
        or not isinstance(diagnostic_status, Mapping)
        or set(diagnostic_status) != {"status", "failure_codes"}
        or diagnostic_status.get("status") != "complete"
        or diagnostic_status.get("failure_codes") != []
    ):
        raise GuidedSttPlanError()

    payload = {
        "schema_version": GUIDED_STT_SUMMARY_CONTRACT_SCHEMA_VERSION,
        "kind": GUIDED_STT_SUMMARY_CONTRACT_KIND,
        "facts": facts,
    }
    return hashlib.sha256(
        _SUMMARY_CONTRACT_DOMAIN + _canonical_json(payload)
    ).hexdigest()


def _parse_exact(raw: bytes) -> GuidedSttPlan:
    expected = canonical_guided_stt_plan_bytes()
    if raw != expected:
        raise GuidedSttPlanError()
    cases: list[GuidedSttCase] = []
    for index, (case_id, text, tags) in enumerate(_CASE_ROWS):
        geometry_tags = tuple(tag for tag in tags if tag.startswith("geometry-"))
        route_tags = tuple(tag for tag in tags if tag.startswith("expected-tool."))
        expected_geometry = "close" if index < 8 else "far"
        if (
            len(geometry_tags) != 1
            or geometry_tags[0] != f"geometry-{expected_geometry}"
            or len(route_tags) != 1
            or route_tags[0].removeprefix("expected-tool.") not in _ROUTES
        ):
            raise GuidedSttPlanError()
        cases.append(
            GuidedSttCase(
                case_id=case_id,
                geometry=expected_geometry,
                expected_route=route_tags[0].removeprefix("expected-tool."),
                expected_text=text,
                tags=tags,
            )
        )
    return GuidedSttPlan(
        plan_id=GUIDED_STT_PLAN_ID,
        schema_version=GUIDED_STT_PLAN_SCHEMA_VERSION,
        sha256=hashlib.sha256(raw).hexdigest(),
        cases=tuple(cases),
    )


def built_in_guided_stt_plan() -> GuidedSttPlan:
    return _parse_exact(canonical_guided_stt_plan_bytes())


def load_private_guided_stt_plan(path: Path | str) -> GuidedSttPlan:
    """Load the exact built-in plan from one owner-private regular inode."""

    parent_descriptor = descriptor = -1
    try:
        candidate = Path(path)
        if not candidate.is_absolute() or candidate.name in {"", ".", ".."}:
            raise GuidedSttPlanError()
        parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        parent_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        parent_descriptor = os.open(candidate.parent, parent_flags)
        parent_before = os.fstat(parent_descriptor)
        if (
            not stat.S_ISDIR(parent_before.st_mode)
            or stat.S_IMODE(parent_before.st_mode) != 0o700
            or parent_before.st_uid != os.getuid()
        ):
            raise GuidedSttPlanError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(candidate.name, flags, dir_fd=parent_descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
            or metadata.st_size > _MAX_PLAN_BYTES
        ):
            raise GuidedSttPlanError()
        raw = bytearray()
        while len(raw) <= _MAX_PLAN_BYTES:
            chunk = os.read(descriptor, min(8192, _MAX_PLAN_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        if len(raw) != metadata.st_size:
            raise GuidedSttPlanError()
        after = os.fstat(descriptor)
        path_after = os.stat(
            candidate.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        parent_after = os.fstat(parent_descriptor)
        if (
            after.st_dev != metadata.st_dev
            or after.st_ino != metadata.st_ino
            or after.st_size != metadata.st_size
            or after.st_mtime_ns != metadata.st_mtime_ns
            or after.st_ctime_ns != metadata.st_ctime_ns
            or path_after.st_dev != metadata.st_dev
            or path_after.st_ino != metadata.st_ino
            or path_after.st_mode != metadata.st_mode
            or path_after.st_nlink != metadata.st_nlink
            or parent_after.st_dev != parent_before.st_dev
            or parent_after.st_ino != parent_before.st_ino
            or parent_after.st_mode != parent_before.st_mode
        ):
            raise GuidedSttPlanError()
        return _parse_exact(bytes(raw))
    except GuidedSttPlanError:
        raise
    except (OSError, TypeError, ValueError):
        raise GuidedSttPlanError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if parent_descriptor >= 0:
            try:
                os.close(parent_descriptor)
            except OSError:
                pass


__all__ = [
    "GUIDED_STT_CAPTURE_PROTOCOL",
    "GUIDED_STT_PLAN_ID",
    "GUIDED_STT_ROUTE_PROFILE",
    "GUIDED_STT_SUMMARY_CONTRACT_KIND",
    "GUIDED_STT_SUMMARY_CONTRACT_SCHEMA_VERSION",
    "GuidedSttCase",
    "GuidedSttPlan",
    "GuidedSttPlanError",
    "built_in_guided_stt_plan",
    "canonical_guided_stt_plan_bytes",
    "guided_stt_capture_config_sha256",
    "guided_stt_route_availability_sha256",
    "guided_stt_sherpa_config_sha256",
    "guided_stt_summary_contract_sha256",
    "load_private_guided_stt_plan",
]
