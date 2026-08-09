"""Strict loading for the private AMI ES2004a natural-turn replay bundle."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Mapping

from tools.capture_replay.corpus import (
    FRAME_SAMPLES,
    SAMPLE_RATE_HZ,
    LoadedReplayCorpus,
    ReplayCorpusError,
    SampleInterval,
    load_corpus,
    verify_corpus_snapshot,
)
from tools.streaming_stt.bounded_io import BoundedReadError, read_regular_bounded


DEFAULT_LOCK = (
    Path(__file__).with_name("ami-es2004a-natural-turn-v1.lock.json").resolve()
)
LOCK_FILE_SHA256 = "02ebfbbbf0871464586f2d62a181cf3dbd2b2699eaec0d72650a335601ea9614"
LOCK_FILE_BYTES = 14_424
FIXTURE_ID = "ami-es2004a-natural-turn-v1"
LOCK_KIND = "ami-es2004a-natural-turn-fixture-lock-v1"
METADATA_FILENAME = "ami-natural-turn.json"
MANIFEST_FILENAME = "capture-replay.json"
PREPARATION_RECEIPT_FILENAME = "preparation-receipt.json"
METADATA_KIND = "ami-es2004a-natural-turn-private-labels-v1"
RECEIPT_KIND = "ami-es2004a-natural-turn-preparation-receipt-v1"
SCHEMA_VERSION = 1
CASE_SAMPLES = 128_000
CASE_COUNT = 4
_MAX_LOCK_BYTES = 256 * 1024
_MAX_JSON_BYTES = 256 * 1024
_MAX_PCM_BYTES = CASE_SAMPLES * 4
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}\Z")
_CANONICALIZATION = (
    "UTF-8 JSON with sort_keys=true, separators=(',', ':'), "
    "ensure_ascii=false, allow_nan=false"
)
_ROOT_FIELDS = {
    "schema_version",
    "kind",
    "fixture_id",
    "production_evidence",
    "self_digest",
    "license",
    "sources",
    "archive_contract",
    "selection",
    "transform",
    "layout",
    "outputs",
    "evidence_scope",
}
_PREPARER_FILES = (
    "core/__init__.py",
    "core/media_session.py",
    "core/wer.py",
    "tools/__init__.py",
    "tools/capture_replay/__init__.py",
    "tools/capture_replay/ami_natural_turn.py",
    "tools/capture_replay/corpus.py",
    "tools/capture_replay/replay.py",
    "tools/prepare_ami_natural_turn_capture_replay.py",
    "tools/streaming_stt/__init__.py",
    "tools/streaming_stt/bounded_io.py",
)


class AmiNaturalTurnFixtureError(RuntimeError):
    """A detail-free lock, bundle, or snapshot validation failure."""


@dataclass(frozen=True, slots=True)
class AmiNaturalTurnTestInjection:
    """Explicit non-production lock identity; unreachable from the CLI."""

    fixture_id: str
    lock_sha256: str
    lock_size_bytes: int


@dataclass(frozen=True, slots=True)
class AmiNaturalTurnDialogueAct:
    dialogue_act_id: str
    speaker_id: str = field(repr=False)
    type_href: str
    source_start_sample: int
    source_end_sample: int
    output_start_sample: int
    output_end_sample: int
    reference: str = field(repr=False)
    reference_sha256: str


@dataclass(frozen=True, slots=True)
class AmiNaturalTurnCase:
    case_id: str
    window_id: str
    channel: str
    replay_case_index: int
    samples: int
    frame_samples: int
    source_start_sample: int
    source_end_sample: int
    source_offset_sample: int
    source_end_output_sample: int
    reference: str = field(repr=False)
    reference_sha256: str
    dialogue_acts: tuple[AmiNaturalTurnDialogueAct, ...] = field(repr=False)
    adjacency_pair_id: str | None
    overlap_interval: SampleInterval | None
    gap_interval: SampleInterval | None

    @property
    def source_complete_sample(self) -> int:
        """The full-case sample count required for procedural completeness."""

        return self.samples

    @property
    def overlap_references(self) -> tuple[str, ...]:
        """Private per-DA references for order-independent overlap scoring."""

        if self.overlap_interval is None:
            return ()
        return tuple(act.reference for act in self.dialogue_acts)


@dataclass(frozen=True, slots=True)
class LoadedAmiNaturalTurnFixture:
    root: Path = field(repr=False)
    fixture_id: str
    production_evidence: bool
    lock_sha256: str
    lock_recipe_sha256: str
    labels_sha256: str
    metadata_sha256: str
    manifest_sha256: str
    receipt_sha256: str
    source_contract_sha256: str
    preparer_closure_sha256: str
    sample_rate_hz: int
    frame_samples: int
    case_samples: int
    replay_corpus: LoadedReplayCorpus = field(repr=False)
    cases: tuple[AmiNaturalTurnCase, ...] = field(repr=False)
    directory_identity: tuple[int, ...] = field(repr=False)
    file_identities: tuple[tuple[str, tuple[int, ...]], ...] = field(repr=False)
    lock_path: Path = field(repr=False)
    lock_size_bytes: int = field(repr=False)


@dataclass(frozen=True, slots=True)
class _MemberPin:
    path: str
    role: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _FixtureLock:
    fixture_id: str
    production_evidence: bool
    raw_sha256: str
    raw_size_bytes: int
    recipe_sha256: str
    value: Mapping[str, object] = field(repr=False)
    members: tuple[_MemberPin, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _PrivateFile:
    name: str
    raw: bytes = field(repr=False)
    sha256: str
    identity: tuple[int, ...] = field(repr=False)


def _bad() -> object:
    raise AmiNaturalTurnFixtureError()


def _strict_json(raw: bytes, *, maximum_bytes: int) -> object:
    if not raw or len(raw) > maximum_bytes:
        raise AmiNaturalTurnFixtureError()

    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if not isinstance(key, str) or key in result:
                raise AmiNaturalTurnFixtureError()
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: _bad(),
        )
    except AmiNaturalTurnFixtureError:
        raise
    except (OverflowError, UnicodeError, ValueError):
        raise AmiNaturalTurnFixtureError() from None


def _canonical_json(value: object, *, newline: bool = False) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (MemoryError, RecursionError, TypeError, UnicodeError, ValueError):
        raise AmiNaturalTurnFixtureError() from None
    return encoded + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _mapping(value: object, fields: set[str]) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise AmiNaturalTurnFixtureError()
    return value


def _sequence(value: object, *, length: int | None = None) -> list[object]:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        raise AmiNaturalTurnFixtureError()
    return value


def _string(value: object, *, maximum: int = 512) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or len(value) > maximum
        or "\x00" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise AmiNaturalTurnFixtureError()
    return value


def _safe_id(value: object) -> str:
    result = _string(value, maximum=96)
    if _SAFE_ID_RE.fullmatch(result) is None:
        raise AmiNaturalTurnFixtureError()
    return result


def _sha256(value: object) -> str:
    result = _string(value, maximum=64)
    if _SHA256_RE.fullmatch(result) is None:
        raise AmiNaturalTurnFixtureError()
    return result


def _integer(value: object, *, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise AmiNaturalTurnFixtureError()
    return value


def _safe_relative(value: object) -> str:
    result = _string(value, maximum=256)
    pure = PurePosixPath(result)
    if (
        pure.is_absolute()
        or not pure.parts
        or pure.as_posix() != result
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in result
        or ":" in pure.parts[0]
    ):
        raise AmiNaturalTurnFixtureError()
    return result


def _canonical_self_digest(value: Mapping[str, object]) -> str:
    copied = dict(value)
    raw_self = copied.get("self_digest")
    if not isinstance(raw_self, dict):
        raise AmiNaturalTurnFixtureError()
    self_digest = dict(raw_self)
    self_digest.pop("value", None)
    copied["self_digest"] = self_digest
    return _canonical_sha256(copied)


def _selected_windows_binding(
    selection: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw_window in _sequence(selection.get("windows"), length=2):
        window = _mapping(
            raw_window,
            {
                "window_id",
                "start_sample",
                "end_sample",
                "linearized_reference_sha256",
                "dialogue_acts",
                "segments",
                "adjacency_pair",
                "relation",
                "timed_words",
                "external_positive_duration_annotations",
                "edge_complete_turn",
            },
        )
        acts = _sequence(window["dialogue_acts"], length=2)
        segments = _sequence(window["segments"], length=2)
        rows.append(
            {
                "meeting_id": selection["meeting_id"],
                "window_id": window["window_id"],
                "start_sample": window["start_sample"],
                "end_sample": window["end_sample"],
                "dialogue_act_ids": [
                    _mapping(
                        act,
                        {
                            "id",
                            "speaker",
                            "type_href",
                            "words_href",
                            "start_sample",
                            "end_sample",
                            "reference_sha256",
                        },
                    )["id"]
                    for act in acts
                ],
                "dialogue_act_reference_sha256": [
                    act["reference_sha256"]
                    for act in acts  # type: ignore[index]
                ],
                "linearized_reference_sha256": window["linearized_reference_sha256"],
                "segment_ids": [
                    _mapping(
                        segment,
                        {
                            "id",
                            "speaker",
                            "words_href",
                            "start_sample",
                            "end_sample",
                        },
                    )["id"]
                    for segment in segments
                ],
                "adjacency_pair_id": (
                    None
                    if window["adjacency_pair"] is None
                    else _mapping(
                        window["adjacency_pair"],
                        {"id", "type_href", "source_href", "target_href"},
                    )["id"]
                ),
            }
        )
    return rows


def _parse_lock(
    value: object,
    *,
    raw_sha256: str,
    raw_size_bytes: int,
    production: bool,
    expected_fixture_id: str,
) -> _FixtureLock:
    root = _mapping(value, _ROOT_FIELDS)
    self_digest = _mapping(
        root["self_digest"],
        {"algorithm", "canonicalization", "excluded_json_pointer", "value"},
    )
    recipe = _sha256(self_digest["value"])
    fixture_id = _safe_id(root["fixture_id"])
    if (
        root["schema_version"] != SCHEMA_VERSION
        or root["kind"] != LOCK_KIND
        or fixture_id != expected_fixture_id
        or root["production_evidence"] is not production
        or self_digest["algorithm"] != "sha256"
        or self_digest["canonicalization"] != _CANONICALIZATION
        or self_digest["excluded_json_pointer"] != "/self_digest/value"
        or _canonical_self_digest(root) != recipe
    ):
        raise AmiNaturalTurnFixtureError()

    license_value = _mapping(
        root["license"],
        {"gate_id", "acceptance_ids", "redistribution", "attribution", "license_url"},
    )
    if license_value["gate_id"] != "CC-BY-4.0" or license_value["acceptance_ids"] != [
        "CC-BY-4.0"
    ]:
        raise AmiNaturalTurnFixtureError()

    sources = _mapping(root["sources"], {"annotation_archive", "audio"})
    annotation = _mapping(
        sources["annotation_archive"],
        {
            "filename",
            "url",
            "size_bytes",
            "sha256",
            "integrity_provenance",
            "artifact_label",
            "embedded_readme_release",
        },
    )
    _safe_relative(annotation["filename"])
    _sha256(annotation["sha256"])
    _integer(annotation["size_bytes"], minimum=1, maximum=32 * 1024 * 1024)
    audio = _sequence(sources["audio"], length=2)
    if [item.get("channel") if isinstance(item, dict) else None for item in audio] != [
        "close",
        "far",
    ]:
        raise AmiNaturalTurnFixtureError()
    for raw_audio in audio:
        item = _mapping(
            raw_audio,
            {
                "channel",
                "filename",
                "url",
                "size_bytes",
                "sha256",
                "integrity_provenance",
                "samples",
                "roomtone_pcm16_sha256",
                "roomtone_f32_sha256",
            },
        )
        _safe_relative(item["filename"])
        _integer(item["size_bytes"], minimum=45, maximum=64 * 1024 * 1024)
        _integer(item["samples"], minimum=CASE_SAMPLES, maximum=32_000_000)
        for key in ("sha256", "roomtone_pcm16_sha256", "roomtone_f32_sha256"):
            _sha256(item[key])

    archive = _mapping(
        root["archive_contract"],
        {
            "member_count",
            "expanded_bytes",
            "compressed_payload_bytes",
            "archive_comment_bytes",
            "encrypted_members",
            "zip64_sized_members",
            "maximum_members",
            "maximum_expanded_bytes",
            "maximum_member_bytes",
            "required_members",
        },
    )
    maximum_members = _integer(archive["maximum_members"], minimum=1, maximum=8192)
    member_count = _integer(archive["member_count"], minimum=1, maximum=maximum_members)
    _integer(
        archive["expanded_bytes"],
        minimum=1,
        maximum=_integer(archive["maximum_expanded_bytes"], minimum=1),
    )
    _integer(archive["compressed_payload_bytes"], minimum=1)
    for key in ("archive_comment_bytes", "encrypted_members", "zip64_sized_members"):
        _integer(archive[key], minimum=0, maximum=member_count)
    maximum_member = _integer(archive["maximum_member_bytes"], minimum=1)
    members: list[_MemberPin] = []
    for raw_member in _sequence(archive["required_members"]):
        member = _mapping(raw_member, {"path", "role", "size_bytes", "sha256"})
        members.append(
            _MemberPin(
                _safe_relative(member["path"]),
                _string(member["role"], maximum=32),
                _integer(member["size_bytes"], minimum=1, maximum=maximum_member),
                _sha256(member["sha256"]),
            )
        )
    if (
        len(members) < 14
        or len({item.path.casefold() for item in members}) != len(members)
        or [item.path for item in members] != sorted(item.path for item in members)
    ):
        raise AmiNaturalTurnFixtureError()

    selection = _mapping(
        root["selection"],
        {"meeting_id", "algorithm", "selected_windows_sha256", "windows"},
    )
    if (
        selection["meeting_id"] != "ES2004a"
        or selection["algorithm"] != "fixed-manual-dialogue-act-envelopes-v1"
        or _canonical_sha256(_selected_windows_binding(selection))
        != _sha256(selection["selected_windows_sha256"])
    ):
        raise AmiNaturalTurnFixtureError()
    windows = _sequence(selection["windows"], length=2)
    if [
        item.get("window_id") if isinstance(item, dict) else None for item in windows
    ] != [
        "nested_backchannel",
        "adjacent_exchange",
    ]:
        raise AmiNaturalTurnFixtureError()
    for raw_window in windows:
        window = _mapping(raw_window, set(raw_window))  # already exact-checked above
        start = _integer(window["start_sample"], minimum=0)
        end = _integer(window["end_sample"], minimum=start + 1)
        _sha256(window["linearized_reference_sha256"])
        if (
            window["external_positive_duration_annotations"] != 0
            or window["edge_complete_turn"] is not False
        ):
            raise AmiNaturalTurnFixtureError()
        acts = _sequence(window["dialogue_acts"], length=2)
        speakers: list[str] = []
        for raw_act in acts:
            act = _mapping(
                raw_act,
                {
                    "id",
                    "speaker",
                    "type_href",
                    "words_href",
                    "start_sample",
                    "end_sample",
                    "reference_sha256",
                },
            )
            speakers.append(_string(act["speaker"], maximum=1))
            _string(act["id"])
            _string(act["type_href"])
            _string(act["words_href"])
            act_start = _integer(act["start_sample"], minimum=start, maximum=end)
            _integer(act["end_sample"], minimum=act_start + 1, maximum=end)
            _sha256(act["reference_sha256"])
        if len(set(speakers)) != 2:
            raise AmiNaturalTurnFixtureError()
        for raw_segment in _sequence(window["segments"], length=2):
            segment = _mapping(
                raw_segment,
                {"id", "speaker", "words_href", "start_sample", "end_sample"},
            )
            _string(segment["id"])
            _string(segment["speaker"], maximum=1)
            _string(segment["words_href"])
            seg_start = _integer(segment["start_sample"], minimum=0)
            _integer(segment["end_sample"], minimum=seg_start + 1)
        relation = _mapping(window["relation"], {"kind", "start_sample", "end_sample"})
        relation_start = _integer(relation["start_sample"], minimum=start, maximum=end)
        _integer(relation["end_sample"], minimum=relation_start + 1, maximum=end)
        for raw_word in _sequence(window["timed_words"]):
            word = _mapping(
                raw_word,
                {"id", "speaker", "start_sample", "end_sample", "punctuation"},
            )
            _string(word["id"])
            _string(word["speaker"], maximum=1)
            word_start = _integer(word["start_sample"], minimum=start, maximum=end)
            word_end = _integer(word["end_sample"], minimum=word_start, maximum=end)
            if not isinstance(word["punctuation"], bool) or (
                word["punctuation"] is (word_end > word_start)
            ):
                raise AmiNaturalTurnFixtureError()

    transform = _mapping(
        root["transform"],
        {
            "sample_rate_hz",
            "frame_samples",
            "case_samples",
            "source_start_rounding",
            "source_end_rounding",
            "input_encoding",
            "output_encoding",
            "float_conversion",
            "silence_window_start_sample",
            "silence_window_end_sample",
            "roomtone_start_sample",
            "roomtone_end_sample",
            "prefix_samples",
            "postroll",
            "source_splice",
        },
    )
    if (
        transform["sample_rate_hz"] != SAMPLE_RATE_HZ
        or transform["frame_samples"] != FRAME_SAMPLES
        or transform["case_samples"] != CASE_SAMPLES
        or transform["source_start_rounding"] != "decimal-floor"
        or transform["source_end_rounding"] != "decimal-ceiling"
        or transform["input_encoding"] != "riff-pcm-s16le-mono"
        or transform["output_encoding"] != "f32le"
        or transform["float_conversion"] != "astype(float32)*float32(1.0/32768.0)"
        or transform["postroll"] != "np.resize(same-channel-roomtone,remaining-samples)"
        or transform["source_splice"] != "untouched-contiguous-envelope"
    ):
        raise AmiNaturalTurnFixtureError()
    prefix = _integer(transform["prefix_samples"], minimum=1, maximum=CASE_SAMPLES)
    room_start = _integer(transform["roomtone_start_sample"], minimum=0)
    room_end = _integer(transform["roomtone_end_sample"], minimum=room_start + 1)
    if prefix != room_end - room_start:
        raise AmiNaturalTurnFixtureError()

    layout = _mapping(
        root["layout"],
        {"window_count", "channel_count", "case_count", "artifact_count", "case_order"},
    )
    outputs = _mapping(root["outputs"], {"cases"})
    output_cases = _sequence(outputs["cases"], length=CASE_COUNT)
    order = _sequence(layout["case_order"], length=CASE_COUNT)
    if (
        layout["window_count"] != 2
        or layout["channel_count"] != 2
        or layout["case_count"] != CASE_COUNT
        or layout["artifact_count"] != CASE_COUNT
        or [
            item.get("case_id") if isinstance(item, dict) else None
            for item in output_cases
        ]
        != order
    ):
        raise AmiNaturalTurnFixtureError()
    pairs: list[tuple[str, str]] = []
    for raw_output in output_cases:
        output = _mapping(
            raw_output,
            {
                "case_id",
                "window_id",
                "channel",
                "file",
                "samples",
                "size_bytes",
                "source_pcm16_sha256",
                "source_f32_sha256",
                "sha256",
            },
        )
        _safe_id(output["case_id"])
        _safe_relative(output["file"])
        if output["samples"] != CASE_SAMPLES or output["size_bytes"] != _MAX_PCM_BYTES:
            raise AmiNaturalTurnFixtureError()
        pairs.append((_string(output["window_id"]), _string(output["channel"])))
        for key in ("source_pcm16_sha256", "source_f32_sha256", "sha256"):
            _sha256(output[key])
    if set(pairs) != {
        ("nested_backchannel", "close"),
        ("nested_backchannel", "far"),
        ("adjacent_exchange", "close"),
        ("adjacent_exchange", "far"),
    }:
        raise AmiNaturalTurnFixtureError()

    evidence = _mapping(
        root["evidence_scope"],
        {
            "diagnostic_only",
            "complete_selected_envelope",
            "complete_acoustic_turn",
            "endpoint_evidence",
            "vad_evidence",
            "latency_evidence",
            "wer_authority",
            "capture_device_evidence",
            "aec_evidence",
            "barge_in_evidence",
            "promotion_authority",
        },
    )
    if evidence != {
        "diagnostic_only": True,
        "complete_selected_envelope": True,
        "complete_acoustic_turn": False,
        "endpoint_evidence": False,
        "vad_evidence": False,
        "latency_evidence": False,
        "wer_authority": False,
        "capture_device_evidence": False,
        "aec_evidence": False,
        "barge_in_evidence": False,
        "promotion_authority": False,
    }:
        raise AmiNaturalTurnFixtureError()
    return _FixtureLock(
        fixture_id,
        production,
        raw_sha256,
        raw_size_bytes,
        recipe,
        root,
        tuple(members),
    )


def _load_fixture_lock(
    path: Path | str = DEFAULT_LOCK,
    *,
    test_injection: AmiNaturalTurnTestInjection | None = None,
) -> _FixtureLock:
    selected = Path(os.path.abspath(Path(path).expanduser()))
    production = test_injection is None
    if production:
        expected_id = FIXTURE_ID
        expected_sha = LOCK_FILE_SHA256
        expected_bytes = LOCK_FILE_BYTES
        try:
            if selected != DEFAULT_LOCK or selected.resolve(strict=True) != selected:
                raise AmiNaturalTurnFixtureError()
        except (OSError, RuntimeError, ValueError):
            raise AmiNaturalTurnFixtureError() from None
    else:
        if (
            type(test_injection) is not AmiNaturalTurnTestInjection
            or test_injection.fixture_id == FIXTURE_ID
            or _SAFE_ID_RE.fullmatch(test_injection.fixture_id) is None
            or _SHA256_RE.fullmatch(test_injection.lock_sha256) is None
            or isinstance(test_injection.lock_size_bytes, bool)
            or not 1 <= test_injection.lock_size_bytes <= _MAX_LOCK_BYTES
        ):
            raise AmiNaturalTurnFixtureError()
        expected_id = test_injection.fixture_id
        expected_sha = test_injection.lock_sha256
        expected_bytes = test_injection.lock_size_bytes
    try:
        snapshot = read_regular_bounded(
            selected,
            maximum_bytes=_MAX_LOCK_BYTES,
            expected_bytes=expected_bytes,
        )
    except BoundedReadError:
        raise AmiNaturalTurnFixtureError() from None
    digest = hashlib.sha256(snapshot.data).hexdigest()
    if digest != expected_sha:
        raise AmiNaturalTurnFixtureError()
    return _parse_lock(
        _strict_json(snapshot.data, maximum_bytes=_MAX_LOCK_BYTES),
        raw_sha256=digest,
        raw_size_bytes=len(snapshot.data),
        production=production,
        expected_fixture_id=expected_id,
    )


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
    )


def _private_root(path: Path | str) -> tuple[Path, int, tuple[int, ...]]:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise AmiNaturalTurnFixtureError()
    root = Path(os.path.abspath(supplied))
    descriptor = -1
    try:
        if root.resolve(strict=True) != root:
            raise AmiNaturalTurnFixtureError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        before = root.lstat()
        descriptor = os.open(root, flags)
        opened = os.fstat(descriptor)
        identity = _file_identity(opened)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o700
            or opened.st_uid != os.getuid()
            or _file_identity(before) != identity
            or _file_identity(root.lstat()) != identity
        ):
            raise AmiNaturalTurnFixtureError()
        return root, descriptor, identity
    except AmiNaturalTurnFixtureError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except (OSError, RuntimeError, ValueError):
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise AmiNaturalTurnFixtureError() from None


def _read_private_leaf(
    root: Path,
    descriptor: int,
    name: str,
    *,
    maximum_bytes: int,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> _PrivateFile:
    if _safe_relative(name) != name or len(PurePosixPath(name).parts) != 1:
        raise AmiNaturalTurnFixtureError()
    file_descriptor = -1
    try:
        before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum_bytes
            or (expected_bytes is not None and before.st_size != expected_bytes)
        ):
            raise AmiNaturalTurnFixtureError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        file_descriptor = os.open(name, flags, dir_fd=descriptor)
        opened = os.fstat(file_descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise AmiNaturalTurnFixtureError()
        raw = bytearray()
        while len(raw) <= maximum_bytes:
            chunk = os.read(file_descriptor, min(65_536, maximum_bytes + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(file_descriptor)
        current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        digest = hashlib.sha256(raw).hexdigest()
        if (
            len(raw) != opened.st_size
            or len(raw) > maximum_bytes
            or _file_identity(after) != _file_identity(opened)
            or _file_identity(current) != _file_identity(opened)
            or _file_identity((root / name).lstat()) != _file_identity(opened)
            or (expected_sha256 is not None and digest != expected_sha256)
        ):
            raise AmiNaturalTurnFixtureError()
        return _PrivateFile(name, bytes(raw), digest, _file_identity(opened))
    except AmiNaturalTurnFixtureError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise AmiNaturalTurnFixtureError() from None
    finally:
        if file_descriptor >= 0:
            try:
                os.close(file_descriptor)
            except OSError:
                pass


def _interval(value: object, *, samples: int) -> SampleInterval | None:
    if value is None:
        return None
    raw = _mapping(value, {"start_sample", "end_sample"})
    start = _integer(raw["start_sample"], minimum=0, maximum=samples)
    end = _integer(raw["end_sample"], minimum=start + 1, maximum=samples)
    return SampleInterval(start, end)


def _parse_metadata_case(value: object, *, index: int) -> AmiNaturalTurnCase:
    raw = _mapping(
        value,
        {
            "case_id",
            "window_id",
            "channel",
            "replay_case_index",
            "samples",
            "frame_samples",
            "source_start_sample",
            "source_end_sample",
            "source_offset_sample",
            "source_end_output_sample",
            "reference",
            "reference_sha256",
            "dialogue_acts",
            "adjacency_pair_id",
            "overlap_interval",
            "gap_interval",
        },
    )
    samples = _integer(raw["samples"], minimum=FRAME_SAMPLES, maximum=CASE_SAMPLES)
    frame_samples = _integer(raw["frame_samples"], minimum=1, maximum=samples)
    source_start = _integer(raw["source_start_sample"], minimum=0)
    source_end = _integer(raw["source_end_sample"], minimum=source_start + 1)
    source_offset = _integer(raw["source_offset_sample"], minimum=0, maximum=samples)
    source_output_end = _integer(
        raw["source_end_output_sample"], minimum=source_offset + 1, maximum=samples
    )
    if source_output_end - source_offset != source_end - source_start:
        raise AmiNaturalTurnFixtureError()
    reference = _string(raw["reference"], maximum=4096)
    reference_sha = _sha256(raw["reference_sha256"])
    if hashlib.sha256(reference.encode("utf-8")).hexdigest() != reference_sha:
        raise AmiNaturalTurnFixtureError()
    acts: list[AmiNaturalTurnDialogueAct] = []
    for raw_act in _sequence(raw["dialogue_acts"], length=2):
        act = _mapping(
            raw_act,
            {
                "dialogue_act_id",
                "speaker_id",
                "type_href",
                "source_start_sample",
                "source_end_sample",
                "output_start_sample",
                "output_end_sample",
                "reference",
                "reference_sha256",
            },
        )
        act_reference = _string(act["reference"], maximum=2048)
        act_hash = _sha256(act["reference_sha256"])
        act_start = _integer(
            act["source_start_sample"], minimum=source_start, maximum=source_end
        )
        act_end = _integer(
            act["source_end_sample"], minimum=act_start + 1, maximum=source_end
        )
        out_start = _integer(
            act["output_start_sample"], minimum=source_offset, maximum=source_output_end
        )
        out_end = _integer(
            act["output_end_sample"], minimum=out_start + 1, maximum=source_output_end
        )
        if (
            hashlib.sha256(act_reference.encode("utf-8")).hexdigest() != act_hash
            or out_start - source_offset != act_start - source_start
            or out_end - source_offset != act_end - source_start
        ):
            raise AmiNaturalTurnFixtureError()
        acts.append(
            AmiNaturalTurnDialogueAct(
                dialogue_act_id=_string(act["dialogue_act_id"]),
                speaker_id=_string(act["speaker_id"], maximum=8),
                type_href=_string(act["type_href"]),
                source_start_sample=act_start,
                source_end_sample=act_end,
                output_start_sample=out_start,
                output_end_sample=out_end,
                reference=act_reference,
                reference_sha256=act_hash,
            )
        )
    if len({act.speaker_id for act in acts}) != 2:
        raise AmiNaturalTurnFixtureError()
    adjacency = raw["adjacency_pair_id"]
    if adjacency is not None:
        adjacency = _string(adjacency)
    overlap = _interval(raw["overlap_interval"], samples=samples)
    gap = _interval(raw["gap_interval"], samples=samples)
    if (overlap is None) == (gap is None):
        raise AmiNaturalTurnFixtureError()
    return AmiNaturalTurnCase(
        case_id=_safe_id(raw["case_id"]),
        window_id=_safe_id(raw["window_id"]),
        channel=_string(raw["channel"], maximum=8),
        replay_case_index=_integer(
            raw["replay_case_index"], minimum=0, maximum=CASE_COUNT - 1
        ),
        samples=samples,
        frame_samples=frame_samples,
        source_start_sample=source_start,
        source_end_sample=source_end,
        source_offset_sample=source_offset,
        source_end_output_sample=source_output_end,
        reference=reference,
        reference_sha256=reference_sha,
        dialogue_acts=tuple(acts),
        adjacency_pair_id=adjacency,
        overlap_interval=overlap,
        gap_interval=gap,
    )


def _label_rows(cases: tuple[AmiNaturalTurnCase, ...]) -> list[dict[str, object]]:
    return [
        {
            "case_id": case.case_id,
            "window_id": case.window_id,
            "channel": case.channel,
            "reference_sha256": case.reference_sha256,
            "dialogue_acts": [
                {
                    "dialogue_act_id": act.dialogue_act_id,
                    "reference_sha256": act.reference_sha256,
                    "source_start_sample": act.source_start_sample,
                    "source_end_sample": act.source_end_sample,
                }
                for act in case.dialogue_acts
            ],
        }
        for case in cases
    ]


def _preparer_closure_sha256() -> str:
    root = Path(__file__).resolve().parents[2]
    rows: list[dict[str, object]] = []
    for relative in _PREPARER_FILES:
        try:
            snapshot = read_regular_bounded(
                root.joinpath(*PurePosixPath(relative).parts),
                maximum_bytes=4 * 1024 * 1024,
            )
        except BoundedReadError:
            raise AmiNaturalTurnFixtureError() from None
        rows.append(
            {
                "path": relative,
                "size_bytes": len(snapshot.data),
                "sha256": hashlib.sha256(snapshot.data).hexdigest(),
            }
        )
    return _canonical_sha256(rows)


def _receipt_binding(value: Mapping[str, object]) -> str:
    copied = dict(value)
    copied.pop("binding_sha256", None)
    return _canonical_sha256(copied)


def _source_contract_sha256(lock: _FixtureLock) -> str:
    """Bind the exact upstream bytes, annotations, selector, transform, and PCM."""

    return _canonical_sha256(
        {
            "fixture_id": lock.fixture_id,
            "lock_recipe_sha256": lock.recipe_sha256,
            "sources": lock.value["sources"],
            "archive_contract": lock.value["archive_contract"],
            "selection": lock.value["selection"],
            "transform": lock.value["transform"],
            "outputs": lock.value["outputs"],
        }
    )


def _locked_case_rows(lock: _FixtureLock) -> tuple[Mapping[str, object], ...]:
    selection = _mapping(
        lock.value["selection"],
        {"meeting_id", "algorithm", "selected_windows_sha256", "windows"},
    )
    windows = {
        _string(window["window_id"]): window
        for window in _sequence(selection["windows"], length=2)
        if isinstance(window, dict)
    }
    outputs = _sequence(
        _mapping(lock.value["outputs"], {"cases"})["cases"], length=CASE_COUNT
    )
    transform = _mapping(lock.value["transform"], set(lock.value["transform"]))
    prefix = _integer(transform["prefix_samples"], minimum=1, maximum=CASE_SAMPLES)
    rows: list[Mapping[str, object]] = []
    for index, raw_output in enumerate(outputs):
        output = _mapping(raw_output, set(raw_output))
        window = windows.get(_string(output["window_id"]))
        if window is None:
            raise AmiNaturalTurnFixtureError()
        start = _integer(window["start_sample"], minimum=0)
        end = _integer(window["end_sample"], minimum=start + 1)
        acts = _sequence(window["dialogue_acts"], length=2)
        relation = _mapping(window["relation"], {"kind", "start_sample", "end_sample"})
        relation_interval = {
            "start_sample": prefix
            + _integer(relation["start_sample"], minimum=start)
            - start,
            "end_sample": prefix
            + _integer(relation["end_sample"], minimum=start + 1)
            - start,
        }
        adjacency = window["adjacency_pair"]
        rows.append(
            {
                "case_id": output["case_id"],
                "window_id": output["window_id"],
                "channel": output["channel"],
                "replay_case_index": index,
                "samples": output["samples"],
                "frame_samples": transform["frame_samples"],
                "source_start_sample": start,
                "source_end_sample": end,
                "source_offset_sample": prefix,
                "source_end_output_sample": prefix + end - start,
                "reference_sha256": window["linearized_reference_sha256"],
                "dialogue_acts": [
                    {
                        "dialogue_act_id": act["id"],
                        "speaker_id": act["speaker"],
                        "type_href": act["type_href"],
                        "source_start_sample": act["start_sample"],
                        "source_end_sample": act["end_sample"],
                        "output_start_sample": prefix + act["start_sample"] - start,
                        "output_end_sample": prefix + act["end_sample"] - start,
                        "reference_sha256": act["reference_sha256"],
                    }
                    for act in acts
                    if isinstance(act, dict)
                ],
                "adjacency_pair_id": (
                    None
                    if adjacency is None
                    else _mapping(
                        adjacency,
                        {"id", "type_href", "source_href", "target_href"},
                    )["id"]
                ),
                "overlap_interval": (
                    relation_interval if relation["kind"] == "nested-overlap" else None
                ),
                "gap_interval": (
                    relation_interval if relation["kind"] == "adjacent-gap" else None
                ),
            }
        )
    return tuple(rows)


def _validate_cases_against_lock(
    lock: _FixtureLock, cases: tuple[AmiNaturalTurnCase, ...]
) -> None:
    expected = _locked_case_rows(lock)
    if len(cases) != len(expected):
        raise AmiNaturalTurnFixtureError()
    for case, row in zip(cases, expected):
        observed = {
            "case_id": case.case_id,
            "window_id": case.window_id,
            "channel": case.channel,
            "replay_case_index": case.replay_case_index,
            "samples": case.samples,
            "frame_samples": case.frame_samples,
            "source_start_sample": case.source_start_sample,
            "source_end_sample": case.source_end_sample,
            "source_offset_sample": case.source_offset_sample,
            "source_end_output_sample": case.source_end_output_sample,
            "reference_sha256": case.reference_sha256,
            "dialogue_acts": [
                {
                    "dialogue_act_id": act.dialogue_act_id,
                    "speaker_id": act.speaker_id,
                    "type_href": act.type_href,
                    "source_start_sample": act.source_start_sample,
                    "source_end_sample": act.source_end_sample,
                    "output_start_sample": act.output_start_sample,
                    "output_end_sample": act.output_end_sample,
                    "reference_sha256": act.reference_sha256,
                }
                for act in case.dialogue_acts
            ],
            "adjacency_pair_id": case.adjacency_pair_id,
            "overlap_interval": (
                None
                if case.overlap_interval is None
                else {
                    "start_sample": case.overlap_interval.start_sample,
                    "end_sample": case.overlap_interval.end_sample,
                }
            ),
            "gap_interval": (
                None
                if case.gap_interval is None
                else {
                    "start_sample": case.gap_interval.start_sample,
                    "end_sample": case.gap_interval.end_sample,
                }
            ),
        }
        if observed != row:
            raise AmiNaturalTurnFixtureError()


def _merged_intervals(
    intervals: list[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if start >= end:
            raise AmiNaturalTurnFixtureError()
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def _validate_replay_semantics(
    lock: _FixtureLock,
    replay: LoadedReplayCorpus,
    cases: tuple[AmiNaturalTurnCase, ...],
) -> None:
    locked_windows = {
        item["window_id"]: item
        for item in _sequence(
            _mapping(
                lock.value["selection"],
                {"meeting_id", "algorithm", "selected_windows_sha256", "windows"},
            )["windows"],
            length=2,
        )
        if isinstance(item, dict)
    }
    expected_purpose = (
        f"AMI ES2004a natural-turn capture replay v1;fixture_id={lock.fixture_id};"
        f"lock_recipe_sha256={lock.recipe_sha256};diagnostic_only=true"
    )
    if replay.purpose != expected_purpose:
        raise AmiNaturalTurnFixtureError()
    for index, case in enumerate(cases):
        replay_case = replay.cases[index]
        window = locked_windows.get(case.window_id)
        if window is None:
            raise AmiNaturalTurnFixtureError()
        expected_tags = (
            "ami",
            "es2004a",
            "natural-turn",
            case.window_id,
            case.channel,
            "calibration-roomtone-2s",
            "postroll-roomtone-repeat",
            "diagnostic-only",
            "overlapping-speakers"
            if case.overlap_interval is not None
            else "turn-transition",
        )
        timed = [
            row
            for row in _sequence(window["timed_words"])
            if isinstance(row, dict) and row.get("punctuation") is False
        ]
        expected_word_geometry = [
            (
                row["speaker"],
                case.source_offset_sample
                + row["start_sample"]
                - case.source_start_sample,
                case.source_offset_sample
                + row["end_sample"]
                - case.source_start_sample,
            )
            for row in timed
        ]
        observed_word_geometry = [
            (word.speaker_id, word.start_sample, word.end_sample)
            for word in replay_case.word_intervals
        ]
        if sorted(observed_word_geometry) != sorted(expected_word_geometry):
            raise AmiNaturalTurnFixtureError()
        if " ".join(word.text for word in replay_case.word_intervals) != case.reference:
            raise AmiNaturalTurnFixtureError()
        expected_speech = _merged_intervals(
            [(start, end) for _speaker, start, end in expected_word_geometry]
        )
        observed_speech = tuple(
            (interval.start_sample, interval.end_sample)
            for interval in replay_case.speech_intervals
        )
        expected_speakers = sorted(
            (
                speaker,
                start,
                end,
            )
            for speaker in {row[0] for row in expected_word_geometry}
            for start, end in _merged_intervals(
                [
                    (left, right)
                    for current, left, right in expected_word_geometry
                    if current == speaker
                ]
            )
        )
        observed_speakers = sorted(
            (item.speaker_id, item.start_sample, item.end_sample)
            for item in replay_case.speaker_intervals
        )
        if (
            replay_case.assertion.value != "transcript"
            or replay_case.expected_text != case.reference
            or replay_case.commands
            or replay_case.aec_delay_samples != 0
            or replay_case.tags != expected_tags
            or observed_speech != expected_speech
            or observed_speakers != expected_speakers
            or set(replay_case.tracks) != {"mic"}
        ):
            raise AmiNaturalTurnFixtureError()


def load_ami_natural_turn_fixture(
    path: Path | str,
    *,
    lock_path: Path | str = DEFAULT_LOCK,
    _test_injection: AmiNaturalTurnTestInjection | None = None,
) -> LoadedAmiNaturalTurnFixture:
    """Load one exact private bundle into immutable in-memory snapshots."""

    lock = _load_fixture_lock(lock_path, test_injection=_test_injection)
    root, descriptor, root_identity = _private_root(path)
    try:
        expected_names = {
            MANIFEST_FILENAME,
            METADATA_FILENAME,
            PREPARATION_RECEIPT_FILENAME,
            *(
                _safe_relative(item["file"])
                for item in _sequence(
                    _mapping(lock.value["outputs"], {"cases"})["cases"],
                    length=CASE_COUNT,
                )
            ),
        }
        try:
            names = set(os.listdir(descriptor))
        except OSError:
            raise AmiNaturalTurnFixtureError() from None
        if names != expected_names:
            raise AmiNaturalTurnFixtureError()
        receipt_file = _read_private_leaf(
            root,
            descriptor,
            PREPARATION_RECEIPT_FILENAME,
            maximum_bytes=_MAX_JSON_BYTES,
        )
        receipt = _mapping(
            _strict_json(receipt_file.raw, maximum_bytes=_MAX_JSON_BYTES),
            {
                "schema_version",
                "kind",
                "fixture_id",
                "production_evidence",
                "accepted_terms",
                "lock_sha256",
                "lock_recipe_sha256",
                "source_contract_sha256",
                "preparer_closure_sha256",
                "labels_sha256",
                "manifest",
                "metadata",
                "artifacts",
                "case_count",
                "case_samples",
                "frame_samples",
                "evidence_scope",
                "binding_sha256",
            },
        )
        if (
            receipt_file.raw != _canonical_json(receipt, newline=True)
            or receipt["schema_version"] != SCHEMA_VERSION
            or receipt["kind"] != RECEIPT_KIND
            or receipt["fixture_id"] != lock.fixture_id
            or receipt["production_evidence"] is not lock.production_evidence
            or receipt["accepted_terms"] != ["CC-BY-4.0"]
            or receipt["lock_sha256"] != lock.raw_sha256
            or receipt["lock_recipe_sha256"] != lock.recipe_sha256
            or receipt["case_count"] != CASE_COUNT
            or receipt["case_samples"] != CASE_SAMPLES
            or receipt["frame_samples"] != FRAME_SAMPLES
            or _receipt_binding(receipt) != _sha256(receipt["binding_sha256"])
            or receipt["evidence_scope"] != lock.value["evidence_scope"]
        ):
            raise AmiNaturalTurnFixtureError()
        source_contract = _sha256(receipt["source_contract_sha256"])
        if source_contract != _source_contract_sha256(lock):
            raise AmiNaturalTurnFixtureError()
        closure = _sha256(receipt["preparer_closure_sha256"])
        if closure != _preparer_closure_sha256():
            raise AmiNaturalTurnFixtureError()

        manifest_pin = _mapping(receipt["manifest"], {"file", "size_bytes", "sha256"})
        metadata_pin = _mapping(receipt["metadata"], {"file", "size_bytes", "sha256"})
        if (
            manifest_pin["file"] != MANIFEST_FILENAME
            or metadata_pin["file"] != METADATA_FILENAME
        ):
            raise AmiNaturalTurnFixtureError()
        manifest_file = _read_private_leaf(
            root,
            descriptor,
            MANIFEST_FILENAME,
            maximum_bytes=_MAX_JSON_BYTES,
            expected_bytes=_integer(manifest_pin["size_bytes"], minimum=1),
            expected_sha256=_sha256(manifest_pin["sha256"]),
        )
        metadata_file = _read_private_leaf(
            root,
            descriptor,
            METADATA_FILENAME,
            maximum_bytes=_MAX_JSON_BYTES,
            expected_bytes=_integer(metadata_pin["size_bytes"], minimum=1),
            expected_sha256=_sha256(metadata_pin["sha256"]),
        )
        metadata = _mapping(
            _strict_json(metadata_file.raw, maximum_bytes=_MAX_JSON_BYTES),
            {
                "schema_version",
                "kind",
                "fixture_id",
                "production_evidence",
                "lock_sha256",
                "lock_recipe_sha256",
                "source_contract_sha256",
                "labels_sha256",
                "sample_rate_hz",
                "frame_samples",
                "case_samples",
                "cases",
            },
        )
        cases = tuple(
            _parse_metadata_case(item, index=index)
            for index, item in enumerate(
                _sequence(metadata["cases"], length=CASE_COUNT)
            )
        )
        _validate_cases_against_lock(lock, cases)
        labels = _canonical_sha256(_label_rows(cases))
        if (
            metadata_file.raw != _canonical_json(metadata, newline=True)
            or metadata["schema_version"] != SCHEMA_VERSION
            or metadata["kind"] != METADATA_KIND
            or metadata["fixture_id"] != lock.fixture_id
            or metadata["production_evidence"] is not lock.production_evidence
            or metadata["lock_sha256"] != lock.raw_sha256
            or metadata["lock_recipe_sha256"] != lock.recipe_sha256
            or metadata["source_contract_sha256"] != source_contract
            or metadata["labels_sha256"] != labels
            or receipt["labels_sha256"] != labels
            or metadata["sample_rate_hz"] != SAMPLE_RATE_HZ
            or metadata["frame_samples"] != FRAME_SAMPLES
            or metadata["case_samples"] != CASE_SAMPLES
        ):
            raise AmiNaturalTurnFixtureError()
        artifact_pins = _sequence(receipt["artifacts"], length=CASE_COUNT)
        artifacts: list[_PrivateFile] = []
        expected_outputs = _sequence(
            _mapping(lock.value["outputs"], {"cases"})["cases"], length=CASE_COUNT
        )
        for index, (raw_pin, raw_output) in enumerate(
            zip(artifact_pins, expected_outputs)
        ):
            pin = _mapping(raw_pin, {"case_id", "file", "size_bytes", "sha256"})
            output = _mapping(raw_output, set(raw_output))
            if (
                pin["case_id"] != cases[index].case_id
                or pin["file"] != output["file"]
                or pin["size_bytes"] != output["size_bytes"]
                or (lock.production_evidence and pin["sha256"] != output["sha256"])
            ):
                raise AmiNaturalTurnFixtureError()
            artifacts.append(
                _read_private_leaf(
                    root,
                    descriptor,
                    _safe_relative(pin["file"]),
                    maximum_bytes=_MAX_PCM_BYTES,
                    expected_bytes=_integer(pin["size_bytes"], minimum=1),
                    expected_sha256=_sha256(pin["sha256"]),
                )
            )

        try:
            replay = load_corpus(root / MANIFEST_FILENAME)
            verify_corpus_snapshot(replay)
        except ReplayCorpusError:
            raise AmiNaturalTurnFixtureError() from None
        if (
            replay.digest != manifest_file.sha256
            or replay.sample_rate_hz != SAMPLE_RATE_HZ
            or replay.frame_samples != FRAME_SAMPLES
            or len(replay.cases) != CASE_COUNT
            or [case.case_id for case in replay.cases]
            != [case.case_id for case in cases]
            or any(case.samples != CASE_SAMPLES for case in replay.cases)
            or any(case.replay_case_index != index for index, case in enumerate(cases))
            or any(
                replay.cases[index].expected_text != case.reference
                for index, case in enumerate(cases)
            )
            or any(
                replay.cases[index].mic.sha256 != artifacts[index].sha256
                for index in range(CASE_COUNT)
            )
        ):
            raise AmiNaturalTurnFixtureError()
        _validate_replay_semantics(lock, replay, cases)
        for index, case in enumerate(cases):
            tags = set(replay.cases[index].tags)
            if (
                case.overlap_interval is not None and "overlapping-speakers" not in tags
            ) or (case.gap_interval is not None and "turn-transition" not in tags):
                raise AmiNaturalTurnFixtureError()

        retained_files = (receipt_file, manifest_file, metadata_file, *artifacts)
        for saved in retained_files:
            maximum = (
                _MAX_PCM_BYTES if saved.name.endswith(".f32le") else _MAX_JSON_BYTES
            )
            current = _read_private_leaf(
                root,
                descriptor,
                saved.name,
                maximum_bytes=maximum,
                expected_bytes=len(saved.raw),
                expected_sha256=saved.sha256,
            )
            if current != saved:
                raise AmiNaturalTurnFixtureError()
        try:
            if set(os.listdir(descriptor)) != expected_names:
                raise AmiNaturalTurnFixtureError()
        except OSError:
            raise AmiNaturalTurnFixtureError() from None
        current_lock = _load_fixture_lock(lock_path, test_injection=_test_injection)
        if current_lock != lock:
            raise AmiNaturalTurnFixtureError()
        current_root = _file_identity(os.fstat(descriptor))
        if (
            current_root != root_identity
            or _file_identity(root.lstat()) != root_identity
        ):
            raise AmiNaturalTurnFixtureError()
        retained = (
            (receipt_file.name, receipt_file.identity),
            (manifest_file.name, manifest_file.identity),
            (metadata_file.name, metadata_file.identity),
            *((item.name, item.identity) for item in artifacts),
        )
        return LoadedAmiNaturalTurnFixture(
            root=root,
            fixture_id=lock.fixture_id,
            production_evidence=lock.production_evidence,
            lock_sha256=lock.raw_sha256,
            lock_recipe_sha256=lock.recipe_sha256,
            labels_sha256=labels,
            metadata_sha256=metadata_file.sha256,
            manifest_sha256=manifest_file.sha256,
            receipt_sha256=receipt_file.sha256,
            source_contract_sha256=source_contract,
            preparer_closure_sha256=closure,
            sample_rate_hz=SAMPLE_RATE_HZ,
            frame_samples=FRAME_SAMPLES,
            case_samples=CASE_SAMPLES,
            replay_corpus=replay,
            cases=cases,
            directory_identity=root_identity,
            file_identities=retained,
            lock_path=Path(lock_path).resolve(),
            lock_size_bytes=lock.raw_size_bytes,
        )
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def verify_ami_natural_turn_fixture_snapshot(
    value: LoadedAmiNaturalTurnFixture,
) -> None:
    """Reopen every bound leaf and reject any bundle or lock drift."""

    if type(value) is not LoadedAmiNaturalTurnFixture:
        raise AmiNaturalTurnFixtureError()
    injection = None
    if not value.production_evidence:
        injection = AmiNaturalTurnTestInjection(
            fixture_id=value.fixture_id,
            lock_sha256=value.lock_sha256,
            lock_size_bytes=value.lock_size_bytes,
        )
    current = load_ami_natural_turn_fixture(
        value.root,
        lock_path=value.lock_path,
        _test_injection=injection,
    )
    if current != value:
        raise AmiNaturalTurnFixtureError()


__all__ = [
    "AmiNaturalTurnCase",
    "AmiNaturalTurnDialogueAct",
    "AmiNaturalTurnFixtureError",
    "AmiNaturalTurnTestInjection",
    "LoadedAmiNaturalTurnFixture",
    "load_ami_natural_turn_fixture",
    "verify_ami_natural_turn_fixture_snapshot",
]
