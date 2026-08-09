"""Prepare the exact private AMI ES2004a natural-turn capture-replay slice.

The command is offline and fail closed.  It accepts only already-present,
private source files; never downloads, runs a model, opens an audio device, or
prints transcript text.  A receipt containing no transcript text is the final
publication step for the seven-leaf bundle.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import string
import struct
from typing import Callable, Mapping, Sequence
import unicodedata
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from tools.capture_replay.ami_natural_turn import (
    CASE_COUNT,
    CASE_SAMPLES,
    DEFAULT_LOCK,
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    METADATA_KIND,
    PREPARATION_RECEIPT_FILENAME,
    RECEIPT_KIND,
    SCHEMA_VERSION,
    AmiNaturalTurnFixtureError,
    AmiNaturalTurnTestInjection,
    _canonical_json,
    _canonical_sha256,
    _label_rows,
    _load_fixture_lock,
    _mapping,
    _preparer_closure_sha256,
    _parse_metadata_case,
    _receipt_binding,
    _sequence,
    _source_contract_sha256,
    _validate_cases_against_lock,
    _validate_replay_semantics,
)
from tools.capture_replay.corpus import (
    FRAME_SAMPLES,
    SAMPLE_RATE_HZ,
    ReplayCorpusError,
    load_corpus,
    verify_corpus_snapshot,
)
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    opened_directory_nofollow,
    read_regular_bounded,
)


_ANNOTATION_MAX_BYTES = 32 * 1024 * 1024
_AUDIO_MAX_BYTES = 64 * 1024 * 1024
_MAX_XML_ELEMENTS = 128_000
_MAX_REFERENCE_CHARS = 4_096
_SAFE_ERROR: Mapping[str, object] = {
    "ok": False,
    "error": "ami_natural_turn_prerequisites_unavailable",
}
_HREF_RANGE_RE = re.compile(
    r"(?P<file>ES2004a\.(?P<speaker>[ABCD])\.words\.xml)#"
    r"id\((?P<start>ES2004a\.(?P=speaker)\.words\d+)\)\.\."
    r"id\((?P<end>ES2004a\.(?P=speaker)\.words\d+)\)\Z"
)
_HREF_SINGLE_RE = re.compile(
    r"(?P<file>ES2004a\.(?P<speaker>[ABCD])\.dialog-act\.xml)#"
    r"id\((?P<id>ES2004a\.(?P=speaker)\.dialog-act\.s\d+\.\d+)\)\Z"
)


class AmiNaturalTurnPreparationError(RuntimeError):
    """A detail-free source, transform, or publication failure."""


class _LifecycleSignal(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
    path: Path = field(repr=False)
    raw: bytes = field(repr=False)
    sha256: str
    size_bytes: int
    identity: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _WordNode:
    source_id: str
    speaker: str
    text: str = field(repr=False)
    start_sample: int
    end_sample: int
    punctuation: bool
    ordinal: int


@dataclass(frozen=True, slots=True)
class _Activity:
    speaker: str
    kind: str
    start_sample: int
    end_sample: int
    ordinal: int


@dataclass(frozen=True, slots=True)
class _RenderedWord:
    source_id: str
    speaker: str
    text: str = field(repr=False)
    start_sample: int
    end_sample: int


@dataclass(frozen=True, slots=True)
class _ParsedWindow:
    window_id: str
    start_sample: int
    end_sample: int
    reference: str = field(repr=False)
    reference_sha256: str
    dialogue_acts: tuple[Mapping[str, object], ...] = field(repr=False)
    words: tuple[_RenderedWord, ...] = field(repr=False)
    adjacency_pair_id: str | None
    relation_kind: str
    relation_start_sample: int
    relation_end_sample: int


@dataclass(frozen=True, slots=True)
class _PreparedCase:
    case_id: str
    filename: str
    pcm: bytes = field(repr=False)
    manifest: Mapping[str, object] = field(repr=False)
    metadata: Mapping[str, object] = field(repr=False)


@dataclass(slots=True)
class _CommitState:
    committed: bool = False
    pending_prepared: PreparedAmiNaturalTurnFixture | None = field(
        default=None,
        repr=False,
    )


@dataclass(frozen=True, slots=True)
class PreparedAmiNaturalTurnFixture:
    """Transcript-free identities known before the terminal receipt link."""

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
    case_count: int
    case_samples: int


def _fail() -> object:
    raise AmiNaturalTurnPreparationError()


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


def _directory_locator(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
    )


def _read_private_source(
    path: Path | str,
    *,
    expected_sha256: str,
    expected_bytes: int,
    maximum_bytes: int,
) -> _SourceSnapshot:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise AmiNaturalTurnPreparationError()
    candidate = Path(os.path.abspath(supplied))
    try:
        with opened_directory_nofollow(candidate.parent, require_private=True):
            before = candidate.lstat()
            if (
                candidate.resolve(strict=True) != candidate
                or not stat.S_ISREG(before.st_mode)
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_uid != os.getuid()
                or before.st_nlink != 1
                or before.st_size != expected_bytes
            ):
                raise AmiNaturalTurnPreparationError()
            snapshot = read_regular_bounded(
                candidate,
                maximum_bytes=maximum_bytes,
                expected_bytes=expected_bytes,
            )
            after = candidate.lstat()
    except AmiNaturalTurnPreparationError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        raise AmiNaturalTurnPreparationError() from None
    digest = hashlib.sha256(snapshot.data).hexdigest()
    if (
        digest != expected_sha256
        or snapshot.path != candidate
        or _file_identity(before) != _file_identity(after)
    ):
        raise AmiNaturalTurnPreparationError()
    return _SourceSnapshot(
        candidate,
        snapshot.data,
        digest,
        len(snapshot.data),
        _file_identity(after),
    )


def _verify_source(value: _SourceSnapshot, *, maximum_bytes: int) -> None:
    current = _read_private_source(
        value.path,
        expected_sha256=value.sha256,
        expected_bytes=value.size_bytes,
        maximum_bytes=maximum_bytes,
    )
    if current != value:
        raise AmiNaturalTurnPreparationError()


def _safe_zip_name(name: str) -> PurePosixPath:
    directory = name.endswith("/")
    lexical = name[:-1] if directory else name
    if (
        not name
        or not lexical
        or "\x00" in name
        or "\\" in name
        or unicodedata.normalize("NFC", name) != name
        or any(ord(character) < 32 for character in name)
    ):
        raise AmiNaturalTurnPreparationError()
    parts = lexical.split("/")
    path = PurePosixPath(lexical)
    if (
        path.is_absolute()
        or path.as_posix() != lexical
        or any(part in {"", ".", ".."} for part in parts)
        or ":" in parts[0]
    ):
        raise AmiNaturalTurnPreparationError()
    return path


def _zip_member_kind(info: zipfile.ZipInfo) -> None:
    mode = (info.external_attr >> 16) & 0xFFFF
    kind = stat.S_IFMT(mode)
    if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise AmiNaturalTurnPreparationError()
    if (info.is_dir() and kind == stat.S_IFREG) or (
        not info.is_dir() and kind == stat.S_IFDIR
    ):
        raise AmiNaturalTurnPreparationError()


def _archive_members(raw: bytes, lock) -> Mapping[str, bytes]:
    archive_contract = _mapping(
        lock.value["archive_contract"],
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
    required = {member.path: member for member in lock.members}
    selected: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r", allowZip64=False) as archive:
            infos = archive.infolist()
            if len(infos) != archive_contract["member_count"]:
                raise AmiNaturalTurnPreparationError()
            seen: set[str] = set()
            expanded = 0
            compressed = 0
            encrypted = 0
            zip64_sized = 0
            for info in infos:
                path = _safe_zip_name(info.filename).as_posix()
                folded = info.filename.casefold()
                if folded in seen:
                    raise AmiNaturalTurnPreparationError()
                seen.add(folded)
                _zip_member_kind(info)
                if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                    raise AmiNaturalTurnPreparationError()
                if (
                    info.file_size < 0
                    or info.file_size > archive_contract["maximum_member_bytes"]
                ):
                    raise AmiNaturalTurnPreparationError()
                expanded += info.file_size
                compressed += info.compress_size
                encrypted += int(bool(info.flag_bits & 0x1))
                zip64_sized += int(
                    info.file_size >= 0xFFFFFFFF or info.compress_size >= 0xFFFFFFFF
                )
                if info.is_dir() or path not in required:
                    continue
                with archive.open(info, "r") as handle:
                    payload = handle.read(info.file_size + 1)
                    trailing = handle.read(1)
                pin = required[path]
                if (
                    len(payload) != info.file_size
                    or trailing
                    or len(payload) != pin.size_bytes
                    or hashlib.sha256(payload).hexdigest() != pin.sha256
                ):
                    raise AmiNaturalTurnPreparationError()
                selected[path] = payload
            if (
                set(selected) != set(required)
                or expanded != archive_contract["expanded_bytes"]
                or compressed != archive_contract["compressed_payload_bytes"]
                or len(archive.comment) != archive_contract["archive_comment_bytes"]
                or encrypted != archive_contract["encrypted_members"]
                or zip64_sized != archive_contract["zip64_sized_members"]
                or expanded > archive_contract["maximum_expanded_bytes"]
                or len(infos) > archive_contract["maximum_members"]
            ):
                raise AmiNaturalTurnPreparationError()
    except AmiNaturalTurnPreparationError:
        raise
    except (
        EOFError,
        NotImplementedError,
        OSError,
        RuntimeError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ):
        raise AmiNaturalTurnPreparationError() from None
    return selected


def _riff_pcm16_mono(raw: bytes) -> np.ndarray:
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise AmiNaturalTurnPreparationError()
    if struct.unpack_from("<I", raw, 4)[0] + 8 != len(raw):
        raise AmiNaturalTurnPreparationError()
    offset = 12
    format_chunk: bytes | None = None
    data_chunk: bytes | None = None
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise AmiNaturalTurnPreparationError()
        chunk_id = raw[offset : offset + 4]
        size = struct.unpack_from("<I", raw, offset + 4)[0]
        offset += 8
        end = offset + size
        padded = end + (size & 1)
        if end > len(raw) or padded > len(raw):
            raise AmiNaturalTurnPreparationError()
        if chunk_id == b"fmt ":
            if format_chunk is not None:
                raise AmiNaturalTurnPreparationError()
            format_chunk = raw[offset:end]
        elif chunk_id == b"data":
            if data_chunk is not None:
                raise AmiNaturalTurnPreparationError()
            data_chunk = raw[offset:end]
        offset = padded
    if (
        offset != len(raw)
        or format_chunk is None
        or len(format_chunk) != 16
        or data_chunk is None
        or not data_chunk
        or len(data_chunk) % 2
    ):
        raise AmiNaturalTurnPreparationError()
    encoding, channels, rate, byte_rate, align, bits = struct.unpack(
        "<HHIIHH", format_chunk
    )
    if (
        encoding != 1
        or channels != 1
        or rate != SAMPLE_RATE_HZ
        or byte_rate != SAMPLE_RATE_HZ * 2
        or align != 2
        or bits != 16
    ):
        raise AmiNaturalTurnPreparationError()
    return np.frombuffer(data_chunk, dtype="<i2")


def _local_name(value: str) -> str:
    return value.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _attributes(element: ET.Element) -> Mapping[str, str]:
    result: dict[str, str] = {}
    for key, value in element.attrib.items():
        name = _local_name(key)
        if name in result:
            raise AmiNaturalTurnPreparationError()
        result[name] = value
    return result


def _xml_root(raw: bytes) -> ET.Element:
    if (
        not raw
        or b"<!doctype" in raw.lower()
        or b"<!entity" in raw.lower()
        or len(raw) > 16 * 1024 * 1024
    ):
        raise AmiNaturalTurnPreparationError()
    try:
        root = ET.fromstring(raw)
    except (ET.ParseError, UnicodeError, ValueError):
        raise AmiNaturalTurnPreparationError() from None
    if sum(1 for _ in root.iter()) > _MAX_XML_ELEMENTS:
        raise AmiNaturalTurnPreparationError()
    return root


def _decimal(value: object) -> Decimal:
    if not isinstance(value, str) or not value or len(value) > 32:
        raise AmiNaturalTurnPreparationError()
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError):
        raise AmiNaturalTurnPreparationError() from None
    if not result.is_finite() or result < 0:
        raise AmiNaturalTurnPreparationError()
    return result


def _sample(value: Decimal, *, end: bool) -> int:
    try:
        return int(
            (value * SAMPLE_RATE_HZ).to_integral_value(
                rounding=ROUND_CEILING if end else ROUND_FLOOR
            )
        )
    except (InvalidOperation, OverflowError, ValueError):
        raise AmiNaturalTurnPreparationError() from None


def _clean_text(value: object, *, punctuation: bool) -> str:
    if not isinstance(value, str):
        raise AmiNaturalTurnPreparationError()
    text = value.strip()
    if (
        not text
        or text != value
        or len(text) > (16 if punctuation else 128)
        or any(ord(character) < 32 for character in text)
        or (not punctuation and any(character.isspace() for character in text))
    ):
        raise AmiNaturalTurnPreparationError()
    if punctuation and any(
        character not in string.punctuation
        and not unicodedata.category(character).startswith("P")
        for character in text
    ):
        raise AmiNaturalTurnPreparationError()
    return text


def _parse_words(
    raw: bytes, *, speaker: str, audio_samples: int
) -> tuple[tuple[_WordNode, ...], tuple[_Activity, ...], tuple[str, ...]]:
    root = _xml_root(raw)
    if _local_name(root.tag) != "root":
        raise AmiNaturalTurnPreparationError()
    nodes: list[_WordNode] = []
    activities: list[_Activity] = []
    ids: set[str] = set()
    ordered_ids: list[str] = []
    previous_start: Decimal | None = None
    previous_word_end: Decimal | None = None
    for element in root:
        if list(element):
            raise AmiNaturalTurnPreparationError()
        name = _local_name(element.tag)
        attributes = _attributes(element)
        source_id = attributes.get("id")
        if not source_id or source_id in ids:
            raise AmiNaturalTurnPreparationError()
        ids.add(source_id)
        ordered_ids.append(source_id)
        has_start = "starttime" in attributes
        has_end = "endtime" in attributes
        if has_start != has_end:
            raise AmiNaturalTurnPreparationError()
        start: Decimal | None = None
        end: Decimal | None = None
        if has_start:
            start = _decimal(attributes["starttime"])
            end = _decimal(attributes["endtime"])
            if (
                end < start
                or _sample(end, end=True) > audio_samples
                or (previous_start is not None and start < previous_start)
            ):
                raise AmiNaturalTurnPreparationError()
            previous_start = start
        punctuation = name == "punc" or (
            name == "w" and attributes.get("punc") == "true"
        )
        if name not in {"w", "punc"}:
            if start is not None and end is not None and end > start:
                activities.append(
                    _Activity(
                        speaker,
                        name,
                        _sample(start, end=False),
                        _sample(end, end=True),
                        len(activities),
                    )
                )
            continue
        text = _clean_text("".join(element.itertext()), punctuation=punctuation)
        if punctuation:
            if (
                start is None
                or end is None
                or start != end
                or previous_word_end is None
                or start != previous_word_end
            ):
                raise AmiNaturalTurnPreparationError()
        else:
            if start is None or end is None or end <= start:
                raise AmiNaturalTurnPreparationError()
            previous_word_end = end
        assert start is not None and end is not None
        nodes.append(
            _WordNode(
                source_id,
                speaker,
                text,
                _sample(start, end=False),
                _sample(end, end=True),
                punctuation,
                len(nodes),
            )
        )
    if not nodes:
        raise AmiNaturalTurnPreparationError()
    return tuple(nodes), tuple(activities), tuple(ordered_ids)


def _range_nodes(
    href: str, words: Mapping[str, tuple[_WordNode, ...]]
) -> tuple[_WordNode, ...]:
    match = _HREF_RANGE_RE.fullmatch(href)
    if match is None:
        raise AmiNaturalTurnPreparationError()
    speaker = match.group("speaker")
    nodes = words[speaker]
    positions = {node.source_id: index for index, node in enumerate(nodes)}
    try:
        start = positions[match.group("start")]
        end = positions[match.group("end")]
    except KeyError:
        raise AmiNaturalTurnPreparationError() from None
    if start > end:
        raise AmiNaturalTurnPreparationError()
    return nodes[start : end + 1]


def _render_nodes(nodes: Sequence[_WordNode]) -> tuple[_RenderedWord, ...]:
    result: list[_RenderedWord] = []
    for node in nodes:
        if node.punctuation:
            if not result:
                raise AmiNaturalTurnPreparationError()
            previous = result[-1]
            result[-1] = _RenderedWord(
                previous.source_id,
                previous.speaker,
                previous.text + node.text,
                previous.start_sample,
                previous.end_sample,
            )
        else:
            result.append(
                _RenderedWord(
                    node.source_id,
                    node.speaker,
                    node.text,
                    node.start_sample,
                    node.end_sample,
                )
            )
    if not result:
        raise AmiNaturalTurnPreparationError()
    return tuple(result)


def _reference(rendered: Sequence[_RenderedWord]) -> str:
    value = " ".join(word.text for word in rendered)
    if not value or len(value) > _MAX_REFERENCE_CHARS:
        raise AmiNaturalTurnPreparationError()
    return value


def _id_elements(root: ET.Element, *, element_name: str) -> Mapping[str, ET.Element]:
    result: dict[str, ET.Element] = {}
    for element in root.iter():
        if _local_name(element.tag) != element_name:
            continue
        identity = _attributes(element).get("id")
        if not identity or identity in result:
            raise AmiNaturalTurnPreparationError()
        result[identity] = element
    return result


def _pointer_map(element: ET.Element) -> tuple[Mapping[str, str], tuple[str, ...]]:
    pointers: dict[str, str] = {}
    children: list[str] = []
    for child in element:
        name = _local_name(child.tag)
        attributes = _attributes(child)
        if list(child):
            raise AmiNaturalTurnPreparationError()
        if name == "pointer":
            role = attributes.get("role")
            href = attributes.get("href")
            if not role or not href or role in pointers:
                raise AmiNaturalTurnPreparationError()
            pointers[role] = href
        elif name == "child":
            href = attributes.get("href")
            if not href:
                raise AmiNaturalTurnPreparationError()
            children.append(href)
        else:
            raise AmiNaturalTurnPreparationError()
    return pointers, tuple(children)


def _ontology_ids(raw: bytes) -> set[str]:
    result: set[str] = set()
    for element in _xml_root(raw).iter():
        identity = _attributes(element).get("id")
        if identity:
            if identity in result:
                raise AmiNaturalTurnPreparationError()
            result.add(identity)
    return result


def _ontology_target(href: str, *, filename: str) -> str:
    prefix = f"{filename}#id("
    if not href.startswith(prefix) or not href.endswith(")"):
        raise AmiNaturalTurnPreparationError()
    value = href[len(prefix) : -1]
    if not value:
        raise AmiNaturalTurnPreparationError()
    return value


def _validate_segment(
    raw: bytes,
    *,
    expected: Mapping[str, object],
    source_ids: Mapping[str, tuple[str, ...]],
) -> None:
    elements = _id_elements(_xml_root(raw), element_name="segment")
    identity = str(expected["id"])
    try:
        element = elements[identity]
    except KeyError:
        raise AmiNaturalTurnPreparationError() from None
    attributes = _attributes(element)
    pointers, children = _pointer_map(element)
    if pointers or children != (expected["words_href"],):
        raise AmiNaturalTurnPreparationError()
    match = _HREF_RANGE_RE.fullmatch(str(expected["words_href"]))
    if match is None or match.group("speaker") != expected["speaker"]:
        raise AmiNaturalTurnPreparationError()
    positions = {
        source_id: index
        for index, source_id in enumerate(source_ids[match.group("speaker")])
    }
    try:
        range_start = positions[match.group("start")]
        range_end = positions[match.group("end")]
    except KeyError:
        raise AmiNaturalTurnPreparationError() from None
    if range_start > range_end:
        raise AmiNaturalTurnPreparationError()
    if (
        _sample(_decimal(attributes.get("transcriber_start")), end=False)
        != expected["start_sample"]
        or _sample(_decimal(attributes.get("transcriber_end")), end=True)
        != expected["end_sample"]
    ):
        raise AmiNaturalTurnPreparationError()


def _validate_adjacency(
    raw: bytes,
    *,
    expected: Mapping[str, object] | None,
    ap_type_ids: set[str],
) -> str | None:
    if expected is None:
        return None
    elements = _id_elements(_xml_root(raw), element_name="adjacency-pair")
    identity = str(expected["id"])
    try:
        element = elements[identity]
    except KeyError:
        raise AmiNaturalTurnPreparationError() from None
    pointers, children = _pointer_map(element)
    if children or pointers != {
        "type": expected["type_href"],
        "source": expected["source_href"],
        "target": expected["target_href"],
    }:
        raise AmiNaturalTurnPreparationError()
    if (
        _ontology_target(str(expected["type_href"]), filename="ap-types.xml")
        not in ap_type_ids
    ):
        raise AmiNaturalTurnPreparationError()
    for role in ("source", "target"):
        match = _HREF_SINGLE_RE.fullmatch(str(expected[f"{role}_href"]))
        if match is None:
            raise AmiNaturalTurnPreparationError()
    return identity


def _parse_locked_windows(
    members: Mapping[str, bytes],
    lock,
    *,
    audio_samples: int,
) -> tuple[
    tuple[_ParsedWindow, ...],
    Mapping[str, tuple[_WordNode, ...]],
    tuple[_Activity, ...],
]:
    words: dict[str, tuple[_WordNode, ...]] = {}
    source_ids: dict[str, tuple[str, ...]] = {}
    activities: list[_Activity] = []
    for speaker in "ABCD":
        parsed_words, parsed_activities, parsed_ids = _parse_words(
            members[f"words/ES2004a.{speaker}.words.xml"],
            speaker=speaker,
            audio_samples=audio_samples,
        )
        words[speaker] = parsed_words
        source_ids[speaker] = parsed_ids
        activities.extend(parsed_activities)
    da_types = _ontology_ids(members["ontologies/da-types.xml"])
    ap_types = _ontology_ids(members["ontologies/ap-types.xml"])
    da_elements = {
        speaker: _id_elements(
            _xml_root(members[f"dialogueActs/ES2004a.{speaker}.dialog-act.xml"]),
            element_name="dact",
        )
        for speaker in "ABCD"
    }
    selection = _mapping(
        lock.value["selection"],
        {"meeting_id", "algorithm", "selected_windows_sha256", "windows"},
    )
    result: list[_ParsedWindow] = []
    for raw_window in _sequence(selection["windows"], length=2):
        window = _mapping(raw_window, set(raw_window))
        start = int(window["start_sample"])
        end = int(window["end_sample"])
        selected_nodes: dict[str, _WordNode] = {}
        acts: list[Mapping[str, object]] = []
        for raw_act in _sequence(window["dialogue_acts"], length=2):
            expected = _mapping(raw_act, set(raw_act))
            speaker = str(expected["speaker"])
            try:
                element = da_elements[speaker][str(expected["id"])]
            except KeyError:
                raise AmiNaturalTurnPreparationError() from None
            pointers, children = _pointer_map(element)
            if pointers != {"da-aspect": expected["type_href"]} or children != (
                expected["words_href"],
            ):
                raise AmiNaturalTurnPreparationError()
            if (
                _ontology_target(str(expected["type_href"]), filename="da-types.xml")
                not in da_types
            ):
                raise AmiNaturalTurnPreparationError()
            nodes = _range_nodes(str(expected["words_href"]), words)
            rendered = _render_nodes(nodes)
            reference = _reference(rendered)
            if (
                hashlib.sha256(reference.encode("utf-8")).hexdigest()
                != expected["reference_sha256"]
                or min(item.start_sample for item in rendered)
                != expected["start_sample"]
                or max(item.end_sample for item in rendered) != expected["end_sample"]
            ):
                raise AmiNaturalTurnPreparationError()
            for node in nodes:
                selected_nodes[node.source_id] = node
            acts.append(
                {
                    "dialogue_act_id": expected["id"],
                    "speaker_id": speaker,
                    "type_href": expected["type_href"],
                    "source_start_sample": expected["start_sample"],
                    "source_end_sample": expected["end_sample"],
                    "reference": reference,
                    "reference_sha256": expected["reference_sha256"],
                }
            )
        timed_expected = {
            str(item["id"]): item
            for item in _sequence(window["timed_words"])
            if isinstance(item, dict)
        }
        if set(selected_nodes) != set(timed_expected):
            raise AmiNaturalTurnPreparationError()
        for identity, node in selected_nodes.items():
            expected = timed_expected[identity]
            if (
                node.speaker != expected["speaker"]
                or node.start_sample != expected["start_sample"]
                or node.end_sample != expected["end_sample"]
                or node.punctuation is not expected["punctuation"]
            ):
                raise AmiNaturalTurnPreparationError()

        selected_positive = {
            identity
            for identity, node in selected_nodes.items()
            if not node.punctuation
        }
        external = sum(
            1
            for speaker_nodes in words.values()
            for node in speaker_nodes
            if (
                not node.punctuation
                and node.source_id not in selected_positive
                and node.start_sample < end
                and node.end_sample > start
            )
        ) + sum(
            1
            for activity in activities
            if activity.start_sample < end and activity.end_sample > start
        )
        if external != window["external_positive_duration_annotations"]:
            raise AmiNaturalTurnPreparationError()

        rendered_all = sorted(
            (
                rendered
                for raw_act in _sequence(window["dialogue_acts"], length=2)
                for rendered in _render_nodes(
                    _range_nodes(str(raw_act["words_href"]), words)  # type: ignore[index]
                )
            ),
            key=lambda item: (
                item.start_sample,
                item.end_sample,
                item.speaker,
                item.source_id,
            ),
        )
        combined = _reference(rendered_all)
        if (
            hashlib.sha256(combined.encode("utf-8")).hexdigest()
            != window["linearized_reference_sha256"]
        ):
            raise AmiNaturalTurnPreparationError()
        for expected_segment in _sequence(window["segments"], length=2):
            speaker = str(expected_segment["speaker"])  # type: ignore[index]
            _validate_segment(
                members[f"segments/ES2004a.{speaker}.segments.xml"],
                expected=expected_segment,  # type: ignore[arg-type]
                source_ids=source_ids,
            )
        adjacency = _validate_adjacency(
            members["dialogueActs/ES2004a.adjacency-pairs.xml"],
            expected=window["adjacency_pair"],  # type: ignore[arg-type]
            ap_type_ids=ap_types,
        )
        relation = _mapping(window["relation"], {"kind", "start_sample", "end_sample"})
        result.append(
            _ParsedWindow(
                str(window["window_id"]),
                start,
                end,
                combined,
                str(window["linearized_reference_sha256"]),
                tuple(acts),
                tuple(rendered_all),
                adjacency,
                str(relation["kind"]),
                int(relation["start_sample"]),
                int(relation["end_sample"]),
            )
        )
    return tuple(result), words, tuple(activities)


def _merged(intervals: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    result: list[list[int]] = []
    for start, end in sorted(intervals):
        if start >= end:
            raise AmiNaturalTurnPreparationError()
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return tuple((start, end) for start, end in result)


def _verify_roomtone_selection(
    words: Mapping[str, tuple[_WordNode, ...]],
    activities: Sequence[_Activity],
    *,
    audio_samples: int,
    transform: Mapping[str, object],
) -> tuple[int, int]:
    occupied = _merged(
        tuple(
            (node.start_sample, node.end_sample)
            for nodes in words.values()
            for node in nodes
            if not node.punctuation
        )
        + tuple((item.start_sample, item.end_sample) for item in activities)
    )
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for start, end in occupied:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < audio_samples:
        gaps.append((cursor, audio_samples))
    if not gaps:
        raise AmiNaturalTurnPreparationError()
    gap_start, gap_end = min(gaps, key=lambda item: (-(item[1] - item[0]), item[0]))
    expected_start = int(transform["silence_window_start_sample"])
    expected_end = int(transform["silence_window_end_sample"])
    length = expected_end - expected_start
    if gap_end - gap_start < length:
        raise AmiNaturalTurnPreparationError()
    crop_start = gap_start + (gap_end - gap_start - length) // 2
    if (crop_start, crop_start + length) != (expected_start, expected_end):
        raise AmiNaturalTurnPreparationError()
    room_start = int(transform["roomtone_start_sample"])
    room_end = int(transform["roomtone_end_sample"])
    prefix = int(transform["prefix_samples"])
    centered = expected_start + (length - prefix) // 2
    if (room_start, room_end) != (centered, centered + prefix):
        raise AmiNaturalTurnPreparationError()
    return room_start, room_end


def _case_pcm(
    source: np.ndarray,
    *,
    window: _ParsedWindow,
    roomtone: tuple[int, int],
    output: Mapping[str, object],
    audio_pin: Mapping[str, object],
    production_evidence: bool,
) -> bytes:
    room_start, room_end = roomtone
    source_i16 = source[window.start_sample : window.end_sample]
    room_i16 = source[room_start:room_end]
    if (
        source_i16.size != window.end_sample - window.start_sample
        or room_i16.size != 32_000
        or hashlib.sha256(source_i16.tobytes()).hexdigest()
        != output["source_pcm16_sha256"]
        or hashlib.sha256(room_i16.tobytes()).hexdigest()
        != audio_pin["roomtone_pcm16_sha256"]
    ):
        raise AmiNaturalTurnPreparationError()
    source_f32 = source_i16.astype(np.float32) * np.float32(1.0 / 32768.0)
    room_f32 = room_i16.astype(np.float32) * np.float32(1.0 / 32768.0)
    if (
        hashlib.sha256(source_f32.tobytes()).hexdigest() != output["source_f32_sha256"]
        or hashlib.sha256(room_f32.tobytes()).hexdigest()
        != audio_pin["roomtone_f32_sha256"]
    ):
        raise AmiNaturalTurnPreparationError()
    result = np.empty(CASE_SAMPLES, dtype="<f4")
    result[:32_000] = room_f32
    source_end = 32_000 + source_f32.size
    result[32_000:source_end] = source_f32
    result[source_end:] = np.resize(room_f32, CASE_SAMPLES - source_end)
    raw = result.tobytes(order="C")
    if (
        len(raw) != output["size_bytes"]
        or not bool(np.all(np.isfinite(result)))
        or bool(np.any(np.abs(result) > 1.0))
        or hashlib.sha256(raw).hexdigest() != output["sha256"]
    ):
        raise AmiNaturalTurnPreparationError()
    del production_evidence
    return raw


def _case_payloads(
    *,
    lock,
    windows: Sequence[_ParsedWindow],
    pcm_by_channel: Mapping[str, np.ndarray],
    roomtone: tuple[int, int],
) -> tuple[_PreparedCase, ...]:
    window_by_id = {window.window_id: window for window in windows}
    sources = _mapping(lock.value["sources"], {"annotation_archive", "audio"})
    audio_by_channel = {
        item["channel"]: item
        for item in _sequence(sources["audio"], length=2)
        if isinstance(item, dict)
    }
    outputs = _sequence(
        _mapping(lock.value["outputs"], {"cases"})["cases"], length=CASE_COUNT
    )
    result: list[_PreparedCase] = []
    for index, raw_output in enumerate(outputs):
        output = _mapping(raw_output, set(raw_output))
        channel = str(output["channel"])
        window = window_by_id[str(output["window_id"])]
        pcm = _case_pcm(
            pcm_by_channel[channel],
            window=window,
            roomtone=roomtone,
            output=output,
            audio_pin=audio_by_channel[channel],
            production_evidence=lock.production_evidence,
        )
        offset = 32_000
        output_words = [
            _RenderedWord(
                word.source_id,
                word.speaker,
                word.text,
                offset + word.start_sample - window.start_sample,
                offset + word.end_sample - window.start_sample,
            )
            for word in window.words
        ]
        speech = [
            {"start_sample": start, "end_sample": end}
            for start, end in _merged(
                tuple((word.start_sample, word.end_sample) for word in output_words)
            )
        ]
        speaker_intervals = [
            {
                "speaker_id": speaker,
                "role": "unknown",
                "start_sample": start,
                "end_sample": end,
            }
            for speaker in sorted({word.speaker for word in output_words})
            for start, end in _merged(
                tuple(
                    (word.start_sample, word.end_sample)
                    for word in output_words
                    if word.speaker == speaker
                )
            )
        ]
        speaker_intervals.sort(
            key=lambda row: (row["start_sample"], row["end_sample"], row["speaker_id"])
        )
        tags = [
            "ami",
            "es2004a",
            "natural-turn",
            window.window_id,
            channel,
            "calibration-roomtone-2s",
            "postroll-roomtone-repeat",
            "diagnostic-only",
            "overlapping-speakers"
            if window.relation_kind == "nested-overlap"
            else "turn-transition",
        ]
        manifest: dict[str, object] = {
            "id": output["case_id"],
            "tracks": {
                "mic": {
                    "file": output["file"],
                    "sha256": output["sha256"],
                    "encoding": "f32le",
                    "samples": CASE_SAMPLES,
                }
            },
            "speech_intervals": speech,
            "speaker_intervals": speaker_intervals,
            "word_intervals": [
                {
                    "speaker_id": word.speaker,
                    "text": word.text,
                    "start_sample": word.start_sample,
                    "end_sample": word.end_sample,
                }
                for word in output_words
            ],
            "assertion": "transcript",
            "expected_text": window.reference,
            "commands": [],
            "tags": tags,
            "aec_delay_samples": 0,
        }
        relation = {
            "start_sample": offset + window.relation_start_sample - window.start_sample,
            "end_sample": offset + window.relation_end_sample - window.start_sample,
        }
        metadata: dict[str, object] = {
            "case_id": output["case_id"],
            "window_id": window.window_id,
            "channel": channel,
            "replay_case_index": index,
            "samples": CASE_SAMPLES,
            "frame_samples": FRAME_SAMPLES,
            "source_start_sample": window.start_sample,
            "source_end_sample": window.end_sample,
            "source_offset_sample": offset,
            "source_end_output_sample": offset
            + window.end_sample
            - window.start_sample,
            "reference": window.reference,
            "reference_sha256": window.reference_sha256,
            "dialogue_acts": [
                {
                    **act,
                    "output_start_sample": offset
                    + int(act["source_start_sample"])
                    - window.start_sample,
                    "output_end_sample": offset
                    + int(act["source_end_sample"])
                    - window.start_sample,
                }
                for act in window.dialogue_acts
            ],
            "adjacency_pair_id": window.adjacency_pair_id,
            "overlap_interval": (
                relation if window.relation_kind == "nested-overlap" else None
            ),
            "gap_interval": (
                relation if window.relation_kind == "adjacent-gap" else None
            ),
        }
        result.append(
            _PreparedCase(
                str(output["case_id"]),
                str(output["file"]),
                pcm,
                manifest,
                metadata,
            )
        )
    if [case.case_id for case in result] != _mapping(
        lock.value["layout"],
        {"window_count", "channel_count", "case_count", "artifact_count", "case_order"},
    )["case_order"]:
        raise AmiNaturalTurnPreparationError()
    return tuple(result)


def _has_git_ancestor(path: Path) -> bool:
    for ancestor in (path, *path.parents):
        try:
            ancestor_mode = ancestor.lstat().st_mode
        except OSError:
            raise AmiNaturalTurnPreparationError() from None
        if ancestor_mode & stat.S_ISVTX and ancestor_mode & stat.S_IWOTH:
            break
        try:
            metadata = (ancestor / ".git").lstat()
        except FileNotFoundError:
            continue
        except OSError:
            raise AmiNaturalTurnPreparationError() from None
        if stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode):
            return True
    return False


def _new_private_output(
    path: Path | str, *, sources: Sequence[_SourceSnapshot]
) -> tuple[Path, int, tuple[int, ...]]:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise AmiNaturalTurnPreparationError()
    candidate = Path(os.path.abspath(supplied))
    if (
        not candidate.name
        or candidate.name in {".", ".."}
        or _has_git_ancestor(candidate.parent)
        or any(
            candidate == source.path
            or candidate in source.path.parents
            or candidate == source.path.parent
            or source.path.parent in candidate.parents
            for source in sources
        )
    ):
        raise AmiNaturalTurnPreparationError()
    descriptor = -1
    try:
        with opened_directory_nofollow(candidate.parent, require_private=True) as (
            _stable,
            parent_fd,
        ):
            os.mkdir(candidate.name, mode=0o700, dir_fd=parent_fd)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(candidate, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o700
            or opened.st_uid != os.getuid()
            or candidate.resolve(strict=True) != candidate
            or _directory_locator(candidate.lstat()) != _directory_locator(opened)
        ):
            raise AmiNaturalTurnPreparationError()
        return candidate, descriptor, _directory_locator(opened)
    except AmiNaturalTurnPreparationError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except (BoundedReadError, FileExistsError, OSError, RuntimeError, ValueError):
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise AmiNaturalTurnPreparationError() from None


def _verify_output_directory(
    path: Path, descriptor: int, identity: tuple[int, ...]
) -> None:
    try:
        if (
            _directory_locator(os.fstat(descriptor)) != identity
            or _directory_locator(path.lstat()) != identity
            or path.resolve(strict=True) != path
        ):
            raise AmiNaturalTurnPreparationError()
    except AmiNaturalTurnPreparationError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise AmiNaturalTurnPreparationError() from None


def _write_leaf(descriptor: int, name: str, raw: bytes) -> tuple[int, ...]:
    if (
        not name
        or len(PurePosixPath(name).parts) != 1
        or PurePosixPath(name).name != name
        or not raw
    ):
        raise AmiNaturalTurnPreparationError()
    file_descriptor = -1
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        file_descriptor = os.open(name, flags, 0o600, dir_fd=descriptor)
        os.fchmod(file_descriptor, 0o600)
        view = memoryview(raw)
        while view:
            written = os.write(file_descriptor, view)
            if written <= 0:
                raise AmiNaturalTurnPreparationError()
            view = view[written:]
        os.fsync(file_descriptor)
        opened = os.fstat(file_descriptor)
        current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if (
            _file_identity(opened) != _file_identity(current)
            or not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or opened.st_size != len(raw)
        ):
            raise AmiNaturalTurnPreparationError()
        return _file_identity(opened)
    except AmiNaturalTurnPreparationError:
        raise
    except (OSError, ValueError):
        raise AmiNaturalTurnPreparationError() from None
    finally:
        if file_descriptor >= 0:
            try:
                os.close(file_descriptor)
            except OSError:
                pass


def _exact_entries(descriptor: int, expected: set[str]) -> None:
    try:
        if set(os.listdir(descriptor)) != expected:
            raise AmiNaturalTurnPreparationError()
    except AmiNaturalTurnPreparationError:
        raise
    except OSError:
        raise AmiNaturalTurnPreparationError() from None


def _terminal_commit(
    directory_path: Path,
    directory_fd: int,
    raw: bytes,
    state: _CommitState,
    prepared: PreparedAmiNaturalTurnFixture,
    *,
    commit_guard: Callable[[], None],
) -> None:
    if (
        state.committed
        or state.pending_prepared is not None
        or type(prepared) is not PreparedAmiNaturalTurnFixture
        or not callable(commit_guard)
        or not hasattr(os, "O_TMPFILE")
    ):
        raise AmiNaturalTurnPreparationError()
    staged_fd = -1
    previous_mask = None
    try:
        flags = os.O_RDWR | os.O_TMPFILE | getattr(os, "O_CLOEXEC", 0)
        staged_fd = os.open(".", flags, 0o600, dir_fd=directory_fd)
        os.fchmod(staged_fd, 0o600)
        view = memoryview(raw)
        while view:
            written = os.write(staged_fd, view)
            if written <= 0:
                raise AmiNaturalTurnPreparationError()
            view = view[written:]
        os.fsync(staged_fd)
        staged = os.fstat(staged_fd)
        staged_identity = _file_identity(staged)
        if (
            not stat.S_ISREG(staged.st_mode)
            or stat.S_IMODE(staged.st_mode) != 0o600
            or staged.st_uid != os.getuid()
            or staged.st_nlink != 0
            or staged.st_size != len(raw)
        ):
            raise AmiNaturalTurnPreparationError()
        directory_identity = _file_identity(os.fstat(directory_fd))
        commit_guard()
        os.lseek(staged_fd, 0, os.SEEK_SET)
        retained = bytearray()
        while len(retained) <= len(raw):
            chunk = os.read(staged_fd, min(65_536, len(raw) + 1 - len(retained)))
            if not chunk:
                break
            retained.extend(chunk)
        current = os.fstat(staged_fd)
        if (
            bytes(retained) != raw
            or hashlib.sha256(retained).digest() != hashlib.sha256(raw).digest()
            or _file_identity(current) != staged_identity
            or current.st_nlink != 0
            or _file_identity(os.fstat(directory_fd)) != directory_identity
            or _file_identity(directory_path.lstat()) != directory_identity
        ):
            raise AmiNaturalTurnPreparationError()
        os.fsync(directory_fd)
        state.pending_prepared = prepared
        if hasattr(signal, "pthread_sigmask"):
            signals = tuple(
                item
                for item in (
                    getattr(signal, "SIGINT", None),
                    getattr(signal, "SIGHUP", None),
                    getattr(signal, "SIGTERM", None),
                )
                if isinstance(item, int)
            )
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
        link_returned = False
        try:
            try:
                os.link(
                    f"/proc/self/fd/{staged_fd}",
                    PREPARATION_RECEIPT_FILENAME,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=True,
                )
                link_returned = True
            finally:
                if link_returned:
                    state.committed = True
                else:
                    try:
                        published = os.stat(
                            PREPARATION_RECEIPT_FILENAME,
                            dir_fd=directory_fd,
                            follow_symlinks=False,
                        )
                        current = os.fstat(staged_fd)
                    except OSError:
                        pass
                    else:
                        if (
                            (published.st_dev, published.st_ino)
                            == (current.st_dev, current.st_ino)
                            and stat.S_ISREG(published.st_mode)
                            and stat.S_IMODE(published.st_mode) == 0o600
                            and published.st_uid == os.getuid()
                            and published.st_nlink == 1
                            and published.st_size == len(raw)
                            and current.st_nlink == 1
                        ):
                            state.committed = True
        except BaseException:
            if not state.committed:
                raise
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
    except AmiNaturalTurnPreparationError:
        raise
    except Exception:
        if state.committed:
            return
        raise AmiNaturalTurnPreparationError() from None
    finally:
        if previous_mask is not None:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            except BaseException:
                if not state.committed:
                    raise
        if staged_fd >= 0:
            try:
                os.close(staged_fd)
            except OSError:
                pass


def _terminal_guard() -> None:
    return None


def _verified_written(path: Path, raw: bytes) -> None:
    try:
        snapshot = read_regular_bounded(
            path,
            maximum_bytes=max(len(raw), 1),
            expected_bytes=len(raw),
        )
        metadata = path.lstat()
    except (BoundedReadError, OSError):
        raise AmiNaturalTurnPreparationError() from None
    if (
        snapshot.data != raw
        or hashlib.sha256(snapshot.data).digest() != hashlib.sha256(raw).digest()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
    ):
        raise AmiNaturalTurnPreparationError()


def prepare_ami_natural_turn_capture_replay(
    *,
    annotations_zip: Path | str,
    close_wav: Path | str,
    far_wav: Path | str,
    output_dir: Path | str,
    accepted_terms: Sequence[str],
    lock_path: Path | str = DEFAULT_LOCK,
    _test_injection: AmiNaturalTurnTestInjection | None = None,
    _commit_state: _CommitState | None = None,
    _commit_guard: Callable[[], None] = _terminal_guard,
) -> PreparedAmiNaturalTurnFixture:
    """Verify exact inputs and publish one terminal-receipt private bundle."""

    try:
        terms = tuple(accepted_terms)
    except (TypeError, ValueError):
        raise AmiNaturalTurnPreparationError() from None
    if (
        isinstance(accepted_terms, (str, bytes))
        or len(terms) != 1
        or set(terms) != {"CC-BY-4.0"}
    ):
        raise AmiNaturalTurnPreparationError()
    try:
        lock = _load_fixture_lock(lock_path, test_injection=_test_injection)
        sources_value = _mapping(lock.value["sources"], {"annotation_archive", "audio"})
        annotation_pin = _mapping(
            sources_value["annotation_archive"],
            set(sources_value["annotation_archive"]),
        )
        audio_pins = {
            item["channel"]: item
            for item in _sequence(sources_value["audio"], length=2)
            if isinstance(item, dict)
        }
        annotations = _read_private_source(
            annotations_zip,
            expected_sha256=str(annotation_pin["sha256"]),
            expected_bytes=int(annotation_pin["size_bytes"]),
            maximum_bytes=_ANNOTATION_MAX_BYTES,
        )
        close = _read_private_source(
            close_wav,
            expected_sha256=str(audio_pins["close"]["sha256"]),
            expected_bytes=int(audio_pins["close"]["size_bytes"]),
            maximum_bytes=_AUDIO_MAX_BYTES,
        )
        far = _read_private_source(
            far_wav,
            expected_sha256=str(audio_pins["far"]["sha256"]),
            expected_bytes=int(audio_pins["far"]["size_bytes"]),
            maximum_bytes=_AUDIO_MAX_BYTES,
        )
        if len({item.path for item in (annotations, close, far)}) != 3:
            raise AmiNaturalTurnPreparationError()
        close_pcm = _riff_pcm16_mono(close.raw)
        far_pcm = _riff_pcm16_mono(far.raw)
        if (
            close_pcm.size != far_pcm.size
            or close_pcm.size != audio_pins["close"]["samples"]
            or far_pcm.size != audio_pins["far"]["samples"]
        ):
            raise AmiNaturalTurnPreparationError()
        members = _archive_members(annotations.raw, lock)
        windows, words, activities = _parse_locked_windows(
            members,
            lock,
            audio_samples=int(close_pcm.size),
        )
        transform = _mapping(lock.value["transform"], set(lock.value["transform"]))
        roomtone = _verify_roomtone_selection(
            words,
            activities,
            audio_samples=int(close_pcm.size),
            transform=transform,
        )
        cases = _case_payloads(
            lock=lock,
            windows=windows,
            pcm_by_channel={"close": close_pcm, "far": far_pcm},
            roomtone=roomtone,
        )
        source_contract = _source_contract_sha256(lock)
        closure = _preparer_closure_sha256()
        purpose = (
            f"AMI ES2004a natural-turn capture replay v1;fixture_id={lock.fixture_id};"
            f"lock_recipe_sha256={lock.recipe_sha256};diagnostic_only=true"
        )
        manifest_value: dict[str, object] = {
            "schema_version": 1,
            "purpose": purpose,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "frame_samples": FRAME_SAMPLES,
            "cases": [case.manifest for case in cases],
        }
        manifest_raw = _canonical_json(manifest_value, newline=True)

        parsed_cases = tuple(
            _parse_metadata_case(case.metadata, index=index)
            for index, case in enumerate(cases)
        )
        _validate_cases_against_lock(lock, parsed_cases)
        labels = _canonical_sha256(_label_rows(parsed_cases))
        metadata_value: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": METADATA_KIND,
            "fixture_id": lock.fixture_id,
            "production_evidence": lock.production_evidence,
            "lock_sha256": lock.raw_sha256,
            "lock_recipe_sha256": lock.recipe_sha256,
            "source_contract_sha256": source_contract,
            "labels_sha256": labels,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "frame_samples": FRAME_SAMPLES,
            "case_samples": CASE_SAMPLES,
            "cases": [case.metadata for case in cases],
        }
        metadata_raw = _canonical_json(metadata_value, newline=True)
        artifacts = [
            {
                "case_id": case.case_id,
                "file": case.filename,
                "size_bytes": len(case.pcm),
                "sha256": hashlib.sha256(case.pcm).hexdigest(),
            }
            for case in cases
        ]
        receipt_value: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": RECEIPT_KIND,
            "fixture_id": lock.fixture_id,
            "production_evidence": lock.production_evidence,
            "accepted_terms": ["CC-BY-4.0"],
            "lock_sha256": lock.raw_sha256,
            "lock_recipe_sha256": lock.recipe_sha256,
            "source_contract_sha256": source_contract,
            "preparer_closure_sha256": closure,
            "labels_sha256": labels,
            "manifest": {
                "file": MANIFEST_FILENAME,
                "size_bytes": len(manifest_raw),
                "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            },
            "metadata": {
                "file": METADATA_FILENAME,
                "size_bytes": len(metadata_raw),
                "sha256": hashlib.sha256(metadata_raw).hexdigest(),
            },
            "artifacts": artifacts,
            "case_count": CASE_COUNT,
            "case_samples": CASE_SAMPLES,
            "frame_samples": FRAME_SAMPLES,
            "evidence_scope": lock.value["evidence_scope"],
        }
        receipt_value["binding_sha256"] = _receipt_binding(receipt_value)
        receipt_raw = _canonical_json(receipt_value, newline=True)
        if any(
            value.encode("utf-8") in receipt_raw
            for window in windows
            for value in (
                window.reference,
                *(str(act["reference"]) for act in window.dialogue_acts),
            )
        ):
            raise AmiNaturalTurnPreparationError()

        for source, maximum in (
            (annotations, _ANNOTATION_MAX_BYTES),
            (close, _AUDIO_MAX_BYTES),
            (far, _AUDIO_MAX_BYTES),
        ):
            _verify_source(source, maximum_bytes=maximum)
        if (
            _load_fixture_lock(lock_path, test_injection=_test_injection) != lock
            or _preparer_closure_sha256() != closure
        ):
            raise AmiNaturalTurnPreparationError()
    except AmiNaturalTurnPreparationError:
        raise
    except AmiNaturalTurnFixtureError:
        raise AmiNaturalTurnPreparationError() from None
    except (KeyError, MemoryError, OSError, RuntimeError, TypeError, ValueError):
        raise AmiNaturalTurnPreparationError() from None

    state = _CommitState() if _commit_state is None else _commit_state
    if type(state) is not _CommitState or state.committed:
        raise AmiNaturalTurnPreparationError()
    destination, output_fd, _output_identity = _new_private_output(
        output_dir,
        sources=(annotations, close, far),
    )
    prepared_result = PreparedAmiNaturalTurnFixture(
        root=destination,
        fixture_id=lock.fixture_id,
        production_evidence=lock.production_evidence,
        lock_sha256=lock.raw_sha256,
        lock_recipe_sha256=lock.recipe_sha256,
        labels_sha256=labels,
        metadata_sha256=hashlib.sha256(metadata_raw).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
        source_contract_sha256=source_contract,
        preparer_closure_sha256=closure,
        case_count=CASE_COUNT,
        case_samples=CASE_SAMPLES,
    )
    try:
        for case in cases:
            _write_leaf(output_fd, case.filename, case.pcm)
        _write_leaf(output_fd, MANIFEST_FILENAME, manifest_raw)
        _write_leaf(output_fd, METADATA_FILENAME, metadata_raw)
        non_receipt = {
            *(case.filename for case in cases),
            MANIFEST_FILENAME,
            METADATA_FILENAME,
        }

        def verify_precommit_bundle() -> None:
            _exact_entries(output_fd, non_receipt)
            for prepared_case in cases:
                _verified_written(
                    destination / prepared_case.filename, prepared_case.pcm
                )
            _verified_written(destination / MANIFEST_FILENAME, manifest_raw)
            _verified_written(destination / METADATA_FILENAME, metadata_raw)
            _verify_output_directory(
                destination,
                output_fd,
                _directory_locator(os.fstat(output_fd)),
            )
            try:
                replay = load_corpus(destination / MANIFEST_FILENAME)
                verify_corpus_snapshot(replay)
                _validate_replay_semantics(lock, replay, parsed_cases)
            except (AmiNaturalTurnFixtureError, ReplayCorpusError):
                raise AmiNaturalTurnPreparationError() from None
            if (
                replay.digest != prepared_result.manifest_sha256
                or [item.case_id for item in replay.cases]
                != [item.case_id for item in parsed_cases]
                or any(
                    replay.cases[index].mic.sha256
                    != hashlib.sha256(cases[index].pcm).hexdigest()
                    for index in range(CASE_COUNT)
                )
            ):
                raise AmiNaturalTurnPreparationError()
            for guarded_source, guarded_maximum in (
                (annotations, _ANNOTATION_MAX_BYTES),
                (close, _AUDIO_MAX_BYTES),
                (far, _AUDIO_MAX_BYTES),
            ):
                _verify_source(guarded_source, maximum_bytes=guarded_maximum)
            if (
                _load_fixture_lock(lock_path, test_injection=_test_injection) != lock
                or _preparer_closure_sha256() != closure
            ):
                raise AmiNaturalTurnPreparationError()

        verify_precommit_bundle()

        def close_guard() -> None:
            _commit_guard()
            verify_precommit_bundle()

        _terminal_commit(
            destination,
            output_fd,
            receipt_raw,
            state,
            prepared_result,
            commit_guard=close_guard,
        )
        return prepared_result
    finally:
        try:
            os.close(output_fd)
        except OSError:
            pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise AmiNaturalTurnPreparationError()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description=(
            "Prepare the exact private AMI ES2004a natural-turn capture replay; "
            "no download, model, network, GPU, audio device, or live execution."
        )
    )
    parser.add_argument("--annotations-zip", type=Path, required=True)
    parser.add_argument("--close-wav", type=Path, required=True)
    parser.add_argument("--far-wav", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accept-license", action="append", default=[])
    return parser


def _lifecycle_handler(signum: int, _frame: object) -> None:
    raise _LifecycleSignal(signum)


def _success_result(
    prepared: PreparedAmiNaturalTurnFixture,
) -> Mapping[str, object]:
    return {
        "ok": True,
        "fixture_id": prepared.fixture_id,
        "production_evidence": prepared.production_evidence,
        "cases": prepared.case_count,
        "case_samples": prepared.case_samples,
        "manifest_sha256": prepared.manifest_sha256,
        "metadata_sha256": prepared.metadata_sha256,
        "receipt_sha256": prepared.receipt_sha256,
    }


def _recover_main_result(
    error: BaseException,
    state: _CommitState,
) -> tuple[Mapping[str, object], int]:
    if state.committed:
        prepared = state.pending_prepared
        if type(prepared) is PreparedAmiNaturalTurnFixture:
            return _success_result(prepared), 0
        return _SAFE_ERROR, 0
    if isinstance(error, KeyboardInterrupt):
        return _SAFE_ERROR, 130
    if isinstance(error, _LifecycleSignal):
        return _SAFE_ERROR, 128 + error.signum
    if isinstance(error, Exception):
        return _SAFE_ERROR, 2
    raise error


def main(argv: Sequence[str] | None = None) -> int:
    state = _CommitState()
    previous_handlers: dict[int, object] = {}
    result: Mapping[str, object] = _SAFE_ERROR
    code = 2
    try:
        try:
            for signum in (
                getattr(signal, "SIGHUP", None),
                getattr(signal, "SIGTERM", None),
            ):
                if isinstance(signum, int):
                    previous_handlers[signum] = signal.signal(
                        signum,
                        _lifecycle_handler,
                    )
            args = _parser().parse_args(argv)
            prepared = prepare_ami_natural_turn_capture_replay(
                annotations_zip=args.annotations_zip,
                close_wav=args.close_wav,
                far_wav=args.far_wav,
                output_dir=args.output_dir,
                accepted_terms=args.accept_license,
                _commit_state=state,
            )
            result = _success_result(prepared)
            code = 0
        except BaseException as error:
            result, code = _recover_main_result(error, state)
        try:
            print(_canonical_json(result).decode("utf-8"))
        except BaseException as error:
            result, code = _recover_main_result(error, state)
    finally:
        for signum, previous in reversed(tuple(previous_handlers.items())):
            try:
                signal.signal(signum, previous)
            except BaseException as error:
                result, code = _recover_main_result(error, state)
    return 0 if state.committed else code


if __name__ == "__main__":
    raise SystemExit(main())
