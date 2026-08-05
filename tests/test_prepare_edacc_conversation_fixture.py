from __future__ import annotations

from array import array
from collections import Counter
import csv
from dataclasses import replace
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tarfile
import wave

import pytest

from tools import prepare_edacc_conversation_fixture as prepare
from tools.streaming_stt.corpus import load_corpus


_ACCENT_FIELD = (
    "How would you describe your accent in English? (e.g. Italian, Glaswegian)"
)
_PARTICIPANT_FIELD = "PARTICIPANT_ID"
_OBSERVED_EDACC_UID_FIELD = b"\x80\x00\x00\x00\x00\x22\x89\x10"
_OBSERVED_EDACC_UID = 0x228910
_OWNER_BASE256_MIN = 8**7
_OWNER_MAX = (1 << 31) - 1
_TAR_NUMERIC_FIELDS = {
    "mode": slice(100, 108),
    "uid": slice(108, 116),
    "gid": slice(116, 124),
    "size": slice(124, 136),
    "mtime": slice(136, 148),
    "checksum": slice(148, 156),
    "devmajor": slice(329, 337),
    "devminor": slice(337, 345),
}

_TEST_ALIASED_RECORDING = "EDACC-C23_P1"
_TEST_ALIASED_CONVERSATION = "EDACC-C23_P7"
_DEV_CONVERSATION_FAMILY = "EDACC-C24"
_DEV_RECORDINGS = ("EDACC-C24_P1", "EDACC-C24_P2")


def _test_recording(index: int) -> str:
    return _TEST_ALIASED_RECORDING if index == 23 else f"EDACC-C{index:02d}"


def _write_wav(path: Path, index: int) -> None:
    frames = prepare.SOURCE_SAMPLE_RATE_HZ * 3 // 2
    sample = 512 + index * 32
    values = array("h", [sample]) * frames
    if sys.byteorder != "little":
        values.byteswap()
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(prepare.SOURCE_SAMPLE_RATE_HZ)
        audio.writeframes(values.tobytes())


def _source_tree(tmp_path: Path) -> Path:
    root = tmp_path / prepare.ARCHIVE_ROOT
    test = root / "test"
    dev = root / "dev"
    data = root / "data"
    test.mkdir(parents=True)
    dev.mkdir()
    data.mkdir()

    segments: list[str] = []
    utt2spk: list[str] = []
    text: list[str] = []
    stm: list[str] = []
    filtered_stm: list[str] = []
    conversations: list[str] = []
    accent_rows: list[dict[str, str]] = []

    def add_row(
        utterance: str,
        recording: str,
        speaker: str,
        start: str,
        end: str,
        reference: str,
        *,
        stm_reference: str | None = None,
        filtered_reference: str | None = None,
    ) -> None:
        segments.append(f"{utterance} {recording} {start} {end}\n")
        utt2spk.append(f"{utterance} {speaker}\n")
        text.append(f"{utterance} {reference}\n")
        rendered = reference if stm_reference is None else stm_reference
        stm.append(f"{recording} 1 {speaker} {start} {end} <female,l1> {rendered}\n")
        filtered = rendered if filtered_reference is None else filtered_reference
        suffix = f" {filtered}" if filtered else ""
        filtered_stm.append(
            f"{recording} 1 {speaker} {start} {end} <female,l1>{suffix}\n"
        )

    for index in range(24):
        recording = _test_recording(index)
        speaker = f"EDACC-C{index:02d}-A"
        conversation = (
            _TEST_ALIASED_CONVERSATION if index == 23 else recording
        )
        conversations.append(f"{conversation}\n")
        _write_wav(data / f"{recording}.wav", index)
        accent_rows.append(
            {
                _PARTICIPANT_FIELD: f"EDAC-C{index:02d}P1",
                _ACCENT_FIELD: f"Synthetic accent {index % 12}",
            }
        )
        if index < 8:
            reference = f"SYNTHETIC CLEAN PHRASE NUMBER {index}"
        elif index < 16:
            reference = f"UM SYNTHETIC PHRASE NUMBER {index}"
        else:
            reference = f"SYNTHETIC NEAR OVERLAP NUMBER {index}"
        add_row(
            f"utt-{index:02d}-selected",
            recording,
            speaker,
            "0.00",
            "0.50",
            reference,
            filtered_reference=(
                f"SYNTHETIC PHRASE NUMBER {index}" if index == 8 else None
            ),
        )
        if index >= 16:
            other = f"EDACC-C{index:02d}-B"
            accent_rows.append(
                {
                    _PARTICIPANT_FIELD: f"EDAC-C{index:02d}P2",
                    _ACCENT_FIELD: f"Synthetic other accent {index}",
                }
            )
            add_row(
                f"utt-{index:02d}-overlap-a",
                recording,
                speaker,
                "0.70",
                "1.20",
                "ACTUAL CROSS TALK FIRST",
            )
            add_row(
                f"utt-{index:02d}-overlap-b",
                recording,
                other,
                "0.80",
                "1.30",
                "ACTUAL CROSS TALK SECOND",
            )

    # These source rows exercise every hard-WER exclusion without changing the
    # exact 24-row eligible set.
    add_row(
        "utt-excluded-noise",
        "EDACC-C00",
        "EDACC-C00-A",
        "0.60",
        "0.65",
        "<laugh>",
    )
    add_row(
        "utt-excluded-alternative",
        "EDACC-C01",
        "EDACC-C01-A",
        "0.60",
        "0.65",
        "(HELLO)/(HI)",
    )
    add_row(
        "utt-excluded-ignore",
        "EDACC-C02",
        "EDACC-C02-A",
        "0.60",
        "0.65",
        "CONTROL SENTENCE WORDS",
        stm_reference="IGNORE_TIME_SEGMENT_IN_SCORING",
    )
    add_row(
        "utt-excluded-filtered-empty",
        "EDACC-C03",
        "EDACC-C03-A",
        "0.60",
        "1.10",
        "SPOKEN HESITATION WORDS",
        filtered_reference="",
    )

    (test / "conv.list").write_text("".join(conversations), encoding="utf-8")
    (test / "company.ctm").write_text(
        "EDACC-C00 1 0.00 0.50 SYNTHETIC\n",
        encoding="utf-8",
    )
    (test / "segments").write_text("".join(segments), encoding="utf-8")
    (test / "utt2spk").write_text("".join(utt2spk), encoding="utf-8")
    (test / "text").write_text("".join(text), encoding="utf-8")
    (test / "stm").write_text("".join(stm), encoding="utf-8")
    (test / "stm.filt").write_text("".join(filtered_stm), encoding="utf-8")

    dev_rows = tuple(
        (
            recording,
            f"dev-utt-{index}",
            f"{_DEV_CONVERSATION_FAMILY}-{'A' if index == 1 else 'B'}",
        )
        for index, recording in enumerate(_DEV_RECORDINGS, start=1)
    )
    for index, (recording, _utterance, _speaker) in enumerate(
        dev_rows,
        start=24,
    ):
        _write_wav(data / f"{recording}.wav", index)
    (dev / "company.ctm").write_text(
        "".join(
            f"{recording} 1 0.00 0.50 DEVELOPMENT\n"
            for recording, _utterance, _speaker in dev_rows
        ),
        encoding="utf-8",
    )
    (dev / "conv.list").write_text(
        f"{_DEV_CONVERSATION_FAMILY}\n",
        encoding="utf-8",
    )
    (dev / "segments").write_text(
        "".join(
            f"{utterance} {recording} 0.00 0.50\n"
            for recording, utterance, _speaker in dev_rows
        ),
        encoding="utf-8",
    )
    (dev / "stm").write_text(
        "".join(
            f"{recording} 1 {speaker} 0.00 0.50 "
            "<female,l1> DEVELOPMENT WORDS\n"
            for recording, _utterance, speaker in dev_rows
        ),
        encoding="utf-8",
    )
    (dev / "stm.filt").write_text(
        "".join(
            f"{recording} 1 {speaker} 0.00 0.50 "
            "<female,l1> DEVELOPMENT WORDS\n"
            for recording, _utterance, speaker in dev_rows
        ),
        encoding="utf-8",
    )
    (dev / "text").write_text(
        "".join(
            f"{utterance} DEVELOPMENT WORDS\n"
            for _recording, utterance, _speaker in dev_rows
        ),
        encoding="utf-8",
    )
    (dev / "utt2spk").write_text(
        "".join(
            f"{utterance} {speaker}\n"
            for _recording, utterance, speaker in dev_rows
        ),
        encoding="utf-8",
    )
    (root / "evaluate.sh").write_text(
        "#!/bin/sh\nexit 0\n",
        encoding="utf-8",
    )
    (root / "glm").write_text(";; synthetic GLM\n", encoding="utf-8")
    (root / "README.txt").write_text(
        "Synthetic EdAcc publisher-layout fixture.\n",
        encoding="utf-8",
    )
    with (root / "linguistic_background.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(_PARTICIPANT_FIELD, _ACCENT_FIELD),
        )
        writer.writeheader()
        writer.writerows(accent_rows)
    root.chmod(0o700)
    return root


def _archive(tmp_path: Path, root: Path, *, name: str = "source.tar.gz") -> Path:
    archive = tmp_path / name
    with tarfile.open(
        archive,
        "w:gz",
        format=tarfile.USTAR_FORMAT,
    ) as package:
        package.add(root, arcname=prepare.ARCHIVE_ROOT)
    archive.chmod(0o600)
    return archive


def _injection(
    archive: Path, fixture_id: str = "edacc-unit-v1"
) -> prepare.TestSourceInjection:
    payload = archive.read_bytes()
    return prepare.TestSourceInjection(
        archive_size_bytes=len(payload),
        archive_md5=hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        fixture_id=fixture_id,
    )


def _mutate_first_byte(path: Path) -> None:
    with path.open("r+b", buffering=0) as handle:
        value = handle.read(1)
        assert value
        handle.seek(0)
        handle.write(bytes((value[0] ^ 1,)))


def _prepare(tmp_path: Path, *, output_name: str = "prepared"):
    root = _source_tree(tmp_path)
    archive = _archive(tmp_path, root)
    result = prepare.prepare_edacc_conversation_fixture(
        source_root=root,
        archive_path=archive,
        output_dir=tmp_path / output_name,
        accepted_terms=prepare.REQUIRED_TERMS,
        test_source_injection=_injection(archive),
    )
    return result, root, archive


def _accent_csv(participant_field: str) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow((participant_field, _ACCENT_FIELD))
    writer.writerow(("EDAC-C07P1", "Synthetic Romanian accent"))
    return output.getvalue().encode("utf-8")


def test_exact_uppercase_participant_header_preserves_accent_lookup():
    direct, keyed = prepare._parse_accents(_accent_csv(_PARTICIPANT_FIELD))

    assert direct == {"EDAC-C07P1": "Synthetic Romanian accent"}
    assert keyed == {(7, "A"): "Synthetic Romanian accent"}
    assert (
        prepare._accent_for_speaker("EDACC-C07-A", direct, keyed)
        == "Synthetic Romanian accent"
    )


@pytest.mark.parametrize(
    "participant_field",
    (
        "Participant_ID",
        " PARTICIPANT_ID",
        "PARTICIPANT_ID ",
        "PARTICIPANT_\u0406D",
    ),
    ids=(
        "case-mismatch",
        "leading-whitespace",
        "trailing-whitespace",
        "cyrillic-i-lookalike",
    ),
)
def test_participant_header_near_misses_fail_closed(participant_field):
    with pytest.raises(prepare.EdaccPreparationError) as failure:
        prepare._parse_accents(_accent_csv(participant_field))

    assert failure.value.args == ()


def _tree_snapshot(root: Path) -> tuple[frozenset[Path], dict[Path, bytes]]:
    directories: set[Path] = set()
    files: dict[Path, bytes] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_dir():
            directories.add(relative)
        else:
            files[relative] = path.read_bytes()
    return frozenset(directories), files


@pytest.fixture(scope="module")
def valid_edacc_archive_bytes(tmp_path_factory) -> bytes:
    fixture_root = tmp_path_factory.mktemp("edacc-raw-archive")
    source = _source_tree(fixture_root / "source")
    return _archive(fixture_root, source).read_bytes()


def _raw_tar_members(payload: bytes | bytearray) -> tuple[tuple[str, int, int], ...]:
    members: list[tuple[str, int, int]] = []
    offset = 0
    while offset + 512 <= len(payload):
        header = payload[offset : offset + 512]
        if header == b"\0" * 512:
            return tuple(members)
        name = bytes(header[:100]).split(b"\0", 1)[0].decode("ascii")
        size_field = bytes(header[124:136]).strip(b"\0 ") or b"0"
        size = int(size_field, 8)
        end = offset + 512 + ((size + 511) // 512) * 512
        assert end <= len(payload)
        members.append((name, offset, end))
        offset = end
    raise AssertionError("synthetic tar has no terminator")


def _raw_tar_member(
    payload: bytes | bytearray,
    name: str,
) -> tuple[int, int]:
    for member_name, start, end in _raw_tar_members(payload):
        if member_name == name:
            return start, end
    raise AssertionError(f"missing synthetic member: {name}")


def _raw_tar_terminator(payload: bytes | bytearray) -> int:
    members = _raw_tar_members(payload)
    assert members
    return members[-1][2]


def _rewrite_raw_tar_header(
    payload: bytearray,
    offset: int,
    *,
    name: str | None = None,
    member_type: bytes | None = None,
    size: int | None = None,
    checksum: bool = True,
) -> None:
    header = bytearray(payload[offset : offset + 512])
    assert len(header) == 512
    if name is not None:
        encoded = name.encode("ascii")
        assert len(encoded) <= 100
        header[:100] = encoded.ljust(100, b"\0")
    if member_type is not None:
        assert len(member_type) == 1
        header[156:157] = member_type
    if size is not None:
        header[124:136] = f"{size:011o}\0".encode("ascii")
    if checksum:
        header[148:156] = b" " * 8
        header[148:156] = f"{sum(header):06o}\0 ".encode("ascii")
    payload[offset : offset + 512] = header


def _positive_base256(value: int, *, length: int = 8) -> bytes:
    assert 0 <= value < 256 ** (length - 1)
    return b"\x80" + value.to_bytes(length - 1, "big")


def _rewrite_raw_tar_numeric_field(
    payload: bytearray,
    offset: int,
    *,
    field: str,
    encoded: bytes,
    checksum: bool = True,
) -> None:
    location = _TAR_NUMERIC_FIELDS[field]
    assert len(encoded) == location.stop - location.start
    payload[offset + location.start : offset + location.stop] = encoded
    if checksum:
        _rewrite_raw_tar_header(payload, offset)


def _append_raw_tar_member(
    payload: bytearray,
    *,
    source_name: str,
    target_name: str,
) -> None:
    start, end = _raw_tar_member(payload, source_name)
    member = bytearray(payload[start:end])
    _rewrite_raw_tar_header(member, 0, name=target_name)
    terminator = _raw_tar_terminator(payload)
    payload[terminator:terminator] = member


def _write_raw_tar_archive(
    tmp_path: Path,
    payload: bytes | bytearray,
    *,
    name: str = "mutated.tar.gz",
) -> Path:
    archive = tmp_path / name
    archive.write_bytes(gzip.compress(bytes(payload), compresslevel=1, mtime=0))
    archive.chmod(0o600)
    return archive


def _assert_archive_scanner_rejects(tmp_path: Path, archive: Path) -> Path:
    extraction_parent = tmp_path / "private-extraction"
    extraction_parent.mkdir(mode=0o700)
    source_root = extraction_parent / prepare.ARCHIVE_ROOT
    canary = extraction_parent / "escape"
    canary.write_bytes(b"unchanged escape canary")
    canary.chmod(0o600)
    output = tmp_path / "must-not-exist"

    with pytest.raises(prepare.EdaccPreparationError) as failure:
        prepare.prepare_edacc_conversation_fixture(
            source_root=source_root,
            archive_path=archive,
            output_dir=output,
            accepted_terms=prepare.REQUIRED_TERMS,
            test_source_injection=_injection(archive),
        )

    assert failure.value.args == ()
    assert source_root.is_dir()
    assert source_root.stat().st_mode & 0o777 == 0o700
    assert canary.read_bytes() == b"unchanged escape canary"
    assert canary.stat().st_mode & 0o777 == 0o600
    assert not output.exists()
    return source_root


def _assert_complete_layout_rejects(tmp_path: Path, root: Path) -> None:
    archive = _archive(tmp_path, root)
    output = tmp_path / "must-not-exist"

    with pytest.raises(prepare.EdaccPreparationError) as failure:
        prepare.prepare_edacc_conversation_fixture(
            source_root=root,
            archive_path=archive,
            output_dir=output,
            accepted_terms=prepare.REQUIRED_TERMS,
            test_source_injection=_injection(archive),
        )

    assert failure.value.args == ()
    assert not output.exists()


def test_absent_source_root_is_extracted_with_exact_private_layout(tmp_path):
    archive_source = _source_tree(tmp_path / "archive-input")
    expected_directories, expected_files = _tree_snapshot(archive_source)
    archive = _archive(tmp_path, archive_source)
    extraction_parent = tmp_path / "private-extraction"
    extraction_parent.mkdir(mode=0o700)
    source_root = extraction_parent / prepare.ARCHIVE_ROOT
    output = tmp_path / "prepared-from-archive"

    result = prepare.prepare_edacc_conversation_fixture(
        source_root=source_root,
        archive_path=archive,
        output_dir=output,
        accepted_terms=prepare.REQUIRED_TERMS,
        test_source_injection=_injection(archive),
    )

    actual_directories, actual_files = _tree_snapshot(source_root)
    assert actual_directories == expected_directories
    assert actual_files == expected_files
    assert set(path.name for path in source_root.iterdir()) == {
        "README.txt",
        "data",
        "dev",
        "evaluate.sh",
        "glm",
        "linguistic_background.csv",
        "test",
    }
    assert actual_files[Path("README.txt")].endswith(b"\n")
    assert all(
        Path(f"data/{recording}.wav") in actual_files
        for recording in _DEV_RECORDINGS
    )
    assert Path(f"data/{_TEST_ALIASED_RECORDING}.wav") in actual_files
    assert actual_files[Path("dev/conv.list")] == (
        f"{_DEV_CONVERSATION_FAMILY}\n".encode()
    )
    assert actual_files[Path("test/conv.list")].endswith(
        f"{_TEST_ALIASED_CONVERSATION}\n".encode()
    )
    assert source_root.stat().st_mode & 0o777 == 0o700
    assert all(
        (source_root / relative).stat().st_mode & 0o777 == 0o700
        for relative in actual_directories
    )
    assert all(
        (source_root / relative).stat().st_mode & 0o777 == 0o600
        for relative in actual_files
    )
    assert result.production_evidence is False
    assert result.corpus.path.parent == output
    assert len(result.corpus.cases) == prepare.CASE_COUNT
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in output.iterdir())


def test_archive_scanner_accepts_exact_observed_edacc_base256_uid(
    valid_edacc_archive_bytes,
):
    payload = bytearray(gzip.decompress(valid_edacc_archive_bytes))
    offset, _end = _raw_tar_member(payload, f"{prepare.ARCHIVE_ROOT}/")
    _rewrite_raw_tar_numeric_field(
        payload,
        offset,
        field="uid",
        encoded=_OBSERVED_EDACC_UID_FIELD,
    )

    member = prepare._parse_tar_header(bytes(payload[offset : offset + 512]))

    assert _OBSERVED_EDACC_UID == 2_263_312
    assert member.name == prepare.ARCHIVE_ROOT
    assert member.is_directory is True


@pytest.mark.parametrize("field", ("uid", "gid"))
@pytest.mark.parametrize("value", (_OWNER_BASE256_MIN, _OWNER_MAX))
def test_archive_scanner_accepts_canonical_positive_base256_owner_boundaries(
    valid_edacc_archive_bytes,
    field,
    value,
):
    payload = bytearray(gzip.decompress(valid_edacc_archive_bytes))
    offset, _end = _raw_tar_member(payload, f"{prepare.ARCHIVE_ROOT}/")
    _rewrite_raw_tar_numeric_field(
        payload,
        offset,
        field=field,
        encoded=_positive_base256(value),
    )

    member = prepare._parse_tar_header(bytes(payload[offset : offset + 512]))

    assert member.name == prepare.ARCHIVE_ROOT


@pytest.mark.parametrize("field", ("uid", "gid"))
@pytest.mark.parametrize(
    "encoded",
    (
        _positive_base256(0),
        _positive_base256(_OWNER_BASE256_MIN - 1),
        b"\xff" * 8,
        b"\x81" + b"\0" * 7,
        _positive_base256(_OWNER_MAX + 1),
    ),
    ids=(
        "zero-base256",
        "octal-max-base256",
        "negative",
        "noncanonical-marker",
        "overflow",
    ),
)
def test_archive_scanner_rejects_invalid_base256_owner_fields(
    valid_edacc_archive_bytes,
    field,
    encoded,
):
    payload = bytearray(gzip.decompress(valid_edacc_archive_bytes))
    offset, _end = _raw_tar_member(payload, f"{prepare.ARCHIVE_ROOT}/")
    _rewrite_raw_tar_numeric_field(
        payload,
        offset,
        field=field,
        encoded=encoded,
    )

    with pytest.raises(prepare.EdaccPreparationError):
        prepare._parse_tar_header(bytes(payload[offset : offset + 512]))


def test_owner_base256_decoder_requires_the_eight_byte_uid_gid_width():
    with pytest.raises(prepare.EdaccPreparationError):
        prepare._tar_number(
            _positive_base256(8**6, length=7),
            maximum=_OWNER_MAX,
            allow_base256=True,
        )


@pytest.mark.parametrize(
    "field",
    ("mode", "size", "mtime", "checksum", "devmajor", "devminor"),
)
def test_archive_scanner_still_rejects_base256_outside_owner_fields(
    valid_edacc_archive_bytes,
    field,
):
    payload = bytearray(gzip.decompress(valid_edacc_archive_bytes))
    offset, _end = _raw_tar_member(
        payload,
        f"{prepare.ARCHIVE_ROOT}/evaluate.sh",
    )
    location = _TAR_NUMERIC_FIELDS[field]
    header = bytearray(payload[offset : offset + 512])
    if field == "checksum":
        header[location] = b" " * 8
        value = sum(header)
        _rewrite_raw_tar_numeric_field(
            payload,
            offset,
            field=field,
            encoded=_positive_base256(value),
            checksum=False,
        )
    else:
        original = bytes(header[location]).strip(b"\0 ") or b"0"
        value = int(original, 8)
        _rewrite_raw_tar_numeric_field(
            payload,
            offset,
            field=field,
            encoded=_positive_base256(
                value,
                length=location.stop - location.start,
            ),
        )

    with pytest.raises(prepare.EdaccPreparationError):
        prepare._parse_tar_header(bytes(payload[offset : offset + 512]))


def test_absent_source_archive_path_accepts_base256_uid_and_gid(tmp_path):
    archive_source = _source_tree(tmp_path / "archive-input")
    expected_directories, expected_files = _tree_snapshot(archive_source)
    original = _archive(tmp_path, archive_source)
    payload = bytearray(gzip.decompress(original.read_bytes()))
    offset, _end = _raw_tar_member(payload, f"{prepare.ARCHIVE_ROOT}/")
    _rewrite_raw_tar_numeric_field(
        payload,
        offset,
        field="uid",
        encoded=_OBSERVED_EDACC_UID_FIELD,
    )
    _rewrite_raw_tar_numeric_field(
        payload,
        offset,
        field="gid",
        encoded=_positive_base256(_OWNER_BASE256_MIN),
    )
    archive = _write_raw_tar_archive(tmp_path, payload, name="owner-base256.tar.gz")
    extraction_parent = tmp_path / "private-extraction"
    extraction_parent.mkdir(mode=0o700)
    source_root = extraction_parent / prepare.ARCHIVE_ROOT

    result = prepare.prepare_edacc_conversation_fixture(
        source_root=source_root,
        archive_path=archive,
        output_dir=tmp_path / "prepared-from-owner-base256",
        accepted_terms=prepare.REQUIRED_TERMS,
        test_source_injection=_injection(archive),
    )

    assert _tree_snapshot(source_root) == (expected_directories, expected_files)
    assert len(result.corpus.cases) == prepare.CASE_COUNT
    assert result.production_evidence is False


@pytest.mark.parametrize(
    "member_name",
    (
        f"{prepare.ARCHIVE_ROOT}/unexpected",
        f"{prepare.ARCHIVE_ROOT}/../escape",
        "/absolute",
        f"{prepare.ARCHIVE_ROOT}\\escape",
    ),
    ids=("unexpected", "traversal", "absolute", "backslash"),
)
def test_archive_scanner_rejects_unsafe_or_unexpected_member_names(
    tmp_path,
    valid_edacc_archive_bytes,
    member_name,
):
    payload = bytearray(gzip.decompress(valid_edacc_archive_bytes))
    offset, _end = _raw_tar_member(
        payload,
        f"{prepare.ARCHIVE_ROOT}/evaluate.sh",
    )
    _rewrite_raw_tar_header(payload, offset, name=member_name)

    archive = _write_raw_tar_archive(tmp_path, payload)
    _assert_archive_scanner_rejects(tmp_path, archive)


@pytest.mark.parametrize(
    "member_type",
    (
        tarfile.SYMTYPE,
        tarfile.LNKTYPE,
        tarfile.FIFOTYPE,
        tarfile.CHRTYPE,
        tarfile.BLKTYPE,
        tarfile.XHDTYPE,
        tarfile.XGLTYPE,
        tarfile.GNUTYPE_LONGNAME,
        tarfile.GNUTYPE_LONGLINK,
        tarfile.GNUTYPE_SPARSE,
        b"Z",
    ),
    ids=(
        "symlink",
        "hardlink",
        "fifo",
        "character-device",
        "block-device",
        "pax",
        "global-pax",
        "gnu-longname",
        "gnu-longlink",
        "gnu-sparse",
        "unknown",
    ),
)
def test_archive_scanner_rejects_non_regular_types_from_the_header(
    tmp_path,
    valid_edacc_archive_bytes,
    member_type,
):
    payload = bytearray(gzip.decompress(valid_edacc_archive_bytes))
    offset, _end = _raw_tar_member(
        payload,
        f"{prepare.ARCHIVE_ROOT}/evaluate.sh",
    )
    _rewrite_raw_tar_header(payload, offset, member_type=member_type)

    archive = _write_raw_tar_archive(tmp_path, payload)
    _assert_archive_scanner_rejects(tmp_path, archive)


@pytest.mark.parametrize("collision", ("duplicate", "casefold"))
def test_archive_scanner_rejects_duplicate_and_casefold_collisions(
    tmp_path,
    valid_edacc_archive_bytes,
    collision,
):
    payload = bytearray(gzip.decompress(valid_edacc_archive_bytes))
    original = f"{prepare.ARCHIVE_ROOT}/evaluate.sh"
    target = (
        original
        if collision == "duplicate"
        else f"{prepare.ARCHIVE_ROOT}/EVALUATE.SH"
    )
    _append_raw_tar_member(
        payload,
        source_name=original,
        target_name=target,
    )

    archive = _write_raw_tar_archive(tmp_path, payload)
    _assert_archive_scanner_rejects(tmp_path, archive)


@pytest.mark.parametrize("filename", ("README.txt", "evaluate.sh"))
def test_archive_scanner_rejects_missing_fixed_member(
    tmp_path,
    valid_edacc_archive_bytes,
    filename,
):
    payload = bytearray(gzip.decompress(valid_edacc_archive_bytes))
    start, end = _raw_tar_member(
        payload,
        f"{prepare.ARCHIVE_ROOT}/{filename}",
    )
    del payload[start:end]

    archive = _write_raw_tar_archive(tmp_path, payload)
    _assert_archive_scanner_rejects(tmp_path, archive)


def test_archive_scanner_rejects_oversize_readme_before_payload(tmp_path):
    source = _source_tree(tmp_path / "archive-input")
    original = _archive(tmp_path, source)
    payload = bytearray(gzip.decompress(original.read_bytes()))
    offset, _end = _raw_tar_member(
        payload,
        f"{prepare.ARCHIVE_ROOT}/README.txt",
    )
    _rewrite_raw_tar_header(
        payload,
        offset,
        size=prepare._ARCHIVE_ROOT_FILE_LIMITS["README.txt"] + 1,
    )

    archive = _write_raw_tar_archive(
        tmp_path,
        payload,
        name="oversize-readme.tar.gz",
    )
    _assert_archive_scanner_rejects(tmp_path, archive)


@pytest.mark.parametrize(
    "invalid_stem",
    (
        "EDACC-C24_P10",
        "EDACC-C24_PX",
        "EDACC-C24_P7_P8",
    ),
    ids=("two-digit-suffix", "nondigit-suffix", "double-suffix"),
)
def test_archive_scanner_rejects_nonpublisher_wav_suffixes(
    tmp_path,
    valid_edacc_archive_bytes,
    invalid_stem,
):
    payload = bytearray(gzip.decompress(valid_edacc_archive_bytes))
    valid_member = f"{prepare.ARCHIVE_ROOT}/data/{_DEV_RECORDINGS[0]}.wav"
    offset, _end = _raw_tar_member(payload, valid_member)
    _rewrite_raw_tar_header(
        payload,
        offset,
        name=f"{prepare.ARCHIVE_ROOT}/data/{invalid_stem}.wav",
    )

    archive = _write_raw_tar_archive(tmp_path, payload)
    _assert_archive_scanner_rejects(tmp_path, archive)


def test_archive_scanner_rejects_unexpected_root_file(
    tmp_path,
    valid_edacc_archive_bytes,
):
    payload = bytearray(gzip.decompress(valid_edacc_archive_bytes))
    _append_raw_tar_member(
        payload,
        source_name=f"{prepare.ARCHIVE_ROOT}/README.txt",
        target_name=f"{prepare.ARCHIVE_ROOT}/unexpected.txt",
    )

    archive = _write_raw_tar_archive(tmp_path, payload)
    _assert_archive_scanner_rejects(tmp_path, archive)


@pytest.mark.parametrize("relationship", ("orphan", "missing-referenced"))
def test_archive_scanner_rejects_audio_split_relationship_drift(
    tmp_path,
    valid_edacc_archive_bytes,
    relationship,
):
    payload = bytearray(gzip.decompress(valid_edacc_archive_bytes))
    referenced = f"{prepare.ARCHIVE_ROOT}/data/{_DEV_RECORDINGS[0]}.wav"
    if relationship == "orphan":
        _append_raw_tar_member(
            payload,
            source_name=referenced,
            target_name=f"{prepare.ARCHIVE_ROOT}/data/EDACC-C999.wav",
        )
    else:
        start, end = _raw_tar_member(payload, referenced)
        del payload[start:end]

    archive = _write_raw_tar_archive(tmp_path, payload)
    _assert_archive_scanner_rejects(tmp_path, archive)


def test_complete_layout_accepts_publisher_conversation_family_aliases(tmp_path):
    result, root, _archive_path = _prepare(tmp_path)

    assert len(result.corpus.cases) == prepare.CASE_COUNT
    assert (root / "test" / "conv.list").read_text(encoding="utf-8").endswith(
        f"{_TEST_ALIASED_CONVERSATION}\n"
    )
    assert (root / "dev" / "conv.list").read_text(
        encoding="utf-8"
    ) == f"{_DEV_CONVERSATION_FAMILY}\n"
    dev_segments = (root / "dev" / "segments").read_text(encoding="utf-8")
    assert all(
        f" {recording} " in dev_segments for recording in _DEV_RECORDINGS
    )


@pytest.mark.parametrize(
    "conversation_id",
    (
        "EDACC-C999_P7",
        "EDACC-C23_P10",
        "EDACC-C23_PX",
        "EDACC-C23_P7_P8",
    ),
    ids=(
        "wrong-family",
        "two-digit-participant",
        "nondigit-participant",
        "double-participant",
    ),
)
def test_complete_layout_rejects_wrong_or_malformed_conversation_id(
    tmp_path,
    conversation_id,
):
    root = _source_tree(tmp_path / "source")
    conversations = root / "test" / "conv.list"
    original = conversations.read_text(encoding="utf-8")
    assert f"{_TEST_ALIASED_CONVERSATION}\n" in original
    conversations.write_text(
        original.replace(
            f"{_TEST_ALIASED_CONVERSATION}\n",
            f"{conversation_id}\n",
        ),
        encoding="utf-8",
    )

    _assert_complete_layout_rejects(tmp_path, root)


def test_complete_layout_rejects_cross_split_conversation_family_overlap(
    tmp_path,
):
    root = _source_tree(tmp_path / "source")
    replacements = dict(
        zip(_DEV_RECORDINGS, ("EDACC-C23_P3", "EDACC-C23_P4"), strict=True)
    )
    for old, new in replacements.items():
        (root / "data" / f"{old}.wav").rename(root / "data" / f"{new}.wav")
    for filename in ("company.ctm", "segments", "stm", "stm.filt", "utt2spk"):
        path = root / "dev" / filename
        payload = path.read_text(encoding="utf-8")
        for old, new in replacements.items():
            payload = payload.replace(old, new)
        path.write_text(payload, encoding="utf-8")
    (root / "dev" / "conv.list").write_text("EDACC-C23\n", encoding="utf-8")

    _assert_complete_layout_rejects(tmp_path, root)


@pytest.mark.parametrize(
    "drift",
    ("missing-wav", "extra-wav", "missing-segment", "extra-segment"),
)
def test_complete_layout_rejects_segment_wav_exact_stem_drift(tmp_path, drift):
    root = _source_tree(tmp_path / "source")
    data = root / "data"
    segments = root / "dev" / "segments"
    if drift == "missing-wav":
        (data / f"{_DEV_RECORDINGS[1]}.wav").unlink()
    elif drift == "extra-wav":
        _write_wav(data / "EDACC-C24_P3.wav", 26)
    elif drift == "missing-segment":
        payload = segments.read_text(encoding="utf-8")
        matching = f"dev-utt-2 {_DEV_RECORDINGS[1]} 0.00 0.50\n"
        assert matching in payload
        segments.write_text(payload.replace(matching, ""), encoding="utf-8")
    else:
        with segments.open("a", encoding="utf-8") as handle:
            handle.write("dev-utt-3 EDACC-C24_P3 0.00 0.50\n")

    _assert_complete_layout_rejects(tmp_path, root)


@pytest.mark.parametrize("corruption", ("checksum", "padding", "tail"))
def test_archive_scanner_rejects_tar_integrity_corruption(
    tmp_path,
    valid_edacc_archive_bytes,
    corruption,
):
    payload = bytearray(gzip.decompress(valid_edacc_archive_bytes))
    start, _end = _raw_tar_member(
        payload,
        f"{prepare.ARCHIVE_ROOT}/evaluate.sh",
    )
    if corruption == "checksum":
        assert payload[start + 148] in b"01234567"
        payload[start + 148] ^= 1
    elif corruption == "padding":
        size = int(bytes(payload[start + 124 : start + 136]).strip(b"\0 "), 8)
        assert size % 512
        payload[start + 512 + size] = 1
    else:
        trailing = _raw_tar_terminator(payload) + 1024
        if trailing == len(payload):
            payload.extend(b"\0" * 512)
        assert trailing < len(payload)
        payload[trailing] = 1

    archive = _write_raw_tar_archive(tmp_path, payload)
    _assert_archive_scanner_rejects(tmp_path, archive)


def test_archive_scanner_rejects_concatenated_gzip_members(
    tmp_path,
    valid_edacc_archive_bytes,
):
    archive = tmp_path / "concatenated.tar.gz"
    archive.write_bytes(
        valid_edacc_archive_bytes
        + gzip.compress(b"\0" * 512, compresslevel=1, mtime=0)
    )
    archive.chmod(0o600)

    _assert_archive_scanner_rejects(tmp_path, archive)


def test_archive_scanner_rejects_huge_extension_before_reading_its_body(
    tmp_path,
    monkeypatch,
):
    extension = tarfile.TarInfo(f"{prepare.ARCHIVE_ROOT}/pax-extension")
    extension.mode = 0o600
    extension.uid = 0
    extension.gid = 0
    extension.mtime = 0
    extension.type = tarfile.XHDTYPE
    extension.size = 1 << 30
    header = extension.tobuf(format=tarfile.USTAR_FORMAT)
    assert len(header) == 512
    archive = _write_raw_tar_archive(tmp_path, header, name="huge-pax.tar.gz")
    read_sizes: list[int] = []
    original_read = prepare._CompressedArchiveStream.read

    def tracking_read(self, size):
        read_sizes.append(size)
        return original_read(self, size)

    monkeypatch.setattr(prepare._CompressedArchiveStream, "read", tracking_read)

    _assert_archive_scanner_rejects(tmp_path, archive)
    assert read_sizes == [512]


def test_materializes_private_schema_v2_and_transcript_free_receipt(tmp_path):
    result, _root, archive = _prepare(tmp_path)
    corpus = result.corpus
    receipt_path = corpus.path.parent / "preparation-receipt.json"
    receipt_raw = receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)

    assert corpus.schema_version == 2
    assert len(corpus.cases) == 24
    assert corpus.audio_bytes == 24 * 8_000 * 4
    assert result.production_evidence is False
    assert hashlib.sha256(receipt_raw).hexdigest() == result.receipt_sha256
    assert receipt["decoder"] == {
        "contract": prepare.DECODER_CONTRACT,
        "closure_sha256": receipt["decoder"]["closure_sha256"],
        "files": 24,
        "bytes": sum(
            (_root / "data" / f"{_test_recording(index)}.wav").stat().st_size
            for index in range(24)
        ),
        "production_evidence": False,
    }
    assert receipt["selection"]["eligible_rows"] == 24
    assert receipt["selection"]["parent_receipt_sha256"] is None
    assert receipt["selection"]["parent_corpus_manifest_sha256"] is None
    assert Counter(case["attributes"]["stratum"] for case in receipt["cases"]) == {
        "clean": 8,
        "disfluent": 8,
        "overlap_adjacent": 8,
    }
    assert len({case["speaker_sha256"] for case in receipt["cases"]}) == 24
    assert len({case["attributes"]["accent_sha256"] for case in receipt["cases"]}) == 12
    assert all(
        case["attributes"]["actual_overlap"] is False for case in receipt["cases"]
    )
    assert all(
        case["attributes"]["reference_source"] == "official_filtered_stm"
        for case in receipt["cases"]
    )
    assert receipt["artifacts"] == [
        {
            "artifact_id": "archive",
            "upstream_algorithm": "md5",
            "upstream_digest": _injection(archive).archive_md5,
            "size_bytes": archive.stat().st_size,
            "local_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        }
    ]

    assert corpus.path.parent.stat().st_mode & 0o777 == 0o700
    assert all(
        path.stat().st_mode & 0o777 == 0o600 for path in corpus.path.parent.iterdir()
    )
    first = corpus.cases[0]
    sample = struct.unpack("<f", first.source_path.read_bytes()[:4])[0]
    assert 0.0 < sample < 0.1
    references = {case.expected_text for case in corpus.cases}
    assert "SYNTHETIC PHRASE NUMBER 8" in references
    assert "UM SYNTHETIC PHRASE NUMBER 8" not in references
    assert load_corpus(corpus.path).digest == corpus.digest

    encoded = receipt_raw.decode("ascii")
    for private in (
        "EDACC-C",
        "EDAC-C",
        "SYNTHETIC",
        "Synthetic accent",
        str(tmp_path),
    ):
        assert private not in encoded


def test_selection_and_raw_evidence_are_deterministic(tmp_path):
    root = _source_tree(tmp_path)
    archive = _archive(tmp_path, root)
    injection = _injection(archive)
    first = prepare.prepare_edacc_conversation_fixture(
        source_root=root,
        archive_path=archive,
        output_dir=tmp_path / "first",
        accepted_terms=prepare.REQUIRED_TERMS,
        test_source_injection=injection,
    )
    second = prepare.prepare_edacc_conversation_fixture(
        source_root=root,
        archive_path=archive,
        output_dir=tmp_path / "second",
        accepted_terms=prepare.REQUIRED_TERMS,
        test_source_injection=injection,
    )
    assert first.selected_rows_sha256 == second.selected_rows_sha256
    assert first.metadata_sha256 == second.metadata_sha256
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.corpus.digest == second.corpus.digest

    filtered = root / "test" / "stm.filt"
    rows = filtered.read_text(encoding="utf-8").splitlines(keepends=True)
    rows[0] = rows[0].rstrip("\n") + " \n"
    filtered.write_text("".join(rows), encoding="utf-8")
    changed_archive = _archive(tmp_path, root, name="changed.tar.gz")
    changed = prepare.prepare_edacc_conversation_fixture(
        source_root=root,
        archive_path=changed_archive,
        output_dir=tmp_path / "changed",
        accepted_terms=prepare.REQUIRED_TERMS,
        test_source_injection=_injection(changed_archive, "edacc-unit-v2"),
    )
    assert changed.selected_rows_sha256 == first.selected_rows_sha256
    assert changed.metadata_sha256 != first.metadata_sha256
    assert changed.receipt_sha256 != first.receipt_sha256


def test_filtered_stm_is_key_aligned_not_positionally_bound(tmp_path):
    reordered_root = _source_tree(tmp_path / "reordered")
    filtered = reordered_root / "test" / "stm.filt"
    rows = filtered.read_text(encoding="utf-8").splitlines(keepends=True)
    filtered.write_text("".join(reversed(rows)), encoding="utf-8")
    reordered_archive = _archive(tmp_path / "reordered", reordered_root)
    reordered = prepare.prepare_edacc_conversation_fixture(
        source_root=reordered_root,
        archive_path=reordered_archive,
        output_dir=tmp_path / "reordered-output",
        accepted_terms=prepare.REQUIRED_TERMS,
        test_source_injection=_injection(reordered_archive),
    )
    references = {case.expected_text for case in reordered.corpus.cases}
    assert "SYNTHETIC PHRASE NUMBER 8" in references
    assert "UM SYNTHETIC PHRASE NUMBER 8" not in references

    mismatched_root = _source_tree(tmp_path / "mismatched")
    mismatched_filtered = mismatched_root / "test" / "stm.filt"
    rows = mismatched_filtered.read_text(encoding="utf-8").splitlines(keepends=True)
    rows[0] = rows[0].replace("EDACC-C00", "EDACC-C99", 1)
    mismatched_filtered.write_text("".join(rows), encoding="utf-8")
    mismatched_archive = _archive(tmp_path / "mismatched", mismatched_root)
    with pytest.raises(prepare.EdaccPreparationError):
        prepare.prepare_edacc_conversation_fixture(
            source_root=mismatched_root,
            archive_path=mismatched_archive,
            output_dir=tmp_path / "mismatched-output",
            accepted_terms=prepare.REQUIRED_TERMS,
            test_source_injection=_injection(mismatched_archive),
        )


def test_extraction_and_archive_tampering_fail_closed(tmp_path):
    root = _source_tree(tmp_path)
    archive = _archive(tmp_path, root)
    injection = _injection(archive)

    text_path = root / "test" / "text"
    text_path.write_text(
        text_path.read_text(encoding="utf-8").replace(
            "SYNTHETIC CLEAN PHRASE NUMBER 0",
            "SYNTHETIC ALTERED PHRASE NUMBER 0",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(prepare.EdaccPreparationError):
        prepare.prepare_edacc_conversation_fixture(
            source_root=root,
            archive_path=archive,
            output_dir=tmp_path / "tampered-tree",
            accepted_terms=prepare.REQUIRED_TERMS,
            test_source_injection=injection,
        )

    pristine_root = _source_tree(tmp_path / "pristine")
    pristine_archive = _archive(tmp_path / "pristine", pristine_root)
    pristine_injection = _injection(pristine_archive)
    with pristine_archive.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(prepare.EdaccPreparationError):
        prepare.prepare_edacc_conversation_fixture(
            source_root=pristine_root,
            archive_path=pristine_archive,
            output_dir=tmp_path / "tampered-archive",
            accepted_terms=prepare.REQUIRED_TERMS,
            test_source_injection=pristine_injection,
        )


def test_official_catalog_pin_and_decoder_contract_are_executable():
    prepare._validate_catalog_contract()
    dataset = prepare.dataset_by_id(prepare.DATASET_ID)
    artifact = dataset.artifacts[0]
    assert prepare.DOI == "10.7488/ds/7914"
    assert prepare.ARCHIVE_FILENAME == "edacc_v1.0.tar.gz"
    assert prepare.ARCHIVE_SIZE_BYTES == 5_916_732_170
    assert prepare.ARCHIVE_MD5 == "146b4b8026b5d0ce9611667c708456b3"
    assert artifact.checksum is not None
    assert artifact.checksum.digest == prepare.ARCHIVE_MD5
    assert prepare.DECODER_CONTRACT == ("edacc-wav-pcm16-mono32k-box2-f32le16k-v1")


def test_filtered_stm_owns_hard_wer_reference_scoreability():
    assert prepare._scoreable_reference(
        "UM SOURCE WORDS",
        "UM SOURCE WORDS",
        "SOURCE WORDS",
    )
    assert not prepare._scoreable_reference(
        "UM SOURCE WORDS",
        "UM SOURCE WORDS",
        "",
    )
    assert not prepare._scoreable_reference(
        "HM SOURCE WORDS",
        "HM SOURCE WORDS",
        "{ HM / HMM } SOURCE WORDS",
    )


def test_lock_recipe_is_the_exact_executable_matrix_selector():
    dataset = prepare.dataset_by_id(prepare.DATASET_ID)
    lock = prepare.load_fixture_lock()
    source = lock.source_by_id(prepare.SOURCE_ID)

    assert dataset.selection.seed == prepare.SELECTION_SEED
    assert source.selection_recipe == prepare._expected_source_recipe(dataset)
    prepare._validate_fixture_source_contract(source, dataset=dataset)

    drifted_recipe = dict(source.selection_recipe)
    drifted_recipe["seed"] = lock.selection_seed
    with pytest.raises(prepare.EdaccPreparationError):
        prepare._validate_fixture_source_contract(
            replace(source, selection_recipe=drifted_recipe),
            dataset=dataset,
        )


def test_cli_failure_is_detail_free(tmp_path, capsys):
    private = tmp_path / "private-source-canary"
    assert (
        prepare.main(
            [
                "--archive",
                str(private / "archive.tar.gz"),
                "--source-root",
                str(private / prepare.ARCHIVE_ROOT),
                "--output-dir",
                str(tmp_path / "output"),
                "--accept-term",
                prepare.LICENSE_ID,
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == prepare._SAFE_ERROR
    assert str(private) not in captured.err


def test_rejects_test_contract_that_could_be_mistaken_for_production(tmp_path):
    with pytest.raises(prepare.EdaccPreparationError):
        prepare._test_contract(
            prepare.TestSourceInjection(
                archive_size_bytes=1,
                archive_md5="not-a-digest",
                fixture_id="unit",
            )
        )
    with pytest.raises(prepare.EdaccPreparationError):
        prepare.prepare_edacc_conversation_fixture(
            source_root=tmp_path / prepare.ARCHIVE_ROOT,
            archive_path=tmp_path / "missing.tar.gz",
            output_dir=tmp_path / "output",
            accepted_terms=frozenset(),
            test_source_injection=prepare.TestSourceInjection(
                archive_size_bytes=1,
                archive_md5="0" * 32,
                fixture_id="unit",
            ),
        )


def test_production_paths_are_private_canonical_and_outside_git(tmp_path):
    git_root = tmp_path / "checkout"
    git_root.mkdir(mode=0o700)
    (git_root / ".git").write_text("gitdir: private-metadata\n", encoding="utf-8")
    source = git_root / prepare.ARCHIVE_ROOT
    source.mkdir(mode=0o700)
    archive = git_root / prepare.ARCHIVE_FILENAME
    archive.write_bytes(b"synthetic archive")
    archive.chmod(0o600)
    output = git_root / "prepared"

    with pytest.raises(prepare.EdaccPreparationError):
        prepare._validate_private_paths(
            source_root=source,
            archive_path=archive,
            output_dir=output,
            production=True,
        )
    validated = prepare._validate_private_paths(
        source_root=source,
        archive_path=archive,
        output_dir=output,
        production=False,
    )
    assert (
        validated.source_root,
        validated.archive_path,
        validated.output_dir,
    ) == (source, archive, output)

    source.chmod(0o755)
    with pytest.raises(prepare.EdaccPreparationError):
        prepare._validate_private_paths(
            source_root=source,
            archive_path=archive,
            output_dir=output,
            production=False,
        )
    source.chmod(0o700)
    archive.chmod(0o644)
    with pytest.raises(prepare.EdaccPreparationError):
        prepare._validate_private_paths(
            source_root=source,
            archive_path=archive,
            output_dir=output,
            production=False,
        )


def test_preexisting_source_root_cannot_bypass_production_owned_extraction(
    tmp_path,
    monkeypatch,
):
    source = _source_tree(tmp_path)
    archive = _archive(
        tmp_path,
        source,
        name=prepare.ARCHIVE_FILENAME,
    )
    identity = _injection(archive)
    output = tmp_path / "must-not-exist"
    acquisition_calls = 0
    consume_calls = 0
    original_validate = prepare._validate_private_acquisition_paths

    def tracking_validate(**kwargs):
        nonlocal acquisition_calls
        acquisition_calls += 1
        return original_validate(**kwargs)

    def forbidden_consume(*args, **kwargs):
        nonlocal consume_calls
        consume_calls += 1
        raise prepare.EdaccPreparationError()

    monkeypatch.setattr(prepare, "_validate_catalog_contract", lambda: None)
    monkeypatch.setattr(
        prepare,
        "ARCHIVE_SIZE_BYTES",
        identity.archive_size_bytes,
    )
    monkeypatch.setattr(prepare, "ARCHIVE_MD5", identity.archive_md5)
    monkeypatch.setattr(
        prepare,
        "_validate_private_acquisition_paths",
        tracking_validate,
    )
    monkeypatch.setattr(prepare, "_consume_archive", forbidden_consume)

    with pytest.raises(prepare.EdaccPreparationError):
        prepare.prepare_edacc_conversation_fixture(
            source_root=source,
            archive_path=archive,
            output_dir=output,
            accepted_terms=prepare.REQUIRED_TERMS,
        )

    assert acquisition_calls == 1
    assert consume_calls == 0
    assert not output.exists()


def test_production_paths_ignore_stable_inert_git_sentinel(tmp_path):
    cache = tmp_path / "non-git-cache"
    cache.mkdir()
    sentinel = cache / ".git"
    sentinel.mkdir()
    (sentinel / "info").mkdir()
    source = cache / prepare.ARCHIVE_ROOT
    source.mkdir(mode=0o700)
    archive = cache / prepare.ARCHIVE_FILENAME
    archive.write_bytes(b"synthetic archive")
    archive.chmod(0o600)
    output = cache / "prepared"

    validated = prepare._validate_private_paths(
        source_root=source,
        archive_path=archive,
        output_dir=output,
        production=True,
    )

    assert validated.output_dir == output
    assert not output.exists()


def test_git_ancestor_probe_ignores_stable_empty_regular_marker(tmp_path):
    cache = tmp_path / "empty-marker-cache"
    cache.mkdir()
    marker = cache / ".git"
    marker.touch()

    with prepare.opened_directory_nofollow(cache) as (_bound, descriptor):
        assert prepare._git_marker_present(descriptor) is False


def test_git_ancestor_probe_rejects_empty_regular_marker_growth(
    tmp_path,
    monkeypatch,
):
    cache = tmp_path / "growing-marker-cache"
    cache.mkdir()
    marker = cache / ".git"
    marker.touch()
    original_stat = prepare.os.stat
    mutated = False

    def growing_stat(path, *args, **kwargs):
        nonlocal mutated
        result = original_stat(path, *args, **kwargs)
        if path == ".git" and kwargs.get("dir_fd") is not None and not mutated:
            marker.write_text("gitdir: private-metadata\n", encoding="utf-8")
            mutated = True
        return result

    monkeypatch.setattr(prepare.os, "stat", growing_stat)

    with prepare.opened_directory_nofollow(cache) as (_bound, descriptor):
        with pytest.raises(prepare.EdaccPreparationError):
            prepare._git_marker_present(descriptor)
    assert mutated is True


def test_production_paths_reject_git_marker_replaced_during_probe(
    tmp_path,
    monkeypatch,
):
    cache = tmp_path / "mutating-cache"
    cache.mkdir()
    marker = cache / ".git"
    marker.mkdir()
    source = cache / prepare.ARCHIVE_ROOT
    source.mkdir(mode=0o700)
    archive = cache / prepare.ARCHIVE_FILENAME
    archive.write_bytes(b"synthetic archive")
    archive.chmod(0o600)
    output = cache / "must-not-exist"
    original_stat = prepare.os.stat
    mutated = False

    def mutating_stat(path, *args, **kwargs):
        nonlocal mutated
        if path == "HEAD" and kwargs.get("dir_fd") is not None and not mutated:
            mutated = True
            marker.rename(cache / ".git-moved")
            marker.mkdir()
            (marker / "HEAD").write_text(
                "ref: refs/heads/main\n",
                encoding="utf-8",
            )
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(prepare.os, "stat", mutating_stat)

    with pytest.raises(prepare.EdaccPreparationError):
        prepare._validate_private_paths(
            source_root=source,
            archive_path=archive,
            output_dir=output,
            production=True,
        )
    assert mutated is True
    assert not output.exists()


def test_git_ancestor_probe_rejects_lexical_ancestor_replacement(
    tmp_path,
    monkeypatch,
):
    parent = tmp_path / "source-parent"
    parent.mkdir()
    source = parent / prepare.ARCHIVE_ROOT
    source.mkdir(mode=0o700)
    source_inode = source.stat().st_ino
    moved = tmp_path / "source-parent-moved"
    original_probe = prepare._git_marker_present
    calls = 0
    mutated = False

    def mutating_probe(descriptor):
        nonlocal calls, mutated
        calls += 1
        result = original_probe(descriptor)
        if calls == 2:
            parent.rename(moved)
            parent.mkdir()
            (moved / source.name).rename(parent / source.name)
            marker = parent / ".git"
            marker.mkdir()
            (marker / "HEAD").write_text(
                "ref: refs/heads/main\n",
                encoding="utf-8",
            )
            mutated = True
        return result

    monkeypatch.setattr(prepare, "_git_marker_present", mutating_probe)

    with pytest.raises(prepare.EdaccPreparationError):
        prepare._has_git_ancestor(source)
    assert mutated is True
    assert source.stat().st_ino == source_inode
    assert (parent / ".git" / "HEAD").is_file()


@pytest.mark.parametrize("target", ("metadata", "audio", "archive"))
def test_source_snapshot_rejects_mutation_during_path_resolution(
    tmp_path,
    monkeypatch,
    target,
):
    root = _source_tree(tmp_path)
    archive = _archive(tmp_path, root)
    output = tmp_path / "must-not-exist"
    paths = prepare._validate_private_paths(
        source_root=root,
        archive_path=archive,
        output_dir=output,
        production=False,
    )
    metadata = prepare._snapshot_metadata(root)
    identity = _injection(archive)
    archive_snapshot = prepare._consume_archive(
        archive,
        expected_identity=prepare._file_identity(archive.lstat()),
        expected_size=identity.archive_size_bytes,
        expected_md5=identity.archive_md5,
    )
    audio_path = root / "data" / "EDACC-C00.wav"
    audio = {"EDACC-C00": prepare._bind_audio(audio_path)}
    selected = {
        "metadata": next(iter(metadata.values())).path,
        "audio": audio_path,
        "archive": archive,
    }[target]
    original_resolve = Path.resolve
    mutated = False

    def mutating_resolve(path, *args, **kwargs):
        nonlocal mutated
        result = original_resolve(path, *args, **kwargs)
        if path == selected and not mutated:
            _mutate_first_byte(selected)
            mutated = True
        return result

    monkeypatch.setattr(Path, "resolve", mutating_resolve)

    with pytest.raises(prepare.EdaccPreparationError):
        prepare._verify_private_source_snapshot(
            paths,
            archive_snapshot=archive_snapshot,
            metadata=metadata,
            audio=audio,
            production=False,
            output_created=False,
        )
    assert mutated is True
    assert not output.exists()


@pytest.mark.parametrize("target", ("metadata", "audio", "archive"))
def test_source_snapshot_rebinds_every_input_after_metadata_reads(
    tmp_path,
    monkeypatch,
    target,
):
    root = _source_tree(tmp_path)
    archive = _archive(tmp_path, root)
    output = tmp_path / "must-not-exist"
    paths = prepare._validate_private_paths(
        source_root=root,
        archive_path=archive,
        output_dir=output,
        production=False,
    )
    metadata = prepare._snapshot_metadata(root)
    identity = _injection(archive)
    archive_snapshot = prepare._consume_archive(
        archive,
        expected_identity=prepare._file_identity(archive.lstat()),
        expected_size=identity.archive_size_bytes,
        expected_md5=identity.archive_md5,
    )
    audio_path = root / "data" / "EDACC-C00.wav"
    audio = {"EDACC-C00": prepare._bind_audio(audio_path)}
    selected = {
        "metadata": next(iter(metadata.values())).path,
        "audio": audio_path,
        "archive": archive,
    }[target]
    original_metadata_bytes = prepare._metadata_bytes
    calls = 0
    mutated = False

    def mutating_metadata_bytes(bindings, relative):
        nonlocal calls, mutated
        result = original_metadata_bytes(bindings, relative)
        calls += 1
        if calls == len(metadata):
            _mutate_first_byte(selected)
            mutated = True
        return result

    monkeypatch.setattr(prepare, "_metadata_bytes", mutating_metadata_bytes)

    with pytest.raises(prepare.EdaccPreparationError):
        prepare._verify_private_source_snapshot(
            paths,
            archive_snapshot=archive_snapshot,
            metadata=metadata,
            audio=audio,
            production=False,
            output_created=False,
        )
    assert calls == len(metadata)
    assert mutated is True
    assert not output.exists()


@pytest.mark.parametrize("target", ("source", "archive"))
def test_rejects_private_input_mode_drift_after_preflight(
    tmp_path,
    monkeypatch,
    target,
):
    root = _source_tree(tmp_path)
    archive = _archive(tmp_path, root)
    injection = _injection(archive)
    output = tmp_path / "must-not-exist"
    original_validate = prepare._validate_private_paths

    def drifting_validate(**kwargs):
        paths = original_validate(**kwargs)
        if target == "source":
            root.chmod(0o755)
        else:
            archive.chmod(0o644)
        return paths

    monkeypatch.setattr(prepare, "_validate_private_paths", drifting_validate)

    with pytest.raises(prepare.EdaccPreparationError):
        prepare.prepare_edacc_conversation_fixture(
            source_root=root,
            archive_path=archive,
            output_dir=output,
            accepted_terms=prepare.REQUIRED_TERMS,
            test_source_injection=injection,
        )
    assert not output.exists()


def test_rejects_destination_parent_replacement_before_publication(
    tmp_path,
    monkeypatch,
):
    root = _source_tree(tmp_path)
    archive = _archive(tmp_path, root)
    injection = _injection(archive)
    parent = tmp_path / "publication-parent"
    parent.mkdir()
    output = parent / "must-not-exist"
    moved = tmp_path / "publication-parent-moved"
    original_validate = prepare._validate_private_paths

    def replacing_validate(**kwargs):
        paths = original_validate(**kwargs)
        parent.rename(moved)
        parent.mkdir()
        return paths

    monkeypatch.setattr(prepare, "_validate_private_paths", replacing_validate)

    with pytest.raises(prepare.EdaccPreparationError):
        prepare.prepare_edacc_conversation_fixture(
            source_root=root,
            archive_path=archive,
            output_dir=output,
            accepted_terms=prepare.REQUIRED_TERMS,
            test_source_injection=injection,
        )
    assert not output.exists()


def test_rejects_destination_parent_replacement_inside_publication(
    tmp_path,
    monkeypatch,
):
    root = _source_tree(tmp_path)
    archive = _archive(tmp_path, root)
    injection = _injection(archive)
    parent = tmp_path / "publisher-parent"
    parent.mkdir()
    output = parent / "must-not-exist"
    moved = tmp_path / "publisher-parent-moved"
    original_publish = prepare._publish_private_corpus_bound
    replaced = False

    def replacing_publish(**kwargs):
        nonlocal replaced
        parent.rename(moved)
        parent.mkdir()
        replaced = True
        return original_publish(**kwargs)

    monkeypatch.setattr(prepare, "_publish_private_corpus_bound", replacing_publish)

    with pytest.raises(prepare.EdaccPreparationError):
        prepare.prepare_edacc_conversation_fixture(
            source_root=root,
            archive_path=archive,
            output_dir=output,
            accepted_terms=prepare.REQUIRED_TERMS,
            test_source_injection=injection,
        )
    assert replaced is True
    assert not output.exists()
    assert not (moved / output.name).exists()


@pytest.mark.parametrize("target", ("source", "archive"))
def test_rejects_private_input_mode_drift_after_publication(
    tmp_path,
    monkeypatch,
    target,
):
    root = _source_tree(tmp_path)
    archive = _archive(tmp_path, root)
    injection = _injection(archive)
    output = tmp_path / "retained-failed-output"
    original_publish = prepare._publish_private_corpus_bound

    def mutating_publish(**kwargs):
        published = original_publish(**kwargs)
        if target == "source":
            root.chmod(0o755)
        else:
            archive.chmod(0o644)
        return published

    monkeypatch.setattr(prepare, "_publish_private_corpus_bound", mutating_publish)

    with pytest.raises(prepare.EdaccPreparationError):
        prepare.prepare_edacc_conversation_fixture(
            source_root=root,
            archive_path=archive,
            output_dir=output,
            accepted_terms=prepare.REQUIRED_TERMS,
            test_source_injection=injection,
        )
    assert output.is_dir()


@pytest.mark.parametrize("target", ("manifest", "receipt", "pcm"))
def test_rejects_public_output_mode_drift_after_publication(
    tmp_path,
    monkeypatch,
    target,
):
    root = _source_tree(tmp_path)
    archive = _archive(tmp_path, root)
    injection = _injection(archive)
    output = tmp_path / "retained-public-output"
    original_publish = prepare._publish_private_corpus_bound

    def mutating_publish(**kwargs):
        published = original_publish(**kwargs)
        if target == "manifest":
            selected = published.path
        elif target == "receipt":
            selected = published.path.parent / "preparation-receipt.json"
        else:
            selected = next(published.path.parent.glob("*.f32le"))
        selected.chmod(0o644)
        return published

    monkeypatch.setattr(prepare, "_publish_private_corpus_bound", mutating_publish)

    with pytest.raises(prepare.EdaccPreparationError):
        prepare.prepare_edacc_conversation_fixture(
            source_root=root,
            archive_path=archive,
            output_dir=output,
            accepted_terms=prepare.REQUIRED_TERMS,
            test_source_injection=injection,
        )
    assert output.is_dir()


@pytest.mark.parametrize("target", ("metadata", "audio", "archive"))
def test_rejects_private_input_content_drift_after_publication(
    tmp_path,
    monkeypatch,
    target,
):
    root = _source_tree(tmp_path)
    archive = _archive(tmp_path, root)
    injection = _injection(archive)
    output = tmp_path / "retained-input-mutation-output"
    original_publish = prepare._publish_private_corpus_bound

    def mutating_publish(**kwargs):
        published = original_publish(**kwargs)
        if target == "metadata":
            selected = root / "test" / "text"
        elif target == "audio":
            selected = next((root / "data").glob("*.wav"))
        else:
            selected = archive
        _mutate_first_byte(selected)
        return published

    monkeypatch.setattr(prepare, "_publish_private_corpus_bound", mutating_publish)

    with pytest.raises(prepare.EdaccPreparationError):
        prepare.prepare_edacc_conversation_fixture(
            source_root=root,
            archive_path=archive,
            output_dir=output,
            accepted_terms=prepare.REQUIRED_TERMS,
            test_source_injection=injection,
        )
    assert output.is_dir()


@pytest.mark.parametrize("target", ("manifest", "receipt", "pcm"))
def test_rejects_public_output_content_drift_after_publication(
    tmp_path,
    monkeypatch,
    target,
):
    root = _source_tree(tmp_path)
    archive = _archive(tmp_path, root)
    injection = _injection(archive)
    output = tmp_path / "retained-content-mutation-output"
    original_publish = prepare._publish_private_corpus_bound

    def mutating_publish(**kwargs):
        published = original_publish(**kwargs)
        if target == "manifest":
            selected = published.path
        elif target == "receipt":
            selected = published.path.parent / "preparation-receipt.json"
        else:
            selected = next(published.path.parent.glob("*.f32le"))
        _mutate_first_byte(selected)
        return published

    monkeypatch.setattr(prepare, "_publish_private_corpus_bound", mutating_publish)

    with pytest.raises(prepare.EdaccPreparationError):
        prepare.prepare_edacc_conversation_fixture(
            source_root=root,
            archive_path=archive,
            output_dir=output,
            accepted_terms=prepare.REQUIRED_TERMS,
            test_source_injection=injection,
        )
    assert output.is_dir()


def test_production_publication_calls_canonical_source_validator(
    tmp_path,
    monkeypatch,
):
    result, _root, _archive_path = _prepare(tmp_path)
    corpus = result.corpus
    assert corpus.provenance is not None
    receipt_payload = (corpus.path.parent / "preparation-receipt.json").read_bytes()
    expected_cases = tuple(
        prepare.CorpusWriteCase(
            case_id=case.case_id,
            audio_bytes=case.audio_bytes,
            reference=case.expected_text,
            tags=case.tags,
            commands=case.commands,
            assertion=case.assertion,
            forbidden_commands=case.forbidden_commands,
        )
        for case in corpus.cases
    )
    lock_sentinel = object()
    calls = []
    parent_identity = prepare._directory_entry_identity(
        corpus.path.parent.parent.lstat()
    )

    class Binding:
        receipt_sha256 = result.receipt_sha256
        corpus_manifest_sha256 = corpus.digest
        cases = prepare.CASE_COUNT
        pcm_bytes = corpus.audio_bytes

    def validate_private_source(path, *, lock, expected_source_id):
        calls.append((path, lock, expected_source_id))
        return Binding()

    monkeypatch.setattr(
        prepare.fixture,
        "validate_private_source",
        validate_private_source,
    )

    validated = prepare._validate_published_output(
        published=corpus,
        destination=corpus.path.parent,
        receipt_payload=receipt_payload,
        receipt_sha256=result.receipt_sha256,
        provenance=corpus.provenance,
        expected_cases=expected_cases,
        lock=lock_sentinel,
        production=True,
        expected_parent_identity=parent_identity,
    )

    assert validated.digest == corpus.digest
    assert calls == [(corpus.path, lock_sentinel, prepare.SOURCE_ID)]


def test_rejects_destination_parent_replacement_at_final_output_snapshot(
    tmp_path,
    monkeypatch,
):
    root = _source_tree(tmp_path)
    archive = _archive(tmp_path, root)
    injection = _injection(archive)
    parent = tmp_path / "final-parent"
    parent.mkdir()
    output = parent / "must-not-return"
    moved = tmp_path / "final-parent-moved"
    original_snapshot = prepare._published_output_snapshot
    calls = 0

    def replacing_snapshot(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            parent.rename(moved)
            parent.mkdir()
            (moved / output.name).rename(parent / output.name)
        return original_snapshot(**kwargs)

    monkeypatch.setattr(
        prepare,
        "_published_output_snapshot",
        replacing_snapshot,
    )

    with pytest.raises(prepare.EdaccPreparationError):
        prepare.prepare_edacc_conversation_fixture(
            source_root=root,
            archive_path=archive,
            output_dir=output,
            accepted_terms=prepare.REQUIRED_TERMS,
            test_source_injection=injection,
        )
    assert calls == 3
    assert output.is_dir()


def test_rejects_pcm_mutation_after_final_snapshot_read(
    tmp_path,
    monkeypatch,
):
    root = _source_tree(tmp_path)
    archive = _archive(tmp_path, root)
    injection = _injection(archive)
    output = tmp_path / "mutated-after-read"
    original_verify = prepare.verify_corpus_snapshot
    calls = 0

    def mutating_verify(corpus):
        nonlocal calls
        calls += 1
        original_verify(corpus)
        if calls == 3:
            _mutate_first_byte(corpus.cases[0].source_path)

    monkeypatch.setattr(prepare, "verify_corpus_snapshot", mutating_verify)

    with pytest.raises(prepare.EdaccPreparationError):
        prepare.prepare_edacc_conversation_fixture(
            source_root=root,
            archive_path=archive,
            output_dir=output,
            accepted_terms=prepare.REQUIRED_TERMS,
            test_source_injection=injection,
        )
    assert calls == 3
    assert output.is_dir()


def _new_archive_extraction_request(tmp_path: Path):
    archive_source = _source_tree(tmp_path / "archive-input")
    archive = _archive(tmp_path, archive_source)
    source_parent = tmp_path / "private-extraction"
    source_parent.mkdir(mode=0o700)
    return (
        source_parent,
        source_parent / prepare.ARCHIVE_ROOT,
        archive,
        tmp_path / "must-not-publish",
        _injection(archive),
    )


@pytest.mark.parametrize("replacement_phase", ("during", "after"))
def test_extraction_rejects_source_parent_replacement_at_consume_handoff(
    tmp_path,
    monkeypatch,
    replacement_phase,
):
    source_parent, source_root, archive, output, injection = (
        _new_archive_extraction_request(tmp_path)
    )
    moved_parent = tmp_path / f"original-parent-{replacement_phase}"
    canary = b"replacement parent must stay untouched\n"
    replacement_happened = False

    def replace_parent() -> None:
        nonlocal replacement_happened
        source_parent.rename(moved_parent)
        source_parent.mkdir(mode=0o700)
        (source_parent / "canary").write_bytes(canary)
        replacement_happened = True

    if replacement_phase == "during":
        original_consume_file = prepare._consume_archive_file

        def replacing_consume_file(stream, member, *, tree):
            result = original_consume_file(stream, member, tree=tree)
            if not replacement_happened:
                replace_parent()
            return result

        monkeypatch.setattr(
            prepare,
            "_consume_archive_file",
            replacing_consume_file,
        )
    else:
        original_consume = prepare._consume_archive

        def replacing_consume(*args, **kwargs):
            snapshot = original_consume(*args, **kwargs)
            replace_parent()
            return snapshot

        monkeypatch.setattr(prepare, "_consume_archive", replacing_consume)

    with pytest.raises(prepare.EdaccPreparationError):
        prepare.prepare_edacc_conversation_fixture(
            source_root=source_root,
            archive_path=archive,
            output_dir=output,
            accepted_terms=prepare.REQUIRED_TERMS,
            test_source_injection=injection,
        )

    assert replacement_happened is True
    assert (source_parent / "canary").read_bytes() == canary
    assert (moved_parent / prepare.ARCHIVE_ROOT).is_dir()
    assert not output.exists()


@pytest.mark.parametrize("target", ("root", "data", "dev", "test"))
def test_extraction_rejects_directory_replacement_before_layout_validation(
    tmp_path,
    monkeypatch,
    target,
):
    _source_parent, source_root, archive, output, injection = (
        _new_archive_extraction_request(tmp_path)
    )
    original_validate = prepare._validate_complete_archive_source_layout
    moved = tmp_path / f"original-{target}"
    canary = b"replacement directory must stay untouched\n"
    replacement_happened = False

    def replacing_validate(root, snapshot):
        nonlocal replacement_happened
        selected = root if target == "root" else root / target
        selected.rename(moved)
        selected.mkdir(mode=0o700)
        (selected / "canary").write_bytes(canary)
        replacement_happened = True
        return original_validate(root, snapshot)

    monkeypatch.setattr(
        prepare,
        "_validate_complete_archive_source_layout",
        replacing_validate,
    )

    with pytest.raises(prepare.EdaccPreparationError):
        prepare.prepare_edacc_conversation_fixture(
            source_root=source_root,
            archive_path=archive,
            output_dir=output,
            accepted_terms=prepare.REQUIRED_TERMS,
            test_source_injection=injection,
        )

    replacement = source_root if target == "root" else source_root / target
    assert replacement_happened is True
    assert (replacement / "canary").read_bytes() == canary
    assert moved.is_dir()
    assert not output.exists()


def test_extraction_rejects_archive_path_replacement_during_raw_consumption(
    tmp_path,
    monkeypatch,
):
    _source_parent, source_root, archive, output, injection = (
        _new_archive_extraction_request(tmp_path)
    )
    original_archive = archive.read_bytes()
    moved_archive = tmp_path / "opened-archive.tar.gz"
    replacement_archive = bytearray(original_archive)
    replacement_archive[-1] ^= 1
    original_read = prepare._CompressedArchiveStream.read
    replacement_happened = False

    def replacing_read(stream, size):
        nonlocal replacement_happened
        payload = original_read(stream, size)
        if payload and not replacement_happened:
            archive.rename(moved_archive)
            archive.write_bytes(replacement_archive)
            archive.chmod(0o600)
            replacement_happened = True
        return payload

    monkeypatch.setattr(prepare._CompressedArchiveStream, "read", replacing_read)

    with pytest.raises(prepare.EdaccPreparationError):
        prepare.prepare_edacc_conversation_fixture(
            source_root=source_root,
            archive_path=archive,
            output_dir=output,
            accepted_terms=prepare.REQUIRED_TERMS,
            test_source_injection=injection,
        )

    assert replacement_happened is True
    assert moved_archive.read_bytes() == original_archive
    assert archive.read_bytes() == bytes(replacement_archive)
    assert source_root.is_dir()
    assert not output.exists()


def test_failed_extraction_retains_private_partial_tree_and_refuses_retry(
    tmp_path,
    monkeypatch,
):
    _source_parent, source_root, archive, output, injection = (
        _new_archive_extraction_request(tmp_path)
    )
    original_consume = prepare._consume_archive
    original_consume_file = prepare._consume_archive_file
    consume_calls = 0
    regular_calls = 0

    def tracking_consume(*args, **kwargs):
        nonlocal consume_calls
        consume_calls += 1
        return original_consume(*args, **kwargs)

    def failing_consume_file(stream, member, *, tree):
        nonlocal regular_calls
        regular_calls += 1
        if regular_calls == 2:
            raise prepare.EdaccPreparationError()
        return original_consume_file(stream, member, tree=tree)

    monkeypatch.setattr(prepare, "ARCHIVE_SIZE_BYTES", injection.archive_size_bytes)
    monkeypatch.setattr(prepare, "ARCHIVE_MD5", injection.archive_md5)
    monkeypatch.setattr(prepare, "_validate_catalog_contract", lambda: None)
    monkeypatch.setattr(
        prepare,
        "_validate_fixture_source_contract",
        lambda source, *, dataset: None,
    )
    monkeypatch.setattr(prepare, "_consume_archive", tracking_consume)
    monkeypatch.setattr(
        prepare,
        "_consume_archive_file",
        failing_consume_file,
    )

    def run_production_attempt() -> None:
        prepare.prepare_edacc_conversation_fixture(
            source_root=source_root,
            archive_path=archive,
            output_dir=output,
            accepted_terms=prepare.REQUIRED_TERMS,
        )

    with pytest.raises(prepare.EdaccPreparationError):
        run_production_attempt()

    retained_files = tuple(
        path.relative_to(source_root)
        for path in source_root.rglob("*")
        if path.is_file()
    )
    assert consume_calls == 1
    assert regular_calls == 2
    assert len(retained_files) == 1
    assert source_root.stat().st_mode & 0o777 == 0o700
    assert all(
        path.stat().st_mode & 0o777 == 0o700
        for path in (source_root / "data", source_root / "dev", source_root / "test")
    )
    assert not output.exists()

    with pytest.raises(prepare.EdaccPreparationError):
        run_production_attempt()

    assert consume_calls == 1
    assert regular_calls == 2
    assert tuple(
        path.relative_to(source_root)
        for path in source_root.rglob("*")
        if path.is_file()
    ) == retained_files
    assert not output.exists()


def test_extraction_rejects_persisted_member_mutation_before_readback(
    tmp_path,
    monkeypatch,
):
    _source_parent, source_root, archive, output, injection = (
        _new_archive_extraction_request(tmp_path)
    )
    original_lseek = prepare.os.lseek
    mutated = False

    def mutating_lseek(descriptor, offset, whence):
        nonlocal mutated
        if not mutated:
            first = prepare.os.pread(descriptor, 1, 0)
            assert first
            assert prepare.os.pwrite(
                descriptor,
                bytes((first[0] ^ 1,)),
                0,
            ) == 1
            mutated = True
        return original_lseek(descriptor, offset, whence)

    monkeypatch.setattr(prepare.os, "lseek", mutating_lseek)

    with pytest.raises(prepare.EdaccPreparationError):
        prepare.prepare_edacc_conversation_fixture(
            source_root=source_root,
            archive_path=archive,
            output_dir=output,
            accepted_terms=prepare.REQUIRED_TERMS,
            test_source_injection=injection,
        )

    assert mutated is True
    assert source_root.is_dir()
    assert not output.exists()


@pytest.mark.parametrize("target", ("root-metadata", "unselected-audio"))
def test_complete_extracted_layout_is_rebound_after_publication(
    tmp_path,
    monkeypatch,
    target,
):
    _source_parent, source_root, archive, output, injection = (
        _new_archive_extraction_request(tmp_path)
    )
    selected = (
        source_root / "evaluate.sh"
        if target == "root-metadata"
        else source_root / "data" / f"{_DEV_RECORDINGS[0]}.wav"
    )
    original_publish = prepare._publish_private_corpus_bound
    mutated = False

    def mutating_publish(**kwargs):
        nonlocal mutated
        result = original_publish(**kwargs)
        _mutate_first_byte(selected)
        mutated = True
        return result

    monkeypatch.setattr(
        prepare,
        "_publish_private_corpus_bound",
        mutating_publish,
    )

    with pytest.raises(prepare.EdaccPreparationError):
        prepare.prepare_edacc_conversation_fixture(
            source_root=source_root,
            archive_path=archive,
            output_dir=output,
            accepted_terms=prepare.REQUIRED_TERMS,
            test_source_injection=injection,
        )

    assert mutated is True
    assert output.is_dir()
