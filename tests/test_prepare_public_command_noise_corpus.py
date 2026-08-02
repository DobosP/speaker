from __future__ import annotations

from array import array
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct
import tarfile
import wave
import zipfile

import pytest

from tools import prepare_public_command_noise_corpus as prepare


_UNKNOWN_WORDS = (
    "blood",
    "riot",
    "bond",
    "damp",
    "ripe",
    "bright",
    "dock",
    "alright",
    "upward",
    "gown",
    "rifle",
    "jointly",
    "buyer",
    "sleek",
    "perverse",
    "because",
    "coast",
    "orange",
    "stomp",
    "peace",
    "nest",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _wav_bytes(*, channels: int, frames: int = 160, sample_rate: int = 16_000) -> bytes:
    output = io.BytesIO()
    values = array("h")
    for index in range(frames):
        clean = 400 if index % 2 else -400
        if channels == 1:
            values.append(clean)
        else:
            values.extend((clean, 100 if index % 2 else -100))
    with wave.open(output, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(values.tobytes())
    return output.getvalue()


def _tar_add(
    archive: tarfile.TarFile,
    name: str,
    payload: bytes | None,
    *,
    symlink: bool = False,
) -> None:
    info = tarfile.TarInfo(name)
    info.mtime = 1
    if payload is None:
        info.type = tarfile.DIRTYPE
        info.mode = 0o750
        archive.addfile(info)
    elif symlink:
        info.type = tarfile.SYMTYPE
        info.linkname = "./README.md"
        info.mode = 0o640
        archive.addfile(info)
    else:
        info.size = len(payload)
        info.mode = 0o640
        archive.addfile(info, io.BytesIO(payload))


def _write_gsc(
    root: Path,
    *,
    reuse_speaker: bool = False,
    invalid_wav: str | None = None,
    unsafe_member: str | None = None,
) -> tuple[Path, prepare.SpeechCommandsArchive]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "speech-commands-test.tar.gz"
    readme = b"synthetic speech commands\n"
    license_payload = b"CC-BY-4.0\n"
    entries: list[tuple[str, bytes]] = []
    speaker_index = 1
    for command in prepare.COMMANDS:
        for _slot in range(2):
            speaker = 1 if reuse_speaker else speaker_index
            channels = 2 if invalid_wav == "stereo" and speaker_index == 1 else 1
            rate = 8_000 if invalid_wav == "rate" and speaker_index == 1 else 16_000
            entries.append(
                (
                    f"./{command}/{speaker:08x}_nohash_0.wav",
                    _wav_bytes(channels=channels, sample_rate=rate),
                )
            )
            speaker_index += 1
    for _slot in range(5):
        speaker = 1 if reuse_speaker else speaker_index
        source_word = _UNKNOWN_WORDS[_slot]
        entries.append(
            (
                f"./_unknown_/{source_word}_{speaker:08x}_nohash_0.wav.wav",
                _wav_bytes(channels=1),
            )
        )
        speaker_index += 1
    for index in range(5):
        entries.append((f"./_silence_/{index}.wav", _wav_bytes(channels=1)))

    labels = (*prepare.COMMANDS, "_unknown_", "_silence_")
    with tarfile.open(path, "w:gz") as archive:
        _tar_add(archive, ".", None)
        _tar_add(archive, "./README.md", readme)
        for label in labels:
            _tar_add(archive, f"./{label}", None)
            for name, payload in entries:
                if name.startswith(f"./{label}/"):
                    _tar_add(archive, name, payload)
        _tar_add(archive, "./LICENSE", license_payload)
        if unsafe_member == "path":
            _tar_add(archive, "../escape.wav", _wav_bytes(channels=1))
        elif unsafe_member == "symlink":
            _tar_add(archive, "./yes/link.wav", b"", symlink=True)
    os.chmod(path, 0o600)
    raw = path.read_bytes()
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
    contract = prepare.SpeechCommandsArchive(
        filename=path.name,
        size_bytes=len(raw),
        sha256=_sha256(raw),
        member_count=len(members),
        file_count=sum(not member.isdir() for member in members),
        uncompressed_bytes=sum(member.size for member in members),
        readme_size_bytes=len(readme),
        readme_sha256=_sha256(readme),
        license_size_bytes=len(license_payload),
        license_sha256=_sha256(license_payload),
    )
    return path, contract


def _eccc_csv(*, invalid_label: bool = False, omit_negative: bool = False) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(prepare._ECCC_HEADERS)
    rows: list[tuple[str, str]] = [
        (command, "word") for command in prepare.COMMANDS[:6]
    ]
    negative_count = 20 if omit_negative else 21
    rows.extend(
        (
            _UNKNOWN_WORDS[index],
            prepare.COMMANDS[index % len(prepare.COMMANDS)],
        )
        for index in range(negative_count)
    )
    for index, (target, confusion) in enumerate(rows, start=1000):
        if invalid_label and index == 1000:
            target = target.upper()
        writer.writerow(
            (
                str(index),
                "80",
                "SSN",
                str(index * 10),
                "-6.0",
                f"s{index % 4 + 1}",
                target,
                "raw",
                "response",
                "3",
                "2 1",
                confusion,
                "2",
                "T",
                "t",
                "1",
                "C",
                "c",
                "1",
                "1",
            )
        )
    return output.getvalue().encode()


def _zip_directory(archive: zipfile.ZipFile, name: str) -> None:
    info = zipfile.ZipInfo(name)
    info.external_attr = (stat.S_IFDIR | 0o755) << 16
    archive.writestr(info, b"")


def _zip_file(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name)
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, payload)


def _write_eccc(
    root: Path,
    *,
    invalid_label: bool = False,
    omit_negative: bool = False,
    invalid_wav: str | None = None,
    unsafe_member: str | None = None,
) -> tuple[Path, prepare.EcccArchive]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "EnglishCCC-test.zip"
    csv_payload = _eccc_csv(
        invalid_label=invalid_label,
        omit_negative=omit_negative,
    )
    readme = b"synthetic English CCC\n"
    license_payload = b"CC-BY-4.0\n"
    row_count = len(csv_payload.decode().splitlines()) - 1
    with zipfile.ZipFile(path, "w", allowZip64=False) as archive:
        for directory in sorted(prepare._ECCC_DIRECTORIES):
            _zip_directory(archive, directory)
        fixed = {
            prepare._ECCC_CSV: csv_payload,
            prepare._ECCC_README: readme,
            prepare._ECCC_LICENSE: license_payload,
            f"{prepare._ECCC_ROOT}/confusionWavs/ReadMe.txt": b"readme\n",
            f"{prepare._ECCC_ROOT}/maskerWavs/BAB4.txt": b"masker\n",
            f"{prepare._ECCC_ROOT}/maskerWavs/BAB4.wav": b"masker-bab4",
            f"{prepare._ECCC_ROOT}/maskerWavs/BMN3.wav": b"masker-bmn3",
            f"{prepare._ECCC_ROOT}/maskerWavs/ReadMe.txt": b"readme\n",
            f"{prepare._ECCC_ROOT}/maskerWavs/SSN.wav": b"masker-ssn",
        }
        for name, payload in fixed.items():
            _zip_file(archive, name, payload)
        channels = 1 if invalid_wav == "mono" else 2
        rate = 8_000 if invalid_wav == "rate" else 16_000
        for index in range(1000, 1000 + row_count):
            _zip_file(
                archive,
                f"{prepare._ECCC_ROOT}/confusionWavs/T_{index}.wav",
                _wav_bytes(channels=channels, frames=6480, sample_rate=rate),
            )
        if unsafe_member == "path":
            _zip_file(archive, "../escape", b"unsafe")
        elif unsafe_member == "symlink":
            info = zipfile.ZipInfo(f"{prepare._ECCC_ROOT}/maskerWavs/link")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, b"target")
    os.chmod(path, 0o600)
    raw = path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
    contract = prepare.EcccArchive(
        filename=path.name,
        size_bytes=len(raw),
        sha256=_sha256(raw),
        member_count=len(infos),
        file_count=sum(not info.is_dir() for info in infos),
        uncompressed_bytes=sum(info.file_size for info in infos),
        csv_rows=row_count,
        csv_size_bytes=len(csv_payload),
        csv_sha256=_sha256(csv_payload),
        readme_size_bytes=len(readme),
        readme_sha256=_sha256(readme),
        license_size_bytes=len(license_payload),
        license_sha256=_sha256(license_payload),
    )
    return path, contract


def _sources(
    tmp_path: Path,
    *,
    reuse_speaker: bool = False,
    gsc_invalid_wav: str | None = None,
    gsc_unsafe: str | None = None,
    eccc_invalid_label: bool = False,
    eccc_omit_negative: bool = False,
    eccc_invalid_wav: str | None = None,
    eccc_unsafe: str | None = None,
) -> tuple[Path, Path, prepare.TestCommandNoiseInjection]:
    gsc_path, gsc = _write_gsc(
        tmp_path / "gsc",
        reuse_speaker=reuse_speaker,
        invalid_wav=gsc_invalid_wav,
        unsafe_member=gsc_unsafe,
    )
    eccc_path, eccc = _write_eccc(
        tmp_path / "eccc",
        invalid_label=eccc_invalid_label,
        omit_negative=eccc_omit_negative,
        invalid_wav=eccc_invalid_wav,
        unsafe_member=eccc_unsafe,
    )
    injection = prepare.TestCommandNoiseInjection(
        speech_commands=gsc,
        eccc=eccc,
        fixture_id="synthetic-command-noise-v1",
    )
    return gsc_path, eccc_path, injection


def _run(
    tmp_path: Path,
    *,
    output_name: str = "prepared",
    **source_options: object,
) -> prepare.PreparedCommandNoiseCorpus:
    gsc_path, eccc_path, injection = _sources(tmp_path, **source_options)
    return prepare.prepare_public_command_noise_corpus(
        speech_commands_archive=gsc_path,
        eccc_archive=eccc_path,
        output_dir=tmp_path / output_name,
        accepted_terms=prepare.REQUIRED_TERMS,
        test_injection=injection,
    )


def test_recipe_lock_is_self_digested_and_pins_official_sources():
    raw = prepare.LOCK_PATH.read_bytes()
    value = json.loads(raw)
    declared = value.pop("recipe_sha256")
    assert declared == prepare.EXPECTED_RECIPE_SHA256
    assert prepare._digest_json(value) == declared
    assert value["totals"] == {
        "sources": 2,
        "cases": 57,
        "command_positive": 26,
        "command_negative": 26,
        "assertions": {"transcript": 47, "speech_negative": 5, "silence": 5},
    }
    gsc, eccc = value["sources"]
    assert (gsc["size_bytes"], gsc["sha256"]) == (
        112_563_277,
        "cc2a00c1147c2254e9be3fa0f779d8c17421dc349b86366567a8edfa9acd51df",
    )
    assert (eccc["size_bytes"], eccc["sha256"]) == (
        173_138_764,
        "dc3a1722d9737f08b8efcbee835cbe711dd8f2267441d1489b058c7b791ae92e",
    )
    assert value["privacy"]["selected_public_eccc_row_ids_committed_in_matrix"]
    assert not value["privacy"]["selected_archive_member_names_committed_in_lock"]


def test_catalog_drift_is_converted_to_detail_free_preparation_error(monkeypatch):
    recipe = prepare._load_recipe_lock()

    def fail(_dataset_id):
        raise prepare.MatrixError("sensitive catalog detail")

    monkeypatch.setattr(prepare, "dataset_by_id", fail)
    with pytest.raises(prepare.CommandNoisePreparationError):
        prepare._source_contract(recipe, None)


def test_prepare_publishes_exact_command_semantics_private_and_deterministic(tmp_path):
    gsc_path, eccc_path, injection = _sources(tmp_path)
    first = prepare.prepare_public_command_noise_corpus(
        speech_commands_archive=gsc_path,
        eccc_archive=eccc_path,
        output_dir=tmp_path / "first",
        accepted_terms=prepare.REQUIRED_TERMS,
        test_injection=injection,
    )
    second = prepare.prepare_public_command_noise_corpus(
        speech_commands_archive=gsc_path,
        eccc_archive=eccc_path,
        output_dir=tmp_path / "second",
        accepted_terms=prepare.REQUIRED_TERMS,
        test_injection=injection,
    )

    assert not first.production_sources
    assert first.corpus.schema_version == 4
    assert first.corpus.digest == second.corpus.digest
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.source_set_sha256 == second.source_set_sha256
    assert [case.case_id for case in first.corpus.cases] == [
        *(f"gsc-case-{index:03d}" for index in range(1, 31)),
        *(f"eccc-case-{index:03d}" for index in range(1, 28)),
    ]
    assertions = [case.assertion for case in first.corpus.cases]
    assert assertions.count("transcript") == 47
    assert assertions.count("speech_negative") == 5
    assert assertions.count("silence") == 5
    assert sum(bool(case.commands) for case in first.corpus.cases) == 26

    gsc_unknown = [
        case for case in first.corpus.cases if "speech-negative" in case.tags
    ]
    assert len(gsc_unknown) == 5
    assert all(
        case.expected_text == ""
        and case.commands == ()
        and case.forbidden_commands == prepare.COMMANDS
        for case in gsc_unknown
    )
    eccc_negative = [
        case for case in first.corpus.cases if "command-negative" in case.tags
    ]
    assert len(eccc_negative) == 26
    assert sum("speech-negative" in case.tags for case in eccc_negative) == 5
    eccc_negative = [case for case in eccc_negative if "eccc" in case.tags]
    assert len(eccc_negative) == 21
    assert all(
        case.expected_text
        and case.assertion == "transcript"
        and case.commands == ()
        and len(case.forbidden_commands) == 1
        for case in eccc_negative
    )
    positives = [case for case in first.corpus.cases if case.commands]
    assert all(
        case.expected_text == case.commands[0]
        and len(case.forbidden_commands) == 9
        and set(case.commands).isdisjoint(case.forbidden_commands)
        for case in positives
    )
    first_eccc = first.corpus.cases[30]
    assert struct.unpack_from("<f", first_eccc.audio_bytes)[0] == pytest.approx(
        -500.0 / 32768.0
    )

    receipt_path = first.corpus.path.parent / "preparation-receipt.json"
    receipt_raw = receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)
    assert _sha256(receipt_raw) == first.receipt_sha256
    assert receipt["counts"] == {
        "sources": 2,
        "cases": 57,
        "command_positive": 26,
        "command_negative": 26,
        "assertions": {"transcript": 47, "speech_negative": 5, "silence": 5},
    }
    assert receipt["accepted_terms"] == ["CC-BY-4.0"]
    rendered = receipt_raw.decode()
    for forbidden in (
        str(tmp_path),
        "_nohash_",
        "T_1000",
        "blood",
        "speaker_id",
        "member_name",
    ):
        assert forbidden not in rendered
    assert stat.S_IMODE(first.corpus.path.parent.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in first.corpus.path.parent.iterdir()
    )


@pytest.mark.parametrize("kind", ["path", "symlink"])
def test_rejects_unsafe_tar_members_even_when_archive_pin_matches(tmp_path, kind):
    with pytest.raises(prepare.CommandNoisePreparationError):
        _run(tmp_path, gsc_unsafe=kind)
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("kind", ["path", "symlink"])
def test_rejects_unsafe_zip_members_even_when_archive_pin_matches(tmp_path, kind):
    with pytest.raises(prepare.CommandNoisePreparationError):
        _run(tmp_path, eccc_unsafe=kind)
    assert not (tmp_path / "prepared").exists()


def test_tar_member_limit_is_enforced_during_progressive_scan(tmp_path):
    path = tmp_path / "oversized-members.tar.gz"
    readme = b"readme"
    license_payload = b"license"
    wav = _wav_bytes(channels=1, frames=1)
    labels = (*prepare.COMMANDS, "_unknown_", "_silence_")
    with tarfile.open(path, "w:gz") as archive:
        _tar_add(archive, ".", None)
        for label in labels:
            _tar_add(archive, f"./{label}", None)
        _tar_add(archive, "./README.md", readme)
        _tar_add(archive, "./LICENSE", license_payload)
        # 15 fixed entries + 5,986 valid files = 6,001 members.
        for index in range(5_986):
            _tar_add(
                archive,
                f"./yes/00000001_nohash_{index}.wav",
                wav,
            )
    os.chmod(path, 0o600)
    raw = path.read_bytes()
    contract = prepare.SpeechCommandsArchive(
        filename=path.name,
        size_bytes=len(raw),
        sha256=_sha256(raw),
        member_count=prepare._MAX_GSC_MEMBERS,
        file_count=5_988,
        uncompressed_bytes=len(readme) + len(license_payload) + 5_986 * len(wav),
        readme_size_bytes=len(readme),
        readme_sha256=_sha256(readme),
        license_size_bytes=len(license_payload),
        license_sha256=_sha256(license_payload),
    )
    with pytest.raises(prepare.CommandNoisePreparationError):
        prepare._prepare_gsc(path, contract)


def test_zip_member_limit_is_bounded_before_zipfile_materialization(tmp_path):
    path = tmp_path / "oversized-members.zip"
    with zipfile.ZipFile(path, "w", allowZip64=False) as archive:
        for index in range(prepare._MAX_ECCC_MEMBERS + 1):
            _zip_file(archive, f"entry-{index:04d}", b"x")
    raw = path.read_bytes()
    with path.open("rb") as handle, pytest.raises(
        prepare.CommandNoisePreparationError
    ):
        prepare._preflight_zip_central_directory(
            handle,
            archive_bytes=len(raw),
            expected_members=prepare._MAX_ECCC_MEMBERS,
        )


def test_rejects_label_tamper_and_selection_insufficiency(tmp_path):
    with pytest.raises(prepare.CommandNoisePreparationError):
        _run(tmp_path / "label", eccc_invalid_label=True)
    with pytest.raises(prepare.CommandNoisePreparationError):
        _run(tmp_path / "rows", eccc_omit_negative=True)
    with pytest.raises(prepare.CommandNoisePreparationError):
        _run(tmp_path / "speakers", reuse_speaker=True)


@pytest.mark.parametrize(
    ("source", "error"),
    [("gsc", "stereo"), ("gsc", "rate"), ("eccc", "mono"), ("eccc", "rate")],
)
def test_rejects_non_pcm16k_channel_geometry(tmp_path, source, error):
    options = (
        {"gsc_invalid_wav": error}
        if source == "gsc"
        else {"eccc_invalid_wav": error}
    )
    with pytest.raises(prepare.CommandNoisePreparationError):
        _run(tmp_path, **options)


def test_rejects_non_private_source_mode_and_source_tamper(tmp_path):
    gsc_path, eccc_path, injection = _sources(tmp_path)
    os.chmod(gsc_path, 0o640)
    with pytest.raises(prepare.CommandNoisePreparationError):
        prepare.prepare_public_command_noise_corpus(
            speech_commands_archive=gsc_path,
            eccc_archive=eccc_path,
            output_dir=tmp_path / "mode-output",
            accepted_terms=prepare.REQUIRED_TERMS,
            test_injection=injection,
        )
    os.chmod(gsc_path, 0o600)
    with gsc_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(prepare.CommandNoisePreparationError):
        prepare.prepare_public_command_noise_corpus(
            speech_commands_archive=gsc_path,
            eccc_archive=eccc_path,
            output_dir=tmp_path / "tamper-output",
            accepted_terms=prepare.REQUIRED_TERMS,
            test_injection=injection,
        )


def test_rejects_same_size_archive_mutation_during_preparation(tmp_path, monkeypatch):
    gsc_path, eccc_path, injection = _sources(tmp_path)
    original = prepare._speaker_first
    mutated = False

    def mutate_then_select(*args, **kwargs):
        nonlocal mutated
        if not mutated:
            with gsc_path.open("r+b", buffering=0) as handle:
                handle.seek(4)  # Gzip MTIME: content-neutral for decompression.
                before = handle.read(1)
                handle.seek(4)
                handle.write(bytes((before[0] ^ 0x01,)))
                os.fsync(handle.fileno())
            mutated = True
        return original(*args, **kwargs)

    monkeypatch.setattr(prepare, "_speaker_first", mutate_then_select)
    with pytest.raises(prepare.CommandNoisePreparationError):
        prepare.prepare_public_command_noise_corpus(
            speech_commands_archive=gsc_path,
            eccc_archive=eccc_path,
            output_dir=tmp_path / "raced-output",
            accepted_terms=prepare.REQUIRED_TERMS,
            test_injection=injection,
        )
    assert mutated
    assert not (tmp_path / "raced-output").exists()


def test_rejects_raw_source_inside_recognized_git_ancestor(tmp_path):
    gsc_path, _eccc_path, injection = _sources(tmp_path)
    marker = gsc_path.parent / ".git"
    marker.mkdir()
    (marker / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    with pytest.raises(prepare.CommandNoisePreparationError):
        with prepare._open_private_archive(
            gsc_path,
            filename=injection.speech_commands.filename,
            size_bytes=injection.speech_commands.size_bytes,
            sha256=injection.speech_commands.sha256,
        ):
            pytest.fail("unsafe archive was opened")


@pytest.mark.parametrize("accepted", [frozenset(), frozenset({"Apache-2.0"})])
def test_rejects_missing_or_wrong_license_acceptance(tmp_path, accepted):
    gsc_path, eccc_path, injection = _sources(tmp_path)
    with pytest.raises(prepare.CommandNoisePreparationError):
        prepare.prepare_public_command_noise_corpus(
            speech_commands_archive=gsc_path,
            eccc_archive=eccc_path,
            output_dir=tmp_path / "terms-output",
            accepted_terms=accepted,
            test_injection=injection,
        )
    assert not (tmp_path / "terms-output").exists()


def test_rejects_stale_recipe_inside_git_and_overwrite(tmp_path):
    gsc_path, eccc_path, injection = _sources(tmp_path)
    stale = tmp_path / "stale.lock.json"
    stale.write_bytes(prepare.LOCK_PATH.read_bytes().replace(b"Google", b"google", 1))
    with pytest.raises(prepare.CommandNoisePreparationError):
        prepare.prepare_public_command_noise_corpus(
            speech_commands_archive=gsc_path,
            eccc_archive=eccc_path,
            output_dir=tmp_path / "stale-output",
            accepted_terms=prepare.REQUIRED_TERMS,
            test_injection=injection,
            recipe_lock_path=stale,
        )
    with pytest.raises(prepare.CommandNoisePreparationError):
        prepare.prepare_public_command_noise_corpus(
            speech_commands_archive=gsc_path,
            eccc_archive=eccc_path,
            output_dir=Path.cwd() / "must-stay-outside-git",
            accepted_terms=prepare.REQUIRED_TERMS,
            test_injection=injection,
        )
    result = prepare.prepare_public_command_noise_corpus(
        speech_commands_archive=gsc_path,
        eccc_archive=eccc_path,
        output_dir=tmp_path / "once",
        accepted_terms=prepare.REQUIRED_TERMS,
        test_injection=injection,
    )
    before = result.corpus.path.read_bytes()
    with pytest.raises(prepare.CommandNoisePreparationError):
        prepare.prepare_public_command_noise_corpus(
            speech_commands_archive=gsc_path,
            eccc_archive=eccc_path,
            output_dir=tmp_path / "once",
            accepted_terms=prepare.REQUIRED_TERMS,
            test_injection=injection,
        )
    assert result.corpus.path.read_bytes() == before


def test_cli_failure_is_aggregate_only(tmp_path, capsys):
    gsc_path, eccc_path, _injection = _sources(tmp_path)
    output = tmp_path / "cli-output"
    assert (
        prepare.main(
            [
                "--speech-commands-archive",
                str(gsc_path),
                "--eccc-archive",
                str(eccc_path),
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )
    rendered = capsys.readouterr().out
    assert json.loads(rendered) == prepare._SAFE_ERROR
    assert str(tmp_path) not in rendered
    assert not output.exists()
