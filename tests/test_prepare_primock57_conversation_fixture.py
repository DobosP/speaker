from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys

import numpy as np
import pytest

import tools.prepare_primock57_conversation_fixture as subject


_AUDIO_A = "audio/day1_consultation01_doctor.wav"
_AUDIO_B = "audio/day1_consultation01_patient.wav"
_GRID_A = "transcripts/day1_consultation01_doctor.TextGrid"
_GRID_B = "transcripts/day1_consultation01_patient.TextGrid"
_LICENSE = "LICENSE.md"
_DURATION = Decimal("8")


@dataclass(frozen=True)
class _SyntheticSource:
    root: Path
    payloads: dict[str, bytes]
    references: tuple[str, ...]
    injection: subject.TestSourceInjection


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _wav(*, samples: int, value: int) -> bytes:
    pcm = struct.pack("<h", value) * samples
    return (
        b"RIFF"
        + struct.pack("<I", len(pcm) + 36)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 16_000, 32_000, 2, 16)
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )


def _textgrid(
    *,
    tier_name: str,
    duration: Decimal,
    boundaries: tuple[Decimal, ...],
    speech: dict[int, str],
) -> bytes:
    assert boundaries[0] == 0
    assert boundaries[-1] == duration
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0 ",
        f"xmax = {duration} ",
        "tiers? <exists> ",
        "size = 1 ",
        "item []: ",
        "    item [1]:",
        '        class = "IntervalTier" ',
        f'        name = "{tier_name}" ',
        "        xmin = 0 ",
        f"        xmax = {duration} ",
        f"        intervals: size = {len(boundaries) - 1} ",
    ]
    for index, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True),
        start=1,
    ):
        text = speech.get(index, "").replace('"', '""')
        lines.extend(
            [
                f"        intervals [{index}]:",
                f"            xmin = {start} ",
                f"            xmax = {end} ",
                f'            text = "{text}" ',
            ]
        )
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def _write_private(root: Path, relative: str, raw: bytes) -> None:
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_bytes(raw)
    path.chmod(0o600)


def _make_source(tmp_path: Path) -> _SyntheticSource:
    root = tmp_path / "private-source"
    root.mkdir(mode=0o700, parents=True)
    root.chmod(0o700)
    references = (
        "alpha private phrase",
        "omega private phrase",
        "beta private phrase",
        "first role overlap",
        "first second overlap",
        "middle role overlap",
        "middle second overlap",
        "last role overlap",
        "last second overlap",
    )
    boundaries_a = tuple(
        Decimal(value)
        for value in (
            "0",
            "0.5",
            "1.0",
            "2.00003",
            "2.80003",
            "4.0",
            "4.7",
            "6.0",
            "6.6",
            "7.2",
            "7.6",
            "8",
        )
    )
    boundaries_b = tuple(
        Decimal(value)
        for value in (
            "0",
            "1.2",
            "1.7",
            "2.40004",
            "3.10004",
            "4.3",
            "5.0",
            "6.2",
            "6.9",
            "8",
        )
    )
    payloads = {
        _AUDIO_A: _wav(samples=8 * 16_000, value=8_000),
        _AUDIO_B: _wav(samples=8 * 16_000, value=-4_000),
        _GRID_A: _textgrid(
            tier_name="PrivateRoleAlpha",
            duration=_DURATION,
            boundaries=boundaries_a,
            speech={
                2: references[0],
                4: references[3],
                6: references[5],
                8: references[7],
                10: references[1],
            },
        ),
        _GRID_B: _textgrid(
            tier_name="PrivateRoleBeta",
            duration=_DURATION,
            boundaries=boundaries_b,
            speech={
                2: references[2],
                4: references[4],
                6: references[6],
                8: references[8],
            },
        ),
        _LICENSE: b"synthetic CC-BY-4.0 test fixture only\n",
    }
    for relative, raw in payloads.items():
        _write_private(root, relative, raw)
    selected = {
        "isolated": (
            ("role-a", 2, "0.5", "1.0", references[0]),
            ("role-b", 2, "1.2", "1.7", references[2]),
            ("role-a", 10, "7.2", "7.6", references[1]),
        ),
        "overlap": (
            (
                4,
                "2.00003",
                "2.80003",
                references[3],
                4,
                "2.40004",
                "3.10004",
                references[4],
            ),
            (6, "4.0", "4.7", references[5], 6, "4.3", "5.0", references[6]),
            (8, "6.0", "6.6", references[7], 8, "6.2", "6.9", references[8]),
        ),
    }
    isolated = tuple(
        subject.TestIntervalSelection(
            role=role,
            interval_index=index,
            start_seconds=Decimal(start),
            end_seconds=Decimal(end),
            reference_sha256=_sha256(reference.encode("utf-8")),
        )
        for role, index, start, end, reference in selected["isolated"]
    )
    overlap = tuple(
        subject.TestOverlapPairSelection(
            role_a=subject.TestIntervalSelection(
                role="role-a",
                interval_index=index_a,
                start_seconds=Decimal(start_a),
                end_seconds=Decimal(end_a),
                reference_sha256=_sha256(reference_a.encode("utf-8")),
            ),
            role_b=subject.TestIntervalSelection(
                role="role-b",
                interval_index=index_b,
                start_seconds=Decimal(start_b),
                end_seconds=Decimal(end_b),
                reference_sha256=_sha256(reference_b.encode("utf-8")),
            ),
            overlap_start_seconds=max(Decimal(start_a), Decimal(start_b)),
            overlap_end_seconds=min(Decimal(end_a), Decimal(end_b)),
        )
        for (
            index_a,
            start_a,
            end_a,
            reference_a,
            index_b,
            start_b,
            end_b,
            reference_b,
        ) in selected["overlap"]
    )
    injection = subject.TestSourceInjection(
        fixture_id="synthetic-primock57-v1",
        file_pins={
            relative: (len(raw), _sha256(raw)) for relative, raw in payloads.items()
        },
        role_a_duration_seconds=_DURATION,
        role_b_duration_seconds=_DURATION,
        isolated_cases=isolated,
        overlap_cases=overlap,
    )
    return _SyntheticSource(root, payloads, references, injection)


def _assert_private_tree(root: Path) -> None:
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    for path in root.iterdir():
        metadata = path.stat()
        assert metadata.st_uid == root.stat().st_uid
        if path.is_file():
            assert stat.S_IMODE(metadata.st_mode) == 0o600
            assert metadata.st_nlink == 1


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def _overlap_source_contract_from_receipt(receipt: dict[str, object]) -> str:
    return _sha256(
        _canonical(
            {
                "accepted_license": subject.LICENSE_ID,
                "fixture_id": receipt["fixture_id"],
                "lock_recipe_sha256": subject.EXPECTED_RECIPE_SHA256,
                "preparer_files": receipt["preparer_files"],
                "production_evidence": receipt["production_evidence"],
                "selection_sha256": receipt["selection_sha256"],
                "source_files": receipt["source_files"],
            }
        )[:-1]
    )


def _prepare(
    tmp_path: Path,
) -> tuple[
    _SyntheticSource,
    Path,
    Path,
    subject.PreparedPrimock57Fixture,
]:
    source = _make_source(tmp_path)
    isolated = tmp_path / "isolated-output"
    overlap = tmp_path / "overlap-output"
    prepared = subject.prepare_primock57_conversation_fixture(
        source_dir=source.root,
        isolated_output_dir=isolated,
        overlap_output_dir=overlap,
        accepted_license=subject.LICENSE_ID,
        test_source_injection=source.injection,
    )
    return source, isolated, overlap, prepared


def _replace_pin(
    source: _SyntheticSource,
    relative: str,
    raw: bytes,
) -> subject.TestSourceInjection:
    pins = dict(source.injection.file_pins)
    pins[relative] = (len(raw), _sha256(raw))
    return replace(source.injection, file_pins=pins)


def test_publishes_independent_isolated_and_overlap_products(
    tmp_path: Path,
) -> None:
    source, isolated, overlap, prepared = _prepare(tmp_path)

    assert prepared.production_evidence is False
    assert prepared.isolated_corpus.schema_version == 2
    assert [case.case_id for case in prepared.isolated_corpus.cases] == [
        "primock57-isolated-00",
        "primock57-isolated-01",
        "primock57-isolated-02",
    ]
    assert [case.samples for case in prepared.isolated_corpus.cases] == [
        8_000,
        8_000,
        6_400,
    ]
    assert [case.expected_text for case in prepared.isolated_corpus.cases] == [
        source.references[0],
        source.references[2],
        source.references[1],
    ]
    expected_samples = (8_000, -4_000, 8_000)
    for case, sample in zip(
        prepared.isolated_corpus.cases,
        expected_samples,
        strict=True,
    ):
        expected = np.full(
            case.samples,
            np.float32(sample) / np.float32(32_768),
            dtype="<f4",
        ).tobytes()
        assert case.audio_bytes == expected

    loaded_isolated = subject.load_primock57_isolated_bundle(isolated)
    loaded_overlap = subject.load_primock57_overlap_bundle(overlap)
    assert loaded_isolated.corpus.digest == prepared.isolated_corpus.digest
    assert loaded_isolated.receipt_sha256 == prepared.isolated_receipt_sha256
    assert loaded_isolated.production_evidence is False
    assert loaded_overlap.digest == prepared.overlap_bundle.digest
    assert loaded_overlap.receipt_sha256 == prepared.overlap_receipt_sha256
    assert loaded_overlap.production_evidence is False
    assert len(loaded_overlap.cases) == 3
    assert loaded_overlap.path.name == subject.OVERLAP_MANIFEST_FILENAME
    _assert_private_tree(isolated)
    _assert_private_tree(overlap)


def test_overlap_manifest_has_exact_geometry_masking_and_float32_mix(
    tmp_path: Path,
) -> None:
    _source, _isolated, overlap, _prepared = _prepare(tmp_path)
    manifest = json.loads((overlap / subject.OVERLAP_MANIFEST_FILENAME).read_bytes())
    assert manifest["kind"] == "primock57-two-role-overlap-diagnostic-v1"
    assert manifest["schema_version"] == 1
    assert manifest["production_evidence"] is False
    assert manifest["evidence_scope"] == {
        "diagnostic_only": True,
        "natural_conversation_alignment": True,
        "ordinary_wer": False,
        "original_device_mix": False,
        "overlap_metric": "not-implemented",
        "promotion_authority": False,
        "qualification_authority": False,
    }
    expected_geometry = (
        (32_000, 49_601, 6_400, 12_801),
        (64_000, 80_000, 4_800, 11_200),
        (96_000, 110_400, 3_200, 9_600),
    )
    for case, geometry in zip(
        manifest["cases"],
        expected_geometry,
        strict=True,
    ):
        source_start, source_end, overlap_start, overlap_end = geometry
        samples = source_end - source_start
        assert case["envelope"] == {
            "samples": samples,
            "source_end_sample": source_end,
            "source_start_sample": source_start,
        }
        assert case["overlap"] == {
            "relative_end_sample": overlap_end,
            "relative_start_sample": overlap_start,
            "samples": overlap_end - overlap_start,
        }
        arrays = {}
        for key in ("role_a", "role_b"):
            row = case[key]
            raw = (overlap / row["file"]).read_bytes()
            assert _sha256(raw) == row["sha256"]
            array = np.frombuffer(raw, dtype="<f4")
            assert array.size == samples
            assert np.all(array[: row["activity_start_sample"]] == 0)
            assert np.all(array[row["activity_end_sample"] :] == 0)
            arrays[key] = array
        mix_row = case["mix"]
        mix_raw = (overlap / mix_row["file"]).read_bytes()
        expected_mix = np.multiply(
            np.add(arrays["role_a"], arrays["role_b"], dtype=np.float32),
            np.float32(0.5),
            dtype=np.float32,
        )
        assert mix_raw == np.asarray(expected_mix, dtype="<f4").tobytes()
        assert _sha256(mix_raw) == mix_row["sha256"]
        assert mix_row["arithmetic"] == ("float32-add-then-multiply-float32-0.5-v1")


def test_receipts_and_safe_stdout_surfaces_exclude_private_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, isolated, overlap, _prepared = _prepare(tmp_path)
    receipt_raws = (
        (isolated / subject.PREPARATION_RECEIPT_FILENAME).read_text(),
        (overlap / subject.PREPARATION_RECEIPT_FILENAME).read_text(),
    )
    forbidden = (
        str(source.root),
        "PrivateRoleAlpha",
        "PrivateRoleBeta",
        *source.references,
    )
    assert all(value not in raw for raw in receipt_raws for value in forbidden)
    for raw in receipt_raws:
        receipt = json.loads(raw)
        assert receipt["production_evidence"] is False
        assert receipt["privacy"] == {
            "local_paths_in_receipt": False,
            "raw_role_labels_in_receipt": False,
            "transcripts_in_receipt": False,
        }

    code = subject.main([])
    stdout = capsys.readouterr().out
    assert code == 2
    assert all(value not in stdout for value in forbidden)


@pytest.mark.parametrize("accepted_license", ["", "cc-by-4.0", "CC0-1.0"])
def test_rejects_wrong_license_before_publication(
    tmp_path: Path,
    accepted_license: str,
) -> None:
    source = _make_source(tmp_path)
    isolated = tmp_path / "isolated-output"
    overlap = tmp_path / "overlap-output"
    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.prepare_primock57_conversation_fixture(
            source_dir=source.root,
            isolated_output_dir=isolated,
            overlap_output_dir=overlap,
            accepted_license=accepted_license,
            test_source_injection=source.injection,
        )
    assert not isolated.exists()
    assert not overlap.exists()


def test_rejects_source_hash_textgrid_tag_and_wav_contract_tamper(
    tmp_path: Path,
) -> None:
    source = _make_source(tmp_path)

    audio = source.root / _AUDIO_A
    audio.write_bytes(audio.read_bytes() + b"\0")
    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.prepare_primock57_conversation_fixture(
            source_dir=source.root,
            isolated_output_dir=tmp_path / "hash-isolated",
            overlap_output_dir=tmp_path / "hash-overlap",
            accepted_license=subject.LICENSE_ID,
            test_source_injection=source.injection,
        )

    tagged = _make_source(tmp_path / "tagged")
    grid = tagged.root / _GRID_A
    raw_grid = grid.read_bytes().replace(
        b"first role overlap",
        b"<UNKNOWN/> first role overlap",
        1,
    )
    grid.write_bytes(raw_grid)
    grid.chmod(0o600)
    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.prepare_primock57_conversation_fixture(
            source_dir=tagged.root,
            isolated_output_dir=tmp_path / "tag-isolated",
            overlap_output_dir=tmp_path / "tag-overlap",
            accepted_license=subject.LICENSE_ID,
            test_source_injection=_replace_pin(tagged, _GRID_A, raw_grid),
        )

    malformed = _make_source(tmp_path / "wav")
    wav = malformed.root / _AUDIO_A
    raw_wav = bytearray(wav.read_bytes())
    raw_wav[22:24] = struct.pack("<H", 2)
    wav.write_bytes(raw_wav)
    wav.chmod(0o600)
    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.prepare_primock57_conversation_fixture(
            source_dir=malformed.root,
            isolated_output_dir=tmp_path / "wav-isolated",
            overlap_output_dir=tmp_path / "wav-overlap",
            accepted_license=subject.LICENSE_ID,
            test_source_injection=_replace_pin(malformed, _AUDIO_A, bytes(raw_wav)),
        )


def test_rejects_lock_and_reference_contract_tamper_before_publication(
    tmp_path: Path,
) -> None:
    source = _make_source(tmp_path)
    lock = json.loads(subject.DEFAULT_LOCK.read_bytes())
    lock["repository"] = "private/tamper"
    tampered_lock = tmp_path / "tampered-lock.json"
    tampered_lock.write_bytes(_canonical(lock))
    tampered_lock.chmod(0o600)
    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.prepare_primock57_conversation_fixture(
            source_dir=source.root,
            isolated_output_dir=tmp_path / "lock-isolated",
            overlap_output_dir=tmp_path / "lock-overlap",
            accepted_license=subject.LICENSE_ID,
            lock_path=tampered_lock,
            test_source_injection=source.injection,
        )
    assert not (tmp_path / "lock-isolated").exists()
    assert not (tmp_path / "lock-overlap").exists()

    bad_case = replace(
        source.injection.isolated_cases[0],
        reference_sha256="0" * 64,
    )
    injection = replace(
        source.injection,
        isolated_cases=(bad_case, *source.injection.isolated_cases[1:]),
    )
    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.prepare_primock57_conversation_fixture(
            source_dir=source.root,
            isolated_output_dir=tmp_path / "reference-isolated",
            overlap_output_dir=tmp_path / "reference-overlap",
            accepted_license=subject.LICENSE_ID,
            test_source_injection=injection,
        )
    assert not (tmp_path / "reference-isolated").exists()
    assert not (tmp_path / "reference-overlap").exists()


def test_rejects_non_private_or_hardlinked_source_leaf(
    tmp_path: Path,
) -> None:
    source = _make_source(tmp_path)
    target = source.root / _GRID_A
    target.chmod(0o644)
    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.prepare_primock57_conversation_fixture(
            source_dir=source.root,
            isolated_output_dir=tmp_path / "mode-isolated",
            overlap_output_dir=tmp_path / "mode-overlap",
            accepted_license=subject.LICENSE_ID,
            test_source_injection=source.injection,
        )

    non_private_root = _make_source(tmp_path / "non-private-root")
    non_private_root.root.chmod(0o755)
    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.prepare_primock57_conversation_fixture(
            source_dir=non_private_root.root,
            isolated_output_dir=tmp_path / "root-isolated",
            overlap_output_dir=tmp_path / "root-overlap",
            accepted_license=subject.LICENSE_ID,
            test_source_injection=non_private_root.injection,
        )

    linked = _make_source(tmp_path / "linked")
    (tmp_path / "second-link").hardlink_to(linked.root / _LICENSE)
    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.prepare_primock57_conversation_fixture(
            source_dir=linked.root,
            isolated_output_dir=tmp_path / "link-isolated",
            overlap_output_dir=tmp_path / "link-overlap",
            accepted_license=subject.LICENSE_ID,
            test_source_injection=linked.injection,
        )


def test_no_clobber_preserves_existing_output_and_rejects_git_parent(
    tmp_path: Path,
) -> None:
    source = _make_source(tmp_path)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    marker = occupied / "keep"
    marker.write_text("do-not-clobber")
    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.prepare_primock57_conversation_fixture(
            source_dir=source.root,
            isolated_output_dir=occupied,
            overlap_output_dir=tmp_path / "unused-overlap",
            accepted_license=subject.LICENSE_ID,
            test_source_injection=source.injection,
        )
    assert marker.read_text() == "do-not-clobber"

    git_parent = tmp_path / "private-git"
    git_parent.mkdir(mode=0o700)
    (git_parent / ".git").mkdir()
    (git_parent / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.prepare_primock57_conversation_fixture(
            source_dir=source.root,
            isolated_output_dir=git_parent / "isolated",
            overlap_output_dir=git_parent / "overlap",
            accepted_license=subject.LICENSE_ID,
            test_source_injection=source.injection,
        )


def test_overlap_terminal_failure_keeps_isolated_and_partial_without_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source(tmp_path)
    isolated = tmp_path / "isolated-output"
    overlap = tmp_path / "overlap-output"

    def fail_before_terminal(_callback: object) -> None:
        raise subject.Primock57PreparationError()

    monkeypatch.setattr(subject, "_terminal_commit_guard", fail_before_terminal)
    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.prepare_primock57_conversation_fixture(
            source_dir=source.root,
            isolated_output_dir=isolated,
            overlap_output_dir=overlap,
            accepted_license=subject.LICENSE_ID,
            test_source_injection=source.injection,
        )

    assert subject.load_primock57_isolated_bundle(isolated).corpus.schema_version == 2
    assert overlap.exists()
    assert (overlap / subject.OVERLAP_MANIFEST_FILENAME).is_file()
    assert not (overlap / subject.PREPARATION_RECEIPT_FILENAME).exists()


def test_close_time_source_mutation_fails_before_overlap_terminal_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source(tmp_path)
    isolated = tmp_path / "isolated-output"
    overlap = tmp_path / "overlap-output"

    def mutate_then_validate(callback: object) -> None:
        license_path = source.root / _LICENSE
        license_path.write_bytes(license_path.read_bytes() + b" ")
        callback()

    monkeypatch.setattr(subject, "_terminal_commit_guard", mutate_then_validate)
    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.prepare_primock57_conversation_fixture(
            source_dir=source.root,
            isolated_output_dir=isolated,
            overlap_output_dir=overlap,
            accepted_license=subject.LICENSE_ID,
            test_source_injection=source.injection,
        )
    assert subject.load_primock57_isolated_bundle(isolated).corpus.schema_version == 2
    assert not (overlap / subject.PREPARATION_RECEIPT_FILENAME).exists()


def test_no_fallible_bundle_revalidation_runs_after_terminal_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source(tmp_path)
    isolated = tmp_path / "isolated-output"
    overlap = tmp_path / "overlap-output"
    real_link = subject._link_terminal_receipt
    real_verify = subject._verify_directory_binding
    terminal_linked = False

    def link(directory_fd: int, descriptor: int) -> None:
        nonlocal terminal_linked
        real_link(directory_fd, descriptor)
        terminal_linked = True

    def verify(*args: object, **kwargs: object) -> None:
        if terminal_linked:
            raise AssertionError("bundle revalidated after terminal commit")
        real_verify(*args, **kwargs)

    monkeypatch.setattr(subject, "_link_terminal_receipt", link)
    monkeypatch.setattr(subject, "_verify_directory_binding", verify)
    prepared = subject.prepare_primock57_conversation_fixture(
        source_dir=source.root,
        isolated_output_dir=isolated,
        overlap_output_dir=overlap,
        accepted_license=subject.LICENSE_ID,
        test_source_injection=source.injection,
    )
    assert prepared.production_evidence is False
    assert (overlap / subject.PREPARATION_RECEIPT_FILENAME).is_file()


def test_keyboard_interrupt_before_terminal_link_is_precommit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _make_source(tmp_path / "api")
    isolated = tmp_path / "api-isolated"
    overlap = tmp_path / "api-overlap"

    def interrupt_before_terminal(_callback: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        subject,
        "_terminal_commit_guard",
        interrupt_before_terminal,
    )
    with pytest.raises(KeyboardInterrupt):
        subject.prepare_primock57_conversation_fixture(
            source_dir=source.root,
            isolated_output_dir=isolated,
            overlap_output_dir=overlap,
            accepted_license=subject.LICENSE_ID,
            test_source_injection=source.injection,
        )
    assert subject.load_primock57_isolated_bundle(isolated).corpus.schema_version == 2
    assert (overlap / subject.OVERLAP_MANIFEST_FILENAME).is_file()
    assert not (overlap / subject.PREPARATION_RECEIPT_FILENAME).exists()

    cli_source = _make_source(tmp_path / "cli")
    cli_isolated = tmp_path / "cli-isolated"
    cli_overlap = tmp_path / "cli-overlap"
    real_prepare = subject.prepare_primock57_conversation_fixture

    def prepare_with_test_source(**kwargs: object) -> subject.PreparedPrimock57Fixture:
        return real_prepare(
            **kwargs,
            test_source_injection=cli_source.injection,
        )

    monkeypatch.setattr(
        subject,
        "prepare_primock57_conversation_fixture",
        prepare_with_test_source,
    )
    code = subject.main(
        [
            "--source-dir",
            str(cli_source.root),
            "--isolated-output-dir",
            str(cli_isolated),
            "--overlap-output-dir",
            str(cli_overlap),
            "--accept-license",
            subject.LICENSE_ID,
        ]
    )
    assert code == 130
    assert json.loads(capsys.readouterr().out) == subject._SAFE_ERROR
    assert not (cli_overlap / subject.PREPARATION_RECEIPT_FILENAME).exists()


def test_interrupt_after_real_terminal_link_recovers_committed_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source(tmp_path)
    isolated = tmp_path / "isolated-output"
    overlap = tmp_path / "overlap-output"
    real_link = subject._link_terminal_receipt
    linked = False

    def link_then_interrupt(directory_fd: int, descriptor: int) -> None:
        nonlocal linked
        real_link(directory_fd, descriptor)
        linked = True
        raise KeyboardInterrupt

    monkeypatch.setattr(subject, "_link_terminal_receipt", link_then_interrupt)
    prepared = subject.prepare_primock57_conversation_fixture(
        source_dir=source.root,
        isolated_output_dir=isolated,
        overlap_output_dir=overlap,
        accepted_license=subject.LICENSE_ID,
        test_source_injection=source.injection,
    )

    assert linked is True
    assert (overlap / subject.PREPARATION_RECEIPT_FILENAME).is_file()
    reopened = subject.load_primock57_overlap_bundle(overlap)
    assert reopened.digest == prepared.overlap_bundle.digest
    assert reopened.receipt_sha256 == prepared.overlap_receipt_sha256


def test_cli_returns_success_when_stdout_breaks_after_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source(tmp_path)
    isolated = tmp_path / "isolated-output"
    overlap = tmp_path / "overlap-output"
    real_prepare = subject.prepare_primock57_conversation_fixture

    def prepare_with_test_source(**kwargs: object) -> subject.PreparedPrimock57Fixture:
        return real_prepare(
            **kwargs,
            test_source_injection=source.injection,
        )

    def broken_stdout(*_args: object, **_kwargs: object) -> None:
        raise BrokenPipeError

    monkeypatch.setattr(
        subject,
        "prepare_primock57_conversation_fixture",
        prepare_with_test_source,
    )
    monkeypatch.setattr(subject, "print", broken_stdout, raising=False)
    code = subject.main(
        [
            "--source-dir",
            str(source.root),
            "--isolated-output-dir",
            str(isolated),
            "--overlap-output-dir",
            str(overlap),
            "--accept-license",
            subject.LICENSE_ID,
        ]
    )

    assert code == 0
    reopened = subject.load_primock57_overlap_bundle(overlap)
    assert reopened.production_evidence is False
    assert (overlap / subject.PREPARATION_RECEIPT_FILENAME).is_file()


def test_overlap_directory_fsync_precedes_terminal_link_after_all_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source(tmp_path)
    isolated = tmp_path / "isolated-output"
    overlap = tmp_path / "overlap-output"
    real_fsync = subject.os.fsync
    real_link = subject._link_terminal_receipt
    events: list[tuple[str, tuple[int, int], frozenset[str]]] = []

    def recording_fsync(descriptor: int) -> None:
        metadata = subject.os.fstat(descriptor)
        if stat.S_ISDIR(metadata.st_mode):
            events.append(
                (
                    "fsync",
                    (metadata.st_dev, metadata.st_ino),
                    frozenset(subject.os.listdir(descriptor)),
                )
            )
        real_fsync(descriptor)

    def recording_link(directory_fd: int, descriptor: int) -> None:
        metadata = subject.os.fstat(directory_fd)
        events.append(
            (
                "link",
                (metadata.st_dev, metadata.st_ino),
                frozenset(subject.os.listdir(directory_fd)),
            )
        )
        real_link(directory_fd, descriptor)

    monkeypatch.setattr(subject.os, "fsync", recording_fsync)
    monkeypatch.setattr(subject, "_link_terminal_receipt", recording_link)
    prepared = subject.prepare_primock57_conversation_fixture(
        source_dir=source.root,
        isolated_output_dir=isolated,
        overlap_output_dir=overlap,
        accepted_license=subject.LICENSE_ID,
        test_source_injection=source.injection,
    )

    link_index = next(index for index, event in enumerate(events) if event[0] == "link")
    _, overlap_identity, names_at_link = events[link_index]
    prior_overlap_fsyncs = [
        event
        for event in events[:link_index]
        if event[0] == "fsync" and event[1] == overlap_identity
    ]
    assert prior_overlap_fsyncs
    names_at_fsync = prior_overlap_fsyncs[-1][2]
    manifest = json.loads((overlap / subject.OVERLAP_MANIFEST_FILENAME).read_bytes())
    artifact_names = {
        case[key]["file"]
        for case in manifest["cases"]
        for key in ("role_a", "role_b", "mix")
    }
    expected_precommit_names = artifact_names | {subject.OVERLAP_MANIFEST_FILENAME}
    assert names_at_fsync == expected_precommit_names
    assert names_at_link == expected_precommit_names
    assert subject.PREPARATION_RECEIPT_FILENAME not in names_at_fsync
    assert subject.load_primock57_overlap_bundle(overlap).digest == (
        prepared.overlap_bundle.digest
    )


def test_overlap_loader_rejects_same_size_mix_tamper(tmp_path: Path) -> None:
    _source, _isolated, overlap, _prepared = _prepare(tmp_path)
    manifest = json.loads((overlap / subject.OVERLAP_MANIFEST_FILENAME).read_bytes())
    mix = overlap / manifest["cases"][0]["mix"]["file"]
    raw = bytearray(mix.read_bytes())
    raw[0] ^= 1
    mix.write_bytes(raw)
    mix.chmod(0o600)
    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.load_primock57_overlap_bundle(overlap)


def test_isolated_loader_rejects_world_readable_pcm(tmp_path: Path) -> None:
    _source, isolated, _overlap, prepared = _prepare(tmp_path)
    pcm = isolated / prepared.isolated_corpus.cases[0].source_path.name
    pcm.chmod(0o644)

    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.load_primock57_isolated_bundle(isolated)


def test_synthetic_isolated_bundle_cannot_self_recertify_as_production(
    tmp_path: Path,
) -> None:
    _source, isolated, _overlap, _prepared = _prepare(tmp_path)
    receipt_path = isolated / subject.PREPARATION_RECEIPT_FILENAME
    manifest_path = isolated / "corpus.json"
    receipt = json.loads(receipt_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    assert receipt["production_evidence"] is False

    receipt["production_evidence"] = True
    receipt_raw = _canonical(receipt)
    manifest["provenance"]["source_set_sha256"] = _sha256(receipt_raw)
    receipt_path.write_bytes(receipt_raw)
    receipt_path.chmod(0o600)
    manifest_path.write_bytes(_canonical(manifest))
    manifest_path.chmod(0o600)

    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.load_primock57_isolated_bundle(isolated)


def test_synthetic_overlap_bundle_cannot_self_recertify_as_production(
    tmp_path: Path,
) -> None:
    _source, _isolated, overlap, _prepared = _prepare(tmp_path)
    receipt_path = overlap / subject.PREPARATION_RECEIPT_FILENAME
    manifest_path = overlap / subject.OVERLAP_MANIFEST_FILENAME
    receipt = json.loads(receipt_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    assert receipt["production_evidence"] is False

    receipt["production_evidence"] = True
    source_contract = _overlap_source_contract_from_receipt(receipt)
    receipt["source_contract_sha256"] = source_contract
    manifest["production_evidence"] = True
    manifest["source_contract_sha256"] = source_contract
    manifest_raw = _canonical(manifest)
    receipt["manifest"]["bytes"] = len(manifest_raw)
    receipt["manifest"]["sha256"] = _sha256(manifest_raw)
    manifest_path.write_bytes(manifest_raw)
    manifest_path.chmod(0o600)
    receipt_path.write_bytes(_canonical(receipt))
    receipt_path.chmod(0o600)

    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.load_primock57_overlap_bundle(overlap)


def test_overlap_loader_rejects_forged_reference_with_recomputed_hashes(
    tmp_path: Path,
) -> None:
    _source, _isolated, overlap, _prepared = _prepare(tmp_path)
    receipt_path = overlap / subject.PREPARATION_RECEIPT_FILENAME
    manifest_path = overlap / subject.OVERLAP_MANIFEST_FILENAME
    receipt = json.loads(receipt_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    role = manifest["cases"][0]["role_a"]
    role["reference"] = "forged replacement reference"
    role["reference_sha256"] = _sha256(role["reference"].encode("utf-8"))
    manifest_raw = _canonical(manifest)
    receipt["manifest"]["bytes"] = len(manifest_raw)
    receipt["manifest"]["sha256"] = _sha256(manifest_raw)
    manifest_path.write_bytes(manifest_raw)
    manifest_path.chmod(0o600)
    receipt_path.write_bytes(_canonical(receipt))
    receipt_path.chmod(0o600)

    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.load_primock57_overlap_bundle(overlap)


def test_test_injection_cannot_claim_the_production_fixture_id(
    tmp_path: Path,
) -> None:
    source = _make_source(tmp_path)
    injection = replace(source.injection, fixture_id=subject.FIXTURE_ID)
    isolated = tmp_path / "isolated-output"
    overlap = tmp_path / "overlap-output"

    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.prepare_primock57_conversation_fixture(
            source_dir=source.root,
            isolated_output_dir=isolated,
            overlap_output_dir=overlap,
            accepted_license=subject.LICENSE_ID,
            test_source_injection=injection,
        )
    assert not isolated.exists()
    assert not overlap.exists()


def test_synthetic_corpus_does_not_inherit_the_production_lock_digest(
    tmp_path: Path,
) -> None:
    _source, _isolated, _overlap, prepared = _prepare(tmp_path)
    provenance = prepared.isolated_corpus.provenance
    assert provenance is not None
    assert provenance.manifest_sha256 != subject.EXPECTED_RECIPE_SHA256


def test_selection_and_source_contract_bind_each_role_overlap_geometry() -> None:
    reference_a = "first role overlap"
    reference_b = "first second overlap"
    first = (
        subject._Interval(
            "role-a",
            1,
            Decimal("2.0"),
            Decimal("4.0"),
            reference_a,
            reference_a,
        ),
        subject._Interval(
            "role-b",
            1,
            Decimal("3.0"),
            Decimal("5.0"),
            reference_b,
            reference_b,
        ),
    )
    shifted = (
        subject._Interval(
            "role-a",
            1,
            Decimal("2.0"),
            Decimal("3.5"),
            reference_a,
            reference_a,
        ),
        subject._Interval(
            "role-b",
            1,
            Decimal("2.5"),
            Decimal("5.0"),
            reference_b,
            reference_b,
        ),
    )
    first_selection = subject._selection_sha256((), (first,))
    shifted_selection = subject._selection_sha256((), (shifted,))
    source_rows = [
        {"role": "test-source", "sha256": "a" * 64, "size_bytes": 1},
    ]
    preparer_files = ({"path": "test-preparer", "sha256": "b" * 64, "size_bytes": 1},)
    first_contract = subject._source_contract_sha256(
        fixture_id="synthetic-primock57-v1",
        production_evidence=False,
        source_rows=source_rows,
        selection_sha256=first_selection,
        preparer_files=preparer_files,
    )
    shifted_contract = subject._source_contract_sha256(
        fixture_id="synthetic-primock57-v1",
        production_evidence=False,
        source_rows=source_rows,
        selection_sha256=shifted_selection,
        preparer_files=preparer_files,
    )

    assert first_selection != shifted_selection
    assert first_contract != shifted_contract


def test_prelink_rejects_unexpected_leaf_and_mutated_staged_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source(tmp_path / "extra")
    isolated = tmp_path / "extra-isolated"
    overlap = tmp_path / "extra-overlap"

    def add_unexpected(callback: object) -> None:
        unexpected = overlap / "unexpected"
        unexpected.write_bytes(b"foreign")
        unexpected.chmod(0o600)
        callback()

    monkeypatch.setattr(subject, "_terminal_commit_guard", add_unexpected)
    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.prepare_primock57_conversation_fixture(
            source_dir=source.root,
            isolated_output_dir=isolated,
            overlap_output_dir=overlap,
            accepted_license=subject.LICENSE_ID,
            test_source_injection=source.injection,
        )
    assert not (overlap / subject.PREPARATION_RECEIPT_FILENAME).exists()

    monkeypatch.undo()
    source = _make_source(tmp_path / "receipt")
    isolated = tmp_path / "receipt-isolated"
    overlap = tmp_path / "receipt-overlap"
    real_stage = subject._stage_terminal_receipt
    retained_fd = -1

    def retain_stage(
        directory_fd: int,
        raw: bytes,
    ) -> tuple[int, subject._PrivateFile]:
        nonlocal retained_fd
        descriptor, staged = real_stage(directory_fd, raw)
        retained_fd = subject.os.dup(descriptor)
        return descriptor, staged

    def mutate_staged(callback: object) -> None:
        assert retained_fd >= 0
        subject.os.pwrite(retained_fd, b"X", 0)
        callback()

    monkeypatch.setattr(subject, "_stage_terminal_receipt", retain_stage)
    monkeypatch.setattr(subject, "_terminal_commit_guard", mutate_staged)
    try:
        with pytest.raises(subject.Primock57PreparationError, match="^$"):
            subject.prepare_primock57_conversation_fixture(
                source_dir=source.root,
                isolated_output_dir=isolated,
                overlap_output_dir=overlap,
                accepted_license=subject.LICENSE_ID,
                test_source_injection=source.injection,
            )
    finally:
        if retained_fd >= 0:
            subject.os.close(retained_fd)
    assert not (overlap / subject.PREPARATION_RECEIPT_FILENAME).exists()


def test_prelink_revalidates_isolated_bundle_and_source_root_privacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source(tmp_path / "isolated")
    isolated = tmp_path / "isolated-output"
    overlap = tmp_path / "isolated-overlap"

    def mutate_isolated(callback: object) -> None:
        pcm = isolated / "primock57-isolated-00.f32le"
        raw = bytearray(pcm.read_bytes())
        raw[0] ^= 1
        pcm.write_bytes(raw)
        pcm.chmod(0o600)
        callback()

    monkeypatch.setattr(subject, "_terminal_commit_guard", mutate_isolated)
    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.prepare_primock57_conversation_fixture(
            source_dir=source.root,
            isolated_output_dir=isolated,
            overlap_output_dir=overlap,
            accepted_license=subject.LICENSE_ID,
            test_source_injection=source.injection,
        )
    assert not (overlap / subject.PREPARATION_RECEIPT_FILENAME).exists()

    monkeypatch.undo()
    source = _make_source(tmp_path / "source-root")
    isolated = tmp_path / "root-isolated"
    overlap = tmp_path / "root-overlap"

    def expose_source_root(callback: object) -> None:
        source.root.chmod(0o755)
        callback()

    monkeypatch.setattr(subject, "_terminal_commit_guard", expose_source_root)
    try:
        with pytest.raises(subject.Primock57PreparationError, match="^$"):
            subject.prepare_primock57_conversation_fixture(
                source_dir=source.root,
                isolated_output_dir=isolated,
                overlap_output_dir=overlap,
                accepted_license=subject.LICENSE_ID,
                test_source_injection=source.injection,
            )
    finally:
        source.root.chmod(0o700)
    assert not (overlap / subject.PREPARATION_RECEIPT_FILENAME).exists()


@pytest.mark.parametrize("product", ["isolated", "overlap"])
def test_loader_rejects_close_time_output_directory_mode_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    product: str,
) -> None:
    _source, isolated, overlap, _prepared = _prepare(tmp_path)
    target = isolated if product == "isolated" else overlap
    real_validate = subject._validate_evidence_identity

    def mutate_directory(*args: object, **kwargs: object) -> object:
        result = real_validate(*args, **kwargs)
        target.chmod(0o755)
        return result

    monkeypatch.setattr(subject, "_validate_evidence_identity", mutate_directory)
    loader = (
        subject.load_primock57_isolated_bundle
        if product == "isolated"
        else subject.load_primock57_overlap_bundle
    )
    try:
        with pytest.raises(subject.Primock57PreparationError, match="^$"):
            loader(target)
    finally:
        target.chmod(0o700)


def test_isolated_loader_reports_canonical_source_contract(tmp_path: Path) -> None:
    _source, isolated, _overlap, _prepared = _prepare(tmp_path)
    receipt = json.loads((isolated / subject.PREPARATION_RECEIPT_FILENAME).read_bytes())
    loaded = subject.load_primock57_isolated_bundle(isolated)
    canonical = subject._source_contract_sha256(
        fixture_id=receipt["fixture_id"],
        production_evidence=receipt["production_evidence"],
        source_rows=receipt["source_files"],
        selection_sha256=receipt["selection_sha256"],
        preparer_files=tuple(receipt["preparer_files"]),
        lock_recipe_sha256=receipt["lock_recipe_sha256"],
    )
    assert loaded.source_contract_sha256 == canonical
    assert loaded.source_contract_sha256 != loaded.receipt_sha256


def test_preparer_closure_binds_eager_local_imports(tmp_path: Path) -> None:
    _source, isolated, _overlap, _prepared = _prepare(tmp_path)
    receipt = json.loads((isolated / subject.PREPARATION_RECEIPT_FILENAME).read_bytes())
    paths = {row["path"] for row in receipt["preparer_files"]}
    expected = {
        "core/__init__.py",
        "core/wer.py",
        "tools/__init__.py",
        "tools/prepare_primock57_conversation_fixture.py",
        "tools/streaming_stt/__init__.py",
        "tools/streaming_stt/bounded_io.py",
        "tools/streaming_stt/corpus.py",
        "tools/streaming_stt/corpus_writer.py",
        "tools/streaming_stt/private_diagnostic_receipt.py",
        "tools/streaming_stt/protocol.py",
    }
    assert paths == expected == set(subject._PREPARER_FILES)

    repo_root = Path(subject.__file__).resolve().parents[1]
    script = """
import json
from pathlib import Path
import sys
import tools.prepare_primock57_conversation_fixture
root = Path.cwd().resolve()
rows = sorted({
    Path(module.__file__).resolve().relative_to(root).as_posix()
    for module in sys.modules.values()
    if getattr(module, "__file__", None)
    and Path(module.__file__).resolve().is_relative_to(root)
    and Path(module.__file__).suffix == ".py"
})
print(json.dumps(rows, separators=(",", ":")))
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert json.loads(completed.stdout) == sorted(expected)


def test_locked_artifact_pins_reject_self_consistent_pcm_rewrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, isolated, overlap, prepared = _prepare(tmp_path)
    overlap_manifest_path = overlap / subject.OVERLAP_MANIFEST_FILENAME
    overlap_receipt_path = overlap / subject.PREPARATION_RECEIPT_FILENAME
    overlap_manifest = json.loads(overlap_manifest_path.read_bytes())
    artifact_pins = {
        case.source_path.name: (case.samples, case.sha256)
        for case in prepared.isolated_corpus.cases
    }
    for case in overlap_manifest["cases"]:
        samples = case["envelope"]["samples"]
        for key in ("role_a", "role_b", "mix"):
            artifact_pins[case[key]["file"]] = (samples, case[key]["sha256"])
    contract = subject._SourceContract(
        fixture_id=source.injection.fixture_id,
        pins=source.injection.file_pins,
        role_a_duration=source.injection.role_a_duration_seconds,
        role_b_duration=source.injection.role_b_duration_seconds,
        isolated=source.injection.isolated_cases,
        overlaps=source.injection.overlap_cases,
        artifact_pins=artifact_pins,
        production_evidence=True,
        lock_recipe_sha256="f" * 64,
    )

    def locked_contract(*_args: object, **_kwargs: object) -> subject._SourceContract:
        return contract

    monkeypatch.setattr(subject, "_validate_evidence_identity", locked_contract)

    corpus_path = isolated / "corpus.json"
    corpus = json.loads(corpus_path.read_bytes())
    pcm_path = isolated / corpus["cases"][0]["file"]
    pcm_raw = bytearray(pcm_path.read_bytes())
    pcm_raw[0] ^= 1
    pcm_path.write_bytes(pcm_raw)
    pcm_path.chmod(0o600)
    corpus["cases"][0]["sha256"] = _sha256(pcm_raw)
    corpus_path.write_bytes(_canonical(corpus))
    corpus_path.chmod(0o600)
    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.load_primock57_isolated_bundle(isolated)

    role = overlap_manifest["cases"][0]["role_a"]
    role_path = overlap / role["file"]
    role_values = np.frombuffer(role_path.read_bytes(), dtype="<f4").copy()
    role_values[role["activity_start_sample"]] = np.float32(123.0)
    role_raw = np.asarray(role_values, dtype="<f4").tobytes()
    role_path.write_bytes(role_raw)
    role_path.chmod(0o600)
    role["sha256"] = _sha256(role_raw)
    other = overlap_manifest["cases"][0]["role_b"]
    other_values = np.frombuffer((overlap / other["file"]).read_bytes(), dtype="<f4")
    mix_values = np.multiply(
        np.add(role_values, other_values, dtype=np.float32),
        np.float32(0.5),
        dtype=np.float32,
    )
    mix = overlap_manifest["cases"][0]["mix"]
    mix_raw = np.asarray(mix_values, dtype="<f4").tobytes()
    mix_path = overlap / mix["file"]
    mix_path.write_bytes(mix_raw)
    mix_path.chmod(0o600)
    mix["sha256"] = _sha256(mix_raw)
    manifest_raw = _canonical(overlap_manifest)
    receipt = json.loads(overlap_receipt_path.read_bytes())
    receipt["manifest"]["bytes"] = len(manifest_raw)
    receipt["manifest"]["sha256"] = _sha256(manifest_raw)
    overlap_manifest_path.write_bytes(manifest_raw)
    overlap_manifest_path.chmod(0o600)
    overlap_receipt_path.write_bytes(_canonical(receipt))
    overlap_receipt_path.chmod(0o600)
    with pytest.raises(subject.Primock57PreparationError, match="^$"):
        subject.load_primock57_overlap_bundle(overlap)
