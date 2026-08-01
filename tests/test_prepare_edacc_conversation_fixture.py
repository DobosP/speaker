from __future__ import annotations

from array import array
from collections import Counter
import csv
from dataclasses import replace
import hashlib
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
    data = root / "data"
    test.mkdir(parents=True)
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
        recording = f"EDACC-C{index:02d}"
        speaker = f"{recording}-A"
        conversations.append(f"{recording}\n")
        _write_wav(data / f"{recording}.wav", index)
        accent_rows.append(
            {
                "Final-Participant_ID": f"EDAC-C{index:02d}P1",
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
            other = f"{recording}-B"
            accent_rows.append(
                {
                    "Final-Participant_ID": f"EDAC-C{index:02d}P2",
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
    (test / "segments").write_text("".join(segments), encoding="utf-8")
    (test / "utt2spk").write_text("".join(utt2spk), encoding="utf-8")
    (test / "text").write_text("".join(text), encoding="utf-8")
    (test / "stm").write_text("".join(stm), encoding="utf-8")
    (test / "stm.filt").write_text("".join(filtered_stm), encoding="utf-8")
    with (root / "linguistic_background.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("Final-Participant_ID", _ACCENT_FIELD),
        )
        writer.writeheader()
        writer.writerows(accent_rows)
    root.chmod(0o700)
    return root


def _archive(tmp_path: Path, root: Path, *, name: str = "source.tar.gz") -> Path:
    archive = tmp_path / name
    with tarfile.open(archive, "w:gz") as package:
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
        "bytes": sum(path.stat().st_size for path in (_root / "data").glob("*.wav")),
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
    assert prepare._validate_private_paths(
        source_root=source,
        archive_path=archive,
        output_dir=output,
        production=False,
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
