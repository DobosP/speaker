from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile
from types import SimpleNamespace

import pytest

from tools import prepare_common_voice_conversation_fixture as projection
from tools import prepare_common_voice_spontaneous_v4 as parent
from tools import public_conversation_fixture as generic_fixture
from tools.streaming_stt.corpus import load_corpus


TERMS = frozenset({"CC0-1.0", "MDC-CONSUMER-TERMS-2026-05-06"})


def _row(index: int, *, duration_ms: int) -> dict[str, str]:
    value = {field: "" for field in parent.MAIN_TSV_FIELDS}
    value.update(
        {
            "client_id": f"speaker-{index:03d}",
            "audio_id": f"audio-{index:03d}",
            "audio_file": f"spontaneous-speech-en-{index:03d}.mp3",
            "duration_ms": str(duration_ms),
            "prompt_id": f"prompt-{index:03d}",
            "prompt": "Describe something familiar.",
            "transcription": f"private reference words {index}",
            "votes": "2",
            "language": "en",
            "prompt_upvotes": "1",
            "prompt_reports": "0",
            "is_edited": "0",
            "split": "test",
            "char_per_sec": "12.5",
        }
    )
    return value


def _rows() -> list[dict[str, str]]:
    return [
        _row(bucket * 8 + offset, duration_ms=duration)
        for bucket, duration in enumerate((3_000, 7_000, 12_000, 18_000))
        for offset in range(8)
    ]


def _tsv(rows: list[dict[str, str]]) -> bytes:
    value = io.StringIO(newline="")
    writer = csv.DictWriter(
        value,
        fieldnames=parent.MAIN_TSV_FIELDS,
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return value.getvalue().encode("utf-8")


def _add(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o600
    archive.addfile(info, io.BytesIO(payload))


def _archive(path: Path, rows: list[dict[str, str]]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        _add(archive, parent._README_MEMBER, b"readme\n")
        _add(archive, parent._QA_MEMBER, b"{}\n")
        _add(archive, parent._REPORTED_MEMBER, b"comment\tprivate-canary\n")
        _add(archive, parent._MAIN_MEMBER, _tsv(rows))
        for index, row in enumerate(rows):
            _add(
                archive,
                f"{parent._AUDIO_PREFIX}{row['audio_file']}",
                f"fake-mp3-{index:03d}".encode("ascii"),
            )
    path.chmod(0o600)
    return path


def _patch_release(monkeypatch, archive: Path, rows: list[dict[str, str]]) -> None:
    raw = archive.read_bytes()
    total_duration = sum(int(row["duration_ms"]) for row in rows)
    monkeypatch.setattr(parent, "ARCHIVE_SIZE_BYTES", len(raw))
    monkeypatch.setattr(parent, "ARCHIVE_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(parent, "EXPECTED_ROWS", len(rows))
    monkeypatch.setattr(parent, "EXPECTED_DURATION_MS", total_duration)
    monkeypatch.setattr(
        parent,
        "EXPECTED_SPLIT_COUNTS",
        {"train": 0, "dev": 0, "test": len(rows), "unassigned": 0},
    )
    monkeypatch.setattr(
        parent,
        "EXPECTED_SPLIT_DURATION_MS",
        {"train": 0, "dev": 0, "test": total_duration, "unassigned": 0},
    )
    monkeypatch.setattr(
        parent,
        "EXPECTED_SPLIT_SPEAKERS",
        {"train": 0, "dev": 0, "test": len(rows)},
    )
    monkeypatch.setattr(parent, "EXPECTED_TOTAL_SPEAKERS", len(rows))


def _decoder(payload: bytes) -> parent.DecodedAudio:
    index = int(payload[-3:])
    duration_ms = (3_000, 7_000, 12_000, 18_000)[index // 8]
    samples = duration_ms * 16
    value = (index % 8 + 1) / 10.0
    return parent.DecodedAudio(struct.pack("<f", value) * samples, samples)


def _api_receipt() -> parent.MdcApiReceipt:
    return parent.MdcApiReceipt(
        filename=parent.ARCHIVE_FILENAME,
        size_bytes=parent.ARCHIVE_SIZE_BYTES,
        checksum_algorithm="sha256",
        checksum_sha256=parent.ARCHIVE_SHA256,
    )


def _parent_fixture(tmp_path: Path, monkeypatch):
    rows = _rows()
    archive = _archive(tmp_path / "source.tar.gz", rows)
    _patch_release(monkeypatch, archive, rows)
    fixture_id = "cv-projection-unit-v1"
    prepared = parent.prepare_common_voice_spontaneous_v4(
        archive_path=archive,
        output_dir=tmp_path / "parent",
        accepted_terms=TERMS,
        api_receipt=_api_receipt(),
        test_decoder_injection=parent.TestDecoderInjection(_decoder, fixture_id),
    )
    return archive, prepared, fixture_id


def _semantic_drift_lock(tmp_path: Path) -> Path:
    root = json.loads(generic_fixture.DEFAULT_LOCK.read_text(encoding="ascii"))
    source = next(
        item for item in root["sources"] if item["source_id"] == projection.SOURCE_ID
    )
    source["selection_recipe"]["eligibility"][0] = "verified_32_drifted"
    source["selection_recipe_sha256"] = generic_fixture._canonical_sha256(
        source["selection_recipe"]
    )
    digest_input = dict(root)
    digest_input.pop("recipe_sha256")
    root["recipe_sha256"] = generic_fixture._canonical_sha256(digest_input)
    path = tmp_path / "semantic-drift.lock.json"
    path.write_text(
        json.dumps(root, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    return path


def test_lock_recipe_is_the_exact_executable_projection_contract():
    lock = generic_fixture.load_fixture_lock()
    source = lock.source_by_id(projection.SOURCE_ID)
    assert dict(source.selection_recipe) == projection._expected_source_recipe(
        seed=lock.selection_seed
    )
    assert source.decoder_contract == projection.DECODER_CONTRACT


def test_fully_redigested_projection_recipe_drift_fails_before_decode(
    tmp_path,
    monkeypatch,
):
    archive, prepared, fixture_id = _parent_fixture(tmp_path, monkeypatch)
    output = tmp_path / "must-not-publish-drifted-recipe"
    decoded = False

    def tracking_decoder(payload: bytes) -> parent.DecodedAudio:
        nonlocal decoded
        decoded = True
        return _decoder(payload)

    with pytest.raises(projection.CvProjectionError):
        projection.prepare_common_voice_conversation_fixture(
            archive_path=archive,
            parent_corpus_path=prepared.corpus.path,
            output_dir=output,
            accepted_terms=TERMS,
            api_receipt=_api_receipt(),
            lock_path=_semantic_drift_lock(tmp_path),
            test_projection_injection=projection.TestProjectionInjection(
                fixture_id,
                tracking_decoder,
            ),
        )
    assert decoded is False
    assert not output.exists()


def test_recomputes_parent_source_and_projects_six_per_bucket(tmp_path, monkeypatch):
    archive, prepared, fixture_id = _parent_fixture(tmp_path, monkeypatch)
    result = projection.prepare_common_voice_conversation_fixture(
        archive_path=archive,
        parent_corpus_path=prepared.corpus.path,
        output_dir=tmp_path / "projected",
        accepted_terms=TERMS,
        api_receipt=_api_receipt(),
        test_projection_injection=projection.TestProjectionInjection(
            fixture_id,
            _decoder,
        ),
    )

    assert result.production_evidence is False
    assert len(result.corpus.cases) == 24
    assert result.corpus.path.stat().st_mode & 0o777 == 0o600
    assert result.corpus.path.parent.stat().st_mode & 0o777 == 0o700
    receipt_path = result.corpus.path.parent / "preparation-receipt.json"
    receipt_raw = receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)
    assert hashlib.sha256(receipt_raw).hexdigest() == result.receipt_sha256
    assert receipt["selection"]["parent_receipt_sha256"] == result.parent_receipt_sha256
    assert (
        receipt["selection"]["parent_corpus_manifest_sha256"]
        == result.parent_corpus_manifest_sha256
    )
    assert receipt["decoder"]["production_evidence"] is False
    assert receipt["decoder"]["contract"] == projection.DECODER_CONTRACT
    assert receipt["selection"]["eligible_rows"] == 32
    assert {item["attributes"]["duration_bucket"] for item in receipt["cases"]} == {
        "d0_2_6s",
        "d1_6_10s",
        "d2_10_15s",
        "d3_15_25s",
    }
    assert {
        bucket: sum(
            item["attributes"]["duration_bucket"] == bucket
            for item in receipt["cases"]
        )
        for bucket, *_rest in parent.DURATION_BUCKETS
    } == {bucket: 6 for bucket, *_rest in parent.DURATION_BUCKETS}
    assert len({item["speaker_sha256"] for item in receipt["cases"]}) == 24
    assert len({item["attributes"]["prompt_sha256"] for item in receipt["cases"]}) == 24
    assert load_corpus(result.corpus.path).digest == result.corpus.digest

    # Exercise the generic validator's case-level duration/quota core without
    # weakening its production-only receipt gate for this synthetic release.
    generic_fixture._validate_source_invariants(
        projection.SOURCE_ID,
        receipt["cases"],
    )
    with pytest.raises(generic_fixture.FixtureError):
        generic_fixture.validate_private_source(
            result.corpus.path,
            lock=generic_fixture.load_fixture_lock(),
        )

    encoded = json.dumps(receipt)
    for private in (
        "private reference words",
        "speaker-",
        "prompt-",
        "spontaneous-speech-en-",
        "private-canary",
        str(tmp_path),
    ):
        assert private not in encoded


def test_parent_receipt_tamper_is_rejected_before_projection(tmp_path, monkeypatch):
    archive, prepared, fixture_id = _parent_fixture(tmp_path, monkeypatch)
    receipt_path = prepared.corpus.path.parent / "preparation-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["cases"][0]["source_mp3_sha256"] = "f" * 64
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    receipt_path.chmod(0o600)

    with pytest.raises(projection.CvProjectionError):
        projection.prepare_common_voice_conversation_fixture(
            archive_path=archive,
            parent_corpus_path=prepared.corpus.path,
            output_dir=tmp_path / "projected-tamper",
            accepted_terms=TERMS,
            api_receipt=_api_receipt(),
            test_projection_injection=projection.TestProjectionInjection(
                fixture_id,
                _decoder,
            ),
        )


def test_rebound_parent_pcm_is_redecoded_from_mp3_and_rejected(tmp_path, monkeypatch):
    archive, prepared, fixture_id = _parent_fixture(tmp_path, monkeypatch)
    receipt_path = prepared.corpus.path.parent / "preparation-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest = json.loads(prepared.corpus.path.read_text(encoding="utf-8"))

    replacement = struct.pack("<f", 0.9875)
    replacement_sha256 = hashlib.sha256(replacement).hexdigest()
    pcm_path = prepared.corpus.cases[0].source_path
    pcm_path.write_bytes(replacement)
    pcm_path.chmod(0o600)
    receipt["cases"][0]["pcm_sha256"] = replacement_sha256
    receipt["cases"][0]["pcm_samples"] = 1
    receipt_raw = parent._canonical_json_bytes(receipt)
    receipt_path.write_bytes(receipt_raw)
    receipt_path.chmod(0o600)
    manifest["cases"][0]["sha256"] = replacement_sha256
    manifest["cases"][0]["samples"] = 1
    manifest["provenance"]["source_set_sha256"] = hashlib.sha256(
        receipt_raw
    ).hexdigest()
    prepared.corpus.path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    prepared.corpus.path.chmod(0o600)

    with pytest.raises(projection.CvProjectionError):
        projection.prepare_common_voice_conversation_fixture(
            archive_path=archive,
            parent_corpus_path=prepared.corpus.path,
            output_dir=tmp_path / "projected-rebound-pcm",
            accepted_terms=TERMS,
            api_receipt=_api_receipt(),
            test_projection_injection=projection.TestProjectionInjection(
                fixture_id,
                _decoder,
            ),
        )


@pytest.mark.parametrize("public_input", ["directory", "pcm"])
def test_parent_directory_and_pcm_must_remain_owner_private(
    tmp_path,
    monkeypatch,
    public_input,
):
    archive, prepared, fixture_id = _parent_fixture(tmp_path, monkeypatch)
    if public_input == "directory":
        prepared.corpus.path.parent.chmod(0o755)
    else:
        prepared.corpus.cases[0].source_path.chmod(0o644)

    with pytest.raises(projection.CvProjectionError):
        projection.prepare_common_voice_conversation_fixture(
            archive_path=archive,
            parent_corpus_path=prepared.corpus.path,
            output_dir=tmp_path / f"projected-public-{public_input}",
            accepted_terms=TERMS,
            api_receipt=_api_receipt(),
            test_projection_injection=projection.TestProjectionInjection(
                fixture_id,
                _decoder,
            ),
        )


def test_parent_directory_change_during_decode_fails_before_publication(
    tmp_path,
    monkeypatch,
):
    archive, prepared, fixture_id = _parent_fixture(tmp_path, monkeypatch)
    output = tmp_path / "projected-raced-parent"
    changed = False

    def racing_decoder(payload: bytes) -> parent.DecodedAudio:
        nonlocal changed
        decoded = _decoder(payload)
        if not changed:
            prepared.corpus.path.parent.chmod(0o755)
            changed = True
        return decoded

    with pytest.raises(projection.CvProjectionError):
        projection.prepare_common_voice_conversation_fixture(
            archive_path=archive,
            parent_corpus_path=prepared.corpus.path,
            output_dir=output,
            accepted_terms=TERMS,
            api_receipt=_api_receipt(),
            test_projection_injection=projection.TestProjectionInjection(
                fixture_id,
                racing_decoder,
            ),
        )
    assert changed is True
    assert not output.exists()


def test_parent_pcm_mode_change_during_decode_fails_before_publication(
    tmp_path,
    monkeypatch,
):
    archive, prepared, fixture_id = _parent_fixture(tmp_path, monkeypatch)
    output = tmp_path / "projected-raced-pcm"
    changed = False

    def racing_decoder(payload: bytes) -> parent.DecodedAudio:
        nonlocal changed
        decoded = _decoder(payload)
        if not changed:
            prepared.corpus.cases[0].source_path.chmod(0o644)
            changed = True
        return decoded

    with pytest.raises(projection.CvProjectionError):
        projection.prepare_common_voice_conversation_fixture(
            archive_path=archive,
            parent_corpus_path=prepared.corpus.path,
            output_dir=output,
            accepted_terms=TERMS,
            api_receipt=_api_receipt(),
            test_projection_injection=projection.TestProjectionInjection(
                fixture_id,
                racing_decoder,
            ),
        )
    assert changed is True
    assert not output.exists()


def test_projection_backtracks_when_bucket_greedy_choice_is_infeasible():
    seed = "cv-projection-backtracking-v1"
    buckets = [bucket for bucket, *_rest in parent.DURATION_BUCKETS]
    values: list[projection._BoundParentCase] = []

    for bucket_index, bucket in enumerate(buckets):
        for offset in range(8):
            identity = hashlib.sha256(
                f"identity:{bucket}:{offset}".encode("ascii")
            ).hexdigest()
            receipt = {
                "identity_sha256": identity,
                "speaker_sha256": hashlib.sha256(
                    f"speaker:{bucket}:{offset}".encode("ascii")
                ).hexdigest(),
                "prompt_sha256": hashlib.sha256(
                    f"placeholder:{bucket}:{offset}".encode("ascii")
                ).hexdigest(),
            }
            values.append(
                projection._BoundParentCase(
                    row=parent.SpsRow(
                        client_id=f"speaker-{bucket_index}-{offset}",
                        audio_id=f"audio-{bucket_index}-{offset}",
                        audio_file=f"audio-{bucket_index}-{offset}.mp3",
                        prompt_id=f"prompt-{bucket_index}-{offset}",
                        duration_ms=(3_000, 7_000, 12_000, 18_000)[bucket_index],
                        transcription="known reference",
                        split="test",
                        language="en",
                        quality_tags=(),
                    ),
                    bucket=bucket,
                    corpus_case=SimpleNamespace(),
                    receipt_case=receipt,
                )
            )

    by_bucket = {
        bucket: sorted(
            (value for value in values if value.bucket == bucket),
            key=lambda value: (
                projection._rank(seed, value),
                value.receipt_case["identity_sha256"],
            ),
        )
        for bucket in buckets
    }
    shared = [
        hashlib.sha256(f"shared:{index}".encode("ascii")).hexdigest()
        for index in range(8)
    ]
    for index, value in enumerate(by_bucket[buckets[0]]):
        value.receipt_case["prompt_sha256"] = shared[index]
    second_prompts = shared[:3] + [
        hashlib.sha256(f"second:{index}".encode("ascii")).hexdigest()
        for index in range(5)
    ]
    for value, prompt in zip(by_bucket[buckets[1]], second_prompts, strict=True):
        value.receipt_case["prompt_sha256"] = prompt
    for bucket_index, bucket in enumerate(buckets[2:], start=2):
        for offset, value in enumerate(by_bucket[bucket]):
            value.receipt_case["prompt_sha256"] = hashlib.sha256(
                f"unique:{bucket_index}:{offset}".encode("ascii")
            ).hexdigest()

    greedy_prompts = {
        value.receipt_case["prompt_sha256"] for value in by_bucket[buckets[0]][:6]
    }
    assert sum(
        value.receipt_case["prompt_sha256"] not in greedy_prompts
        for value in by_bucket[buckets[1]]
    ) == 5

    selected = projection._project_cases(values, seed=seed)
    assert len(selected) == 24
    assert len({value.receipt_case["speaker_sha256"] for value in selected}) == 24
    assert len({value.receipt_case["prompt_sha256"] for value in selected}) == 24


def test_projection_is_deterministic_and_cli_errors_are_detail_free(
    tmp_path, monkeypatch, capsys
):
    archive, prepared, fixture_id = _parent_fixture(tmp_path, monkeypatch)
    first = projection.prepare_common_voice_conversation_fixture(
        archive_path=archive,
        parent_corpus_path=prepared.corpus.path,
        output_dir=tmp_path / "projected-a",
        accepted_terms=TERMS,
        api_receipt=_api_receipt(),
        test_projection_injection=projection.TestProjectionInjection(
            fixture_id,
            _decoder,
        ),
    )
    second = projection.prepare_common_voice_conversation_fixture(
        archive_path=archive,
        parent_corpus_path=prepared.corpus.path,
        output_dir=tmp_path / "projected-b",
        accepted_terms=TERMS,
        api_receipt=_api_receipt(),
        test_projection_injection=projection.TestProjectionInjection(
            fixture_id,
            _decoder,
        ),
    )
    assert first.corpus.digest == second.corpus.digest
    assert first.receipt_sha256 == second.receipt_sha256

    secret = tmp_path / "private-secret"
    assert projection.main(["--archive", str(secret)]) == 2
    captured = capsys.readouterr()
    assert str(secret) not in captured.out
    assert json.loads(captured.out) == {
        "error": "cvss4 fixture preparation failed",
        "ok": False,
    }
