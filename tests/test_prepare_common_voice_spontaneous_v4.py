from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile

import pytest

from tools import prepare_common_voice_spontaneous_v4 as prepare
from tools.streaming_stt.corpus import load_corpus


TERMS = frozenset({"CC0-1.0", "MDC-CONSUMER-TERMS-2026-05-06"})


def _row(
    index: int,
    *,
    split: str,
    duration_ms: int,
    speaker: str | None = None,
    transcription: str | None = None,
    quality_tags: str = "",
) -> dict[str, str]:
    values = {field: "" for field in prepare.MAIN_TSV_FIELDS}
    values.update(
        {
            "client_id": speaker or f"speaker-{index:03d}",
            "audio_id": f"audio-{index:03d}",
            "audio_file": f"spontaneous-speech-en-{index:03d}.mp3",
            "duration_ms": str(duration_ms),
            "prompt_id": f"prompt-{index:03d}",
            "prompt": "Describe something familiar.",
            "transcription": transcription or f"private reference words {index}",
            "votes": "2",
            "language": "en",
            "prompt_upvotes": "1",
            "prompt_reports": "0",
            "is_edited": "0",
            "split": split,
            "char_per_sec": "12.5",
            "quality_tags": quality_tags,
        }
    )
    return values


def _valid_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    durations = (3_000, 7_000, 12_000, 18_000)
    for bucket, duration_ms in enumerate(durations):
        for offset in range(8):
            index = bucket * 8 + offset
            rows.append(_row(index, split="test", duration_ms=duration_ms))
    rows.extend(
        [
            _row(32, split="train", duration_ms=1_000),
            _row(33, split="dev", duration_ms=1_100),
            _row(34, split="unassigned", duration_ms=1_200),
        ]
    )
    return rows


def _tsv(rows: list[dict[str, str]], *, fields=prepare.MAIN_TSV_FIELDS) -> bytes:
    buffer = io.StringIO(newline="")
    import csv

    writer = csv.DictWriter(
        buffer,
        fieldnames=fields,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _add_file(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o600
    archive.addfile(info, io.BytesIO(payload))


def _archive(
    path: Path,
    rows: list[dict[str, str]],
    *,
    tsv_payload: bytes | None = None,
    extras: tuple[tuple[str, bytes], ...] = (),
) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        _add_file(archive, prepare._README_MEMBER, b"public readme\n")
        _add_file(archive, prepare._QA_MEMBER, b"{}\n")
        _add_file(
            archive,
            prepare._REPORTED_MEMBER,
            b"comment\tprivate-reported-sentinel\n",
        )
        _add_file(
            archive,
            prepare._MAIN_MEMBER,
            _tsv(rows) if tsv_payload is None else tsv_payload,
        )
        for row in rows:
            _add_file(
                archive,
                f"{prepare._AUDIO_PREFIX}{row['audio_file']}",
                f"fake-mp3-{row['audio_id']}".encode("ascii"),
            )
        for name, payload in extras:
            _add_file(archive, name, payload)
    path.chmod(0o600)
    return path


def _patch_release(monkeypatch, path: Path, rows: list[dict[str, str]]) -> None:
    counts = {name: 0 for name in prepare.EXPECTED_SPLIT_COUNTS}
    durations = {name: 0 for name in prepare.EXPECTED_SPLIT_COUNTS}
    speakers: dict[str, set[str]] = {name: set() for name in counts}
    for row in rows:
        split = row["split"] or "unassigned"
        counts[split] += 1
        durations[split] += int(row["duration_ms"])
        speakers[split].add(row["client_id"])
    payload = path.read_bytes()
    monkeypatch.setattr(prepare, "ARCHIVE_SIZE_BYTES", len(payload))
    monkeypatch.setattr(prepare, "ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(prepare, "EXPECTED_ROWS", len(rows))
    monkeypatch.setattr(
        prepare,
        "EXPECTED_DURATION_MS",
        sum(int(row["duration_ms"]) for row in rows),
    )
    monkeypatch.setattr(prepare, "EXPECTED_SPLIT_COUNTS", counts)
    monkeypatch.setattr(prepare, "EXPECTED_SPLIT_DURATION_MS", durations)
    monkeypatch.setattr(
        prepare,
        "EXPECTED_SPLIT_SPEAKERS",
        {name: len(speakers[name]) for name in ("train", "dev", "test")},
    )
    monkeypatch.setattr(
        prepare,
        "EXPECTED_TOTAL_SPEAKERS",
        len(set().union(*speakers.values())),
    )
    monkeypatch.setattr(prepare, "MAX_DURATION_DELTA_MS", 100_000)


def _api_receipt() -> prepare.MdcApiReceipt:
    return prepare.MdcApiReceipt(
        filename=prepare.ARCHIVE_FILENAME,
        size_bytes=prepare.ARCHIVE_SIZE_BYTES,
        checksum_algorithm="sha256",
        checksum_sha256=prepare.ARCHIVE_SHA256,
    )


def _fake_decoder(_payload: bytes) -> prepare.DecodedAudio:
    return prepare.DecodedAudio(struct.pack("<f", 0.125), 1)


def _injection(fixture_id: str = "unit-v1") -> prepare.TestDecoderInjection:
    return prepare.TestDecoderInjection(_fake_decoder, fixture_id)


def _prepare_fixture(tmp_path: Path, monkeypatch):
    rows = _valid_rows()
    archive = _archive(tmp_path / "source.tar.gz", rows)
    _patch_release(monkeypatch, archive, rows)
    result = prepare.prepare_common_voice_spontaneous_v4(
        archive_path=archive,
        output_dir=tmp_path / "prepared",
        accepted_terms=TERMS,
        api_receipt=_api_receipt(),
        test_decoder_injection=_injection("unit-v1"),
    )
    return result, rows, archive


def test_prepares_private_schema_v2_slice_and_path_free_receipt(tmp_path, monkeypatch):
    result, rows, _archive_path = _prepare_fixture(tmp_path, monkeypatch)

    corpus = result.corpus
    assert corpus.schema_version == 2
    assert len(corpus.cases) == 32
    assert corpus.audio_bytes == 32 * 4
    assert corpus.path.stat().st_mode & 0o777 == 0o600
    assert corpus.path.parent.stat().st_mode & 0o777 == 0o700
    assert {case.tags[-1] for case in corpus.cases} == {
        "d0_2_6s",
        "d1_6_10s",
        "d2_10_15s",
        "d3_15_25s",
    }
    assert len({case.case_id for case in corpus.cases}) == 32
    assert [case.expected_text for case in corpus.cases] != [
        row["transcription"] for row in rows[:32]
    ]

    receipt_path = corpus.path.parent / "preparation-receipt.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert hashlib.sha256(receipt_bytes).hexdigest() == result.receipt_sha256
    assert corpus.provenance is not None
    assert corpus.provenance.source_set_sha256 == result.receipt_sha256
    assert corpus.provenance.manifest_sha256 == result.selection_recipe_sha256
    assert receipt["dataset"]["release_date"] == "2026-06-12"
    assert receipt["dataset"]["mdc_published_date"] == "2026-06-17"
    assert receipt["decoder"] == {
        "contract": "test-injected-f32le-v1",
        "fixture_id": "unit-v1",
        "production_evidence": False,
    }
    assert receipt["preparer"]["contract"].endswith("-v2")
    assert set(receipt["preparer"]["files"]) == {
        "core/wer.py",
        "tools/prepare_common_voice_spontaneous_v4.py",
        "tools/public_voice_eval_matrix.py",
        "tools/streaming_stt/bounded_io.py",
        "tools/streaming_stt/corpus.py",
        "tools/streaming_stt/corpus_writer.py",
        "tools/streaming_stt/protocol.py",
    }
    assert receipt["evidence_scope"]["pcm_preparation_complete"] is True
    assert receipt["evidence_scope"]["stt_model_run"] is False
    assert receipt["evidence_scope"]["timing_scope"] == "prepared-pcm-only"
    assert receipt["metadata"]["reported_tsv_content_read"] is False
    assert receipt["selection"]["bucket_counts"] == {
        "d0_2_6s": 8,
        "d1_6_10s": 8,
        "d2_10_15s": 8,
        "d3_15_25s": 8,
    }
    encoded = json.dumps(receipt)
    for private in (
        "private reference words",
        "speaker-",
        "spontaneous-speech-en-",
        "private-reported-sentinel",
        str(tmp_path),
    ):
        assert private not in encoded
    assert load_corpus(corpus.path).digest == corpus.digest


def test_executable_release_pins_match_the_catalog():
    prepare._validate_catalog_contract()


def test_safe_result_and_cli_error_never_emit_private_values(
    tmp_path, monkeypatch, capsys
):
    result, _rows, archive = _prepare_fixture(tmp_path, monkeypatch)
    safe = prepare._safe_result(result)
    encoded = json.dumps(safe)
    assert "private reference" not in encoded
    assert str(tmp_path) not in encoded
    assert str(archive) not in encoded

    code = prepare.main(
        [
            "--archive",
            str(tmp_path / "private-does-not-exist.tar.gz"),
            "--output-dir",
            str(tmp_path / "other"),
            "--accept-term",
            "CC0-1.0",
            "--accept-term",
            "MDC-CONSUMER-TERMS-2026-05-06",
            "--mdc-api-filename",
            prepare.ARCHIVE_FILENAME,
            "--mdc-api-size-bytes",
            str(prepare.ARCHIVE_SIZE_BYTES),
            "--mdc-api-checksum",
            f"sha256:{prepare.ARCHIVE_SHA256}",
        ]
    )
    assert code == 2
    stdout = capsys.readouterr().out
    assert json.loads(stdout) == prepare._SAFE_ERROR
    assert "private-does-not-exist" not in stdout
    assert str(tmp_path) not in stdout


def test_missing_terms_and_mismatched_api_receipt_fail_before_archive(tmp_path):
    with pytest.raises(prepare.SpsPreparationError, match="terms"):
        prepare.prepare_common_voice_spontaneous_v4(
            archive_path=tmp_path / "private-name.tar.gz",
            output_dir=tmp_path / "output",
            accepted_terms=frozenset({"CC0-1.0"}),
            api_receipt=_api_receipt(),
        )
    bad = prepare.MdcApiReceipt(
        filename=prepare.ARCHIVE_FILENAME,
        size_bytes=prepare.ARCHIVE_SIZE_BYTES,
        checksum_algorithm="sha256",
        checksum_sha256="a" * 64,
    )
    with pytest.raises(prepare.SpsPreparationError, match="MDC content"):
        prepare.prepare_common_voice_spontaneous_v4(
            archive_path=tmp_path / "private-name.tar.gz",
            output_dir=tmp_path / "output",
            accepted_terms=TERMS,
            api_receipt=bad,
        )


def test_archive_hash_size_and_private_mode_are_required(tmp_path, monkeypatch):
    rows = _valid_rows()
    archive = _archive(tmp_path / "source.tar.gz", rows)
    _patch_release(monkeypatch, archive, rows)
    monkeypatch.setattr(prepare, "ARCHIVE_SHA256", "a" * 64)
    with pytest.raises(prepare.SpsPreparationError, match="archive prerequisite"):
        with prepare._verified_archive(archive):
            pass

    monkeypatch.setattr(
        prepare, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    archive.chmod(0o644)
    with pytest.raises(prepare.SpsPreparationError, match="archive prerequisite"):
        with prepare._verified_archive(archive):
            pass


def test_archive_is_rehashed_through_the_same_descriptor_at_close(
    tmp_path, monkeypatch
):
    rows = _valid_rows()
    archive = _archive(tmp_path / "source.tar.gz", rows)
    _patch_release(monkeypatch, archive, rows)
    original = prepare._hash_archive_descriptor
    calls = 0

    def changed_on_second_hash(descriptor: int) -> str:
        nonlocal calls
        calls += 1
        digest = original(descriptor)
        return digest if calls == 1 else "0" * 64

    monkeypatch.setattr(prepare, "_hash_archive_descriptor", changed_on_second_hash)
    with pytest.raises(prepare.SpsPreparationError, match="archive changed"):
        with prepare._verified_archive(archive) as opened:
            prepare._scan_archive(opened)
    assert calls == 2


def test_archive_and_output_paths_must_be_absolute_and_outside_git(tmp_path):
    with pytest.raises(prepare.SpsPreparationError, match="archive prerequisite"):
        with prepare._verified_archive(Path("relative-private.tar.gz")):
            pass
    with pytest.raises(prepare.SpsPreparationError, match="output prerequisite"):
        prepare._validated_output_path(Path("relative-output"))
    with pytest.raises(prepare.SpsPreparationError, match="output prerequisite"):
        prepare._validated_output_path(prepare._REPO_ROOT / "private-corpus")
    assert prepare._validated_output_path(tmp_path / "new-output") == (
        tmp_path / "new-output"
    )


@pytest.mark.parametrize("git_marker", ("file", "directory"))
def test_archive_and_output_reject_any_git_checkout_ancestor(tmp_path, git_marker):
    repository = tmp_path / f"sibling-{git_marker}"
    repository.mkdir()
    marker = repository / ".git"
    if git_marker == "file":
        marker.write_text("gitdir: /private/redacted\n", encoding="utf-8")
    else:
        marker.mkdir()
        (marker / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    archive = _archive(repository / "private.tar.gz", _valid_rows())

    with pytest.raises(prepare.SpsPreparationError, match="archive prerequisite"):
        with prepare._verified_archive(archive):
            pass
    with pytest.raises(prepare.SpsPreparationError, match="output prerequisite"):
        prepare._validated_output_path(repository / "private-output")


def test_empty_dot_git_sentinel_does_not_block_a_non_git_cache(tmp_path):
    cache = tmp_path / "non-git-cache"
    cache.mkdir()
    (cache / ".git").mkdir()

    assert prepare._has_git_ancestor(cache) is False
    assert prepare._validated_output_path(cache / "new-output") == (
        cache / "new-output"
    )


@pytest.mark.parametrize("bad_name", ["../escape", "/absolute", "other/file"])
def test_archive_rejects_unsafe_or_unexpected_members(tmp_path, monkeypatch, bad_name):
    rows = _valid_rows()
    archive = _archive(
        tmp_path / "source.tar.gz",
        rows,
        extras=((bad_name, b"bad"),),
    )
    _patch_release(monkeypatch, archive, rows)
    with prepare._verified_archive(archive) as opened:
        with pytest.raises(prepare.SpsPreparationError, match="archive layout"):
            prepare._scan_archive(opened)


def test_archive_rejects_duplicate_members(tmp_path, monkeypatch):
    rows = _valid_rows()
    duplicate = f"{prepare._AUDIO_PREFIX}{rows[0]['audio_file']}"
    archive = _archive(
        tmp_path / "source.tar.gz",
        rows,
        extras=((duplicate, b"duplicate"),),
    )
    _patch_release(monkeypatch, archive, rows)
    with prepare._verified_archive(archive) as opened:
        with pytest.raises(prepare.SpsPreparationError, match="archive layout"):
            prepare._scan_archive(opened)


def test_archive_rejects_gnu_sparse_regular_member(tmp_path, monkeypatch):
    archive_path = tmp_path / "sparse.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo(
            f"{prepare._AUDIO_PREFIX}spontaneous-speech-en-sparse.mp3"
        )
        info.type = tarfile.GNUTYPE_SPARSE
        info.size = 3
        archive.addfile(info, io.BytesIO(b"bad"))
    archive_path.chmod(0o600)
    payload = archive_path.read_bytes()
    monkeypatch.setattr(prepare, "ARCHIVE_SIZE_BYTES", len(payload))
    monkeypatch.setattr(prepare, "ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())
    with prepare._verified_archive(archive_path) as opened:
        with pytest.raises(prepare.SpsPreparationError, match="archive layout"):
            prepare._scan_archive(opened)


def test_exact_tsv_header_and_release_statistics_are_required(tmp_path, monkeypatch):
    rows = _valid_rows()
    bad_fields = prepare.MAIN_TSV_FIELDS[:-1]
    archive = _archive(
        tmp_path / "source.tar.gz",
        rows,
        tsv_payload=_tsv(rows, fields=bad_fields),
    )
    _patch_release(monkeypatch, archive, rows)
    with prepare._verified_archive(archive) as opened:
        layout = prepare._scan_archive(opened)
        payload = prepare._read_member(
            opened, layout.main_tsv, maximum_bytes=prepare._MAX_MAIN_TSV_BYTES
        )
    with pytest.raises(prepare.SpsPreparationError, match="metadata contract"):
        prepare._parse_main_tsv(payload)

    good = _tsv(rows)
    monkeypatch.setattr(prepare, "EXPECTED_ROWS", len(rows) + 1)
    with pytest.raises(prepare.SpsPreparationError, match="metadata statistics"):
        prepare._parse_main_tsv(good)


def test_assigned_speaker_overlap_fails_closed(tmp_path, monkeypatch):
    rows = _valid_rows()
    rows[-3]["client_id"] = rows[0]["client_id"]
    archive = _archive(tmp_path / "source.tar.gz", rows)
    _patch_release(monkeypatch, archive, rows)
    with pytest.raises(prepare.SpsPreparationError, match="speaker split"):
        prepare._parse_main_tsv(_tsv(rows))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("votes", "not-an-integer"),
        ("prompt_upvotes", "NaN"),
        ("char_per_sec", "inf"),
        ("is_edited", "true"),
    ],
)
def test_typed_metadata_fields_fail_closed(tmp_path, monkeypatch, field, value):
    rows = _valid_rows()
    rows[0][field] = value
    archive = _archive(tmp_path / "source.tar.gz", rows)
    _patch_release(monkeypatch, archive, rows)
    with pytest.raises(prepare.SpsPreparationError, match="metadata contract"):
        prepare._parse_main_tsv(_tsv(rows))


@pytest.mark.parametrize(
    "change",
    [
        {"quality_tags": "speech-rate"},
        {"transcription": "words [noise] here"},
    ],
)
def test_clean_selection_filter_is_explicit_and_has_no_fallback(
    tmp_path, monkeypatch, change
):
    rows = _valid_rows()
    rows[0].update(change)
    archive = _archive(tmp_path / "source.tar.gz", rows)
    _patch_release(monkeypatch, archive, rows)
    metadata = prepare._parse_main_tsv(_tsv(rows))
    with pytest.raises(prepare.SpsPreparationError, match="selection contract"):
        prepare._select_rows(metadata)


def test_selection_is_input_order_independent(tmp_path, monkeypatch):
    rows = _valid_rows()
    first_archive = _archive(tmp_path / "first.tar.gz", rows)
    _patch_release(monkeypatch, first_archive, rows)
    first = prepare._select_rows(prepare._parse_main_tsv(_tsv(rows)))
    reordered = list(reversed(rows))
    second = prepare._select_rows(prepare._parse_main_tsv(_tsv(reordered)))

    assert [(row.audio_id, bucket) for row, bucket in first[0]] == [
        (row.audio_id, bucket) for row, bucket in second[0]
    ]
    assert first[1:] == second[1:]


def test_decode_duration_and_total_budget_fail_closed(tmp_path, monkeypatch):
    rows = _valid_rows()
    archive = _archive(tmp_path / "source.tar.gz", rows)
    _patch_release(monkeypatch, archive, rows)
    monkeypatch.setattr(prepare, "MAX_DURATION_DELTA_MS", 1)
    with pytest.raises(prepare.SpsPreparationError, match="audio decode"):
        prepare.prepare_common_voice_spontaneous_v4(
            archive_path=archive,
            output_dir=tmp_path / "duration-output",
            accepted_terms=TERMS,
            api_receipt=_api_receipt(),
            test_decoder_injection=_injection(),
        )

    monkeypatch.setattr(prepare, "MAX_DURATION_DELTA_MS", 100_000)
    monkeypatch.setattr(prepare, "MAX_CORPUS_BYTES", 64)
    with pytest.raises(prepare.SpsPreparationError, match="corpus budget"):
        prepare.prepare_common_voice_spontaneous_v4(
            archive_path=archive,
            output_dir=tmp_path / "budget-output",
            accepted_terms=TERMS,
            api_receipt=_api_receipt(),
            test_decoder_injection=_injection(),
        )


def test_publication_refuses_overwrite_and_decoder_closure_changes_receipt(
    tmp_path, monkeypatch
):
    first, rows, archive = _prepare_fixture(tmp_path, monkeypatch)
    with pytest.raises(prepare.SpsPreparationError, match="output prerequisite"):
        prepare.prepare_common_voice_spontaneous_v4(
            archive_path=archive,
            output_dir=first.corpus.path.parent,
            accepted_terms=TERMS,
            api_receipt=_api_receipt(),
            test_decoder_injection=_injection("unit-v1"),
        )

    second = prepare.prepare_common_voice_spontaneous_v4(
        archive_path=archive,
        output_dir=tmp_path / "second",
        accepted_terms=TERMS,
        api_receipt=_api_receipt(),
        test_decoder_injection=_injection("unit-v2"),
    )
    assert first.receipt_sha256 != second.receipt_sha256
    assert first.metadata_sha256 == second.metadata_sha256
    assert first.selection_recipe_sha256 == second.selection_recipe_sha256
    assert len(rows) == 35


@pytest.mark.parametrize(
    "invalid",
    (
        {"contract": "direct-pyav-mp3-mono-f32le-16000-v1"},
        prepare.TestDecoderInjection(_fake_decoder, "/home/private/decoder"),
        prepare.TestDecoderInjection(_fake_decoder, "private transcript words"),
    ),
)
def test_test_decoder_injection_cannot_forge_or_leak_provenance(invalid):
    with pytest.raises(prepare.SpsPreparationError, match="decoder provenance"):
        prepare._test_decoder_binding(invalid)  # type: ignore[arg-type]


def test_preparer_code_binding_mismatch_fails_after_publication(tmp_path, monkeypatch):
    rows = _valid_rows()
    archive = _archive(tmp_path / "source.tar.gz", rows)
    _patch_release(monkeypatch, archive, rows)
    calls = 0

    def changing_binding() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"contract": "unit-code-v1", "generation": calls}

    monkeypatch.setattr(prepare, "_preparer_code_provenance", changing_binding)
    destination = tmp_path / "code-mismatch"
    with pytest.raises(prepare.SpsPreparationError, match="provenance changed"):
        prepare.prepare_common_voice_spontaneous_v4(
            archive_path=archive,
            output_dir=destination,
            accepted_terms=TERMS,
            api_receipt=_api_receipt(),
            test_decoder_injection=_injection(),
        )
    assert calls == 2
    assert destination.is_dir()


def test_default_decoder_closure_mismatch_fails_after_publication(
    tmp_path, monkeypatch
):
    rows = _valid_rows()
    archive = _archive(tmp_path / "source.tar.gz", rows)
    _patch_release(monkeypatch, archive, rows)
    monkeypatch.setattr(prepare, "_validate_catalog_contract", lambda: None)
    monkeypatch.setattr(prepare, "_decode_mp3_pyav", _fake_decoder)
    calls = 0

    def changing_decoder() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"contract": "unit-decoder-v1", "generation": calls}

    monkeypatch.setattr(prepare, "_pyav_decoder_provenance", changing_decoder)
    destination = tmp_path / "decoder-mismatch"
    with pytest.raises(prepare.SpsPreparationError, match="provenance changed"):
        prepare.prepare_common_voice_spontaneous_v4(
            archive_path=archive,
            output_dir=destination,
            accepted_terms=TERMS,
            api_receipt=_api_receipt(),
        )
    assert calls == 2
    assert destination.is_dir()


def test_decoder_tree_walk_stops_at_the_entry_limit(tmp_path, monkeypatch):
    root = tmp_path / "runtime-tree"
    root.mkdir()
    for index in range(3):
        (root / f"file-{index}.bin").write_bytes(b"bounded")
    monkeypatch.setattr(prepare, "_MAX_RUNTIME_TREE_ENTRIES", 2)

    with pytest.raises(prepare.SpsPreparationError, match="decoder provenance"):
        prepare._runtime_tree_digest(root)


def test_direct_pyav_decoder_emits_finite_mono_16k_f32le():
    av = pytest.importorskip("av", reason="optional pinned evaluation dependency")
    import numpy as np

    sample_rate = 16_000
    samples = np.sin(
        2.0 * np.pi * 440.0 * np.arange(sample_rate // 2) / sample_rate
    ).astype(np.float32)
    encoded = io.BytesIO()
    with av.open(encoded, mode="w", format="mp3") as container:
        stream = container.add_stream("mp3", rate=sample_rate)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(
            samples.reshape(1, -1),
            format="flt",
            layout="mono",
        )
        frame.sample_rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)

    decoded = prepare._decode_mp3_pyav(encoded.getvalue())
    prepare._verify_decoded_audio(decoded, duration_ms=500)
    assert decoded.samples > 0
    assert len(decoded.raw_f32le) == decoded.samples * 4


def test_decoder_receipt_binds_binary_trees_without_local_paths():
    pytest.importorskip("av", reason="optional pinned evaluation dependency")
    receipt = prepare._pyav_decoder_provenance()
    assert receipt["contract"] == "direct-pyav-mp3-mono-f32le-16000-v1"
    for key in (
        "pyav_package",
        "pyav_bundled_ffmpeg",
        "numpy_package",
        "numpy_bundled_libraries",
    ):
        assert receipt[key]["files"] > 0
        assert receipt[key]["bytes"] > 0
        assert len(receipt[key]["tree_sha256"]) == 64
    encoded = json.dumps(receipt)
    assert "/home/" not in encoded
    assert "site-packages" not in encoded


def test_invalid_cli_arguments_use_only_safe_json(tmp_path, capsys):
    private = str(tmp_path / "private-canary.tar.gz")
    code = prepare.main(
        [
            "--archive",
            private,
            "--output-dir",
            str(tmp_path / "output"),
            "--mdc-api-size-bytes",
            "private-invalid-integer",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert json.loads(captured.out) == prepare._SAFE_ERROR
    assert captured.err == ""
    assert private not in captured.out
    assert "private-invalid-integer" not in captured.out


def test_help_states_no_download_model_network_or_audio_device():
    help_text = " ".join(prepare._parser().format_help().split())
    assert "no download, model, network, or audio-device execution" in help_text
