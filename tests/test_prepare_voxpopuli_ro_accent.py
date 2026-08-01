from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import struct
import sys

import pytest

from tools import prepare_voxpopuli_ro_accent as prepare
from tools import voxpopuli_ro_accent_parquet_worker as worker
from tools.streaming_stt.corpus import load_corpus


TERMS = frozenset({"CC0-1.0", "EUROPEAN-PARLIAMENT-LEGAL-NOTICE"})
TINY_BUCKETS = (
    ("d0_2_6s", 2, 4, False, 6),
    ("d1_6_10s", 4, 6, False, 6),
    ("d2_10_15s", 6, 8, False, 6),
    ("d3_15_25s", 8, 10, True, 6),
)


def _pcm16_wav(samples: int, value: int = 1000) -> bytes:
    pcm = struct.pack(f"<{samples}h", *([value] * samples))
    fmt = struct.pack("<HHIIHH", 1, 1, 16_000, 32_000, 2, 16)
    payload = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    payload += b"data" + struct.pack("<I", len(pcm)) + pcm
    return b"RIFF" + struct.pack("<I", len(payload) + 4) + b"WAVE" + payload


def _wav_chunk(kind: bytes, payload: bytes) -> bytes:
    assert len(kind) == 4
    return kind + struct.pack("<I", len(payload)) + payload


def _riff_wave(*chunks: bytes) -> bytes:
    payload = b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(payload) + 4) + b"WAVE" + payload


def _float32_wav(
    samples: int,
    value: float = 0.25,
    *,
    fact_samples: int | None = None,
) -> bytes:
    pcm = struct.pack(f"<{samples}f", *([value] * samples))
    fmt = struct.pack("<HHIIHH", 3, 1, 16_000, 64_000, 4, 32) + b"\x00\x00"
    fact = struct.pack("<I", samples if fact_samples is None else fact_samples)
    return _riff_wave(
        _wav_chunk(b"fmt ", fmt),
        _wav_chunk(b"fact", fact),
        _wav_chunk(b"data", pcm),
    )


def _patch_tiny_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    payloads = (b"synthetic-shard-zero", b"synthetic-shard-one")
    paths: list[Path] = []
    shards: list[prepare.SourceShard] = []
    for index, payload in enumerate(payloads):
        path = tmp_path / f"source-{index}.parquet"
        path.write_bytes(payload)
        path.chmod(0o600)
        paths.append(path)
        shards.append(
            prepare.SourceShard(
                shard_index=index,
                filename=f"test-{index:05d}-of-00002.parquet",
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                download_url=f"https://example.invalid/source-{index}.parquet",
            )
        )
    monkeypatch.setattr(prepare, "SOURCE_SHARDS", tuple(shards))
    monkeypatch.setattr(prepare, "DURATION_BUCKETS", TINY_BUCKETS)
    monkeypatch.setattr(worker, "DURATION_BUCKETS", TINY_BUCKETS)
    return tuple(paths), tuple(payloads), tuple(shards)


def _fake_worker_result(output: Path, shards: tuple[prepare.SourceShard, ...]) -> None:
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    rows: list[dict[str, object]] = []
    candidates: list[tuple[str, str, str, int, str]] = []
    sample_by_bucket = {
        "d0_2_6s": 2,
        "d1_6_10s": 4,
        "d2_10_15s": 6,
        "d3_15_25s": 8,
    }
    for bucket_index, (bucket, *_rest) in enumerate(TINY_BUCKETS):
        bucket_rows: list[tuple[str, str, str, int, str]] = []
        for offset in range(6):
            speaker = f"private-speaker-{bucket_index}-{offset}"
            audio_id = f"private-audio-{bucket_index}-{offset}"
            transcript = f"known private reference {bucket_index} {offset}"
            bucket_rows.append(
                (
                    worker.speaker_bucket_rank(bucket, speaker),
                    worker.row_rank(speaker, audio_id),
                    speaker,
                    offset,
                    audio_id,
                )
            )
        candidates.extend(
            (
                speaker_rank,
                row_rank,
                speaker,
                bucket_index * 6 + offset,
                audio_id,
            )
            for speaker_rank, row_rank, speaker, offset, audio_id in sorted(bucket_rows)
        )
    for index, (speaker_rank, row_rank, speaker, original_index, audio_id) in enumerate(
        candidates
    ):
        bucket_index = original_index // 6
        offset = original_index % 6
        bucket = TINY_BUCKETS[bucket_index][0]
        samples = sample_by_bucket[bucket]
        transcript = f"known private reference {bucket_index} {offset}"
        wav = _float32_wav(samples, value=(index + 1) / 100.0)
        filename = f"case-{index:02d}.wav"
        path = output / filename
        path.write_bytes(wav)
        path.chmod(0o400)
        rows.append(
            {
                "selection_index": index,
                "shard_index": index % 2,
                "row_index": index,
                "audio_id": audio_id,
                "speaker_id": speaker,
                "gender": "female" if index % 2 else "male",
                "transcription": transcript,
                "transcription_sha256": hashlib.sha256(
                    transcript.encode("utf-8")
                ).hexdigest(),
                "duration_samples": samples,
                "duration_bucket": bucket,
                "row_rank_sha256": row_rank,
                "speaker_bucket_rank_sha256": speaker_rank,
                "audio_file": filename,
                "audio_sha256": hashlib.sha256(wav).hexdigest(),
                "audio_size_bytes": len(wav),
            }
        )
    result = {
        "schema_version": worker.PROTOCOL_VERSION,
        "pyarrow_version": worker.PYARROW_VERSION,
        "source_revision": worker.SOURCE_REVISION,
        "selection_seed": worker.SELECTION_SEED,
        "schema_contract_sha256": worker.SCHEMA_CONTRACT_SHA256,
        "source_total_rows": worker.SOURCE_TOTAL_ROWS,
        "source_shards": [
            {
                "shard_index": shard.shard_index,
                "filename": shard.filename,
                "size_bytes": shard.size_bytes,
                "sha256": shard.sha256,
                "rows": 4_193 if shard.shard_index == 0 else 4_194,
                "row_groups": 1,
            }
            for shard in shards
        ],
        "target_accent_speakers": worker.EXPECTED_EN_RO_SPEAKERS,
        "eligible_rows": 24,
        "eligible_speakers": 24,
        "eligible_set_sha256": "e" * 64,
        "bucket_counts": {name: quota for name, *_middle, quota in TINY_BUCKETS},
        "cases": rows,
    }
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result_path.chmod(0o400)


def _injection(shards: tuple[prepare.SourceShard, ...]):
    return prepare.TestWorkerInjection(
        lambda _descriptors, output: _fake_worker_result(output, shards),
        "synthetic-v1",
    )


def _corrupt_worker_result(
    output: Path,
    shards: tuple[prepare.SourceShard, ...],
    field: str,
) -> None:
    _fake_worker_result(output, shards)
    result_path = output / "result.json"
    result_path.chmod(0o600)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    case = result["cases"][0]
    if field == "audio_file":
        case[field] = "case-wrong.wav"
    elif field == "audio_size_bytes":
        case[field] += 4
    elif field == "audio_sha256":
        case[field] = "0" * 64
    else:  # pragma: no cover - test helper contract
        raise AssertionError(field)
    result_path.write_text(
        json.dumps(result, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result_path.chmod(0o400)


def _prepare_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths, _payloads, shards = _patch_tiny_sources(tmp_path, monkeypatch)
    result = prepare.prepare_voxpopuli_ro_accent(
        source_shard_0=paths[0],
        source_shard_1=paths[1],
        parquet_python=None,
        output_dir=tmp_path / "prepared",
        accepted_terms=TERMS,
        test_worker_injection=_injection(shards),
    )
    return result, paths, shards


def _candidate(
    speaker: str,
    audio: str,
    bucket: str,
    *,
    row: int,
) -> worker.Candidate:
    samples = {
        "d0_2_6s": 3 * 16_000,
        "d1_6_10s": 7 * 16_000,
        "d2_10_15s": 12 * 16_000,
        "d3_15_25s": 18 * 16_000,
    }[bucket]
    return worker.Candidate(
        shard_index=row % 2,
        row_index=row,
        audio_id=audio,
        speaker_id=speaker,
        gender="unknown",
        transcription=f"known words {row}",
        duration_samples=samples,
        duration_bucket=bucket,
        row_rank_sha256=worker.row_rank(speaker, audio),
        speaker_bucket_rank_sha256=worker.speaker_bucket_rank(bucket, speaker),
        audio_sha256=hashlib.sha256(audio.encode()).hexdigest(),
        audio_size_bytes=100,
    )


def _hf_metadata() -> dict[bytes, bytes]:
    values = {}
    for name in (
        "audio_id",
        "raw_text",
        "normalized_text",
        "gender",
        "speaker_id",
        "accent",
    ):
        values[name] = {"dtype": "string", "_type": "Value"}
    values["language"] = {
        "names": list(worker.LANGUAGE_NAMES),
        "_type": "ClassLabel",
    }
    values["audio"] = {"sampling_rate": 16_000, "_type": "Audio"}
    values["is_gold_transcript"] = {"dtype": "bool", "_type": "Value"}
    ordered = {name: values[name] for name in worker.FEATURE_NAMES}
    return {b"huggingface": json.dumps({"info": {"features": ordered}}).encode("utf-8")}


def test_exact_official_revision_artifacts_and_legal_provenance_are_pinned():
    assert worker.PROTOCOL_VERSION == 2
    assert worker.SCHEMA_CONTRACT["embedded_audio"] == {
        "container": "exact-riff-wave-v2",
        "chunk_order": [["fmt ", 18], ["fact", 4], ["data", "4*n"]],
        "fmt_le": [3, 1, 16_000, 64_000, 4, 32],
        "fmt_extra_hex": "0000",
        "fact_le": "uint32-samples-equals-data-bytes-div-4",
        "data": "little-endian-ieee-binary32",
        "empty_data": "valid-container-ineligible-row",
        "selected_values": "finite-no-amplitude-bound-no-transform",
    }
    assert prepare.SOURCE_REVISION == ("42f01879c780b4a2e90ec0b4f616c2ece526e4f1")
    assert [(item.size_bytes, item.sha256) for item in prepare.SOURCE_SHARDS] == [
        (
            2_471_743_754,
            "202d71bde2680f30aa7ae1050a98cdf85972458f94e6493db2aa7b8dd8ff46ec",
        ),
        (
            2_474_983_297,
            "7d61b4c7a2e5c565a382543236442a7a8dd081c985e52dd1084905e4becc3901",
        ),
    ]
    assert sum(item.size_bytes for item in prepare.SOURCE_SHARDS) == 4_946_727_051
    assert all(
        prepare.SOURCE_REVISION in item.download_url for item in prepare.SOURCE_SHARDS
    )
    assert prepare.CC0_URL.startswith("https://creativecommons.org/")
    assert prepare.EP_LEGAL_NOTICE_URL == (
        "https://www.europarl.europa.eu/legal-notice/en/"
    )
    assert prepare.REQUIRED_TERMS == TERMS


def test_preparer_provenance_binds_shared_process_boundary(monkeypatch):
    baseline = prepare._preparer_provenance()
    original = prepare._module_digest

    def changed_digest(path: Path) -> str:
        if path.name == "public_voice_fixtures.py":
            return "0" * 64
        return original(path)

    monkeypatch.setattr(prepare, "_module_digest", changed_digest)
    changed = prepare._preparer_provenance()

    assert "tools/public_voice_fixtures.py" in baseline["files"]
    assert changed != baseline


def test_selection_is_order_independent_duration_stratified_and_speaker_unique():
    candidates: list[worker.Candidate] = []
    row = 0
    for bucket_index, (bucket, *_rest) in enumerate(worker.DURATION_BUCKETS):
        for offset in range(7):
            candidates.append(
                _candidate(
                    f"speaker-{bucket_index}-{offset}",
                    f"audio-{bucket_index}-{offset}",
                    bucket,
                    row=row,
                )
            )
            row += 1
    # Cross-band candidates exercise assignment rather than greedy row picking.
    candidates.append(_candidate("speaker-0-0", "cross-a", "d1_6_10s", row=row))
    candidates.append(_candidate("speaker-1-0", "cross-b", "d2_10_15s", row=row + 1))

    selected = worker.select_candidates(candidates)
    reversed_selected = worker.select_candidates(tuple(reversed(candidates)))

    assert selected == reversed_selected
    assert len(selected) == 24
    assert len({item.speaker_id for item in selected}) == 24
    assert {
        bucket: sum(item.duration_bucket == bucket for item in selected)
        for bucket, *_rest in worker.DURATION_BUCKETS
    } == {bucket: 6 for bucket, *_rest in worker.DURATION_BUCKETS}
    assert [item.duration_bucket for item in selected] == [
        bucket for bucket, *_rest in worker.DURATION_BUCKETS for _index in range(6)
    ]


def test_rank_contract_is_nul_delimited_and_revision_bound():
    speaker = "speaker-id"
    audio = "audio-id"
    expected = hashlib.sha256(
        worker.SELECTION_SEED.encode("ascii")
        + b"\0"
        + worker.SOURCE_REVISION.encode("ascii")
        + b"\0row\0"
        + speaker.encode("utf-8")
        + b"\0"
        + audio.encode("utf-8")
    ).hexdigest()

    assert worker.row_rank(speaker, audio) == expected
    assert worker.row_rank(speaker, audio) != worker.row_rank(speaker, audio + "-2")


def test_huggingface_metadata_contract_is_semantically_exact():
    metadata = _hf_metadata()
    worker._validate_huggingface_metadata(metadata)

    payload = json.loads(metadata[b"huggingface"])
    payload["info"]["features"]["audio"]["sampling_rate"] = 48_000
    with pytest.raises(worker.WorkerError):
        worker._validate_huggingface_metadata(
            {b"huggingface": json.dumps(payload).encode("utf-8")}
        )

    payload = json.loads(metadata[b"huggingface"])
    payload["info"]["features"]["unexpected"] = {
        "dtype": "string",
        "_type": "Value",
    }
    with pytest.raises(worker.WorkerError):
        worker._validate_huggingface_metadata(
            {b"huggingface": json.dumps(payload).encode("utf-8")}
        )


def test_embedded_audio_path_requires_the_audio_id_wav_basename():
    assert worker._audio_path_matches("audio-id.wav", "audio-id")
    assert worker._audio_path_matches("nested/audio-id.wav", "audio-id")
    assert not worker._audio_path_matches("nested/other.wav", "audio-id")
    assert not worker._audio_path_matches("audio-id.mp3", "audio-id")


def test_exact_float32_wav_parser_preserves_finite_samples_without_clipping():
    wav = _float32_wav(4, value=-1.5)
    expected = struct.pack("<4f", -1.5, -1.5, -1.5, -1.5)

    assert worker._wav_samples(wav) == 4
    assert prepare._wav_to_f32le(wav, expected_samples=4) == expected


def test_exact_empty_float32_wav_is_valid_but_ineligible_and_not_decodable():
    wav = _float32_wav(0)

    assert worker._wav_samples(wav) == 0
    assert worker.duration_bucket(worker._wav_samples(wav)) is None
    with pytest.raises(prepare.VoxPopuliPreparationError):
        prepare._wav_to_f32le(wav, expected_samples=0)


def test_collect_candidates_skips_exact_empty_row_and_never_selects_it(monkeypatch):
    monkeypatch.setattr(worker, "SOURCE_TOTAL_ROWS", 25)
    monkeypatch.setattr(worker, "EXPECTED_EN_RO_SPEAKERS", 25)
    monkeypatch.setattr(worker, "DURATION_BUCKETS", TINY_BUCKETS)

    class FakeShard:
        def __init__(self, shard_index: int):
            self.record = {"shard_index": shard_index}

    shards = (FakeShard(0), FakeShard(1))
    metadata_rows: list[dict[str, object]] = []
    audio_rows: list[tuple[int, bytes, str]] = []
    for index in range(25):
        audio_id = f"audio-{index}"
        metadata_rows.append(
            {
                "audio_id": audio_id,
                "language": worker.LANGUAGE_ID,
                "raw_text": f"reference {index}",
                "normalized_text": f"reference {index}",
                "gender": "unknown",
                "speaker_id": f"speaker-{index}",
                "is_gold_transcript": True,
                "accent": worker.ACCENT,
            }
        )
        samples = 0 if index == 0 else (2, 4, 6, 8)[(index - 1) // 6]
        audio_rows.append((index, _float32_wav(samples), f"{audio_id}.wav"))

    def fake_metadata(shard):
        return iter(metadata_rows if shard.record["shard_index"] == 0 else ())

    def fake_audio(shard, requested):
        rows = audio_rows if shard.record["shard_index"] == 0 else ()
        return iter(row for row in rows if row[0] in requested)

    monkeypatch.setattr(worker, "_metadata_rows", fake_metadata)
    monkeypatch.setattr(worker, "_audio_rows", fake_audio)

    candidates, accent_speakers, eligible_speakers, _digest = (
        worker._collect_candidates(shards)
    )
    selected = worker.select_candidates(candidates)

    assert accent_speakers == 25
    assert eligible_speakers == 24
    assert len(candidates) == len(selected) == 24
    assert all(
        item.audio_id != "audio-0" and item.duration_samples > 0 for item in selected
    )


def test_pcm16_and_non_exact_float_wav_shapes_are_rejected_independently():
    exact = _float32_wav(4)
    fmt = exact[20:38]
    fact = exact[46:50]
    data = exact[58:]
    invalid = {
        "pcm16": _pcm16_wav(4),
        "wrong-riff-size": (exact[:4] + struct.pack("<I", len(exact) - 9) + exact[8:]),
        "missing-fmt": _riff_wave(
            _wav_chunk(b"fact", fact),
            _wav_chunk(b"data", data),
        ),
        "duplicate-fmt": _riff_wave(
            _wav_chunk(b"fmt ", fmt),
            _wav_chunk(b"fmt ", fmt),
            _wav_chunk(b"fact", fact),
            _wav_chunk(b"data", data),
        ),
        "short-fmt": _riff_wave(
            _wav_chunk(b"fmt ", fmt[:16]),
            _wav_chunk(b"fact", fact),
            _wav_chunk(b"data", data),
        ),
        "wrong-fmt-tag": _riff_wave(
            _wav_chunk(
                b"fmt ",
                struct.pack("<HHIIHH", 1, 1, 16_000, 64_000, 4, 32) + b"\x00\x00",
            ),
            _wav_chunk(b"fact", fact),
            _wav_chunk(b"data", data),
        ),
        "wrong-fmt-extra": _riff_wave(
            _wav_chunk(b"fmt ", fmt[:16] + b"\x01\x00"),
            _wav_chunk(b"fact", fact),
            _wav_chunk(b"data", data),
        ),
        "missing-fact": _riff_wave(
            _wav_chunk(b"fmt ", fmt),
            _wav_chunk(b"data", data),
        ),
        "duplicate-fact": _riff_wave(
            _wav_chunk(b"fmt ", fmt),
            _wav_chunk(b"fact", fact),
            _wav_chunk(b"fact", fact),
            _wav_chunk(b"data", data),
        ),
        "wrong-fact-size": _riff_wave(
            _wav_chunk(b"fmt ", fmt),
            _wav_chunk(b"fact", fact + b"\x00\x00\x00\x00"),
            _wav_chunk(b"data", data),
        ),
        "wrong-fact-value": _float32_wav(4, fact_samples=3),
        "wrong-order": _riff_wave(
            _wav_chunk(b"fact", fact),
            _wav_chunk(b"fmt ", fmt),
            _wav_chunk(b"data", data),
        ),
        "trailing-chunk": _riff_wave(
            _wav_chunk(b"fmt ", fmt),
            _wav_chunk(b"fact", fact),
            _wav_chunk(b"data", data),
            _wav_chunk(b"JUNK", b""),
        ),
    }

    for wav in invalid.values():
        with pytest.raises(worker.WorkerError):
            worker._wav_samples(wav)
        with pytest.raises(prepare.VoxPopuliPreparationError):
            prepare._wav_to_f32le(wav, expected_samples=4)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_selected_float32_wav_rejects_non_finite_samples(value: float):
    wav = _float32_wav(4, value=value)

    assert worker._wav_samples(wav) == 4
    with pytest.raises(prepare.VoxPopuliPreparationError):
        prepare._wav_to_f32le(wav, expected_samples=4)


def test_prepares_private_schema_v2_corpus_and_transcript_free_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    result, paths, _shards = _prepare_fixture(tmp_path, monkeypatch)
    corpus = result.corpus

    assert corpus.schema_version == 2
    assert len(corpus.cases) == 24
    assert corpus.path.stat().st_mode & 0o777 == 0o600
    assert corpus.path.parent.stat().st_mode & 0o777 == 0o700
    assert all(
        case.source_path.stat().st_mode & 0o777 == 0o600 for case in corpus.cases
    )
    assert len({case.case_id for case in corpus.cases}) == 24
    assert len({case.tags[-1] for case in corpus.cases}) == 4
    assert corpus.provenance is not None
    assert corpus.provenance.source_set_sha256 == result.receipt_sha256
    assert corpus.provenance.manifest_sha256 == result.selection_recipe_sha256
    assert corpus.provenance.metadata_sha256 == result.metadata_sha256
    assert load_corpus(corpus.path).digest == corpus.digest

    receipt_path = corpus.path.parent / "preparation-receipt.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert hashlib.sha256(receipt_bytes).hexdigest() == result.receipt_sha256
    assert receipt["license"]["dataset_license_id"] == "CC0-1.0"
    assert receipt["license"]["raw_source_legal_notice_url"] == (
        "https://www.europarl.europa.eu/legal-notice/en/"
    )
    assert receipt["selection"]["selected_distinct_speakers"] == 24
    assert set(receipt["preparer"]["files"]) == {
        "core/wer.py",
        "tools/public_voice_fixtures.py",
        "tools/prepare_voxpopuli_ro_accent.py",
        "tools/voxpopuli_ro_accent_parquet_worker.py",
        "tools/streaming_stt/bounded_io.py",
        "tools/streaming_stt/corpus.py",
        "tools/streaming_stt/corpus_writer.py",
        "tools/streaming_stt/protocol.py",
    }
    assert receipt["preparer"]["contract"] == ("voxpopuli-en-ro-private-preparer-v2")
    assert receipt["preparer"]["pcm_decoder"] == (
        "exact-ieee-float32-wav-finite-identity-16000-v2"
    )
    assert receipt["preparer"]["amplitude_policy"] == (
        "finite-binary32-no-bound-no-clipping-no-normalization"
    )
    assert all(
        {"gender_sha256", "shard_index", "source_row_index"}.isdisjoint(case)
        for case in receipt["cases"]
    )
    assert receipt["privacy"] == {
        "receipt_contains_transcripts": False,
        "receipt_contains_speaker_ids": False,
        "receipt_contains_audio_ids": False,
        "receipt_contains_local_paths": False,
        "cli_is_aggregate_only": True,
    }
    serialized = receipt_bytes.decode("utf-8")
    assert "known private reference" not in serialized
    assert "private-speaker" not in serialized
    assert "private-audio" not in serialized
    assert str(tmp_path) not in serialized
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in paths)
    safe = prepare._safe_result(result)
    assert str(tmp_path) not in json.dumps(safe)
    assert "known private reference" not in json.dumps(safe)


@pytest.mark.parametrize(
    "field",
    ["audio_file", "audio_size_bytes", "audio_sha256"],
)
def test_parent_rejects_worker_audio_path_size_or_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
):
    paths, _payloads, shards = _patch_tiny_sources(tmp_path, monkeypatch)
    destination = tmp_path / f"invalid-{field}"

    with pytest.raises(prepare.VoxPopuliPreparationError):
        prepare.prepare_voxpopuli_ro_accent(
            source_shard_0=paths[0],
            source_shard_1=paths[1],
            parquet_python=None,
            output_dir=destination,
            accepted_terms=TERMS,
            test_worker_injection=prepare.TestWorkerInjection(
                lambda _descriptors, output: _corrupt_worker_result(
                    output, shards, field
                ),
                f"invalid-{field.replace('_', '-')}",
            ),
        )

    assert not destination.exists()


def test_no_overwrite_and_missing_term_fail_before_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    result, paths, shards = _prepare_fixture(tmp_path, monkeypatch)
    before = result.corpus.path.read_bytes()
    calls = 0

    def should_not_run(_descriptors, _output):
        nonlocal calls
        calls += 1

    with pytest.raises(prepare.VoxPopuliPreparationError):
        prepare.prepare_voxpopuli_ro_accent(
            source_shard_0=paths[0],
            source_shard_1=paths[1],
            parquet_python=None,
            output_dir=result.corpus.path.parent,
            accepted_terms=TERMS,
            test_worker_injection=prepare.TestWorkerInjection(
                should_not_run, "no-overwrite-v1"
            ),
        )
    with pytest.raises(prepare.VoxPopuliPreparationError):
        prepare.prepare_voxpopuli_ro_accent(
            source_shard_0=paths[0],
            source_shard_1=paths[1],
            parquet_python=None,
            output_dir=tmp_path / "missing-terms",
            accepted_terms=frozenset({"CC0-1.0"}),
            test_worker_injection=_injection(shards),
        )
    assert calls == 0
    assert result.corpus.path.read_bytes() == before


def test_source_change_during_worker_is_rejected_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths, payloads, shards = _patch_tiny_sources(tmp_path, monkeypatch)

    def mutate(_descriptors: tuple[int, ...], output: Path) -> None:
        paths[0].write_bytes(b"X" * len(payloads[0]))
        _fake_worker_result(output, shards)

    destination = tmp_path / "changed-source"
    with pytest.raises(prepare.VoxPopuliPreparationError, match="source verification"):
        prepare.prepare_voxpopuli_ro_accent(
            source_shard_0=paths[0],
            source_shard_1=paths[1],
            parquet_python=None,
            output_dir=destination,
            accepted_terms=TERMS,
            test_worker_injection=prepare.TestWorkerInjection(mutate, "mutate-v1"),
        )

    assert not destination.exists()


@pytest.mark.parametrize("kind", ["broad-mode", "symlink", "hardlink"])
def test_source_overrides_must_be_canonical_private_single_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
):
    paths, _payloads, shards = _patch_tiny_sources(tmp_path, monkeypatch)
    source = paths[0]
    if kind == "broad-mode":
        source.chmod(0o644)
        supplied = source
    elif kind == "symlink":
        supplied = tmp_path / "source-link.parquet"
        supplied.symlink_to(source)
    else:
        supplied = tmp_path / "source-hard.parquet"
        os.link(source, supplied)

    with pytest.raises(prepare.VoxPopuliPreparationError):
        prepare.prepare_voxpopuli_ro_accent(
            source_shard_0=supplied,
            source_shard_1=paths[1],
            parquet_python=None,
            output_dir=tmp_path / f"invalid-{kind}",
            accepted_terms=TERMS,
            test_worker_injection=_injection(shards),
        )


def test_real_worker_runner_is_isolated_descriptor_only_and_one_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths, _payloads, _shards = _patch_tiny_sources(tmp_path, monkeypatch)
    monkeypatch.setattr(
        prepare.public_fixtures,
        "_validate_parquet_python",
        lambda _value: Path(sys.executable),
    )
    monkeypatch.setenv("SHOULD_NOT_REACH_WORKER", "secret")
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 999_991

        def __init__(self, argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            self.returncode = None

        def wait(self, timeout=None):
            captured.setdefault("timeouts", []).append(timeout)
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(prepare.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        prepare.public_fixtures,
        "_terminate_worker_process_group",
        lambda process: captured.setdefault("reaped", process.pid),
    )
    with prepare._opened_sources(paths) as handles:
        with prepare._run_worker(
            handles,
            output_parent=tmp_path,
            parquet_python=Path(sys.executable),
            test_worker_injection=None,
        ) as run:
            assert run.binding["production_evidence"] is True
            assert run.output_dir.is_dir()

    argv = captured["argv"]
    kwargs = captured["kwargs"]
    assert argv[:3] == [str(Path(sys.executable)), "-I", "-B"]
    assert Path(argv[3]).name == "voxpopuli_ro_accent_parquet_worker.py"
    assert kwargs["stdin"] is prepare.subprocess.DEVNULL
    assert kwargs["stdout"] is prepare.subprocess.DEVNULL
    assert kwargs["stderr"] is prepare.subprocess.DEVNULL
    assert kwargs["close_fds"] is True
    assert kwargs["start_new_session"] is True
    assert kwargs["pass_fds"]
    assert "shell" not in kwargs
    assert "SHOULD_NOT_REACH_WORKER" not in kwargs["env"]
    assert [
        kwargs["env"][name]
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "ARROW_NUM_THREADS",
        )
    ] == ["1", "1", "1", "1", "1"]
    assert captured["timeouts"] == [prepare._WORKER_TIMEOUT_SEC]
    assert captured["reaped"] == FakeProcess.pid


def test_worker_request_is_strict_and_contains_descriptors_not_paths(tmp_path: Path):
    request = {
        "schema_version": worker.PROTOCOL_VERSION,
        "source_revision": worker.SOURCE_REVISION,
        "selection_seed": worker.SELECTION_SEED,
        "source_descriptors": [
            {
                "shard_index": item["shard_index"],
                "descriptor": 10 + int(item["shard_index"]),
                "filename": item["filename"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in worker.SOURCE_SHARDS
        ],
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    path.chmod(0o400)

    loaded = worker._load_request(path)

    assert loaded == request
    assert "path" not in json.dumps(loaded).casefold()
    path.chmod(0o600)
    path.write_text(
        '{"schema_version":2,"schema_version":2}',
        encoding="utf-8",
    )
    path.chmod(0o400)
    with pytest.raises(worker.WorkerError):
        worker._load_request(path)


def test_help_and_runtime_dependencies_have_no_download_or_pyarrow_leak():
    help_text = " ".join(prepare._parser().format_help().split())

    assert "no download, network, model" in help_text
    assert "--source-shard-0" in help_text
    assert "--source-shard-1" in help_text
    assert "--parquet-python" in help_text
    assert "--accept-download" not in help_text
    assert "urllib" not in inspect.getsource(prepare)
    assert "requests" not in inspect.getsource(prepare)
    assert "import pyarrow" not in inspect.getsource(prepare)
    for dependency in (
        Path("requirements.txt"),
        Path("requirements-agent.txt"),
        Path("requirements-ondevice.txt"),
        Path("requirements-remote.txt"),
    ):
        assert "pyarrow" not in dependency.read_text(encoding="utf-8").casefold()


def test_cli_failure_is_aggregate_only(capsys, tmp_path: Path):
    secret = tmp_path / "private-sentinel.parquet"

    assert (
        prepare.main(
            [
                "--source-shard-0",
                str(secret),
                "--source-shard-1",
                str(secret),
                "--parquet-python",
                str(secret),
                "--output-dir",
                str(tmp_path / "output"),
                "--accept-term",
                "CC0-1.0",
                "--accept-term",
                "EUROPEAN-PARLIAMENT-LEGAL-NOTICE",
            ]
        )
        == 2
    )
    output = capsys.readouterr().out
    assert json.loads(output) == prepare._SAFE_ERROR
    assert str(tmp_path) not in output
    assert "sentinel" not in output
