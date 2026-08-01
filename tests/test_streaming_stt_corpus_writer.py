from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import struct

import pytest

from tools.streaming_stt.corpus import CorpusError, CorpusProvenance, load_corpus
from tools.streaming_stt import corpus_writer


def _provenance() -> CorpusProvenance:
    return CorpusProvenance(
        kind="public-voice-v1",
        suite="spontaneous-en",
        manifest_sha256="a" * 64,
        metadata_sha256="b" * 64,
        source_set_sha256="c" * 64,
    )


def _case(case_id: str = "speaker-01") -> corpus_writer.CorpusWriteCase:
    return corpus_writer.CorpusWriteCase(
        case_id=case_id,
        audio_bytes=struct.pack("<ffff", 0.0, 0.25, -0.25, 0.0),
        reference="a spontaneous answer",
        tags=("public-voice", "spontaneous-en"),
    )


def test_publishes_private_schema_v2_corpus_and_sidecar(tmp_path: Path):
    destination = tmp_path / "private-corpus"
    receipt = b'{"schema_version":1}\n'

    loaded = corpus_writer.publish_private_corpus(
        cases=(_case(),),
        provenance=_provenance(),
        output_dir=destination,
        purpose="spontaneous speech evaluation slice",
        sidecars={"preparation-receipt.json": receipt},
    )

    assert loaded.schema_version == 2
    assert loaded.cases[0].case_id == "speaker-01"
    assert loaded.cases[0].expected_text == "a spontaneous answer"
    assert loaded.provenance == _provenance()
    assert (destination.stat().st_mode & 0o777) == 0o700
    assert (destination / "preparation-receipt.json").read_bytes() == receipt
    assert {
        path.name: path.stat().st_mode & 0o777 for path in destination.iterdir()
    } == {
        "corpus.json": 0o600,
        "preparation-receipt.json": 0o600,
        "speaker-01.f32le": 0o600,
    }
    manifest = json.loads((destination / "corpus.json").read_text(encoding="utf-8"))
    assert manifest["provenance"] == _provenance().as_dict()


def test_private_diagnostic_provenance_publishes_schema_v3_only(tmp_path: Path):
    destination = tmp_path / "private-diagnostic-corpus"
    provenance = replace(
        _provenance(),
        kind="private-diagnostic-v1",
        suite="final-model-input",
    )

    loaded = corpus_writer.publish_private_corpus(
        cases=(_case(),),
        provenance=provenance,
        output_dir=destination,
        purpose="exact private diagnostic final input",
    )

    assert loaded.schema_version == 3
    assert loaded.provenance == provenance
    payload = json.loads((destination / "corpus.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["provenance"]["kind"] == "private-diagnostic-v1"

    payload["schema_version"] = 2
    (destination / "corpus.json").write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CorpusError):
        load_corpus(destination / "corpus.json")


def test_publication_preserves_explicit_command_targets(tmp_path: Path):
    loaded = corpus_writer.publish_private_corpus(
        cases=(replace(_case(), commands=("spontaneous answer",)),),
        provenance=_provenance(),
        output_dir=tmp_path / "command-corpus",
        purpose="command target preservation",
    )

    assert loaded.cases[0].commands == ("spontaneous answer",)


def test_arbitrary_sidecar_digest_is_not_the_source_set_digest(tmp_path: Path):
    destination = tmp_path / "private-corpus"
    receipt = b'{"schema_version":1,"kind":"independent-receipt"}\n'
    provenance = _provenance()
    assert hashlib.sha256(receipt).hexdigest() != provenance.source_set_sha256

    loaded = corpus_writer.publish_private_corpus(
        cases=(_case(),),
        provenance=provenance,
        output_dir=destination,
        purpose="independently bound sidecar",
        sidecars={"preparation-receipt.json": receipt},
    )

    assert loaded.provenance == provenance
    assert (destination / "preparation-receipt.json").read_bytes() == receipt


def test_publication_never_overwrites_existing_destination(tmp_path: Path):
    destination = tmp_path / "private-corpus"
    first = corpus_writer.publish_private_corpus(
        cases=(_case(),),
        provenance=_provenance(),
        output_dir=destination,
        purpose="first",
    )
    before = {path.name: path.read_bytes() for path in destination.iterdir()}

    with pytest.raises(corpus_writer.CorpusWriterError):
        corpus_writer.publish_private_corpus(
            cases=(_case(),),
            provenance=_provenance(),
            output_dir=destination,
            purpose="second",
        )

    assert first.path.read_bytes() == before["corpus.json"]
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == before


@pytest.mark.parametrize(
    "filename",
    ("../receipt.json", "/receipt.json", "corpus.json", "speaker-01.f32le"),
)
def test_unsafe_or_reserved_sidecar_is_rejected_before_output(
    tmp_path: Path,
    filename: str,
):
    destination = tmp_path / "private-corpus"

    with pytest.raises(corpus_writer.CorpusWriterError):
        corpus_writer.publish_private_corpus(
            cases=(_case(),),
            provenance=_provenance(),
            output_dir=destination,
            purpose="invalid sidecar",
            sidecars={filename: b"receipt\n"},
        )

    assert not destination.exists()


def test_loader_rejects_nonfinite_pcm_and_partial_evidence_is_retained(
    tmp_path: Path,
):
    destination = tmp_path / "private-corpus"
    invalid = corpus_writer.CorpusWriteCase(
        case_id="nonfinite",
        audio_bytes=struct.pack("<f", float("nan")),
        reference="invalid audio",
        tags=("public-voice",),
    )

    with pytest.raises(corpus_writer.CorpusWriterError):
        corpus_writer.publish_private_corpus(
            cases=(invalid,),
            provenance=_provenance(),
            output_dir=destination,
            purpose="loader validation",
        )

    assert destination.is_dir()
    assert (destination / "nonfinite.f32le").is_file()
    assert (destination / "corpus.json").is_file()


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    (
        ("kind", "unknown-private-v1"),
        ("suite", "Spontaneous-English"),
        ("manifest_sha256", "a" * 63),
        ("metadata_sha256", "B" * 64),
        ("source_set_sha256", "not-a-sha256"),
    ),
)
def test_invalid_provenance_is_rejected_before_output(
    tmp_path: Path,
    field_name: str,
    invalid: str,
):
    destination = tmp_path / "private-corpus"

    with pytest.raises(corpus_writer.CorpusWriterError):
        corpus_writer.publish_private_corpus(
            cases=(_case(),),
            provenance=replace(_provenance(), **{field_name: invalid}),
            output_dir=destination,
            purpose="invalid provenance",
        )

    assert not destination.exists()


@pytest.mark.parametrize("reference", ("", "   ", "?!—"))
def test_invalid_transcript_reference_is_rejected_before_output(
    tmp_path: Path,
    reference: str,
):
    destination = tmp_path / "private-corpus"

    with pytest.raises(corpus_writer.CorpusWriterError):
        corpus_writer.publish_private_corpus(
            cases=(replace(_case(), reference=reference),),
            provenance=_provenance(),
            output_dir=destination,
            purpose="invalid transcript reference",
        )

    assert not destination.exists()


def test_case_materialization_stops_at_one_over_the_limit(tmp_path: Path):
    destination = tmp_path / "private-corpus"
    yielded: list[int] = []

    def endless_cases():
        index = 0
        while True:
            yielded.append(index)
            yield _case(f"case-{index:03d}")
            index += 1

    with pytest.raises(corpus_writer.CorpusWriterError):
        corpus_writer.publish_private_corpus(
            cases=endless_cases(),
            provenance=_provenance(),
            output_dir=destination,
            purpose="bounded materialization",
        )

    assert len(yielded) == corpus_writer._MAX_CASES + 1
    assert not destination.exists()


def test_case_materialization_validates_before_requesting_another_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "private-corpus"
    yielded: list[str] = []

    def cases():
        yielded.append("oversized")
        yield _case("oversized")
        raise AssertionError("writer advanced past the invalid first case")

    monkeypatch.setattr(corpus_writer, "MAX_PCM_BYTES", 4)

    with pytest.raises(corpus_writer.CorpusWriterError):
        corpus_writer.publish_private_corpus(
            cases=cases(),
            provenance=_provenance(),
            output_dir=destination,
            purpose="bounded materialization",
        )

    assert yielded == ["oversized"]
    assert not destination.exists()


def test_oversized_serialized_manifest_is_rejected_before_output(tmp_path: Path):
    destination = tmp_path / "private-corpus"
    cases = tuple(
        replace(
            _case(f"case-{index:03d}"),
            reference="x" * corpus_writer._MAX_REFERENCE_CHARS,
        )
        for index in range(65)
    )

    with pytest.raises(corpus_writer.CorpusWriterError):
        corpus_writer.publish_private_corpus(
            cases=cases,
            provenance=_provenance(),
            output_dir=destination,
            purpose="manifest byte bound",
        )

    assert not destination.exists()


def test_serialized_manifest_accepts_exact_limit_and_rejects_next_byte(
    tmp_path: Path,
):
    def cases_for_size(target: int) -> tuple[corpus_writer.CorpusWriteCase, ...]:
        cases = [
            replace(_case(f"case-{index:03d}"), reference="x") for index in range(64)
        ]
        _rows, baseline = corpus_writer._serialize_manifest(
            tuple(cases),
            _provenance(),
            "manifest byte boundary",
        )
        remaining = target - len(baseline)
        assert 0 <= remaining <= len(cases) * (corpus_writer._MAX_REFERENCE_CHARS - 1)
        for index, case in enumerate(cases):
            added = min(remaining, corpus_writer._MAX_REFERENCE_CHARS - 1)
            cases[index] = replace(case, reference="x" * (added + 1))
            remaining -= added
        assert remaining == 0
        return tuple(cases)

    exact_destination = tmp_path / "exact-limit"
    exact_cases = cases_for_size(corpus_writer._MAX_CORPUS_MANIFEST_BYTES)
    loaded = corpus_writer.publish_private_corpus(
        cases=exact_cases,
        provenance=_provenance(),
        output_dir=exact_destination,
        purpose="manifest byte boundary",
    )

    assert loaded.path.stat().st_size == corpus_writer._MAX_CORPUS_MANIFEST_BYTES

    oversized_destination = tmp_path / "over-limit"
    with pytest.raises(corpus_writer.CorpusWriterError):
        corpus_writer.publish_private_corpus(
            cases=cases_for_size(corpus_writer._MAX_CORPUS_MANIFEST_BYTES + 1),
            provenance=_provenance(),
            output_dir=oversized_destination,
            purpose="manifest byte boundary",
        )

    assert not oversized_destination.exists()


def test_oversized_sidecar_is_rejected_before_output(tmp_path: Path):
    destination = tmp_path / "private-corpus"

    with pytest.raises(corpus_writer.CorpusWriterError):
        corpus_writer.publish_private_corpus(
            cases=(_case(),),
            provenance=_provenance(),
            output_dir=destination,
            purpose="sidecar byte bound",
            sidecars={
                "preparation-receipt.json": b"x"
                * (corpus_writer._MAX_SIDECAR_BYTES + 1)
            },
        )

    assert not destination.exists()


def test_mid_publication_failure_keeps_completed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "private-corpus"
    original = corpus_writer._write_new_private

    def fail_second(directory_fd: int, filename: str, payload: bytes):
        if filename == "second.f32le":
            raise corpus_writer.CorpusWriterError()
        return original(directory_fd, filename, payload)

    monkeypatch.setattr(corpus_writer, "_write_new_private", fail_second)

    with pytest.raises(corpus_writer.CorpusWriterError):
        corpus_writer.publish_private_corpus(
            cases=(_case("first"), _case("second")),
            provenance=_provenance(),
            output_dir=destination,
            purpose="injected failure",
        )

    assert (destination / "first.f32le").is_file()
    assert not (destination / "second.f32le").exists()
    assert not (destination / "corpus.json").exists()


def test_destination_replacement_during_validation_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "private-corpus"
    displaced = tmp_path / "displaced-corpus"
    original = corpus_writer.load_corpus

    def relocate_after_load(path: Path):
        loaded = original(path)
        destination.rename(displaced)
        destination.mkdir(mode=0o700)
        return loaded

    monkeypatch.setattr(corpus_writer, "load_corpus", relocate_after_load)

    with pytest.raises(corpus_writer.CorpusWriterError):
        corpus_writer.publish_private_corpus(
            cases=(_case(),),
            provenance=_provenance(),
            output_dir=destination,
            purpose="directory replacement race",
            sidecars={"preparation-receipt.json": b"receipt\n"},
        )

    assert list(destination.iterdir()) == []
    assert (displaced / "corpus.json").is_file()
    assert (displaced / "speaker-01.f32le").is_file()
    assert (displaced / "preparation-receipt.json").is_file()


def test_directory_replacement_between_creation_stat_and_open_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "private-corpus"
    displaced = tmp_path / "displaced-corpus"
    original = os.open
    replaced = False

    def replace_before_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if not replaced and path == destination.name and dir_fd is not None:
            replaced = True
            destination.rename(displaced)
            destination.mkdir(mode=0o700)
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replace_before_open)

    with pytest.raises(corpus_writer.CorpusWriterError):
        corpus_writer.publish_private_corpus(
            cases=(_case(),),
            provenance=_provenance(),
            output_dir=destination,
            purpose="directory creation race",
        )

    assert replaced
    assert list(destination.iterdir()) == []
    assert list(displaced.iterdir()) == []


def test_writes_remain_bound_to_original_directory_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "private-corpus"
    displaced = tmp_path / "displaced-corpus"
    original = corpus_writer._write_new_private
    replaced = False

    def relocate_before_audio(directory_fd: int, filename: str, payload: bytes):
        nonlocal replaced
        if filename == "speaker-01.f32le":
            destination.rename(displaced)
            destination.mkdir(mode=0o700)
            replaced = True
        return original(directory_fd, filename, payload)

    monkeypatch.setattr(corpus_writer, "_write_new_private", relocate_before_audio)

    with pytest.raises(corpus_writer.CorpusWriterError):
        corpus_writer.publish_private_corpus(
            cases=(_case(),),
            provenance=_provenance(),
            output_dir=destination,
            purpose="directory descriptor binding",
            sidecars={"preparation-receipt.json": b"receipt\n"},
        )

    assert replaced
    assert list(destination.iterdir()) == []
    assert {path.name for path in displaced.iterdir()} == {
        "corpus.json",
        "preparation-receipt.json",
        "speaker-01.f32le",
    }


def test_close_time_same_size_sidecar_corruption_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "private-corpus"
    receipt = b'{"schema_version":1}\n'
    corrupted = receipt.replace(b"1", b"2")
    assert len(corrupted) == len(receipt)
    original = corpus_writer.load_corpus

    def corrupt_after_load(path: Path):
        loaded = original(path)
        sidecar = destination / "preparation-receipt.json"
        descriptor = os.open(sidecar, os.O_WRONLY | os.O_TRUNC)
        try:
            assert os.write(descriptor, corrupted) == len(corrupted)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return loaded

    monkeypatch.setattr(corpus_writer, "load_corpus", corrupt_after_load)

    with pytest.raises(corpus_writer.CorpusWriterError):
        corpus_writer.publish_private_corpus(
            cases=(_case(),),
            provenance=_provenance(),
            output_dir=destination,
            purpose="sidecar corruption race",
            sidecars={"preparation-receipt.json": receipt},
        )

    assert (destination / "preparation-receipt.json").read_bytes() == corrupted


def test_close_time_same_content_sidecar_replacement_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "private-corpus"
    receipt = b'{"schema_version":1}\n'
    original = corpus_writer.load_corpus

    def replace_after_load(path: Path):
        loaded = original(path)
        sidecar = destination / "preparation-receipt.json"
        sidecar.unlink()
        sidecar.write_bytes(receipt)
        sidecar.chmod(0o600)
        return loaded

    monkeypatch.setattr(corpus_writer, "load_corpus", replace_after_load)

    with pytest.raises(corpus_writer.CorpusWriterError):
        corpus_writer.publish_private_corpus(
            cases=(_case(),),
            provenance=_provenance(),
            output_dir=destination,
            purpose="sidecar replacement race",
            sidecars={"preparation-receipt.json": receipt},
        )

    assert (destination / "preparation-receipt.json").read_bytes() == receipt
