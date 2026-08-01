from __future__ import annotations

from array import array
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import wave
import zipfile

import numpy as np
import pytest

from tools import prepare_demand_noise_streaming_stt_corpus as prepare
from tools.streaming_stt.corpus import CorpusProvenance
from tools.streaming_stt.corpus_writer import CorpusWriteCase, publish_private_corpus


TERMS = frozenset({"CC-BY-SA-3.0"})


def _wav(seed: int, *, samples: int = 3 * 16_000) -> bytes:
    values = array(
        "h",
        (((index * (seed * 13 + 17)) % 20_001) - 10_000 for index in range(samples)),
    )
    if sys.byteorder != "little":
        values.byteswap()
    raw = values.tobytes()
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(raw)
    return output.getvalue()


def _archive(tmp_path: Path, artifact_id: str, root: str, seed: int):
    filename = f"{root}_16k.zip"
    path = tmp_path / filename
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        directory = zipfile.ZipInfo(f"{root}/", date_time=(2024, 1, 1, 0, 0, 0))
        directory.external_attr = (0o40700 << 16) | 0x10
        archive.writestr(directory, b"")
        for channel in range(1, 17):
            info = zipfile.ZipInfo(
                f"{root}/ch{channel:02d}.wav",
                date_time=(2024, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100600 << 16
            archive.writestr(info, _wav(seed + channel))
    path.chmod(0o600)
    payload = path.read_bytes()
    return path, prepare.DemandArchive(
        artifact_id=artifact_id,
        filename=filename,
        root=root,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _source_corpus(tmp_path: Path) -> Path:
    source = tmp_path / "source-corpus"
    provenance = CorpusProvenance(
        kind="public-voice-v1",
        suite="test-commands",
        manifest_sha256="a" * 64,
        metadata_sha256="b" * 64,
        source_set_sha256="c" * 64,
    )
    cases = []
    for index, text in enumerate(("sentinel first command", "sentinel second command")):
        values = [0.0] * 160 + [0.1 + index * 0.02] * 15_680 + [0.0] * 160
        cases.append(
            CorpusWriteCase(
                case_id=f"source-{index}",
                audio_bytes=struct.pack(f"<{len(values)}f", *values),
                reference=text,
                tags=("public-voice", "commands"),
                commands=("command",),
            )
        )
    return publish_private_corpus(
        cases=tuple(cases),
        provenance=provenance,
        output_dir=source,
        purpose="test command source",
    ).path


def _inputs(tmp_path: Path):
    rows = (
        _archive(tmp_path, "dkitchen", "DKITCHEN", 1),
        _archive(tmp_path, "dliving", "DLIVING", 2),
        _archive(tmp_path, "dwashing", "DWASHING", 3),
    )
    paths = tuple(row[0] for row in rows)
    injection = prepare.TestDemandInjection(
        archives=tuple(row[1] for row in rows),
        fixture_id="test-demand-three-environments",
    )
    return paths, injection


def test_publishes_deterministic_three_environment_strata(tmp_path: Path):
    source = _source_corpus(tmp_path)
    paths, injection = _inputs(tmp_path)

    first = prepare.prepare_demand_noise_corpus(
        source_corpus=source,
        archive_paths=paths,
        output_dir=tmp_path / "noise-a",
        accepted_terms=TERMS,
        test_injection=injection,
    )
    second = prepare.prepare_demand_noise_corpus(
        source_corpus=source,
        archive_paths=paths,
        output_dir=tmp_path / "noise-b",
        accepted_terms=TERMS,
        test_injection=injection,
    )

    assert first.production_sources is False
    assert first.corpus.digest == second.corpus.digest
    assert first.receipt_sha256 == second.receipt_sha256
    assert len(first.corpus.cases) == 6
    assert first.corpus.audio_bytes == 6 * 16_000 * 4
    assert {case.commands for case in first.corpus.cases} == {("command",)}
    assert {tag for case in first.corpus.cases for tag in case.tags} >= {
        "noise",
        "demand",
        "dkitchen",
        "dliving",
        "dwashing",
        "snr0",
        "snr10",
        "snr20",
    }
    assert [
        sorted(tag for tag in case.tags if tag.startswith("snr"))
        for case in first.corpus.cases
    ] == [["snr20"], ["snr10"], ["snr0"], ["snr10"], ["snr0"], ["snr20"]]

    receipt_path = first.corpus.path.parent / "preparation-receipt.json"
    receipt_raw = receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)
    assert hashlib.sha256(receipt_raw).hexdigest() == first.receipt_sha256
    assert receipt["production_sources"] is False
    assert receipt["dataset"] == {
        "accepted_terms": ["CC-BY-SA-3.0"],
        "canonical_url": "https://zenodo.org/records/1227121",
        "dataset_id": "demand_domestic_16k",
        "doi": "10.5281/zenodo.1227121",
        "license_id": "CC-BY-SA-3.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/3.0/legalcode",
        "upstream_id": "zenodo:1227121:DKITCHEN+DLIVING+DWASHING",
    }
    assert receipt["accepted_terms"] == ["CC-BY-SA-3.0"]
    assert receipt["preparer"]["contract"] == (
        "demand-domestic-private-noise-preparer-v3"
    )
    assert set(receipt["preparer"]["files"]) == {
        "core/wer.py",
        "tools/prepare_demand_noise_streaming_stt_corpus.py",
        "tools/public_voice_eval_matrix.py",
        "tools/streaming_stt/bounded_io.py",
        "tools/streaming_stt/corpus.py",
        "tools/streaming_stt/corpus_writer.py",
        "tools/streaming_stt/protocol.py",
    }
    assert all(
        len(digest) == 64 and int(digest, 16) >= 0
        for digest in receipt["preparer"]["files"].values()
    )
    assert receipt["runtime"]["contract"] == (
        "cpython-stdlib-numpy-pcm16-f32le-noise-mix-v2"
    )
    assert receipt["runtime"]["python"]["implementation"] == (
        sys.implementation.name
    )
    assert receipt["runtime"]["numpy"]["version"] == np.__version__
    assert receipt["runtime"]["numpy"]["input_dtype"] == np.dtype("<i2").str
    assert receipt["runtime"]["numpy"]["output_dtype"] == np.dtype("<f4").str
    stdlib = receipt["runtime"]["python"]["stdlib_modules"]
    assert set(stdlib) == set(prepare._STDLIB_RUNTIME_MODULES)
    assert {"json", "math", "wave", "zipfile", "zlib"} <= set(stdlib)
    for binding in stdlib.values():
        if binding["implementation"] == "file":
            assert set(binding) == {"implementation", "sha256"}
            assert len(binding["sha256"]) == 64
            assert int(binding["sha256"], 16) >= 0
        else:
            assert binding == {
                "implementation": binding["implementation"],
                "bound_by": "python_executable_sha256",
            }
            assert binding["implementation"] in {"built-in", "frozen"}
    numpy_package = receipt["runtime"]["numpy"]["package"]
    assert numpy_package["files"] > 0
    assert numpy_package["bytes"] > 0
    assert len(numpy_package["tree_sha256"]) == 64
    numpy_libraries = receipt["runtime"]["numpy"]["bundled_libraries"]
    assert type(numpy_libraries["present"]) is bool
    if numpy_libraries["present"]:
        assert numpy_libraries["files"] > 0
        assert numpy_libraries["bytes"] > 0
        assert len(numpy_libraries["tree_sha256"]) == 64
    assert receipt["transform"]["non_overlapping_intervals_per_environment"] is True
    assert "sentinel first command" not in receipt_raw.decode("ascii")
    assert "sentinel second command" not in receipt_raw.decode("ascii")
    assert str(tmp_path) not in receipt_raw.decode("ascii")
    for environment in ("dkitchen", "dliving", "dwashing"):
        rows = [row for row in receipt["cases"] if row["environment"] == environment]
        intervals = sorted(
            (row["noise_sample_offset"], row["noise_sample_offset"] + row["source_samples"])
            for row in rows
        )
        assert intervals[0][1] <= intervals[1][0]
    assert {row["target_snr_db"] for row in receipt["cases"]} == {0, 10, 20}
    assert all(
        row["measured_snr_db"] == pytest.approx(row["target_snr_db"], abs=1e-6)
        for row in receipt["cases"]
    )


def test_missing_terms_or_changed_archive_fails_before_output(tmp_path: Path):
    source = _source_corpus(tmp_path)
    paths, injection = _inputs(tmp_path)

    with pytest.raises(prepare.DemandPreparationError):
        prepare.prepare_demand_noise_corpus(
            source_corpus=source,
            archive_paths=paths,
            output_dir=tmp_path / "missing-terms",
            accepted_terms=frozenset(),
            test_injection=injection,
        )
    assert not (tmp_path / "missing-terms").exists()

    paths[0].chmod(0o600)
    with paths[0].open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(prepare.DemandPreparationError):
        prepare.prepare_demand_noise_corpus(
            source_corpus=source,
            archive_paths=paths,
            output_dir=tmp_path / "changed-source",
            accepted_terms=TERMS,
            test_injection=injection,
        )
    assert not (tmp_path / "changed-source").exists()


def test_bad_zip_layout_and_private_mode_fail_closed(tmp_path: Path):
    source = _source_corpus(tmp_path)
    paths, injection = _inputs(tmp_path)

    paths[1].chmod(0o644)
    with pytest.raises(prepare.DemandPreparationError):
        prepare.prepare_demand_noise_corpus(
            source_corpus=source,
            archive_paths=paths,
            output_dir=tmp_path / "public-mode",
            accepted_terms=TERMS,
            test_injection=injection,
        )
    assert not (tmp_path / "public-mode").exists()


def test_output_must_be_absolute_new_and_outside_any_git_worktree(tmp_path: Path):
    with pytest.raises(prepare.DemandPreparationError):
        prepare._validated_output_path(Path("relative-demand-output"))
    with pytest.raises(prepare.DemandPreparationError):
        prepare._validated_output_path(prepare._REPO_ROOT / "private-demand-output")

    nested_git = tmp_path / "nested-git"
    (nested_git / ".git").mkdir(parents=True)
    (nested_git / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    with pytest.raises(prepare.DemandPreparationError):
        prepare._validated_output_path(nested_git / "private-demand-output")

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(prepare.DemandPreparationError):
        prepare._validated_output_path(existing)


def test_close_rechecks_archive_and_runtime_bindings(tmp_path: Path, monkeypatch):
    paths, injection = _inputs(tmp_path)
    noise = prepare._load_noise(paths[0], injection.archives[0])
    paths[0].touch()
    with pytest.raises(prepare.DemandPreparationError):
        prepare._recheck_noise_sources((noise,))

    source = _source_corpus(tmp_path)
    original_runtime = prepare._runtime_contract
    calls = 0

    def changing_runtime():
        nonlocal calls
        calls += 1
        binding = deepcopy(original_runtime())
        if calls > 1:
            binding["python"]["hexversion"] += 1
        return binding

    monkeypatch.setattr(prepare, "_runtime_contract", changing_runtime)
    with pytest.raises(prepare.DemandPreparationError):
        prepare.prepare_demand_noise_corpus(
            source_corpus=source,
            archive_paths=paths,
            output_dir=tmp_path / "changed-runtime",
            accepted_terms=TERMS,
            test_injection=injection,
        )
    assert calls == 2


def test_catalog_contract_and_safe_cli_error(tmp_path: Path, capsys):
    prepare._validate_catalog_contract()

    private = tmp_path / "sentinel-private-source"
    private.write_bytes(b"not a corpus")
    code = prepare.main(
        [
            "--source-corpus",
            str(private),
            "--dkitchen",
            str(private),
            "--dliving",
            str(private),
            "--dwashing",
            str(private),
            "--output-dir",
            str(tmp_path / "output"),
            "--accept-term",
            "CC-BY-SA-3.0",
        ]
    )

    output = capsys.readouterr().out
    assert code == 2
    assert json.loads(output) == prepare._SAFE_ERROR
    assert "sentinel-private-source" not in output
