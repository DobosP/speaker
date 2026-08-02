from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import struct

import pytest

from tools import public_conversation_fixture as fixture
from tools import streaming_stt_suite
from tools.public_voice_eval_matrix import dataset_by_id, selection_policy_sha256
from tools.streaming_stt.corpus import CorpusProvenance
from tools.streaming_stt.corpus_writer import CorpusWriteCase, publish_private_corpus


def _sha(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


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


def _artifact_receipts(dataset_id: str) -> list[dict[str, object]]:
    dataset = dataset_by_id(dataset_id)
    result: list[dict[str, object]] = []
    for artifact in dataset.artifacts:
        local_sha = _sha(f"{dataset_id}:{artifact.artifact_id}:content")
        if artifact.checksum is None:
            algorithm = "sha256"
            upstream_digest = local_sha
        else:
            algorithm = artifact.checksum.algorithm.value
            upstream_digest = artifact.checksum.digest
            if algorithm == "sha256":
                local_sha = upstream_digest
        result.append(
            {
                "artifact_id": artifact.artifact_id,
                "upstream_algorithm": algorithm,
                "upstream_digest": upstream_digest,
                "size_bytes": artifact.size_bytes or 1,
                "local_sha256": local_sha,
            }
        )
    return result


def _attributes(
    source_id: str,
    *,
    ordinal: int,
    stratum: str,
    channel: str | None,
) -> tuple[str | None, dict[str, object]]:
    if source_id == "harper-valley":
        task_index = ordinal // 3
        return _sha(f"speaker:{source_id}:{ordinal}"), {
            "task_type_index": task_index,
            "task_type_sha256": _sha(f"task:{task_index}"),
            "channel": "caller",
            "pii_scan_passed": True,
        }
    if source_id == "edacc-test":
        return _sha(f"speaker:{source_id}:{ordinal}"), {
            "stratum": stratum,
            "accent_sha256": _sha(f"accent:{ordinal % 8}"),
            "actual_overlap": False,
            "reference_source": "official_filtered_stm",
            "scoreability": "hard_wer",
        }
    if source_id == "ami-es2004a-close-far":
        window = ordinal // 2
        speaker = _sha(f"speaker:{source_id}:{window}") if stratum == "isolated" else None
        return speaker, {
            "window_index": window,
            "window_sha256": _sha(f"window:{window}"),
            "window_kind": stratum,
            "channel": channel,
            "start_sample": window * 2_000,
            "end_sample": window * 2_000 + 1_600,
            "actual_overlap": False,
            "scoreability": "hard_wer",
        }
    if source_id == "cvss4-en":
        duration = {
            "d0_2_6s": 4_000,
            "d1_6_10s": 8_000,
            "d2_10_15s": 12_000,
            "d3_15_25s": 20_000,
        }[stratum]
        return _sha(f"speaker:{source_id}:{ordinal}"), {
            "duration_bucket": stratum,
            "duration_ms": duration,
            "prompt_sha256": _sha(f"prompt:{ordinal}"),
        }
    raise AssertionError(source_id)


ReceiptMutator = Callable[[dict[str, object]], None]


def _publish_source(
    root: Path,
    lock: fixture.FixtureLock,
    source_id: str,
    *,
    mutate_receipt: ReceiptMutator | None = None,
) -> Path:
    """Publish production-shaped validator grammar, never source evidence."""

    source = lock.source_by_id(source_id)
    dataset = dataset_by_id(source.dataset_id)
    slots = lock.slots_for(source_id)
    write_cases: list[CorpusWriteCase] = []
    receipt_cases: list[dict[str, object]] = []
    total_samples = 0

    for slot in slots:
        pair_reference = (
            slot.ordinal // 2
            if source_id == "ami-es2004a-close-far"
            else slot.ordinal
        )
        reference = f"known reference {source_id} {pair_reference}"
        case_id = f"{source_id}-{slot.ordinal:02d}"
        speaker, attributes = _attributes(
            source_id,
            ordinal=slot.ordinal,
            stratum=slot.stratum,
            channel=slot.channel,
        )
        if source_id == "ami-es2004a-close-far":
            samples = int(attributes["end_sample"]) - int(attributes["start_sample"])
        elif source_id == "cvss4-en":
            samples = int(attributes["duration_ms"]) * 16
        else:
            samples = 1_600
        audio = struct.pack("<f", 0.1 + (slot.ordinal % 4) * 0.1) * samples
        extra_tags = [slot.stratum]
        if source_id == "ami-es2004a-close-far" and slot.channel is not None:
            extra_tags.append(slot.channel)
        tags = tuple(dict.fromkeys((*source.required_tags, *extra_tags)))
        write_cases.append(
            CorpusWriteCase(
                case_id=case_id,
                audio_bytes=audio,
                reference=reference,
                tags=tags,
            )
        )
        receipt_cases.append(
            {
                "slot_id": slot.slot_id,
                "case_id": case_id,
                "identity_sha256": _sha(f"identity:{source_id}:{slot.ordinal}"),
                "speaker_sha256": speaker,
                "reference_sha256": _sha(reference),
                "source_audio_sha256": _sha(f"source-audio:{source_id}:{slot.ordinal}"),
                "source_audio_bytes": len(audio),
                "pcm_sha256": _sha(audio),
                "pcm_samples": len(audio) // 4,
                "tags": list(tags),
                "attributes": attributes,
            }
        )
        total_samples += len(audio) // 4

    metadata_sha = _sha(f"metadata:{source_id}")
    preparer_files = {
        relative: fixture._repo_file_sha256(relative)
        for relative in source.preparer_files
    }
    receipt: dict[str, object] = {
        "schema_version": 1,
        "kind": fixture.RECEIPT_KIND,
        "fixture_id": fixture.FIXTURE_ID,
        "lock_recipe_sha256": lock.recipe_sha256,
        "source_id": source_id,
        "source_recipe_sha256": source.selection_recipe_sha256,
        "dataset_contract_sha256": source.dataset_contract_sha256,
        "accepted_terms": sorted(dataset.license.acceptance_ids),
        "artifacts": _artifact_receipts(source.dataset_id),
        "metadata_sha256": metadata_sha,
        "selection": {
            "policy_sha256": selection_policy_sha256(dataset.selection),
            "eligible_rows": 48,
            "eligible_set_sha256": _sha(f"eligible:{source_id}"),
            "selected_rows_sha256": _sha(f"selected:{source_id}"),
            "selected_cases": 24,
            "selected_case_set_sha256": fixture._canonical_sha256(receipt_cases),
            "parent_receipt_sha256": (
                _sha("synthetic-cv-parent-receipt")
                if source_id == "cvss4-en"
                else None
            ),
            "parent_corpus_manifest_sha256": (
                _sha("synthetic-cv-parent-corpus")
                if source_id == "cvss4-en"
                else None
            ),
        },
        "preparer": {
            "contract": source.preparer_contract,
            "files": preparer_files,
        },
        "decoder": {
            "contract": source.decoder_contract,
            "closure_sha256": fixture.decoder_closure_sha256(
                source.decoder_contract,
                preparer_files,
            ),
            "files": 1,
            "bytes": 1,
            "production_evidence": True,
        },
        "cases": receipt_cases,
        "totals": {
            "cases": 24,
            "pcm_samples": total_samples,
            "pcm_bytes": total_samples * 4,
        },
        "privacy": dict(fixture._PRIVATE_RECEIPT_PRIVACY),
        "evidence_scope": dict(fixture._RECEIPT_EVIDENCE_SCOPE),
    }
    if mutate_receipt is not None:
        mutate_receipt(receipt)
        receipt["selection"]["selected_case_set_sha256"] = fixture._canonical_sha256(
            receipt["cases"]
        )
    receipt_raw = _canonical(receipt)
    provenance = CorpusProvenance(
        kind="public-voice-v1",
        suite=source.corpus_suite,
        manifest_sha256=source.selection_recipe_sha256,
        metadata_sha256=metadata_sha,
        source_set_sha256=_sha(receipt_raw),
    )
    loaded = publish_private_corpus(
        cases=write_cases,
        provenance=provenance,
        output_dir=root / source_id,
        purpose=f"Synthetic contract fixture for {source_id}",
        sidecars={"preparation-receipt.json": receipt_raw},
    )
    return loaded.path


def _publish_set(
    tmp_path: Path,
    *,
    source_mutator: tuple[str, ReceiptMutator] | None = None,
) -> tuple[fixture.FixtureLock, tuple[Path, ...]]:
    lock = fixture.load_fixture_lock()
    paths = tuple(
        _publish_source(
            tmp_path,
            lock,
            source_id,
            mutate_receipt=(
                source_mutator[1]
                if source_mutator is not None and source_mutator[0] == source_id
                else None
            ),
        )
        for source_id in fixture.SOURCE_IDS
    )
    return lock, paths


def test_committed_lock_binds_four_sources_and_all_96_abstract_slots():
    lock = fixture.load_fixture_lock()

    assert lock.recipe_sha256 == (
        "7a35945d20bb1e91f251edac155d126ca6cb8abe44880e359be5725f48ca212e"
    )
    assert tuple(item.source_id for item in lock.sources) == fixture.SOURCE_IDS
    assert tuple(item.dataset_id for item in lock.sources) == fixture.DATASET_IDS
    assert len(lock.slots) == 96
    assert len({item.slot_id for item in lock.slots}) == 96
    assert all(len(lock.slots_for(source_id)) == 24 for source_id in fixture.SOURCE_IDS)

    encoded = fixture.DEFAULT_LOCK.read_text(encoding="utf-8")
    assert "expected_text" not in encoded
    assert "source_path" not in encoded
    assert "/home/" not in encoded
    assert ".wav" not in encoded
    assert "human_transcript" in encoded
    assert all(
        item.dataset_contract_sha256
        == fixture.dataset_contract_sha256(dataset_by_id(item.dataset_id))
        for item in lock.sources
    )
    assert all(
        item.preparer_files == fixture._SOURCE_PREPARER_FILES[item.source_id]
        and item.preparer_contract
        == fixture._SOURCE_PREPARER_CONTRACTS[item.source_id]
        and item.decoder_contract
        == fixture._SOURCE_DECODER_CONTRACTS[item.source_id]
        for item in lock.sources
    )


def test_lock_rejects_duplicate_keys_paths_digest_and_catalog_drift(tmp_path, monkeypatch):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(fixture.FixtureError):
        fixture.load_fixture_lock(duplicate)

    root = json.loads(fixture.DEFAULT_LOCK.read_text(encoding="utf-8"))
    root["sources"][0]["selection_recipe"]["eligibility"][0] = "/private/canary"
    unsafe = tmp_path / "unsafe.json"
    unsafe.write_bytes(_canonical(root))
    with pytest.raises(fixture.FixtureError):
        fixture.load_fixture_lock(unsafe)

    root = json.loads(fixture.DEFAULT_LOCK.read_text(encoding="utf-8"))
    edacc = root["sources"][1]
    edacc["selection_recipe"]["seed"] = root["selection_seed"]
    edacc["selection_recipe_sha256"] = fixture._canonical_sha256(
        edacc["selection_recipe"]
    )
    digest_input = dict(root)
    digest_input.pop("recipe_sha256")
    root["recipe_sha256"] = fixture._canonical_sha256(digest_input)
    contradictory = tmp_path / "contradictory-selector-seed.json"
    contradictory.write_bytes(_canonical(root))
    with pytest.raises(fixture.FixtureError):
        fixture.load_fixture_lock(contradictory)

    original = fixture.dataset_by_id

    def drifted(dataset_id: str):
        dataset = original(dataset_id)
        return replace(dataset, revision="changed") if dataset_id == fixture.DATASET_IDS[0] else dataset

    monkeypatch.setattr(fixture, "dataset_by_id", drifted)
    with pytest.raises(fixture.FixtureError):
        fixture.load_fixture_lock()


def test_four_private_corpora_validate_in_lock_order_and_snapshot(tmp_path, capsys):
    lock, paths = _publish_set(tmp_path)
    result = fixture.validate_fixture_set(reversed(paths))

    assert result.lock.recipe_sha256 == lock.recipe_sha256
    assert tuple(item.source_id for item in result.sources) == fixture.SOURCE_IDS
    assert result.cases == 96
    assert result.pcm_bytes > 96 * 4
    assert len(result.fixture_set_sha256) == 64
    assert len(set(result.corpus_paths)) == 4
    fixture.verify_fixture_set_snapshot(result)

    argv: list[str] = []
    for path in paths:
        argv.extend(("--corpus", str(path)))
    assert fixture.main(argv) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["sources"] == 4
    assert payload["cases"] == 96
    assert "known reference" not in captured.out
    assert str(tmp_path) not in captured.out
    assert captured.err == ""


def test_receipt_corpus_and_snapshot_tampering_fail_closed(tmp_path):
    lock, paths = _publish_set(tmp_path)
    result = fixture.validate_fixture_set(paths)

    receipt_path = paths[0].parent / "preparation-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["cases"][0]["reference_sha256"] = "f" * 64
    receipt["selection"]["selected_case_set_sha256"] = fixture._canonical_sha256(
        receipt["cases"]
    )
    receipt_path.write_bytes(_canonical(receipt))
    os.chmod(receipt_path, 0o600)

    with pytest.raises(fixture.FixtureError):
        fixture.validate_private_source(paths[0], lock=lock)
    with pytest.raises(fixture.FixtureError):
        fixture.verify_fixture_set_snapshot(result)


def test_rebound_receipt_manifest_and_pcm_cross_bindings_fail_closed(tmp_path):
    lock = fixture.load_fixture_lock()

    (tmp_path / "rebound").mkdir()
    rebound = _publish_source(
        tmp_path / "rebound",
        lock,
        "harper-valley",
        mutate_receipt=lambda value: value["cases"][0].__setitem__(
            "reference_sha256", "f" * 64
        ),
    )
    with pytest.raises(fixture.FixtureError):
        fixture.validate_private_source(rebound, lock=lock)

    (tmp_path / "manifest").mkdir()
    changed_manifest = _publish_source(
        tmp_path / "manifest",
        lock,
        "harper-valley",
    )
    manifest = json.loads(changed_manifest.read_text(encoding="utf-8"))
    manifest["cases"][0]["expected_text"] = "different scoreable reference"
    changed_manifest.write_bytes(_canonical(manifest))
    changed_manifest.chmod(0o600)
    with pytest.raises(fixture.FixtureError):
        fixture.validate_private_source(changed_manifest, lock=lock)

    (tmp_path / "pcm").mkdir()
    changed_pcm = _publish_source(
        tmp_path / "pcm",
        lock,
        "harper-valley",
    )
    loaded = fixture.load_corpus(changed_pcm)
    pcm_path = loaded.cases[0].source_path
    raw = bytearray(pcm_path.read_bytes())
    raw[0] ^= 1
    pcm_path.write_bytes(raw)
    pcm_path.chmod(0o600)
    with pytest.raises(fixture.FixtureError):
        fixture.validate_private_source(changed_pcm, lock=lock)


@pytest.mark.parametrize(
    ("source_id", "mutator"),
    [
        (
            "harper-valley",
            lambda value: value["cases"][0]["attributes"].__setitem__(
                "pii_scan_passed", False
            ),
        ),
        (
            "edacc-test",
            lambda value: value["cases"][0]["attributes"].__setitem__(
                "actual_overlap", True
            ),
        ),
        (
            "edacc-test",
            lambda value: value["cases"][0]["attributes"].__setitem__(
                "reference_source", "raw_stm"
            ),
        ),
        (
            "ami-es2004a-close-far",
            lambda value: value["cases"][1]["attributes"].__setitem__(
                "end_sample", 9_999
            ),
        ),
        (
            "cvss4-en",
            lambda value: value["cases"][1]["attributes"].__setitem__(
                "prompt_sha256", value["cases"][0]["attributes"]["prompt_sha256"]
            ),
        ),
    ],
)
def test_source_specific_scoreability_invariants_fail_closed(
    tmp_path,
    source_id: str,
    mutator: ReceiptMutator,
):
    lock, paths = _publish_set(tmp_path, source_mutator=(source_id, mutator))
    path = paths[fixture.SOURCE_IDS.index(source_id)]
    with pytest.raises(fixture.FixtureError):
        fixture.validate_private_source(path, lock=lock, expected_source_id=source_id)


def test_private_mode_hardlink_and_exact_four_source_requirements(tmp_path):
    lock, paths = _publish_set(tmp_path)
    receipt = paths[0].parent / "preparation-receipt.json"
    alias = paths[0].parent / "receipt-alias.json"
    os.link(receipt, alias)
    with pytest.raises(fixture.FixtureError):
        fixture.validate_private_source(paths[0], lock=lock)

    with pytest.raises(fixture.FixtureError):
        fixture.validate_fixture_set(paths[:3])
    with pytest.raises(fixture.FixtureError):
        fixture.validate_fixture_set((paths[0], paths[0], paths[2], paths[3]))


def test_cli_failure_is_detail_free(tmp_path, capsys):
    canary = tmp_path / "private-canary" / "corpus.json"
    assert fixture.main(["--corpus", str(canary)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "fixture validation failed"
    assert str(canary) not in captured.err


def _fake_suite_report(worker_manifest_paths, corpus_paths, **_kwargs):
    workers = tuple(worker_manifest_paths)
    corpora = tuple(corpus_paths)
    digests = [load.digest for load in map(fixture.load_corpus, corpora)]
    runs = [
        {
            "binding": {
                "worker_index": worker_index,
                "corpus_index": corpus_index,
                "corpus_manifest_sha256": digest,
            },
            "report": {},
        }
        for worker_index in range(len(workers))
        for corpus_index, digest in enumerate(digests)
    ]
    return {
        "ok": True,
        "complete": True,
        "controller": {
            "state": "complete",
            "planned_worker_manifests": len(workers),
            "worker_manifests": len(workers),
            "planned_corpora": len(corpora),
            "corpora": len(corpora),
            "planned_runs": len(runs),
            "runs": len(runs),
            "corpus_set_sha256": fixture._canonical_sha256(digests),
        },
        "config": {},
        "runs": runs,
    }


def test_validated_suite_uses_lock_order_and_revalidates_after_run(
    tmp_path, monkeypatch
):
    lock, paths = _publish_set(tmp_path)
    observed: dict[str, object] = {}

    def run_suite(workers, corpora, **kwargs):
        observed["workers"] = tuple(workers)
        observed["corpora"] = tuple(corpora)
        observed["scratch"] = kwargs["scratch_parent"]
        return _fake_suite_report(observed["workers"], observed["corpora"])

    monkeypatch.setattr(streaming_stt_suite, "run_suite", run_suite)
    result = fixture.run_validated_suite(
        (tmp_path / "worker.json",),
        reversed(paths),
        scratch_parent=tmp_path / "scratch",
    )

    assert result["ok"] is True
    assert result["fixture"]["validation"] == "before_and_after_suite"
    assert result["fixture"]["lock_recipe_sha256"] == lock.recipe_sha256
    assert tuple(path.parent.name for path in observed["corpora"]) == fixture.SOURCE_IDS
    assert len(result["suite"]["runs"]) == 4


def test_validated_suite_rejects_pcm_changed_during_model_run(tmp_path, monkeypatch):
    _lock, paths = _publish_set(tmp_path)

    def mutate_during_suite(workers, corpora, **_kwargs):
        selected = tuple(corpora)
        loaded = fixture.load_corpus(selected[0])
        pcm = loaded.cases[0].source_path
        raw = bytearray(pcm.read_bytes())
        raw[0] ^= 1
        pcm.write_bytes(raw)
        pcm.chmod(0o600)
        return _fake_suite_report(tuple(workers), selected)

    monkeypatch.setattr(streaming_stt_suite, "run_suite", mutate_during_suite)
    with pytest.raises(fixture.FixtureError):
        fixture.run_validated_suite(
            (tmp_path / "worker.json",),
            paths,
            scratch_parent=tmp_path / "scratch",
        )
