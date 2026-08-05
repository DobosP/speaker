from __future__ import annotations

import json
import hashlib
import stat
import os
import wave
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import tools.recorded_stt_eval as stt_eval
from core.diagnostic_bundle import FinalModelInputReceipt, FinalModelInputRole
from tools.streaming_stt.corpus import CorpusProvenance
from tools.streaming_stt.corpus_writer import CorpusWriteCase, publish_private_corpus
from tools.streaming_stt.private_diagnostic_receipt import (
    PrivateDiagnosticCaseSurface,
    create_private_diagnostic_receipt,
)


def _totals(pairs, *, keywords=(), verifier_outcomes=None):
    measured = stt_eval._measure(pairs, keywords=keywords)
    return stt_eval.EvaluationTotals(
        streaming=measured,
        offline=measured,
        selected=measured,
        clips=len(pairs),
        decisions=len(pairs),
        offline_outcomes={"decoded": len(pairs)},
        verifier_outcomes=verifier_outcomes or {},
        selected_sources={"streaming": len(pairs)},
        selected_sources_attested=True,
    )


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _schema_v3_corpus(
    tmp_path,
    *,
    tags=("private-diagnostic", "selected_asr_segment", "owner-voice"),
    role=FinalModelInputRole.SELECTED_ASR_SEGMENT,
    suite="final-model-input",
    commands=(),
):
    samples = np.array([1.25, -1.5, 0.1234567, -0.7654321], dtype="<f4")
    audio = samples.tobytes()
    receipt = FinalModelInputReceipt(
        input_index=7,
        monotonic_ns=10,
        stream_id="sherpa-11111111111111111111111111111111",
        utterance_id="u1",
        capture_epoch=2,
        capture_generation=3,
        revision=1,
        role=role,
        sample_rate_hz=16_000,
        sample_start=0,
        sample_end=len(samples),
        sha256=hashlib.sha256(audio).hexdigest(),
    )
    case = CorpusWriteCase(
        case_id="owner-command-001",
        audio_bytes=audio,
        reference="search in my vault",
        tags=tuple(tags),
        commands=tuple(commands),
    )
    selection = create_private_diagnostic_receipt(
        diagnostic_manifest_sha256="1" * 64,
        labels_sha256="2" * 64,
        cases=(
            PrivateDiagnosticCaseSurface(
                case_id=case.case_id,
                filename=f"{case.case_id}.f32le",
                sha256=receipt.sha256,
                samples=receipt.samples,
                expected_text=case.reference,
                assertion=case.assertion,
                commands=case.commands,
                tags=case.tags,
            ),
        ),
        selected_inputs=(receipt,),
    )
    provenance = CorpusProvenance(
        kind="private-diagnostic-v1",
        suite=suite,
        manifest_sha256="1" * 64,
        metadata_sha256="2" * 64,
        source_set_sha256=hashlib.sha256(selection).hexdigest(),
    )
    loaded = publish_private_corpus(
        cases=(case,),
        provenance=provenance,
        output_dir=tmp_path / "private-v3",
        purpose="exact private diagnostic final-selection input v1",
        sidecars={"preparation-receipt.json": selection},
    )
    return loaded, audio


def _loaded_fixture(item, digest="c" * 64):
    return stt_eval._CorpusLoad(
        (item,),
        digest,
        stt_eval._LEGACY_CORPUS_WITNESS,
    )


def test_aggregate_accuracy_counts_micro_wer_cer_and_keyword_recall():
    totals = stt_eval._measure(
        [
            ("alpha beta", "alpha gamma"),
            ("find vault now", "find vault now"),
        ],
        keywords=("vault",),
    )

    assert totals.clips == 2
    assert totals.nonempty == 2
    assert totals.exact == 1
    assert totals.substitutions == 1
    assert totals.insertions == 0
    assert totals.deletions == 0
    assert totals.ref_words == 5
    assert totals.wer == 0.2
    assert 0.0 < totals.cer < 1.0
    assert totals.keyword_attempts == 1
    assert totals.keyword_hits == 1


def test_candidate_requires_measured_improvement_without_regression():
    references = ["alpha beta", "gamma delta"]
    baseline_rows = list(zip(references, ["alpha wrong", "bad delta"]))
    candidate_rows = list(zip(references, ["alpha beta", "bad delta"]))
    comparison = stt_eval.compare_candidate(
        _totals(baseline_rows),
        _totals(candidate_rows),
        [stt_eval._pair_errors(*pair) for pair in baseline_rows],
        [stt_eval._pair_errors(*pair) for pair in candidate_rows],
    )

    assert comparison.promotable is True
    assert (comparison.wins, comparison.ties, comparison.losses) == (1, 1, 0)

    regressed_rows = list(zip(references, ["alpha beta", "totally wrong"]))
    regressed = stt_eval.compare_candidate(
        _totals(baseline_rows),
        _totals(regressed_rows),
        [stt_eval._pair_errors(*pair) for pair in baseline_rows],
        [stt_eval._pair_errors(*pair) for pair in regressed_rows],
    )
    assert regressed.promotable is False
    assert regressed.losses >= 1


def test_candidate_counts_per_clip_cer_losses_when_wer_ties():
    references = ["abcdefghij", "cat", "dog"]
    baseline_rows = list(zip(references, ["xxxxxxxxxx", "bat", "fog"]))
    candidate_rows = list(zip(references, ["abcdefghik", "zzzz", "zzzz"]))

    comparison = stt_eval.compare_candidate(
        _totals(baseline_rows),
        _totals(candidate_rows),
        [stt_eval._pair_errors(*pair) for pair in baseline_rows],
        [stt_eval._pair_errors(*pair) for pair in candidate_rows],
    )

    assert candidate_rows and _totals(candidate_rows).selected.char_edits < _totals(
        baseline_rows
    ).selected.char_edits
    assert (comparison.wins, comparison.losses) == (1, 2)
    assert comparison.promotable is False


def test_private_report_is_mode_600_and_aggregate_only(tmp_path):
    report = tmp_path / "report.json"
    payload = {"ok": True, "selected": {"wer": 0.0}}

    stt_eval._write_report(report, payload)

    assert stat.S_IMODE(report.stat().st_mode) == 0o600
    assert json.loads(report.read_text(encoding="utf-8")) == payload


def test_private_report_replaces_symlink_without_touching_target(tmp_path):
    target = tmp_path / "target.txt"
    report = tmp_path / "report.json"
    target.write_text("keep", encoding="utf-8")
    report.symlink_to(target)

    stt_eval._write_report(report, {"ok": True})

    assert target.read_text(encoding="utf-8") == "keep"
    assert report.is_file() and not report.is_symlink()
    assert stat.S_IMODE(report.stat().st_mode) == 0o600


def test_artifact_digest_binds_bias_dictionary_contents(tmp_path):
    @dataclass(frozen=True)
    class _Config:
        asr_encoder: str
        asr_final_hr_dict_dir: str

    model = tmp_path / "model.onnx"
    dictionary = tmp_path / "bias"
    dictionary.mkdir()
    lexicon = dictionary / "lexicon.txt"
    model.write_bytes(b"model")
    lexicon.write_text("first", encoding="utf-8")
    config = _Config(str(model), str(dictionary))

    first = stt_eval._model_digest(config)
    lexicon.write_text("second", encoding="utf-8")
    second = stt_eval._model_digest(config)

    assert first != second


def test_artifact_digest_binds_only_active_bpe_hotword_vocab(tmp_path):
    @dataclass(frozen=True)
    class _Config:
        asr_encoder: str
        asr_hotwords: str
        asr_decoding_method: str
        asr_modeling_unit: str
        asr_bpe_vocab: str

    model = tmp_path / "model.onnx"
    vocab = tmp_path / "bpe.vocab"
    model.write_bytes(b"model")
    vocab.write_bytes(b"first")
    active = _Config(
        str(model), "VAULT", "modified_beam_search", "bpe", str(vocab)
    )

    first = stt_eval._model_digest(active)
    vocab.write_bytes(b"second")
    second = stt_eval._model_digest(active)

    assert first != second
    stale = _Config(
        str(model), "", "modified_beam_search", "bpe", str(tmp_path / "missing.vocab")
    )
    clean = _Config(str(model), "", "modified_beam_search", "bpe", "")
    assert stt_eval._model_digest(stale) == stt_eval._model_digest(clean)
    for method, unit in (("greedy_search", "bpe"), ("modified_beam_search", "cjkchar")):
        inert = _Config(
            str(model), "VAULT", method, unit, str(tmp_path / "missing.vocab")
        )
        assert stt_eval._model_digest(inert) == stt_eval._model_digest(
            _Config(str(model), "VAULT", method, unit, "")
        )


def test_artifact_digest_binds_final_verifier_snapshot_contents(tmp_path):
    @dataclass(frozen=True)
    class _Config:
        asr_encoder: str
        asr_final_verifier_model: str

    model = tmp_path / "streaming.onnx"
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    weights = verifier / "model.bin"
    model.write_bytes(b"streaming")
    weights.write_bytes(b"first")
    config = _Config(str(model), str(verifier))

    first = stt_eval._model_digest(config)
    weights.write_bytes(b"second")
    second = stt_eval._model_digest(config)

    assert first != second


def test_artifact_digest_binds_nemo_joiner_contents(tmp_path):
    @dataclass(frozen=True)
    class _Config:
        asr_encoder: str
        asr_final_joiner: str

    model = tmp_path / "streaming.onnx"
    joiner = tmp_path / "joiner.onnx"
    model.write_bytes(b"streaming")
    joiner.write_bytes(b"first")
    config = _Config(str(model), str(joiner))

    first = stt_eval._model_digest(config)
    joiner.write_bytes(b"second")
    second = stt_eval._model_digest(config)

    assert first != second


def test_artifact_digest_fails_when_configured_resource_is_missing(tmp_path):
    @dataclass(frozen=True)
    class _Config:
        asr_encoder: str
        asr_final_hr_lexicon: str

    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    config = _Config(str(model), str(tmp_path / "missing.txt"))

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._model_digest(config)


def test_artifact_digest_ignores_stale_disabled_verifier_path(tmp_path):
    @dataclass(frozen=True)
    class _Config:
        asr_encoder: str
        asr_final_verifier_backend: str
        asr_final_verifier_model: str

    model = tmp_path / "streaming.onnx"
    model.write_bytes(b"model")
    stale = str(tmp_path / "missing-verifier")

    disabled = _Config(str(model), "", stale)
    clean = _Config(str(model), "", "")

    assert stt_eval._model_digest(disabled) == stt_eval._model_digest(clean)


@pytest.mark.parametrize(
    "outcome",
    [
        "consensus",
        "empty",
        "tie",
        "no_quorum",
        "control_guard",
        "attested_control",
        "empty_veto",
        "empty_streaming_guard",
    ],
)
def test_enabled_verifier_accepts_completed_safe_fallback_outcomes(outcome):
    config = type(
        "Config",
        (),
        {"asr_final_verifier_backend": "faster_whisper"},
    )()
    totals = _totals([("alpha", "alpha")], verifier_outcomes={outcome: 1})

    assert stt_eval._enabled_verifier_evaluation_ok(config, totals)


@pytest.mark.parametrize(
    "outcomes",
    [
        {},
        {"skipped": 1},
        {"unavailable": 1},
        {"error": 1},
        {"consensus": 1, "error": 1},
    ],
)
def test_enabled_verifier_rejects_errors_or_zero_completed_decodes(outcomes):
    config = type(
        "Config",
        (),
        {"asr_final_verifier_backend": "faster_whisper"},
    )()
    totals = _totals([("alpha", "alpha")], verifier_outcomes=outcomes)

    assert not stt_eval._enabled_verifier_evaluation_ok(config, totals)


def test_disabled_verifier_does_not_require_verifier_outcomes():
    config = type("Config", (), {"asr_final_verifier_backend": ""})()
    totals = _totals([("alpha", "alpha")])

    assert stt_eval._enabled_verifier_evaluation_ok(config, totals)


@pytest.mark.parametrize("speech_sec", [True, float("inf"), 1.1])
def test_corpus_rejects_impossible_attested_speech_duration(tmp_path, speech_sec):
    clip = tmp_path / "clip.wav"
    with wave.open(str(clip), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 16000)
    digest = stt_eval._sha256_file(clip)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "clip_dir": str(tmp_path),
                "clips": [
                    {
                        "id": "clip",
                        "expected_text": "private reference",
                        "sha256": digest,
                        "speech_sec": speech_sec,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._load_corpus(manifest)


def test_schema_v3_loads_exact_owned_final_input_and_preserves_tags(tmp_path):
    published, audio = _schema_v3_corpus(tmp_path)

    loaded = stt_eval._load_corpus(published.path)
    items, digest = loaded

    assert digest == published.digest
    assert len(items) == 1
    item = items[0]
    assert item.sample_rate == 16_000
    assert item.speech_sec is None
    assert item.tags == (
        "private-diagnostic",
        "selected_asr_segment",
        "owner-voice",
    )
    assert isinstance(item.samples, np.ndarray)
    assert item.samples.flags.owndata
    assert item.samples.dtype == np.dtype(np.float32)
    assert item.samples.astype("<f4", copy=False).tobytes() == audio
    (published.path.parent / "owner-command-001.f32le").write_bytes(b"\0" * len(audio))
    assert item.samples.astype("<f4", copy=False).tobytes() == audio


def test_schema_v3_accepts_model_gate_role_when_receipt_and_tag_match(tmp_path):
    published, _audio = _schema_v3_corpus(
        tmp_path,
        tags=("private-diagnostic", "model_gate_segment"),
        role=FinalModelInputRole.MODEL_GATE_SEGMENT,
    )

    items, _digest = stt_eval._load_corpus(published.path)

    assert items[0].tags == ("private-diagnostic", "model_gate_segment")


def test_legacy_loader_digest_and_fields_remain_byte_compatible(tmp_path):
    clip = tmp_path / "clip.wav"
    with wave.open(str(clip), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * 160)
    clip_digest = stt_eval._sha256_file(clip)
    manifest = tmp_path / "legacy.json"
    payload = {
        "clip_dir": str(tmp_path),
        "clips": [
            {
                "id": "clip",
                "expected_text": "legacy reference",
                "sha256": clip_digest,
                "speech_sec": 0.005,
            }
        ],
    }
    raw = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    manifest.write_bytes(raw)
    expected = hashlib.sha256(raw)
    expected.update(bytes.fromhex(clip_digest))

    loaded = stt_eval._load_corpus(manifest)
    items, digest = loaded

    assert digest == expected.hexdigest()
    assert items[0].sample_rate == 16_000
    assert items[0].speech_sec == 0.005
    assert items[0].tags == ()
    stt_eval._verify_loaded_corpus(loaded)
    for protected in (manifest, clip):
        with pytest.raises(stt_eval.EvaluationPrerequisiteError):
            stt_eval._guard_output_path(loaded, protected)


@pytest.mark.parametrize("version", [1, 2, 4, "3", True])
def test_reserved_versioned_dispatch_never_falls_back_to_legacy(tmp_path, version):
    manifest = tmp_path / "reserved.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": version,
                "clip_dir": str(tmp_path),
                "clips": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._load_corpus(manifest)


def test_dispatch_rejects_manifest_above_legacy_compatible_bound(tmp_path):
    manifest = tmp_path / "oversized.json"
    with manifest.open("wb") as handle:
        handle.truncate(stt_eval._MAX_RECORDED_CORPUS_MANIFEST_BYTES + 1)

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._load_corpus(manifest)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is POSIX-only")
def test_dispatch_rejects_fifo_without_opening_it(tmp_path):
    manifest = tmp_path / "manifest.fifo"
    os.mkfifo(manifest)

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._load_corpus(manifest)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_schema_v3_manifest_links_fail_closed(tmp_path, link_kind):
    published, _audio = _schema_v3_corpus(tmp_path)
    alias = tmp_path / "manifest-alias.json"
    if link_kind == "symlink":
        alias.symlink_to(published.path)
    else:
        os.link(published.path, alias)

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._load_corpus(alias if link_kind == "symlink" else published.path)


def test_schema_v3_rejects_wrong_suite_and_cross_version_provenance(tmp_path):
    wrong_suite, _audio = _schema_v3_corpus(tmp_path, suite="other-suite")
    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._load_corpus(wrong_suite.path)

    other = tmp_path / "other"
    other.mkdir(mode=0o700)
    other.chmod(0o700)
    published, _audio = _schema_v3_corpus(other)
    payload = json.loads(published.path.read_text(encoding="utf-8"))
    payload["provenance"]["kind"] = "public-voice-v1"
    published.path.write_bytes(_canonical(payload))
    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._load_corpus(published.path)


@pytest.mark.parametrize(
    ("tags", "role"),
    [
        (("selected_asr_segment",), FinalModelInputRole.SELECTED_ASR_SEGMENT),
        (("private-diagnostic",), FinalModelInputRole.SELECTED_ASR_SEGMENT),
        (
            (
                "private-diagnostic",
                "selected_asr_segment",
                "model_gate_segment",
            ),
            FinalModelInputRole.SELECTED_ASR_SEGMENT,
        ),
        (
            ("private-diagnostic", "selected_asr_segment"),
            FinalModelInputRole.MODEL_GATE_SEGMENT,
        ),
    ],
)
def test_schema_v3_requires_marker_and_exactly_matching_role_tag(
    tmp_path,
    tags,
    role,
):
    published, _audio = _schema_v3_corpus(tmp_path, tags=tags, role=role)

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._load_corpus(published.path)


def test_schema_v3_rejects_non_exporter_command_rows(tmp_path):
    published, _audio = _schema_v3_corpus(
        tmp_path,
        commands=("search in my vault",),
    )

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._load_corpus(published.path)


@pytest.mark.parametrize("target", ["receipt", "pcm"])
def test_schema_v3_rejects_bound_artifact_tamper(tmp_path, target):
    published, audio = _schema_v3_corpus(tmp_path)
    path = (
        published.path.parent / "preparation-receipt.json"
        if target == "receipt"
        else published.path.parent / "owner-command-001.f32le"
    )
    path.write_bytes(b"x" * path.stat().st_size)
    path.chmod(0o600)

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._load_corpus(published.path)
    assert len(audio) > 0


@pytest.mark.parametrize("target", ["receipt", "pcm"])
@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_schema_v3_rejects_linked_artifacts(tmp_path, target, link_kind):
    published, _audio = _schema_v3_corpus(tmp_path)
    root = published.path.parent
    path = (
        root / "preparation-receipt.json"
        if target == "receipt"
        else root / "owner-command-001.f32le"
    )
    alias = root / f"{target}-alias"
    if link_kind == "hardlink":
        os.link(path, alias)
    else:
        path.rename(alias)
        path.symlink_to(alias.name)

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._load_corpus(published.path)


def test_schema_v3_rejects_oversized_selection_receipt(tmp_path):
    published, _audio = _schema_v3_corpus(tmp_path)
    receipt = published.path.parent / "preparation-receipt.json"
    oversized = b"x" * (stt_eval._MAX_PRIVATE_DIAGNOSTIC_RECEIPT_BYTES + 1)
    receipt.write_bytes(oversized)
    payload = json.loads(published.path.read_text(encoding="utf-8"))
    payload["provenance"]["source_set_sha256"] = hashlib.sha256(
        oversized
    ).hexdigest()
    published.path.write_bytes(_canonical(payload))

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._load_corpus(published.path)


@pytest.mark.parametrize("target", ["directory", "manifest", "receipt", "pcm"])
def test_schema_v3_requires_canonical_private_metadata(tmp_path, target):
    published, _audio = _schema_v3_corpus(tmp_path)
    paths = {
        "directory": published.path.parent,
        "manifest": published.path,
        "receipt": published.path.parent / "preparation-receipt.json",
        "pcm": published.path.parent / "owner-command-001.f32le",
    }
    paths[target].chmod(0o755 if target == "directory" else 0o644)

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._load_corpus(published.path)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrapper-version",
        "wrapper-bool",
        "wrapper-kind",
        "manifest",
        "labels",
        "case-binding",
        "missing-row",
        "duplicate-row",
        "sha256",
        "samples",
        "sample-rate",
        "role",
    ],
)
def test_schema_v3_semantically_rechecks_coordinated_receipt_change(
    tmp_path,
    mutation,
):
    published, _audio = _schema_v3_corpus(tmp_path)
    receipt_path = published.path.parent / "preparation-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mutation == "wrapper-version":
        receipt["schema_version"] = 1
    elif mutation == "wrapper-bool":
        receipt["schema_version"] = True
    elif mutation == "wrapper-kind":
        receipt["kind"] = "other-selection-v2"
    elif mutation == "manifest":
        receipt["diagnostic_manifest_sha256"] = "0" * 64
    elif mutation == "labels":
        receipt["labels_sha256"] = "0" * 64
    elif mutation == "case-binding":
        receipt["case_binding_sha256"] = "0" * 64
    elif mutation == "missing-row":
        receipt["selected_inputs"] = []
    elif mutation == "duplicate-row":
        receipt["selected_inputs"].append(dict(receipt["selected_inputs"][0]))
    elif mutation == "sha256":
        receipt["selected_inputs"][0]["sha256"] = "0" * 64
    elif mutation == "samples":
        receipt["selected_inputs"][0]["samples"] += 1
    elif mutation == "sample-rate":
        receipt["selected_inputs"][0]["sample_rate_hz"] = 8_000
    else:
        receipt["selected_inputs"][0]["role"] = "model_gate_segment"
    changed = _canonical(receipt)
    receipt_path.write_bytes(changed)
    payload = json.loads(published.path.read_text(encoding="utf-8"))
    payload["provenance"]["source_set_sha256"] = hashlib.sha256(changed).hexdigest()
    published.path.write_bytes(_canonical(payload))

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._load_corpus(published.path)


def test_schema_v3_requires_strictly_increasing_receipt_indexes(tmp_path):
    published, audio = _schema_v3_corpus(tmp_path)
    root = published.path.parent
    second_audio = root / "owner-command-002.f32le"
    second_audio.write_bytes(audio)
    second_audio.chmod(0o600)
    payload = json.loads(published.path.read_text(encoding="utf-8"))
    second_case = dict(payload["cases"][0])
    second_case.update(id="owner-command-002", file=second_audio.name)
    payload["cases"].append(second_case)
    receipt_path = root / "preparation-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    second_receipt = dict(receipt["selected_inputs"][0])
    second_receipt["input_index"] = 6
    receipt["selected_inputs"].append(second_receipt)
    receipt_bytes = _canonical(receipt)
    receipt_path.write_bytes(receipt_bytes)
    payload["provenance"]["source_set_sha256"] = hashlib.sha256(
        receipt_bytes
    ).hexdigest()
    published.path.write_bytes(_canonical(payload))

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._load_corpus(published.path)


@pytest.mark.parametrize(
    "mutation",
    ["expected-text", "route-tag", "case-id-and-file", "tag-order", "commands"],
)
def test_schema_v3_rejects_published_case_surface_tampering(tmp_path, mutation):
    published, _audio = _schema_v3_corpus(
        tmp_path,
        tags=(
            "private-diagnostic",
            "selected_asr_segment",
            "owner-voice",
            "expected-tool.vault.search",
        ),
    )
    payload = json.loads(published.path.read_text(encoding="utf-8"))
    row = payload["cases"][0]
    if mutation == "expected-text":
        row["expected_text"] = "find in my vault"
    elif mutation == "route-tag":
        row["tags"][-1] = "expected-tool.web.search"
    elif mutation == "case-id-and-file":
        source = published.path.parent / row["file"]
        row.update(id="owner-command-renamed", file="owner-command-renamed.f32le")
        source.rename(published.path.parent / row["file"])
    elif mutation == "tag-order":
        row["tags"][-2:] = reversed(row["tags"][-2:])
    else:
        row["commands"] = ["search in my vault"]
    published.path.write_bytes(_canonical(payload))

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._load_corpus(published.path)


def test_schema_v3_rejects_exact_legacy_v1_selection_receipt(tmp_path):
    published, _audio = _schema_v3_corpus(tmp_path)
    receipt_path = published.path.parent / "preparation-receipt.json"
    current = json.loads(receipt_path.read_text(encoding="utf-8"))
    legacy = _canonical(
        {
            "schema_version": 1,
            "kind": "private-diagnostic-selection-v1",
            "diagnostic_manifest_sha256": current[
                "diagnostic_manifest_sha256"
            ],
            "selected_inputs": current["selected_inputs"],
        }
    )
    receipt_path.write_bytes(legacy)
    payload = json.loads(published.path.read_text(encoding="utf-8"))
    payload["provenance"]["source_set_sha256"] = hashlib.sha256(legacy).hexdigest()
    published.path.write_bytes(_canonical(payload))

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._load_corpus(published.path)


def test_schema_v3_cross_binds_source_label_digest(tmp_path):
    published, _audio = _schema_v3_corpus(tmp_path)
    payload = json.loads(published.path.read_text(encoding="utf-8"))
    payload["provenance"]["metadata_sha256"] = "0" * 64
    published.path.write_bytes(_canonical(payload))

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._load_corpus(published.path)


def test_cli_rejects_case_binding_tamper_before_model_or_report(
    tmp_path,
    monkeypatch,
    capsys,
):
    published, _audio = _schema_v3_corpus(tmp_path)
    payload = json.loads(published.path.read_text(encoding="utf-8"))
    payload["cases"][0]["expected_text"] = "find in my vault"
    published.path.write_bytes(_canonical(payload))
    report = tmp_path / "report.json"

    def forbidden_model_load():
        raise AssertionError("model setup reached")

    monkeypatch.setattr(stt_eval, "_load_config", forbidden_model_load)

    assert stt_eval.main(
        ["--manifest", str(published.path), "--output", str(report)]
    ) == 2
    assert not report.exists()
    assert json.loads(capsys.readouterr().out) == stt_eval._SAFE_ERROR


def test_schema_v3_calls_snapshot_verifier_after_owned_copy(tmp_path, monkeypatch):
    published, _audio = _schema_v3_corpus(tmp_path)
    from tools.streaming_stt import corpus as corpus_module

    original = corpus_module.verify_corpus_snapshot
    calls = []

    def verify(corpus):
        calls.append(corpus.digest)
        original(corpus)

    monkeypatch.setattr(corpus_module, "verify_corpus_snapshot", verify)

    loaded = stt_eval._load_corpus(published.path)

    assert loaded.digest == published.digest
    assert calls == [published.digest]


def test_schema_v3_post_copy_verifier_catches_pcm_mutation(tmp_path, monkeypatch):
    published, audio = _schema_v3_corpus(tmp_path)
    from tools.streaming_stt import corpus as corpus_module

    original = corpus_module.verify_corpus_snapshot

    def mutate_then_verify(corpus):
        pcm = corpus.path.parent / "owner-command-001.f32le"
        pcm.write_bytes(b"\0" * len(audio))
        pcm.chmod(0o600)
        original(corpus)

    monkeypatch.setattr(
        corpus_module,
        "verify_corpus_snapshot",
        mutate_then_verify,
    )

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._load_corpus(published.path)


def test_selected_source_vocabulary_matches_production_enum():
    from core.engines.sherpa import FinalTranscriptSource

    assert stt_eval._SELECTED_SOURCES == {
        value.value for value in FinalTranscriptSource
    }


def test_terminal_text_helper_preserves_empty_decision_boundaries():
    private = "SENTINEL_PRIVATE_HYPOTHESIS"
    decisions = (
        SimpleNamespace(streaming_raw="", offline_raw=None, selected=""),
        SimpleNamespace(
            streaming_raw=private,
            offline_raw=" offline ",
            selected=" selected ",
        ),
    )

    texts = stt_eval._terminal_decision_texts(decisions)

    assert texts.streaming == ("", private)
    assert texts.offline == ("", " offline ")
    assert texts.selected == ("", " selected ")
    assert private not in repr(texts)


def test_evaluate_attests_all_five_selected_sources(monkeypatch):
    from core.engines import file_replay
    from core.engines.sherpa import FinalTranscriptSource

    decisions = tuple(
        SimpleNamespace(
            streaming_raw=text,
            offline_raw=text,
            selected=text,
            offline_outcome="decoded",
            verifier_outcome="unavailable",
            selected_source=source,
        )
        for source, text in zip(
            FinalTranscriptSource,
            ("", "one", "two", "three", "four"),
        )
    )

    stopped = []

    class Engine:
        def __init__(self, _config, *, asr_only):
            assert asr_only is True

        def start(self, _callbacks):
            return None

        def evaluate_samples(self, _samples, sample_rate, *, speech_sec):
            assert sample_rate == 16_000
            assert speech_sec is None
            return SimpleNamespace(decisions=decisions)

        def stop(self):
            stopped.append(True)

    monkeypatch.setattr(file_replay, "FileReplayEngine", Engine)
    item = stt_eval._CorpusItem(
        "one two three four",
        np.zeros(16, dtype=np.float32),
        16_000,
        None,
    )

    totals, errors = stt_eval._evaluate(object(), (item,), ())

    assert errors == ((0, 0),)
    assert totals.decisions == 5
    assert totals.selected_sources == {
        "none": 1,
        "streaming": 1,
        "offline": 1,
        "verifier_consensus": 1,
        "established_override": 1,
    }
    assert totals.selected_sources_attested is True
    assert totals.selected_source_accounting_complete is True
    assert totals.as_dict()["selected_source_accounting_complete"] is True
    assert stopped == [True]


def test_unknown_selected_source_fails_detail_free(monkeypatch):
    from core.engines import file_replay

    canary = "SENTINEL_PRIVATE_SELECTED_SOURCE"
    decision = SimpleNamespace(
        streaming_raw="private",
        offline_raw="private",
        selected="private",
        offline_outcome="decoded",
        verifier_outcome="unavailable",
        selected_source=canary,
    )

    class Engine:
        def __init__(self, _config, *, asr_only):
            assert asr_only

        def start(self, _callbacks):
            return None

        def evaluate_samples(self, *_args, **_kwargs):
            return SimpleNamespace(decisions=(decision,))

        def stop(self):
            return None

    monkeypatch.setattr(file_replay, "FileReplayEngine", Engine)
    item = stt_eval._CorpusItem("private", np.zeros(1), 16_000, None)

    with pytest.raises(stt_eval.EvaluationPrerequisiteError) as caught:
        stt_eval._evaluate(object(), (item,), ())

    assert canary not in str(caught.value)
    assert canary not in repr(caught.value)


@pytest.mark.parametrize("source", [None, True, False, 1, "unknown"])
def test_selected_source_normalizer_rejects_nonproduction_values(source):
    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._selected_source(source)


def test_selected_source_accounting_requires_exact_attestation_and_sum():
    measured = stt_eval._measure((("one", "one"),))
    generic = stt_eval.EvaluationTotals(
        measured,
        measured,
        measured,
        1,
        1,
        {},
    )
    mismatch = stt_eval.EvaluationTotals(
        measured,
        measured,
        measured,
        1,
        2,
        {},
        selected_sources={"streaming": 1},
        selected_sources_attested=True,
    )

    assert generic.complete is True
    assert generic.selected_source_accounting_complete is False
    assert mismatch.complete is True
    assert mismatch.selected_source_accounting_complete is False
    assert not stt_eval._recorded_evaluation_ok(object(), generic)
    assert not stt_eval._recorded_evaluation_ok(object(), mismatch)


@pytest.mark.parametrize("decisions", [0, -1, True, 1.0, "1"])
def test_selected_source_accounting_requires_positive_exact_int_decisions(decisions):
    measured = stt_eval._measure((("one", "one"),))
    totals = stt_eval.EvaluationTotals(
        measured,
        measured,
        measured,
        1,
        decisions,
        {},
        selected_sources={"streaming": 1},
        selected_sources_attested=True,
    )

    assert totals.selected_source_accounting_complete is False


def test_selected_source_map_is_detached_and_hidden_from_repr():
    measured = stt_eval._measure((("one", "one"),))
    source_map = {"streaming": 1, "offline": 0}
    totals = stt_eval.EvaluationTotals(
        measured,
        measured,
        measured,
        1,
        1,
        {},
        selected_sources=source_map,
        selected_sources_attested=True,
    )
    source_map["streaming"] = 999
    source_map["offline"] = 1

    assert totals.selected_sources == {"streaming": 1}
    assert totals.selected_source_accounting_complete is True
    assert "selected_sources" not in repr(totals)


@pytest.mark.parametrize(
    "mapping",
    [
        {"SENTINEL_PRIVATE_SOURCE": 1},
        {"streaming": True},
        {"streaming": 1.0},
        {"streaming": -1},
    ],
)
def test_malformed_selected_source_maps_fail_without_repr_leak(mapping):
    measured = stt_eval._measure((("one", "one"),))
    canary = "SENTINEL_PRIVATE_SOURCE"

    with pytest.raises(stt_eval.EvaluationPrerequisiteError) as caught:
        stt_eval.EvaluationTotals(
            measured,
            measured,
            measured,
            1,
            1,
            {},
            selected_sources=mapping,
            selected_sources_attested=True,
        )

    assert canary not in str(caught.value)
    assert canary not in repr(caught.value)


def test_selected_source_helpers_sanitize_hostile_mapping_and_value_properties():
    from collections.abc import Mapping

    canary = "SENTINEL_PRIVATE_SOURCE_DETAIL"

    class HostileMapping(Mapping):
        def __getitem__(self, _key):
            raise RuntimeError(canary)

        def __iter__(self):
            raise RuntimeError(canary)

        def __len__(self):
            raise RuntimeError(canary)

    class HostileValue:
        @property
        def value(self):
            raise RuntimeError(canary)

    for call in (
        lambda: stt_eval._closed_selected_sources(HostileMapping()),
        lambda: stt_eval._selected_source(HostileValue()),
    ):
        with pytest.raises(stt_eval.EvaluationPrerequisiteError) as caught:
            call()
        assert canary not in str(caught.value)
        assert canary not in repr(caught.value)


def test_selected_source_attestation_signal_requires_exact_bool():
    measured = stt_eval._measure((("one", "one"),))

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval.EvaluationTotals(
            measured,
            measured,
            measured,
            1,
            1,
            {},
            selected_sources={"streaming": 1},
            selected_sources_attested=1,
        )


def test_cli_never_prints_private_rows_or_config(capsys, monkeypatch):
    private_reference = "SENTINEL_PRIVATE_REFERENCE"
    private_hypothesis = "SENTINEL_PRIVATE_HYPOTHESIS"

    @dataclass(frozen=True)
    class _Config:
        value: int = 1

    item = stt_eval._CorpusItem(private_reference, object(), 16000, None)
    measured = stt_eval._measure([(private_reference, private_hypothesis)])
    totals = stt_eval.EvaluationTotals(
        streaming=measured,
        offline=measured,
        selected=measured,
        clips=1,
        decisions=1,
        offline_outcomes={"decoded": 1},
        selected_sources={"streaming": 1},
        selected_sources_attested=True,
    )
    monkeypatch.setattr(stt_eval, "_load_corpus", lambda _path: _loaded_fixture(item))
    monkeypatch.setattr(stt_eval, "_load_config", lambda: _Config())
    monkeypatch.setattr(
        stt_eval,
        "_evaluate",
        lambda _config, _corpus, _keywords: (
            totals,
            [stt_eval._pair_errors(private_reference, private_hypothesis)],
        ),
    )
    monkeypatch.setattr(stt_eval, "_config_digest", lambda _config: "d" * 64)
    monkeypatch.setattr(stt_eval, "_model_digest", lambda _config: "e" * 64)

    assert stt_eval.main([]) == 0
    output = capsys.readouterr().out
    assert private_reference not in output
    assert private_hypothesis not in output
    payload = json.loads(output)
    assert payload["baseline"]["selected"]["word_errors"] > 0
    assert "baseline_final_stt_profile" not in payload
    assert "baseline_final_stt_profile_digest" not in payload
    assert "candidate_final_stt_profile" not in payload
    assert "candidate_final_stt_profile_digest" not in payload


def test_cli_rejects_unattested_baseline_source_accounting(capsys, monkeypatch):
    item = stt_eval._CorpusItem("reference", object(), 16_000, None)
    measured = stt_eval._measure((("reference", "reference"),))
    totals = stt_eval.EvaluationTotals(
        measured,
        measured,
        measured,
        1,
        1,
        {"decoded": 1},
    )
    monkeypatch.setattr(stt_eval, "_load_corpus", lambda _path: _loaded_fixture(item))
    monkeypatch.setattr(stt_eval, "_load_config", object)
    monkeypatch.setattr(
        stt_eval,
        "_evaluate",
        lambda _config, _corpus, _keywords: (totals, ((0, 0),)),
    )
    monkeypatch.setattr(stt_eval, "_config_digest", lambda _config: "d" * 64)
    monkeypatch.setattr(stt_eval, "_model_digest", lambda _config: "e" * 64)

    assert stt_eval.main([]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["baseline"]["complete"] is True
    assert payload["baseline"]["selected_sources_attested"] is False
    assert payload["baseline"]["selected_source_accounting_complete"] is False


@pytest.mark.parametrize("source_state", ["missing", "mismatch"])
def test_cli_rejects_candidate_with_bad_source_accounting(
    capsys,
    monkeypatch,
    source_state,
):
    @dataclass(frozen=True)
    class Config:
        value: int = 1

    item = stt_eval._CorpusItem("alpha beta", object(), 16_000, None)
    baseline = _totals((("alpha beta", "alpha wrong"),))
    measured = stt_eval._measure((("alpha beta", "alpha beta"),))
    candidate = stt_eval.EvaluationTotals(
        measured,
        measured,
        measured,
        1,
        2 if source_state == "mismatch" else 1,
        {"decoded": 1},
        selected_sources=(
            {"streaming": 1} if source_state == "mismatch" else {}
        ),
        selected_sources_attested=source_state == "mismatch",
    )

    def evaluate(config, _corpus, _keywords):
        if config.value == 2:
            return candidate, ((0, 0),)
        return baseline, ((1, 5),)

    monkeypatch.setattr(stt_eval, "_load_corpus", lambda _path: _loaded_fixture(item))
    monkeypatch.setattr(stt_eval, "_load_config", Config)
    monkeypatch.setattr(stt_eval, "_evaluate", evaluate)
    monkeypatch.setattr(stt_eval, "_config_digest", lambda _config: "d" * 64)
    monkeypatch.setattr(stt_eval, "_model_digest", lambda _config: "e" * 64)

    assert stt_eval.main(["--set", "value=2"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert (
        payload["candidate"]["selected_source_accounting_complete"] is False
    )
    assert payload["comparison"]["promotable"] is False
    assert payload["ok"] is False


def test_cli_rechecks_schema_v3_after_evaluation_before_report(
    tmp_path,
    capsys,
    monkeypatch,
):
    published, _audio = _schema_v3_corpus(tmp_path)
    report = tmp_path / "report.json"
    measured = stt_eval._measure((("search in my vault", "search in my vault"),))
    totals = stt_eval.EvaluationTotals(
        measured,
        measured,
        measured,
        1,
        1,
        {"decoded": 1},
        selected_sources={"streaming": 1},
        selected_sources_attested=True,
    )

    def evaluate(_config, _corpus, _keywords):
        receipt = published.path.parent / "preparation-receipt.json"
        receipt.write_bytes(b"x" * receipt.stat().st_size)
        receipt.chmod(0o600)
        return totals, ((0, 0),)

    monkeypatch.setattr(stt_eval, "_load_config", object)
    monkeypatch.setattr(stt_eval, "_evaluate", evaluate)
    monkeypatch.setattr(stt_eval, "_config_digest", lambda _config: "d" * 64)
    monkeypatch.setattr(stt_eval, "_model_digest", lambda _config: "e" * 64)

    assert stt_eval.main(
        ["--manifest", str(published.path), "--output", str(report)]
    ) == 2
    assert not report.exists()
    assert json.loads(capsys.readouterr().out) == stt_eval._SAFE_ERROR


@pytest.mark.parametrize(
    "alias_kind",
    ["manifest", "receipt", "pcm", "symlink-pcm"],
)
def test_cli_refuses_output_aliases_to_loaded_schema_v3_inputs(
    tmp_path,
    capsys,
    monkeypatch,
    alias_kind,
):
    published, _audio = _schema_v3_corpus(tmp_path)
    protected = {
        "manifest": published.path,
        "receipt": published.path.parent / "preparation-receipt.json",
        "pcm": published.path.parent / "owner-command-001.f32le",
        "symlink-pcm": published.path.parent / "owner-command-001.f32le",
    }[alias_kind]
    before = {
        path: path.read_bytes()
        for path in (
            published.path,
            published.path.parent / "preparation-receipt.json",
            published.path.parent / "owner-command-001.f32le",
        )
    }
    output = protected
    if alias_kind == "symlink-pcm":
        output = tmp_path / "report-alias.json"
        output.symlink_to(protected)
    measured = stt_eval._measure((("search in my vault", "search in my vault"),))
    totals = stt_eval.EvaluationTotals(
        measured,
        measured,
        measured,
        1,
        1,
        {"decoded": 1},
        selected_sources={"streaming": 1},
        selected_sources_attested=True,
    )
    monkeypatch.setattr(stt_eval, "_load_config", object)
    monkeypatch.setattr(
        stt_eval,
        "_evaluate",
        lambda _config, _corpus, _keywords: (totals, ((0, 0),)),
    )
    monkeypatch.setattr(stt_eval, "_config_digest", lambda _config: "d" * 64)
    monkeypatch.setattr(stt_eval, "_model_digest", lambda _config: "e" * 64)

    assert stt_eval.main(
        ["--manifest", str(published.path), "--output", str(output)]
    ) == 2
    assert json.loads(capsys.readouterr().out) == stt_eval._SAFE_ERROR
    assert all(path.read_bytes() == payload for path, payload in before.items())
    if alias_kind == "symlink-pcm":
        assert output.is_symlink()


def test_cli_fails_closed_when_enabled_baseline_verifier_errors(
    capsys,
    monkeypatch,
):
    @dataclass(frozen=True)
    class _Config:
        asr_final_verifier_backend: str = "faster_whisper"

    item = stt_eval._CorpusItem("reference", object(), 16000, None)
    totals = _totals(
        [("reference", "reference")],
        verifier_outcomes={"error": 1},
    )
    monkeypatch.setattr(stt_eval, "_load_corpus", lambda _path: _loaded_fixture(item))
    monkeypatch.setattr(stt_eval, "_load_config", _Config)
    monkeypatch.setattr(
        stt_eval,
        "_evaluate",
        lambda _config, _corpus, _keywords: (totals, [(0, 0)]),
    )
    monkeypatch.setattr(stt_eval, "_config_digest", lambda _config: "d" * 64)
    monkeypatch.setattr(stt_eval, "_model_digest", lambda _config: "e" * 64)

    assert stt_eval.main([]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["baseline"]["verifier_outcomes"] == {"error": 1}


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        ({"decoded": 1}, True),
        ({"empty": 1}, True),
        ({"error": 1, "decoded": 1}, False),
        ({"unavailable": 1}, False),
        ({"skipped": 1}, False),
    ],
)
def test_selected_offline_evaluation_requires_completed_error_free_decode(
    outcomes,
    expected,
):
    @dataclass(frozen=True)
    class _Config:
        asr_final_backend: str = "nemo_transducer"

    totals = _totals([("reference", "reference")])
    totals = stt_eval.EvaluationTotals(
        streaming=totals.streaming,
        offline=totals.offline,
        selected=totals.selected,
        clips=totals.clips,
        decisions=totals.decisions,
        offline_outcomes=outcomes,
        verifier_outcomes=totals.verifier_outcomes,
    )
    assert stt_eval._enabled_offline_evaluation_ok(_Config(), totals) is expected


def test_cli_fails_closed_when_selected_offline_model_never_decodes(
    capsys,
    monkeypatch,
):
    @dataclass(frozen=True)
    class _Config:
        asr_final_backend: str = "nemo_transducer"

    item = stt_eval._CorpusItem("reference", object(), 16000, None)
    measured = stt_eval._measure([("reference", "reference")])
    totals = stt_eval.EvaluationTotals(
        streaming=measured,
        offline=measured,
        selected=measured,
        clips=1,
        decisions=1,
        offline_outcomes={"unavailable": 1},
    )
    monkeypatch.setattr(stt_eval, "_load_corpus", lambda _path: _loaded_fixture(item))
    monkeypatch.setattr(stt_eval, "_load_config", _Config)
    monkeypatch.setattr(
        stt_eval,
        "_evaluate",
        lambda _config, _corpus, _keywords: (totals, [(0, 0)]),
    )
    monkeypatch.setattr(stt_eval, "_config_digest", lambda _config: "d" * 64)
    monkeypatch.setattr(stt_eval, "_model_digest", lambda _config: "e" * 64)

    assert stt_eval.main([]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["baseline"]["offline_outcomes"] == {"unavailable": 1}


def test_cli_rejects_improved_candidate_when_enabled_verifier_errors(
    capsys,
    monkeypatch,
):
    @dataclass(frozen=True)
    class _Config:
        asr_final_verifier_backend: str = ""

    item = stt_eval._CorpusItem("alpha beta", object(), 16000, None)
    baseline = _totals([("alpha beta", "alpha wrong")])
    candidate = _totals(
        [("alpha beta", "alpha beta")],
        verifier_outcomes={"error": 1},
    )

    def evaluate(config, _corpus, _keywords):
        if config.asr_final_verifier_backend:
            return candidate, [(0, 0)]
        return baseline, [(1, 5)]

    monkeypatch.setattr(stt_eval, "_load_corpus", lambda _path: _loaded_fixture(item))
    monkeypatch.setattr(stt_eval, "_load_config", _Config)
    monkeypatch.setattr(stt_eval, "_evaluate", evaluate)
    monkeypatch.setattr(stt_eval, "_config_digest", lambda _config: "d" * 64)
    monkeypatch.setattr(stt_eval, "_model_digest", lambda _config: "e" * 64)

    assert stt_eval.main(
        ["--set", "asr_final_verifier_backend=faster_whisper"]
    ) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate"]["selected"]["word_errors"] == 0
    assert payload["comparison"]["wins"] == 1
    assert payload["comparison"]["promotable"] is False
    assert payload["ok"] is False


def test_cli_prerequisite_failure_is_detail_free(capsys, monkeypatch):
    monkeypatch.setattr(
        stt_eval,
        "_load_corpus",
        lambda _path: (_ for _ in ()).throw(
            stt_eval.EvaluationPrerequisiteError("SENTINEL_PRIVATE_PATH")
        ),
    )

    assert stt_eval.main([]) == 2
    output = capsys.readouterr().out
    assert "SENTINEL_PRIVATE_PATH" not in output
    assert json.loads(output) == stt_eval._SAFE_ERROR


def test_cli_unexpected_failure_is_also_detail_free(capsys, monkeypatch):
    monkeypatch.setattr(
        stt_eval,
        "_load_corpus",
        lambda _path: (_ for _ in ()).throw(
            RuntimeError("SENTINEL_PRIVATE_TRANSCRIPT")
        ),
    )

    assert stt_eval.main([]) == 2
    output = capsys.readouterr().out
    assert "SENTINEL_PRIVATE_TRANSCRIPT" not in output
    assert json.loads(output) == stt_eval._SAFE_ERROR


def test_profile_pair_uses_shared_resolver_from_one_effective_base(monkeypatch):
    effective = {"sherpa": {"streaming": "unchanged"}}
    calls = []

    def apply_profile(config, name):
        calls.append((config, name))
        return (
            {"sherpa": {"profile": name}},
            SimpleNamespace(
                name=name,
                sha256=("a" if len(calls) == 1 else "b") * 64,
                schema_version=1,
            ),
        )

    monkeypatch.setattr(stt_eval, "_load_effective_config", lambda: effective)
    monkeypatch.setattr(
        "core.config.apply_final_stt_profile",
        apply_profile,
    )
    monkeypatch.setattr(
        stt_eval,
        "_sherpa_config",
        lambda config: config["sherpa"]["profile"],
    )

    resolved = stt_eval._load_profile_configs(
        "sense-voice",
        "parakeet-faster-whisper",
    )

    assert calls == [
        (effective, "sense-voice"),
        (effective, "parakeet-faster-whisper"),
    ]
    assert resolved == (
        "sense-voice",
        ("sense-voice", "a" * 64),
        "parakeet-faster-whisper",
        ("parakeet-faster-whisper", "b" * 64),
    )
    assert effective == {"sherpa": {"streaming": "unchanged"}}


@pytest.mark.parametrize(
    "argv",
    [
        ["--baseline-final-stt-profile", "sense-voice"],
        ["--candidate-final-stt-profile", "parakeet-faster-whisper"],
        [
            "--baseline-final-stt-profile",
            "sense-voice",
            "--candidate-final-stt-profile",
            "sense-voice",
        ],
        [
            "--baseline-final-stt-profile",
            "not-a-profile",
            "--candidate-final-stt-profile",
            "parakeet-faster-whisper",
        ],
        [
            "--baseline-final-stt-profile",
            "sense-voice",
            "--candidate-final-stt-profile",
            "parakeet-faster-whisper",
            "--set",
            "value=2",
        ],
    ],
)
def test_cli_rejects_non_atomic_or_mixed_profile_comparison_before_input_work(
    argv,
    capsys,
    monkeypatch,
):
    monkeypatch.setattr(
        stt_eval,
        "_load_corpus",
        lambda _path: pytest.fail("invalid profile comparison reached corpus work"),
    )
    monkeypatch.setattr(
        stt_eval,
        "_load_profile_configs",
        lambda *_args: pytest.fail("invalid profile comparison reached config work"),
    )
    monkeypatch.setattr(
        stt_eval,
        "_load_config",
        lambda: pytest.fail("invalid profile comparison reached config work"),
    )

    assert stt_eval.main(argv) == 2
    assert json.loads(capsys.readouterr().out) == stt_eval._SAFE_ERROR


def test_cli_resolves_and_hashes_complete_profile_pair_before_replay(
    capsys,
    monkeypatch,
):
    @dataclass(frozen=True)
    class Config:
        profile: str

    item = stt_eval._CorpusItem("alpha beta", object(), 16_000, None)
    baseline = _totals((("alpha beta", "alpha wrong"),))
    candidate = _totals((("alpha beta", "alpha beta"),))
    events = []

    def load_profiles(baseline_name, candidate_name):
        assert baseline_name == "sense-voice"
        assert candidate_name == "parakeet-faster-whisper"
        return (
            Config("baseline"),
            (baseline_name, "a" * 64),
            Config("candidate"),
            (candidate_name, "b" * 64),
        )

    def config_digest(config):
        events.append(f"config:{config.profile}")
        return ("c" if config.profile == "baseline" else "d") * 64

    def model_digest(config):
        events.append(f"model:{config.profile}")
        return ("e" if config.profile == "baseline" else "f") * 64

    def evaluate(config, _corpus, _keywords):
        events.append(f"evaluate:{config.profile}")
        if config.profile == "baseline":
            return baseline, ((1, 5),)
        return candidate, ((0, 0),)

    monkeypatch.setattr(stt_eval, "_load_corpus", lambda _path: _loaded_fixture(item))
    monkeypatch.setattr(stt_eval, "_load_profile_configs", load_profiles)
    monkeypatch.setattr(stt_eval, "_config_digest", config_digest)
    monkeypatch.setattr(stt_eval, "_model_digest", model_digest)
    monkeypatch.setattr(stt_eval, "_evaluate", evaluate)

    assert (
        stt_eval.main(
            [
                "--baseline-final-stt-profile",
                "sense-voice",
                "--candidate-final-stt-profile",
                "parakeet-faster-whisper",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert events == [
        "config:baseline",
        "model:baseline",
        "config:candidate",
        "model:candidate",
        "evaluate:baseline",
        "evaluate:candidate",
    ]
    assert payload["baseline_final_stt_profile"] == "sense-voice"
    assert payload["baseline_final_stt_profile_digest"] == "a" * 64
    assert payload["candidate_final_stt_profile"] == "parakeet-faster-whisper"
    assert payload["candidate_final_stt_profile_digest"] == "b" * 64
    assert payload["baseline_config_digest"] == "c" * 64
    assert payload["candidate_config_digest"] == "d" * 64
    assert payload["baseline_model_digest"] == "e" * 64
    assert payload["candidate_model_digest"] == "f" * 64
    assert payload["comparison"]["promotable"] is True
    assert payload["ok"] is True


def test_cli_profile_pair_digest_failure_runs_neither_cell(capsys, monkeypatch):
    @dataclass(frozen=True)
    class Config:
        profile: str

    item = stt_eval._CorpusItem("reference", object(), 16_000, None)
    monkeypatch.setattr(stt_eval, "_load_corpus", lambda _path: _loaded_fixture(item))
    monkeypatch.setattr(
        stt_eval,
        "_load_profile_configs",
        lambda *_args: (
            Config("baseline"),
            ("sense-voice", "a" * 64),
            Config("candidate"),
            ("parakeet-faster-whisper", "b" * 64),
        ),
    )
    monkeypatch.setattr(stt_eval, "_config_digest", lambda _config: "c" * 64)

    def model_digest(config):
        if config.profile == "candidate":
            raise stt_eval.EvaluationPrerequisiteError()
        return "d" * 64

    monkeypatch.setattr(stt_eval, "_model_digest", model_digest)
    monkeypatch.setattr(
        stt_eval,
        "_evaluate",
        lambda *_args, **_kwargs: pytest.fail("profile cell ran before pair binding"),
    )

    assert (
        stt_eval.main(
            [
                "--baseline-final-stt-profile",
                "sense-voice",
                "--candidate-final-stt-profile",
                "parakeet-faster-whisper",
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out) == stt_eval._SAFE_ERROR


@pytest.mark.parametrize(
    "metadata",
    [
        SimpleNamespace(name="sense-voice", sha256="A" * 64, schema_version=1),
        SimpleNamespace(name="sense-voice", sha256="a" * 63, schema_version=1),
        SimpleNamespace(
            name="parakeet-faster-whisper",
            sha256="a" * 64,
            schema_version=1,
        ),
        SimpleNamespace(name="sense-voice", sha256="a" * 64, schema_version=True),
    ],
)
def test_profile_binding_rejects_noncanonical_or_mismatched_metadata(metadata):
    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._profile_binding(metadata, "sense-voice")
