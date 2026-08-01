from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from core.diagnostic_bundle import (
    CaptureCoordinate,
    DiagnosticObservation,
    DiagnosticStage,
    DiagnosticTrack,
    FinalModelInputRole,
    SynchronizedDiagnosticBundle,
    validate_manifest,
)
from tools.prepare_diagnostic_streaming_stt_corpus import (
    DiagnosticCorpusPreparationError,
    main,
    prepare_diagnostic_corpus,
)
from tools.streaming_stt_eval import _corpus_binding


_STREAM_ID = "sherpa-11111111111111111111111111111111"


def _diagnostic(tmp_path: Path):
    tmp_path.chmod(0o700)
    tracks = {
        role: tmp_path / f"run.{role.value}.wav" for role in DiagnosticTrack
    }
    timeline = tmp_path / "run.timeline.jsonl"
    manifest = tmp_path / "run.diagnostic.json"
    spool = tmp_path / "run.final-input.f32le"
    bundle = SynchronizedDiagnosticBundle(
        tracks,
        timeline,
        manifest,
        16_000,
        final_model_input_path=spool,
    )
    coordinate = CaptureCoordinate(
        sequence=1,
        sample_rate_hz=48_000,
        captured_started_at=10.0,
        captured_at=10.01,
        capture_epoch=2,
        source_generation=3,
        capture_generation=4,
        source_sample_start=0,
        source_sample_end=480,
    )
    span = bundle.write_frame(
        {
            role: np.full(160, 0.1 + index * 0.1, dtype="float32")
            for index, role in enumerate(DiagnosticTrack)
        },
        coordinate,
    )
    assert span is not None
    before = (
        DiagnosticObservation(
            stage=DiagnosticStage.ENDPOINT_EVALUATED,
            monotonic_ns=10,
            span=span,
            reason="asr",
            basis="acoustic",
            completion_state="unavailable",
            acoustic_endpoint=True,
            vad_active=False,
            early_endpoint_allowed=True,
            trailing_silence_sec=0.8,
            boolean_value=True,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.ENDPOINT_COMMITTED,
            monotonic_ns=10,
            span=span,
            stream_id=_STREAM_ID,
            utterance_id="u1",
            revision=1,
            reason="asr",
            basis="acoustic",
            completion_state="unavailable",
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.ASR_STREAMING_FINAL,
            monotonic_ns=11,
            span=span,
            stream_id=_STREAM_ID,
            utterance_id="u1",
            revision=1,
            outcome="nonempty",
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.FINALIZER_STARTED,
            monotonic_ns=12,
            span=span,
            stream_id=_STREAM_ID,
            utterance_id="u1",
            revision=1,
        ),
    )
    for observation in before:
        assert bundle.observe(observation)
    exact = np.array([1.25, -1.5, 0.1234567, -0.7654321], dtype="float32")
    receipt = bundle.write_final_model_input(
        exact,
        stream_id=_STREAM_ID,
        utterance_id="u1",
        capture_epoch=2,
        capture_generation=4,
        revision=1,
        role=FinalModelInputRole.SELECTED_ASR_SEGMENT,
    )
    assert receipt is not None
    for observation in (
        DiagnosticObservation(
            stage=DiagnosticStage.FINAL_SELECTION,
            monotonic_ns=13,
            span=span,
            stream_id=_STREAM_ID,
            utterance_id="u1",
            revision=1,
            selected_source="streaming",
            outcome="unavailable",
            support=0,
            boolean_value=False,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.FINAL_DISPATCHED,
            monotonic_ns=14,
            span=span,
            stream_id=_STREAM_ID,
            utterance_id="u1",
            revision=1,
            outcome="callback_returned",
        ),
    ):
        assert bundle.observe(observation)
    bundle.close()
    assert validate_manifest(manifest)
    return manifest, tracks[DiagnosticTrack.MODEL_ASR_TAP], receipt, exact


def _labels(
    path: Path,
    *,
    manifest: Path,
    input_index: int,
    input_sha256: str,
    expected_text: str = "search in my vault",
) -> Path:
    payload = {
        "schema_version": 1,
        "diagnostic_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "cases": [
            {
                "id": "paul-vault-001",
                "input_index": input_index,
                "input_sha256": input_sha256,
                "expected_text": expected_text,
                "tags": ["owner-voice", "quiet-room"],
            }
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def test_publishes_exact_spool_slice_as_private_schema_v3(tmp_path: Path) -> None:
    manifest, continuous_asr, receipt, exact = _diagnostic(tmp_path)
    labels = _labels(
        tmp_path / "labels.json",
        manifest=manifest,
        input_index=receipt.input_index,
        input_sha256=receipt.sha256,
    )
    destination = tmp_path / "private-corpus"

    loaded = prepare_diagnostic_corpus(
        diagnostic_manifest=manifest,
        labels=labels,
        output_dir=destination,
    )

    expected = exact.astype("<f4", copy=False).tobytes()
    assert loaded.schema_version == 3
    assert loaded.provenance is not None
    assert loaded.provenance.kind == "private-diagnostic-v1"
    assert loaded.provenance.suite == "final-model-input"
    binding = _corpus_binding(loaded)
    assert binding["schema_version"] == 3
    assert binding["provenance"] == loaded.provenance.as_dict()
    assert loaded.cases[0].audio_bytes == expected
    assert (destination / "paul-vault-001.f32le").read_bytes() == expected
    assert continuous_asr.read_bytes() != expected
    receipt_payload = (destination / "preparation-receipt.json").read_text(
        encoding="utf-8"
    )
    assert "search in my vault" not in receipt_payload
    assert str(manifest) not in receipt_payload
    assert str(labels) not in receipt_payload
    assert {
        path.name: path.stat().st_mode & 0o777 for path in destination.iterdir()
    } == {
        "corpus.json": 0o600,
        "paul-vault-001.f32le": 0o600,
        "preparation-receipt.json": 0o600,
    }


@pytest.mark.parametrize("mutation", ("manifest", "input", "index"))
def test_label_bindings_fail_closed_before_output(
    tmp_path: Path, mutation: str
) -> None:
    manifest, _continuous_asr, receipt, _exact = _diagnostic(tmp_path)
    labels = _labels(
        tmp_path / "labels.json",
        manifest=manifest,
        input_index=receipt.input_index,
        input_sha256=receipt.sha256,
    )
    payload = json.loads(labels.read_text(encoding="utf-8"))
    if mutation == "manifest":
        payload["diagnostic_manifest_sha256"] = "0" * 64
    elif mutation == "input":
        payload["cases"][0]["input_sha256"] = "0" * 64
    else:
        payload["cases"][0]["input_index"] = receipt.input_index + 1
    labels.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    destination = tmp_path / "private-corpus"

    with pytest.raises(DiagnosticCorpusPreparationError):
        prepare_diagnostic_corpus(
            diagnostic_manifest=manifest,
            labels=labels,
            output_dir=destination,
        )

    assert not destination.exists()


def test_labels_must_be_private_and_single_link(tmp_path: Path) -> None:
    manifest, _continuous_asr, receipt, _exact = _diagnostic(tmp_path)
    labels = _labels(
        tmp_path / "labels.json",
        manifest=manifest,
        input_index=receipt.input_index,
        input_sha256=receipt.sha256,
    )
    labels.chmod(0o644)

    with pytest.raises(DiagnosticCorpusPreparationError):
        prepare_diagnostic_corpus(
            diagnostic_manifest=manifest,
            labels=labels,
            output_dir=tmp_path / "private-corpus",
        )


def test_cli_output_is_aggregate_only_on_success_and_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, _continuous_asr, receipt, _exact = _diagnostic(tmp_path)
    canary = "private transcript canary words"
    labels = _labels(
        tmp_path / "private-labels.json",
        manifest=manifest,
        input_index=receipt.input_index,
        input_sha256=receipt.sha256,
        expected_text=canary,
    )
    destination = tmp_path / "private-corpus"

    assert main(
        [
            "--diagnostic-manifest",
            str(manifest),
            "--labels",
            str(labels),
            "--output-dir",
            str(destination),
        ]
    ) == 0
    success = capsys.readouterr().out
    assert canary not in success
    assert str(manifest) not in success
    assert str(labels) not in success
    assert json.loads(success)["ok"] is True

    missing = tmp_path / "private-missing-canary.json"
    assert main(
        [
            "--diagnostic-manifest",
            str(missing),
            "--labels",
            str(labels),
            "--output-dir",
            str(tmp_path / "unused"),
        ]
    ) == 2
    failure = capsys.readouterr().out
    assert canary not in failure
    assert str(missing) not in failure
    assert json.loads(failure) == {
        "ok": False,
        "error": "diagnostic_streaming_corpus_prerequisites_unavailable",
    }


@pytest.mark.parametrize("argument_failure", ("malformed", "unknown"))
def test_cli_argument_failures_never_echo_private_arguments(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    argument_failure: str,
) -> None:
    canary = "private-transcript-canary"
    private_path = tmp_path / canary
    if argument_failure == "malformed":
        argv = ["--diagnostic-manifest", str(private_path), "--labels"]
    else:
        argv = [
            "--diagnostic-manifest",
            str(tmp_path / "manifest.json"),
            "--labels",
            str(tmp_path / "labels.json"),
            "--output-dir",
            str(tmp_path / "output"),
            canary,
        ]

    assert main(argv) == 2
    captured = capsys.readouterr()
    assert canary not in captured.out
    assert canary not in captured.err
    assert str(private_path) not in captured.out
    assert str(private_path) not in captured.err
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "ok": False,
        "error": "diagnostic_streaming_corpus_prerequisites_unavailable",
    }
