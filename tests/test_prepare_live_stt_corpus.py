from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from core.diagnostic_bundle import (
    CaptureCoordinate,
    DiagnosticObservation,
    DiagnosticStage,
    DiagnosticTrack,
    FinalModelInputReceipt,
    FinalModelInputRole,
    SynchronizedDiagnosticBundle,
    validate_manifest,
)
from tools import prepare_live_stt_corpus as live_stt_corpus
from tools import recorded_stt_eval
from tools.prepare_diagnostic_streaming_stt_corpus import (
    DiagnosticCorpusPreparationError,
)
from tools.prepare_live_stt_corpus import (
    LiveSttCorpusPreparationError,
    main,
    prepare_live_stt_corpus,
)


_STREAM_ID = "sherpa-11111111111111111111111111111111"
_CASES = (
    {
        "id": "owner-vault-001",
        "expected_text": "search in my vault for the private canary alpha",
        "tags": ["owner-voice", "expected-tool.vault.search"],
    },
    {
        "id": "owner-reminder-002",
        "expected_text": "remind me about the private canary beta tomorrow",
        "tags": ["owner-voice", "expected-tool.reminder.create"],
    },
)


@dataclass(frozen=True)
class _Diagnostic:
    manifest: Path
    timeline: Path
    spool: Path
    receipts: tuple[FinalModelInputReceipt, ...]
    samples: tuple[np.ndarray, ...]


def _private_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    path.chmod(0o700)
    return path


def _complete_turn(
    bundle: SynchronizedDiagnosticBundle,
    span,
    *,
    index: int,
    samples: np.ndarray,
) -> FinalModelInputReceipt:
    revision = index + 1
    utterance_id = f"u{revision}"
    timestamp = 100 + revision * 10
    for observation in (
        DiagnosticObservation(
            stage=DiagnosticStage.ENDPOINT_EVALUATED,
            monotonic_ns=timestamp,
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
            monotonic_ns=timestamp,
            span=span,
            stream_id=_STREAM_ID,
            utterance_id=utterance_id,
            revision=revision,
            reason="asr",
            basis="acoustic",
            completion_state="unavailable",
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.ASR_STREAMING_FINAL,
            monotonic_ns=timestamp + 2,
            span=span,
            stream_id=_STREAM_ID,
            utterance_id=utterance_id,
            revision=revision,
            outcome="nonempty",
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.FINALIZER_STARTED,
            monotonic_ns=timestamp + 3,
            span=span,
            stream_id=_STREAM_ID,
            utterance_id=utterance_id,
            revision=revision,
        ),
    ):
        assert bundle.observe(observation)
    receipt = bundle.write_final_model_input(
        samples,
        stream_id=_STREAM_ID,
        utterance_id=utterance_id,
        capture_epoch=span.coordinate.capture_epoch,
        capture_generation=span.coordinate.capture_generation,
        revision=revision,
        role=FinalModelInputRole.SELECTED_ASR_SEGMENT,
    )
    assert receipt is not None
    for observation in (
        DiagnosticObservation(
            stage=DiagnosticStage.FINAL_SELECTION,
            monotonic_ns=timestamp + 4,
            span=span,
            stream_id=_STREAM_ID,
            utterance_id=utterance_id,
            revision=revision,
            selected_source="streaming",
            outcome="unavailable",
            support=0,
            boolean_value=False,
        ),
        DiagnosticObservation(
            stage=DiagnosticStage.FINAL_DISPATCHED,
            monotonic_ns=timestamp + 5,
            span=span,
            stream_id=_STREAM_ID,
            utterance_id=utterance_id,
            revision=revision,
            outcome="callback_returned",
        ),
    ):
        assert bundle.observe(observation)
    return receipt


def _diagnostic(path: Path) -> _Diagnostic:
    root = _private_dir(path)
    tracks = {
        role: root / f"run.{role.value}.wav" for role in DiagnosticTrack
    }
    timeline = root / "run.timeline.jsonl"
    manifest = root / "run.diagnostic.json"
    spool = root / "run.final-input.f32le"
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
    samples = (
        np.array([1.25, -1.5, 0.1234567, -0.7654321], dtype="float32"),
        np.array([-0.33333334, 0.75, -0.125], dtype="float32"),
    )
    receipts = tuple(
        _complete_turn(bundle, span, index=index, samples=value)
        for index, value in enumerate(samples)
    )
    bundle.close()
    assert validate_manifest(manifest)
    return _Diagnostic(manifest, timeline, spool, receipts, samples)


def _plan(
    path: Path,
    *,
    cases: tuple[dict[str, object], ...] = _CASES,
    payload: dict[str, object] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    value = payload or {"schema_version": 1, "cases": list(cases)}
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def _prepare(
    tmp_path: Path,
    *,
    diagnostic: _Diagnostic | None = None,
    plan: Path | None = None,
    labels: Path | None = None,
    output: Path | None = None,
):
    diagnostic = diagnostic or _diagnostic(tmp_path / "diagnostic")
    plan = plan or _plan(tmp_path / "private" / "reference-plan.json")
    labels = labels or tmp_path / "private" / "generated-labels.json"
    output = output or tmp_path / "private" / "prepared-corpus"
    return prepare_live_stt_corpus(
        diagnostic_manifest=diagnostic.manifest,
        reference_plan=plan,
        labels_output=labels,
        output_dir=output,
    )


def test_binds_every_receipt_in_order_and_exports_exact_f32_corpus(
    tmp_path: Path,
) -> None:
    diagnostic = _diagnostic(tmp_path / "diagnostic")
    plan = _plan(tmp_path / "private" / "reference-plan.json")
    labels = tmp_path / "private" / "generated-labels.json"
    output = tmp_path / "private" / "prepared-corpus"

    loaded = _prepare(
        tmp_path,
        diagnostic=diagnostic,
        plan=plan,
        labels=labels,
        output=output,
    )

    label_bytes = labels.read_bytes()
    label_payload = json.loads(label_bytes)
    assert label_payload == {
        "schema_version": 1,
        "diagnostic_manifest_sha256": hashlib.sha256(
            diagnostic.manifest.read_bytes()
        ).hexdigest(),
        "cases": [
            {
                **case,
                "input_index": receipt.input_index,
                "input_sha256": receipt.sha256,
            }
            for case, receipt in zip(_CASES, diagnostic.receipts, strict=True)
        ],
    }
    assert labels.stat().st_mode & 0o777 == 0o600
    assert labels.stat().st_nlink == 1
    assert output.stat().st_mode & 0o777 == 0o700
    assert {
        child.name: child.stat().st_mode & 0o777 for child in output.iterdir()
    } == {
        "corpus.json": 0o600,
        "owner-reminder-002.f32le": 0o600,
        "owner-vault-001.f32le": 0o600,
        "preparation-receipt.json": 0o600,
    }

    evaluation_load = recorded_stt_eval._load_corpus(loaded.path)
    items, digest = evaluation_load
    assert digest == loaded.digest
    assert [item.reference for item in items] == [
        case["expected_text"] for case in _CASES
    ]
    assert [item.tags for item in items] == [
        (
            "private-diagnostic",
            "selected_asr_segment",
            *case["tags"],
        )
        for case in _CASES
    ]
    assert [
        item.samples.astype("<f4", copy=False).tobytes() for item in items
    ] == [
        samples.astype("<f4", copy=False).tobytes()
        for samples in diagnostic.samples
    ]
    recorded_stt_eval._verify_loaded_corpus(evaluation_load)


@pytest.mark.parametrize("case_count", (1, 3))
def test_reference_count_must_exactly_match_receipt_count_before_publication(
    tmp_path: Path,
    case_count: int,
) -> None:
    diagnostic = _diagnostic(tmp_path / "diagnostic")
    cases = list(_CASES)
    if case_count == 3:
        cases.append(
            {
                "id": "owner-app-003",
                "expected_text": "open the calculator",
                "tags": ["owner-voice", "expected-tool.app.open"],
            }
        )
    plan = _plan(
        tmp_path / "private" / "reference-plan.json",
        cases=tuple(cases[:case_count]),
    )
    labels = tmp_path / "private" / "generated-labels.json"
    output = tmp_path / "private" / "prepared-corpus"

    with pytest.raises(LiveSttCorpusPreparationError):
        _prepare(
            tmp_path,
            diagnostic=diagnostic,
            plan=plan,
            labels=labels,
            output=output,
        )

    assert not labels.exists()
    assert not output.exists()


def test_empty_reference_plan_is_rejected_before_publication(tmp_path: Path) -> None:
    diagnostic = _diagnostic(tmp_path / "diagnostic")
    plan = _plan(tmp_path / "private" / "reference-plan.json", cases=())
    labels = tmp_path / "private" / "generated-labels.json"
    output = tmp_path / "private" / "prepared-corpus"

    with pytest.raises(LiveSttCorpusPreparationError):
        _prepare(
            tmp_path,
            diagnostic=diagnostic,
            plan=plan,
            labels=labels,
            output=output,
        )

    assert not labels.exists()
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "schema-bool",
        "top-level-extra",
        "case-missing-field",
        "case-extra-field",
        "bad-id",
        "duplicate-id",
        "expected-not-string",
        "expected-empty",
        "expected-non-normalizing",
        "tags-not-list",
        "bad-tag",
        "duplicate-tag",
    ),
)
def test_malformed_reference_plan_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    diagnostic = _diagnostic(tmp_path / "diagnostic")
    payload: dict[str, object] = {
        "schema_version": 1,
        "cases": [dict(case) for case in _CASES],
    }
    cases = payload["cases"]
    assert isinstance(cases, list)
    first = cases[0]
    assert isinstance(first, dict)
    if mutation == "schema-bool":
        payload["schema_version"] = True
    elif mutation == "top-level-extra":
        payload["private_note"] = "must not be accepted"
    elif mutation == "case-missing-field":
        first.pop("expected_text")
    elif mutation == "case-extra-field":
        first["input_index"] = 0
    elif mutation == "bad-id":
        first["id"] = "Owner Voice"
    elif mutation == "duplicate-id":
        second = cases[1]
        assert isinstance(second, dict)
        second["id"] = first["id"]
    elif mutation == "expected-not-string":
        first["expected_text"] = 123
    elif mutation == "expected-empty":
        first["expected_text"] = "   "
    elif mutation == "expected-non-normalizing":
        first["expected_text"] = "!?..."
    elif mutation == "tags-not-list":
        first["tags"] = "owner-voice"
    elif mutation == "bad-tag":
        first["tags"] = ["Owner Voice"]
    else:
        first["tags"] = ["owner-voice", "owner-voice"]
    plan = _plan(
        tmp_path / "private" / "reference-plan.json",
        payload=payload,
    )
    labels = tmp_path / "private" / "generated-labels.json"
    output = tmp_path / "private" / "prepared-corpus"

    with pytest.raises(LiveSttCorpusPreparationError):
        _prepare(
            tmp_path,
            diagnostic=diagnostic,
            plan=plan,
            labels=labels,
            output=output,
        )

    assert not labels.exists()
    assert not output.exists()


def test_duplicate_json_keys_in_reference_plan_are_rejected(tmp_path: Path) -> None:
    diagnostic = _diagnostic(tmp_path / "diagnostic")
    plan = tmp_path / "private" / "reference-plan.json"
    _private_dir(plan.parent)
    plan.write_text(
        '{"schema_version":1,"schema_version":1,"cases":[]}\n',
        encoding="utf-8",
    )
    plan.chmod(0o600)
    labels = plan.parent / "generated-labels.json"
    output = plan.parent / "prepared-corpus"

    with pytest.raises(LiveSttCorpusPreparationError):
        _prepare(
            tmp_path,
            diagnostic=diagnostic,
            plan=plan,
            labels=labels,
            output=output,
        )

    assert not labels.exists()
    assert not output.exists()


@pytest.mark.parametrize(
    "evidence_failure",
    (
        "missing-manifest",
        "incomplete-manifest",
        "mutated-manifest",
        "missing-timeline",
        "mutated-timeline",
        "missing-spool",
        "mutated-spool",
    ),
)
def test_invalid_or_mutated_diagnostic_evidence_creates_no_outputs(
    tmp_path: Path,
    evidence_failure: str,
) -> None:
    diagnostic = _diagnostic(tmp_path / "diagnostic")
    manifest = diagnostic.manifest
    if evidence_failure == "missing-manifest":
        manifest = diagnostic.manifest.parent / "missing.diagnostic.json"
    elif evidence_failure == "incomplete-manifest":
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["complete"] = False
        manifest.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif evidence_failure == "mutated-manifest":
        manifest.write_text("{}\n", encoding="utf-8")
    elif evidence_failure == "missing-timeline":
        diagnostic.timeline.unlink()
    elif evidence_failure == "mutated-timeline":
        diagnostic.timeline.write_bytes(diagnostic.timeline.read_bytes() + b"x")
    elif evidence_failure == "missing-spool":
        diagnostic.spool.unlink()
    else:
        changed = bytearray(diagnostic.spool.read_bytes())
        changed[0] ^= 0x01
        diagnostic.spool.write_bytes(changed)
    plan = _plan(tmp_path / "private" / "reference-plan.json")
    labels = plan.parent / "generated-labels.json"
    output = plan.parent / "prepared-corpus"

    with pytest.raises(LiveSttCorpusPreparationError):
        prepare_live_stt_corpus(
            diagnostic_manifest=manifest,
            reference_plan=plan,
            labels_output=labels,
            output_dir=output,
        )

    assert not labels.exists()
    assert not output.exists()


@pytest.mark.parametrize(
    "plan_failure",
    ("public-mode", "symlink", "hardlink", "public-parent"),
)
def test_reference_plan_must_be_private_regular_and_single_link(
    tmp_path: Path,
    plan_failure: str,
) -> None:
    diagnostic = _diagnostic(tmp_path / "diagnostic")
    original = _plan(tmp_path / "private" / "reference-plan.json")
    plan = original
    if plan_failure == "public-mode":
        original.chmod(0o644)
    elif plan_failure == "symlink":
        plan = original.parent / "plan-link.json"
        plan.symlink_to(original.name)
    elif plan_failure == "hardlink":
        plan = original.parent / "plan-hardlink.json"
        os.link(original, plan)
    else:
        original.parent.chmod(0o755)
    labels = tmp_path / "destination" / "generated-labels.json"
    output = tmp_path / "destination" / "prepared-corpus"

    with pytest.raises(LiveSttCorpusPreparationError):
        _prepare(
            tmp_path,
            diagnostic=diagnostic,
            plan=plan,
            labels=labels,
            output=output,
        )

    assert not labels.exists()
    assert not output.exists()


def test_existing_labels_are_not_overwritten_and_corpus_is_not_created(
    tmp_path: Path,
) -> None:
    diagnostic = _diagnostic(tmp_path / "diagnostic")
    plan = _plan(tmp_path / "private" / "reference-plan.json")
    labels = plan.parent / "generated-labels.json"
    sentinel = b"private-existing-label-sentinel\n"
    labels.write_bytes(sentinel)
    labels.chmod(0o600)
    output = plan.parent / "prepared-corpus"

    with pytest.raises(LiveSttCorpusPreparationError):
        _prepare(
            tmp_path,
            diagnostic=diagnostic,
            plan=plan,
            labels=labels,
            output=output,
        )

    assert labels.read_bytes() == sentinel
    assert not output.exists()


def test_existing_output_preflight_does_not_create_labels_or_overwrite_output(
    tmp_path: Path,
) -> None:
    diagnostic = _diagnostic(tmp_path / "diagnostic")
    plan = _plan(tmp_path / "private" / "reference-plan.json")
    labels = plan.parent / "generated-labels.json"
    output = plan.parent / "prepared-corpus"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_bytes(b"existing-output")

    with pytest.raises(LiveSttCorpusPreparationError):
        _prepare(
            tmp_path,
            diagnostic=diagnostic,
            plan=plan,
            labels=labels,
            output=output,
        )

    assert not labels.exists()
    assert sentinel.read_bytes() == b"existing-output"


def test_downstream_export_failure_retains_complete_generated_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic = _diagnostic(tmp_path / "diagnostic")
    plan = _plan(tmp_path / "private" / "reference-plan.json")
    labels = plan.parent / "generated-labels.json"
    output = plan.parent / "prepared-corpus"
    observed: dict[str, object] = {}

    def fail_export(**kwargs):
        generated = Path(kwargs["labels"])
        observed["bytes"] = generated.read_bytes()
        observed["digest"] = hashlib.sha256(generated.read_bytes()).hexdigest()
        observed["mode"] = generated.stat().st_mode & 0o777
        observed["nlink"] = generated.stat().st_nlink
        assert kwargs["expected_labels_sha256"] == observed["digest"]
        raise DiagnosticCorpusPreparationError()

    monkeypatch.setattr(live_stt_corpus, "prepare_diagnostic_corpus", fail_export)

    with pytest.raises(LiveSttCorpusPreparationError):
        _prepare(
            tmp_path,
            diagnostic=diagnostic,
            plan=plan,
            labels=labels,
            output=output,
        )

    assert labels.read_bytes() == observed["bytes"]
    assert hashlib.sha256(labels.read_bytes()).hexdigest() == observed["digest"]
    assert observed["mode"] == 0o600
    assert observed["nlink"] == 1
    assert not output.exists()


def test_generated_label_replacement_race_is_rejected_before_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic = _diagnostic(tmp_path / "diagnostic")
    plan = _plan(tmp_path / "private" / "reference-plan.json")
    labels = plan.parent / "generated-labels.json"
    output = plan.parent / "prepared-corpus"
    original_export = live_stt_corpus.prepare_diagnostic_corpus
    replacement_digest: str | None = None

    def replace_then_export(**kwargs):
        nonlocal replacement_digest
        generated = Path(kwargs["labels"])
        original_digest = hashlib.sha256(generated.read_bytes()).hexdigest()
        assert kwargs["expected_labels_sha256"] == original_digest
        payload = json.loads(generated.read_text(encoding="utf-8"))
        payload["cases"][0]["expected_text"] = "replacement race private canary"
        replacement = generated.with_name("replacement-labels.json")
        replacement.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        replacement.chmod(0o600)
        replacement_digest = hashlib.sha256(replacement.read_bytes()).hexdigest()
        os.replace(replacement, generated)
        return original_export(**kwargs)

    monkeypatch.setattr(
        live_stt_corpus,
        "prepare_diagnostic_corpus",
        replace_then_export,
    )

    with pytest.raises(LiveSttCorpusPreparationError):
        _prepare(
            tmp_path,
            diagnostic=diagnostic,
            plan=plan,
            labels=labels,
            output=output,
        )

    assert replacement_digest is not None
    assert hashlib.sha256(labels.read_bytes()).hexdigest() == replacement_digest
    assert labels.stat().st_mode & 0o777 == 0o600
    assert labels.stat().st_nlink == 1
    assert not output.exists()


def test_same_bytes_label_replacement_race_is_rejected_by_inode_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic = _diagnostic(tmp_path / "diagnostic")
    plan = _plan(tmp_path / "private" / "reference-plan.json")
    labels = plan.parent / "generated-labels.json"
    output = plan.parent / "prepared-corpus"
    original_export = live_stt_corpus.prepare_diagnostic_corpus
    original_inode: tuple[int, int] | None = None

    def replace_then_export(**kwargs):
        nonlocal original_inode
        generated = Path(kwargs["labels"])
        raw = generated.read_bytes()
        metadata = generated.stat()
        original_inode = (metadata.st_dev, metadata.st_ino)
        assert kwargs["expected_labels_inode"] == original_inode
        replacement = generated.with_name("same-bytes-labels.json")
        replacement.write_bytes(raw)
        replacement.chmod(0o600)
        os.replace(replacement, generated)
        assert (generated.stat().st_dev, generated.stat().st_ino) != original_inode
        return original_export(**kwargs)

    monkeypatch.setattr(
        live_stt_corpus,
        "prepare_diagnostic_corpus",
        replace_then_export,
    )

    with pytest.raises(LiveSttCorpusPreparationError):
        _prepare(
            tmp_path,
            diagnostic=diagnostic,
            plan=plan,
            labels=labels,
            output=output,
        )

    assert original_inode is not None
    assert labels.is_file()
    assert labels.stat().st_mode & 0o777 == 0o600
    assert labels.stat().st_nlink == 1
    assert not output.exists()


def test_reference_plan_mutation_after_label_publication_stops_before_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic = _diagnostic(tmp_path / "diagnostic")
    plan = _plan(tmp_path / "private" / "reference-plan.json")
    labels = plan.parent / "generated-labels.json"
    output = plan.parent / "prepared-corpus"
    original_publish = live_stt_corpus._publish_private_labels

    def publish_then_mutate(
        path: Path,
        payload: bytes,
    ) -> live_stt_corpus._PublishedLabelWitness:
        witness = original_publish(path, payload)
        changed = json.loads(plan.read_text(encoding="utf-8"))
        changed["cases"][0]["expected_text"] = "post-publication private canary"
        plan.write_text(
            json.dumps(changed, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        plan.chmod(0o600)
        return witness

    monkeypatch.setattr(
        live_stt_corpus,
        "_publish_private_labels",
        publish_then_mutate,
    )

    with pytest.raises(LiveSttCorpusPreparationError):
        _prepare(
            tmp_path,
            diagnostic=diagnostic,
            plan=plan,
            labels=labels,
            output=output,
        )

    assert labels.is_file()
    assert labels.stat().st_mode & 0o777 == 0o600
    assert labels.stat().st_nlink == 1
    assert not output.exists()


@pytest.mark.parametrize("tag_count, succeeds", ((30, True), (31, False)))
def test_plan_tag_budget_reserves_exporter_owned_tags(
    tmp_path: Path,
    tag_count: int,
    succeeds: bool,
) -> None:
    diagnostic = _diagnostic(tmp_path / "diagnostic")
    cases = [dict(case) for case in _CASES]
    cases[0]["tags"] = [
        "expected-tool.vault.search",
        *(f"private-tag-{index:02d}" for index in range(tag_count - 1)),
    ]
    plan = _plan(
        tmp_path / "private" / "reference-plan.json",
        cases=tuple(cases),
    )
    labels = plan.parent / "generated-labels.json"
    output = plan.parent / "prepared-corpus"

    if succeeds:
        loaded = _prepare(
            tmp_path,
            diagnostic=diagnostic,
            plan=plan,
            labels=labels,
            output=output,
        )
        assert len(loaded.cases[0].tags) == 32
    else:
        with pytest.raises(LiveSttCorpusPreparationError):
            _prepare(
                tmp_path,
                diagnostic=diagnostic,
                plan=plan,
                labels=labels,
                output=output,
            )
        assert not labels.exists()
        assert not output.exists()


@pytest.mark.parametrize(
    "reserved_tags",
    (
        ["private-diagnostic"],
        ["selected_asr_segment"],
        ["model_gate_segment"],
        ["expected-tool.unknown"],
        ["expected-tool.none", "expected-tool.vault.search"],
    ),
)
def test_plan_rejects_exporter_owned_unknown_and_multiple_route_tags(
    tmp_path: Path,
    reserved_tags: list[str],
) -> None:
    diagnostic = _diagnostic(tmp_path / "diagnostic")
    cases = [dict(case) for case in _CASES]
    cases[0]["tags"] = ["owner-voice", *reserved_tags]
    plan = _plan(
        tmp_path / "private" / "reference-plan.json",
        cases=tuple(cases),
    )
    labels = plan.parent / "generated-labels.json"
    output = plan.parent / "prepared-corpus"

    with pytest.raises(LiveSttCorpusPreparationError):
        _prepare(
            tmp_path,
            diagnostic=diagnostic,
            plan=plan,
            labels=labels,
            output=output,
        )

    assert not labels.exists()
    assert not output.exists()


@pytest.mark.parametrize(
    "destination_failure",
    ("dangling-label-symlink", "public-label-parent", "symlink-label-parent"),
)
def test_labels_destination_requires_absent_entry_and_private_real_parent(
    tmp_path: Path,
    destination_failure: str,
) -> None:
    diagnostic = _diagnostic(tmp_path / "diagnostic")
    plan = _plan(tmp_path / "private" / "reference-plan.json")
    labels_parent = _private_dir(tmp_path / "labels-destination")
    labels = labels_parent / "generated-labels.json"
    if destination_failure == "dangling-label-symlink":
        labels.symlink_to("missing-private-target.json")
    elif destination_failure == "public-label-parent":
        labels_parent.chmod(0o755)
    else:
        target = _private_dir(tmp_path / "real-label-parent")
        alias = tmp_path / "label-parent-alias"
        alias.symlink_to(target, target_is_directory=True)
        labels = alias / "generated-labels.json"
    output_parent = _private_dir(tmp_path / "corpus-destination")
    output = output_parent / "prepared-corpus"

    with pytest.raises(LiveSttCorpusPreparationError):
        _prepare(
            tmp_path,
            diagnostic=diagnostic,
            plan=plan,
            labels=labels,
            output=output,
        )

    assert not output.exists()
    if destination_failure != "dangling-label-symlink":
        assert not labels.exists()


@pytest.mark.parametrize("alias_kind", ("same", "lexical-alias"))
def test_labels_and_corpus_destinations_cannot_alias(
    tmp_path: Path,
    alias_kind: str,
) -> None:
    diagnostic = _diagnostic(tmp_path / "diagnostic")
    plan = _plan(tmp_path / "private" / "reference-plan.json")
    shared = plan.parent / "shared-output"
    labels = shared
    output = shared
    if alias_kind == "lexical-alias":
        alias_parent = _private_dir(plan.parent / "alias-parent")
        labels = alias_parent / ".." / "shared-output"

    with pytest.raises(LiveSttCorpusPreparationError):
        _prepare(
            tmp_path,
            diagnostic=diagnostic,
            plan=plan,
            labels=labels,
            output=output,
        )

    assert not shared.exists()


@pytest.mark.parametrize("containment", ("labels-in-output", "output-in-labels"))
def test_labels_and_corpus_destinations_cannot_contain_each_other(
    tmp_path: Path,
    containment: str,
) -> None:
    diagnostic = _diagnostic(tmp_path / "diagnostic")
    plan = _plan(tmp_path / "private" / "reference-plan.json")
    labels = plan.parent / "generated-labels.json"
    output = plan.parent / "prepared-corpus"
    if containment == "labels-in-output":
        labels = output / "generated-labels.json"
    else:
        output = labels / "prepared-corpus"

    with pytest.raises(LiveSttCorpusPreparationError):
        _prepare(
            tmp_path,
            diagnostic=diagnostic,
            plan=plan,
            labels=labels,
            output=output,
        )

    assert not labels.exists()
    assert not output.exists()


def test_cli_success_and_failure_are_aggregate_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    diagnostic = _diagnostic(tmp_path / "diagnostic-private-path-canary")
    plan = _plan(tmp_path / "private-plan-path-canary" / "reference-plan.json")
    labels = (
        _private_dir(tmp_path / "labels-path-canary") / "generated-labels.json"
    )
    output = _private_dir(tmp_path / "corpus-path-canary") / "prepared-corpus"
    private_values = (
        *(str(case["expected_text"]) for case in _CASES),
        *(str(tag) for case in _CASES for tag in case["tags"]),
        str(diagnostic.manifest),
        str(plan),
        str(labels),
        str(output),
    )

    assert main(
        [
            "--diagnostic-manifest",
            str(diagnostic.manifest),
            "--reference-plan",
            str(plan),
            "--labels-output",
            str(labels),
            "--output-dir",
            str(output),
        ]
    ) == 0
    captured = capsys.readouterr()
    success = json.loads(captured.out)
    assert success["ok"] is True
    assert success["cases"] == 2
    assert captured.err == ""
    assert all(value not in captured.out for value in private_values)

    missing_plan = tmp_path / "private-missing-plan-path-canary.json"
    assert main(
        [
            "--diagnostic-manifest",
            str(diagnostic.manifest),
            "--reference-plan",
            str(missing_plan),
            "--labels-output",
            str(tmp_path / "unused-labels"),
            "--output-dir",
            str(tmp_path / "unused-corpus"),
        ]
    ) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["ok"] is False
    assert all(value not in captured.out for value in (*private_values, str(missing_plan)))


@pytest.mark.parametrize("argument_failure", ("missing-value", "unknown"))
def test_cli_argument_errors_never_echo_private_arguments(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    argument_failure: str,
) -> None:
    canary = "private-argument-canary"
    private_path = tmp_path / canary
    if argument_failure == "missing-value":
        argv = ["--diagnostic-manifest", str(private_path), "--reference-plan"]
    else:
        argv = [
            "--diagnostic-manifest",
            str(private_path),
            "--reference-plan",
            str(tmp_path / "plan.json"),
            "--labels-output",
            str(tmp_path / "labels.json"),
            "--output-dir",
            str(tmp_path / "output"),
            canary,
        ]

    assert main(argv) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert canary not in captured.out
    assert str(private_path) not in captured.out
    assert json.loads(captured.out)["ok"] is False
