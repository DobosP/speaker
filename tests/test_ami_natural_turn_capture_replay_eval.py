from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import threading
from types import MappingProxyType

import pytest

from core.engine import TranscriptAbortReason
from tools import ami_natural_turn_capture_replay_eval as subject
from tools import capture_replay_eval
from tools.capture_replay import (
    LoadedReplayCorpus,
    ReplayAssertion,
    ReplayCase,
    ReplayTrack,
    SampleInterval,
)
from tools.capture_replay.ami_natural_turn import (
    AmiNaturalTurnCase,
    AmiNaturalTurnDialogueAct,
    LoadedAmiNaturalTurnFixture,
)
from tools.capture_replay.metrics import (
    ReplayRunRecord,
    TimedFinal,
    aggregate_metrics,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _private_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    path.chmod(0o700)
    return path


def _act(
    identity: str,
    speaker: str,
    reference: str,
    start: int,
    end: int,
) -> AmiNaturalTurnDialogueAct:
    return AmiNaturalTurnDialogueAct(
        dialogue_act_id=identity,
        speaker_id=speaker,
        type_href="da-types.xml#id(test)",
        source_start_sample=start,
        source_end_sample=end,
        output_start_sample=32_000 + start,
        output_end_sample=32_000 + end,
        reference=reference,
        reference_sha256=_sha(reference),
    )


def _fixture(root: Path, *, production: bool = False) -> LoadedAmiNaturalTurnFixture:
    pcm = bytes(128_000 * 4)
    windows = (
        (
            "nested-backchannel",
            "host phrase",
            "brief answer",
            SampleInterval(37_760, 43_040),
            None,
        ),
        (
            "adjacent-exchange",
            "first adjacent turn",
            "second adjacent turn",
            None,
            SampleInterval(45_280, 46_720),
        ),
    )
    metadata: list[AmiNaturalTurnCase] = []
    replay: list[ReplayCase] = []
    for window_index, (window, left, right, overlap, gap) in enumerate(windows):
        for channel in ("close", "far"):
            index = len(metadata)
            case_id = f"ami-es2004a-{window}-{channel}"
            references = (
                _act(f"da-{window_index}-left", "D", left, 100, 200),
                _act(f"da-{window_index}-right", "B", right, 210, 300),
            )
            reference = f"{left} {right}"
            track = ReplayTrack(
                role="mic",
                path=root / f"{case_id}.f32le",
                sha256=hashlib.sha256(pcm).hexdigest(),
                samples=128_000,
                pcm_bytes=pcm,
            )
            tags = (
                "ami",
                "natural-turn",
                "overlapping-speakers" if overlap is not None else "turn-transition",
            )
            replay.append(
                ReplayCase(
                    case_id=case_id,
                    sample_rate_hz=16_000,
                    frame_samples=1_600,
                    tracks=MappingProxyType({"mic": track}),
                    speech_intervals=(),
                    speaker_intervals=(),
                    word_intervals=(),
                    assertion=ReplayAssertion.TRANSCRIPT,
                    expected_text=reference,
                    commands=(),
                    tags=tags,
                    aec_delay_samples=0,
                )
            )
            metadata.append(
                AmiNaturalTurnCase(
                    case_id=case_id,
                    window_id=window,
                    channel=channel,
                    replay_case_index=index,
                    samples=128_000,
                    frame_samples=1_600,
                    source_start_sample=1_000,
                    source_end_sample=18_600,
                    source_offset_sample=32_000,
                    source_end_output_sample=49_600,
                    reference=reference,
                    reference_sha256=_sha(reference),
                    dialogue_acts=references,
                    adjacency_pair_id=(
                        None if overlap is not None else "adjacency-pair-23"
                    ),
                    overlap_interval=overlap,
                    gap_interval=gap,
                )
            )
    corpus = LoadedReplayCorpus(
        path=root / "capture-replay.json",
        digest="a" * 64,
        schema_version=1,
        purpose="ami-natural-turn-test",
        sample_rate_hz=16_000,
        frame_samples=1_600,
        cases=tuple(replay),
        audio_bytes=sum(len(case.mic.pcm_bytes) for case in replay),
    )
    return LoadedAmiNaturalTurnFixture(
        root=root,
        fixture_id="ami-es2004a-natural-turn-v1-test",
        production_evidence=production,
        lock_sha256="b" * 64,
        lock_recipe_sha256="c" * 64,
        labels_sha256="d" * 64,
        metadata_sha256="e" * 64,
        manifest_sha256=corpus.digest,
        receipt_sha256="f" * 64,
        source_contract_sha256="1" * 64,
        preparer_closure_sha256="2" * 64,
        sample_rate_hz=16_000,
        frame_samples=1_600,
        case_samples=128_000,
        replay_corpus=corpus,
        cases=tuple(metadata),
        directory_identity=(1, 2, 3),
        file_identities=(("capture-replay.json", (4, 5, 6)),),
        lock_path=root / "test.lock.json",
        lock_size_bytes=100,
    )


def _records(fixture: LoadedAmiNaturalTurnFixture) -> tuple[ReplayRunRecord, ...]:
    result: list[ReplayRunRecord] = []
    for repeat in range(subject.REPEATS):
        for case_index, case in enumerate(fixture.cases):
            text = (
                " ".join(reversed(case.overlap_references))
                if case.overlap_interval is not None
                else case.reference
            )
            result.append(
                ReplayRunRecord(
                    case_index=case_index,
                    repeat=repeat,
                    finals=(
                        TimedFinal(
                            text,
                            emitted_at=0.4,
                            speech_start_at=0.1,
                            speech_end_at=0.2,
                            endpoint_committed_at=0.3,
                            utterance_id=f"u-{repeat}-{case_index}",
                        ),
                    ),
                    wall_seconds=0.01,
                )
            )
    return tuple(result)


def _outcome_factory(fixture: LoadedAmiNaturalTurnFixture):
    def factory(corpus, configured, **kwargs):
        assert corpus is fixture.replay_corpus
        assert kwargs == {
            "repeats": subject.REPEATS,
            "provider": None,
            "asr_threads": None,
        }
        records = _records(fixture)
        executed = capture_replay_eval._replay_config(
            configured,
            provider=None,
            asr_threads=None,
        )
        report = {
            "execution_complete": True,
            "quality_verdict": "diagnostic_only",
            "schema_version": 1,
            "corpus": {
                "manifest_sha256": corpus.digest,
                "cases": len(corpus.cases),
                "audio_bytes": corpus.audio_bytes,
                "audio_seconds": round(
                    sum(case.duration_seconds for case in corpus.cases),
                    3,
                ),
            },
            "engine": {
                "configured_config_sha256": capture_replay_eval._json_digest(
                    vars(configured)
                ),
                "executed_config_sha256": capture_replay_eval._json_digest(
                    vars(executed)
                ),
                "artifact_metadata_sha256": (
                    capture_replay_eval._artifact_metadata_digest(executed)
                ),
                "evaluator_source_sha256": (
                    capture_replay_eval._evaluator_source_digest()
                ),
                "evaluator_source_files": len(
                    capture_replay_eval._evaluator_source_files()
                ),
                "provider": executed.provider,
                "asr_final_backend_configured": "",
                "asr_final_verifier_backend_configured": "",
                "asr_final_verifier_execution": {
                    "requested": False,
                    "available_after_run": False,
                    "attempted_decodes": 0,
                    "completed_decodes": 0,
                    "decode_errors": 0,
                    "nonempty_decodes": 0,
                },
                "final_execution": "inline-deterministic-replay",
                "streaming_decode_execution": "single-session-synchronous-replay",
                "public_identity_models_loaded": False,
                "output_models_loaded": False,
                "model_load_ms": 0.0,
            },
            "metrics": aggregate_metrics(
                corpus,
                records,
                repeats=subject.REPEATS,
            ),
        }
        return capture_replay_eval._CaptureReplayEvaluationOutcome(
            report=report,
            records=records,
        )

    return factory


class Harness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.fixture_root = _private_dir(tmp_path / "fixture-input")
        self.config_root = _private_dir(tmp_path / "config-input")
        self.artifact_root = _private_dir(tmp_path / "model-input")
        self.output_root = _private_dir(tmp_path / "report-output")
        self.model = self.artifact_root / "encoder.onnx"
        self.model.write_bytes(b"synthetic model identity")
        self.config = self.config_root / "config.json"
        self.local = self.config_root / "config.local.json"
        self.config.write_text(
            json.dumps(
                {
                    "device": "test",
                    "sherpa": {"asr_encoder": str(self.model)},
                }
            ),
            encoding="utf-8",
        )
        self.fixture = _fixture(self.fixture_root)
        monkeypatch.setattr(
            subject,
            "load_ami_natural_turn_fixture",
            lambda path: (
                self.fixture
                if Path(path) == self.fixture_root
                else (_ for _ in ()).throw(RuntimeError())
            ),
        )
        monkeypatch.setattr(
            subject,
            "verify_ami_natural_turn_fixture_snapshot",
            lambda value: (
                None if value is self.fixture else (_ for _ in ()).throw(RuntimeError())
            ),
        )
        # The managed host intentionally keeps /tmp below a Git ancestor;
        # publication mechanics are exercised here with synthetic inputs only.
        monkeypatch.setattr(subject, "_has_git_ancestor", lambda _path: False)

    def run(self, *, output: Path | None = None) -> dict[str, object]:
        return subject.run_ami_natural_turn_capture_replay_eval(
            self.fixture_root,
            config_path=self.config,
            local_config_path=self.local,
            output_path=output,
            _outcome_factory=_outcome_factory(self.fixture),
        )

    def binding(self) -> subject._InputBinding:
        return subject._load_input_binding(
            self.fixture_root,
            self.config,
            self.local,
            device=None,
            provider=None,
            asr_threads=None,
        )


def _rebind(report: dict[str, object]) -> None:
    report["binding_sha256"] = subject._canonical_sha256(
        {key: value for key, value in report.items() if key != "binding_sha256"}
    )


def test_fake_run_preserves_generic_report_and_scores_overlap_min_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    report = harness.run()

    assert set(report) == subject._ROOT_FIELDS
    assert report["schema_version"] == 1
    assert report["kind"] == subject.KIND
    assert report["evidence"] == {
        "diagnostic_only": True,
        "manual_dialogue_acts_descriptive_only": True,
        "ordinary_overlap_wer": False,
        "orc_wer": False,
        "endpoint_ground_truth": False,
        "complete_acoustic_turn": False,
        "capture_device": False,
        "live_latency": False,
        "qualification_authority": False,
        "promotion_authority": False,
        "default_change": False,
        "model_executed": False,
    }
    assert report["fixture"] == {
        "manifest_sha256": "a" * 64,
        "receipt_sha256": "f" * 64,
        "labels_sha256": "d" * 64,
        "lock_recipe_sha256": "c" * 64,
        "cases": 4,
        "windows": 2,
        "audio_bytes": 2_048_000,
        "samples": 512_000,
        "frames": 320,
        "production_evidence": False,
    }
    assert report["natural_turn"] == {
        "coverage": {
            "cases": 4,
            "repeats": 3,
            "evaluations": 12,
            "source_complete_evaluations": 12,
            "overlap_evaluations": 6,
        },
        "overlap": {
            "ordinary_wer": None,
            "reference_order_comparable": False,
            "diagnostic": {
                "protocol": "two-utterance-min-order-wer-v1",
                "evaluations": 6,
                "substitutions": 0,
                "insertions": 0,
                "deletions": 0,
                "word_errors": 0,
                "reference_words": 24,
                "hypothesis_words": 24,
                "wer": 0.0,
            },
        },
        "terminal_counts": {
            "protocol": "manual-dialogue-act-exposure-v1",
            "manual_dialogue_act_exposures": 24,
            "typed_final_callbacks": 12,
            "typed_abort_callbacks": 0,
            "descriptive_only": True,
        },
    }
    assert (
        report["capture_replay"]["metrics"]["accuracy"]["by_condition"][
            "human_overlap"
        ]["wer"]
        is None
    )
    assert (
        report["capture_replay"]["metrics"]["accuracy"]["by_condition"][
            "human_overlap"
        ]["linearized_wer"]
        != 0.0
    )
    assert report["execution"] == {
        "closure_sha256": subject._closure_sha256(),
        "source_files": 56,
    }
    assert "brief answer host phrase" not in json.dumps(report, sort_keys=True)


def test_injected_outcome_cannot_claim_production_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.fixture = replace(harness.fixture, production_evidence=True)
    invoked = False

    def factory(*args, **kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError

    with pytest.raises(subject.AmiNaturalTurnCaptureReplayEvalError):
        subject.run_ami_natural_turn_capture_replay_eval(
            harness.fixture_root,
            config_path=harness.config,
            local_config_path=harness.local,
            _outcome_factory=factory,
        )
    assert invoked is False


@pytest.mark.parametrize("impostor", [False, 0, "", (), {}, object()])
def test_commit_state_rejects_every_non_state_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    impostor: object,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    invoked = False

    def factory(*args, **kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError

    with pytest.raises(subject.AmiNaturalTurnCaptureReplayEvalError):
        subject.run_ami_natural_turn_capture_replay_eval(
            harness.fixture_root,
            config_path=harness.config,
            local_config_path=harness.local,
            _outcome_factory=factory,
            _commit_state=impostor,
        )
    assert invoked is False


@pytest.mark.parametrize("mutation", ["order", "duplicate_id", "decreasing"])
def test_record_order_and_final_callback_order_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    real = _outcome_factory(harness.fixture)

    def bad(corpus, configured, **kwargs):
        outcome = real(corpus, configured, **kwargs)
        records = list(outcome.records)
        if mutation == "order":
            records[0], records[1] = records[1], records[0]
        else:
            first = records[0]
            extra = TimedFinal(
                "second private callback",
                emitted_at=0.2 if mutation == "decreasing" else 0.5,
                speech_start_at=0.1,
                speech_end_at=0.2,
                endpoint_committed_at=0.3,
                utterance_id=(
                    first.finals[0].utterance_id
                    if mutation == "duplicate_id"
                    else "second-id"
                ),
            )
            records[0] = replace(first, finals=(*first.finals, extra))
        return replace(outcome, records=tuple(records))

    with pytest.raises(subject.AmiNaturalTurnCaptureReplayEvalError):
        subject.run_ami_natural_turn_capture_replay_eval(
            harness.fixture_root,
            config_path=harness.config,
            local_config_path=harness.local,
            _outcome_factory=bad,
        )


def test_mid_source_final_is_valid_after_procedural_source_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    report = harness.run()
    assert report["natural_turn"]["coverage"]["source_complete_evaluations"] == 12
    assert all(
        final.emitted_at < 1.0
        for row in _records(harness.fixture)
        for final in row.finals
    )


@pytest.mark.parametrize(
    "surface",
    ["latency", "streaming", "acoustic", "throughput", "resources", "verifier"],
)
def test_generic_metrics_must_exactly_match_the_retained_record_reduction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    real = _outcome_factory(harness.fixture)

    def tampered(corpus, configured, **kwargs):
        outcome = real(corpus, configured, **kwargs)
        report = deepcopy(outcome.report)
        metrics = report["metrics"]
        if surface == "latency":
            latency = metrics["latency"]
            latency["endpoint_to_final_p50_ms"] += 1.0
            latency["endpoint_to_final_p95_ms"] += 1.0
        elif surface == "streaming":
            metrics["streaming"]["churn_token_edits"] += 1
        elif surface == "acoustic":
            metrics["acoustic"]["events"]["barge_in"] = 1
        elif surface == "throughput":
            metrics["throughput"]["wall_total_sec"] = 0.24
            metrics["throughput"]["wall_to_audio_ratio"] = 0.0025
        elif surface == "resources":
            metrics["resources"]["peak_rss_mb"] = 1.0
        elif surface == "verifier":
            verifier = report["engine"]["asr_final_verifier_execution"]
            verifier.update(
                {
                    "requested": True,
                    "available_after_run": True,
                    "attempted_decodes": 1,
                    "completed_decodes": 1,
                    "nonempty_decodes": 1,
                }
            )
        return replace(outcome, report=report)

    with pytest.raises(subject.AmiNaturalTurnCaptureReplayEvalError):
        subject.run_ami_natural_turn_capture_replay_eval(
            harness.fixture_root,
            config_path=harness.config,
            local_config_path=harness.local,
            _outcome_factory=tampered,
        )


def _mutate_bound_input(kind: str, harness: Harness) -> None:
    if kind == "fixture":
        harness.fixture = replace(harness.fixture, manifest_sha256="9" * 64)
    elif kind == "lock":
        harness.fixture = replace(harness.fixture, lock_recipe_sha256="8" * 64)
    elif kind == "config":
        harness.config.write_text(
            harness.config.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
    elif kind == "artifact":
        harness.model.write_bytes(harness.model.read_bytes() + b" drift")
    else:  # pragma: no cover - test helper contract
        raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["fixture", "lock", "config", "artifact", "closure"])
@pytest.mark.parametrize("phase", ["after_outcome", "commit_guard"])
def test_every_bound_input_is_rechecked_after_outcome_and_at_commit_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    phase: str,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    output = harness.output_root / "report.json"
    real_factory = _outcome_factory(harness.fixture)
    closure_drift = False
    real_closure = subject._closure_sha256

    def closure() -> str:
        return "7" * 64 if closure_drift else real_closure()

    monkeypatch.setattr(subject, "_closure_sha256", closure)

    def mutate() -> None:
        nonlocal closure_drift
        if kind == "closure":
            closure_drift = True
        else:
            _mutate_bound_input(kind, harness)

    def outcome(corpus, configured, **kwargs):
        value = real_factory(corpus, configured, **kwargs)
        if phase == "after_outcome":
            mutate()
        return value

    if phase == "commit_guard":
        real_publish = subject._publish_report

        def publish(*args, **kwargs):
            mutate()
            return real_publish(*args, **kwargs)

        monkeypatch.setattr(subject, "_publish_report", publish)

    with pytest.raises(subject.AmiNaturalTurnCaptureReplayEvalError):
        subject.run_ami_natural_turn_capture_replay_eval(
            harness.fixture_root,
            config_path=harness.config,
            local_config_path=harness.local,
            output_path=output,
            _outcome_factory=outcome,
        )
    assert not output.exists()


def test_outputless_return_has_its_own_final_rebind_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    real_assert = subject._assert_rebound
    calls = 0

    def rebound(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            _mutate_bound_input("config", harness)
        return real_assert(*args, **kwargs)

    monkeypatch.setattr(subject, "_assert_rebound", rebound)
    with pytest.raises(subject.AmiNaturalTurnCaptureReplayEvalError):
        harness.run()
    assert calls == 4


@pytest.mark.parametrize(
    "mutation",
    [
        "root_extra",
        "latency_extra",
        "condition_extra",
        "abort_unknown",
        "closure",
        "reference_denominator",
        "diagnostic_substitution_bounds",
        "condition_sum",
        "locked_denominators",
        "hypothesis_algebra",
        "command_counts",
        "silence_counts",
        "latency_null_equation",
        "throughput_ratio",
        "verifier_balance",
        "generic_provider_path",
        "nonfinite",
    ],
)
def test_closed_report_schema_privacy_and_math_reject_adversaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    report = deepcopy(harness.run())
    if mutation == "root_extra":
        report["debug"] = True
    elif mutation == "latency_extra":
        report["capture_replay"]["metrics"]["latency"]["rows"] = []
    elif mutation == "condition_extra":
        report["capture_replay"]["metrics"]["accuracy"]["by_condition"][
            "human_overlap"
        ]["case_ids"] = []
    elif mutation == "abort_unknown":
        report["capture_replay"]["metrics"]["streaming"]["abort_reasons"][
            "private-reason"
        ] = 0
    elif mutation == "closure":
        report["execution"]["source_files"] = 55
    elif mutation == "reference_denominator":
        diagnostic = report["natural_turn"]["overlap"]["diagnostic"]
        diagnostic["reference_words"] += 1
        diagnostic["hypothesis_words"] += 1
    elif mutation == "diagnostic_substitution_bounds":
        diagnostic = report["natural_turn"]["overlap"]["diagnostic"]
        diagnostic["substitutions"] = diagnostic["reference_words"] + 1
        diagnostic["word_errors"] = diagnostic["substitutions"]
        diagnostic["wer"] = diagnostic["word_errors"] / diagnostic["reference_words"]
    elif mutation == "condition_sum":
        accuracy = report["capture_replay"]["metrics"]["accuracy"]
        for key in subject._ACCURACY_COUNT_FIELDS - {"transcript_runs"}:
            accuracy[key] = 0
        accuracy["wer"] = None
        accuracy["cer"] = None
    elif mutation == "locked_denominators":
        accuracy = report["capture_replay"]["metrics"]["accuracy"]
        transition = accuracy["by_condition"]["turn_transition"]
        for row in (accuracy, transition):
            row["reference_words"] += 1
            row["hypothesis_words"] += 1
            row["wer"] = round(row["word_errors"] / row["reference_words"], 4)
    elif mutation == "hypothesis_algebra":
        accuracy = report["capture_replay"]["metrics"]["accuracy"]
        accuracy["hypothesis_words"] += 1
        accuracy["by_condition"]["turn_transition"]["hypothesis_words"] += 1
    elif mutation == "command_counts":
        accuracy = report["capture_replay"]["metrics"]["accuracy"]
        accuracy["command_attempts"] = 1
        accuracy["command_recall"] = 0.0
    elif mutation == "silence_counts":
        report["capture_replay"]["metrics"]["accuracy"]["silence_runs"] = 1
    elif mutation == "latency_null_equation":
        latency = report["capture_replay"]["metrics"]["latency"]
        assert latency["first_partial_observations"] == 0
        latency["first_partial_from_speech_start_p50_ms"] = 0.0
        latency["first_partial_from_speech_start_p95_ms"] = 0.0
    elif mutation == "throughput_ratio":
        report["capture_replay"]["metrics"]["throughput"]["wall_to_audio_ratio"] += 0.01
    elif mutation == "verifier_balance":
        verifier = report["capture_replay"]["engine"]["asr_final_verifier_execution"]
        verifier.update(
            {
                "requested": True,
                "available_after_run": True,
                "attempted_decodes": 1,
                "completed_decodes": 1,
                "nonempty_decodes": 1,
            }
        )
    elif mutation == "generic_provider_path":
        report["capture_replay"]["engine"]["provider"] = str(harness.fixture_root)
    elif mutation == "nonfinite":
        report["capture_replay"]["metrics"]["resources"]["peak_rss_mb"] = float("nan")
    if mutation != "nonfinite":
        _rebind(report)
    with pytest.raises(subject.AmiNaturalTurnCaptureReplayEvalError):
        subject._validate_report(report, initial=harness.binding())


@pytest.mark.parametrize("protected", ["fixture", "config", "model"])
def test_report_tree_cannot_overlap_any_input_or_artifact_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protected: str,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    parent = {
        "fixture": harness.fixture_root,
        "config": harness.config_root,
        "model": harness.artifact_root,
    }[protected]
    output = parent / "report.json"
    invoked = False

    def factory(*args, **kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError

    with pytest.raises(subject.AmiNaturalTurnCaptureReplayEvalError):
        subject.run_ami_natural_turn_capture_replay_eval(
            harness.fixture_root,
            config_path=harness.config,
            local_config_path=harness.local,
            output_path=output,
            _outcome_factory=factory,
        )
    assert invoked is False
    assert not output.exists()


@pytest.mark.parametrize(
    "hazard", ["parent_symlink", "output_symlink", "git", "public_parent"]
)
def test_report_parent_and_leaf_aliases_fail_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hazard: str,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    output_parent = harness.output_root
    output = output_parent / "unsafe.json"
    if hazard == "parent_symlink":
        target = _private_dir(tmp_path / "real-output")
        alias = tmp_path / "output-alias"
        alias.symlink_to(target, target_is_directory=True)
        output = alias / "unsafe.json"
    elif hazard == "output_symlink":
        output.symlink_to(harness.output_root / "missing-target")
    elif hazard == "git":
        monkeypatch.setattr(subject, "_has_git_ancestor", lambda path: True)
    elif hazard == "public_parent":
        output_parent.chmod(0o755)
    invoked = False

    def factory(*args, **kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError

    try:
        with pytest.raises(subject.AmiNaturalTurnCaptureReplayEvalError):
            subject.run_ami_natural_turn_capture_replay_eval(
                harness.fixture_root,
                config_path=harness.config,
                local_config_path=harness.local,
                output_path=output,
                _outcome_factory=factory,
            )
    finally:
        output_parent.chmod(0o700)
    assert invoked is False


def test_private_otmpfile_report_is_mode_600_single_link_and_no_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    output = harness.output_root / "report.json"
    report = harness.run(output=output)
    metadata = output.stat()

    assert stat.S_ISREG(metadata.st_mode)
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_nlink == 1
    assert output.read_bytes() == subject._canonical_json(report, newline=True)
    with pytest.raises(subject.AmiNaturalTurnCaptureReplayEvalError):
        harness.run(output=output)


def test_ambiguous_link_exception_recovers_only_the_staged_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    report = harness.run()
    output = harness.output_root / "ambiguous.json"
    state = subject._ReportCommitState(pending_report=report)
    real_link = os.link

    def ambiguous(*args, **kwargs):
        real_link(*args, **kwargs)
        raise OSError("ambiguous link result")

    monkeypatch.setattr(subject.os, "link", ambiguous)
    digest = subject._publish_report(
        output,
        report,
        commit_guard=lambda: True,
        state=state,
    )
    assert state.committed is True
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()


@pytest.mark.parametrize("signal_name", ["interrupt", "hup", "term"])
def test_prelink_interrupt_leaves_no_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signal_name: str,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    report = harness.run()
    output = harness.output_root / "interrupted.json"
    state = subject._ReportCommitState(pending_report=report)
    failure = (
        KeyboardInterrupt()
        if signal_name == "interrupt"
        else subject._LifecycleSignal(
            int(getattr(signal, "SIGHUP" if signal_name == "hup" else "SIGTERM"))
        )
    )

    with pytest.raises(type(failure)):
        subject._publish_report(
            output,
            report,
            commit_guard=lambda: (_ for _ in ()).throw(failure),
            state=state,
        )
    assert state.committed is False
    assert not output.exists()


def test_guard_mutation_of_staged_bytes_is_detected_before_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    report = harness.run()
    output = harness.output_root / "mutated-stage.json"
    state = subject._ReportCommitState(pending_report=report)
    real_read = os.read
    reads = 0

    def corrupt_second_read(descriptor: int, count: int) -> bytes:
        nonlocal reads
        reads += 1
        raw = real_read(descriptor, count)
        return raw if reads == 1 else raw + b"x"

    monkeypatch.setattr(subject.os, "read", corrupt_second_read)
    with pytest.raises(subject.AmiNaturalTurnCaptureReplayEvalError):
        subject._publish_report(
            output,
            report,
            commit_guard=lambda: True,
            state=state,
        )
    assert not output.exists()


def test_report_parent_mutation_in_final_guard_prevents_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    report = harness.run()
    output = harness.output_root / "parent-mutated.json"
    state = subject._ReportCommitState(pending_report=report)

    def mutate_parent() -> bool:
        harness.output_root.chmod(0o755)
        return True

    try:
        with pytest.raises(subject.AmiNaturalTurnCaptureReplayEvalError):
            subject._publish_report(
                output,
                report,
                commit_guard=mutate_parent,
                state=state,
            )
    finally:
        harness.output_root.chmod(0o700)
    assert not output.exists()


def test_reopen_rebinds_canonical_retained_report_and_rejects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    output = harness.output_root / "report.json"
    report = harness.run(output=output)
    retained, digest = subject._verify_published_report(
        output,
        fixture_dir=harness.fixture_root,
        config_path=harness.config,
        local_config_path=harness.local,
        device=None,
        provider=None,
        asr_threads=None,
    )
    assert retained == report
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()

    output.write_bytes(output.read_bytes() + b" ")
    with pytest.raises(subject.AmiNaturalTurnCaptureReplayEvalError):
        subject._verify_published_report(
            output,
            fixture_dir=harness.fixture_root,
            config_path=harness.config,
            local_config_path=harness.local,
            device=None,
            provider=None,
            asr_threads=None,
        )


def test_exact_clean_import_closure_matches_persisted_source_list() -> None:
    root = Path(subject.__file__).resolve().parents[1]
    script = """
import json
import pathlib
import sys
root = pathlib.Path.cwd().resolve()
import tools.ami_natural_turn_capture_replay_eval
files = sorted({
    pathlib.Path(module.__file__).resolve().relative_to(root).as_posix()
    for module in sys.modules.values()
    if getattr(module, "__file__", None)
    and pathlib.Path(module.__file__).resolve().is_relative_to(root)
    and pathlib.Path(module.__file__).suffix == ".py"
})
print(json.dumps(files))
"""
    completed = subprocess.run(
        [str(Path(subject.sys.executable)), "-B", "-c", script],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8")
    assert json.loads(completed.stdout) == list(subject._SOURCE_FILES)
    assert len(subject._SOURCE_FILES) == 56


class _SpawnGapProcess:
    def __init__(self) -> None:
        self.pid = 12345
        self.stdout = io.BytesIO()
        self.waited = 0
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.waited += 1
        self.returncode = -signal.SIGTERM
        return self.returncode


def test_spawn_ownership_gap_interrupt_stops_and_reaps_exact_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _SpawnGapProcess()
    masks = 0
    killed: list[tuple[int, int]] = []

    def mask(operation, signals):
        nonlocal masks
        masks += 1
        if masks == 2:
            raise KeyboardInterrupt
        return set()

    monkeypatch.setattr(subject.signal, "pthread_sigmask", mask)
    monkeypatch.setattr(subject.subprocess, "Popen", lambda *a, **kw: process)

    def killpg(pid: int, signum: int) -> None:
        if signum == 0 and process.returncode is not None:
            raise ProcessLookupError
        if signum != 0:
            killed.append((pid, signum))

    monkeypatch.setattr(subject.os, "killpg", killpg)

    with pytest.raises(KeyboardInterrupt):
        subject._run_guarded_worker([], timeout=1)
    assert killed == [(process.pid, signal.SIGTERM)]
    assert process.waited == 1


class _CompletedSpawnProcess:
    def __init__(self) -> None:
        self.pid = 23456
        self.stdout = io.BytesIO()
        self.returncode: int | None = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        del timeout
        self.returncode = 0
        return self.returncode


def test_spawned_child_preexec_unblocks_every_lifecycle_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _CompletedSpawnProcess()
    calls: list[tuple[int, tuple[int, ...]]] = []
    previous = {signal.SIGUSR1}
    spawned: dict[str, object] = {}

    def mask(operation, signals):
        calls.append((operation, tuple(signals)))
        return previous if operation == signal.SIG_BLOCK else set()

    def popen(*args, **kwargs):
        spawned.update(kwargs)
        kwargs["preexec_fn"]()
        return process

    monkeypatch.setattr(subject.signal, "pthread_sigmask", mask)
    monkeypatch.setattr(subject.subprocess, "Popen", popen)
    monkeypatch.setattr(
        subject.os,
        "killpg",
        lambda _pid, _signum: (_ for _ in ()).throw(ProcessLookupError()),
    )

    assert subject._run_guarded_worker([], timeout=1) == (0, b"")
    lifecycle = subject._commit_signals()
    assert calls == [
        (signal.SIG_BLOCK, lifecycle),
        (signal.SIG_UNBLOCK, lifecycle),
        (signal.SIG_SETMASK, tuple(previous)),
        (signal.SIG_BLOCK, lifecycle),
        (signal.SIG_SETMASK, tuple(previous)),
    ]
    assert spawned["start_new_session"] is True
    assert spawned["close_fds"] is True


class _StubbornProcess:
    pid = 34567

    def __init__(self) -> None:
        self.waits = 0

    def poll(self):
        return None

    def wait(self, timeout=None):
        self.waits += 1
        if self.waits == 1:
            raise subprocess.TimeoutExpired("private-command", timeout)
        return -signal.SIGKILL


def test_terminate_reaps_after_sigkill(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _StubbornProcess()
    killed: list[tuple[int, int]] = []
    group_alive = True

    def killpg(pid: int, signum: int) -> None:
        nonlocal group_alive
        if signum == 0:
            if not group_alive:
                raise ProcessLookupError
            return
        killed.append((pid, signum))
        if signum == signal.SIGKILL:
            group_alive = False

    monkeypatch.setattr(subject.os, "killpg", killpg)

    assert subject._terminate(process) is True

    assert killed == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]
    assert process.waits == 2


class _HeldGroupPipe:
    def __init__(self, released: threading.Event) -> None:
        self._released = released

    def read(self, _count: int) -> bytes:
        if not self._released.wait(timeout=1):
            raise AssertionError("owned descendant was not stopped")
        return b""

    def close(self) -> None:
        self._released.set()


class _ExitedLeaderWithDescendant:
    pid = 45678

    def __init__(self, released: threading.Event) -> None:
        self.stdout = _HeldGroupPipe(released)
        self.returncode = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        del timeout
        return self.returncode


def test_normal_leader_exit_still_kills_descendant_group_holding_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released = threading.Event()
    process = _ExitedLeaderWithDescendant(released)
    group_alive = True
    signals: list[int] = []

    def killpg(pid: int, signum: int) -> None:
        nonlocal group_alive
        assert pid == process.pid
        if signum == 0:
            if not group_alive:
                raise ProcessLookupError
            return
        signals.append(signum)
        if signum == signal.SIGKILL:
            group_alive = False
            released.set()

    monkeypatch.setattr(subject.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(subject.os, "killpg", killpg)
    monkeypatch.setattr(subject, "_PROCESS_GROUP_TERM_SECONDS", 0.01)

    assert subject._run_guarded_worker([], timeout=1) == (0, b"")
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert group_alive is False
    assert released.is_set()


def _cli_args(harness: Harness, output: Path) -> list[str]:
    return [
        "--fixture-dir",
        str(harness.fixture_root),
        "--config",
        str(harness.config),
        "--local-config",
        str(harness.local),
        "--watchdog-seconds",
        "1",
        "--report",
        str(output),
    ]


def _lifecycle_failure(signal_name: str) -> BaseException:
    if signal_name == "interrupt":
        return KeyboardInterrupt()
    return subject._LifecycleSignal(
        int(getattr(signal, "SIGHUP" if signal_name == "hup" else "SIGTERM"))
    )


@pytest.mark.parametrize("signal_name", ["interrupt", "hup", "term"])
@pytest.mark.parametrize("phase", ["prelink", "postlink"])
def test_worker_and_guarded_cli_preserve_lifecycle_terminal_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    signal_name: str,
    phase: str,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    output = harness.output_root / "terminal.json"
    report = deepcopy(harness.run())
    report["evidence"]["model_executed"] = True
    _rebind(report)
    assert (
        subject._validate_report(
            report,
            initial=harness.binding(),
            expected_model_executed=True,
        )
        == report
    )

    def worker_run(*_args, output_path, _commit_state, **_kwargs):
        if phase == "prelink":
            raise _lifecycle_failure(signal_name)
        _commit_state.pending_report = report
        subject._publish_report(
            Path(output_path),
            report,
            commit_guard=lambda: True,
            state=_commit_state,
        )
        raise _lifecycle_failure(signal_name)

    monkeypatch.setattr(
        subject,
        "run_ami_natural_turn_capture_replay_eval",
        worker_run,
    )

    def same_process_worker(raw_argv, *, timeout, _ownership):
        assert timeout == 1
        _ownership.spawned = True
        _ownership.cleanup_verified = True
        captured = bytearray()
        real_write = os.write

        def capture(descriptor: int, raw: bytes) -> int:
            if descriptor == 1:
                captured.extend(raw)
                return len(raw)
            return real_write(descriptor, raw)

        with monkeypatch.context() as local:
            local.setattr(subject.os, "write", capture)
            returncode = subject._worker_main(raw_argv)
        if phase == "postlink":
            # Exercise the parent's terminal reopen path as well as the worker's
            # shared commit-state recovery.
            raise _lifecycle_failure(signal_name)
        return returncode, bytes(captured)

    monkeypatch.setattr(subject, "_run_guarded_worker", same_process_worker)
    code = subject.guarded_main(_cli_args(harness, output))
    payload = json.loads(capfd.readouterr().out)

    if phase == "prelink":
        expected = (
            130
            if signal_name == "interrupt"
            else 128
            + int(
                getattr(
                    signal,
                    "SIGHUP" if signal_name == "hup" else "SIGTERM",
                )
            )
        )
        assert code == expected
        assert payload == subject._SAFE_ERROR
        assert not output.exists()
    else:
        assert code == 0
        assert payload == subject._receipt(
            report,
            hashlib.sha256(output.read_bytes()).hexdigest(),
        )
        assert output.read_bytes() == subject._canonical_json(report, newline=True)


def test_guarded_cli_never_recovers_a_valid_report_owned_by_an_earlier_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    output = harness.output_root / "preexisting.json"
    report = deepcopy(harness.run())
    report["evidence"]["model_executed"] = True
    _rebind(report)
    state = subject._ReportCommitState(pending_report=report)
    subject._publish_report(
        output,
        report,
        commit_guard=lambda: True,
        state=state,
    )
    original = output.read_bytes()
    spawned = False

    def worker(*args, **kwargs):
        nonlocal spawned
        spawned = True
        raise AssertionError

    monkeypatch.setattr(subject, "_run_guarded_worker", worker)
    code = subject.guarded_main(_cli_args(harness, output))

    assert code == 2
    assert json.loads(capfd.readouterr().out) == subject._SAFE_ERROR
    assert spawned is False
    assert output.read_bytes() == original


@pytest.mark.parametrize("failure_kind", ["timeout", "interrupt", "hup", "term"])
def test_unverified_worker_cleanup_can_never_be_recovered_as_cli_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    failure_kind: str,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    output = harness.output_root / "cleanup-failed.json"
    report = deepcopy(harness.run())
    report["evidence"]["model_executed"] = True
    _rebind(report)
    cleanups = 0

    class Process:
        pid = 56789
        stdout = io.BytesIO()

        def wait(self, timeout=None):
            state = subject._ReportCommitState(pending_report=report)
            subject._publish_report(
                output,
                report,
                commit_guard=lambda: True,
                state=state,
            )
            if failure_kind == "timeout":
                raise subprocess.TimeoutExpired("private-command", timeout)
            raise _lifecycle_failure(failure_kind)

    def cleanup(_process) -> bool:
        nonlocal cleanups
        cleanups += 1
        return False

    monkeypatch.setattr(subject.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(subject, "_terminate", cleanup)

    code = subject.guarded_main(_cli_args(harness, output))

    assert code == 2
    assert json.loads(capfd.readouterr().out) == subject._SAFE_ERROR
    assert cleanups == 1
    assert output.read_bytes() == subject._canonical_json(report, newline=True)


@pytest.mark.parametrize("signal_name", ["interrupt", "hup", "term"])
@pytest.mark.parametrize("phase", ["precommit", "postcommit"])
def test_cleanup_lifecycle_is_deferred_until_group_and_pipe_are_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    signal_name: str,
    phase: str,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    output = harness.output_root / "cleanup-lifecycle.json"
    report = deepcopy(harness.run())
    report["evidence"]["model_executed"] = True
    _rebind(report)
    released = threading.Event()
    group_alive = True
    cleanups = 0

    class Process:
        pid = 67890
        stdout = _HeldGroupPipe(released)

        def wait(self, timeout=None):
            del timeout
            if phase == "postcommit":
                state = subject._ReportCommitState(pending_report=report)
                subject._publish_report(
                    output,
                    report,
                    commit_guard=lambda: True,
                    state=state,
                )
            return 0

    def cleanup(_process) -> bool:
        nonlocal cleanups, group_alive
        cleanups += 1
        if cleanups == 1:
            raise _lifecycle_failure(signal_name)
        group_alive = False
        released.set()
        return True

    monkeypatch.setattr(subject.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(subject, "_terminate", cleanup)

    code = subject.guarded_main(_cli_args(harness, output))
    payload = json.loads(capfd.readouterr().out)

    assert cleanups == 2
    assert group_alive is False
    assert released.is_set()
    if phase == "precommit":
        expected = (
            130
            if signal_name == "interrupt"
            else 128
            + int(
                getattr(
                    signal,
                    "SIGHUP" if signal_name == "hup" else "SIGTERM",
                )
            )
        )
        assert code == expected
        assert payload == subject._SAFE_ERROR
        assert not output.exists()
    else:
        assert code == 0
        assert output.read_bytes() == subject._canonical_json(report, newline=True)
        assert payload == subject._receipt(
            report,
            hashlib.sha256(output.read_bytes()).hexdigest(),
        )


@pytest.mark.parametrize("signal_name", ["hup", "term"])
@pytest.mark.parametrize("injection", ["recovery", "stdout"])
def test_guarded_cli_keeps_lifecycle_handlers_through_terminal_recovery_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    signal_name: str,
    injection: str,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    output = harness.output_root / "late-lifecycle.json"
    report = deepcopy(harness.run())
    report["evidence"]["model_executed"] = True
    _rebind(report)
    signum = int(getattr(signal, "SIGHUP" if signal_name == "hup" else "SIGTERM"))

    def worker(_raw_argv, *, timeout, _ownership):
        assert timeout == 1
        _ownership.spawned = True
        _ownership.cleanup_verified = True
        state = subject._ReportCommitState(pending_report=report)
        subject._publish_report(
            output,
            report,
            commit_guard=lambda: True,
            state=state,
        )
        raise subject._LifecycleSignal(signum)

    monkeypatch.setattr(subject, "_run_guarded_worker", worker)
    if injection == "recovery":
        real_verify = subject._verify_published_report
        calls = 0

        def interrupted_verify(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise subject._LifecycleSignal(signum)
            return real_verify(*args, **kwargs)

        monkeypatch.setattr(subject, "_verify_published_report", interrupted_verify)
    else:
        real_write = os.write
        interrupted = False

        def interrupted_write(descriptor: int, raw: bytes) -> int:
            nonlocal interrupted
            if descriptor == 1 and not interrupted:
                interrupted = True
                raise subject._LifecycleSignal(signum)
            return real_write(descriptor, raw)

        monkeypatch.setattr(subject.os, "write", interrupted_write)

    code = subject.guarded_main(_cli_args(harness, output))
    captured = capfd.readouterr().out

    assert code == 0
    assert output.read_bytes() == subject._canonical_json(report, newline=True)
    if injection == "recovery":
        assert json.loads(captured) == subject._receipt(
            report,
            hashlib.sha256(output.read_bytes()).hexdigest(),
        )
    else:
        assert captured == ""


def test_worker_watchdog_limit_rejects_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    output = _private_dir(tmp_path / "output") / "report.json"
    invoked = False

    def run(*args, **kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError

    monkeypatch.setattr(subject, "_run_guarded_worker", run)
    code = subject.guarded_main(
        [
            "--fixture-dir",
            str(tmp_path / "fixture"),
            "--config",
            str(tmp_path / "config.json"),
            "--local-config",
            str(tmp_path / "local.json"),
            "--watchdog-seconds",
            str(subject._MAX_WATCHDOG_SECONDS + 1),
            "--report",
            str(output),
        ]
    )
    assert code == 2
    assert invoked is False
    assert json.loads(capfd.readouterr().out) == subject._SAFE_ERROR
    assert not output.exists()


def test_parser_has_no_repeat_or_quality_override() -> None:
    actions = {
        option
        for action in subject._parser()._actions
        for option in action.option_strings
    }
    assert "--fixture-dir" in actions
    assert "--watchdog-seconds" in actions
    assert "--repeats" not in actions
    assert "--accept-quality" not in actions


def test_abort_callbacks_are_descriptive_not_compared_to_manual_exposures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    real = _outcome_factory(harness.fixture)

    def with_abort(corpus, configured, **kwargs):
        outcome = real(corpus, configured, **kwargs)
        first = replace(
            outcome.records[0],
            abort_reasons=(TranscriptAbortReason.SHUTDOWN,),
        )
        records = (first, *outcome.records[1:])
        report = deepcopy(outcome.report)
        report["metrics"] = aggregate_metrics(
            corpus,
            records,
            repeats=subject.REPEATS,
        )
        return replace(outcome, report=report, records=records)

    report = subject.run_ami_natural_turn_capture_replay_eval(
        harness.fixture_root,
        config_path=harness.config,
        local_config_path=harness.local,
        _outcome_factory=with_abort,
    )
    counts = report["natural_turn"]["terminal_counts"]
    assert counts["manual_dialogue_act_exposures"] == 24
    assert counts["typed_final_callbacks"] == 12
    assert counts["typed_abort_callbacks"] == 1
    assert counts["descriptive_only"] is True
