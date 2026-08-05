from __future__ import annotations

import hashlib
import json
import logging
import os
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import stat
import struct
import sys
from types import SimpleNamespace

import pytest

from core.engine import EngineCallbacks
from core.engines.sherpa import (
    FinalTranscriptDecision,
    FinalTranscriptSource,
    SherpaConfig,
)
from tools import production_final_stt_eval as evaluate
from tools.streaming_stt.corpus import (
    CorpusCase,
    CorpusProvenance,
    LoadedCorpus,
    load_corpus,
)
from tools.streaming_stt.final_metrics import (
    FinalDecisionRecord,
    aggregate_final_metrics,
)


_LOCK_RAW_SHA256 = (
    "1ae5a5845a70b69aa7e1d32e6f799566e2c715f8202b2d86760a6cef73d51564"
)
_RECIPE_SHA256 = (
    "4d4e7a1cab6e498fdcda7a74cfab08a01e6bd8ac9c292a4ad940f4c09046433e"
)
_REAL_CORPUS_ENV = "SPEAKER_PRODUCTION_FINAL_STT_CORPUS"


def _case(
    case_id: str,
    *,
    assertion: str,
    tags: tuple[str, ...],
    commands: tuple[str, ...] = (),
    expected_text: str = "private ordinary speech",
    forbidden_commands: tuple[str, ...] = ("forbidden private command",),
    samples: int = 1,
) -> CorpusCase:
    audio = struct.pack("<f", 0.25) * samples
    return CorpusCase(
        case_id=case_id,
        source_path=Path(f"/private/audio/{case_id}.f32le"),
        sha256=hashlib.sha256(audio).hexdigest(),
        samples=samples,
        expected_text=expected_text,
        assertion=assertion,
        commands=commands,
        forbidden_commands=forbidden_commands,
        tags=tags,
        audio_bytes=audio,
    )


def _exact_shape_corpus(tmp_path: Path) -> LoadedCorpus:
    cases: list[CorpusCase] = []
    for index in range(20):
        cases.append(
            _case(
                f"gsc-positive-{index}",
                assertion="transcript",
                expected_text="private positive command",
                commands=("private positive command",),
                tags=("public-command-noise", "gsc", "command-positive"),
            )
        )
    for index in range(5):
        cases.append(
            _case(
                f"gsc-speech-negative-{index}",
                assertion="speech_negative",
                expected_text="",
                tags=(
                    "public-command-noise",
                    "gsc",
                    "command-negative",
                    "speech-negative",
                ),
            )
        )
    for index in range(5):
        cases.append(
            _case(
                f"gsc-silence-{index}",
                assertion="silence",
                expected_text="",
                tags=("public-command-noise", "gsc", "silence"),
            )
        )
    for index in range(6):
        cases.append(
            _case(
                f"eccc-positive-{index}",
                assertion="transcript",
                expected_text="private noisy positive command",
                commands=("private noisy positive command",),
                tags=(
                    "public-command-noise",
                    "eccc",
                    "noisy",
                    "command-positive",
                ),
            )
        )
    for index in range(21):
        cases.append(
            _case(
                f"eccc-negative-{index}",
                assertion="transcript",
                tags=(
                    "public-command-noise",
                    "eccc",
                    "noisy",
                    "command-negative",
                ),
            )
        )
    return LoadedCorpus(
        path=tmp_path / "private-corpus" / "corpus.json",
        digest="c" * 64,
        schema_version=4,
        purpose=evaluate._EXPECTED_PURPOSE,
        provenance=CorpusProvenance(
            kind="public-command-noise-v1",
            suite="command-noise-v1",
            manifest_sha256=_LOCK_RAW_SHA256,
            metadata_sha256="d" * 64,
            source_set_sha256="e" * 64,
        ),
        cases=tuple(cases),
        audio_bytes=sum(len(case.audio_bytes) for case in cases),
    )


def _receipt(corpus: LoadedCorpus) -> dict[str, object]:
    assert corpus.provenance is not None
    return {
        "schema_version": 1,
        "kind": "public-command-noise-preparation-receipt-v1",
        "recipe_sha256": _RECIPE_SHA256,
        "recipe_lock_sha256": _LOCK_RAW_SHA256,
        "production_sources": True,
        "accepted_terms": ["CC-BY-4.0"],
        "source_set_sha256": corpus.provenance.source_set_sha256,
        "counts": {
            "sources": 2,
            "cases": 57,
            "command_positive": 26,
            "command_negative": 26,
            "assertions": {
                "transcript": 47,
                "speech_negative": 5,
                "silence": 5,
            },
        },
        "audio": {
            "encoding": "f32le",
            "sample_rate_hz": 16_000,
            "channels": 1,
            "samples": sum(case.samples for case in corpus.cases),
            "pcm_set_sha256": "f" * 64,
        },
        "source_aggregates": [
            {
                "source_id": "google-speech-commands-v002-test",
                "cases": 30,
                "selection_sha256": "1" * 64,
            },
            {
                "source_id": "english-consistent-confusion-v1.2",
                "cases": 27,
                "selection_sha256": "2" * 64,
            },
        ],
    }


def _install_exact_shape_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    receipt: dict[str, object],
) -> str:
    monkeypatch.setattr(
        evaluate,
        "_load_recipe_lock",
        lambda _path: SimpleNamespace(
            raw_sha256=_LOCK_RAW_SHA256,
            recipe_sha256=_RECIPE_SHA256,
        ),
    )
    payload = json.dumps(
        receipt,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    receipt_sha256 = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        evaluate,
        "read_regular_bounded",
        lambda _path, **_kwargs: SimpleNamespace(data=payload),
    )
    monkeypatch.setattr(evaluate, "_EXPECTED_CORPUS_SHA256", "c" * 64)
    monkeypatch.setattr(
        evaluate,
        "_EXPECTED_RECEIPT_SHA256",
        receipt_sha256,
    )
    monkeypatch.setattr(evaluate, "_EXPECTED_SOURCE_SET_SHA256", "e" * 64)
    monkeypatch.setattr(evaluate, "_EXPECTED_PCM_SET_SHA256", "f" * 64)
    monkeypatch.setattr(evaluate, "_require_private_corpus", lambda _corpus: None)
    return receipt_sha256


def _config_inputs(tmp_path: Path) -> evaluate._ConfigInputs:
    config_path = tmp_path / "config.json"
    local_path = tmp_path / "config.local.json"
    config_bytes = json.dumps(
        {
            "device": "desktop",
            "device_profiles": {
                "desktop": {"sherpa": {"asr_num_threads": 1}},
            },
            "sherpa": {
                "asr_encoder": "models/private-encoder.onnx",
                "provider": "cpu",
                "asr_final_backend": "sense_voice",
                "asr_final_verifier_backend": "faster_whisper",
            },
        }
    ).encode("utf-8")
    local_bytes = b'{"sherpa":{"provider":"cpu"}}'
    config_path.write_bytes(config_bytes)
    local_path.write_bytes(local_bytes)
    return evaluate._load_config_inputs(config_path, local_path)


def _execution(
    *,
    decisions: int = 1,
    offline: dict[str, int] | None = None,
    verifier: dict[str, int] | None = None,
    selected: dict[str, int] | None = None,
) -> evaluate._Execution:
    return evaluate._Execution(
        streaming=(),
        selected=(),
        decisions=decisions,
        offline_outcomes={"decoded": decisions} if offline is None else offline,
        verifier_outcomes={"consensus": decisions}
        if verifier is None
        else verifier,
        selected_sources={"offline": decisions} if selected is None else selected,
    )


def _evaluation_corpus(tmp_path: Path, *cases: CorpusCase) -> LoadedCorpus:
    return LoadedCorpus(
        path=tmp_path / "SENTINEL_PRIVATE_CORPUS.json",
        digest="3" * 64,
        schema_version=4,
        purpose="SENTINEL_PRIVATE_PURPOSE",
        provenance=None,
        cases=tuple(cases),
        audio_bytes=sum(len(case.audio_bytes) for case in cases),
    )


def _install_evaluation_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    execution: evaluate._Execution,
) -> tuple[dict[str, object], dict[str, str]]:
    runtime = {
        "python": {
            "version": "3.12.7",
            "executable_sha256": "9" * 64,
        },
        "distributions": {
            label: {
                "version": "1.2.3",
                "record_sha256": hashlib.sha256(label.encode("ascii")).hexdigest(),
            }
            for label, _distribution in evaluate._RUNTIME_DISTRIBUTIONS
        },
    }
    provider = {
        "streaming_offline_requested": "cpu",
        "streaming_offline_attestation": "explicit_cpu_provider_available",
        "verifier_requested": "cuda:0-fp16",
        "verifier_attestation": "cuda_only_adapter_completed_decode",
    }
    monkeypatch.setattr(
        evaluate,
        "_validate_exact_corpus",
        lambda _corpus: "5" * 64,
    )
    monkeypatch.setattr(
        evaluate,
        "_verify_private_corpus_snapshot",
        lambda _corpus: None,
    )
    monkeypatch.setattr(evaluate, "_verify_config_inputs", lambda _inputs: None)
    monkeypatch.setattr(evaluate, "_config_digest", lambda _config: "6" * 64)
    monkeypatch.setattr(evaluate, "_source_digest", lambda: "7" * 64)
    monkeypatch.setattr(evaluate, "_runtime_identities", lambda: runtime)
    monkeypatch.setattr(
        evaluate,
        "_provider_attestation",
        lambda _config: provider,
    )
    monkeypatch.setattr(
        evaluate,
        "_configured_model_digest",
        lambda _config, _root: "8" * 64,
    )
    monkeypatch.setattr(
        evaluate,
        "_execute_replay",
        lambda *_args, **_kwargs: execution,
    )
    return runtime, provider


def test_cli_failure_is_one_generic_privacy_safe_receipt(capsys) -> None:
    private = "/private/SENTINEL-corpus-and-transcript"

    code = evaluate.main(["--corpus", private])

    captured = capsys.readouterr()
    assert code == 2
    assert json.loads(captured.out) == {
        "error": "production_final_stt_prerequisites_unavailable",
        "ok": False,
    }
    assert captured.err == ""
    assert private not in captured.out
    assert "required" not in captured.out


def test_config_inputs_are_absolute_same_parent_strict_snapshots(
    tmp_path: Path,
) -> None:
    inputs = _config_inputs(tmp_path)

    assert inputs.root == tmp_path.resolve()
    assert inputs.config_path == (tmp_path / "config.json").resolve()
    assert inputs.local_path == (tmp_path / "config.local.json").resolve()
    assert inputs.digest == evaluate._digest_labeled_bytes(
        (("config", inputs.config_bytes), ("local", inputs.local_bytes))
    )
    effective = evaluate._effective_config(inputs, None)
    assert effective.asr_encoder == "models/private-encoder.onnx"
    assert effective.provider == "cpu"
    assert effective.asr_final_backend == "sense_voice"
    assert effective.asr_final_verifier_backend == "faster_whisper"

    with pytest.raises(evaluate.ProductionFinalEvaluationError):
        evaluate._load_config_inputs("config.json", inputs.local_path)

    other = tmp_path / "other"
    other.mkdir()
    other_local = other / "config.local.json"
    other_local.write_bytes(b"{}")
    with pytest.raises(evaluate.ProductionFinalEvaluationError):
        evaluate._load_config_inputs(inputs.config_path, other_local)

    duplicate = replace(
        inputs,
        config_bytes=(
            b'{"sherpa":{"asr_encoder":"first","asr_encoder":"second"}}'
        ),
    )
    with pytest.raises(evaluate.ProductionFinalEvaluationError):
        evaluate._effective_config(duplicate, None)


def test_config_snapshot_rejects_mutation_and_symlink(
    tmp_path: Path,
) -> None:
    inputs = _config_inputs(tmp_path)
    (tmp_path / "config.local.json").write_bytes(
        b'{"sherpa":{"provider":"cpu","asr_num_threads":2}}'
    )

    with pytest.raises(evaluate.ProductionFinalEvaluationError):
        evaluate._verify_config_inputs(inputs)

    link = tmp_path / "linked.local.json"
    link.symlink_to(tmp_path / "config.local.json")
    with pytest.raises(evaluate.ProductionFinalEvaluationError):
        evaluate._load_config_inputs(inputs.config_path, link)


def test_fixed_profile_pair_resolves_one_device_base_and_prebinds_both_closures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _config_inputs(tmp_path)
    calls: list[tuple[str, int, int]] = []
    events: list[str] = []

    def apply_profile(config: dict, name: str):
        calls.append((name, id(config), config["sherpa"]["asr_num_threads"]))
        candidate = name == "parakeet-faster-whisper"
        final = {
            "asr_final_backend": "nemo_transducer" if candidate else "sense_voice",
            "asr_final_model": f"{name}.onnx",
            "asr_final_tokens": f"{name}.txt",
            "asr_final_verifier_backend": "faster_whisper" if candidate else "",
            "asr_final_verifier_model": "faster-whisper" if candidate else "",
        }
        return (
            evaluate.deep_merge(config, {"sherpa": final}),
            SimpleNamespace(
                name=name,
                sha256=("b" if candidate else "a") * 64,
                schema_version=1,
            ),
        )

    def config_digest(config: SherpaConfig) -> str:
        profile = str(config.asr_final_backend)
        events.append(f"config:{profile}")
        return ("d" if profile == "nemo_transducer" else "c") * 64

    def model_digest(config: SherpaConfig, root: Path) -> str:
        assert root == tmp_path
        profile = str(config.asr_final_backend)
        events.append(f"model:{profile}")
        return ("f" if profile == "nemo_transducer" else "e") * 64

    monkeypatch.setattr(evaluate, "apply_final_stt_profile", apply_profile)
    monkeypatch.setattr(evaluate, "_config_digest", config_digest)
    monkeypatch.setattr(evaluate, "_configured_model_digest", model_digest)
    monkeypatch.setattr(evaluate, "_source_digest", lambda: "7" * 64)
    monkeypatch.setattr(
        evaluate,
        "_runtime_identities",
        lambda: {"safe_runtime": "bound"},
    )

    pair = evaluate._bind_final_stt_profile_pair(
        inputs,
        None,
        "sense-voice",
        "parakeet-faster-whisper",
    )

    assert calls == [
        ("sense-voice", calls[0][1], 1),
        ("parakeet-faster-whisper", calls[0][1], 1),
    ]
    assert events == [
        "config:sense_voice",
        "model:sense_voice",
        "config:nemo_transducer",
        "model:nemo_transducer",
    ]
    assert pair.baseline.config.asr_final_verifier_backend == ""
    assert pair.candidate.config.asr_final_verifier_backend == "faster_whisper"
    assert pair.device_profile == "desktop"
    assert pair.evaluator_source_sha256 == "7" * 64

    failure_events: list[str] = []

    def fail_candidate_model(config: SherpaConfig, _root: Path) -> str:
        profile = str(config.asr_final_backend)
        failure_events.append(profile)
        if profile == "nemo_transducer":
            raise evaluate.ProductionFinalEvaluationError()
        return "e" * 64

    monkeypatch.setattr(evaluate, "_configured_model_digest", fail_candidate_model)
    with pytest.raises(evaluate.ProductionFinalEvaluationError):
        evaluate._bind_final_stt_profile_pair(
            inputs,
            None,
            "sense-voice",
            "parakeet-faster-whisper",
        )
    assert failure_events == ["sense_voice", "nemo_transducer"]


@pytest.mark.parametrize(
    "profile_args",
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
            "parakeet-faster-whisper",
            "--candidate-final-stt-profile",
            "sense-voice",
        ],
    ],
)
def test_cli_rejects_non_atomic_profile_pair_before_input_work(
    profile_args: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        evaluate,
        "load_corpus",
        lambda _path: pytest.fail("invalid pair reached corpus loading"),
    )
    argv = [
        "--corpus",
        "/private/corpus.json",
        "--config",
        "/private/config.json",
        "--local-config",
        "/private/config.local.json",
        "--report",
        "/private/report.json",
        *profile_args,
    ]

    assert evaluate.main(argv) == 2
    assert json.loads(capsys.readouterr().out) == evaluate._SAFE_ERROR


def test_optional_real_prepared_corpus_passes_exact_validation() -> None:
    supplied = os.environ.get(_REAL_CORPUS_ENV, "").strip()
    if not supplied:
        pytest.skip("exact prepared public command/noise corpus is not provisioned")
    corpus_path = Path(supplied).expanduser()
    if not corpus_path.is_absolute() or not corpus_path.is_file():
        pytest.fail("explicit exact prepared corpus is unavailable")

    corpus = load_corpus(corpus_path)

    assert len(corpus.cases) == 57
    assert evaluate._validate_exact_corpus(corpus) == _LOCK_RAW_SHA256


def test_exact_shape_and_tampered_corpus_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _exact_shape_corpus(tmp_path)
    receipt_sha256 = _install_exact_shape_dependencies(
        monkeypatch,
        _receipt(corpus),
    )
    assert corpus.provenance is not None
    corpus = replace(
        corpus,
        provenance=replace(corpus.provenance, metadata_sha256=receipt_sha256),
    )
    assert evaluate._validate_exact_corpus(corpus) == _LOCK_RAW_SHA256

    first = corpus.cases[0]
    tampered = replace(
        corpus,
        cases=(
            replace(
                first,
                tags=("public-command-noise", "gsc", "command-negative"),
            ),
            *corpus.cases[1:],
        ),
    )
    with pytest.raises(evaluate.ProductionFinalEvaluationError):
        evaluate._validate_exact_corpus(tampered)


@pytest.mark.parametrize("tamper", ["counts", "source-shape", "extra-key"])
def test_exact_corpus_rejects_tampered_preparation_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    corpus = _exact_shape_corpus(tmp_path)
    receipt = _receipt(corpus)
    if tamper == "counts":
        receipt["counts"] = {**receipt["counts"], "cases": 56}  # type: ignore[arg-type]
    elif tamper == "source-shape":
        receipt["source_aggregates"] = [
            *receipt["source_aggregates"][:-1],  # type: ignore[index]
            {
                "source_id": "english-consistent-confusion-v1.2",
                "cases": 28,
                "selection_sha256": "2" * 64,
            },
        ]
    else:
        receipt["private_path"] = "/must/not/be/accepted"
    receipt_sha256 = _install_exact_shape_dependencies(monkeypatch, receipt)
    assert corpus.provenance is not None
    corpus = replace(
        corpus,
        provenance=replace(corpus.provenance, metadata_sha256=receipt_sha256),
    )

    with pytest.raises(evaluate.ProductionFinalEvaluationError):
        evaluate._validate_exact_corpus(corpus)


def test_attest_execution_accepts_completed_offline_and_verifier_attempts() -> None:
    config = SherpaConfig(
        asr_final_backend="sense_voice",
        asr_final_verifier_backend="faster_whisper",
    )

    evaluate._attest_execution(
        config,
        _execution(
            decisions=2,
            offline={"decoded": 1, "empty": 1},
            verifier={"consensus": 1, "empty": 1},
            selected={"offline": 1, "streaming": 1},
        ),
    )
    evaluate._attest_execution(
        SherpaConfig(
            asr_final_backend="sense_voice",
            asr_final_verifier_backend="",
        ),
        _execution(
            decisions=2,
            offline={"decoded": 1, "empty": 1},
            verifier={"unavailable": 2},
            selected={"offline": 1, "streaming": 1},
        ),
    )


@pytest.mark.parametrize(
    ("decisions", "offline", "verifier", "selected"),
    [
        (1, {"error": 1}, {"consensus": 1}, {"offline": 1}),
        (1, {"skipped": 1}, {"consensus": 1}, {"streaming": 1}),
        (1, {"decoded": 1}, {"error": 1}, {"offline": 1}),
        (1, {"decoded": 1}, {"skipped": 1}, {"offline": 1}),
        (1, {"unavailable": 1}, {"unavailable": 1}, {"streaming": 1}),
        (0, {}, {}, {}),
    ],
    ids=(
        "offline-error",
        "offline-skipped-with-zero-attempts",
        "verifier-error",
        "verifier-skipped-with-zero-attempts",
        "both-unavailable",
        "zero-decisions",
    ),
)
def test_attest_execution_rejects_errors_skips_and_zero_attempts(
    decisions: int,
    offline: dict[str, int],
    verifier: dict[str, int],
    selected: dict[str, int],
) -> None:
    config = SherpaConfig(
        asr_final_backend="sense_voice",
        asr_final_verifier_backend="faster_whisper",
    )

    with pytest.raises(evaluate.ProductionFinalEvaluationError):
        evaluate._attest_execution(
            config,
            _execution(
                decisions=decisions,
                offline=offline,
                verifier=verifier,
                selected=selected,
            ),
        )


def test_evaluation_rejects_covered_run_with_one_zero_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = (
        _case(
            "private-real-decision",
            assertion="transcript",
            expected_text="private first reference",
            commands=("private first reference",),
            tags=("private",),
        ),
        _case(
            "private-zero-decision",
            assertion="silence",
            expected_text="",
            tags=("private",),
        ),
    )
    corpus = _evaluation_corpus(tmp_path, *cases)
    records = (
        FinalDecisionRecord(0, 0, ("private actual decision",)),
        FinalDecisionRecord(1, 0, ()),
    )
    metrics = aggregate_final_metrics(corpus, records, repeats=1)
    assert metrics["coverage_complete"] is True
    assert metrics["decisions"]["zero_decision_evaluations"] == 1
    execution = evaluate._Execution(
        streaming=metrics,
        selected=metrics,
        decisions=1,
        offline_outcomes={"decoded": 1},
        verifier_outcomes={"consensus": 1},
        selected_sources={"offline": 1},
    )
    _install_evaluation_dependencies(monkeypatch, execution)
    inputs = evaluate._ConfigInputs(
        root=tmp_path,
        config_path=tmp_path / "private-config.json",
        local_path=tmp_path / "private-local.json",
        config_bytes=b"private credential bytes",
        local_bytes=b"private local bytes",
        digest="credential-derived-private-digest",
    )
    config = SherpaConfig(
        asr_encoder="private-model",
        provider="cpu",
        asr_final_backend="sense_voice",
        asr_final_verifier_backend="faster_whisper",
    )

    with pytest.raises(evaluate.ProductionFinalEvaluationError):
        evaluate.evaluate_production_final(
            corpus,
            inputs,
            config,
            repeats=1,
            stratum_tags=(),
            device_profile="desktop",
        )


def test_evaluation_accepts_one_decision_whose_selected_text_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _evaluation_corpus(
        tmp_path,
        _case(
            "private-empty-selected",
            assertion="speech_negative",
            expected_text="",
            tags=("private",),
        ),
    )
    one_empty_decision = (FinalDecisionRecord(0, 0, ("",)),)
    metrics = aggregate_final_metrics(corpus, one_empty_decision, repeats=1)
    assert metrics["coverage_complete"] is True
    assert metrics["decisions"] == {
        "total": 1,
        "zero_decision_evaluations": 0,
        "multi_decision_evaluations": 0,
    }
    execution = evaluate._Execution(
        streaming=metrics,
        selected=metrics,
        decisions=1,
        offline_outcomes={"empty": 1},
        verifier_outcomes={"empty": 1},
        selected_sources={"none": 1},
    )
    _install_evaluation_dependencies(monkeypatch, execution)
    inputs = evaluate._ConfigInputs(
        root=tmp_path,
        config_path=tmp_path / "private-config.json",
        local_path=tmp_path / "private-local.json",
        config_bytes=b"private credential bytes",
        local_bytes=b"private local bytes",
        digest="credential-derived-private-digest",
    )
    config = SherpaConfig(
        asr_encoder="private-model",
        provider="cpu",
        asr_final_backend="sense_voice",
        asr_final_verifier_backend="faster_whisper",
    )

    report = evaluate.evaluate_production_final(
        corpus,
        inputs,
        config,
        repeats=1,
        stratum_tags=(),
        device_profile="desktop",
    )

    assert report["decisions"] == 1
    assert report["views"][
        "file_replay_selected_without_owned_timing_recovery"
    ]["decisions"] == {
        "total": 1,
        "zero_decision_evaluations": 0,
        "multi_decision_evaluations": 0,
    }


def test_evaluation_revalidates_private_corpus_at_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _evaluation_corpus(
        tmp_path,
        _case(
            "private-close-check",
            assertion="transcript",
            expected_text="private reference",
            commands=("private reference",),
            tags=("private",),
        ),
    )
    metrics = aggregate_final_metrics(
        corpus,
        (FinalDecisionRecord(0, 0, ("private reference",)),),
        repeats=1,
    )
    execution = evaluate._Execution(
        streaming=metrics,
        selected=metrics,
        decisions=1,
        offline_outcomes={"decoded": 1},
        verifier_outcomes={"consensus": 1},
        selected_sources={"offline": 1},
    )
    _install_evaluation_dependencies(monkeypatch, execution)
    checks: list[LoadedCorpus] = []

    def fail_at_close(value: LoadedCorpus) -> None:
        checks.append(value)
        if len(checks) == 2:
            raise evaluate.ProductionFinalEvaluationError()

    monkeypatch.setattr(
        evaluate,
        "_verify_private_corpus_snapshot",
        fail_at_close,
    )
    inputs = evaluate._ConfigInputs(
        root=tmp_path,
        config_path=tmp_path / "private-config.json",
        local_path=tmp_path / "private-local.json",
        config_bytes=b"private config",
        local_bytes=b"private local config",
        digest="private-config-digest",
    )
    config = SherpaConfig(
        asr_encoder="private-model",
        provider="cpu",
        asr_final_backend="sense_voice",
        asr_final_verifier_backend="faster_whisper",
    )

    with pytest.raises(evaluate.ProductionFinalEvaluationError) as caught:
        evaluate.evaluate_production_final(
            corpus,
            inputs,
            config,
            repeats=1,
            stratum_tags=(),
            device_profile="desktop",
        )

    assert checks == [corpus, corpus]
    assert str(caught.value) == ""


def test_preflight_requires_no_tts_both_final_models_and_warms_verifier() -> None:
    warm_calls: list[bool] = []
    verifier = SimpleNamespace(warm=lambda: warm_calls.append(True))

    evaluate._preflight_engine(
        SimpleNamespace(
            _tts=None,
            _final_recognizer=object(),
            _final_verifier=verifier,
        )
    )

    assert warm_calls == [True]
    evaluate._preflight_engine(
        SimpleNamespace(
            _tts=None,
            _final_recognizer=object(),
            _final_verifier=None,
        ),
        verifier_required=False,
    )
    invalid = (
        SimpleNamespace(
            _tts=object(),
            _final_recognizer=object(),
            _final_verifier=verifier,
        ),
        SimpleNamespace(
            _tts=None,
            _final_recognizer=None,
            _final_verifier=verifier,
        ),
        SimpleNamespace(
            _tts=None,
            _final_recognizer=object(),
            _final_verifier=None,
        ),
        SimpleNamespace(
            _tts=None,
            _final_recognizer=object(),
            _final_verifier=object(),
        ),
    )
    for engine in invalid:
        with pytest.raises(evaluate.ProductionFinalEvaluationError):
            evaluate._preflight_engine(engine)


def test_preflight_warm_failure_is_detail_free() -> None:
    private = "SENTINEL private verifier model path and credential"

    def fail_warm() -> None:
        raise RuntimeError(private)

    engine = SimpleNamespace(
        _tts=None,
        _final_recognizer=object(),
        _final_verifier=SimpleNamespace(warm=fail_warm),
    )

    with pytest.raises(evaluate.ProductionFinalEvaluationError) as caught:
        evaluate._preflight_engine(engine)

    assert str(caught.value) == ""
    assert private not in repr(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("failure_stage", ["start", "warm"])
def test_execute_replay_stops_when_start_or_warm_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    private = f"SENTINEL private {failure_stage} failure"
    corpus = _evaluation_corpus(
        tmp_path,
        _case(
            "private-cleanup-case",
            assertion="transcript",
            expected_text="private reference",
            commands=("private reference",),
            tags=("private",),
        ),
    )
    instances: list[object] = []

    class FailingEngine:
        def __init__(self, _config: SherpaConfig, *, asr_only: bool) -> None:
            assert asr_only is True
            self._tts = None
            self._final_recognizer = object()
            self._final_verifier = SimpleNamespace(warm=self._warm)
            self.stop_calls = 0
            instances.append(self)

        def _warm(self) -> None:
            if failure_stage == "warm":
                raise RuntimeError(private)

        def start(self, callbacks: EngineCallbacks) -> None:
            assert callbacks == EngineCallbacks()
            if failure_stage == "start":
                raise RuntimeError(private)

        def stop(self) -> None:
            self.stop_calls += 1

    monkeypatch.setattr(evaluate, "FileReplayEngine", FailingEngine)
    monkeypatch.setattr(evaluate, "_quiet_model_output", nullcontext)

    with pytest.raises(evaluate.ProductionFinalEvaluationError) as caught:
        evaluate._execute_replay(
            corpus,
            SherpaConfig(asr_encoder="private-model"),
            root=tmp_path,
            repeats=1,
            stratum_tags=(),
        )

    [engine] = instances
    assert engine.stop_calls == 1
    assert str(caught.value) == ""
    assert private not in repr(caught.value)


def test_execute_replay_is_asr_only_callback_free_f32_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    samples = np.asarray([0.25, -0.5, 0.75], dtype="<f4")
    case = _case(
        "SENTINEL-private-replay-case",
        assertion="transcript",
        tags=("private",),
        expected_text="SENTINEL private reference",
        commands=("SENTINEL private reference",),
        samples=len(samples),
    )
    case = replace(case, audio_bytes=samples.tobytes())
    corpus = LoadedCorpus(
        path=tmp_path / "private-corpus.json",
        digest="a" * 64,
        schema_version=4,
        purpose="private",
        provenance=None,
        cases=(case,),
        audio_bytes=len(case.audio_bytes),
    )
    first = FinalTranscriptDecision(
        streaming_raw="SENTINEL private streaming",
        offline_raw="SENTINEL private offline",
        selected="SENTINEL private selected",
        offline_outcome="decoded",
        selected_source=FinalTranscriptSource.OFFLINE,
        verifier_raw="SENTINEL private verifier",
        verifier_outcome="consensus",
    )
    second = FinalTranscriptDecision(
        streaming_raw="SENTINEL private second streaming",
        offline_raw="",
        selected="SENTINEL private second selected",
        offline_outcome="empty",
        selected_source=FinalTranscriptSource.STREAMING,
        verifier_raw="",
        verifier_outcome="empty",
    )
    instances: list[object] = []

    class FakeReplayEngine:
        def __init__(self, config: SherpaConfig, *, asr_only: bool) -> None:
            self.config = config
            self.asr_only = asr_only
            self.started = False
            self.stopped = False
            self.calls = 0
            self.callbacks: EngineCallbacks | None = None
            self._tts = None
            self._final_recognizer = object()
            self._final_verifier = SimpleNamespace(warm=lambda: None)
            instances.append(self)

        def start(self, callbacks: EngineCallbacks) -> None:
            self.started = True
            self.callbacks = callbacks
            assert callbacks == EngineCallbacks()
            assert callbacks.on_final_result is None
            assert callbacks.on_partial_result is None
            assert callbacks.on_command_result is None

        def evaluate_samples(
            self,
            observed,
            sample_rate: int,
            *,
            speech_sec: float | None,
        ) -> SimpleNamespace:
            assert Path.cwd() == tmp_path
            assert observed.dtype == np.dtype("<f4")
            assert observed.flags.owndata
            assert observed.tolist() == pytest.approx(samples.tolist())
            assert sample_rate == 16_000
            assert speech_sec is None
            self.calls += 1
            return SimpleNamespace(decisions=(first, second))

        def stop(self) -> None:
            self.stopped = True

    monkeypatch.setattr(evaluate, "FileReplayEngine", FakeReplayEngine)
    monkeypatch.setattr(evaluate, "_quiet_model_output", nullcontext)
    config = SherpaConfig(
        asr_encoder="private-model",
        asr_final_verifier_backend="faster_whisper",
    )

    result = evaluate._execute_replay(
        corpus,
        config,
        root=tmp_path,
        repeats=2,
        stratum_tags=(),
    )

    [engine] = instances
    assert engine.config is config
    assert engine.asr_only is True
    assert engine.started is True
    assert engine.stopped is True
    assert engine.calls == 2
    assert result.decisions == 4
    assert result.offline_outcomes == {"decoded": 2, "empty": 2}
    assert result.verifier_outcomes == {"consensus": 2, "empty": 2}
    assert result.selected_sources == {"offline": 2, "streaming": 2}
    assert result.streaming["coverage_complete"] is True
    assert result.streaming["decisions"] == {
        "total": 4,
        "zero_decision_evaluations": 0,
        "multi_decision_evaluations": 2,
    }
    assert result.selected["coverage_complete"] is True
    assert "SENTINEL" not in repr(result)


@pytest.mark.parametrize("raise_during_evaluation", [False, True])
def test_real_quiet_model_output_suppresses_engine_output_and_restores_process_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    raise_during_evaluation: bool,
) -> None:
    private = "SENTINEL_PRIVATE_MODEL_OUTPUT"
    visible = "VISIBLE_AFTER_MODEL_OUTPUT_GUARD"
    corpus = _evaluation_corpus(
        tmp_path,
        _case(
            "private-output-case",
            assertion="transcript",
            expected_text="private reference",
            commands=("private reference",),
            tags=("private",),
        ),
    )
    decision = FinalTranscriptDecision(
        streaming_raw="private streaming",
        offline_raw="private offline",
        selected="private selected",
        offline_outcome="decoded",
        selected_source=FinalTranscriptSource.OFFLINE,
        verifier_raw="private verifier",
        verifier_outcome="consensus",
    )

    def emit_private(stage: str) -> None:
        logging.getLogger("speaker.tests.private-model-output").warning(
            "%s logging %s",
            private,
            stage,
        )
        print(f"{private} stdout {stage}", flush=True)
        print(f"{private} stderr {stage}", file=sys.stderr, flush=True)
        os.write(1, f"{private} fd1 {stage}\n".encode("ascii"))
        os.write(2, f"{private} fd2 {stage}\n".encode("ascii"))

    class NoisyReplayEngine:
        def __init__(self, _config: SherpaConfig, *, asr_only: bool) -> None:
            assert asr_only is True
            self._tts = None
            self._final_recognizer = object()
            self._final_verifier = SimpleNamespace(warm=self._warm)

        def _warm(self) -> None:
            emit_private("warm")

        def start(self, callbacks: EngineCallbacks) -> None:
            assert callbacks == EngineCallbacks()
            emit_private("start")

        def evaluate_samples(
            self,
            _samples,
            _sample_rate: int,
            *,
            speech_sec: float | None,
        ) -> SimpleNamespace:
            assert speech_sec is None
            emit_private("evaluate")
            if raise_during_evaluation:
                raise RuntimeError(private)
            return SimpleNamespace(decisions=(decision,))

        def stop(self) -> None:
            emit_private("stop")

    monkeypatch.setattr(evaluate, "FileReplayEngine", NoisyReplayEngine)
    caplog.set_level(logging.DEBUG)
    previous_disable = logging.root.manager.disable
    previous_stdout = sys.stdout
    previous_stderr = sys.stderr
    stdout_identity = (os.fstat(1).st_dev, os.fstat(1).st_ino)
    stderr_identity = (os.fstat(2).st_dev, os.fstat(2).st_ino)

    if raise_during_evaluation:
        with pytest.raises(evaluate.ProductionFinalEvaluationError) as caught:
            evaluate._execute_replay(
                corpus,
                SherpaConfig(
                    asr_encoder="private-model",
                    asr_final_verifier_backend="faster_whisper",
                ),
                root=tmp_path,
                repeats=1,
                stratum_tags=(),
            )
        assert str(caught.value) == ""
        assert private not in repr(caught.value)
    else:
        result = evaluate._execute_replay(
            corpus,
            SherpaConfig(
                asr_encoder="private-model",
                asr_final_verifier_backend="faster_whisper",
            ),
            root=tmp_path,
            repeats=1,
            stratum_tags=(),
        )
        assert result.decisions == 1

    assert logging.root.manager.disable == previous_disable
    assert sys.stdout is previous_stdout
    assert sys.stderr is previous_stderr
    assert (os.fstat(1).st_dev, os.fstat(1).st_ino) == stdout_identity
    assert (os.fstat(2).st_dev, os.fstat(2).st_ino) == stderr_identity

    print(f"{visible} stdout", flush=True)
    print(f"{visible} stderr", file=sys.stderr, flush=True)
    os.write(1, f"{visible} fd1\n".encode("ascii"))
    os.write(2, f"{visible} fd2\n".encode("ascii"))
    logging.getLogger("speaker.tests.private-model-output").warning(
        "%s logging",
        visible,
    )
    captured = capfd.readouterr()
    observed = captured.out + captured.err + caplog.text
    assert private not in observed
    assert f"{visible} stdout" in captured.out
    assert f"{visible} fd1" in captured.out
    assert f"{visible} stderr" in captured.err
    assert f"{visible} fd2" in captured.err
    assert f"{visible} logging" in caplog.text


def test_private_writer_is_mode_600_no_clobber_and_refuses_git_and_symlink(
    tmp_path: Path,
) -> None:
    report = tmp_path / "aggregate-report.json"
    payload = b'{"execution_complete":true}\n'

    evaluate._write_new_private(report, payload)

    assert report.read_bytes() == payload
    assert stat.S_IMODE(report.stat().st_mode) == 0o600
    with pytest.raises(evaluate.ProductionFinalEvaluationError):
        evaluate._write_new_private(report, b"different")
    assert report.read_bytes() == payload

    git_root = tmp_path / "recognized-git"
    git_root.mkdir()
    (git_root / ".git").mkdir()
    inside_git = git_root / "must-not-exist.json"
    with pytest.raises(evaluate.ProductionFinalEvaluationError):
        evaluate._write_new_private(inside_git, payload)
    assert not inside_git.exists()

    target = tmp_path / "symlink-target.json"
    target.write_bytes(b"target-must-stay")
    linked_report = tmp_path / "linked-report.json"
    linked_report.symlink_to(target)
    with pytest.raises(evaluate.ProductionFinalEvaluationError):
        evaluate._write_new_private(linked_report, payload)
    assert linked_report.is_symlink()
    assert target.read_bytes() == b"target-must-stay"


def test_private_writer_completes_forced_partial_writes_with_exact_hash_and_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "partial-write-report.json"
    payload = (b'{"aggregate":"safe"}' * 257) + b"\n"
    real_write = evaluate.os.write
    write_sizes: list[int] = []

    def partial_write(descriptor: int, data) -> int:
        requested = len(data)
        count = real_write(descriptor, data[: min(7, requested)])
        write_sizes.append(count)
        return count

    monkeypatch.setattr(evaluate.os, "write", partial_write)

    evaluate._write_new_private(report, payload)

    published = report.read_bytes()
    assert len(write_sizes) > 1
    assert max(write_sizes) <= 7
    assert published == payload
    assert hashlib.sha256(published).hexdigest() == hashlib.sha256(payload).hexdigest()
    assert stat.S_IMODE(report.stat().st_mode) == 0o600


def test_private_writer_rejects_hardlink_and_widened_parent(
    tmp_path: Path,
) -> None:
    payload = b'{"aggregate":"safe"}\n'
    target = tmp_path / "existing-target.json"
    target.write_bytes(b"must-stay-private")
    hardlink = tmp_path / "hardlinked-report.json"
    os.link(target, hardlink)

    with pytest.raises(evaluate.ProductionFinalEvaluationError):
        evaluate._write_new_private(hardlink, payload)
    assert target.read_bytes() == b"must-stay-private"
    assert hardlink.samefile(target)

    widened = tmp_path / "widened-parent"
    widened.mkdir(mode=0o700)
    widened.chmod(0o755)
    refused = widened / "must-not-exist.json"
    try:
        with pytest.raises(evaluate.ProductionFinalEvaluationError):
            evaluate._write_new_private(refused, payload)
        assert not refused.exists()
    finally:
        widened.chmod(0o700)


def test_private_writer_removes_publication_when_parent_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "parent-fsync-failure.json"
    payload = b'{"aggregate":"safe"}\n'
    real_fsync = evaluate.os.fsync
    failed = False

    def fail_published_parent_fsync(descriptor: int) -> None:
        nonlocal failed
        if (
            not failed
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
            and report.exists()
        ):
            failed = True
            raise OSError("SENTINEL private parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(evaluate.os, "fsync", fail_published_parent_fsync)

    with pytest.raises(evaluate.ProductionFinalEvaluationError) as caught:
        evaluate._write_new_private(report, payload)

    assert failed is True
    assert str(caught.value) == ""
    assert not report.exists()
    assert list(tmp_path.glob(".production-final-stt-*.tmp")) == []


def test_private_writer_removes_publication_when_final_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "post-link-validation-failure.json"
    payload = b'{"aggregate":"safe"}\n'
    real_stat = evaluate.os.stat
    candidate_stat_calls = 0

    def fail_one_post_link_validation(
        path,
        *,
        dir_fd=None,
        follow_symlinks=True,
    ):
        nonlocal candidate_stat_calls
        result = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if path == report.name and dir_fd is not None:
            candidate_stat_calls += 1
            if candidate_stat_calls == 1:
                values = list(result)
                values[3] = result.st_nlink + 1
                return os.stat_result(values)
        return result

    monkeypatch.setattr(evaluate.os, "stat", fail_one_post_link_validation)

    with pytest.raises(evaluate.ProductionFinalEvaluationError):
        evaluate._write_new_private(report, payload)

    assert candidate_stat_calls >= 1
    assert not report.exists()
    assert list(tmp_path.glob(".production-final-stt-*.tmp")) == []


def test_private_writer_never_unlinks_foreign_replacement_during_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "foreign-replacement-report.json"
    payload = b'{"aggregate":"owned-publication"}\n'
    foreign = tmp_path / "foreign-source.json"
    foreign_payload = b"SENTINEL foreign inode must survive"
    foreign.write_bytes(foreign_payload)
    foreign.chmod(0o600)
    foreign_identity = (foreign.stat().st_dev, foreign.stat().st_ino)
    real_fsync = evaluate.os.fsync
    replaced = False

    def replace_before_parent_fsync(descriptor: int) -> None:
        nonlocal replaced
        if (
            not replaced
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
            and report.exists()
        ):
            os.replace(foreign, report)
            replaced = True
            raise OSError("SENTINEL private post-replacement failure")
        real_fsync(descriptor)

    monkeypatch.setattr(evaluate.os, "fsync", replace_before_parent_fsync)

    with pytest.raises(evaluate.ProductionFinalEvaluationError):
        evaluate._write_new_private(report, payload)

    assert replaced is True
    assert report.read_bytes() == foreign_payload
    assert (report.stat().st_dev, report.stat().st_ino) == foreign_identity
    assert list(tmp_path.glob(".production-final-stt-*.tmp")) == []


@pytest.mark.parametrize("widened", ["manifest", "pcm"])
def test_private_corpus_rejects_widened_manifest_and_pcm_modes(
    tmp_path: Path,
    widened: str,
) -> None:
    root = tmp_path / "private-corpus"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    manifest = root / "corpus.json"
    receipt = root / "preparation-receipt.json"
    pcm = root / "case.f32le"
    for path, payload in (
        (manifest, b"{}\n"),
        (receipt, b"{}\n"),
        (pcm, struct.pack("<f", 0.25)),
    ):
        path.write_bytes(payload)
        path.chmod(0o600)
    case = replace(
        _case(
            "private-permission-case",
            assertion="transcript",
            expected_text="private reference",
            commands=("private reference",),
            tags=("private",),
        ),
        source_path=pcm,
        audio_bytes=pcm.read_bytes(),
    )
    corpus = LoadedCorpus(
        path=manifest,
        digest="a" * 64,
        schema_version=4,
        purpose="private",
        provenance=None,
        cases=(case,),
        audio_bytes=len(case.audio_bytes),
    )
    evaluate._require_private_corpus(corpus)

    changed = manifest if widened == "manifest" else pcm
    changed.chmod(0o640)

    with pytest.raises(evaluate.ProductionFinalEvaluationError):
        evaluate._require_private_corpus(corpus)


def test_private_corpus_rejects_hardlinked_pcm(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private-hardlink-corpus"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    manifest = root / "corpus.json"
    receipt = root / "preparation-receipt.json"
    pcm = root / "case.f32le"
    for path in (manifest, receipt, pcm):
        path.write_bytes(b"private")
        path.chmod(0o600)
    duplicate = root / "duplicate.f32le"
    os.link(pcm, duplicate)
    case = replace(
        _case(
            "private-hardlink-case",
            assertion="transcript",
            commands=("private",),
            tags=("private",),
        ),
        source_path=pcm,
        audio_bytes=pcm.read_bytes(),
    )
    corpus = LoadedCorpus(
        path=manifest,
        digest="a" * 64,
        schema_version=4,
        purpose="private",
        provenance=None,
        cases=(case,),
        audio_bytes=len(case.audio_bytes),
    )

    with pytest.raises(evaluate.ProductionFinalEvaluationError):
        evaluate._require_private_corpus(corpus)


def test_runtime_identities_have_exact_safe_keys_versions_and_record_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python_payload = b"exact private python executable bytes"
    versions = {
        distribution: f"1.2.{index}"
        for index, (_label, distribution) in enumerate(
            evaluate._RUNTIME_DISTRIBUTIONS,
            start=1,
        )
    }
    records = {
        distribution: f"{distribution}/module.py,sha256=SENTINEL-{index},12\n"
        for index, (_label, distribution) in enumerate(
            evaluate._RUNTIME_DISTRIBUTIONS,
            start=1,
        )
    }
    requested: list[str] = []

    class Distribution:
        def __init__(self, name: str) -> None:
            self.name = name
            self.version = versions[name]

        def read_text(self, filename: str) -> str:
            assert filename == "RECORD"
            return records[self.name]

    def distribution(name: str) -> Distribution:
        requested.append(name)
        return Distribution(name)

    monkeypatch.setattr(evaluate.importlib_metadata, "distribution", distribution)
    monkeypatch.setattr(
        evaluate,
        "read_regular_bounded",
        lambda _path, **_kwargs: SimpleNamespace(data=python_payload),
    )

    identities = evaluate._runtime_identities()

    assert set(identities) == {"python", "distributions"}
    assert set(identities["python"]) == {"version", "executable_sha256"}
    assert identities["python"] == {
        "version": (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "executable_sha256": hashlib.sha256(python_payload).hexdigest(),
    }
    expected_labels = [label for label, _name in evaluate._RUNTIME_DISTRIBUTIONS]
    expected_names = [name for _label, name in evaluate._RUNTIME_DISTRIBUTIONS]
    assert list(identities["distributions"]) == expected_labels
    assert requested == expected_names
    for label, distribution_name in evaluate._RUNTIME_DISTRIBUTIONS:
        assert set(identities["distributions"][label]) == {
            "version",
            "record_sha256",
        }
        assert identities["distributions"][label] == {
            "version": versions[distribution_name],
            "record_sha256": hashlib.sha256(
                records[distribution_name].encode("utf-8")
            ).hexdigest(),
        }
    assert "SENTINEL" not in json.dumps(identities, sort_keys=True)


def test_provider_attestation_is_exact_and_rejects_non_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        ),
    )

    assert evaluate._provider_attestation(
        SherpaConfig(
            provider="cpu",
            asr_final_verifier_backend="faster_whisper",
        )
    ) == {
        "streaming_offline_requested": "cpu",
        "streaming_offline_attestation": "explicit_cpu_provider_available",
        "verifier_requested": "cuda:0-fp16",
        "verifier_attestation": "cuda_only_adapter_completed_decode",
    }
    assert evaluate._provider_attestation(SherpaConfig(provider="cpu")) == {
        "streaming_offline_requested": "cpu",
        "streaming_offline_attestation": "explicit_cpu_provider_available",
        "verifier_requested": "disabled",
        "verifier_attestation": "not_configured",
    }
    with pytest.raises(evaluate.ProductionFinalEvaluationError):
        evaluate._provider_attestation(SherpaConfig(provider="cuda"))


def test_sense_voice_report_attests_verifier_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _evaluation_corpus(
        tmp_path,
        _case(
            "private-sense-case",
            assertion="transcript",
            expected_text="private reference",
            commands=("private reference",),
            tags=("private",),
        ),
    )
    aggregate = aggregate_final_metrics(
        corpus,
        (FinalDecisionRecord(0, 0, ("private hypothesis",)),),
        repeats=1,
    )
    execution = evaluate._Execution(
        streaming=aggregate,
        selected=aggregate,
        decisions=1,
        offline_outcomes={"decoded": 1},
        verifier_outcomes={"unavailable": 1},
        selected_sources={"offline": 1},
    )
    _install_evaluation_dependencies(monkeypatch, execution)
    disabled = {
        "streaming_offline_requested": "cpu",
        "streaming_offline_attestation": "explicit_cpu_provider_available",
        "verifier_requested": "disabled",
        "verifier_attestation": "not_configured",
    }
    monkeypatch.setattr(evaluate, "_provider_attestation", lambda _config: disabled)
    config = SherpaConfig(
        asr_encoder="private-streaming-model",
        provider="cpu",
        asr_final_backend="sense_voice",
        asr_final_verifier_backend="",
    )
    inputs = evaluate._ConfigInputs(
        root=tmp_path,
        config_path=tmp_path / "private-config.json",
        local_path=tmp_path / "private-local.json",
        config_bytes=b"private config",
        local_bytes=b"private local",
        digest="private digest",
    )

    report = evaluate.evaluate_production_final(
        corpus,
        inputs,
        config,
        repeats=1,
        stratum_tags=(),
        device_profile="desktop",
    )

    assert report["engine"]["provider_attestation"] == disabled
    assert report["engine"]["verifier_execution"] == {
        "requested": False,
        "outcomes": {"unavailable": 1},
    }


def test_report_is_aggregate_only_and_excludes_text_and_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_case_id = "SENTINEL_PRIVATE_CASE_ID"
    private_reference = "SENTINEL_PRIVATE_REFERENCE"
    private_hypothesis = "SENTINEL_PRIVATE_HYPOTHESIS"
    private_command = "SENTINEL_PRIVATE_COMMAND"
    case = _case(
        private_case_id,
        assertion="transcript",
        expected_text=private_reference,
        commands=(private_command,),
        forbidden_commands=("SENTINEL_PRIVATE_FORBIDDEN",),
        tags=("private-tag",),
    )
    corpus = LoadedCorpus(
        path=tmp_path / "SENTINEL_PRIVATE_CORPUS_PATH.json",
        digest="3" * 64,
        schema_version=4,
        purpose="SENTINEL_PRIVATE_PURPOSE",
        provenance=None,
        cases=(case,),
        audio_bytes=len(case.audio_bytes),
    )
    config_inputs = evaluate._ConfigInputs(
        root=tmp_path,
        config_path=tmp_path / "SENTINEL_PRIVATE_CONFIG_PATH.json",
        local_path=tmp_path / "SENTINEL_PRIVATE_LOCAL_PATH.json",
        config_bytes=b"SENTINEL_PRIVATE_CONFIG_BYTES credential=DO_NOT_EXPORT",
        local_bytes=b"SENTINEL_PRIVATE_LOCAL_BYTES",
        digest="CREDENTIAL_DERIVED_SENTINEL_DIGEST_DO_NOT_EXPORT",
    )
    config = SherpaConfig(
        asr_encoder=str(tmp_path / "SENTINEL_PRIVATE_MODEL_PATH.onnx"),
        provider="cpu",
        asr_final_backend="sense_voice",
        asr_final_verifier_backend="faster_whisper",
        asr_final_verifier_model=str(tmp_path / "SENTINEL_PRIVATE_VERIFIER"),
    )
    aggregate = aggregate_final_metrics(
        corpus,
        (FinalDecisionRecord(0, 0, (private_hypothesis,)),),
        repeats=1,
    )
    execution = evaluate._Execution(
        streaming=aggregate,
        selected=aggregate,
        decisions=1,
        offline_outcomes={"decoded": 1},
        verifier_outcomes={"consensus": 1},
        selected_sources={"offline": 1},
    )
    runtime, provider = _install_evaluation_dependencies(monkeypatch, execution)

    report = evaluate.evaluate_production_final(
        corpus,
        config_inputs,
        config,
        repeats=1,
        stratum_tags=(),
        device_profile="desktop-production",
    )
    encoded = json.dumps(report, sort_keys=True)

    assert report["execution_complete"] is True
    assert report["schema_version"] == 2
    assert report["quality_verdict"] == "diagnostic_only"
    assert report["engine"]["offline_execution"] == {
        "requested": True,
        "outcomes": {"decoded": 1},
    }
    assert report["engine"]["verifier_execution"] == {
        "requested": True,
        "outcomes": {"consensus": 1},
    }
    assert report["engine"]["selection_inputs"] == {
        "vad_owned_speech_duration": False,
        "offline_recovery_authorization": False,
    }
    assert report["engine"]["runtime"] == runtime
    assert report["engine"]["requested_provider"] == "cpu"
    assert report["engine"]["provider_attestation"] == provider
    assert report["engine"]["transport"] == {
        "block_ms": 100,
        "synthetic_tail_ms": 600,
    }
    assert report["engine"]["device_profile"] == "desktop-production"
    assert report["limits"]["chatbot_tools_tts"] is False
    assert report["limits"]["audio_device"] is False
    assert report["limits"]["selection_has_vad_owned_speech_duration"] is False
    assert (
        report["limits"]["selection_has_offline_recovery_authorization"]
        is False
    )
    assert report["limits"]["native_execution_watchdog"] is False
    assert set(report["views"]) == {
        "file_replay_streaming_raw",
        "file_replay_selected_without_owned_timing_recovery",
    }
    assert str(tmp_path) not in encoded
    for private in (
        private_case_id,
        private_reference,
        private_hypothesis,
        private_command,
        "SENTINEL_PRIVATE_FORBIDDEN",
        "SENTINEL_PRIVATE_PURPOSE",
        "SENTINEL_PRIVATE_CONFIG_BYTES",
        "SENTINEL_PRIVATE_LOCAL_BYTES",
        "SENTINEL_PRIVATE_MODEL_PATH",
        "CREDENTIAL_DERIVED_SENTINEL_DIGEST_DO_NOT_EXPORT",
        "DO_NOT_EXPORT",
    ):
        assert private not in encoded


def test_profile_pair_report_is_schema3_sequential_and_aggregate_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = object()
    inputs = object()
    runtime = {"safe_runtime": "bound"}
    pair = evaluate._BoundProfilePair(
        baseline=evaluate._BoundProfile(
            name="sense-voice",
            profile_sha256="a" * 64,
            profile_schema_version=1,
            config=SimpleNamespace(
                asr_final_backend="sense_voice",
                private="SENTINEL_PRIVATE_BASELINE_PATH",
            ),
            config_sha256="c" * 64,
            model_sha256="e" * 64,
        ),
        candidate=evaluate._BoundProfile(
            name="parakeet-faster-whisper",
            profile_sha256="b" * 64,
            profile_schema_version=1,
            config=SimpleNamespace(
                asr_final_backend="nemo_transducer",
                private="SENTINEL_PRIVATE_CANDIDATE_PATH",
            ),
            config_sha256="d" * 64,
            model_sha256="f" * 64,
        ),
        device_profile="desktop",
        evaluator_source_sha256="7" * 64,
        runtime_identities=runtime,
    )
    events: list[str] = []

    def run_cell(observed_corpus, observed_inputs, config, **kwargs):
        assert observed_corpus is corpus
        assert observed_inputs is inputs
        profile = str(config.asr_final_backend)
        events.append(profile)
        suffixes = ("c", "e") if profile == "sense_voice" else ("d", "f")
        assert kwargs["expected_config_sha256"] == suffixes[0] * 64
        assert kwargs["expected_model_sha256"] == suffixes[1] * 64
        assert kwargs["expected_evaluator_source_sha256"] == "7" * 64
        assert kwargs["expected_runtime_identities"] is runtime
        return {
            "corpus": {"manifest_sha256": "8" * 64},
            "engine": {
                "evaluator_source_sha256": "7" * 64,
                "runtime": runtime,
            },
            "evaluations": 57,
            "repeats": 1,
            "stratum_tags": ["noisy"],
            "views": {
                "file_replay_streaming_raw": {"coverage_complete": True},
                "file_replay_selected_without_owned_timing_recovery": {
                    "coverage_complete": True,
                    "profile": profile,
                },
            },
            "limits": {},
        }

    monkeypatch.setattr(evaluate, "evaluate_production_final", run_cell)

    report = evaluate.evaluate_production_final_pair(
        corpus,
        inputs,
        pair,
        repeats=1,
        stratum_tags=("noisy",),
    )
    encoded = json.dumps(report, sort_keys=True)

    assert events == ["sense_voice", "nemo_transducer"]
    assert report["schema_version"] == 3
    assert report["streaming_control_match"] is True
    assert report["comparison"] == {
        "same_input": True,
        "streaming_control_match": True,
        "valid": True,
        "verdict": "comparable",
        "latency_comparison": False,
        "runtime_default_promotion": False,
    }
    assert report["total_evaluations"] == 114
    assert report["execution_order"] == ["baseline", "candidate"]
    assert report["baseline_final_stt_profile"] == "sense-voice"
    assert report["candidate_final_stt_profile"] == "parakeet-faster-whisper"
    assert "promotable" not in encoded
    assert "winner" not in encoded
    assert "SENTINEL" not in encoded


def test_profile_pair_streaming_mismatch_is_explicitly_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = evaluate._BoundProfilePair(
        baseline=evaluate._BoundProfile(
            name="sense-voice",
            profile_sha256="a" * 64,
            profile_schema_version=1,
            config=SimpleNamespace(asr_final_backend="sense_voice"),
            config_sha256="c" * 64,
            model_sha256="e" * 64,
        ),
        candidate=evaluate._BoundProfile(
            name="parakeet-faster-whisper",
            profile_sha256="b" * 64,
            profile_schema_version=1,
            config=SimpleNamespace(asr_final_backend="nemo_transducer"),
            config_sha256="d" * 64,
            model_sha256="f" * 64,
        ),
        device_profile="desktop",
        evaluator_source_sha256="7" * 64,
        runtime_identities={"safe_runtime": "bound"},
    )

    def run_cell(_corpus, _inputs, config, **_kwargs):
        baseline = config.asr_final_backend == "sense_voice"
        return {
            "evaluations": 57,
            "repeats": 1,
            "stratum_tags": ["noisy"],
            "views": {
                "file_replay_streaming_raw": {
                    "coverage_complete": True,
                    "command_hits": 1 if baseline else 2,
                },
                "file_replay_selected_without_owned_timing_recovery": {
                    "coverage_complete": True,
                },
            },
        }

    monkeypatch.setattr(evaluate, "evaluate_production_final", run_cell)

    report = evaluate.evaluate_production_final_pair(
        object(),
        object(),
        pair,
        repeats=1,
        stratum_tags=("noisy",),
    )

    assert report["quality_verdict"] == "diagnostic_only"
    assert report["streaming_control_match"] is False
    assert report["comparison"] == {
        "same_input": True,
        "streaming_control_match": False,
        "valid": False,
        "verdict": "inconclusive_streaming_control_mismatch",
        "latency_comparison": False,
        "runtime_default_promotion": False,
    }
    assert report["limits"]["runtime_default_promotion"] is False


def test_cli_pair_receipt_exposes_inconclusive_comparison(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    complete_views = {
        "file_replay_streaming_raw": {"coverage_complete": True},
        "file_replay_selected_without_owned_timing_recovery": {
            "coverage_complete": True,
        },
    }
    fake_report = {
        "schema_version": 3,
        "baseline": {"views": complete_views},
        "candidate": {"views": complete_views},
        "total_evaluations": 114,
        "comparison": {
            "valid": False,
            "verdict": "inconclusive_streaming_control_mismatch",
        },
    }
    published: list[bytes] = []
    monkeypatch.setattr(evaluate, "load_corpus", lambda _path: object())
    monkeypatch.setattr(evaluate, "_load_config_inputs", lambda *_args: object())
    monkeypatch.setattr(evaluate, "_bind_final_stt_profile_pair", lambda *_args: object())
    monkeypatch.setattr(
        evaluate,
        "evaluate_production_final_pair",
        lambda *_args, **_kwargs: fake_report,
    )
    monkeypatch.setattr(
        evaluate,
        "_write_new_private",
        lambda _path, payload: published.append(payload),
    )

    code = evaluate.main(
        [
            "--corpus",
            "/private/corpus.json",
            "--config",
            "/private/config.json",
            "--local-config",
            "/private/config.local.json",
            "--report",
            "/private/report.json",
            "--baseline-final-stt-profile",
            "sense-voice",
            "--candidate-final-stt-profile",
            "parakeet-faster-whisper",
        ]
    )

    receipt = json.loads(capsys.readouterr().out)
    assert code == 0
    assert len(published) == 1
    assert receipt["ok"] is True
    assert receipt["comparison_valid"] is False
    assert (
        receipt["comparison_verdict"]
        == "inconclusive_streaming_control_mismatch"
    )
    assert receipt["quality_verdict"] == "diagnostic_only"


def test_cli_pair_candidate_failure_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(evaluate, "load_corpus", lambda _path: object())
    monkeypatch.setattr(evaluate, "_load_config_inputs", lambda *_args: object())
    monkeypatch.setattr(evaluate, "_bind_final_stt_profile_pair", lambda *_args: object())
    monkeypatch.setattr(
        evaluate,
        "evaluate_production_final_pair",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            evaluate.ProductionFinalEvaluationError()
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "_write_new_private",
        lambda *_args: pytest.fail("failed pair reached publication"),
    )

    code = evaluate.main(
        [
            "--corpus",
            "/private/SENTINEL-corpus.json",
            "--config",
            "/private/SENTINEL-config.json",
            "--local-config",
            "/private/SENTINEL-local.json",
            "--report",
            "/private/SENTINEL-report.json",
            "--baseline-final-stt-profile",
            "sense-voice",
            "--candidate-final-stt-profile",
            "parakeet-faster-whisper",
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert json.loads(captured.out) == evaluate._SAFE_ERROR
    assert captured.err == ""
    assert "SENTINEL" not in captured.out


def test_cli_legacy_receipt_and_schema2_payload_shape_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_report = {
        "schema_version": 2,
        "evaluations": 57,
        "views": {
            "file_replay_streaming_raw": {"coverage_complete": True},
            "file_replay_selected_without_owned_timing_recovery": {
                "coverage_complete": True,
            },
        },
    }
    published: list[bytes] = []
    monkeypatch.setattr(evaluate, "load_corpus", lambda _path: object())
    monkeypatch.setattr(evaluate, "_load_config_inputs", lambda *_args: object())
    monkeypatch.setattr(
        evaluate,
        "_effective_config_with_profile",
        lambda *_args: (object(), "desktop"),
    )
    monkeypatch.setattr(
        evaluate,
        "evaluate_production_final",
        lambda *_args, **_kwargs: fake_report,
    )
    monkeypatch.setattr(
        evaluate,
        "_write_new_private",
        lambda _path, payload: published.append(payload),
    )

    code = evaluate.main(
        [
            "--corpus",
            "/private/corpus.json",
            "--config",
            "/private/config.json",
            "--local-config",
            "/private/config.local.json",
            "--report",
            "/private/report.json",
        ]
    )

    receipt = json.loads(capsys.readouterr().out)
    assert code == 0
    assert set(receipt) == {
        "coverage_complete",
        "evaluations",
        "execution_complete",
        "ok",
        "quality_verdict",
        "report_sha256",
    }
    assert receipt["coverage_complete"] is True
    assert receipt["evaluations"] == 57
    assert json.loads(published[0]) == fake_report


def test_cli_refuses_to_overwrite_existing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    existing = tmp_path / "existing-report.json"
    original = b"SENTINEL existing private report"
    existing.write_bytes(original)
    fake_report = {
        "evaluations": 1,
        "views": {
            "file_replay_streaming_raw": {"coverage_complete": True},
            "file_replay_selected_without_owned_timing_recovery": {
                "coverage_complete": True
            },
        },
    }
    monkeypatch.setattr(evaluate, "load_corpus", lambda _path: object())
    monkeypatch.setattr(evaluate, "_load_config_inputs", lambda *_args: object())
    monkeypatch.setattr(
        evaluate,
        "_effective_config_with_profile",
        lambda *_args: (object(), "desktop"),
    )
    monkeypatch.setattr(
        evaluate,
        "evaluate_production_final",
        lambda *_args, **_kwargs: fake_report,
    )

    code = evaluate.main(
        [
            "--corpus",
            str(tmp_path / "SENTINEL-private-corpus.json"),
            "--config",
            str(tmp_path / "SENTINEL-private-config.json"),
            "--local-config",
            str(tmp_path / "SENTINEL-private-local.json"),
            "--report",
            str(existing),
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert json.loads(captured.out) == evaluate._SAFE_ERROR
    assert captured.err == ""
    assert "SENTINEL" not in captured.out
    assert existing.read_bytes() == original
