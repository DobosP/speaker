from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_public_conversation_fixture import _publish_source
from tests.test_public_conversation_qualification import _complete_metrics
from tests.test_streaming_stt_suite import _aggregate_report
from tools import edacc_stt_comparison as comparison
from tools import public_conversation_fixture as fixture
from tools import public_conversation_qualification as qualification
from tools import streaming_stt_eval, streaming_stt_suite
from tools.streaming_stt.corpus import load_corpus
from tools.streaming_stt.manifest import (
    FASTER_WHISPER_ENDPOINT_ADAPTER,
    SHERPA_ZIPFORMER_ADAPTER,
    BoundArtifact,
    BoundFile,
    FasterWhisperEndpointConfig,
    SherpaZipformerConfig,
)


_TAGS = ("clean", "disfluent", "overlap_adjacent")
_COUNTS = (8, 8, 8)
_BASELINE_ID = "production-gigaspeech-zipformer-fp32-benchmark-1t"
_CANDIDATE_ID = "faster-whisper-small-local"


def _sha(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class _ManifestDouble:
    digest: str
    schema_version: int
    model_id: str
    adapter: str
    python: BoundFile
    worker: BoundFile
    artifacts: tuple[BoundArtifact, ...]
    adapter_config: SherpaZipformerConfig | FasterWhisperEndpointConfig

    @property
    def artifact_by_name(self) -> dict[str, BoundArtifact]:
        return {artifact.name: artifact for artifact in self.artifacts}


def _bound_file(root: Path, label: str) -> BoundFile:
    return BoundFile(
        path=root / label,
        sha256=_sha(f"bound:{label}"),
        size_bytes=1,
    )


def _bound_artifacts(root: Path, *, candidate: bool) -> tuple[BoundArtifact, ...]:
    names = (
        ("runtime-receipt", "model-receipt", "venv-marker")
        if candidate
        else ("model-encoder", "model-decoder", "model-joiner", "model-tokens")
    )
    return tuple(
        BoundArtifact(
            name=name,
            path=root / name,
            sha256=_sha(f"artifact:{name}"),
            size_bytes=1,
        )
        for name in names
    )


def _manifest_double(
    path: Path | str,
    *,
    root: Path,
    state: dict[str, Any],
) -> _ManifestDouble:
    selected = Path(path)
    candidate = "candidate" in selected.name
    role = "candidate" if candidate else "baseline"
    adapter = state[f"{role}_adapter"]
    schema_version = state[f"{role}_schema"]
    model_id = state[f"{role}_model_id"]
    language = state["candidate_language"] if candidate else "en"
    adapter_config = (
        FasterWhisperEndpointConfig(language=language)
        if candidate
        else SherpaZipformerConfig()
    )
    digest_label = state.get(f"{role}_digest", f"manifest:{role}")
    if state.get("same_digest"):
        digest_label = "same-manifest"
    role_root = root / role
    fixed_worker = comparison._FIXED_WORKER.resolve(strict=True)
    return _ManifestDouble(
        digest=_sha(str(digest_label)),
        schema_version=schema_version,
        model_id=model_id,
        adapter=adapter,
        python=_bound_file(role_root, "python"),
        worker=BoundFile(
            path=fixed_worker,
            sha256=comparison._worker_source_sha256(),
            size_bytes=fixed_worker.lstat().st_size,
        ),
        artifacts=_bound_artifacts(role_root, candidate=candidate),
        adapter_config=adapter_config,
    )


@pytest.fixture
def worker_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {
        "baseline_adapter": SHERPA_ZIPFORMER_ADAPTER,
        "baseline_schema": 4,
        "baseline_model_id": _BASELINE_ID,
        "candidate_adapter": FASTER_WHISPER_ENDPOINT_ADAPTER,
        "candidate_schema": 6,
        "candidate_model_id": _CANDIDATE_ID,
        "candidate_language": "en",
        "same_digest": False,
    }

    def load(path: Path | str) -> _ManifestDouble:
        return _manifest_double(path, root=tmp_path / "worker-evidence", state=state)

    monkeypatch.setattr(comparison, "load_worker_manifest", load)
    # The comparison intentionally reuses these qualification helpers. Patch
    # their module-global loader as well so generated doubles never touch a
    # runtime, model, receipt tree, GPU, or worker process.
    monkeypatch.setattr(qualification, "load_worker_manifest", load)
    return state


@dataclass(frozen=True)
class _SyntheticEdacc:
    corpus_path: Path
    receipt_path: Path
    corpus_sha256: str
    receipt_sha256: str


def _synthetic_edacc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _SyntheticEdacc:
    lock = fixture.load_fixture_lock()
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir(mode=0o700)
    corpus_path = _publish_source(fixture_root, lock, "edacc-test")
    binding = fixture.validate_private_source(
        corpus_path,
        lock=lock,
        expected_source_id="edacc-test",
    )
    monkeypatch.setattr(
        comparison,
        "EXPECTED_CORPUS_SHA256",
        binding.corpus_manifest_sha256,
    )
    monkeypatch.setattr(
        comparison,
        "EXPECTED_RECEIPT_SHA256",
        binding.receipt_sha256,
    )
    monkeypatch.setattr(
        comparison,
        "EXPECTED_PCM_BYTES",
        binding.pcm_bytes,
    )
    return _SyntheticEdacc(
        corpus_path=corpus_path,
        receipt_path=binding.receipt_path,
        corpus_sha256=binding.corpus_manifest_sha256,
        receipt_sha256=binding.receipt_sha256,
    )


def _raw_report(
    worker: Path,
    corpus: Path,
    *,
    manifest: _ManifestDouble,
    repeats: int = 3,
) -> dict[str, object]:
    selected_stream = streaming_stt_eval._selected_stream(
        manifest,  # type: ignore[arg-type]
        None,
        chunk_samples=1600,
        pace="burst",
        partial_interval_ms=200,
        tail_padding_samples=0,
    )
    report = _aggregate_report(
        worker,
        corpus,
        worker_digest=manifest.digest,
        corpus_digest=load_corpus(corpus).digest,
        stratum_tags=_TAGS,
    )
    report["metrics"] = _complete_metrics(
        (_TAGS, _COUNTS),
        adapter=manifest.adapter,
        repeats=repeats,
    )
    report["evidence"] = streaming_stt_eval._evidence_binding(
        manifest.adapter,
        pace=selected_stream.pace,
        adapter_config=manifest.adapter_config,
        stratum_tags=_TAGS,
    )
    config: dict[str, object] = {
        **selected_stream.as_dict(),
        "repeats": repeats,
        "stratum_tags": list(_TAGS),
    }
    config["contract_sha256"] = qualification._canonical_sha256(config)
    report["config"] = config
    report["evaluator"] = streaming_stt_eval._evaluator_binding(manifest.adapter)
    return report


def _install_fake_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    worker_state: dict[str, Any],
    *,
    calls: list[tuple[str, str, dict[str, object]]] | None = None,
    fail_call: int | None = None,
) -> None:
    call_count = 0

    def benchmark(worker: Path, corpus: Path, **kwargs: object) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        if calls is not None:
            calls.append((worker.name, corpus.name, dict(kwargs)))
        if call_count == fail_call:
            raise RuntimeError("PRIVATE_SECOND_CELL_FAILURE")
        manifest = _manifest_double(
            worker,
            root=tmp_path / "worker-evidence",
            state=worker_state,
        )
        return _raw_report(worker, corpus, manifest=manifest)

    monkeypatch.setattr(streaming_stt_eval, "run_benchmark", benchmark)


def _run(
    tmp_path: Path,
    source: _SyntheticEdacc,
) -> dict[str, object]:
    return comparison.run_comparison(
        baseline_worker_manifest=tmp_path / "baseline.json",
        candidate_worker_manifest=tmp_path / "candidate.json",
        corpus_path=source.corpus_path,
        scratch_parent=tmp_path / "scratch",
        output_path=tmp_path / "report.json",
    )


def test_fixed_contract_pins_are_exact_and_not_publicly_overridable() -> None:
    assert comparison.EXPECTED_SOURCE_ID == "edacc-test"
    assert comparison.EXPECTED_LOCK_RAW_SHA256 == (
        "68ac2c85b8825fabda10b578646780fc16a37a1b2ab0ca3285fa9f8c93a3168b"
    )
    assert comparison.EXPECTED_CORPUS_SHA256 == (
        "4da392c39a0b6bd18057f63c96b4f67c0dfdaf4c14e1d2d76176cdbee0920772"
    )
    assert comparison.EXPECTED_RECEIPT_SHA256 == (
        "3c516b6fbb8ce139498cd2e01d83a4faf825e58bf2b9abcaa52896bdab4c853f"
    )
    assert comparison.EXPECTED_PCM_BYTES == 4_790_400
    assert comparison.STRATUM_TAGS == _TAGS
    assert comparison.STRATUM_COUNTS == _COUNTS
    assert comparison.FIXED_REPEATS == 3
    assert comparison.FIXED_CHUNK_SAMPLES == 1600
    assert comparison.FIXED_PACE == "burst"
    assert comparison.FIXED_PARTIAL_INTERVAL_MS == 200
    assert comparison.FIXED_TAIL_PADDING_SAMPLES == 0
    assert set(inspect.signature(comparison.run_comparison).parameters) == {
        "baseline_worker_manifest",
        "candidate_worker_manifest",
        "corpus_path",
        "scratch_parent",
        "output_path",
        "lock_path",
    }


def test_default_controller_runs_exact_sequential_two_by_one_and_is_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_state: dict[str, Any],
) -> None:
    source = _synthetic_edacc(tmp_path, monkeypatch)
    calls: list[tuple[str, str, dict[str, object]]] = []
    active = 0
    maximum_active = 0

    def benchmark(worker: Path, corpus: Path, **kwargs: object) -> dict[str, object]:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        calls.append((worker.name, corpus.name, dict(kwargs)))
        manifest = _manifest_double(
            worker,
            root=tmp_path / "worker-evidence",
            state=worker_state,
        )
        report = _raw_report(worker, corpus, manifest=manifest)
        report["worker"]["verbatim_transcript"] = "PRIVATE_TRANSCRIPT_CANARY"
        report["evidence"]["relative_source"] = "private/source.f32le"
        report["worker_stderr"]["speaker_identity"] = "PRIVATE_SPEAKER_CANARY"
        active -= 1
        return report

    monkeypatch.setattr(streaming_stt_eval, "run_benchmark", benchmark)
    result = _run(tmp_path, source)

    assert maximum_active == 1
    assert [(worker, corpus) for worker, corpus, _kwargs in calls] == [
        ("baseline.json", "corpus.json"),
        ("candidate.json", "corpus.json"),
    ]
    for _worker, _corpus, kwargs in calls:
        assert kwargs == {
            "scratch_parent": (tmp_path / "scratch").resolve(),
            "repeats": 3,
            "stream": None,
            "chunk_samples": 1600,
            "pace": "burst",
            "partial_interval_ms": 200,
            "tail_padding_samples": 0,
            "stratum_tags": _TAGS,
        }

    assert result["ok"] is True
    assert result["execution_complete"] is True
    assert result["execution_semantics"] == "coverage_only"
    assert result["quality_decision"] == "not_evaluated"
    assert result["development_only"] is True
    assert result["promotional"] is False
    assert result["source"]["source_id"] == "edacc-test"
    assert result["source"]["lock_raw_sha256"] == (
        "68ac2c85b8825fabda10b578646780fc16a37a1b2ab0ca3285fa9f8c93a3168b"
    )
    assert result["source"]["cases"] == 24
    assert result["source"]["strata"] == [
        {"tag": tag, "cases": count}
        for tag, count in zip(_TAGS, _COUNTS, strict=True)
    ]
    assert [row["role"] for row in result["model_bindings"]] == [
        "baseline",
        "candidate",
    ]
    assert [row["adapter"] for row in result["model_bindings"]] == [
        SHERPA_ZIPFORMER_ADAPTER,
        FASTER_WHISPER_ENDPOINT_ADAPTER,
    ]
    assert [row["model_id"] for row in result["model_bindings"]] == [
        _BASELINE_ID,
        _CANDIDATE_ID,
    ]
    assert result["suite"]["controller"]["runs"] == 2
    assert [
        (run["binding"]["worker_index"], run["binding"]["corpus_index"])
        for run in result["suite"]["runs"]
    ] == [(0, 0), (1, 0)]
    assert comparison.validate_comparison_report(
        json.loads(comparison._canonical_json_bytes(result))
    ) == result

    encoded = json.dumps(result, sort_keys=True)
    assert str(tmp_path) not in encoded
    assert "PRIVATE_TRANSCRIPT_CANARY" not in encoded
    assert "private/source.f32le" not in encoded
    assert "PRIVATE_SPEAKER_CANARY" not in encoded
    assert "expected_text" not in encoded

    scratch = tmp_path / "scratch"
    assert scratch.stat().st_mode & 0o777 == 0o700
    checkpoints = tuple(path for path in scratch.iterdir() if path.is_file())
    assert len(checkpoints) == 1
    assert checkpoints[0].stat().st_mode & 0o777 == 0o600
    raw = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    assert raw["complete"] is True
    assert raw["controller"]["completed_runs"] == 2
    assert tmp_path.joinpath("report.json").exists() is False


def test_synthetic_source_does_not_bypass_retained_digest_pins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_state: dict[str, Any],
) -> None:
    lock = fixture.load_fixture_lock()
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir(mode=0o700)
    corpus_path = _publish_source(fixture_root, lock, "edacc-test")
    called = False

    def runner(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(comparison, "_PRODUCTION_RUNNER", runner)
    with pytest.raises(comparison.ComparisonError):
        comparison.run_comparison(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_path=corpus_path,
            scratch_parent=tmp_path / "scratch",
            output_path=tmp_path / "report.json",
        )

    assert called is False
    assert (tmp_path / "scratch").exists() is False
    assert (tmp_path / "report.json").exists() is False


def test_nonproduction_edacc_receipt_fails_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_state: dict[str, Any],
) -> None:
    lock = fixture.load_fixture_lock()

    def mark_nonproduction(receipt: dict[str, object]) -> None:
        decoder = receipt["decoder"]
        assert isinstance(decoder, dict)
        decoder["production_evidence"] = False

    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir(mode=0o700)
    corpus_path = _publish_source(
        fixture_root,
        lock,
        "edacc-test",
        mutate_receipt=mark_nonproduction,
    )
    monkeypatch.setattr(comparison, "EXPECTED_CORPUS_SHA256", load_corpus(corpus_path).digest)
    monkeypatch.setattr(
        comparison,
        "EXPECTED_RECEIPT_SHA256",
        hashlib.sha256((corpus_path.parent / "preparation-receipt.json").read_bytes()).hexdigest(),
    )
    called = False

    def runner(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(comparison, "_PRODUCTION_RUNNER", runner)
    with pytest.raises(comparison.ComparisonError):
        comparison.run_comparison(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_path=corpus_path,
            scratch_parent=tmp_path / "scratch",
            output_path=tmp_path / "report.json",
        )
    assert called is False


@pytest.mark.parametrize("mutation", ("missing", "sibling"))
def test_exact_edacc_case_tag_layout_fails_before_runner(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_state: dict[str, Any],
) -> None:
    source = _synthetic_edacc(tmp_path, monkeypatch)
    real_load = comparison.load_corpus
    called = False

    def changed_load(path: Path | str):
        corpus = real_load(path)
        case = corpus.cases[0]
        tags = (
            tuple(tag for tag in case.tags if tag != "clean")
            if mutation == "missing"
            else (*case.tags, "disfluent")
        )
        return replace(corpus, cases=(replace(case, tags=tags), *corpus.cases[1:]))

    def runner(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(comparison, "load_corpus", changed_load)
    monkeypatch.setattr(comparison, "_PRODUCTION_RUNNER", runner)
    with pytest.raises(comparison.ComparisonError):
        _run(tmp_path, source)
    assert called is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("baseline_adapter", FASTER_WHISPER_ENDPOINT_ADAPTER),
        ("baseline_schema", 5),
        ("baseline_model_id", "wrong-zipformer"),
        ("candidate_model_id", "faster-whisper-endpoint-local"),
        ("candidate_language", "ro"),
        ("same_digest", True),
    ),
)
def test_fixed_worker_roles_fail_before_runner(
    field: str,
    value: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_state: dict[str, Any],
) -> None:
    source = _synthetic_edacc(tmp_path, monkeypatch)
    worker_state[field] = value
    called = False

    def runner(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(comparison, "_PRODUCTION_RUNNER", runner)
    with pytest.raises(comparison.ComparisonError):
        _run(tmp_path, source)
    assert called is False
    assert (tmp_path / "scratch").exists() is False


def _mutate_bound_input(
    target: str,
    *,
    source: _SyntheticEdacc,
    worker_state: dict[str, Any],
    implementation_state: dict[str, bool],
) -> None:
    if target == "receipt":
        source.receipt_path.write_bytes(source.receipt_path.read_bytes() + b" ")
        source.receipt_path.chmod(0o600)
    elif target == "worker":
        worker_state["baseline_digest"] = "changed-after-preflight"
    elif target == "implementation":
        implementation_state["changed"] = True
    else:  # pragma: no cover - the parametrization is closed above
        raise AssertionError(target)


@pytest.mark.parametrize("target", ("receipt", "worker", "implementation"))
def test_bound_inputs_are_rechecked_after_delegation(
    target: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_state: dict[str, Any],
) -> None:
    source = _synthetic_edacc(tmp_path, monkeypatch)
    _install_fake_benchmark(monkeypatch, tmp_path, worker_state)
    real_runner = streaming_stt_suite.run_suite
    original_implementation = comparison._implementation_sha256()
    implementation_state = {"changed": False}
    monkeypatch.setattr(
        comparison,
        "_implementation_sha256",
        lambda: (
            "f" * 64
            if implementation_state["changed"]
            else original_implementation
        ),
    )

    def mutate_after_run(*args: object, **kwargs: object) -> dict[str, object]:
        result = real_runner(*args, **kwargs)
        _mutate_bound_input(
            target,
            source=source,
            worker_state=worker_state,
            implementation_state=implementation_state,
        )
        return result

    monkeypatch.setattr(comparison, "_PRODUCTION_RUNNER", mutate_after_run)
    with pytest.raises(comparison.ComparisonError):
        _run(tmp_path, source)

    assert (tmp_path / "report.json").exists() is False
    checkpoint = next((tmp_path / "scratch").iterdir())
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["complete"] is True


@pytest.mark.parametrize("target", ("receipt", "worker", "implementation"))
def test_bound_inputs_are_rechecked_after_safe_reconstruction(
    target: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_state: dict[str, Any],
) -> None:
    source = _synthetic_edacc(tmp_path, monkeypatch)
    _install_fake_benchmark(monkeypatch, tmp_path, worker_state)
    real_validate = comparison.validate_comparison_report
    original_implementation = comparison._implementation_sha256()
    implementation_state = {"changed": False}
    monkeypatch.setattr(
        comparison,
        "_implementation_sha256",
        lambda: (
            "f" * 64
            if implementation_state["changed"]
            else original_implementation
        ),
    )

    def validate_then_mutate(value: object) -> dict[str, object]:
        result = real_validate(value)
        _mutate_bound_input(
            target,
            source=source,
            worker_state=worker_state,
            implementation_state=implementation_state,
        )
        return result

    monkeypatch.setattr(
        comparison,
        "validate_comparison_report",
        validate_then_mutate,
    )
    with pytest.raises(comparison.ComparisonError):
        _run(tmp_path, source)
    assert (tmp_path / "report.json").exists() is False


@pytest.mark.parametrize("failure", (SystemExit(7), KeyboardInterrupt()))
def test_final_closure_baseexception_is_translated(
    failure: BaseException,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_state: dict[str, Any],
) -> None:
    source = _synthetic_edacc(tmp_path, monkeypatch)
    _install_fake_benchmark(monkeypatch, tmp_path, worker_state)
    real_source_state = comparison._source_state
    calls = 0

    def fail_on_final_recheck(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise failure
        return real_source_state(*args, **kwargs)

    monkeypatch.setattr(comparison, "_source_state", fail_on_final_recheck)
    with pytest.raises(comparison.ComparisonError) as caught:
        _run(tmp_path, source)

    assert calls == 3
    assert str(caught.value) == ""
    assert (tmp_path / "report.json").exists() is False


@pytest.mark.parametrize("failure", (SystemExit(7), KeyboardInterrupt()))
def test_runner_baseexception_is_translated_and_leaves_no_output(
    failure: BaseException,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_state: dict[str, Any],
) -> None:
    source = _synthetic_edacc(tmp_path, monkeypatch)

    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise failure

    monkeypatch.setattr(comparison, "_PRODUCTION_RUNNER", fail)
    with pytest.raises(comparison.ComparisonError) as caught:
        _run(tmp_path, source)
    assert str(caught.value) == ""
    assert (tmp_path / "report.json").exists() is False


def test_complete_red_matrix_stays_red_and_nonpromotional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_state: dict[str, Any],
) -> None:
    source = _synthetic_edacc(tmp_path, monkeypatch)
    calls = 0

    def benchmark(worker: Path, corpus: Path, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        manifest = _manifest_double(
            worker,
            root=tmp_path / "worker-evidence",
            state=worker_state,
        )
        report = _raw_report(worker, corpus, manifest=manifest)
        if calls == 1:
            report["ok"] = False
            report["metrics"]["coverage_complete"] = False
        return report

    monkeypatch.setattr(streaming_stt_eval, "run_benchmark", benchmark)
    result = _run(tmp_path, source)

    assert calls == 2
    assert result["ok"] is False
    assert result["execution_complete"] is True
    assert result["quality_decision"] == "not_evaluated"
    assert result["development_only"] is True
    assert result["promotional"] is False
    assert comparison.validate_comparison_report(result) == result


def test_second_cell_failure_preserves_private_red_checkpoint_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_state: dict[str, Any],
) -> None:
    source = _synthetic_edacc(tmp_path, monkeypatch)
    _install_fake_benchmark(
        monkeypatch,
        tmp_path,
        worker_state,
        fail_call=2,
    )

    with pytest.raises(comparison.ComparisonError) as caught:
        _run(tmp_path, source)

    assert "PRIVATE_SECOND_CELL_FAILURE" not in str(caught.value)
    assert (tmp_path / "report.json").exists() is False
    scratch = tmp_path / "scratch"
    checkpoints = tuple(path for path in scratch.iterdir() if path.is_file())
    assert len(checkpoints) == 1
    assert checkpoints[0].stat().st_mode & 0o777 == 0o600
    text = checkpoints[0].read_text(encoding="utf-8")
    checkpoint = json.loads(text)
    assert checkpoint["ok"] is False
    assert checkpoint["complete"] is False
    assert checkpoint["controller"]["state"] == "failed"
    assert checkpoint["controller"]["completed_runs"] == 1
    assert "PRIVATE_SECOND_CELL_FAILURE" not in text
    assert str(tmp_path) not in text


def _rebind_safe_run(result: dict[str, object], run_index: int = 0) -> None:
    suite = result["suite"]
    assert isinstance(suite, dict)
    runs = suite["runs"]
    assert isinstance(runs, list)
    run = runs[run_index]
    report = run["report"]
    binding = run["binding"]
    binding.pop("binding_sha256")
    binding["report_sha256"] = qualification._canonical_sha256(report)
    binding["binding_sha256"] = qualification._canonical_sha256(binding)
    matrix = qualification._canonical_sha256([item["binding"] for item in runs])
    controller = suite["controller"]
    controller["matrix_sha256"] = matrix
    controller["completed_matrix_sha256"] = matrix


def _rebind_contract(result: dict[str, object]) -> None:
    contract = result["contract"]
    assert isinstance(contract, dict)
    body = dict(contract)
    body.pop("contract_sha256")
    contract["contract_sha256"] = qualification._canonical_sha256(body)


@pytest.mark.parametrize(
    "mutation",
    (
        "promotional",
        "run_order",
        "adapter",
        "stratum_count",
        "pcm_bytes",
        "lock_raw_sha256",
    ),
)
def test_strict_retained_validator_rejects_tamper(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_state: dict[str, Any],
) -> None:
    source = _synthetic_edacc(tmp_path, monkeypatch)
    _install_fake_benchmark(monkeypatch, tmp_path, worker_state)
    result = _run(tmp_path, source)

    if mutation == "promotional":
        result["promotional"] = True
    elif mutation == "run_order":
        result["suite"]["runs"].reverse()
        controller = result["suite"]["controller"]
        matrix = qualification._canonical_sha256(
            [item["binding"] for item in result["suite"]["runs"]]
        )
        controller["matrix_sha256"] = matrix
        controller["completed_matrix_sha256"] = matrix
    elif mutation == "adapter":
        run = result["suite"]["runs"][0]
        run["report"]["worker"]["adapter"] = FASTER_WHISPER_ENDPOINT_ADAPTER
        run["binding"]["worker_adapter"] = FASTER_WHISPER_ENDPOINT_ADAPTER
        _rebind_safe_run(result)
    elif mutation == "stratum_count":
        run = result["suite"]["runs"][0]
        run["report"]["metrics"]["strata"][0]["cases"] = 7
        _rebind_safe_run(result)
    elif mutation == "pcm_bytes":
        changed = int(result["source"]["pcm_bytes"]) + 4
        result["source"]["pcm_bytes"] = changed
        result["contract"]["expected_pcm_bytes"] = changed
        _rebind_contract(result)
    else:
        changed = "0" * 64
        result["source"]["lock_raw_sha256"] = changed
        result["contract"]["expected_lock_raw_sha256"] = changed
        _rebind_contract(result)

    with pytest.raises(comparison.ComparisonError):
        comparison.validate_comparison_report(result)


@pytest.mark.parametrize(
    "mutation",
    (
        "outer_schema",
        "model_schema",
        "model_index",
        "controller_schema",
        "controller_corpora",
        "suite_config_tail",
        "source_cases",
        "source_pcm",
        "source_stratum_count",
        "run_corpus_index",
        "run_worker_index",
    ),
)
def test_retained_validator_rejects_boolean_integer_aliases(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_state: dict[str, Any],
) -> None:
    source = _synthetic_edacc(tmp_path, monkeypatch)
    _install_fake_benchmark(monkeypatch, tmp_path, worker_state)
    result = _run(tmp_path, source)

    if mutation == "outer_schema":
        result["schema_version"] = True
    elif mutation in {"model_schema", "model_index"}:
        row = result["model_bindings"][1]
        if mutation == "model_schema":
            row["schema_version"] = True
        else:
            row["worker_index"] = True
        result["contract"]["model_binding_set_sha256"] = (
            qualification._canonical_sha256(result["model_bindings"])
        )
        _rebind_contract(result)
    elif mutation in {"controller_schema", "controller_corpora"}:
        controller = result["suite"]["controller"]
        if mutation == "controller_schema":
            controller["schema_version"] = True
        else:
            controller["planned_corpora"] = True
    elif mutation == "suite_config_tail":
        config = result["suite"]["config"]
        config["overrides"]["tail_padding_samples"] = False
        body = dict(config)
        body.pop("contract_sha256")
        config["contract_sha256"] = qualification._canonical_sha256(body)
    elif mutation == "source_cases":
        result["source"]["cases"] = True
    elif mutation == "source_pcm":
        result["source"]["pcm_bytes"] = True
    elif mutation == "source_stratum_count":
        result["source"]["strata"][0]["cases"] = True
        result["contract"]["source_strata"][0]["cases"] = True
        _rebind_contract(result)
    else:
        run_index = 1 if mutation == "run_worker_index" else 0
        binding = result["suite"]["runs"][run_index]["binding"]
        if mutation == "run_worker_index":
            binding["worker_index"] = True
        else:
            binding["corpus_index"] = False
        _rebind_safe_run(result, run_index)

    with pytest.raises(comparison.ComparisonError):
        comparison.validate_comparison_report(result)


@pytest.mark.parametrize("placement", ("scratch_in_corpus", "output_in_corpus", "overlap"))
def test_private_path_separation_fails_without_creation(
    placement: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_state: dict[str, Any],
) -> None:
    source = _synthetic_edacc(tmp_path, monkeypatch)
    scratch = tmp_path / "scratch"
    output = tmp_path / "report.json"
    if placement == "scratch_in_corpus":
        scratch = source.corpus_path.parent / "scratch"
    elif placement == "output_in_corpus":
        output = source.corpus_path.parent / "report.json"
    else:
        output = tmp_path / "future-output"
        scratch = output / "scratch"

    with pytest.raises(comparison.ComparisonError):
        comparison.run_comparison(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_path=source.corpus_path,
            scratch_parent=scratch,
            output_path=output,
        )
    assert scratch.exists() is False
    assert output.exists() is False


def test_private_report_has_exact_digest_and_no_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_state: dict[str, Any],
) -> None:
    source = _synthetic_edacc(tmp_path, monkeypatch)
    _install_fake_benchmark(monkeypatch, tmp_path, worker_state)
    result = _run(tmp_path, source)
    output = tmp_path / "published.json"

    digest = comparison.write_new_report(output, result)

    assert output.stat().st_mode & 0o777 == 0o600
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert json.loads(output.read_text(encoding="ascii")) == result
    with pytest.raises(comparison.ComparisonError):
        comparison.write_new_report(output, result)


def test_cli_success_prints_exact_digest_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    worker_state: dict[str, Any],
) -> None:
    source = _synthetic_edacc(tmp_path, monkeypatch)
    _install_fake_benchmark(monkeypatch, tmp_path, worker_state)
    output = tmp_path / "report.json"

    code = comparison.main(
        [
            "--baseline-worker-manifest",
            str(tmp_path / "baseline.json"),
            "--candidate-worker-manifest",
            str(tmp_path / "candidate.json"),
            "--corpus",
            str(source.corpus_path),
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert code == 0
    assert captured.err == ""
    assert output.stat().st_mode & 0o777 == 0o600
    assert summary == {
        "ok": True,
        "kind": comparison.KIND,
        "execution_complete": True,
        "quality_decision": "not_evaluated",
        "development_only": True,
        "promotional": False,
        "report_written": True,
        "report_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def test_cli_failure_hides_private_paths_and_does_not_clobber(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private = tmp_path / "PRIVATE_CORPUS_CANARY" / "corpus.json"
    output = tmp_path / "report.json"

    code = comparison.main(
        [
            "--baseline-worker-manifest",
            str(tmp_path / "baseline.json"),
            "--candidate-worker-manifest",
            str(tmp_path / "candidate.json"),
            "--corpus",
            str(private),
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert json.loads(captured.err) == comparison._SAFE_ERROR
    assert str(private) not in captured.err
    assert "PRIVATE_CORPUS_CANARY" not in captured.err
    assert output.exists() is False
