from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from tools import primock57_isolated_eval as subject
from tools import prepare_primock57_conversation_fixture as fixture
from tools.streaming_stt.corpus import (
    CorpusCase,
    CorpusProvenance,
    LoadedCorpus,
)


def _bundle(
    tmp_path: Path,
    *,
    production: bool = True,
    corpus_path: Path | None = None,
) -> fixture.LoadedPrimock57IsolatedBundle:
    root = tmp_path / "private-fixture"
    root.mkdir(mode=0o700, exist_ok=True)
    selected_corpus_path = corpus_path or root / "corpus.json"
    selected_corpus_path.write_bytes(b"{}\n")
    selected_corpus_path.chmod(0o600)
    cases: list[CorpusCase] = []
    for index in range(3):
        audio = bytes((index + 1, 0, 0, 0))
        pcm = root / f"primock57-isolated-{index:02d}.f32le"
        pcm.write_bytes(audio)
        pcm.chmod(0o600)
        cases.append(
            CorpusCase(
                case_id=f"primock57-isolated-{index:02d}",
                source_path=pcm.resolve(),
                sha256=hashlib.sha256(audio).hexdigest(),
                samples=1,
                expected_text=f"private reference phrase number {index}",
                assertion="transcript",
                commands=(),
                forbidden_commands=(),
                tags=("primock57", "isolated", "role-a"),
                audio_bytes=audio,
            )
        )
    corpus = LoadedCorpus(
        path=selected_corpus_path.resolve(),
        digest="a" * 64,
        schema_version=2,
        purpose="isolated test purpose",
        provenance=CorpusProvenance(
            kind="public-voice-v1",
            suite=fixture.FIXTURE_ID,
            manifest_sha256="b" * 64,
            metadata_sha256="c" * 64,
            source_set_sha256="d" * 64,
        ),
        cases=tuple(cases),
        audio_bytes=sum(len(case.audio_bytes) for case in cases),
    )
    return fixture.LoadedPrimock57IsolatedBundle(
        corpus=corpus,
        receipt_sha256="e" * 64,
        source_contract_sha256="f" * 64,
        production_evidence=production,
        identities=((1, 2, 3), (4, 5, 6)),
    )


def _raw_report(
    bundle: fixture.LoadedPrimock57IsolatedBundle,
) -> dict[str, object]:
    return {
        "ok": True,
        "evidence": {"aggregate_only": True},
        "worker": {"manifest_sha256": "1" * 64},
        "corpus": subject._corpus_binding(bundle),
        "config": {"repeats": 1},
        "metrics": {
            "evaluations": 3,
            "transcript": {"wer": 0.25, "reference_words": 12},
        },
        "worker_stderr": {
            "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "truncated": False,
        },
        "evaluator": {
            "schema_version": 1,
            "kind": "isolated_streaming_stt_fake_harness",
            "production_model": False,
            "real_candidate_model": False,
            "model_executed": False,
            "files": {"tools/fake.py": "2" * 64},
        },
    }


def _install_success_fakes(
    monkeypatch: pytest.MonkeyPatch,
    bundle: fixture.LoadedPrimock57IsolatedBundle,
    *,
    events: list[str] | None = None,
) -> list[tuple[tuple[object, ...], dict[str, object]]]:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load(_root: Path) -> fixture.LoadedPrimock57IsolatedBundle:
        if events is not None:
            events.append("validate")
        return bundle

    def run(*args: object, **kwargs: object) -> dict[str, object]:
        if events is not None:
            events.append("runner")
        calls.append((args, kwargs))
        return _raw_report(bundle)

    monkeypatch.setattr(
        subject.primock57_fixture,
        "load_primock57_isolated_bundle",
        load,
    )
    monkeypatch.setattr(subject, "_PRODUCTION_RUNNER", run)
    return calls


def test_validates_before_runner_and_adds_private_aggregate_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    events: list[str] = []
    calls = _install_success_fakes(monkeypatch, bundle, events=events)

    report = subject.run_primock57_isolated_eval(
        tmp_path / "worker-manifest.json",
        bundle.corpus.path,
        scratch_parent=tmp_path / "scratch",
        repeats=1,
        chunk_samples=1_600,
        pace="burst",
        partial_interval_ms=200,
        tail_padding_samples=0,
        stratum_tags=("isolated",),
    )

    assert events[:2] == ["validate", "runner"]
    assert events == ["validate", "runner", "validate", "validate"]
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (
        tmp_path / "worker-manifest.json",
        bundle.corpus.path,
    )
    assert kwargs == {
        "scratch_parent": tmp_path / "scratch",
        "repeats": 1,
        "stream": None,
        "chunk_samples": 1_600,
        "pace": "burst",
        "partial_interval_ms": 200,
        "tail_padding_samples": 0,
        "stratum_tags": ("isolated",),
    }
    binding = report[subject.BINDING_FIELD]
    assert isinstance(binding, dict)
    assert binding["kind"] == subject.KIND
    assert binding["schema_version"] == subject.SCHEMA_VERSION
    assert binding["fixture_id"] == fixture.FIXTURE_ID
    assert binding["production_evidence"] is True
    assert binding["corpus_manifest_sha256"] == bundle.corpus.digest
    assert binding["corpus_cases"] == 3
    assert binding["receipt_sha256"] == bundle.receipt_sha256
    assert binding["source_contract_sha256"] == bundle.source_contract_sha256
    assert len(str(binding["wrapper_closure_sha256"])) == 64
    assert len(str(binding["evaluator_report_sha256"])) == 64
    assert len(str(binding["binding_sha256"])) == 64

    encoded = json.dumps(report, sort_keys=True)
    assert str(tmp_path) not in encoded
    for case in bundle.corpus.cases:
        assert case.expected_text not in encoded
        assert str(case.source_path) not in encoded
    assert report["metrics"] == {
        "evaluations": 3,
        "transcript": {"wer": 0.25, "reference_words": 12},
    }


def test_wrapper_eager_local_imports_are_covered_by_bound_closures() -> None:
    expected = {
        "core/__init__.py",
        "core/config.py",
        "core/wer.py",
        "tools/__init__.py",
        "tools/prepare_primock57_conversation_fixture.py",
        "tools/primock57_isolated_eval.py",
        "tools/recorded_stt_eval.py",
        "tools/streaming_stt/__init__.py",
        "tools/streaming_stt/ami_endpoint_proxy.py",
        "tools/streaming_stt/bounded_io.py",
        "tools/streaming_stt/cgroup_observation.py",
        "tools/streaming_stt/corpus.py",
        "tools/streaming_stt/corpus_writer.py",
        "tools/streaming_stt/manifest.py",
        "tools/streaming_stt/metrics.py",
        "tools/streaming_stt/private_diagnostic_receipt.py",
        "tools/streaming_stt/protocol.py",
        "tools/streaming_stt/runtime_receipt.py",
        "tools/streaming_stt/source_bundle.py",
        "tools/streaming_stt/supervisor.py",
        "tools/streaming_stt_eval.py",
    }
    bound = (
        set(fixture._PREPARER_FILES)
        | set(subject.streaming_stt_eval._EVALUATOR_FILES)
        | {"tools/primock57_isolated_eval.py"}
    )
    assert "core/config.py" in subject.streaming_stt_eval._EVALUATOR_FILES
    assert expected <= bound

    repo_root = Path(subject.__file__).resolve().parents[1]
    script = """
import json
from pathlib import Path
import sys
import tools.primock57_isolated_eval
root = Path.cwd().resolve()
rows = sorted({
    Path(module.__file__).resolve().relative_to(root).as_posix()
    for module in sys.modules.values()
    if getattr(module, "__file__", None)
    and Path(module.__file__).resolve().is_relative_to(root)
    and Path(module.__file__).suffix == ".py"
})
print(json.dumps(rows, separators=(",", ":")))
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert json.loads(completed.stdout) == sorted(expected)


def test_synthetic_bundle_is_rejected_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path, production=False)
    runner_called = False

    monkeypatch.setattr(
        subject.primock57_fixture,
        "load_primock57_isolated_bundle",
        lambda _root: bundle,
    )

    def run(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal runner_called
        runner_called = True
        return _raw_report(bundle)

    monkeypatch.setattr(subject, "_PRODUCTION_RUNNER", run)
    with pytest.raises(subject.Primock57IsolatedEvalError):
        subject.run_primock57_isolated_eval(
            "worker.json",
            bundle.corpus.path,
            scratch_parent=tmp_path / "scratch",
        )
    assert runner_called is False


def test_wrong_input_corpus_path_is_rejected_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = tmp_path / "requested" / "corpus.json"
    requested.parent.mkdir()
    requested.write_bytes(b"{}\n")
    bundle = _bundle(tmp_path)
    runner_called = False
    monkeypatch.setattr(
        subject.primock57_fixture,
        "load_primock57_isolated_bundle",
        lambda _root: bundle,
    )

    def run(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal runner_called
        runner_called = True
        return _raw_report(bundle)

    monkeypatch.setattr(subject, "_PRODUCTION_RUNNER", run)
    with pytest.raises(subject.Primock57IsolatedEvalError):
        subject.run_primock57_isolated_eval(
            "worker.json",
            requested,
            scratch_parent=tmp_path / "scratch",
        )
    assert runner_called is False


@pytest.mark.parametrize("mutation", ["receipt", "pcm"])
def test_pre_runner_receipt_or_pcm_validation_failure_never_invokes_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    bundle = _bundle(tmp_path)
    runner_called = False

    def load(_root: Path) -> fixture.LoadedPrimock57IsolatedBundle:
        raise fixture.Primock57PreparationError(mutation)

    def run(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal runner_called
        runner_called = True
        return _raw_report(bundle)

    monkeypatch.setattr(
        subject.primock57_fixture,
        "load_primock57_isolated_bundle",
        load,
    )
    monkeypatch.setattr(subject, "_PRODUCTION_RUNNER", run)
    with pytest.raises(subject.Primock57IsolatedEvalError):
        subject.run_primock57_isolated_eval(
            "worker.json",
            bundle.corpus.path,
            scratch_parent=tmp_path / "scratch",
        )
    assert runner_called is False


@pytest.mark.parametrize("mutation", ["receipt", "pcm"])
def test_post_runner_receipt_or_pcm_mutation_rejects_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    initial = _bundle(tmp_path)
    if mutation == "receipt":
        changed = replace(initial, receipt_sha256="0" * 64)
    else:
        changed_case = replace(
            initial.corpus.cases[0],
            audio_bytes=b"\x09\x00\x00\x00",
            sha256=hashlib.sha256(b"\x09\x00\x00\x00").hexdigest(),
        )
        changed_corpus = replace(
            initial.corpus,
            cases=(changed_case, *initial.corpus.cases[1:]),
        )
        changed = replace(initial, corpus=changed_corpus)
    load_calls = 0

    def load(_root: Path) -> fixture.LoadedPrimock57IsolatedBundle:
        nonlocal load_calls
        load_calls += 1
        return initial if load_calls == 1 else changed

    monkeypatch.setattr(
        subject.primock57_fixture,
        "load_primock57_isolated_bundle",
        load,
    )
    monkeypatch.setattr(
        subject,
        "_PRODUCTION_RUNNER",
        lambda *_args, **_kwargs: _raw_report(initial),
    )
    with pytest.raises(subject.Primock57IsolatedEvalError):
        subject.run_primock57_isolated_eval(
            "worker.json",
            initial.corpus.path,
            scratch_parent=tmp_path / "scratch",
        )
    assert load_calls == 2


def test_post_runner_directory_chmod_restore_rejects_retained_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    root = bundle.corpus.path.parent
    snapshots: list[tuple[int, ...]] = []

    def directory_snapshot() -> tuple[int, ...]:
        metadata = root.lstat()
        return (
            metadata.st_dev,
            metadata.st_ino,
            stat.S_IFMT(metadata.st_mode),
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
            metadata.st_nlink,
            stat.S_IMODE(metadata.st_mode),
            metadata.st_uid,
        )

    def load(_root: Path) -> fixture.LoadedPrimock57IsolatedBundle:
        snapshot = directory_snapshot()
        snapshots.append(snapshot)
        return replace(
            bundle,
            identities=(snapshot, *bundle.identities[1:]),
        )

    def run(*_args: object, **_kwargs: object) -> dict[str, object]:
        before = root.lstat()
        root.chmod(0o755)
        root.chmod(0o700)
        os.utime(
            root,
            ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
        )
        return _raw_report(bundle)

    monkeypatch.setattr(
        subject.primock57_fixture,
        "load_primock57_isolated_bundle",
        load,
    )
    monkeypatch.setattr(subject, "_PRODUCTION_RUNNER", run)
    with pytest.raises(subject.Primock57IsolatedEvalError) as error:
        subject.run_primock57_isolated_eval(
            "worker.json",
            bundle.corpus.path,
            scratch_parent=tmp_path / "scratch",
        )

    assert str(error.value) == ""
    assert stat.S_IMODE(root.lstat().st_mode) == 0o700
    assert len(snapshots) == 2
    assert snapshots[0] != snapshots[1]
    assert snapshots[0][1] == snapshots[1][1]


@pytest.mark.parametrize("field", ["manifest_sha256", "cases"])
def test_raw_evaluator_wrong_corpus_binding_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(
        subject.primock57_fixture,
        "load_primock57_isolated_bundle",
        lambda _root: bundle,
    )

    def run(*_args: object, **_kwargs: object) -> dict[str, object]:
        report = _raw_report(bundle)
        corpus = dict(report["corpus"])  # type: ignore[arg-type]
        corpus[field] = "0" * 64 if field == "manifest_sha256" else 2
        report["corpus"] = corpus
        return report

    monkeypatch.setattr(subject, "_PRODUCTION_RUNNER", run)
    with pytest.raises(subject.Primock57IsolatedEvalError):
        subject.run_primock57_isolated_eval(
            "worker.json",
            bundle.corpus.path,
            scratch_parent=tmp_path / "scratch",
        )


def test_runner_exception_is_rejected_after_source_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    load_calls = 0

    def load(_root: Path) -> fixture.LoadedPrimock57IsolatedBundle:
        nonlocal load_calls
        load_calls += 1
        return bundle

    def run(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("private runner detail")

    monkeypatch.setattr(
        subject.primock57_fixture,
        "load_primock57_isolated_bundle",
        load,
    )
    monkeypatch.setattr(subject, "_PRODUCTION_RUNNER", run)
    with pytest.raises(subject.Primock57IsolatedEvalError) as error:
        subject.run_primock57_isolated_eval(
            "worker.json",
            bundle.corpus.path,
            scratch_parent=tmp_path / "scratch",
        )
    assert str(error.value) == ""
    assert load_calls == 2


def test_raw_report_cannot_claim_wrapper_binding_or_leak_private_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(
        subject.primock57_fixture,
        "load_primock57_isolated_bundle",
        lambda _root: bundle,
    )

    def forged(*_args: object, **_kwargs: object) -> dict[str, object]:
        report = _raw_report(bundle)
        report[subject.BINDING_FIELD] = {"kind": subject.KIND}
        return report

    monkeypatch.setattr(subject, "_PRODUCTION_RUNNER", forged)
    with pytest.raises(subject.Primock57IsolatedEvalError):
        subject.run_primock57_isolated_eval(
            "worker.json",
            bundle.corpus.path,
            scratch_parent=tmp_path / "scratch",
        )

    def leaking(*_args: object, **_kwargs: object) -> dict[str, object]:
        report = _raw_report(bundle)
        metrics = dict(report["metrics"])  # type: ignore[arg-type]
        metrics["debug"] = {
            "leak": bundle.corpus.cases[0].expected_text,
        }
        report["metrics"] = metrics
        return report

    monkeypatch.setattr(subject, "_PRODUCTION_RUNNER", leaking)
    with pytest.raises(subject.Primock57IsolatedEvalError):
        subject.run_primock57_isolated_eval(
            "worker.json",
            bundle.corpus.path,
            scratch_parent=tmp_path / "scratch",
        )


@pytest.mark.parametrize(
    "detail",
    [
        "failed at /tmp/private-model",
        r"prefix C:\Users\paul\model",
        r"C:Users\paul\model",
        r"\Users\paul\model",
        r"prefix \Users\paul\model",
        r"\secret",
        r"prefix \secret",
    ],
)
def test_embedded_local_path_is_rejected_in_closed_worker_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    detail: str,
) -> None:
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(
        subject.primock57_fixture,
        "load_primock57_isolated_bundle",
        lambda _root: bundle,
    )

    def leaking(*_args: object, **_kwargs: object) -> dict[str, object]:
        report = _raw_report(bundle)
        report["worker_stderr"] = {
            "bytes": len(detail),
            "sha256": hashlib.sha256(detail.encode("utf-8")).hexdigest(),
            "truncated": False,
            "detail": detail,
            "privacy": {"local_paths": False},
        }
        return report

    monkeypatch.setattr(subject, "_PRODUCTION_RUNNER", leaking)
    with pytest.raises(subject.Primock57IsolatedEvalError) as error:
        subject.run_primock57_isolated_eval(
            "worker.json",
            bundle.corpus.path,
            scratch_parent=tmp_path / "scratch",
        )
    assert str(error.value) == ""
    assert detail not in str(error.value)


@pytest.mark.parametrize(
    "detail",
    [
        "nested failure at /tmp/private-model",
        r"nested prefix C:\Users\paul\model",
        r"nested C:Users\paul\model",
        r"\Users\paul\model",
        r"nested prefix \Users\paul\model",
        r"\secret",
        r"nested prefix \secret",
    ],
)
def test_embedded_local_path_is_rejected_in_arbitrary_nested_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    detail: str,
) -> None:
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(
        subject.primock57_fixture,
        "load_primock57_isolated_bundle",
        lambda _root: bundle,
    )

    def leaking(*_args: object, **_kwargs: object) -> dict[str, object]:
        report = _raw_report(bundle)
        metrics = dict(report["metrics"])  # type: ignore[arg-type]
        metrics["debug"] = {
            "detail": detail,
            "privacy": {"local_paths": False},
        }
        report["metrics"] = metrics
        return report

    monkeypatch.setattr(subject, "_PRODUCTION_RUNNER", leaking)
    with pytest.raises(subject.Primock57IsolatedEvalError) as error:
        subject.run_primock57_isolated_eval(
            "worker.json",
            bundle.corpus.path,
            scratch_parent=tmp_path / "scratch",
        )
    assert str(error.value) == ""
    assert detail not in str(error.value)


@pytest.mark.parametrize(
    "fragment",
    [
        "private reference phrase",
        "reference phrase number 0",
    ],
)
def test_verbatim_reference_fragments_are_rejected_despite_privacy_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fragment: str,
) -> None:
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(
        subject.primock57_fixture,
        "load_primock57_isolated_bundle",
        lambda _root: bundle,
    )

    def leaking(*_args: object, **_kwargs: object) -> dict[str, object]:
        report = _raw_report(bundle)
        metrics = dict(report["metrics"])  # type: ignore[arg-type]
        metrics["debug"] = {
            "detail": fragment,
            "privacy": {"verbatim_transcripts": False},
        }
        report["metrics"] = metrics
        return report

    monkeypatch.setattr(subject, "_PRODUCTION_RUNNER", leaking)
    with pytest.raises(subject.Primock57IsolatedEvalError) as error:
        subject.run_primock57_isolated_eval(
            "worker.json",
            bundle.corpus.path,
            scratch_parent=tmp_path / "scratch",
        )
    assert str(error.value) == ""
    assert fragment not in str(error.value)


@pytest.mark.parametrize(
    ("value", "forbidden"),
    [
        ("failed at /tmp/private-model", ()),
        (r"prefix C:\Users\paul\model", ()),
        (r"C:Users\paul\model", ()),
        (r"\Users\paul\model", ()),
        (r"prefix \Users\paul\model", ()),
        (r"\secret", ()),
        (r"prefix \secret", ()),
        (
            "private reference phrase",
            ("private reference phrase number 0",),
        ),
        (
            "reference phrase number 0",
            ("private reference phrase number 0",),
        ),
    ],
)
def test_recursive_privacy_guard_rejects_direct_audit_probes(
    value: str,
    forbidden: tuple[str, ...],
) -> None:
    with pytest.raises(subject.Primock57IsolatedEvalError) as error:
        subject._assert_aggregate_private(
            {"nested": {"detail": value}},
            forbidden_text=forbidden,
        )
    assert str(error.value) == ""
    assert value not in str(error.value)
