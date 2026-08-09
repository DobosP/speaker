from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from tools import prepare_primock57_conversation_fixture as fixture
from tools import primock57_overlap_eval as subject
from tests.test_prepare_primock57_conversation_fixture import _prepare
from tools.streaming_stt.metrics import RunRecord
from tools.streaming_stt.protocol import (
    CaseTrace,
    FinalEvent,
    NativeFinalEvent,
    PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION,
    ParakeetCppFinalEvent,
    ReadyEvent,
    ResourceUsage,
    StreamConfig,
)


def _private_dir(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True)
    path.chmod(0o700)
    return path


def _private_file(path: Path, payload: bytes = b"bound\n") -> Path:
    path.write_bytes(payload)
    path.chmod(0o600)
    return path


@contextmanager
def _real_lifecycle_delivery(signum: int) -> Iterator[list[int]]:
    delivered: list[int] = []

    def handler(number: int, _frame: object) -> None:
        delivered.append(number)
        subject._raise_lifecycle(number)

    previous_mask = subject.signal.pthread_sigmask(
        subject.signal.SIG_UNBLOCK,
        (signum,),
    )
    previous_handler = subject.signal.signal(signum, handler)
    try:
        yield delivered
    finally:
        subject.signal.pthread_sigmask(subject.signal.SIG_BLOCK, (signum,))
        subject.signal.signal(signum, previous_handler)
        subject.signal.pthread_sigmask(subject.signal.SIG_SETMASK, previous_mask)


def _fake_lock() -> subject._EvaluatorLock:
    return subject._EvaluatorLock(
        raw_sha256="a" * 64,
        recipe_sha256="b" * 64,
        manifest_sha256="c" * 64,
        receipt_sha256="d" * 64,
    )


def _bundle_snapshot(tmp_path: Path) -> subject._BundleSnapshot:
    root = _private_dir(tmp_path / "private-bundle")
    manifest_path = _private_file(root / fixture.OVERLAP_MANIFEST_FILENAME)
    inputs: list[subject._PrivateInput] = []
    for case_index in range(3):
        reference_a = f"alpha private utterance number {case_index}"
        reference_b = f"beta confidential response number {case_index}"
        for kind_index, kind in enumerate(("role_a", "role_b", "mix")):
            ordinal = case_index * 3 + kind_index
            audio = bytes((ordinal + 1, 0, 0, 0))
            path = _private_file(root / f"private-input-{ordinal:02d}.f32le", audio)
            inputs.append(
                subject._PrivateInput(
                    case_index=case_index,
                    kind=kind,
                    path=path,
                    sha256=hashlib.sha256(audio).hexdigest(),
                    samples=1,
                    audio=audio,
                    reference_a=reference_a,
                    reference_b=reference_b,
                )
            )
    loaded = fixture.LoadedPrimock57OverlapBundle(
        path=manifest_path,
        digest="c" * 64,
        cases=tuple(
            {"case_id": f"primock57-overlap-{case_index:02d}"}
            for case_index in range(3)
        ),
        receipt_sha256="d" * 64,
        source_contract_sha256="e" * 64,
        production_evidence=True,
    )
    return subject._BundleSnapshot(
        bundle=loaded,
        directory_snapshot=(1, 2, 3),
        file_snapshots=tuple(range(11)),
        inputs=tuple(inputs),
    )


def _manifest(tmp_path: Path) -> SimpleNamespace:
    root = _private_dir(tmp_path / "worker-binding")
    manifest_path = _private_file(root / "worker-manifest.json")
    python_root = _private_dir(tmp_path / "python-binding")
    python_path = _private_file(python_root / "python")
    artifact_root = _private_dir(tmp_path / "artifact-binding")
    artifact_path = _private_file(artifact_root / "fake-script.json")
    return SimpleNamespace(
        path=manifest_path,
        digest="f" * 64,
        adapter="fake-json-v1",
        adapter_config=None,
        worker=SimpleNamespace(path=subject.streaming_stt_eval._FIXED_WORKER),
        python=SimpleNamespace(path=python_path),
        artifacts=(SimpleNamespace(path=artifact_path),),
    )


class _FakeRunLock:
    instances: list[_FakeRunLock] = []

    def __init__(self) -> None:
        self.assertions = 0
        self.closed = False
        self.__class__.instances.append(self)

    @classmethod
    def acquire(cls, _path: Path) -> _FakeRunLock:
        return cls()

    def assert_bound(self) -> None:
        if self.closed:
            raise ValueError
        self.assertions += 1

    def close(self) -> None:
        self.closed = True


class _FakeWorker:
    instances: list[_FakeWorker] = []
    requests: list[object] = []
    text_by_digest: dict[str, str] = {}

    def __init__(self, manifest: object, scratch: Path, source_bundle: object) -> None:
        self.manifest = manifest
        self.scratch = scratch
        self.source_bundle = source_bundle
        self.ready: ReadyEvent | None = None
        self.fully_stopped = False
        self.close_calls = 0
        self.enter_started = False
        self.stderr_summary = {
            "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "truncated": False,
        }
        self.cgroup_evidence = None
        self.resource_observations = None
        self.__class__.instances.append(self)

    def __enter__(self) -> _FakeWorker:
        self.enter_started = True
        self.ready = ReadyEvent(
            model_id="fake-stream-v1",
            manifest_sha256=self.manifest.digest,
            source_bundle_sha256=self.source_bundle.tree_sha256,
            adapter="fake-json-v1",
            model_load_ms=1.0,
            resources=ResourceUsage(rss_mb=1.0, threads=1, vram_mb=None),
            runtime={"python": "test", "platform": "test"},
        )
        return self

    def transcribe(self, request: object) -> CaseTrace:
        self.__class__.requests.append(request)
        text = self.__class__.text_by_digest[request.pcm.sha256]
        return CaseTrace(
            partials=(),
            final=FinalEvent(
                request_id=request.request_id,
                seq=0,
                text=text,
                samples_seen=request.pcm.samples,
                elapsed_ms=1.0,
                finalization_ms=1.0,
                compute_ms=1.0,
                audio_seconds=request.pcm.samples / 16_000,
                chunks=1,
                deadline_misses=0,
                max_backlog_ms=0.0,
                resources=ResourceUsage(rss_mb=1.0, threads=1, vram_mb=None),
            ),
        )

    def __exit__(self, *_exc: object) -> None:
        self.fully_stopped = True

    def close(self) -> None:
        self.close_calls += 1
        self.fully_stopped = True


@dataclass
class _Harness:
    snapshot: subject._BundleSnapshot
    manifest: SimpleNamespace
    scratch_parent: Path
    output_parent: Path
    verify_events: list[str]
    lock_calls: list[str]
    snapshot_calls: list[str]
    closure_calls: list[str]
    manifest_calls: list[str]


def _install_run_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _Harness:
    snapshot = _bundle_snapshot(tmp_path)
    manifest = _manifest(tmp_path)
    scratch_parent = _private_dir(tmp_path / "scratch-parent")
    output_parent = _private_dir(tmp_path / "output-parent")
    verify_events: list[str] = []
    lock_calls: list[str] = []
    snapshot_calls: list[str] = []
    closure_calls: list[str] = []
    manifest_calls: list[str] = []
    source_bundle = SimpleNamespace(
        tree_sha256="1" * 64,
        files=(object(),),
    )

    _FakeRunLock.instances = []
    _FakeWorker.instances = []
    _FakeWorker.requests = []
    _FakeWorker.text_by_digest = {}
    for item in snapshot.inputs:
        if item.kind == "role_a":
            text = item.reference_a
        elif item.kind == "role_b":
            text = item.reference_b
        else:
            text = f"{item.reference_a} {item.reference_b}"
        _FakeWorker.text_by_digest[item.sha256] = text

    def load_lock() -> subject._EvaluatorLock:
        lock_calls.append("lock")
        return _fake_lock()

    def closure() -> str:
        closure_calls.append("closure")
        return "2" * 64

    def load_snapshot(*_args: object) -> subject._BundleSnapshot:
        snapshot_calls.append("snapshot")
        return snapshot

    def load_manifest(_path: object) -> SimpleNamespace:
        manifest_calls.append("manifest")
        return manifest

    monkeypatch.setattr(subject, "_load_lock", load_lock)
    monkeypatch.setattr(subject, "_closure_sha256", closure)
    monkeypatch.setattr(subject, "_snapshot_bundle", load_snapshot)
    monkeypatch.setattr(subject, "StreamingWorker", _FakeWorker)
    monkeypatch.setattr(
        subject, "stage_worker_source_bundle", lambda *_args: source_bundle
    )
    monkeypatch.setattr(
        subject,
        "verify_source_bundle",
        lambda *_args, **_kwargs: verify_events.append("source"),
    )
    monkeypatch.setattr(
        subject.streaming_stt_eval,
        "_BenchmarkRunLock",
        _FakeRunLock,
    )
    monkeypatch.setattr(
        subject.streaming_stt_eval,
        "load_worker_manifest",
        load_manifest,
    )
    monkeypatch.setattr(
        subject.streaming_stt_eval,
        "_selected_stream",
        lambda *_args, **_kwargs: StreamConfig(1, "burst", 1, 0),
    )
    monkeypatch.setattr(
        subject.streaming_stt_eval,
        "_verified_runtime_receipt_digest",
        lambda _manifest: None,
    )
    monkeypatch.setattr(
        subject.streaming_stt_eval,
        "_verified_model_receipt_digest",
        lambda _manifest: None,
    )
    monkeypatch.setattr(
        subject.streaming_stt_eval,
        "_evaluator_binding",
        lambda _adapter: {
            "schema_version": 1,
            "kind": "isolated_streaming_stt_fake_harness",
            "production_model": False,
            "real_candidate_model": False,
            "model_executed": False,
            "files": {"tools/fake.py": "3" * 64},
        },
    )
    monkeypatch.setattr(
        subject.streaming_stt_eval,
        "_worker_binding",
        lambda *_args, **_kwargs: {
            "manifest_sha256": manifest.digest,
            "model_id": "fake-stream-v1",
            "adapter": "fake-json-v1",
            "python_sha256": "4" * 64,
            "worker_sha256": "5" * 64,
            "source_bundle_sha256": source_bundle.tree_sha256,
            "source_bundle_files": 1,
            "artifact_set_sha256": "6" * 64,
            "artifacts": [],
        },
    )
    monkeypatch.setattr(
        subject.streaming_stt_eval,
        "_worker_report",
        lambda _manifest, binding, runtime, *_args, **_kwargs: {
            **binding,
            "runtime": dict(runtime),
        },
    )
    monkeypatch.setattr(
        subject,
        "aggregate_metrics",
        lambda *_args, **_kwargs: {"coverage_complete": True},
    )
    return _Harness(
        snapshot=snapshot,
        manifest=manifest,
        scratch_parent=scratch_parent,
        output_parent=output_parent,
        verify_events=verify_events,
        lock_calls=lock_calls,
        snapshot_calls=snapshot_calls,
        closure_calls=closure_calls,
        manifest_calls=manifest_calls,
    )


def _trace(item: subject._PrivateInput, *, final: object | None = None) -> CaseTrace:
    selected = final or FinalEvent(
        request_id="private-request",
        seq=0,
        text=(
            f"{item.reference_a} {item.reference_b}"
            if item.kind == "mix"
            else item.reference_a
            if item.kind == "role_a"
            else item.reference_b
        ),
        samples_seen=item.samples,
        elapsed_ms=1.0,
        finalization_ms=1.0,
        compute_ms=1.0,
        audio_seconds=item.samples / 16_000,
        chunks=1,
        deadline_misses=0,
        max_backlog_ms=0.0,
        resources=ResourceUsage(rss_mb=1.0, threads=1, vram_mb=None),
    )
    return CaseTrace(partials=(), final=selected)


def _records(snapshot: subject._BundleSnapshot) -> list[RunRecord]:
    return [
        RunRecord(case_index=index, repeat=repeat, trace=_trace(item))
        for repeat in range(3)
        for index, item in enumerate(snapshot.inputs)
    ]


def _lifecycle_failure(kind: str) -> BaseException:
    if kind == "keyboard":
        return KeyboardInterrupt()
    assert kind == "term"
    return subject._LifecycleSignal(getattr(subject.signal, "SIGTERM", 15))


def _inject_lifecycle_failure(
    monkeypatch: pytest.MonkeyPatch,
    point: str,
    failure: BaseException,
) -> None:
    if point.startswith("bundle_"):
        original = subject._snapshot_bundle
        target = 2 if point == "bundle_post" else 4
        calls = 0

        def snapshot(*args: object, **kwargs: object) -> subject._BundleSnapshot:
            nonlocal calls
            calls += 1
            if calls == target:
                raise failure
            return original(*args, **kwargs)

        monkeypatch.setattr(subject, "_snapshot_bundle", snapshot)
        return
    if point.startswith("scratch_"):
        original = subject._scratch_binding
        target = 1 if point == "scratch_post" else 2
        calls = 0

        def scratch(*args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            if calls == target:
                raise failure
            return original(*args, **kwargs)

        monkeypatch.setattr(subject, "_scratch_binding", scratch)
        return
    assert point == "cleanup"

    def cleanup(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(subject.streaming_stt_eval, "_remove_private_scratch", cleanup)


def _inject_lifecycle_after_scratch_prepare(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> tuple[
    list[tuple[Path, tuple[int, int]]],
    list[tuple[Path, tuple[int, int]]],
]:
    created: list[tuple[Path, tuple[int, int]]] = []
    removed: list[tuple[Path, tuple[int, int]]] = []
    delivered = False
    real_prepare = subject.streaming_stt_eval._prepare_scratch
    real_remove = subject.streaming_stt_eval._remove_private_scratch

    def prepare(parent: Path) -> Path:
        path = real_prepare(parent)
        metadata = path.lstat()
        created.append((path, (metadata.st_dev, metadata.st_ino)))
        return path

    def remove(path: Path, *, expected_identity: tuple[int, int]) -> None:
        removed.append((path, expected_identity))
        real_remove(path, expected_identity=expected_identity)

    def signal_mask(how: object, _mask: object) -> set[int]:
        nonlocal delivered
        if created and not delivered and how == subject.signal.SIG_SETMASK:
            delivered = True
            raise failure
        return set()

    monkeypatch.setattr(subject.streaming_stt_eval, "_prepare_scratch", prepare)
    monkeypatch.setattr(subject.streaming_stt_eval, "_remove_private_scratch", remove)
    monkeypatch.setattr(subject.signal, "pthread_sigmask", signal_mask)
    return created, removed


def _install_mask_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[set[int], list[tuple[object, frozenset[int]]]]:
    blocked: set[int] = set()
    operations: list[tuple[object, frozenset[int]]] = []

    def signal_mask(how: object, mask: object) -> set[int]:
        requested = set(mask)
        previous = set(blocked)
        if how == subject.signal.SIG_BLOCK:
            blocked.update(requested)
        elif how == subject.signal.SIG_SETMASK:
            blocked.clear()
            blocked.update(requested)
        else:
            raise AssertionError
        operations.append((how, frozenset(requested)))
        return previous

    monkeypatch.setattr(subject.signal, "pthread_sigmask", signal_mask)
    return blocked, operations


def _inject_lifecycle_after_worker_exit(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> tuple[list[_FakeWorker], list[_FakeWorker], set[int]]:
    blocked, _operations = _install_mask_tracker(monkeypatch)
    tracked_mask = subject.signal.pthread_sigmask
    expected = set(subject._commit_signal_numbers())
    exited: list[_FakeWorker] = []
    delivered: list[_FakeWorker] = []
    real_exit = _FakeWorker.__exit__

    def worker_exit(worker: _FakeWorker, *args: object) -> None:
        assert expected <= blocked
        real_exit(worker, *args)
        assert worker.fully_stopped is True
        exited.append(worker)

    def signal_mask(how: object, mask: object) -> set[int]:
        before = set(blocked)
        previous = tracked_mask(how, mask)
        if exited and not delivered and how == subject.signal.SIG_SETMASK:
            assert expected <= before
            assert exited[-1].fully_stopped is True
            delivered.append(exited[-1])
            raise failure
        return previous

    monkeypatch.setattr(_FakeWorker, "__exit__", worker_exit)
    monkeypatch.setattr(subject.signal, "pthread_sigmask", signal_mask)
    return exited, delivered, blocked


def _rebind(report: dict[str, object]) -> None:
    report["binding_sha256"] = subject._canonical_sha256(
        {key: value for key, value in report.items() if key != "binding_sha256"}
    )


def test_evaluator_lock_is_exact_self_digested_and_alias_rejected(
    tmp_path: Path,
) -> None:
    loaded = subject._load_lock()

    assert loaded.raw_sha256 == subject.LOCK_FILE_SHA256
    assert loaded.recipe_sha256 == subject.LOCK_RECIPE_SHA256
    assert loaded.manifest_sha256 == (
        "e69c85356a77cd87b0c7f4c100614dccbf1484e16532a5a2239a768162e443b4"
    )
    assert loaded.receipt_sha256 == (
        "69ec7df35591a2b1a7203109c58077540540c32deec99dc933a5ddef0840f507"
    )

    alias = tmp_path / subject.DEFAULT_LOCK.name
    alias.symlink_to(subject.DEFAULT_LOCK)
    with pytest.raises(subject.Primock57OverlapEvalError):
        subject._load_lock(alias)


def test_lock_rejects_byte_drift_and_duplicate_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = subject.DEFAULT_LOCK.read_bytes()
    monkeypatch.setattr(
        subject,
        "read_regular_bounded",
        lambda *_args, **_kwargs: SimpleNamespace(data=raw + b" "),
    )
    with pytest.raises(subject.Primock57OverlapEvalError):
        subject._load_lock()

    with pytest.raises(subject.Primock57OverlapEvalError):
        subject._strict_json_bytes(b'{"a":1,"a":1}', maximum_bytes=100)


def test_preflight_reopens_exact_snapshot_and_is_aggregate_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _bundle_snapshot(tmp_path)
    calls = 0

    def reopen(*_args: object) -> subject._BundleSnapshot:
        nonlocal calls
        calls += 1
        return snapshot

    monkeypatch.setattr(subject, "_load_lock", lambda: _fake_lock())
    monkeypatch.setattr(subject, "_closure_sha256", lambda: "9" * 64)
    monkeypatch.setattr(subject, "_snapshot_bundle", reopen)
    report = subject.preflight_primock57_overlap_bundle(snapshot.bundle.path.parent)

    assert calls == 2
    assert report == {
        "fixture_id": fixture.FIXTURE_ID,
        "cases": 3,
        "pcm_inputs": 9,
        "manifest_sha256": snapshot.bundle.digest,
        "receipt_sha256": snapshot.bundle.receipt_sha256,
        "source_contract_sha256": snapshot.bundle.source_contract_sha256,
        "evaluator_lock_recipe_sha256": "b" * 64,
        "wrapper_closure_sha256": "9" * 64,
    }
    encoded = json.dumps(report, sort_keys=True)
    assert str(tmp_path) not in encoded
    for item in snapshot.inputs:
        assert item.reference_a not in encoded
        assert item.reference_b not in encoded
        assert item.path.name not in encoded


@pytest.mark.parametrize("surface", ["snapshot", "closure"])
def test_preflight_rejects_retained_snapshot_or_closure_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    snapshot = _bundle_snapshot(tmp_path)
    changed = replace(snapshot, directory_snapshot=(9, 9, 9))
    snapshots = iter((snapshot, changed if surface == "snapshot" else snapshot))
    closures = iter(("9" * 64, "8" * 64 if surface == "closure" else "9" * 64))
    monkeypatch.setattr(subject, "_load_lock", lambda: _fake_lock())
    monkeypatch.setattr(subject, "_closure_sha256", lambda: next(closures))
    monkeypatch.setattr(subject, "_snapshot_bundle", lambda *_args: next(snapshots))
    with pytest.raises(subject.Primock57OverlapEvalError):
        subject.preflight_primock57_overlap_bundle(snapshot.bundle.path.parent)


def test_retained_snapshot_uses_generated_private_tree_and_detects_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, _isolated, overlap, _prepared = _prepare(tmp_path)
    real_load = fixture.load_primock57_overlap_bundle
    loaded = real_load(overlap)
    evaluator_lock = subject._EvaluatorLock(
        raw_sha256="a" * 64,
        recipe_sha256="b" * 64,
        manifest_sha256=loaded.digest,
        receipt_sha256=loaded.receipt_sha256,
    )

    def production_load(path: Path | str) -> fixture.LoadedPrimock57OverlapBundle:
        return replace(real_load(path), production_evidence=True)

    monkeypatch.setattr(fixture, "load_primock57_overlap_bundle", production_load)
    first = subject._snapshot_bundle(overlap, evaluator_lock)

    assert len(first.inputs) == 9
    assert len(first.file_snapshots) == 11
    assert (
        tuple(item.kind for item in first.inputs)
        == (
            "role_a",
            "role_b",
            "mix",
        )
        * 3
    )
    assert (
        len({(item.snapshot[0], item.snapshot[1]) for item in first.file_snapshots})
        == 11
    )

    target = first.inputs[0].path
    replacement = target.with_name("replacement.f32le")
    replacement.write_bytes(target.read_bytes())
    replacement.chmod(0o600)
    os.replace(replacement, target)
    second = subject._snapshot_bundle(overlap, evaluator_lock)
    assert subject._same_bundle(first, second) is False


def test_retained_snapshot_rejects_permission_and_symlink_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, _isolated, overlap, _prepared = _prepare(tmp_path)
    real_load = fixture.load_primock57_overlap_bundle
    loaded = real_load(overlap)
    evaluator_lock = subject._EvaluatorLock(
        raw_sha256="a" * 64,
        recipe_sha256="b" * 64,
        manifest_sha256=loaded.digest,
        receipt_sha256=loaded.receipt_sha256,
    )
    monkeypatch.setattr(
        fixture,
        "load_primock57_overlap_bundle",
        lambda path: replace(real_load(path), production_evidence=True),
    )
    target = overlap / "primock57-overlap-00-role-a.f32le"
    target.chmod(0o644)
    with pytest.raises(subject.Primock57OverlapEvalError):
        subject._snapshot_bundle(overlap, evaluator_lock)

    target.chmod(0o600)
    alias = tmp_path / "overlap-alias"
    alias.symlink_to(overlap, target_is_directory=True)
    with pytest.raises(subject.Primock57OverlapEvalError):
        subject._snapshot_bundle(alias, evaluator_lock)


def test_fake_worker_runs_one_process_in_exact_27_request_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)

    report = subject.run_primock57_overlap_eval(
        harness.manifest.path,
        harness.snapshot.bundle.path.parent,
        scratch_parent=harness.scratch_parent,
    )

    assert len(_FakeWorker.instances) == 1
    assert len(_FakeWorker.requests) == 27
    assert [request.request_id for request in _FakeWorker.requests] == [
        f"overlap-{index:02d}-r{repeat}" for repeat in range(3) for index in range(9)
    ]
    assert [request.pcm.sha256 for request in _FakeWorker.requests] == [
        item.sha256 for _repeat in range(3) for item in harness.snapshot.inputs
    ]
    assert [request.pcm.path.name for request in _FakeWorker.requests] == [
        f"input-{index:02d}.f32le" for _repeat in range(3) for index in range(9)
    ]
    assert _FakeWorker.instances[0].fully_stopped is True
    assert len(_FakeRunLock.instances) == 1
    assert _FakeRunLock.instances[0].closed is True
    assert report["metrics"] == {
        "coverage": {
            "cases": 3,
            "pcm_inputs": 9,
            "repeats": 3,
            "evaluations": 27,
            "source_complete_evaluations": 27,
        },
        "stem_accuracy": {
            "evaluations": 18,
            "substitutions": 0,
            "insertions": 0,
            "deletions": 0,
            "word_errors": 0,
            "reference_words": 90,
            "hypothesis_words": 90,
            "wer": 0.0,
        },
        "mix_accuracy": {
            "protocol": "two-utterance-min-order-wer-v1",
            "evaluations": 9,
            "substitutions": 0,
            "insertions": 0,
            "deletions": 0,
            "word_errors": 0,
            "reference_words": 90,
            "hypothesis_words": 90,
            "wer": 0.0,
        },
        "mix_minus_stem_wer": 0.0,
        "repeat_disagreement_inputs": 0,
    }
    encoded = json.dumps(report, sort_keys=True)
    assert str(tmp_path) not in encoded
    assert "role_a" not in encoded
    assert "role_b" not in encoded
    assert "primock57-overlap-00" not in encoded
    for item in harness.snapshot.inputs:
        assert item.reference_a not in encoded
        assert item.reference_b not in encoded
        assert item.path.name not in encoded


def test_end_to_end_fake_run_publishes_exact_report_then_removes_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    output = harness.output_parent / "aggregate-report.json"
    state = subject._ReportCommitState()

    report = subject.run_primock57_overlap_eval(
        harness.manifest.path,
        harness.snapshot.bundle.path.parent,
        scratch_parent=harness.scratch_parent,
        output_path=output,
        _commit_state=state,
    )

    expected = subject._canonical_json(report, newline=True)
    assert state.committed is True
    assert output.read_bytes() == expected
    assert state.digest == hashlib.sha256(expected).hexdigest()
    assert stat.S_IMODE(output.lstat().st_mode) == 0o600
    assert output.lstat().st_nlink == 1
    assert list(harness.scratch_parent.iterdir()) == []
    assert report["evidence"]["model_executed"] is False
    assert len(harness.snapshot_calls) >= 3
    assert len(harness.closure_calls) >= 3
    assert len(harness.manifest_calls) >= 3
    assert len(harness.lock_calls) >= 3
    assert len(harness.verify_events) == 2


@pytest.mark.parametrize(
    ("surface", "stage"),
    [
        (surface, stage)
        for surface in ("bundle", "lock", "closure", "manifest", "source")
        for stage in ("post", "commit")
    ]
    + [("worker_input", "post")],
)
def test_mutation_at_post_or_commit_stage_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    surface: str,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    output = harness.output_parent / "aggregate-report.json"
    changed_snapshot = replace(harness.snapshot, directory_snapshot=(7, 8, 9))
    changed_manifest = SimpleNamespace(**vars(harness.manifest))
    changed_manifest.digest = "0" * 64
    # Retained identities are opened initially, after the worker, once before
    # teardown, and finally inside the O_TMPFILE publication guard.
    target_call = 2 if stage == "post" else 4

    if surface == "bundle":
        calls = 0

        def bundle(*_args: object) -> subject._BundleSnapshot:
            nonlocal calls
            calls += 1
            return changed_snapshot if calls >= target_call else harness.snapshot

        monkeypatch.setattr(subject, "_snapshot_bundle", bundle)
    elif surface == "lock":
        calls = 0

        def evaluator_lock() -> subject._EvaluatorLock:
            nonlocal calls
            calls += 1
            selected = _fake_lock()
            return (
                replace(selected, raw_sha256="0" * 64)
                if calls >= target_call
                else selected
            )

        monkeypatch.setattr(subject, "_load_lock", evaluator_lock)
    elif surface == "closure":
        calls = 0

        def closure() -> str:
            nonlocal calls
            calls += 1
            return "0" * 64 if calls >= target_call else "2" * 64

        monkeypatch.setattr(subject, "_closure_sha256", closure)
    elif surface == "manifest":
        calls = 0

        def manifest(_path: object) -> SimpleNamespace:
            nonlocal calls
            calls += 1
            return changed_manifest if calls >= target_call else harness.manifest

        monkeypatch.setattr(
            subject.streaming_stt_eval, "load_worker_manifest", manifest
        )
    elif surface == "source":
        calls = 0

        def source(*_args: object, **_kwargs: object) -> None:
            nonlocal calls
            calls += 1
            if calls >= (1 if stage == "post" else 2):
                raise RuntimeError("private source detail")

        monkeypatch.setattr(subject, "verify_source_bundle", source)
    else:
        assert stage == "post"

        def mutate_on_close(worker: _FakeWorker, *_exc: object) -> None:
            worker.fully_stopped = True
            (worker.scratch / "input-00.f32le").write_bytes(b"\x00\x00\x00\x00")

        monkeypatch.setattr(_FakeWorker, "__exit__", mutate_on_close)

    with pytest.raises(subject.Primock57OverlapEvalError) as error:
        subject.run_primock57_overlap_eval(
            harness.manifest.path,
            harness.snapshot.bundle.path.parent,
            scratch_parent=harness.scratch_parent,
            output_path=output,
            _commit_state=subject._ReportCommitState(),
        )
    assert str(error.value) == ""
    assert not output.exists()


@pytest.mark.parametrize("failure", ["worker", "not_stopped", "scratch_cleanup"])
def test_worker_or_cleanup_failure_is_detail_free_and_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    output = harness.output_parent / "aggregate-report.json"
    if failure == "worker":

        def fail(_self: object, _request: object) -> object:
            raise RuntimeError("private worker failure")

        monkeypatch.setattr(_FakeWorker, "transcribe", fail)
    elif failure == "not_stopped":
        monkeypatch.setattr(
            _FakeWorker,
            "__exit__",
            lambda _self, *_exc: None,
        )
    else:
        monkeypatch.setattr(
            subject.streaming_stt_eval,
            "_remove_private_scratch",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("private cleanup failure")
            ),
        )

    with pytest.raises(subject.Primock57OverlapEvalError) as error:
        subject.run_primock57_overlap_eval(
            harness.manifest.path,
            harness.snapshot.bundle.path.parent,
            scratch_parent=harness.scratch_parent,
            output_path=output,
            _commit_state=subject._ReportCommitState(),
        )
    assert str(error.value) == ""
    assert not output.exists()


@pytest.mark.parametrize("terminal_kind", ["native", "parakeet_cpp", "plain"])
def test_metrics_reject_any_early_or_incomplete_source_terminal(
    tmp_path: Path,
    terminal_kind: str,
) -> None:
    snapshot = _bundle_snapshot(tmp_path)
    records = _records(snapshot)
    item = snapshot.inputs[0]
    usage = ResourceUsage(rss_mb=1.0, threads=1, vram_mb=None)
    if terminal_kind == "native":
        final = NativeFinalEvent(
            request_id="private-request",
            seq=0,
            text=item.reference_a,
            samples_seen=0,
            elapsed_ms=1.0,
            finalization_ms=1.0,
            compute_ms=1.0,
            audio_seconds=item.samples / 16_000,
            chunks=1,
            deadline_misses=0,
            max_backlog_ms=0.0,
            resources=usage,
            model_padding_samples=0,
            source_samples=item.samples,
            source_samples_consumed=0,
            declared_tail_samples=0,
            tail_samples_consumed=0,
            endpoint_reason="eou",
            native_endpoint=True,
            endpoint_probability=1.0,
            endpoint_sample=0,
            endpoint_latency_ms=0.0,
            authoritative=False,
        )
    elif terminal_kind == "parakeet_cpp":
        final = ParakeetCppFinalEvent(
            request_id="private-request",
            seq=0,
            text=item.reference_a,
            samples_seen=0,
            elapsed_ms=1.0,
            finalization_ms=1.0,
            compute_ms=1.0,
            audio_seconds=item.samples / 16_000,
            chunks=1,
            deadline_misses=0,
            max_backlog_ms=0.0,
            resources=usage,
            model_padding_samples=0,
            source_samples_offered=item.samples,
            source_samples_consumed=0,
            tail_samples_offered=0,
            tail_samples_consumed=0,
            observed_events=(),
            endpoint_reason="tail_exhausted",
            native_endpoint=False,
            encoder_frame=None,
            encoder_time_seconds=None,
            event_origin="none",
            endpoint_sample=None,
            endpoint_latency_ms=None,
            authoritative=False,
            protocol_version=PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION,
        )
    else:
        final = FinalEvent(
            request_id="private-request",
            seq=0,
            text=item.reference_a,
            samples_seen=0,
            elapsed_ms=1.0,
            finalization_ms=1.0,
            compute_ms=1.0,
            audio_seconds=0.0,
            chunks=1,
            deadline_misses=0,
            max_backlog_ms=0.0,
            resources=usage,
        )
    records[0] = replace(records[0], trace=_trace(item, final=final))

    with pytest.raises(subject.Primock57OverlapEvalError):
        subject._overlap_metrics(snapshot, records)


def test_private_path_aliases_and_overlapping_roots_are_rejected(
    tmp_path: Path,
) -> None:
    real = _private_dir(tmp_path / "real-private")
    alias = tmp_path / "private-alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(subject.Primock57OverlapEvalError):
        subject._stable_existing_directory(alias)

    snapshot = _bundle_snapshot(tmp_path / "bundle-scope")
    manifest = _manifest(tmp_path / "manifest-scope")
    with pytest.raises(subject.Primock57OverlapEvalError):
        subject._validate_paths(
            initial=snapshot,
            manifest=manifest,
            scratch_parent=snapshot.bundle.path.parent,
            output_path=None,
        )


@pytest.mark.parametrize(
    "protected_name",
    ["bundle", "manifest", "python", "artifact"],
)
@pytest.mark.parametrize("destination", ["scratch", "output"])
def test_every_protected_tree_is_rejected_for_scratch_and_output(
    tmp_path: Path,
    protected_name: str,
    destination: str,
) -> None:
    snapshot = _bundle_snapshot(tmp_path / "bundle-scope")
    manifest = _manifest(tmp_path / "manifest-scope")
    protected = {
        "bundle": snapshot.bundle.path.parent,
        "manifest": manifest.path.parent,
        "python": manifest.python.path.parent,
        "artifact": manifest.artifacts[0].path.parent,
    }[protected_name]
    safe = _private_dir(tmp_path / "safe-private")
    scratch = protected if destination == "scratch" else safe
    output = protected / "new-report.json" if destination == "output" else None

    with pytest.raises(subject.Primock57OverlapEvalError):
        subject._validate_paths(
            initial=snapshot,
            manifest=manifest,
            scratch_parent=scratch,
            output_path=output,
        )


def test_scratch_output_overlap_and_git_ancestor_are_rejected(tmp_path: Path) -> None:
    snapshot = _bundle_snapshot(tmp_path / "bundle-scope")
    manifest = _manifest(tmp_path / "manifest-scope")
    scratch = _private_dir(tmp_path / "scratch")
    with pytest.raises(subject.Primock57OverlapEvalError):
        subject._validate_paths(
            initial=snapshot,
            manifest=manifest,
            scratch_parent=scratch,
            output_path=scratch / "report.json",
        )

    output_parent = _private_dir(tmp_path / "output-root")
    nested_scratch = _private_dir(output_parent / "nested-scratch")
    with pytest.raises(subject.Primock57OverlapEvalError):
        subject._validate_paths(
            initial=snapshot,
            manifest=manifest,
            scratch_parent=nested_scratch,
            output_path=output_parent / "report.json",
        )

    git_root = _private_dir(tmp_path / "git-root")
    marker = _private_dir(git_root / ".git")
    _private_file(marker / "HEAD", b"ref: refs/heads/main\n")
    git_scratch = _private_dir(git_root / "scratch")
    with pytest.raises(subject.Primock57OverlapEvalError):
        subject._validate_paths(
            initial=snapshot,
            manifest=manifest,
            scratch_parent=git_scratch,
            output_path=None,
        )
    safe_scratch = _private_dir(tmp_path / "safe-scratch")
    git_output = _private_dir(git_root / "output")
    with pytest.raises(subject.Primock57OverlapEvalError):
        subject._validate_paths(
            initial=snapshot,
            manifest=manifest,
            scratch_parent=safe_scratch,
            output_path=git_output / "report.json",
        )


def test_private_report_publisher_is_single_link_mode_600_and_no_clobber(
    tmp_path: Path,
) -> None:
    parent = _private_dir(tmp_path / "private-output")
    output = parent / "report.json"
    report = {"ok": True, "aggregate": {"evaluations": 27}}
    state = subject._ReportCommitState()

    digest = subject._publish_report(
        output,
        report,
        commit_guard=lambda: True,
        state=state,
    )

    metadata = output.lstat()
    assert state.committed is True
    assert state.digest == digest
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    assert metadata.st_nlink == 1
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    with pytest.raises(subject.Primock57OverlapEvalError):
        subject._publish_report(
            output,
            report,
            commit_guard=lambda: True,
            state=subject._ReportCommitState(),
        )
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest


def test_terminal_commit_mask_contains_interrupt_hangup_and_termination() -> None:
    assert set(subject._commit_signal_numbers()) >= {
        getattr(subject.signal, "SIGINT", 2),
        getattr(subject.signal, "SIGHUP", 1),
        getattr(subject.signal, "SIGTERM", 15),
    }


def test_worker_enter_and_transcribe_are_unmasked_but_exit_is_masked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    blocked, _operations = _install_mask_tracker(monkeypatch)
    expected = set(subject._commit_signal_numbers())
    observations: list[str] = []
    real_enter = _FakeWorker.__enter__
    real_transcribe = _FakeWorker.transcribe
    real_exit = _FakeWorker.__exit__

    def worker_enter(worker: _FakeWorker) -> _FakeWorker:
        assert expected.isdisjoint(blocked)
        observations.append("enter-unmasked")
        return real_enter(worker)

    def transcribe(worker: _FakeWorker, request: object) -> CaseTrace:
        assert expected.isdisjoint(blocked)
        observations.append("transcribe-unmasked")
        return real_transcribe(worker, request)

    def worker_exit(worker: _FakeWorker, *args: object) -> None:
        assert expected <= blocked
        observations.append("exit-masked")
        real_exit(worker, *args)

    monkeypatch.setattr(_FakeWorker, "__enter__", worker_enter)
    monkeypatch.setattr(_FakeWorker, "transcribe", transcribe)
    monkeypatch.setattr(_FakeWorker, "__exit__", worker_exit)
    report = subject.run_primock57_overlap_eval(
        harness.manifest.path,
        harness.snapshot.bundle.path.parent,
        scratch_parent=harness.scratch_parent,
    )

    assert report["ok"] is True
    assert observations[0] == "enter-unmasked"
    assert observations[1:28] == ["transcribe-unmasked"] * 27
    assert observations[-1] == "exit-masked"
    assert blocked == set()


@pytest.mark.parametrize(
    "signum",
    [
        getattr(subject.signal, "SIGINT", 2),
        getattr(subject.signal, "SIGHUP", 1),
        getattr(subject.signal, "SIGTERM", 15),
    ],
)
def test_fake_popen_lifecycle_is_deferred_until_child_is_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signum: int,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    output = harness.output_parent / "aggregate-report.json"
    blocked, _operations = _install_mask_tracker(monkeypatch)
    expected = set(subject._commit_signal_numbers())
    previous_handlers = {number: object() for number in expected}
    handlers = dict(previous_handlers)
    restore_masks: list[set[int]] = []

    def install_handler(number: int, handler: object) -> object:
        previous = handlers[number]
        handlers[number] = handler
        if handler is previous_handlers[number]:
            restore_masks.append(set(blocked))
        return previous

    monkeypatch.setattr(subject.signal, "signal", install_handler)

    class PopenWorker(_FakeWorker):
        def __init__(
            self,
            manifest: object,
            scratch: Path,
            source_bundle: object,
        ) -> None:
            super().__init__(manifest, scratch, source_bundle)
            self._process: object | None = None
            self.child: SimpleNamespace | None = None
            self.spawn_masks: list[set[int]] = []
            self.close_masks: list[set[int]] = []
            self._popen_factory = self._spawn

        def _spawn(self, *_args: object, **_kwargs: object) -> object:
            self.spawn_masks.append(set(blocked))
            self.child = SimpleNamespace(stopped=False, group_stopped=False)
            handler = handlers[signum]
            assert callable(handler)
            handler(signum, None)
            return self.child

        def __enter__(self) -> PopenWorker:
            self._process = self._popen_factory()
            return super().__enter__()

        def close(self) -> None:
            self.close_calls += 1
            self.close_masks.append(set(blocked))
            assert self.child is not None
            assert self._process is self.child
            self.child.stopped = True
            self.child.group_stopped = True
            self.fully_stopped = True

    monkeypatch.setattr(subject, "StreamingWorker", PopenWorker)
    expected_exception = (
        KeyboardInterrupt
        if signum == getattr(subject.signal, "SIGINT", 2)
        else subject._LifecycleSignal
    )
    with pytest.raises(expected_exception) as caught:
        subject.run_primock57_overlap_eval(
            harness.manifest.path,
            harness.snapshot.bundle.path.parent,
            scratch_parent=harness.scratch_parent,
            output_path=output,
            _commit_state=subject._ReportCommitState(),
        )

    worker = _FakeWorker.instances[0]
    assert isinstance(worker, PopenWorker)
    assert worker.spawn_masks == [set()]
    assert worker.close_masks == [expected]
    assert worker.close_calls == 1
    assert worker.fully_stopped is True
    assert worker.child is not None and worker.child.stopped is True
    assert worker.child.group_stopped is True
    assert worker._process is worker.child
    assert blocked == set()
    assert len(restore_masks) == len(expected)
    assert all(expected <= mask for mask in restore_masks)
    assert all(handlers[number] is previous_handlers[number] for number in expected)
    if expected_exception is subject._LifecycleSignal:
        assert caught.value.signum == signum
    assert not output.exists()
    assert list(harness.scratch_parent.iterdir()) == []


@pytest.mark.parametrize(
    "failure",
    [
        KeyboardInterrupt(),
        subject._LifecycleSignal(getattr(subject.signal, "SIGTERM", 15)),
        subject._LifecycleSignal(getattr(subject.signal, "SIGHUP", 1)),
    ],
)
def test_pre_link_interrupt_leaves_no_report(
    tmp_path: Path,
    failure: BaseException,
) -> None:
    output = _private_dir(tmp_path / "private-output") / "report.json"
    state = subject._ReportCommitState()

    def interrupt() -> bool:
        raise failure

    with pytest.raises(type(failure)):
        subject._publish_report(
            output,
            {"ok": True},
            commit_guard=interrupt,
            state=state,
        )
    assert state.committed is False
    assert not output.exists()


@pytest.mark.parametrize(
    "signum",
    [
        getattr(subject.signal, "SIGINT", 2),
        getattr(subject.signal, "SIGHUP", 1),
        getattr(subject.signal, "SIGTERM", 15),
    ],
)
def test_real_os_lifecycle_from_guard_interrupts_before_link(
    tmp_path: Path,
    signum: int,
) -> None:
    output = _private_dir(tmp_path / "private-output") / "report.json"
    state = subject._ReportCommitState()
    expected_mask = set(subject._commit_signal_numbers())
    guard_masks: list[set[int]] = []
    expected_exception = (
        KeyboardInterrupt
        if signum == getattr(subject.signal, "SIGINT", 2)
        else subject._LifecycleSignal
    )

    def interrupt_guard() -> bool:
        guard_masks.append(
            set(
                subject.signal.pthread_sigmask(
                    subject.signal.SIG_BLOCK,
                    (),
                )
            )
        )
        os.kill(os.getpid(), signum)
        raise AssertionError("lifecycle signal was not delivered")

    with _real_lifecycle_delivery(signum) as delivered:
        with pytest.raises(expected_exception) as caught:
            subject._publish_report(
                output,
                {"ok": True},
                commit_guard=interrupt_guard,
                state=state,
            )

    assert delivered == [signum]
    assert len(guard_masks) == 1 and expected_mask.isdisjoint(guard_masks[0])
    if expected_exception is subject._LifecycleSignal:
        assert caught.value.signum == signum
    assert state.committed is False
    assert state.digest == ""
    assert not output.exists()


@pytest.mark.parametrize(
    "signum",
    [
        getattr(subject.signal, "SIGINT", 2),
        getattr(subject.signal, "SIGHUP", 1),
        getattr(subject.signal, "SIGTERM", 15),
    ],
)
def test_real_os_lifecycle_during_final_parent_validation_is_prelink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signum: int,
) -> None:
    output = _private_dir(tmp_path / "private-output") / "report.json"
    state = subject._ReportCommitState()
    expected_exception = (
        KeyboardInterrupt
        if signum == getattr(subject.signal, "SIGINT", 2)
        else subject._LifecycleSignal
    )
    guard_complete = False
    delivered_from_validation = False
    expected_mask = set(subject._commit_signal_numbers())
    validation_masks: list[set[int]] = []
    real_snapshot = subject.fixture._file_snapshot

    def commit_guard() -> bool:
        nonlocal guard_complete
        guard_complete = True
        return True

    def snapshot(metadata: os.stat_result) -> tuple[int, ...]:
        nonlocal delivered_from_validation
        if guard_complete and not delivered_from_validation:
            delivered_from_validation = True
            validation_masks.append(
                set(
                    subject.signal.pthread_sigmask(
                        subject.signal.SIG_BLOCK,
                        (),
                    )
                )
            )
            os.kill(os.getpid(), signum)
            raise AssertionError("lifecycle signal was not delivered")
        return real_snapshot(metadata)

    monkeypatch.setattr(subject.fixture, "_file_snapshot", snapshot)
    with _real_lifecycle_delivery(signum) as delivered:
        with pytest.raises(expected_exception) as caught:
            subject._publish_report(
                output,
                {"ok": True},
                commit_guard=commit_guard,
                state=state,
            )

    assert guard_complete is True
    assert delivered_from_validation is True
    assert delivered == [signum]
    assert len(validation_masks) == 1
    assert expected_mask.isdisjoint(validation_masks[0])
    if expected_exception is subject._LifecycleSignal:
        assert caught.value.signum == signum
    assert state.committed is False
    assert state.digest == ""
    assert not output.exists()


@pytest.mark.parametrize(
    "failure",
    [
        KeyboardInterrupt(),
        subject._LifecycleSignal(getattr(subject.signal, "SIGTERM", 15)),
        subject._LifecycleSignal(getattr(subject.signal, "SIGHUP", 1)),
    ],
)
def test_interrupt_after_link_is_recovered_as_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    output = _private_dir(tmp_path / "private-output") / "report.json"
    state = subject._ReportCommitState()
    real_link = subject.os.link

    def link_then_interrupt(*args: object, **kwargs: object) -> None:
        real_link(*args, **kwargs)
        raise failure

    monkeypatch.setattr(subject.os, "link", link_then_interrupt)
    digest = subject._publish_report(
        output,
        {"ok": True},
        commit_guard=lambda: True,
        state=state,
    )

    assert state.committed is True
    assert state.digest == digest
    assert output.is_file()
    assert output.lstat().st_nlink == 1
    assert stat.S_IMODE(output.lstat().st_mode) == 0o600


@pytest.mark.parametrize(
    "signum",
    [
        getattr(subject.signal, "SIGINT", 2),
        getattr(subject.signal, "SIGHUP", 1),
        getattr(subject.signal, "SIGTERM", 15),
    ],
)
def test_real_pending_os_lifecycle_after_link_is_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signum: int,
) -> None:
    output = _private_dir(tmp_path / "private-output") / "report.json"
    real_link = subject.os.link
    expected_mask = set(subject._commit_signal_numbers())
    link_masks: list[set[int]] = []
    commit_masks: list[set[int]] = []

    class ObservedCommitState:
        def __init__(self) -> None:
            self._committed = False
            self._digest = ""

        @property
        def committed(self) -> bool:
            return self._committed

        @committed.setter
        def committed(self, value: bool) -> None:
            if value:
                commit_masks.append(
                    set(
                        subject.signal.pthread_sigmask(
                            subject.signal.SIG_BLOCK,
                            (),
                        )
                    )
                )
            self._committed = value

        @property
        def digest(self) -> str:
            return self._digest

        @digest.setter
        def digest(self, value: str) -> None:
            self._digest = value

    state = ObservedCommitState()

    def link_then_signal(*args: object, **kwargs: object) -> None:
        link_masks.append(
            set(
                subject.signal.pthread_sigmask(
                    subject.signal.SIG_BLOCK,
                    (),
                )
            )
        )
        real_link(*args, **kwargs)
        os.kill(os.getpid(), signum)

    monkeypatch.setattr(subject.os, "link", link_then_signal)
    with _real_lifecycle_delivery(signum) as delivered:
        digest = subject._publish_report(
            output,
            {"ok": True},
            commit_guard=lambda: True,
            state=state,
        )

    assert delivered == [signum]
    assert len(link_masks) == 1 and expected_mask <= link_masks[0]
    assert len(commit_masks) == 1 and expected_mask <= commit_masks[0]
    assert state.committed is True
    assert state.digest == digest
    assert output.is_file()
    assert output.lstat().st_nlink == 1
    assert stat.S_IMODE(output.lstat().st_mode) == 0o600


def test_run_lock_close_failure_after_terminal_commit_keeps_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    output = harness.output_parent / "aggregate-report.json"
    state = subject._ReportCommitState()

    def close_after_marking(lock: _FakeRunLock) -> None:
        lock.closed = True
        raise RuntimeError("private lock detail")

    monkeypatch.setattr(_FakeRunLock, "close", close_after_marking)
    report = subject.run_primock57_overlap_eval(
        harness.manifest.path,
        harness.snapshot.bundle.path.parent,
        scratch_parent=harness.scratch_parent,
        output_path=output,
        _commit_state=state,
    )

    assert report["ok"] is True
    assert state.committed is True
    assert output.is_file()


@pytest.mark.parametrize(
    "failure",
    [
        KeyboardInterrupt(),
        subject._LifecycleSignal(getattr(subject.signal, "SIGTERM", 15)),
    ],
)
def test_api_lifecycle_interrupt_is_preserved_after_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    output = harness.output_parent / "aggregate-report.json"

    def interrupt(_self: object, _request: object) -> object:
        raise failure

    monkeypatch.setattr(_FakeWorker, "transcribe", interrupt)
    with pytest.raises(type(failure)):
        subject.run_primock57_overlap_eval(
            harness.manifest.path,
            harness.snapshot.bundle.path.parent,
            scratch_parent=harness.scratch_parent,
            output_path=output,
            _commit_state=subject._ReportCommitState(),
        )
    assert not output.exists()
    assert list(harness.scratch_parent.iterdir()) == []
    assert _FakeWorker.instances[0].fully_stopped is True


@pytest.mark.parametrize("failure_kind", ["keyboard", "term"])
def test_api_worker_entry_lifecycle_closes_worker_and_removes_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    output = harness.output_parent / "aggregate-report.json"
    failure = _lifecycle_failure(failure_kind)
    blocked, _operations = _install_mask_tracker(monkeypatch)
    expected_mask = set(subject._commit_signal_numbers())
    real_enter = _FakeWorker.__enter__

    def enter_then_interrupt(worker: _FakeWorker) -> _FakeWorker:
        assert expected_mask.isdisjoint(blocked)
        real_enter(worker)
        raise failure

    monkeypatch.setattr(_FakeWorker, "__enter__", enter_then_interrupt)
    with pytest.raises(type(failure)) as caught:
        subject.run_primock57_overlap_eval(
            harness.manifest.path,
            harness.snapshot.bundle.path.parent,
            scratch_parent=harness.scratch_parent,
            output_path=output,
            _commit_state=subject._ReportCommitState(),
        )

    worker = _FakeWorker.instances[0]
    assert caught.value is failure
    assert worker.enter_started is True
    assert worker.close_calls == 1
    assert worker.fully_stopped is True
    assert not output.exists()
    assert list(harness.scratch_parent.iterdir()) == []


@pytest.mark.parametrize("failure_kind", ["keyboard", "term"])
def test_api_delayed_exit_lifecycle_arrives_only_after_worker_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    output = harness.output_parent / "aggregate-report.json"
    failure = _lifecycle_failure(failure_kind)
    exited, delivered, blocked = _inject_lifecycle_after_worker_exit(
        monkeypatch,
        failure,
    )

    with pytest.raises(type(failure)) as caught:
        subject.run_primock57_overlap_eval(
            harness.manifest.path,
            harness.snapshot.bundle.path.parent,
            scratch_parent=harness.scratch_parent,
            output_path=output,
            _commit_state=subject._ReportCommitState(),
        )

    worker = _FakeWorker.instances[0]
    assert caught.value is failure
    assert exited == [worker]
    assert delivered == [worker]
    assert worker.fully_stopped is True
    assert worker.close_calls == 0
    assert blocked == set()
    assert not output.exists()
    assert list(harness.scratch_parent.iterdir()) == []


@pytest.mark.parametrize("failure_kind", ["keyboard", "term"])
def test_api_lifecycle_immediately_after_scratch_prepare_removes_exact_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    output = harness.output_parent / "aggregate-report.json"
    failure = _lifecycle_failure(failure_kind)
    created, removed = _inject_lifecycle_after_scratch_prepare(
        monkeypatch,
        failure,
    )

    with pytest.raises(type(failure)) as caught:
        subject.run_primock57_overlap_eval(
            harness.manifest.path,
            harness.snapshot.bundle.path.parent,
            scratch_parent=harness.scratch_parent,
            output_path=output,
            _commit_state=subject._ReportCommitState(),
        )

    assert caught.value is failure
    assert len(created) == 1
    assert removed == created
    assert not output.exists()
    assert list(harness.scratch_parent.iterdir()) == []


@pytest.mark.parametrize(
    "point",
    ["bundle_post", "bundle_final", "scratch_post", "scratch_final", "cleanup"],
)
@pytest.mark.parametrize("failure_kind", ["keyboard", "term"])
def test_api_rebind_and_cleanup_lifecycle_interrupt_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    point: str,
    failure_kind: str,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    output = harness.output_parent / "aggregate-report.json"
    failure = _lifecycle_failure(failure_kind)
    _inject_lifecycle_failure(monkeypatch, point, failure)

    with pytest.raises(type(failure)) as caught:
        subject.run_primock57_overlap_eval(
            harness.manifest.path,
            harness.snapshot.bundle.path.parent,
            scratch_parent=harness.scratch_parent,
            output_path=output,
            _commit_state=subject._ReportCommitState(),
        )

    assert caught.value is failure
    assert not output.exists()
    if point != "cleanup":
        assert list(harness.scratch_parent.iterdir()) == []
    assert _FakeWorker.instances[0].fully_stopped is True


@pytest.mark.parametrize(
    ("failure", "expected_rc"),
    [
        (KeyboardInterrupt(), 130),
        (
            subject._LifecycleSignal(getattr(subject.signal, "SIGTERM", 15)),
            128 + getattr(subject.signal, "SIGTERM", 15),
        ),
        (
            subject._LifecycleSignal(getattr(subject.signal, "SIGHUP", 1)),
            128 + getattr(subject.signal, "SIGHUP", 1),
        ),
    ],
)
def test_integrated_cli_lifecycle_interrupt_preserves_rc_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    failure: BaseException,
    expected_rc: int,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    output = harness.output_parent / "aggregate-report.json"
    monkeypatch.setattr(subject, "_signal_numbers", lambda: ())

    def interrupt(_self: object, _request: object) -> object:
        raise failure

    monkeypatch.setattr(_FakeWorker, "transcribe", interrupt)
    code = subject.main(
        [
            "--worker-manifest",
            str(harness.manifest.path),
            "--overlap-bundle",
            str(harness.snapshot.bundle.path.parent),
            "--scratch-root",
            str(harness.scratch_parent),
            "--output",
            str(output),
        ]
    )

    assert code == expected_rc
    assert json.loads(capfd.readouterr().out) == subject._SAFE_ERROR
    assert not output.exists()
    assert list(harness.scratch_parent.iterdir()) == []
    assert _FakeWorker.instances[0].fully_stopped is True


@pytest.mark.parametrize("failure_kind", ["keyboard", "term"])
def test_integrated_cli_worker_entry_lifecycle_closes_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    failure_kind: str,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    output = harness.output_parent / "aggregate-report.json"
    failure = _lifecycle_failure(failure_kind)
    expected_rc = (
        130
        if failure_kind == "keyboard"
        else 128 + getattr(subject.signal, "SIGTERM", 15)
    )
    blocked, _operations = _install_mask_tracker(monkeypatch)
    expected_mask = set(subject._commit_signal_numbers())
    real_enter = _FakeWorker.__enter__

    def enter_then_interrupt(worker: _FakeWorker) -> _FakeWorker:
        assert expected_mask.isdisjoint(blocked)
        real_enter(worker)
        raise failure

    monkeypatch.setattr(_FakeWorker, "__enter__", enter_then_interrupt)
    code = subject.main(
        [
            "--worker-manifest",
            str(harness.manifest.path),
            "--overlap-bundle",
            str(harness.snapshot.bundle.path.parent),
            "--scratch-root",
            str(harness.scratch_parent),
            "--output",
            str(output),
        ]
    )

    worker = _FakeWorker.instances[0]
    assert code == expected_rc
    assert json.loads(capfd.readouterr().out) == subject._SAFE_ERROR
    assert worker.enter_started is True
    assert worker.close_calls == 1
    assert worker.fully_stopped is True
    assert not output.exists()
    assert list(harness.scratch_parent.iterdir()) == []


@pytest.mark.parametrize("failure_kind", ["keyboard", "term"])
def test_integrated_cli_delayed_exit_lifecycle_arrives_after_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    failure_kind: str,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    output = harness.output_parent / "aggregate-report.json"
    failure = _lifecycle_failure(failure_kind)
    expected_rc = (
        130
        if failure_kind == "keyboard"
        else 128 + getattr(subject.signal, "SIGTERM", 15)
    )
    exited, delivered, blocked = _inject_lifecycle_after_worker_exit(
        monkeypatch,
        failure,
    )

    code = subject.main(
        [
            "--worker-manifest",
            str(harness.manifest.path),
            "--overlap-bundle",
            str(harness.snapshot.bundle.path.parent),
            "--scratch-root",
            str(harness.scratch_parent),
            "--output",
            str(output),
        ]
    )

    worker = _FakeWorker.instances[0]
    assert code == expected_rc
    assert json.loads(capfd.readouterr().out) == subject._SAFE_ERROR
    assert exited == [worker]
    assert delivered == [worker]
    assert worker.fully_stopped is True
    assert worker.close_calls == 0
    assert blocked == set()
    assert not output.exists()
    assert list(harness.scratch_parent.iterdir()) == []


@pytest.mark.parametrize("failure_kind", ["keyboard", "term"])
def test_integrated_cli_lifecycle_after_scratch_prepare_removes_exact_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    failure_kind: str,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    output = harness.output_parent / "aggregate-report.json"
    failure = _lifecycle_failure(failure_kind)
    expected_rc = (
        130
        if failure_kind == "keyboard"
        else 128 + getattr(subject.signal, "SIGTERM", 15)
    )
    monkeypatch.setattr(subject, "_signal_numbers", lambda: ())
    created, removed = _inject_lifecycle_after_scratch_prepare(
        monkeypatch,
        failure,
    )

    code = subject.main(
        [
            "--worker-manifest",
            str(harness.manifest.path),
            "--overlap-bundle",
            str(harness.snapshot.bundle.path.parent),
            "--scratch-root",
            str(harness.scratch_parent),
            "--output",
            str(output),
        ]
    )

    assert code == expected_rc
    assert json.loads(capfd.readouterr().out) == subject._SAFE_ERROR
    assert len(created) == 1
    assert removed == created
    assert not output.exists()
    assert list(harness.scratch_parent.iterdir()) == []


@pytest.mark.parametrize(
    "point",
    ["bundle_post", "bundle_final", "scratch_post", "scratch_final", "cleanup"],
)
@pytest.mark.parametrize("failure_kind", ["keyboard", "term"])
def test_integrated_cli_rebind_and_cleanup_lifecycle_has_safe_rc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    point: str,
    failure_kind: str,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    output = harness.output_parent / "aggregate-report.json"
    failure = _lifecycle_failure(failure_kind)
    expected_rc = (
        130
        if failure_kind == "keyboard"
        else 128 + getattr(subject.signal, "SIGTERM", 15)
    )
    monkeypatch.setattr(subject, "_signal_numbers", lambda: ())
    _inject_lifecycle_failure(monkeypatch, point, failure)

    code = subject.main(
        [
            "--worker-manifest",
            str(harness.manifest.path),
            "--overlap-bundle",
            str(harness.snapshot.bundle.path.parent),
            "--scratch-root",
            str(harness.scratch_parent),
            "--output",
            str(output),
        ]
    )

    assert code == expected_rc
    assert json.loads(capfd.readouterr().out) == subject._SAFE_ERROR
    assert not output.exists()
    if point != "cleanup":
        assert list(harness.scratch_parent.iterdir()) == []
    assert _FakeWorker.instances[0].fully_stopped is True


@pytest.mark.parametrize(
    ("failure", "expected_rc"),
    [
        (KeyboardInterrupt(), 130),
        (subject._LifecycleSignal(getattr(subject.signal, "SIGTERM", 15)), 143),
        (RuntimeError("private detail"), 2),
    ],
)
def test_cli_precommit_failures_are_detail_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    failure: BaseException,
    expected_rc: int,
) -> None:
    monkeypatch.setattr(subject, "_signal_numbers", lambda: ())

    def fail(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(subject, "run_primock57_overlap_eval", fail)
    code = subject.main(
        [
            "--worker-manifest",
            str(tmp_path / "worker.json"),
            "--overlap-bundle",
            str(tmp_path / "bundle"),
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert code == expected_rc
    assert json.loads(capfd.readouterr().out) == subject._SAFE_ERROR


@pytest.mark.parametrize(
    "failure",
    [
        KeyboardInterrupt(),
        subject._LifecycleSignal(getattr(subject.signal, "SIGTERM", 15)),
        subject._LifecycleSignal(getattr(subject.signal, "SIGHUP", 1)),
    ],
)
def test_cli_postlink_interrupt_returns_success_without_safe_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    failure: BaseException,
) -> None:
    monkeypatch.setattr(subject, "_signal_numbers", lambda: ())

    def committed(*_args: object, **kwargs: object) -> object:
        state = kwargs["_commit_state"]
        state.committed = True
        state.digest = "a" * 64
        raise failure

    monkeypatch.setattr(subject, "run_primock57_overlap_eval", committed)
    code = subject.main(
        [
            "--worker-manifest",
            str(tmp_path / "worker.json"),
            "--overlap-bundle",
            str(tmp_path / "bundle"),
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert code == 0
    assert capfd.readouterr().out == ""


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "nan", "unsupported", "wrong_delta", "wrong_binding"],
)
def test_report_schema_rejects_malformed_and_nonfinite_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    report = subject.run_primock57_overlap_eval(
        harness.manifest.path,
        harness.snapshot.bundle.path.parent,
        scratch_parent=harness.scratch_parent,
    )
    if mutation == "missing":
        report.pop("config")
    elif mutation == "extra":
        report["debug"] = True
    elif mutation == "nan":
        metrics = dict(report["metrics"])
        metrics["nonfinite"] = float("nan")
        report["metrics"] = metrics
    elif mutation == "unsupported":
        metrics = dict(report["metrics"])
        metrics["unsupported"] = object()
        report["metrics"] = metrics
    elif mutation == "wrong_delta":
        metrics = dict(report["metrics"])
        metrics["mix_minus_stem_wer"] = 0.25
        report["metrics"] = metrics
    else:
        report["binding_sha256"] = "0" * 64
    if mutation in {"missing", "extra", "wrong_delta"}:
        _rebind(report)

    with pytest.raises(subject.Primock57OverlapEvalError) as error:
        subject._validate_report(report, initial=harness.snapshot)
    assert str(error.value) == ""


def test_report_rejects_self_consistent_impossible_reference_denominator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    report = subject.run_primock57_overlap_eval(
        harness.manifest.path,
        harness.snapshot.bundle.path.parent,
        scratch_parent=harness.scratch_parent,
    )
    metrics = dict(report["metrics"])
    for field in ("stem_accuracy", "mix_accuracy"):
        accuracy = dict(metrics[field])
        accuracy["reference_words"] += 1
        accuracy["hypothesis_words"] += 1
        assert accuracy["word_errors"] == (
            accuracy["substitutions"] + accuracy["insertions"] + accuracy["deletions"]
        )
        assert accuracy["hypothesis_words"] == (
            accuracy["reference_words"] - accuracy["deletions"] + accuracy["insertions"]
        )
        assert accuracy["wer"] == (
            accuracy["word_errors"] / accuracy["reference_words"]
        )
        metrics[field] = accuracy
    report["metrics"] = metrics
    _rebind(report)

    with pytest.raises(subject.Primock57OverlapEvalError) as error:
        subject._validate_report(report, initial=harness.snapshot)
    assert str(error.value) == ""


@pytest.mark.parametrize(
    "leak",
    [
        "/tmp/private/model",
        r"C:\\Users\\paul\\model",
        "alpha private utterance",
        "private-input-00.f32le",
        "role_a",
        "primock57-overlap-00",
    ],
)
def test_nested_report_privacy_rejects_paths_references_and_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    leak: str,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    report = subject.run_primock57_overlap_eval(
        harness.manifest.path,
        harness.snapshot.bundle.path.parent,
        scratch_parent=harness.scratch_parent,
    )
    metrics = dict(report["metrics"])
    metrics["debug"] = {"nested": {"leak": leak}}
    report["metrics"] = metrics
    _rebind(report)

    with pytest.raises(subject.Primock57OverlapEvalError) as error:
        subject._validate_report(report, initial=harness.snapshot)
    assert str(error.value) == ""
    assert leak not in str(error.value)


def test_worker_stderr_shape_cannot_hide_nested_private_detail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    report = subject.run_primock57_overlap_eval(
        harness.manifest.path,
        harness.snapshot.bundle.path.parent,
        scratch_parent=harness.scratch_parent,
    )
    report["worker_stderr"] = {
        **report["worker_stderr"],
        "detail": {"path": "/tmp/private/model"},
    }
    _rebind(report)

    with pytest.raises(subject.Primock57OverlapEvalError) as error:
        subject._validate_report(report, initial=harness.snapshot)
    assert str(error.value) == ""
