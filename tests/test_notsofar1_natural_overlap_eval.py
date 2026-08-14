from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from tools import notsofar1_natural_overlap_eval as subject
from tools import prepare_notsofar1_natural_overlap_fixture as fixture
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


def _fake_lock() -> subject._EvaluatorLock:
    return subject._EvaluatorLock(
        raw_sha256=subject.LOCK_FILE_SHA256,
        recipe_sha256=subject.LOCK_RECIPE_SHA256,
        manifest_sha256=subject.FIXTURE_MANIFEST_SHA256,
        receipt_sha256=subject.FIXTURE_RECEIPT_SHA256,
    )


def _bundle_snapshot(tmp_path: Path) -> subject._BundleSnapshot:
    root = _private_dir(tmp_path / "private-bundle")
    manifest_path = _private_file(root / fixture.MANIFEST_FILENAME)
    words = ("zero", "one", "two", "three", "four", "five", "six")
    inputs: list[subject._PrivateInput] = []
    for window_index, word in enumerate(words):
        reference_a = f"alpha private utterance {word}"
        reference_b = f"beta confidential response {word}"
        for channel_index, channel in enumerate(("channel-a", "channel-b")):
            ordinal = window_index * 2 + channel_index
            audio = bytes((ordinal + 1, 0, 0, 0))
            name = f"notsofar-overlap-w{window_index:02d}-{channel}.f32le"
            path = _private_file(root / name, audio)
            inputs.append(
                subject._PrivateInput(
                    window_index=window_index,
                    channel=channel,
                    path=path,
                    sha256=hashlib.sha256(audio).hexdigest(),
                    samples=1,
                    audio=audio,
                    reference_a=reference_a,
                    reference_b=reference_b,
                )
            )
    windows = tuple(
        {
            "window_id": f"notsofar-overlap-w{i:02d}",
            "envelope": {
                "samples": 1,
                "source_start_sample": i,
                "source_end_sample": i + 1,
            },
            "overlap": {
                "relative_start_sample": 0,
                "relative_end_sample": 1,
                "samples": 1,
            },
            "role_a": {
                "activity_relative_start_sample": 0,
                "activity_relative_end_sample": 1,
                "reference": inputs[i * 2].reference_a,
                "reference_sha256": hashlib.sha256(
                    inputs[i * 2].reference_a.encode()
                ).hexdigest(),
            },
            "role_b": {
                "activity_relative_start_sample": 0,
                "activity_relative_end_sample": 1,
                "reference": inputs[i * 2].reference_b,
                "reference_sha256": hashlib.sha256(
                    inputs[i * 2].reference_b.encode()
                ).hexdigest(),
            },
            "channels": [
                {
                    "case_id": f"notsofar-overlap-w{i:02d}-channel-a",
                    "channel": "channel-a",
                    "file": inputs[i * 2].path.name,
                    "samples": 1,
                    "sha256": inputs[i * 2].sha256,
                    "size_bytes": 4,
                },
                {
                    "case_id": f"notsofar-overlap-w{i:02d}-channel-b",
                    "channel": "channel-b",
                    "file": inputs[i * 2 + 1].path.name,
                    "samples": 1,
                    "sha256": inputs[i * 2 + 1].sha256,
                    "size_bytes": 4,
                },
            ],
        }
        for i in range(7)
    )
    loaded = fixture.LoadedNotsofarNaturalOverlapBundle(
        path=manifest_path,
        digest=subject.FIXTURE_MANIFEST_SHA256,
        receipt_sha256=subject.FIXTURE_RECEIPT_SHA256,
        source_contract_sha256="e" * 64,
        production_evidence=True,
        windows=windows,
        identities=tuple((i,) * 9 for i in range(16)),
    )
    return subject._BundleSnapshot(bundle=loaded, inputs=tuple(inputs))


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
        self.stderr_summary = {
            "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "truncated": False,
        }
        self.cgroup_evidence = None
        self.resource_observations = None
        self.__class__.instances.append(self)

    def __enter__(self) -> _FakeWorker:
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
                audio_seconds=request.pcm.samples / fixture.SAMPLE_RATE_HZ,
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
    ordinary_metric_calls: list[str]
    current_source_calls: list[str]


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
    ordinary_metric_calls: list[str] = []
    current_source_calls: list[str] = []
    source_bundle = SimpleNamespace(tree_sha256="1" * 64, files=(object(),))

    _FakeRunLock.instances = []
    _FakeWorker.instances = []
    _FakeWorker.requests = []
    _FakeWorker.text_by_digest = {}
    for item in snapshot.inputs:
        _FakeWorker.text_by_digest[item.sha256] = (
            f"{item.reference_a} {item.reference_b}"
        )

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
    monkeypatch.setattr(subject, "_has_git_ancestor", lambda _path: False)
    monkeypatch.setattr(subject, "StreamingWorker", _FakeWorker)
    monkeypatch.setattr(
        subject, "stage_worker_source_bundle", lambda *_args: source_bundle
    )

    def current_source_bundle() -> object:
        current_source_calls.append("source")
        return source_bundle

    monkeypatch.setattr(subject, "_current_worker_source_bundle", current_source_bundle)
    monkeypatch.setattr(
        subject,
        "verify_source_bundle",
        lambda *_args, **_kwargs: verify_events.append("source"),
    )
    monkeypatch.setattr(subject.streaming_stt_eval, "_BenchmarkRunLock", _FakeRunLock)
    monkeypatch.setattr(
        subject.streaming_stt_eval, "load_worker_manifest", load_manifest
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
        lambda *_args, **_kwargs: (
            ordinary_metric_calls.append("called"),
            (_ for _ in ()).throw(
                AssertionError("ordinary generic accuracy scorer was invoked")
            ),
        )[1],
        raising=False,
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
        ordinary_metric_calls=ordinary_metric_calls,
        current_source_calls=current_source_calls,
    )


def _trace(item: subject._PrivateInput, *, final: object | None = None) -> CaseTrace:
    selected = final or FinalEvent(
        request_id="private-request",
        seq=0,
        text=f"{item.reference_a} {item.reference_b}",
        samples_seen=item.samples,
        elapsed_ms=1.0,
        finalization_ms=1.0,
        compute_ms=1.0,
        audio_seconds=item.samples / fixture.SAMPLE_RATE_HZ,
        chunks=1,
        deadline_misses=0,
        max_backlog_ms=0.0,
        resources=ResourceUsage(rss_mb=1.0, threads=1, vram_mb=None),
    )
    return CaseTrace(partials=(), final=selected)


def _records(snapshot: subject._BundleSnapshot) -> list[RunRecord]:
    return [
        RunRecord(case_index=index, repeat=0, trace=_trace(item))
        for index, item in enumerate(snapshot.inputs)
    ]


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
    assert loaded.manifest_sha256 == subject.FIXTURE_MANIFEST_SHA256
    assert loaded.receipt_sha256 == subject.FIXTURE_RECEIPT_SHA256
    assert (
        loaded.manifest_sha256
        == "898e291c713154e3cbd78ba9c38fca6fed08284edace98ac615ebdd37265da7b"
    )
    assert (
        loaded.receipt_sha256
        == "29b5d0f1f11f2ac56dee92a7d5ab6ef02a18c4d695a0b9b981b67dde3a24e2ba"
    )

    alias = tmp_path / subject.DEFAULT_LOCK.name
    alias.symlink_to(subject.DEFAULT_LOCK)
    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject._load_lock(alias)


def test_lock_rejects_byte_drift_duplicate_json_and_bool_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = subject.DEFAULT_LOCK.read_bytes()
    monkeypatch.setattr(
        subject,
        "read_regular_bounded",
        lambda *_args, **_kwargs: SimpleNamespace(data=raw + b" "),
    )
    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject._load_lock()
    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject._strict_json_bytes(b'{"a":1,"a":1}', maximum_bytes=100)
    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject._strict_json_bytes(b'{"x":NaN}', maximum_bytes=100)


def _install_reanchored_lock(
    monkeypatch: pytest.MonkeyPatch,
    value: dict[str, object],
) -> None:
    recipe_value = copy.deepcopy(value)
    recipe_value.pop("recipe_sha256", None)
    recipe = subject._canonical_sha256(recipe_value)
    value["recipe_sha256"] = recipe
    raw = json.dumps(value, indent=2, ensure_ascii=True).encode("ascii") + b"\n"
    monkeypatch.setattr(subject, "LOCK_RECIPE_SHA256", recipe)
    monkeypatch.setattr(subject, "LOCK_FILE_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(subject, "LOCK_FILE_BYTES", len(raw))
    monkeypatch.setattr(
        subject,
        "read_regular_bounded",
        lambda *_args, **_kwargs: SimpleNamespace(data=raw),
    )


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("root", "schema_version"),
        ("execution", "repeats"),
        ("execution", "workers"),
        ("evidence_scope", "device_identity_authority"),
        ("evidence_scope", "diagnostic_only"),
        ("evidence_scope", "diarization_evidence"),
        ("evidence_scope", "endpoint_evidence"),
        ("evidence_scope", "latency_evidence"),
        ("evidence_scope", "live_evidence"),
        ("evidence_scope", "promotion_authority"),
        ("evidence_scope", "qualification_authority"),
        ("evidence_scope", "runtime_default_authority"),
    ],
)
def test_self_consistent_lock_rejects_every_bool_int_schema_alias(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
) -> None:
    value = json.loads(subject.DEFAULT_LOCK.read_bytes())
    target = value if section == "root" else value[section]
    original = target[field]
    target[field] = int(original) if type(original) is bool else True
    _install_reanchored_lock(monkeypatch, value)

    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject._load_lock()


@pytest.mark.parametrize(
    "field",
    ["bundle", "evidence_scope", "execution", "privacy"],
)
def test_self_consistent_lock_rejects_nested_container_swaps(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    value = json.loads(subject.DEFAULT_LOCK.read_bytes())
    value[field] = []
    _install_reanchored_lock(monkeypatch, value)

    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject._load_lock()


def test_preflight_reopens_injected_exact_production_snapshot_without_private_rows(
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
    report = subject.preflight_notsofar1_natural_overlap_bundle(
        snapshot.bundle.path.parent
    )

    assert calls == 2
    assert report == {
        "fixture_id": fixture.FIXTURE_ID,
        "windows": 7,
        "pcm_inputs": 14,
        "manifest_sha256": snapshot.bundle.digest,
        "receipt_sha256": snapshot.bundle.receipt_sha256,
        "source_contract_sha256": snapshot.bundle.source_contract_sha256,
        "evaluator_lock_recipe_sha256": subject.LOCK_RECIPE_SHA256,
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
    changed = replace(snapshot, inputs=tuple(reversed(snapshot.inputs)))
    snapshots = iter((snapshot, changed if surface == "snapshot" else snapshot))
    closures = iter(("9" * 64, "8" * 64 if surface == "closure" else "9" * 64))
    monkeypatch.setattr(subject, "_load_lock", lambda: _fake_lock())
    monkeypatch.setattr(subject, "_closure_sha256", lambda: next(closures))
    monkeypatch.setattr(subject, "_snapshot_bundle", lambda *_args: next(snapshots))
    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject.preflight_notsofar1_natural_overlap_bundle(snapshot.bundle.path.parent)


def test_snapshot_loader_reads_exact_14_generated_leaves_in_window_channel_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _bundle_snapshot(tmp_path)
    loads: list[Path | str] = []

    def load(path: Path | str) -> fixture.LoadedNotsofarNaturalOverlapBundle:
        loads.append(path)
        return expected.bundle

    monkeypatch.setattr(fixture, "load_notsofar1_natural_overlap_bundle", load)
    monkeypatch.setattr(
        fixture,
        "verify_notsofar1_natural_overlap_bundle",
        lambda bundle: None,
    )

    observed = subject._snapshot_bundle(
        expected.bundle.path.parent,
        _fake_lock(),
    )

    assert observed == expected
    assert len(loads) == 2
    assert [(item.window_index, item.channel) for item in observed.inputs] == [
        (window, channel)
        for window in range(7)
        for channel in ("channel-a", "channel-b")
    ]


@pytest.mark.parametrize("failure", ["leaf_bytes", "leaf_mode", "loader_close"])
def test_snapshot_rejects_leaf_or_loader_close_time_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    expected = _bundle_snapshot(tmp_path)
    calls = 0

    def load(_path: Path | str) -> fixture.LoadedNotsofarNaturalOverlapBundle:
        nonlocal calls
        calls += 1
        if failure == "loader_close" and calls == 2:
            identities = list(expected.bundle.identities)
            identities[0] = (999,) * 9
            return replace(expected.bundle, identities=tuple(identities))
        return expected.bundle

    monkeypatch.setattr(fixture, "load_notsofar1_natural_overlap_bundle", load)
    monkeypatch.setattr(
        fixture,
        "verify_notsofar1_natural_overlap_bundle",
        lambda bundle: None,
    )
    target = expected.inputs[0].path
    if failure == "leaf_bytes":
        target.write_bytes(b"\x00\x00\x00\x00")
    elif failure == "leaf_mode":
        target.chmod(0o644)

    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject._snapshot_bundle(expected.bundle.path.parent, _fake_lock())


def test_metrics_score_both_references_tie_order_and_exact_aggregate_equations(
    tmp_path: Path,
) -> None:
    snapshot = _bundle_snapshot(tmp_path)
    records = _records(snapshot)
    # Reverse-order hypotheses are exact under the fixed min-order metric and
    # differ across every paired channel after normalization.
    for index in range(1, 14, 2):
        item = snapshot.inputs[index]
        final = replace(
            records[index].trace.final,
            text=f"{item.reference_b} {item.reference_a}",
        )
        records[index] = replace(records[index], trace=_trace(item, final=final))

    metrics = subject._overlap_metrics(snapshot, records)

    assert metrics["coverage"] == {
        "windows": 7,
        "pcm_inputs": 14,
        "repeats": 1,
        "evaluations": 14,
        "source_complete_evaluations": 14,
    }
    assert metrics["paired_device"] == {
        "windows": 7,
        "comparable_windows": 7,
        "exact_match_agreement_windows": 0,
        "selected_order_agreement_windows": 0,
        "absolute_word_error_difference": 0,
        "absolute_hypothesis_word_difference": 0,
    }
    assert set(metrics["channel_accuracy"]) == {"channel_a", "channel_b"}
    aggregate = metrics["aggregate_accuracy"]
    channels = metrics["channel_accuracy"]
    assert aggregate["evaluations"] == 14
    assert aggregate["protocol"] == "two-utterance-min-order-wer-v1"
    assert aggregate["word_errors"] == 0
    assert aggregate["wer"] == 0.0
    assert channels["channel_a"]["evaluations"] == 7
    assert channels["channel_b"]["evaluations"] == 7
    for field in (
        "substitutions",
        "insertions",
        "deletions",
        "word_errors",
        "reference_words",
        "hypothesis_words",
    ):
        assert aggregate[field] == sum(channel[field] for channel in channels.values())
    assert "ordinary_wer" not in metrics
    assert "winner" not in metrics
    assert "repeat_disagreement" not in json.dumps(metrics, sort_keys=True)


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "duplicate", "swapped", "repeat"]
)
def test_metrics_reject_missing_extra_duplicate_swapped_or_repeated_record(
    tmp_path: Path,
    mutation: str,
) -> None:
    snapshot = _bundle_snapshot(tmp_path)
    records = _records(snapshot)
    if mutation == "missing":
        records.pop()
    elif mutation == "extra":
        records.append(records[-1])
    elif mutation == "duplicate":
        records[1] = replace(records[1], case_index=0)
    elif mutation == "swapped":
        records[0], records[1] = records[1], records[0]
    else:
        records[0] = replace(records[0], repeat=1)
    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject._overlap_metrics(snapshot, records)


@pytest.mark.parametrize("terminal_kind", ["native", "parakeet_cpp", "plain"])
def test_metrics_reject_any_early_or_incomplete_source_terminal(
    tmp_path: Path,
    terminal_kind: str,
) -> None:
    snapshot = _bundle_snapshot(tmp_path)
    records = _records(snapshot)
    item = snapshot.inputs[0]
    usage = ResourceUsage(rss_mb=1.0, threads=1, vram_mb=None)
    common = dict(
        request_id="private-request",
        seq=0,
        text=item.reference_a,
        samples_seen=0,
        elapsed_ms=1.0,
        finalization_ms=1.0,
        compute_ms=1.0,
        audio_seconds=item.samples / fixture.SAMPLE_RATE_HZ,
        chunks=1,
        deadline_misses=0,
        max_backlog_ms=0.0,
        resources=usage,
    )
    if terminal_kind == "native":
        final = NativeFinalEvent(
            **common,
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
            **common,
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
        final = FinalEvent(**{**common, "audio_seconds": 0.0})
    records[0] = replace(records[0], trace=_trace(item, final=final))
    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject._overlap_metrics(snapshot, records)


@pytest.mark.parametrize("terminal_kind", ["native", "parakeet_cpp"])
def test_metrics_reject_complete_source_terminal_with_wrong_audio_duration(
    tmp_path: Path,
    terminal_kind: str,
) -> None:
    snapshot = _bundle_snapshot(tmp_path)
    records = _records(snapshot)
    item = snapshot.inputs[0]
    usage = ResourceUsage(rss_mb=1.0, threads=1, vram_mb=None)
    common = dict(
        request_id="private-request",
        seq=0,
        text=f"{item.reference_a} {item.reference_b}",
        samples_seen=item.samples,
        elapsed_ms=1.0,
        finalization_ms=1.0,
        compute_ms=1.0,
        audio_seconds=0.0,
        chunks=1,
        deadline_misses=0,
        max_backlog_ms=0.0,
        resources=usage,
    )
    if terminal_kind == "native":
        final = NativeFinalEvent(
            **common,
            model_padding_samples=0,
            source_samples=item.samples,
            source_samples_consumed=item.samples,
            declared_tail_samples=0,
            tail_samples_consumed=0,
            endpoint_reason="eou",
            native_endpoint=True,
            endpoint_probability=1.0,
            endpoint_sample=item.samples,
            endpoint_latency_ms=0.0,
            authoritative=False,
        )
    else:
        final = ParakeetCppFinalEvent(
            **common,
            model_padding_samples=0,
            source_samples_offered=item.samples,
            source_samples_consumed=item.samples,
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
    records[0] = replace(records[0], trace=_trace(item, final=final))

    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject._overlap_metrics(snapshot, records)


def test_private_paths_protected_roots_and_git_ancestors_are_rejected(
    tmp_path: Path,
) -> None:
    real = _private_dir(tmp_path / "real-private")
    alias = tmp_path / "private-alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject._stable_existing_directory(alias)

    snapshot = _bundle_snapshot(tmp_path / "bundle-scope")
    manifest = _manifest(tmp_path / "manifest-scope")
    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject._validate_paths(
            initial=snapshot,
            manifest=manifest,
            scratch_parent=snapshot.bundle.path.parent,
            output_path=None,
        )
    safe = _private_dir(tmp_path / "safe")
    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject._validate_paths(
            initial=snapshot,
            manifest=manifest,
            scratch_parent=safe,
            output_path=safe / "report.json",
        )
    git_root = _private_dir(tmp_path / "git-root")
    _private_file(git_root / ".git")
    git_scratch = _private_dir(git_root / "scratch")
    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject._validate_paths(
            initial=snapshot,
            manifest=manifest,
            scratch_parent=git_scratch,
            output_path=None,
        )


def test_private_report_publisher_is_atomic_mode_600_single_link_and_no_clobber(
    tmp_path: Path,
) -> None:
    parent = _private_dir(tmp_path / "private-output")
    output = parent / "report.json"
    report = {"ok": True, "aggregate": {"evaluations": 14}}
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
    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject._publish_report(
            output,
            report,
            commit_guard=lambda: True,
            state=subject._ReportCommitState(),
        )
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest


def test_fake_worker_is_one_process_with_exact_14_window_major_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)

    report = subject.run_notsofar1_natural_overlap_eval(
        harness.manifest.path,
        harness.snapshot.bundle.path.parent,
        scratch_parent=harness.scratch_parent,
    )

    assert len(_FakeWorker.instances) == 1
    assert _FakeWorker.instances[0].fully_stopped is True
    assert len(_FakeWorker.requests) == 14
    assert [request.request_id for request in _FakeWorker.requests] == [
        f"notsofar1-natural-overlap-{index:02d}" for index in range(14)
    ]
    assert [request.pcm.sha256 for request in _FakeWorker.requests] == [
        item.sha256 for item in harness.snapshot.inputs
    ]
    assert [request.pcm.path.name for request in _FakeWorker.requests] == [
        f"input-{index:02d}.f32le" for index in range(14)
    ]
    assert len(_FakeRunLock.instances) == 1
    assert _FakeRunLock.instances[0].closed is True
    assert harness.ordinary_metric_calls == []
    assert report["metrics"]["coverage"]["evaluations"] == 14
    assert report["metrics"]["paired_device"]["exact_match_agreement_windows"] == 7
    assert "repeat_disagreement" not in json.dumps(report, sort_keys=True)
    encoded = json.dumps(report, sort_keys=True)
    assert str(tmp_path) not in encoded
    for item in harness.snapshot.inputs:
        assert item.reference_a not in encoded
        assert item.reference_b not in encoded
        assert item.path.name not in encoded


def test_end_to_end_fake_run_publishes_then_removes_private_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    output = harness.output_parent / "aggregate-report.json"
    state = subject._ReportCommitState()

    report = subject.run_notsofar1_natural_overlap_eval(
        harness.manifest.path,
        harness.snapshot.bundle.path.parent,
        scratch_parent=harness.scratch_parent,
        output_path=output,
        _commit_state=state,
    )

    expected = subject._canonical_json(report, newline=True)
    assert state.committed is True
    assert state.digest == hashlib.sha256(expected).hexdigest()
    assert output.read_bytes() == expected
    assert stat.S_IMODE(output.lstat().st_mode) == 0o600
    assert output.lstat().st_nlink == 1
    assert list(harness.scratch_parent.iterdir()) == []
    assert len(harness.verify_events) == 2
    assert len(harness.snapshot_calls) >= 3
    assert len(harness.lock_calls) >= 3
    assert len(harness.closure_calls) >= 3
    assert len(harness.manifest_calls) >= 3


@pytest.mark.parametrize("failure", ["worker", "not_stopped", "cleanup"])
def test_worker_timeout_stop_or_cleanup_failure_is_detail_free_and_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    output = harness.output_parent / "aggregate-report.json"
    if failure == "worker":
        monkeypatch.setattr(
            _FakeWorker,
            "transcribe",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                TimeoutError("private worker timeout detail")
            ),
        )
    elif failure == "not_stopped":
        monkeypatch.setattr(_FakeWorker, "__exit__", lambda *_args: None)
    else:
        monkeypatch.setattr(
            subject.streaming_stt_eval,
            "_remove_private_scratch",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("private cleanup detail")
            ),
        )

    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError) as error:
        subject.run_notsofar1_natural_overlap_eval(
            harness.manifest.path,
            harness.snapshot.bundle.path.parent,
            scratch_parent=harness.scratch_parent,
            output_path=output,
            _commit_state=subject._ReportCommitState(),
        )
    assert str(error.value) == ""
    assert not output.exists()


@pytest.mark.parametrize(
    ("surface", "stage"),
    [
        (surface, stage)
        for surface in ("bundle", "lock", "closure", "manifest", "source")
        for stage in ("post", "commit")
    ],
)
def test_bundle_lock_closure_manifest_or_source_race_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
    stage: str,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    output = harness.output_parent / "aggregate-report.json"
    changed_snapshot = replace(
        harness.snapshot,
        inputs=tuple(reversed(harness.snapshot.inputs)),
    )
    changed_manifest = SimpleNamespace(**vars(harness.manifest))
    changed_manifest.digest = "0" * 64
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
    else:
        calls = 0

        def source(*_args: object, **_kwargs: object) -> None:
            nonlocal calls
            calls += 1
            if calls >= (1 if stage == "post" else 2):
                raise RuntimeError("private source race")

        monkeypatch.setattr(subject, "verify_source_bundle", source)

    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError) as error:
        subject.run_notsofar1_natural_overlap_eval(
            harness.manifest.path,
            harness.snapshot.bundle.path.parent,
            scratch_parent=harness.scratch_parent,
            output_path=output,
            _commit_state=subject._ReportCommitState(),
        )
    assert str(error.value) == ""
    assert not output.exists()


def test_metric_rejects_trace_with_extra_terminal_surface(tmp_path: Path) -> None:
    snapshot = _bundle_snapshot(tmp_path)
    records = _records(snapshot)
    trace = records[0].trace
    records[0] = replace(
        records[0],
        trace=SimpleNamespace(
            partials=(),
            final=trace.final,
            extra_final=trace.final,
        ),
    )
    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject._overlap_metrics(snapshot, records)


def test_disagreement_uses_normalized_hypothesis_inequality_per_window(
    tmp_path: Path,
) -> None:
    snapshot = _bundle_snapshot(tmp_path)
    records = _records(snapshot)
    first_a = snapshot.inputs[0]
    first_b = snapshot.inputs[1]
    second_b = snapshot.inputs[3]
    records[0] = replace(
        records[0],
        trace=_trace(
            first_a, final=replace(records[0].trace.final, text="HELLO, world!")
        ),
    )
    records[1] = replace(
        records[1],
        trace=_trace(
            first_b, final=replace(records[1].trace.final, text="hello world")
        ),
    )
    records[3] = replace(
        records[3],
        trace=_trace(second_b, final=replace(records[3].trace.final, text="different")),
    )

    metrics = subject._overlap_metrics(snapshot, records)

    assert metrics["paired_device"]["exact_match_agreement_windows"] == 6


def test_two_distinct_equally_wrong_hypotheses_are_not_exact_agreement(
    tmp_path: Path,
) -> None:
    snapshot = _bundle_snapshot(tmp_path)
    records = _records(snapshot)
    first_a = snapshot.inputs[0]
    first_b = snapshot.inputs[1]
    records[0] = replace(
        records[0],
        trace=_trace(
            first_a, final=replace(records[0].trace.final, text="wrong alpha")
        ),
    )
    records[1] = replace(
        records[1],
        trace=_trace(first_b, final=replace(records[1].trace.final, text="wrong beta")),
    )

    metrics = subject._overlap_metrics(snapshot, records)

    paired = metrics["paired_device"]
    assert paired["exact_match_agreement_windows"] == 6
    assert (
        metrics["channel_accuracy"]["channel_a"]["word_errors"]
        == metrics["channel_accuracy"]["channel_b"]["word_errors"]
    )


def test_min_order_metric_tie_is_deterministic_and_private() -> None:
    tied = subject.score_two_utterance_min_order_wer(
        "secretalpha",
        "secretbeta",
        "secrethypothesis",
    )
    assert tied.selected_order == "role_a_then_role_b"
    assert "secretalpha" not in repr(tied)
    assert "secretbeta" not in repr(tied)
    assert "secrethypothesis" not in repr(tied)


@pytest.mark.parametrize("receipt", ["runtime", "model"])
@pytest.mark.parametrize("stage", ["post", "commit"])
def test_runtime_or_model_receipt_race_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt: str,
    stage: str,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    output = harness.output_parent / "aggregate-report.json"
    calls = 0
    target = 2 if stage == "post" else 3

    def digest(_manifest: object) -> str | None:
        nonlocal calls
        calls += 1
        return "0" * 64 if calls >= target else None

    name = (
        "_verified_runtime_receipt_digest"
        if receipt == "runtime"
        else "_verified_model_receipt_digest"
    )
    monkeypatch.setattr(subject.streaming_stt_eval, name, digest)

    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError) as error:
        subject.run_notsofar1_natural_overlap_eval(
            harness.manifest.path,
            harness.snapshot.bundle.path.parent,
            scratch_parent=harness.scratch_parent,
            output_path=output,
            _commit_state=subject._ReportCommitState(),
        )
    assert str(error.value) == ""
    assert not output.exists()


def test_closing_worker_input_mutation_is_rejected_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    output = harness.output_parent / "aggregate-report.json"

    def mutate_on_exit(worker: _FakeWorker, *_exc: object) -> None:
        worker.fully_stopped = True
        (worker.scratch / "input-00.f32le").write_bytes(b"\x00\x00\x00\x00")

    monkeypatch.setattr(_FakeWorker, "__exit__", mutate_on_exit)
    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject.run_notsofar1_natural_overlap_eval(
            harness.manifest.path,
            harness.snapshot.bundle.path.parent,
            scratch_parent=harness.scratch_parent,
            output_path=output,
            _commit_state=subject._ReportCommitState(),
        )
    assert not output.exists()


def _valid_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_Harness, dict[str, object]]:
    harness = _install_run_fakes(tmp_path, monkeypatch)
    report = subject.run_notsofar1_natural_overlap_eval(
        harness.manifest.path,
        harness.snapshot.bundle.path.parent,
        scratch_parent=harness.scratch_parent,
    )
    return harness, report


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "nan",
        "unsupported",
        "wrong_binding",
        "bool_count",
        "pooled_not_sum",
        "wrong_denominator",
        "disagreement_too_high",
        "selected_order_too_high",
        "comparable_wrong",
        "absolute_word_error_impossible",
        "absolute_hypothesis_impossible",
        "repeat_field",
        "ordinary_wer",
        "winner",
    ],
)
def test_report_is_closed_finite_and_equationally_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    harness, original = _valid_report(tmp_path, monkeypatch)
    report = copy.deepcopy(original)
    should_rebind = True
    if mutation == "missing":
        report.pop("config")
    elif mutation == "extra":
        report["debug"] = True
    elif mutation == "nan":
        report["metrics"]["aggregate_accuracy"]["wer"] = float("nan")
        should_rebind = False
    elif mutation == "unsupported":
        report["metrics"]["aggregate_accuracy"]["wer"] = object()
        should_rebind = False
    elif mutation == "wrong_binding":
        report["binding_sha256"] = "0" * 64
        should_rebind = False
    elif mutation == "bool_count":
        report["metrics"]["coverage"]["evaluations"] = True
    elif mutation == "pooled_not_sum":
        aggregate = report["metrics"]["aggregate_accuracy"]
        aggregate["insertions"] += 1
        aggregate["word_errors"] += 1
        aggregate["hypothesis_words"] += 1
        aggregate["wer"] = aggregate["word_errors"] / aggregate["reference_words"]
    elif mutation == "wrong_denominator":
        aggregate = report["metrics"]["aggregate_accuracy"]
        channel_a = report["metrics"]["channel_accuracy"]["channel_a"]
        channel_b = report["metrics"]["channel_accuracy"]["channel_b"]
        for summary in (aggregate, channel_a, channel_b):
            increment = 2 if summary is aggregate else 1
            summary["reference_words"] += increment
            summary["hypothesis_words"] += increment
            summary["wer"] = summary["word_errors"] / summary["reference_words"]
    elif mutation == "disagreement_too_high":
        report["metrics"]["paired_device"]["exact_match_agreement_windows"] = 8
    elif mutation == "selected_order_too_high":
        report["metrics"]["paired_device"]["selected_order_agreement_windows"] = 8
    elif mutation == "comparable_wrong":
        report["metrics"]["paired_device"]["comparable_windows"] = 6
    elif mutation == "absolute_word_error_impossible":
        report["metrics"]["paired_device"]["absolute_word_error_difference"] = 10**9
    elif mutation == "absolute_hypothesis_impossible":
        report["metrics"]["paired_device"]["absolute_hypothesis_word_difference"] = (
            10**9
        )
    elif mutation == "repeat_field":
        report["metrics"]["repeat_disagreement_inputs"] = 0
    elif mutation == "ordinary_wer":
        report["metrics"]["ordinary_wer"] = 0.0
    else:
        report["metrics"]["winner"] = "channel-a"
    if should_rebind:
        _rebind(report)

    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError) as error:
        subject._validate_report(report, initial=harness.snapshot)
    assert str(error.value) == ""


def test_report_rejects_dict_subclass_without_invoking_hostile_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, original = _valid_report(tmp_path, monkeypatch)

    class HostileDict(dict[str, object]):
        def __eq__(self, _other: object) -> bool:
            raise AssertionError("hostile equality invoked")

        def __bool__(self) -> bool:
            raise AssertionError("hostile truthiness invoked")

    report = HostileDict(original)
    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject._validate_report(report, initial=harness.snapshot)


def test_public_validator_rejects_hostile_mapping_and_key_before_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, original = _valid_report(tmp_path, monkeypatch)
    hooks: list[str] = []

    class HostileDict(dict[str, object]):
        def keys(self) -> object:
            hooks.append("keys")
            raise AssertionError("hostile keys invoked")

        def get(self, *_args: object) -> object:
            hooks.append("get")
            raise AssertionError("hostile get invoked")

    hostile_mapping = HostileDict(original)
    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject.validate_notsofar1_natural_overlap_report(
            hostile_mapping,
            bundle_dir=harness.snapshot.bundle.path.parent,
            worker_manifest_path=harness.manifest.path,
        )
    assert hooks == []

    nested = copy.deepcopy(original)
    nested["worker"] = HostileDict(nested["worker"])
    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject.validate_notsofar1_natural_overlap_report(
            nested,
            bundle_dir=harness.snapshot.bundle.path.parent,
            worker_manifest_path=harness.manifest.path,
        )
    assert hooks == []


@pytest.mark.parametrize(
    "leak",
    [
        "/tmp/private/notsofar/model",
        r"C:\\Users\\paul\\private-model",
        "alpha private utterance zero",
        "notsofar-overlap-w00-channel-a.f32le",
        "notsofar-overlap-w00",
        "notsofar-overlap-w00-channel-a",
        "sc_meetup_0",
        "sc_rockfall_2",
        "role_a",
        "role_b",
    ],
)
def test_nested_report_privacy_rejects_paths_references_hypotheses_and_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    leak: str,
) -> None:
    harness, original = _valid_report(tmp_path, monkeypatch)
    report = copy.deepcopy(original)
    report["worker"]["runtime"]["private_probe"] = leak
    _rebind(report)

    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError) as error:
        subject.validate_notsofar1_natural_overlap_report(
            report,
            bundle_dir=harness.snapshot.bundle.path.parent,
            worker_manifest_path=harness.manifest.path,
        )
    assert str(error.value) == ""
    assert leak not in str(error.value)


@pytest.mark.parametrize(
    "forbidden_key",
    ["hypothesis", "reference", "window_rows", "speaker_role", "raw_device_id"],
)
def test_report_rejects_forbidden_private_schema_terms_even_without_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forbidden_key: str,
) -> None:
    harness, original = _valid_report(tmp_path, monkeypatch)
    report = copy.deepcopy(original)
    report["worker"]["runtime"][forbidden_key] = "redacted"
    _rebind(report)
    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject._validate_report(report, initial=harness.snapshot)


def test_public_report_validator_reopens_bundle_lock_and_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, report = _valid_report(tmp_path, monkeypatch)
    before = (
        len(harness.snapshot_calls),
        len(harness.lock_calls),
        len(harness.closure_calls),
    )

    assert (
        subject.validate_notsofar1_natural_overlap_report(
            report,
            bundle_dir=harness.snapshot.bundle.path.parent,
            worker_manifest_path=harness.manifest.path,
        )
        == report
    )
    assert len(harness.snapshot_calls) == before[0] + 2
    assert len(harness.lock_calls) == before[1] + 2
    assert len(harness.closure_calls) == before[2] + 2
    assert harness.current_source_calls == ["source", "source", "source"]
    assert not (harness.manifest.path.parent / "source-bundle").exists()


def test_current_worker_source_bundle_is_recomputed_without_staging_or_adjacent_tree() -> (
    None
):
    adjacent = subject.streaming_stt_eval._REPO_ROOT / "source-bundle.json"
    assert not adjacent.exists()

    bundle = subject._current_worker_source_bundle()

    assert not adjacent.exists()
    assert bundle.root == subject.streaming_stt_eval._REPO_ROOT.resolve(strict=True)
    assert bundle.manifest_path == adjacent
    assert bundle.worker_relative_path == subject.WORKER_RELATIVE_PATH
    assert tuple(item.relative_path for item in bundle.files) == tuple(
        sorted(subject.WORKER_SOURCE_FILES)
    )
    assert all(item.size_bytes > 0 and len(item.sha256) == 64 for item in bundle.files)


@pytest.mark.parametrize("failed_open", [2, 3])
def test_current_worker_source_bundle_closes_prior_descriptors_on_nested_open_failure(
    monkeypatch: pytest.MonkeyPatch,
    failed_open: int,
) -> None:
    real_open = subject.source_bundle_support._open_child_directory
    real_close = subject.os.close
    opened: list[int] = []
    closed: list[int] = []
    calls = 0

    def child(parent_fd: int, name: str) -> int:
        nonlocal calls
        calls += 1
        if calls == failed_open:
            raise OSError("injected child open failure")
        descriptor = real_open(parent_fd, name)
        opened.append(descriptor)
        return descriptor

    def close(descriptor: int) -> None:
        if descriptor in opened:
            closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(subject.source_bundle_support, "_open_child_directory", child)
    monkeypatch.setattr(subject.os, "close", close)

    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject._current_worker_source_bundle()
    assert closed == list(reversed(opened))
    for descriptor in opened:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_current_worker_source_bundle_attempts_all_closes_after_one_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = subject.source_bundle_support._open_child_directory
    real_read = subject.source_bundle_support._read_source_at
    real_close = subject.os.close
    opened: list[int] = []
    attempted: list[int] = []
    failed = False

    def child(parent_fd: int, name: str) -> int:
        descriptor = real_open(parent_fd, name)
        opened.append(descriptor)
        return descriptor

    def read_then_fail(parent_fd: int, name: str) -> bytes:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected source read failure")
        return real_read(parent_fd, name)

    def close(descriptor: int) -> None:
        if len(opened) == 3 and descriptor in opened:
            attempted.append(descriptor)
        if len(opened) == 3 and descriptor == opened[-1]:
            raise OSError("injected first close failure")
        real_close(descriptor)

    monkeypatch.setattr(subject.source_bundle_support, "_open_child_directory", child)
    monkeypatch.setattr(
        subject.source_bundle_support, "_read_source_at", read_then_fail
    )
    monkeypatch.setattr(subject.os, "close", close)

    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject._current_worker_source_bundle()
    assert attempted == list(reversed(opened))
    # The injected first failure deliberately leaves its descriptor open; every
    # later descriptor must nevertheless have received a real close attempt.
    for descriptor in opened[:-1]:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    real_close(opened[-1])


def test_public_validator_rejects_current_worker_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, report = _valid_report(tmp_path, monkeypatch)
    stable = SimpleNamespace(tree_sha256="1" * 64, files=(object(),))
    changed = SimpleNamespace(tree_sha256="0" * 64, files=(object(),))
    values = iter((stable, changed))
    monkeypatch.setattr(subject, "_current_worker_source_bundle", lambda: next(values))

    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject.validate_notsofar1_natural_overlap_report(
            report,
            bundle_dir=harness.snapshot.bundle.path.parent,
            worker_manifest_path=harness.manifest.path,
        )


@pytest.mark.parametrize("surface", ["worker", "evaluator", "config"])
def test_public_validator_rejects_forged_worker_evaluator_or_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    harness, original = _valid_report(tmp_path, monkeypatch)
    report = copy.deepcopy(original)
    if surface == "worker":
        report["worker"]["manifest_sha256"] = "0" * 64
    elif surface == "evaluator":
        report["evaluator"]["wrapper_closure_sha256"] = "0" * 64
    else:
        report["config"]["chunk_samples"] += 1
        report["config"]["contract_sha256"] = subject._canonical_sha256(
            {
                key: value
                for key, value in report["config"].items()
                if key != "contract_sha256"
            }
        )
    _rebind(report)

    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject.validate_notsofar1_natural_overlap_report(
            report,
            bundle_dir=harness.snapshot.bundle.path.parent,
            worker_manifest_path=harness.manifest.path,
        )


@pytest.mark.parametrize("receipt", ["runtime", "model"])
def test_public_validator_rejects_runtime_or_model_receipt_close_time_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt: str,
) -> None:
    harness, report = _valid_report(tmp_path, monkeypatch)
    values = iter((None, "0" * 64))
    name = (
        "_verified_runtime_receipt_digest"
        if receipt == "runtime"
        else "_verified_model_receipt_digest"
    )
    monkeypatch.setattr(
        subject.streaming_stt_eval, name, lambda _manifest: next(values)
    )

    with pytest.raises(subject.Notsofar1NaturalOverlapEvalError):
        subject.validate_notsofar1_natural_overlap_report(
            report,
            bundle_dir=harness.snapshot.bundle.path.parent,
            worker_manifest_path=harness.manifest.path,
        )


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), subject._LifecycleSignal(15)])
def test_prelink_lifecycle_leaves_no_report(
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
    monkeypatch.setattr(
        subject,
        "run_notsofar1_natural_overlap_eval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )
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
