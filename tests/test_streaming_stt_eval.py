from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from tests.streaming_stt_helpers import (
    WORKER_PATH,
    bound_file,
    resource,
    scripted_case,
    write_fixture,
)
from tools import streaming_stt_eval
from tools.streaming_stt.corpus import load_corpus
from tools.streaming_stt.metrics import RunRecord, aggregate_metrics
from tools.streaming_stt.manifest import (
    MOONSHINE_ADAPTER,
    MOONSHINE_ARTIFACT_NAMES,
    MoonshineConfig,
)
from tools.streaming_stt.protocol import (
    CaseTrace,
    FinalEvent,
    PartialEvent,
    ReadyEvent,
    ResourceUsage,
    StreamConfig,
)


@pytest.fixture(autouse=True)
def _isolated_host_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(
        streaming_stt_eval,
        "_HOST_LOCK_PATH",
        tmp_path / "host-locks" / "streaming-stt.lock",
    )


def _benchmark_fixture(tmp_path: Path) -> tuple[Path, Path]:
    manifest, corpus, _ = write_fixture(
        tmp_path,
        [
            {
                "id": "sentinel-private-command",
                "values": [0.01] * 160,
                "expected_text": "stop now",
                "commands": ["stop"],
                "tags": ["command"],
            },
            {
                "id": "sentinel-private-silence",
                "values": [0.0] * 80,
                "expected_text": "",
                "assertion": "silence",
                "tags": ["silence"],
            },
        ],
        [
            scripted_case(
                partials=[
                    (80, "sto", 40.0, 1.0),
                    (160, "stop now", 80.0, 2.0),
                ],
                final="stop now",
                elapsed_ms=100.0,
                finalization_ms=20.0,
                compute_ms=4.0,
                deadline_misses=1,
                max_backlog_ms=3.0,
                resources=resource(rss_mb=90.0, threads=2, vram_mb=20.0),
            ),
            scripted_case(
                partials=[(80, "", 20.0, 0.5)],
                final="",
                elapsed_ms=30.0,
                finalization_ms=10.0,
                compute_ms=1.0,
                resources=resource(rss_mb=80.0, threads=1, vram_mb=10.0),
            ),
        ],
    )
    return manifest, corpus


def test_end_to_end_report_is_aggregate_exact_bound_and_fake_labelled(tmp_path):
    manifest, corpus = _benchmark_fixture(tmp_path)
    scratch = tmp_path / "scratch-parent"

    report = streaming_stt_eval.run_benchmark(
        manifest,
        corpus,
        scratch_parent=scratch,
        repeats=3,
        stream=StreamConfig(64, "burst", 100, 16),
    )

    assert report["ok"] is True
    assert report["evidence"] == {
        "kind": "bounded_pcm_streaming_harness",
        "fake_adapter": True,
        "production_model": False,
        "real_candidate_model": False,
        "model_executed": False,
        "replay_pace": "burst",
        "accelerated_replay": True,
        "conversational_latency_valid": False,
        "timing_scope": "adapter_after_pcm_load_to_result",
        "compute_rtf_scope": "candidate_calls_only",
        "end_to_end_rtf": False,
        "pcm_snapshot_io_included": False,
        "controller_ipc_included": False,
        "capture_to_text_latency": False,
        "vad": False,
        "endpointing": False,
        "aec": False,
        "live_hardware": False,
        "adoption_authority": False,
    }
    assert report["worker"]["adapter"] == "fake-json-v1"
    assert "manifest_schema_version" not in report["worker"]
    assert "adapter_config" not in report["worker"]
    assert "runtime_receipt_sha256" not in report["worker"]
    assert len(report["worker"]["manifest_sha256"]) == 64
    assert len(report["worker"]["source_bundle_sha256"]) == 64
    assert report["worker"]["source_bundle_files"] > 1
    assert len(report["worker"]["artifact_set_sha256"]) == 64
    assert report["corpus"]["cases"] == 2
    assert "schema_version" not in report["corpus"]
    assert "provenance" not in report["corpus"]
    metrics = report["metrics"]
    assert metrics["evaluations"] == 6
    assert metrics["coverage_complete"] is True
    assert metrics["accuracy"]["transcript"]["clips"] == 3
    assert metrics["accuracy"]["transcript"]["exact"] == 3
    assert metrics["accuracy"]["command_recall"] == 1.0
    assert metrics["accuracy"]["silence_nonempty_finals"] == 0
    assert metrics["streaming"]["deadline_misses"] == 3
    assert metrics["latency"]["first_nonempty_partial_p50_ms"] == 40.0
    assert metrics["latency"]["stable_partial_p95_ms"] == 80.0
    assert metrics["resources"]["peak_vram_mb"] == 20.0
    assert metrics["determinism"]["cases_with_final_disagreement"] == 0
    assert report["evaluator"]["kind"] == "isolated_streaming_stt_fake_harness"
    assert report["evaluator"]["production_model"] is False
    assert report["evaluator"]["real_candidate_model"] is False
    assert report["evaluator"]["model_executed"] is False
    assert {
        "tools/streaming_stt/runtime_receipt.py",
        "tools/streaming_stt/adapters/moonshine.py",
    } <= set(report["evaluator"]["files"])
    assert {
        "tools/prepare_public_streaming_stt_corpus.py",
        "tools/provision_moonshine_candidate.py",
    }.isdisjoint(report["evaluator"]["files"])
    assert not any(scratch.glob(".streaming-stt-*"))

    encoded = json.dumps(report)
    for private in (
        "stop now",
        "sto",
        str(corpus),
        str(tmp_path / "audio-0.f32le"),
        "sentinel-private-command",
        "sentinel-private-silence",
    ):
        assert private not in encoded


def test_moonshine_worker_and_evidence_bindings_are_closed_and_non_adopting(
    tmp_path,
):
    artifacts = tuple(
        SimpleNamespace(
            name=name,
            sha256=f"{index + 1:064x}",
            size_bytes=index + 1,
            path=tmp_path / f"private-artifact-{index}",
        )
        for index, name in enumerate(MOONSHINE_ARTIFACT_NAMES)
    )
    manifest = SimpleNamespace(
        digest="a" * 64,
        schema_version=2,
        model_id="moonshine-small-streaming",
        adapter=MOONSHINE_ADAPTER,
        python=SimpleNamespace(sha256="b" * 64),
        worker=SimpleNamespace(sha256="c" * 64),
        artifacts=artifacts,
        adapter_config=MoonshineConfig(
            package_version="0.1.0",
            api_version=30000,
            model_arch="small-streaming",
            provider="cpu",
            language="en",
        ),
    )
    source_bundle = SimpleNamespace(
        tree_sha256="d" * 64,
        files=(object(), object()),
    )

    binding = streaming_stt_eval._worker_binding(
        manifest,
        source_bundle,
        runtime_receipt_sha256="e" * 64,
    )

    assert binding["manifest_schema_version"] == 2
    assert binding["adapter_config"] == {
        "package_version": "0.1.0",
        "api_version": 30000,
        "model_arch": "small-streaming",
        "provider": "cpu",
        "language": "en",
    }
    assert binding["runtime_receipt_sha256"] == "e" * 64
    assert str(tmp_path) not in json.dumps(binding)
    assert streaming_stt_eval._evidence_binding(
        MOONSHINE_ADAPTER,
        pace="realtime",
    ) == {
        "kind": "bounded_pcm_streaming_harness",
        "fake_adapter": False,
        "production_model": False,
        "real_candidate_model": True,
        "model_executed": True,
        "replay_pace": "realtime",
        "accelerated_replay": False,
        "conversational_latency_valid": True,
        "timing_scope": "adapter_after_pcm_load_to_result",
        "compute_rtf_scope": "candidate_calls_only",
        "end_to_end_rtf": False,
        "pcm_snapshot_io_included": False,
        "controller_ipc_included": False,
        "capture_to_text_latency": False,
        "vad": False,
        "endpointing": False,
        "aec": False,
        "live_hardware": False,
        "adoption_authority": False,
    }
    evaluator = streaming_stt_eval._evaluator_binding(MOONSHINE_ADAPTER)
    assert evaluator["production_model"] is False
    assert evaluator["real_candidate_model"] is True
    assert evaluator["model_executed"] is True
    assert {
        "tools/prepare_public_streaming_stt_corpus.py",
        "tools/provision_moonshine_candidate.py",
    }.isdisjoint(evaluator["files"])
    with pytest.raises(ValueError):
        streaming_stt_eval._evidence_binding("unknown-adapter", pace="burst")
    with pytest.raises(ValueError):
        streaming_stt_eval._evidence_binding(MOONSHINE_ADAPTER, pace="warp")


def test_moonshine_runtime_receipt_is_loaded_and_tree_verified(
    tmp_path,
    monkeypatch,
):
    artifact = SimpleNamespace(
        path=tmp_path / "runtime-receipt.json",
        sha256="a" * 64,
    )
    manifest = SimpleNamespace(
        adapter=MOONSHINE_ADAPTER,
        artifact_by_name={"runtime-receipt": artifact},
        python=SimpleNamespace(path=tmp_path / "venv" / "bin" / "python"),
    )
    receipt = SimpleNamespace(digest=artifact.sha256)
    calls: list[object] = []

    def load(path, *, expected_digest):
        calls.append(("load", path, expected_digest))
        return receipt

    def verify(value):
        calls.append(("verify", value))

    def verify_location(value, python_path):
        calls.append(("location", value, python_path))

    monkeypatch.setattr(streaming_stt_eval, "load_runtime_tree_receipt", load)
    monkeypatch.setattr(streaming_stt_eval, "verify_runtime_tree_receipt", verify)
    monkeypatch.setattr(
        streaming_stt_eval,
        "verify_venv_runtime_location",
        verify_location,
    )

    assert (
        streaming_stt_eval._verified_runtime_receipt_digest(manifest) == artifact.sha256
    )
    assert calls == [
        ("load", artifact.path, artifact.sha256),
        ("location", receipt, manifest.python.path),
        ("verify", receipt),
    ]


def test_controller_checks_runtime_receipt_boundary_before_and_after_worker(
    tmp_path,
    monkeypatch,
):
    manifest, corpus = _benchmark_fixture(tmp_path)
    calls: list[str] = []
    real_verify = streaming_stt_eval._verified_runtime_receipt_digest

    def record(candidate):
        calls.append(candidate.adapter)
        return real_verify(candidate)

    monkeypatch.setattr(
        streaming_stt_eval,
        "_verified_runtime_receipt_digest",
        record,
    )

    report = streaming_stt_eval.run_benchmark(
        manifest,
        corpus,
        scratch_parent=tmp_path / "scratch",
        repeats=1,
    )

    assert report["ok"] is True
    assert calls == ["fake-json-v1", "fake-json-v1"]


def test_cli_description_is_manifest_adapter_generic():
    help_text = " ".join(streaming_stt_eval._parser().format_help().split())

    assert "manifest-selected adapter" in help_text
    assert "fake worker receipts only" not in help_text


def test_host_run_lock_rejects_a_second_controller(tmp_path):
    lock_path = tmp_path / "host.lock"
    first = streaming_stt_eval._BenchmarkRunLock.acquire(lock_path)
    try:
        with pytest.raises(ValueError):
            streaming_stt_eval._BenchmarkRunLock.acquire(lock_path)
    finally:
        first.close()

    second = streaming_stt_eval._BenchmarkRunLock.acquire(lock_path)
    second.close()


def test_host_run_lock_path_is_independent_of_ambient_temp_directories(
    tmp_path,
    monkeypatch,
):
    expected = streaming_stt_eval._trusted_host_lock_path()

    monkeypatch.setenv("TMPDIR", str(tmp_path / "ambient-tmp"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "ambient-runtime"))

    assert streaming_stt_eval._trusted_host_lock_path() == expected
    assert expected.is_absolute()
    assert str(tmp_path) not in str(expected)


def test_host_run_lock_rename_never_permits_a_second_lease(tmp_path):
    lock_path = tmp_path / "host.lock"
    moved_path = tmp_path / "moved.lock"
    first = streaming_stt_eval._BenchmarkRunLock.acquire(lock_path)
    lock_path.rename(moved_path)

    with pytest.raises(ValueError):
        streaming_stt_eval._BenchmarkRunLock.acquire(lock_path)
    with pytest.raises(ValueError):
        first.assert_bound()
    with pytest.raises(ValueError):
        first.close()

    replacement = streaming_stt_eval._BenchmarkRunLock.acquire(lock_path)
    replacement.close()


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_host_run_lock_never_chmods_an_external_link_target(tmp_path, link_kind):
    external = tmp_path / "external-lock-target"
    external.write_text("sentinel", encoding="utf-8")
    external.chmod(0o640)
    original_mode = stat.S_IMODE(external.stat().st_mode)
    lock_path = tmp_path / "host.lock"
    if link_kind == "symlink":
        lock_path.symlink_to(external)
    else:
        os.link(external, lock_path)

    with pytest.raises(ValueError):
        streaming_stt_eval._BenchmarkRunLock.acquire(lock_path)

    assert external.read_text(encoding="utf-8") == "sentinel"
    assert stat.S_IMODE(external.stat().st_mode) == original_mode


def test_private_cleanup_never_chmods_linked_external_targets(tmp_path):
    external_file = tmp_path / "external-file"
    external_file.write_text("sentinel", encoding="utf-8")
    external_file.chmod(0o640)
    external_directory = tmp_path / "external-directory"
    external_directory.mkdir(mode=0o750)
    external_directory.chmod(0o750)
    file_mode = stat.S_IMODE(external_file.stat().st_mode)
    directory_mode = stat.S_IMODE(external_directory.stat().st_mode)

    scratch = tmp_path / "private-scratch"
    scratch.mkdir(mode=0o700)
    os.link(external_file, scratch / "hardlink")
    (scratch / "directory-link").symlink_to(
        external_directory, target_is_directory=True
    )
    locked = scratch / "locked"
    locked.mkdir()
    (locked / "payload").write_text("private", encoding="utf-8")
    locked.chmod(0o000)

    metadata = scratch.lstat()
    streaming_stt_eval._remove_private_scratch(
        scratch,
        expected_identity=(metadata.st_dev, metadata.st_ino),
    )

    assert not scratch.exists()
    assert external_file.read_text(encoding="utf-8") == "sentinel"
    assert stat.S_IMODE(external_file.stat().st_mode) == file_mode
    assert stat.S_IMODE(external_directory.stat().st_mode) == directory_mode


def test_private_cleanup_rejects_a_replaced_scratch_root(tmp_path):
    scratch = tmp_path / "private-scratch"
    scratch.mkdir(mode=0o700)
    metadata = scratch.lstat()
    original_identity = (metadata.st_dev, metadata.st_ino)
    moved = tmp_path / "moved-original"
    scratch.rename(moved)
    scratch.mkdir(mode=0o700)
    sentinel = scratch / "replacement-sentinel"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError):
        streaming_stt_eval._remove_private_scratch(
            scratch,
            expected_identity=original_identity,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert moved.is_dir()


def test_scratch_creation_rejects_a_symlinked_ancestor_without_writing_through_it(
    tmp_path,
):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError):
        streaming_stt_eval._prepare_scratch(linked_parent / "nested")

    assert not (real_parent / "nested").exists()


def test_invalid_stream_is_rejected_before_inputs_or_worker_are_touched(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        streaming_stt_eval,
        "load_worker_manifest",
        lambda _path: (_ for _ in ()).throw(AssertionError),
    )

    with pytest.raises(ValueError):
        streaming_stt_eval.run_benchmark(
            tmp_path / "missing-worker.json",
            tmp_path / "missing-corpus.json",
            scratch_parent=tmp_path / "scratch",
            stream=StreamConfig(0, "burst", 100, 0),
        )

    assert not (tmp_path / "scratch").exists()


def test_controller_rejects_identical_worker_bytes_from_an_unbound_source_tree(
    tmp_path,
):
    manifest_path, corpus_path = _benchmark_fixture(tmp_path)
    copied_worker = tmp_path / "copied-tree" / "tools" / "streaming_stt" / "worker.py"
    copied_worker.parent.mkdir(parents=True)
    copied_worker.write_bytes(WORKER_PATH.read_bytes())
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["worker"] = bound_file(copied_worker)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        streaming_stt_eval.run_benchmark(
            manifest_path,
            corpus_path,
            scratch_parent=tmp_path / "scratch",
            repeats=1,
        )


def test_private_bundle_is_cleaned_after_worker_start_failure(tmp_path, monkeypatch):
    manifest_path, corpus_path = _benchmark_fixture(tmp_path)
    scratch_parent = tmp_path / "scratch"
    monkeypatch.setattr(
        streaming_stt_eval.StreamingWorker,
        "start",
        lambda _self: (_ for _ in ()).throw(RuntimeError("forced start failure")),
    )

    with pytest.raises(RuntimeError, match="forced start failure"):
        streaming_stt_eval.run_benchmark(
            manifest_path,
            corpus_path,
            scratch_parent=scratch_parent,
            repeats=1,
        )

    assert not any(scratch_parent.glob(".streaming-stt-*"))


def test_repo_ancestor_retarget_cannot_execute_alternate_modules_and_fails_report(
    tmp_path,
    monkeypatch,
):
    manifest_path, corpus_path = _benchmark_fixture(tmp_path)
    source_link = tmp_path / "worker-source"
    source_link.symlink_to(streaming_stt_eval._REPO_ROOT, target_is_directory=True)
    alternate_root = tmp_path / "alternate"
    alternate_worker = alternate_root / "tools" / "streaming_stt" / "worker.py"
    alternate_worker.parent.mkdir(parents=True)
    alternate_worker.write_bytes(WORKER_PATH.read_bytes())
    marker = tmp_path / "alternate-module-executed"
    alternate_adapter = (
        alternate_root / "tools" / "streaming_stt" / "adapters" / "__init__.py"
    )
    alternate_adapter.parent.mkdir()
    alternate_adapter.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n",
        encoding="utf-8",
    )

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["worker"]["path"] = str(
        source_link / "tools" / "streaming_stt" / "worker.py"
    )
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    real_stage = streaming_stt_eval.stage_worker_source_bundle

    def stage_then_retarget(repository_root, destination):
        bundle = real_stage(repository_root, destination)
        source_link.unlink()
        source_link.symlink_to(alternate_root, target_is_directory=True)
        return bundle

    monkeypatch.setattr(
        streaming_stt_eval,
        "stage_worker_source_bundle",
        stage_then_retarget,
    )

    with pytest.raises(ValueError):
        streaming_stt_eval.run_benchmark(
            manifest_path,
            corpus_path,
            scratch_parent=tmp_path / "scratch",
            repeats=1,
        )

    assert not marker.exists()


def test_metric_reducer_counts_final_disagreement_retraction_and_missing_stability(
    tmp_path,
):
    _, corpus_path = _benchmark_fixture(tmp_path)
    corpus = load_corpus(corpus_path)
    usage = ResourceUsage(rss_mb=10.0, threads=1, vram_mb=None)
    ready = ReadyEvent(
        model_id="fake-stream-v1",
        manifest_sha256="a" * 64,
        source_bundle_sha256="b" * 64,
        adapter="fake-json-v1",
        model_load_ms=5.0,
        resources=usage,
        runtime={"python": "3.12", "platform": "linux"},
    )

    def trace(final_text: str) -> CaseTrace:
        return CaseTrace(
            partials=(
                PartialEvent("x", 0, "stop now please", 80, 20.0, 1.0),
                PartialEvent("x", 1, "stop", 160, 40.0, 1.0),
            ),
            final=FinalEvent(
                "x",
                2,
                final_text,
                160,
                60.0,
                20.0,
                2.0,
                0.01,
                1,
                0,
                0.0,
                usage,
            ),
        )

    metrics = aggregate_metrics(
        corpus,
        (
            RunRecord(0, 0, trace("stop")),
            RunRecord(0, 1, trace("cancel")),
        ),
        ready,
        repeats=2,
    )

    assert metrics["coverage_complete"] is False
    assert metrics["determinism"]["cases_with_final_disagreement"] == 1
    assert metrics["streaming"]["retracted_words"] > 0
    assert metrics["latency"]["stable_partial_missing"] == 1


def test_metric_reducer_counts_last_partial_to_final_rewrite(tmp_path):
    _, corpus_path = _benchmark_fixture(tmp_path)
    corpus = load_corpus(corpus_path)
    usage = ResourceUsage(rss_mb=10.0, threads=1, vram_mb=None)
    ready = ReadyEvent(
        model_id="fake-stream-v1",
        manifest_sha256="a" * 64,
        source_bundle_sha256="b" * 64,
        adapter="fake-json-v1",
        model_load_ms=5.0,
        resources=usage,
        runtime={"python": "3.12", "platform": "linux"},
    )
    trace = CaseTrace(
        partials=(PartialEvent("x", 0, "stop now please", 160, 40.0, 1.0),),
        final=FinalEvent(
            "x",
            1,
            "cancel",
            160,
            60.0,
            20.0,
            2.0,
            0.01,
            1,
            0,
            0.0,
            usage,
        ),
    )

    metrics = aggregate_metrics(
        corpus,
        (RunRecord(0, 0, trace),),
        ready,
        repeats=1,
    )

    assert metrics["streaming"]["churn_token_edits"] > 0
    assert metrics["streaming"]["retracted_words"] == 3


def test_silence_punctuation_hallucinations_are_counted_nonempty(tmp_path):
    _, corpus_path = _benchmark_fixture(tmp_path)
    corpus = load_corpus(corpus_path)
    usage = ResourceUsage(rss_mb=10.0, threads=1, vram_mb=None)
    ready = ReadyEvent(
        model_id="fake-stream-v1",
        manifest_sha256="a" * 64,
        source_bundle_sha256="b" * 64,
        adapter="fake-json-v1",
        model_load_ms=5.0,
        resources=usage,
        runtime={"python": "3.12", "platform": "linux"},
    )
    trace = CaseTrace(
        partials=(PartialEvent("x", 0, "!!!", 80, 20.0, 1.0),),
        final=FinalEvent(
            "x",
            1,
            "???",
            80,
            30.0,
            10.0,
            2.0,
            0.005,
            1,
            0,
            0.0,
            usage,
        ),
    )

    metrics = aggregate_metrics(
        corpus,
        (RunRecord(1, 0, trace),),
        ready,
        repeats=1,
    )

    assert metrics["accuracy"]["silence_nonempty_partials"] == 1
    assert metrics["accuracy"]["silence_nonempty_finals"] == 1


def test_metric_reducer_rejects_finite_inputs_whose_sum_overflows(tmp_path):
    _, corpus_path = _benchmark_fixture(tmp_path)
    corpus = load_corpus(corpus_path)
    usage = ResourceUsage(rss_mb=10.0, threads=1, vram_mb=None)
    ready = ReadyEvent(
        model_id="fake-stream-v1",
        manifest_sha256="a" * 64,
        source_bundle_sha256="b" * 64,
        adapter="fake-json-v1",
        model_load_ms=5.0,
        resources=usage,
        runtime={"python": "3.12", "platform": "linux"},
    )
    trace = CaseTrace(
        partials=(),
        final=FinalEvent(
            "x",
            0,
            "stop now",
            160,
            30.0,
            10.0,
            1e308,
            0.01,
            1,
            0,
            0.0,
            usage,
        ),
    )

    with pytest.raises(ValueError, match="non-finite aggregate"):
        aggregate_metrics(
            corpus,
            (
                RunRecord(0, 0, trace),
                RunRecord(0, 1, trace),
            ),
            ready,
            repeats=2,
        )


def test_cli_writes_mode_600_report_and_never_prints_private_rows(
    tmp_path,
    capsys,
):
    manifest, corpus = _benchmark_fixture(tmp_path)
    output = tmp_path / "report.json"

    assert (
        streaming_stt_eval.main(
            [
                "--worker-manifest",
                str(manifest),
                "--corpus",
                str(corpus),
                "--scratch-root",
                str(tmp_path / "scratch"),
                "--repeats",
                "1",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out
    assert json.loads(stdout) == json.loads(output.read_text(encoding="utf-8"))
    assert "stop now" not in stdout
    assert "audio-0" not in stdout
    assert os.stat(output).st_mode & 0o777 == 0o600


def test_cli_failure_is_detail_free(monkeypatch, capsys):
    monkeypatch.setattr(
        streaming_stt_eval,
        "run_benchmark",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("SENTINEL_PRIVATE_TRANSCRIPT_AND_PATH")
        ),
    )

    assert (
        streaming_stt_eval.main(
            ["--worker-manifest", "/missing", "--corpus", "/missing"]
        )
        == 2
    )
    output = capsys.readouterr().out
    assert "SENTINEL" not in output
    assert json.loads(output) == streaming_stt_eval._SAFE_ERROR


def test_cli_never_writes_or_prints_nonstandard_infinity(tmp_path, monkeypatch, capsys):
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        streaming_stt_eval,
        "run_benchmark",
        lambda *_args, **_kwargs: {"ok": True, "metric": float("inf")},
    )

    assert (
        streaming_stt_eval.main(
            [
                "--worker-manifest",
                "/unused",
                "--corpus",
                "/unused",
                "--output",
                str(output),
            ]
        )
        == 2
    )

    stdout = capsys.readouterr().out
    assert "Infinity" not in stdout
    assert json.loads(stdout) == streaming_stt_eval._SAFE_ERROR
    assert not output.exists()
