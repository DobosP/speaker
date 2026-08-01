from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from tests.streaming_stt_helpers import (
    scripted_case,
    stage_test_source_bundle,
    write_fixture,
)
from tools.streaming_stt.corpus import load_corpus
from tools.streaming_stt.manifest import (
    BoundArtifact,
    BoundFile,
    NEMOTRON_ADAPTER,
    PARAKEET_REALTIME_EOU_ADAPTER,
    NemotronConfig,
    ParakeetRealtimeEouConfig,
    load_worker_manifest,
)
from tools.streaming_stt.protocol import (
    FinalEvent,
    MAX_STDERR_BYTES,
    NATIVE_ENDPOINT_PROTOCOL_VERSION,
    NativeFinalEvent,
    PartialEvent,
    PROTOCOL_VERSION,
    PcmInput,
    ResourceUsage,
    StreamConfig,
    TranscribeRequest,
)
from tools.streaming_stt.runtime_receipt import PARAKEET_RUNTIME_TREE_LIMITS
from tools.streaming_stt.supervisor import (
    StreamingWorker,
    WorkerError,
    _await_parakeet_cgroup,
    _nemotron_bwrap_command,
    _parakeet_systemd_scope_command,
    _parse_unified_cgroup_path,
    _verified_user_runtime_directory,
    _verify_parakeet_cgroup_evidence,
)


def _worker(
    manifest_path: Path,
    scratch: Path,
    **kwargs,
) -> StreamingWorker:
    manifest = load_worker_manifest(manifest_path)
    bundle = stage_test_source_bundle(scratch, manifest.worker.path)
    return StreamingWorker(manifest, scratch, bundle, **kwargs)


def _request(
    scratch: Path,
    corpus_path: Path,
    digest: str,
    *,
    request_id: str = "case-0",
) -> TranscribeRequest:
    corpus = load_corpus(corpus_path)
    source = corpus.cases[0]
    path = scratch / "case.f32le"
    path.write_bytes(source.audio_bytes)
    path.chmod(0o600)
    return TranscribeRequest(
        request_id=request_id,
        pcm=PcmInput(
            path=path.resolve(),
            sha256=digest,
            samples=source.samples,
        ),
        stream=StreamConfig(
            chunk_samples=2,
            pace="burst",
            partial_interval_ms=100,
            tail_padding_samples=1,
        ),
    )


def _fixture(
    tmp_path: Path,
    *,
    hang_sec: float = 0.0,
    case_timeout_sec: float = 2.0,
) -> tuple[Path, Path, str]:
    manifest, corpus, digests = write_fixture(
        tmp_path,
        [
            {
                "values": [0.0, 0.1, -0.1, 0.0],
                "expected_text": "stop now",
                "commands": ["stop"],
            }
        ],
        [
            scripted_case(
                partials=[
                    (2, "stop", 50.0, 1.0),
                    (5, "stop now", 100.0, 1.5),
                ],
                final="stop now",
                elapsed_ms=150.0,
                hang_sec=hang_sec,
            )
        ],
        case_timeout_sec=case_timeout_sec,
    )
    return manifest, corpus, digests[0]


def _nemotron_manifest_for_launch(
    manifest_path: Path,
    tmp_path: Path,
):
    manifest = load_worker_manifest(manifest_path)
    venv_python = tmp_path / "candidate-venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_bytes(b"python")
    model_root = tmp_path / "model"
    model_root.mkdir()
    receipt = tmp_path / "runtime-receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    wheel_lock = tmp_path / "runtime-wheel-lock.json"
    wheel_lock.write_text("{}", encoding="utf-8")
    model_files = (
        ("model-config", "config.json"),
        ("model-generation-config", "generation_config.json"),
        ("model-weights", "model.safetensors"),
        ("model-processor-config", "processor_config.json"),
        ("model-tokenizer", "tokenizer.json"),
        ("model-tokenizer-config", "tokenizer_config.json"),
    )
    model_artifacts = []
    for index, (name, filename) in enumerate(model_files, start=1):
        path = model_root / filename
        path.write_bytes(bytes((index,)) * (index + 1))
        model_artifacts.append(
            BoundArtifact(
                path,
                f"{index:x}".rjust(64, "0"),
                path.stat().st_size,
                name,
            )
        )
    artifacts = (
        BoundArtifact(receipt, "a" * 64, receipt.stat().st_size, "runtime-receipt"),
        BoundArtifact(
            wheel_lock,
            "d" * 64,
            wheel_lock.stat().st_size,
            "runtime-wheel-lock",
        ),
        BoundArtifact(
            venv_python.parent.parent / "pyvenv.cfg",
            "b" * 64,
            1,
            "venv-marker",
        ),
        *model_artifacts,
    )
    return replace(
        manifest,
        schema_version=3,
        adapter=NEMOTRON_ADAPTER,
        python=BoundFile(venv_python, "c" * 64, venv_python.stat().st_size),
        artifacts=artifacts,
        adapter_config=NemotronConfig(),
    )


def _parakeet_config() -> ParakeetRealtimeEouConfig:
    return ParakeetRealtimeEouConfig()


def _parakeet_manifest_for_launch(
    manifest_path: Path,
    tmp_path: Path,
):
    return replace(
        _nemotron_manifest_for_launch(manifest_path, tmp_path),
        schema_version=5,
        adapter=PARAKEET_REALTIME_EOU_ADAPTER,
        adapter_config=_parakeet_config(),
    )


def _parakeet_request(
    scratch: Path,
    corpus_path: Path,
    digest: str,
    *,
    protocol_version: int = NATIVE_ENDPOINT_PROTOCOL_VERSION,
) -> TranscribeRequest:
    request = _request(scratch, corpus_path, digest)
    return replace(
        request,
        stream=replace(
            request.stream,
            chunk_samples=1_280,
            tail_padding_samples=48_000,
        ),
        protocol_version=protocol_version,
    )


def _parakeet_native_final(request: TranscribeRequest) -> NativeFinalEvent:
    source_samples = request.pcm.samples
    tail_samples = 1_280
    samples_seen = source_samples + tail_samples
    return NativeFinalEvent(
        request_id=request.request_id,
        seq=0,
        text="stop now",
        samples_seen=samples_seen,
        elapsed_ms=90.0,
        finalization_ms=2.0,
        compute_ms=8.0,
        audio_seconds=source_samples / request.pcm.sample_rate,
        chunks=2,
        deadline_misses=0,
        max_backlog_ms=0.0,
        resources=ResourceUsage(rss_mb=1.0, threads=1, vram_mb=1.0),
        model_padding_samples=2 * 1_280 - samples_seen,
        source_samples=source_samples,
        source_samples_consumed=source_samples,
        declared_tail_samples=request.stream.tail_padding_samples,
        tail_samples_consumed=tail_samples,
        endpoint_reason="eou",
        native_endpoint=True,
        endpoint_probability=0.75,
        endpoint_sample=2 * 1_280,
        endpoint_latency_ms=(tail_samples + 2 * 1_280 - samples_seen) / 16.0,
        authoritative=True,
    )


def _stub_parakeet_worker(
    manifest_path: Path,
    scratch: Path,
) -> StreamingWorker:
    manifest = replace(
        load_worker_manifest(manifest_path),
        schema_version=5,
        adapter=PARAKEET_REALTIME_EOU_ADAPTER,
        adapter_config=_parakeet_config(),
    )
    bundle = stage_test_source_bundle(scratch, manifest.worker.path)
    worker = StreamingWorker(manifest, scratch, bundle)
    worker.ready = object()
    return worker


def test_real_worker_is_a_new_process_session_and_closes_cleanly(tmp_path):
    manifest_path, corpus_path, digest = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)

    ready = worker.start()
    pid = worker.pid
    assert pid is not None
    if os.name == "posix":
        assert os.getpgid(pid) == pid
    trace = worker.transcribe(_request(scratch, corpus_path, digest))
    worker.close()

    assert ready.adapter == "fake-json-v1"
    assert [partial.seq for partial in trace.partials] == [0, 1]
    assert trace.final.seq == 2
    assert trace.partials[-1].samples_seen == trace.final.samples_seen == 5
    assert trace.final.chunks == 3
    assert worker._process is not None  # noqa: SLF001 - lifecycle proof
    assert worker._process.poll() == 0  # noqa: SLF001 - lifecycle proof


@pytest.mark.parametrize(
    ("chunks", "model_padding_samples", "accepted"),
    [
        (1, 4_035, True),
        (3, 4_035, False),
        (1, 0, False),
    ],
)
def test_nemotron_supervisor_validates_native_final_evidence(
    tmp_path,
    monkeypatch,
    chunks,
    model_padding_samples,
    accepted,
):
    manifest_path, corpus_path, digest = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    manifest = replace(
        load_worker_manifest(manifest_path),
        adapter=NEMOTRON_ADAPTER,
        adapter_config=NemotronConfig(),
    )
    bundle = stage_test_source_bundle(scratch, manifest.worker.path)
    worker = StreamingWorker(manifest, scratch, bundle)
    worker.ready = object()
    request = _request(scratch, corpus_path, digest)
    event = FinalEvent(
        request_id=request.request_id,
        seq=0,
        text="stop now",
        samples_seen=5,
        elapsed_ms=10.0,
        finalization_ms=2.0,
        compute_ms=8.0,
        audio_seconds=4 / 16_000,
        chunks=chunks,
        deadline_misses=0,
        max_backlog_ms=0.0,
        resources=ResourceUsage(rss_mb=1.0, threads=1, vram_mb=1.0),
        model_padding_samples=model_padding_samples,
    )
    monkeypatch.setattr(worker, "_verify_pcm", lambda _request: None)
    monkeypatch.setattr(worker, "_send", lambda _message: None)
    monkeypatch.setattr(worker, "_next_event", lambda _timeout: event)

    if accepted:
        trace = worker.transcribe(request)
        assert trace.final == event
    else:
        with pytest.raises(WorkerError, match="worker_protocol"):
            worker.transcribe(request)


@pytest.mark.parametrize(
    "endpoint_kind",
    ["tail-eou", "complete-eob", "early-eob", "exhausted"],
)
def test_parakeet_supervisor_accepts_exact_native_consumption_evidence(
    tmp_path,
    monkeypatch,
    endpoint_kind,
):
    manifest_path, corpus_path, digest = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _stub_parakeet_worker(manifest_path, scratch)
    request = _parakeet_request(scratch, corpus_path, digest)
    event = _parakeet_native_final(request)
    if endpoint_kind == "complete-eob":
        event = replace(
            event,
            endpoint_reason="eob",
            endpoint_latency_ms=None,
            authoritative=False,
        )
    elif endpoint_kind == "early-eob":
        event = replace(
            event,
            samples_seen=3,
            chunks=1,
            model_padding_samples=1_277,
            source_samples_consumed=3,
            tail_samples_consumed=0,
            endpoint_reason="eob",
            endpoint_sample=1_280,
            endpoint_latency_ms=None,
            authoritative=False,
        )
    elif endpoint_kind == "exhausted":
        samples_seen = request.pcm.samples + request.stream.tail_padding_samples
        event = replace(
            event,
            samples_seen=samples_seen,
            elapsed_ms=3_010.0,
            chunks=38,
            model_padding_samples=38 * 1_280 - samples_seen,
            tail_samples_consumed=request.stream.tail_padding_samples,
            endpoint_reason="tail_exhausted",
            native_endpoint=False,
            endpoint_probability=None,
            endpoint_sample=None,
            endpoint_latency_ms=None,
            authoritative=False,
        )
    monkeypatch.setattr(worker, "_verify_pcm", lambda _request: None)
    monkeypatch.setattr(worker, "_send", lambda _message: None)
    monkeypatch.setattr(worker, "_next_event", lambda _timeout: event)

    trace = worker.transcribe(request)

    assert type(trace.final) is NativeFinalEvent
    assert trace.final == event
    if endpoint_kind == "complete-eob":
        assert trace.final.source_samples_consumed == trace.final.source_samples
        assert trace.final.endpoint_latency_ms is None
        assert trace.final.authoritative is False


@pytest.mark.parametrize(
    "mutation",
    [
        {"endpoint_sample": 1},
        {"chunks": 1},
        {"model_padding_samples": 0},
        {"source_samples": 5},
        {"declared_tail_samples": 47_999},
        {
            "samples_seen": 4,
            "chunks": 1,
            "model_padding_samples": 1_276,
            "source_samples_consumed": 3,
            "tail_samples_consumed": 1,
            "endpoint_sample": 4,
            "endpoint_latency_ms": None,
            "authoritative": False,
        },
        {
            "endpoint_reason": "tail_exhausted",
            "native_endpoint": False,
            "endpoint_probability": None,
            "endpoint_sample": None,
            "endpoint_latency_ms": None,
            "authoritative": False,
        },
    ],
)
def test_parakeet_supervisor_rejects_inconsistent_native_evidence(
    tmp_path,
    monkeypatch,
    mutation,
):
    manifest_path, corpus_path, digest = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _stub_parakeet_worker(manifest_path, scratch)
    request = _parakeet_request(scratch, corpus_path, digest)
    event = replace(_parakeet_native_final(request), **mutation)
    monkeypatch.setattr(worker, "_verify_pcm", lambda _request: None)
    monkeypatch.setattr(worker, "_send", lambda _message: None)
    monkeypatch.setattr(worker, "_next_event", lambda _timeout: event)

    with pytest.raises(WorkerError, match="worker_protocol"):
        worker.transcribe(request)


def test_parakeet_supervisor_rejects_partial_beyond_native_endpoint(
    tmp_path,
    monkeypatch,
):
    manifest_path, corpus_path, digest = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _stub_parakeet_worker(manifest_path, scratch)
    request = _parakeet_request(scratch, corpus_path, digest)
    final = replace(_parakeet_native_final(request), seq=1)
    partial = PartialEvent(
        request_id=request.request_id,
        seq=0,
        text="stop",
        samples_seen=final.samples_seen + 1,
        elapsed_ms=50.0,
        decode_ms=1.0,
        protocol_version=NATIVE_ENDPOINT_PROTOCOL_VERSION,
    )
    events = iter((partial, final))
    monkeypatch.setattr(worker, "_verify_pcm", lambda _request: None)
    monkeypatch.setattr(worker, "_send", lambda _message: None)
    monkeypatch.setattr(worker, "_next_event", lambda _timeout: next(events))

    with pytest.raises(WorkerError, match="worker_protocol"):
        worker.transcribe(request)


def test_parakeet_supervisor_rejects_legacy_request_and_terminal(
    tmp_path,
    monkeypatch,
):
    manifest_path, corpus_path, digest = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _stub_parakeet_worker(manifest_path, scratch)
    legacy_request = _parakeet_request(
        scratch,
        corpus_path,
        digest,
        protocol_version=PROTOCOL_VERSION,
    )
    monkeypatch.setattr(worker, "_verify_pcm", lambda _request: None)
    monkeypatch.setattr(worker, "_send", lambda _message: None)

    with pytest.raises(WorkerError, match="invalid_request"):
        worker.transcribe(legacy_request)

    request = replace(
        legacy_request,
        protocol_version=NATIVE_ENDPOINT_PROTOCOL_VERSION,
    )
    native = _parakeet_native_final(request)
    legacy_final = FinalEvent(
        request_id=request.request_id,
        seq=0,
        text=native.text,
        samples_seen=request.pcm.samples + request.stream.tail_padding_samples,
        elapsed_ms=native.elapsed_ms,
        finalization_ms=native.finalization_ms,
        compute_ms=native.compute_ms,
        audio_seconds=native.audio_seconds,
        chunks=38,
        deadline_misses=0,
        max_backlog_ms=0.0,
        resources=native.resources,
    )
    monkeypatch.setattr(worker, "_next_event", lambda _timeout: legacy_final)

    with pytest.raises(WorkerError, match="worker_protocol"):
        worker.transcribe(request)


def test_worker_launch_argv_is_exact_isolated_and_bytecode_free(tmp_path):
    manifest_path, _, _ = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    captured: dict[str, object] = {}

    def rejecting_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        raise OSError

    worker = _worker(
        manifest_path,
        scratch,
        popen_factory=rejecting_popen,
    )
    with pytest.raises(WorkerError, match="worker_start"):
        worker.start()

    manifest = worker.manifest
    bundle_fd = worker._bundle_descriptor  # noqa: SLF001 - exact launch proof
    assert bundle_fd == -1
    command = captured["command"]
    assert isinstance(command, list)
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    bundle_fd, worker_fd, python_fd = kwargs["pass_fds"]
    assert command == [
        str(manifest.python.path),
        "-I",
        "-S",
        "-B",
        f"/proc/self/fd/{worker_fd}",
        "--manifest",
        str(manifest.path),
        "--scratch-root",
        str(scratch.resolve()),
        "--source-bundle-manifest",
        f"/proc/self/fd/{bundle_fd}/source-bundle.json",
        "--source-bundle-sha256",
        worker.source_bundle.tree_sha256,
        "--source-bundle-fd",
        str(bundle_fd),
    ]
    assert str(manifest.worker.path) not in command
    assert kwargs["cwd"] == str(scratch.resolve())
    assert kwargs["executable"] == f"/proc/self/fd/{python_fd}"
    assert kwargs["start_new_session"] is True
    assert kwargs["close_fds"] is True
    assert len(kwargs["pass_fds"]) == 3
    assert "PATH" not in kwargs["env"]
    assert "shell" not in kwargs


def test_nemotron_bwrap_argv_has_closed_mount_and_namespace_boundary(tmp_path):
    manifest_path, _, _ = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    source_bundle = stage_test_source_bundle(
        scratch,
        load_worker_manifest(manifest_path).worker.path,
    )
    manifest = _nemotron_manifest_for_launch(manifest_path, tmp_path)
    system_paths = (
        Path("/usr"),
        Path("/lib"),
        Path("/lib64"),
        Path("/etc/ld.so.cache"),
        Path("/etc/passwd"),
        Path("/etc/group"),
        Path("/sys/bus/pci/drivers/nvidia"),
        Path("/sys/devices/pci0000:00/0000:01:00.0"),
        Path("/sys/module/nvidia"),
        Path("/sys/module/nvidia_uvm"),
    )
    device_nodes = (
        Path("/dev/nvidia0"),
        Path("/dev/nvidiactl"),
        Path("/dev/nvidia-uvm"),
        Path("/dev/nvidia-uvm-tools"),
    )

    command = _nemotron_bwrap_command(
        manifest,
        scratch.resolve(),
        source_bundle,
        bundle_descriptor=41,
        worker_descriptor=42,
        python_descriptor=43,
        device_nodes=device_nodes,
        system_ro_paths=system_paths,
        proc_driver_available=True,
    )

    assert command[:19] == [
        "/usr/bin/bwrap",
        "--unshare-user-try",
        "--unshare-ipc",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--die-with-parent",
        "--new-session",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/lib",
        "/lib",
    ]
    assert command.count("--unshare-pid") == 1
    assert command.count("--unshare-net") == 1
    assert command.count("--die-with-parent") == 1
    assert command.count("--new-session") == 1
    assert command.count("--bind") == 1
    scratch_index = command.index("--bind")
    assert command[scratch_index : scratch_index + 3] == [
        "--bind",
        str(scratch.resolve()),
        str(scratch.resolve()),
    ]
    assert "--ro-bind-try" not in command
    assert "--dev-bind-try" not in command
    assert [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--dev-bind"
    ] == [str(path) for path in device_nodes]
    assert all(
        command[index + 2] == command[index + 1]
        for index, value in enumerate(command)
        if value == "--dev-bind"
    )

    ro_pairs = {
        (command[index + 1], command[index + 2])
        for index, value in enumerate(command)
        if value == "--ro-bind"
    }
    assert {(str(path), str(path)) for path in system_paths} <= ro_pairs
    assert (
        str(Path("/proc/driver/nvidia")),
        str(Path("/proc/driver/nvidia")),
    ) in ro_pairs
    assert (
        str(source_bundle.root),
        str(source_bundle.root),
    ) in ro_pairs
    assert (
        str(source_bundle.worker_path),
        str(manifest.worker.path),
    ) in ro_pairs
    assert (str(manifest.path), str(manifest.path)) in ro_pairs
    assert (
        str(manifest.python.path.parent.parent),
        str(manifest.python.path.parent.parent),
    ) in ro_pairs
    model_root = manifest.artifact_by_name["model-config"].path.parent
    assert (str(model_root), str(model_root)) in ro_pairs
    receipt = manifest.artifact_by_name["runtime-receipt"].path
    assert (str(receipt), str(receipt)) in ro_pairs
    wheel_lock = manifest.artifact_by_name["runtime-wheel-lock"].path
    assert (str(wheel_lock), str(wheel_lock)) in ro_pairs
    assert ("/", "/") not in ro_pairs
    assert str(manifest.worker.path) not in {
        source for source, _destination in ro_pairs
    }

    separator = command.index("--")
    assert command[separator - 4 : separator] == [
        "--chdir",
        str(scratch.resolve()),
        "--argv0",
        str(manifest.python.path),
    ]
    assert command[separator + 1 :] == [
        "/proc/self/fd/43",
        "-I",
        "-S",
        "-B",
        "/proc/self/fd/42",
        "--manifest",
        str(manifest.path),
        "--scratch-root",
        str(scratch.resolve()),
        "--source-bundle-manifest",
        "/proc/self/fd/41/source-bundle.json",
        "--source-bundle-sha256",
        source_bundle.tree_sha256,
        "--source-bundle-fd",
        "41",
    ]


def test_nemotron_start_executes_verified_bwrap_fd_in_outer_process_group(
    tmp_path,
    monkeypatch,
):
    manifest_path, _, _ = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)
    worker.manifest = _nemotron_manifest_for_launch(manifest_path, tmp_path)
    captured: dict[str, object] = {}

    monkeypatch.setattr(worker, "_verify_bound_files", lambda: None)
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._nemotron_device_nodes",
        lambda: (Path("/dev/nvidia0"),),
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._nemotron_system_ro_paths",
        lambda: (Path("/usr"),),
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._NVIDIA_PROC_DRIVER",
        tmp_path / "missing-proc-driver",
    )

    def open_bwrap():
        return os.open("/usr/bin/bwrap", os.O_RDONLY)

    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._open_bwrap_descriptor",
        open_bwrap,
    )
    monkeypatch.setattr(
        worker,
        "_open_bound_python_descriptor",
        lambda: os.open("/usr/bin/python3", os.O_RDONLY),
    )

    def rejecting_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        raise OSError

    worker._popen_factory = rejecting_popen  # noqa: SLF001 - launch seam
    with pytest.raises(WorkerError, match="worker_start"):
        worker.start()

    command = captured["command"]
    kwargs = captured["kwargs"]
    assert isinstance(command, list)
    assert isinstance(kwargs, dict)
    assert command[0] == "/usr/bin/bwrap"
    assert kwargs["start_new_session"] is True
    assert kwargs["close_fds"] is True
    assert len(kwargs["pass_fds"]) == 4
    bwrap_fd = kwargs["pass_fds"][-1]
    assert kwargs["executable"] == f"/proc/self/fd/{bwrap_fd}"
    assert worker._bundle_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._worker_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._python_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._bwrap_descriptor == -1  # noqa: SLF001 - cleanup proof


def test_parakeet_scope_command_has_exact_hard_limits_and_pinned_inner_exec():
    command = _parakeet_systemd_scope_command(
        [
            "/usr/bin/bwrap",
            "--unshare-net",
            "--die-with-parent",
            "--",
            "/proc/self/fd/9",
        ],
        bwrap_descriptor=42,
    )

    assert command == [
        "/usr/bin/systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "--no-ask-password",
        "--expand-environment=no",
        "--slice-inherit",
        "--nice=15",
        "--description=speaker-parakeet-realtime-eou-worker",
        "--property",
        "MemoryHigh=6442450944",
        "--property",
        "MemoryMax=8589934592",
        "--property",
        "MemorySwapMax=0",
        "--property",
        "CPUQuota=200%",
        "--property",
        "TasksMax=128",
        "--property",
        "OOMPolicy=kill",
        "--",
        "/proc/self/fd/42",
        "--unsetenv",
        "XDG_RUNTIME_DIR",
        "--unsetenv",
        "INVOCATION_ID",
        "--unshare-net",
        "--die-with-parent",
        "--",
        "/proc/self/fd/9",
    ]
    assert "/usr/bin/bwrap" not in command
    assert "--pipe" not in command
    assert "--wait" not in command
    assert "--service-type" not in command


@pytest.mark.parametrize(
    ("inner", "descriptor"),
    [
        ([], 1),
        (["/usr/bin/bwrap"], -1),
        (["/alternate/bwrap"], 1),
        (["/usr/bin/bwrap", "bad\x00value"], 1),
    ],
)
def test_parakeet_scope_command_rejects_unpinned_or_unsafe_inputs(
    inner,
    descriptor,
):
    with pytest.raises(WorkerError, match="worker_prerequisite"):
        _parakeet_systemd_scope_command(inner, bwrap_descriptor=descriptor)


def test_parakeet_start_executes_pinned_scope_then_pinned_bwrap(
    tmp_path,
    monkeypatch,
):
    manifest_path, _, _ = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)
    worker.manifest = _parakeet_manifest_for_launch(manifest_path, tmp_path)
    captured: dict[str, object] = {}
    runtime_directory = tmp_path / "user-runtime"

    monkeypatch.setattr(worker, "_verify_bound_files", lambda: None)
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._nemotron_device_nodes",
        lambda: (Path("/dev/nvidia0"),),
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._nemotron_system_ro_paths",
        lambda: (Path("/usr"),),
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._NVIDIA_PROC_DRIVER",
        tmp_path / "missing-proc-driver",
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._verified_user_runtime_directory",
        lambda: runtime_directory,
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._open_bwrap_descriptor",
        lambda: os.open("/usr/bin/bwrap", os.O_RDONLY),
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._open_systemd_run_descriptor",
        lambda: os.open("/usr/bin/systemd-run", os.O_RDONLY),
    )
    monkeypatch.setattr(
        worker,
        "_open_bound_python_descriptor",
        lambda: os.open("/usr/bin/python3", os.O_RDONLY),
    )

    def rejecting_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        raise OSError

    worker._popen_factory = rejecting_popen  # noqa: SLF001 - launch seam
    with pytest.raises(WorkerError, match="worker_start"):
        worker.start()

    command = captured["command"]
    kwargs = captured["kwargs"]
    assert isinstance(command, list)
    assert isinstance(kwargs, dict)
    bundle_fd, worker_fd, python_fd, bwrap_fd, systemd_fd = kwargs["pass_fds"]
    assert command[0] == "/usr/bin/systemd-run"
    assert command[command.index("--") + 1] == f"/proc/self/fd/{bwrap_fd}"
    assert command.count("--") == 2
    assert "/usr/bin/bwrap" not in command
    assert kwargs["executable"] == f"/proc/self/fd/{systemd_fd}"
    assert kwargs["start_new_session"] is True
    assert kwargs["close_fds"] is True
    assert len({bundle_fd, worker_fd, python_fd, bwrap_fd, systemd_fd}) == 5
    assert kwargs["env"]["XDG_RUNTIME_DIR"] == str(runtime_directory)
    assert kwargs["env"]["PATH"] == "/usr/bin"
    assert "DBUS_SESSION_BUS_ADDRESS" not in kwargs["env"]
    assert "INVOCATION_ID" not in kwargs["env"]
    assert "shell" not in kwargs
    assert worker._bundle_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._worker_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._python_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._bwrap_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._systemd_run_descriptor == -1  # noqa: SLF001


def test_supervisor_runtime_receipt_load_is_the_single_tree_scan(
    tmp_path,
    monkeypatch,
):
    manifest_path, _, _ = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)
    worker.manifest = _parakeet_manifest_for_launch(manifest_path, tmp_path)
    receipt_artifact = worker.manifest.artifact_by_name["runtime-receipt"]
    receipt = SimpleNamespace(digest=receipt_artifact.sha256)
    receipt_loads: list[tuple[object, ...]] = []
    digest_by_path = {
        worker.manifest.worker.path: worker.manifest.worker.sha256,
        worker.manifest.python.path: worker.manifest.python.sha256,
        **{
            artifact.path: artifact.sha256
            for artifact in worker.manifest.artifacts
        },
    }

    def digest(path, **_kwargs):
        return SimpleNamespace(sha256=digest_by_path[path])

    def load(path, *, expected_digest, limits):
        receipt_loads.append((path, expected_digest, limits))
        return receipt

    monkeypatch.setattr(
        "tools.streaming_stt.supervisor.hash_regular_bounded",
        digest,
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor.artifact_maximum_bytes",
        lambda *_args: 1024,
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor.load_runtime_tree_receipt",
        load,
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor.verify_venv_runtime_location",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._verify_nemotron_runtime_layout",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor.verify_source_bundle",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor.verify_runtime_tree_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("redundant runtime scan")
        ),
        raising=False,
    )

    worker._verify_bound_files()  # noqa: SLF001 - exact scan-count seam

    assert receipt_loads == [
        (
            receipt_artifact.path,
            receipt_artifact.sha256,
            PARAKEET_RUNTIME_TREE_LIMITS,
        )
    ]


def test_parakeet_refuses_missing_systemd_run_without_launching(
    tmp_path,
    monkeypatch,
):
    manifest_path, _, _ = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)
    worker.manifest = _parakeet_manifest_for_launch(manifest_path, tmp_path)
    launched = False

    monkeypatch.setattr(worker, "_verify_bound_files", lambda: None)
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._nemotron_device_nodes",
        lambda: (Path("/dev/nvidia0"),),
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._nemotron_system_ro_paths",
        lambda: (Path("/usr"),),
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._NVIDIA_PROC_DRIVER",
        tmp_path / "missing-proc-driver",
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._verified_user_runtime_directory",
        lambda: tmp_path / "user-runtime",
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._open_bwrap_descriptor",
        lambda: os.open("/usr/bin/bwrap", os.O_RDONLY),
    )

    def missing_systemd():
        raise WorkerError("worker_prerequisite")

    def popen(*_args, **_kwargs):
        nonlocal launched
        launched = True
        raise AssertionError

    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._open_systemd_run_descriptor",
        missing_systemd,
    )
    worker._popen_factory = popen  # noqa: SLF001 - launch seam

    with pytest.raises(WorkerError, match="worker_prerequisite"):
        worker.start()

    assert launched is False
    assert worker.pid is None
    assert worker._bundle_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._worker_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._python_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._bwrap_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._systemd_run_descriptor == -1  # noqa: SLF001


def _write_parakeet_cgroup_tree(
    root: Path,
    *,
    overrides: dict[str, str] | None = None,
) -> tuple[bytes, Path]:
    root.mkdir()
    (root / "cgroup.controllers").write_text(
        "cpuset cpu io memory pids\n",
        encoding="ascii",
    )
    relative = Path(
        "user.slice/user-1000.slice/user@1000.service/app.slice/run-r1234.scope"
    )
    scope = root / relative
    scope.mkdir(parents=True)
    values = {
        "memory.high": "6442450944",
        "memory.max": "8589934592",
        "memory.swap.max": "0",
        "memory.oom.group": "1",
        "cpu.max": "200000 100000",
        "pids.max": "128",
    }
    values.update(overrides or {})
    for name, value in values.items():
        (scope / name).write_text(value + "\n", encoding="ascii")
    return f"0::/{relative}\n".encode("ascii"), scope


def test_parakeet_cgroup_v2_evidence_is_exact_and_path_independent(tmp_path):
    payload, _scope = _write_parakeet_cgroup_tree(tmp_path / "cgroup")

    evidence = _verify_parakeet_cgroup_evidence(
        payload,
        cgroup_root=tmp_path / "cgroup",
    )

    assert evidence == {
        "kind": "systemd-user-scope-cgroup-v2",
        "memory_high_bytes": 6442450944,
        "memory_max_bytes": 8589934592,
        "memory_swap_max_bytes": 0,
        "cpu_quota_percent": 200,
        "tasks_max": 128,
        "oom_policy": "kill",
        "verified": True,
    }
    assert "run-r1234.scope" not in repr(evidence)


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"0::/\n",
        b"1:name=/run-r1234.scope\n",
        b"0::/run-r1234.scope\nextra\n",
        b"0::/../run-r1234.scope\n",
        b"0::/app.slice/not-a-run-scope.scope\n",
        b"0::/app.slice/run-r1234.scope",
        b"0::/app.slice/run-r1234.scope\r\n",
        b"0::/app.slice/run-r\xff.scope\n",
    ],
)
def test_parakeet_cgroup_path_rejects_non_v2_or_unsafe_values(payload):
    with pytest.raises(WorkerError, match="worker_prerequisite"):
        _parse_unified_cgroup_path(payload)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("memory.high", "max"),
        ("memory.max", "8589934591"),
        ("memory.swap.max", "1"),
        ("memory.oom.group", "0"),
        ("cpu.max", "max 100000"),
        ("cpu.max", "100000 100000"),
        ("pids.max", "129"),
    ],
)
def test_parakeet_cgroup_evidence_rejects_any_relaxed_or_missing_bound(
    tmp_path,
    name,
    value,
):
    payload, _scope = _write_parakeet_cgroup_tree(
        tmp_path / "cgroup",
        overrides={name: value},
    )

    with pytest.raises(WorkerError, match="worker_prerequisite"):
        _verify_parakeet_cgroup_evidence(
            payload,
            cgroup_root=tmp_path / "cgroup",
        )


def test_parakeet_cgroup_evidence_rejects_missing_controller(tmp_path):
    root = tmp_path / "cgroup"
    payload, _scope = _write_parakeet_cgroup_tree(root)
    (root / "cgroup.controllers").write_text("cpu memory\n", encoding="ascii")

    with pytest.raises(WorkerError, match="worker_prerequisite"):
        _verify_parakeet_cgroup_evidence(payload, cgroup_root=root)


def test_parakeet_cgroup_wait_retries_migration_then_returns_evidence(monkeypatch):
    expected = {"verified": True}
    attempts = 0
    sleeps: list[float] = []

    class Process:
        pid = 123

        @staticmethod
        def poll():
            return None

    def read(_pid):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise WorkerError("worker_prerequisite")
        return expected

    ticks = iter((0.0, 0.0))
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._read_parakeet_cgroup_evidence",
        read,
    )

    result = _await_parakeet_cgroup(
        Process(),  # type: ignore[arg-type]
        monotonic=lambda: next(ticks),
        sleeper=sleeps.append,
    )

    assert result == expected
    assert attempts == 2
    assert sleeps == [0.01]


def test_parakeet_cgroup_wait_fails_immediately_if_scope_launcher_dies():
    class Process:
        pid = 123

        @staticmethod
        def poll():
            return 1

    with pytest.raises(WorkerError, match="worker_prerequisite"):
        _await_parakeet_cgroup(Process())  # type: ignore[arg-type]


def test_parakeet_cgroup_wait_fails_closed_at_setup_deadline(monkeypatch):
    class Process:
        pid = 123

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._read_parakeet_cgroup_evidence",
        lambda _pid: (_ for _ in ()).throw(WorkerError("worker_prerequisite")),
    )
    ticks = iter((0.0, 5.0))

    with pytest.raises(WorkerError, match="worker_prerequisite"):
        _await_parakeet_cgroup(
            Process(),  # type: ignore[arg-type]
            monotonic=lambda: next(ticks),
            sleeper=lambda _delay: None,
        )


def test_user_runtime_directory_requires_private_owned_exact_directory(tmp_path):
    root = tmp_path / "run-user"
    runtime = root / str(os.getuid())
    runtime.mkdir(parents=True)
    runtime.chmod(0o700)

    assert _verified_user_runtime_directory(root=root) == runtime

    runtime.chmod(0o750)
    with pytest.raises(WorkerError, match="worker_prerequisite"):
        _verified_user_runtime_directory(root=root)


def test_nemotron_refuses_missing_gpu_before_runtime_verification(
    tmp_path,
    monkeypatch,
):
    manifest_path, _, _ = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)
    worker.manifest = replace(
        worker.manifest,
        schema_version=3,
        adapter=NEMOTRON_ADAPTER,
        adapter_config=NemotronConfig(),
    )
    verified = False

    def record_verification():
        nonlocal verified
        verified = True

    def no_gpu():
        raise WorkerError("worker_prerequisite")

    monkeypatch.setattr(worker, "_verify_bound_files", record_verification)
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._nemotron_device_nodes",
        no_gpu,
    )

    with pytest.raises(WorkerError, match="worker_prerequisite"):
        worker.start()

    assert verified is False
    assert worker.pid is None


def test_site_free_launch_never_processes_runtime_pth_before_explicit_import(
    tmp_path,
):
    venv = tmp_path / "candidate-venv"
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(sys.executable).resolve(strict=True))
    (venv / "pyvenv.cfg").write_text(
        "home = /usr/bin\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )
    runtime = (
        venv
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    runtime.mkdir(parents=True)
    sentinel = tmp_path / "pth-executed"
    (runtime / "candidate-startup.pth").write_text(
        f"import pathlib; pathlib.Path({str(sentinel)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    (runtime / "candidate_runtime.py").write_text("VALUE = 1\n", encoding="utf-8")

    control = subprocess.run(
        [str(python), "-I", "-B", "-c", "pass"],
        check=False,
        capture_output=True,
        timeout=5.0,
    )
    assert control.returncode == 0
    assert sentinel.read_text(encoding="utf-8") == "executed"
    sentinel.unlink()

    probe = (
        "import pathlib,sys;"
        "runtime=pathlib.Path(sys.argv[1]);"
        "sentinel=pathlib.Path(sys.argv[2]);"
        "assert not sentinel.exists();"
        "sys.path.insert(0,str(runtime));"
        "import candidate_runtime;"
        "assert candidate_runtime.VALUE==1;"
        "assert not sentinel.exists()"
    )
    isolated = subprocess.run(
        [str(python), "-I", "-S", "-B", "-c", probe, str(runtime), str(sentinel)],
        check=False,
        capture_output=True,
        timeout=5.0,
    )

    assert isolated.returncode == 0
    assert isolated.stdout == b""
    assert isolated.stderr == b""
    assert not sentinel.exists()


@pytest.mark.skipif(os.name != "posix", reason="process-group proof is POSIX")
@pytest.mark.parametrize("failed_start", [1, 2])
def test_drainer_start_failure_reaps_worker_and_closes_launch_descriptors(
    tmp_path,
    monkeypatch,
    failed_start,
):
    manifest_path, _, _ = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)
    original_start = threading.Thread.start
    starts = 0

    def fail_selected_start(thread):
        nonlocal starts
        starts += 1
        if starts == failed_start:
            raise RuntimeError
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_selected_start)
    monkeypatch.setattr("tools.streaming_stt.supervisor._TERM_GRACE_SEC", 0.1)
    monkeypatch.setattr("tools.streaming_stt.supervisor._KILL_GRACE_SEC", 0.5)

    with pytest.raises(WorkerError, match="worker_start"):
        worker.start()

    pid = worker.pid
    assert pid is not None
    assert worker._process is not None  # noqa: SLF001 - cleanup proof
    assert worker._process.poll() is not None  # noqa: SLF001 - cleanup proof
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)
    assert worker._stdout_thread is not None  # noqa: SLF001 - cleanup proof
    assert not worker._stdout_thread.is_alive()  # noqa: SLF001 - cleanup proof
    if worker._stderr_thread is not None:  # noqa: SLF001 - cleanup proof
        assert not worker._stderr_thread.is_alive()  # noqa: SLF001
    assert (worker._stderr_thread is None) is (failed_start == 1)  # noqa: SLF001
    assert worker._bundle_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._worker_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._python_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._process.stdin is not None  # noqa: SLF001 - cleanup proof
    assert worker._process.stdout is not None  # noqa: SLF001 - cleanup proof
    assert worker._process.stderr is not None  # noqa: SLF001 - cleanup proof
    assert worker._process.stdin.closed  # noqa: SLF001 - cleanup proof
    assert worker._process.stdout.closed  # noqa: SLF001 - cleanup proof
    assert worker._process.stderr.closed  # noqa: SLF001 - cleanup proof


def test_pipe_cleanup_skips_output_with_live_drainer(tmp_path):
    manifest_path, _, _ = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)

    class CloseProbe:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class ProcessProbe:
        def __init__(self):
            self.stdin = CloseProbe()
            self.stdout = CloseProbe()
            self.stderr = CloseProbe()

    release = threading.Event()
    started = threading.Event()

    def live_drainer():
        started.set()
        release.wait()

    process = ProcessProbe()
    drainer = threading.Thread(target=live_drainer, daemon=True)
    drainer.start()
    assert started.wait(timeout=1.0)
    worker._process = process  # type: ignore[assignment]  # noqa: SLF001
    worker._stdout_thread = drainer  # noqa: SLF001 - bounded close seam

    try:
        worker._close_process_pipes()  # noqa: SLF001 - bounded close seam

        assert process.stdin.closed
        assert not process.stdout.closed
        assert process.stderr.closed
    finally:
        release.set()
    drainer.join(timeout=1.0)
    assert not drainer.is_alive()

    worker._close_process_pipes()  # noqa: SLF001 - stopped drainer closes

    assert process.stdout.closed


def test_worker_rejects_group_readable_scratch_before_launch(tmp_path):
    manifest_path, _, _ = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    scratch.chmod(0o750)
    worker = _worker(manifest_path, scratch)

    with pytest.raises(WorkerError, match="worker_prerequisite"):
        worker.start()

    assert worker.pid is None


def test_case_timeout_terminates_and_reaps_the_worker_group(tmp_path, monkeypatch):
    manifest_path, corpus_path, digest = _fixture(
        tmp_path,
        hang_sec=5.0,
        case_timeout_sec=0.1,
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)
    worker.start()
    pid = worker.pid
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._TERM_GRACE_SEC",
        0.2,
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._KILL_GRACE_SEC",
        0.5,
    )

    with pytest.raises(WorkerError, match="worker_timeout"):
        worker.transcribe(_request(scratch, corpus_path, digest))

    assert pid is not None
    assert worker._process is not None  # noqa: SLF001 - lifecycle proof
    assert worker._process.poll() is not None  # noqa: SLF001 - lifecycle proof
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.killpg(pid, 0)


@pytest.mark.skipif(os.name != "posix", reason="process-group proof is POSIX")
def test_timeout_kills_pipe_holding_descendant_before_joining_drainers(
    tmp_path,
    monkeypatch,
):
    worker_script = tmp_path / "pipe-holder-worker.py"
    worker_script.write_text(
        "import hashlib,json,platform,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "manifest_path=Path(sys.argv[sys.argv.index('--manifest')+1])\n"
        "scratch=Path(sys.argv[sys.argv.index('--scratch-root')+1])\n"
        "raw=manifest_path.read_bytes()\n"
        "manifest=json.loads(raw)\n"
        f"ready={{'v':{PROTOCOL_VERSION},'type':'ready','model_id':manifest['model_id'],"
        "'manifest_sha256':hashlib.sha256(raw).hexdigest(),"
        "'source_bundle_sha256':"
        "sys.argv[sys.argv.index('--source-bundle-sha256')+1],"
        "'adapter':manifest['adapter'],'model_load_ms':0.0,"
        "'resources':{'rss_mb':1.0,'threads':1,'vram_mb':None},"
        "'runtime':{'python':platform.python_version(),'platform':sys.platform}}\n"
        "print(json.dumps(ready),flush=True)\n"
        "request=json.loads(sys.stdin.readline())\n"
        "marker=scratch/'descendant-ready'\n"
        'code="import signal,sys,time;'
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "open(sys.argv[1],'w').write('ready');time.sleep(60)\"\n"
        "child=subprocess.Popen([sys.executable,'-I','-B','-c',code,str(marker)])\n"
        "deadline=time.monotonic()+2.0\n"
        "while not marker.exists() and time.monotonic()<deadline:time.sleep(0.005)\n"
        "(scratch/'descendant-pid').write_text(str(child.pid))\n"
        f"partial={{'v':{PROTOCOL_VERSION},'type':'partial','id':request['id'],'seq':0,'text':'',"
        "'samples_seen':0,'elapsed_ms':0.0,'decode_ms':0.0}\n"
        "print(json.dumps(partial),flush=True)\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    manifest_path, corpus_path, digests = write_fixture(
        tmp_path,
        [{"values": [0.0], "expected_text": "x"}],
        [scripted_case(partials=[], final="x")],
        case_timeout_sec=0.3,
        worker_path=worker_script,
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)
    worker.start()
    pid = worker.pid
    monkeypatch.setattr("tools.streaming_stt.supervisor._TERM_GRACE_SEC", 0.1)
    monkeypatch.setattr("tools.streaming_stt.supervisor._KILL_GRACE_SEC", 1.0)

    with pytest.raises(WorkerError, match="worker_timeout"):
        worker.transcribe(_request(scratch, corpus_path, digests[0]))

    assert pid is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)
    assert worker._stdout_thread is not None  # noqa: SLF001 - lifecycle proof
    assert worker._stderr_thread is not None  # noqa: SLF001 - lifecycle proof
    assert not worker._stdout_thread.is_alive()  # noqa: SLF001
    assert not worker._stderr_thread.is_alive()  # noqa: SLF001


def test_second_request_is_rejected_while_one_case_is_active(tmp_path):
    manifest_path, corpus_path, digest = _fixture(
        tmp_path,
        hang_sec=1.0,
        case_timeout_sec=0.2,
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)
    worker.start()
    request = _request(scratch, corpus_path, digest)
    failures: list[WorkerError] = []

    def first_call() -> None:
        try:
            worker.transcribe(request)
        except WorkerError as exc:
            failures.append(exc)

    first = threading.Thread(target=first_call)
    first.start()
    deadline = time.monotonic() + 1.0
    while not worker._active and time.monotonic() < deadline:  # noqa: SLF001
        time.sleep(0.005)

    with pytest.raises(WorkerError, match="worker_state"):
        worker.transcribe(request)

    first.join(timeout=2.0)
    assert not first.is_alive()
    assert failures and failures[0].code == "worker_timeout"


def test_supervisor_rejects_pcm_outside_private_scratch_without_sending(
    tmp_path,
):
    manifest_path, corpus_path, digest = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    corpus = load_corpus(corpus_path)
    request = TranscribeRequest(
        request_id="outside",
        pcm=PcmInput(
            path=corpus.cases[0].source_path,
            sha256=digest,
            samples=corpus.cases[0].samples,
        ),
        stream=StreamConfig(2, "burst", 100, 0),
    )

    with _worker(manifest_path, scratch) as worker:
        with pytest.raises(WorkerError) as failure:
            worker.transcribe(request)

    assert str(failure.value) == "worker_prerequisite"
    assert str(corpus.cases[0].source_path) not in str(failure.value)


def test_worker_artifact_change_after_start_is_detected_on_close(tmp_path):
    manifest_path, _, _ = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    manifest = load_worker_manifest(manifest_path)
    bundle = stage_test_source_bundle(scratch, manifest.worker.path)
    worker = StreamingWorker(manifest, scratch, bundle)
    worker.start()
    artifact = manifest.artifact_by_name["fake-script"].path
    artifact.write_bytes(artifact.read_bytes() + b"x")

    with pytest.raises(WorkerError, match="worker_artifact_changed"):
        worker.close()


def test_staged_source_mutation_is_rejected_before_launch(tmp_path):
    manifest_path, _, _ = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)
    staged_adapter = (
        worker.source_bundle.root / "tools" / "streaming_stt" / "adapters" / "fake.py"
    )
    staged_adapter.chmod(0o600)
    staged_adapter.write_bytes(staged_adapter.read_bytes() + b"\n")

    with pytest.raises(WorkerError, match="worker_artifact_changed"):
        worker.start()

    assert worker.pid is None


@pytest.mark.skipif(os.name != "posix", reason="bounded process signal check is POSIX")
def test_dead_worker_is_detected_without_waiting_for_full_timeout(tmp_path):
    manifest_path, _, _ = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)
    worker.start()
    assert worker.pid is not None
    os.kill(worker.pid, 9)
    started = time.monotonic()

    with pytest.raises(WorkerError):
        worker._next_event(2.0)  # noqa: SLF001 - bounded EOF detection seam

    assert time.monotonic() - started < 1.0
    worker.close()


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
def test_worker_output_caps_fail_closed_without_pipe_deadlock(
    tmp_path,
    stream_name,
):
    worker_script = tmp_path / "noisy-worker.py"
    target = "stdout" if stream_name == "stdout" else "stderr"
    size = "4 * 1024 * 1024 + 1" if target == "stdout" else "256 * 1024 + 1"
    worker_script.write_text(
        "import sys,time\n"
        f"sys.{target}.buffer.write(b'x' * ({size}) + b'\\n')\n"
        f"sys.{target}.buffer.flush()\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    manifest_path, _, _ = write_fixture(
        tmp_path,
        [{"values": [0.0], "expected_text": "x"}],
        [scripted_case(partials=[], final="x")],
        worker_path=worker_script,
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)

    with pytest.raises(WorkerError, match="worker_output_limit"):
        worker.start()

    assert worker._process is not None  # noqa: SLF001 - lifecycle proof
    assert worker._process.poll() is not None  # noqa: SLF001 - lifecycle proof
    if stream_name == "stderr":
        assert worker.stderr_summary["truncated"] is True
        assert worker.stderr_summary["bytes"] == MAX_STDERR_BYTES + 1
