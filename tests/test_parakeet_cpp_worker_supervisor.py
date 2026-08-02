from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.streaming_stt.manifest import (
    BoundArtifact,
    BoundFile,
    PARAKEET_CPP_ADAPTER,
    PARAKEET_CPP_ARTIFACT_NAMES,
    ParakeetCppConfig,
    WorkerLimits,
    WorkerManifest,
)
from tools.streaming_stt.protocol import (
    PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION,
    ParakeetCppFinalEvent,
    ParakeetCppObservedEvent,
    PcmInput,
    ProtocolError,
    ResourceUsage,
    StreamConfig,
    TranscribeRequest,
    encode_message,
    parse_response,
)
from tools.streaming_stt.source_bundle import (
    WORKER_SOURCE_FILES,
    stage_worker_source_bundle,
)
from tools.streaming_stt.supervisor import (
    StreamingWorker,
    WorkerError,
    _NEMOTRON_CORE_RO_PATHS,
    _parakeet_cpp_bwrap_command,
    _parakeet_cpp_systemd_scope_command,
    _validate_parakeet_cpp_final,
    _verify_parakeet_cpp_cgroup_evidence,
    _wire_protocol_version as supervisor_wire_protocol_version,
)
from tools.streaming_stt.worker import (
    _parakeet_cpp_final_payload,
    _wire_protocol_version as worker_wire_protocol_version,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = ParakeetCppConfig()
ADAPTER_SOURCE = "tools/streaming_stt/adapters/parakeet_cpp.py"
NATIVE_FILENAMES = {
    "source-receipt": "source-receipt.json",
    "build-receipt": "build-receipt.json",
    "libparakeet": "libparakeet.so",
    "bridge-library": "libspeaker_parakeet_bridge.so",
}
MODEL_FILENAMES = {
    "model-receipt": "model-receipt.json",
    "model-gguf": "realtime_eou_120m-v1-f16.gguf",
}


def _request(tmp_path: Path, *, tail_samples: int = 48_000) -> TranscribeRequest:
    return TranscribeRequest(
        request_id="case-1",
        pcm=PcmInput(
            path=(tmp_path / "case.f32le").resolve(),
            sha256="a" * 64,
            samples=16_000,
        ),
        stream=StreamConfig(
            chunk_samples=CONFIG.native_chunk_samples,
            pace="burst",
            partial_interval_ms=200,
            tail_padding_samples=tail_samples,
        ),
        protocol_version=PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION,
    )


def _case(
    request: TranscribeRequest,
    *,
    source_consumed: int = 16_000,
    tail_consumed: int = 640,
    reason: str = "eou",
    origin: str = "feed",
    native: bool = True,
    encoder_frame: int | None = 13,
    encoder_time: float | None = 1.04,
    endpoint_sample: int | None = None,
    endpoint_latency_ms: float | None = 40.0,
    authoritative: bool = True,
    observed_events: tuple[ParakeetCppObservedEvent, ...] | None = None,
) -> SimpleNamespace:
    samples_seen = source_consumed + tail_consumed
    if observed_events is None:
        observed_events = (
            ParakeetCppObservedEvent(
                event_type="eou",
                event_origin="feed",
                observed_sample=samples_seen,
                encoder_frame=13,
                encoder_time_seconds=1.04,
            ),
        )
    return SimpleNamespace(
        partials=(),
        final="stop now",
        elapsed_ms=150.0,
        finalization_ms=4.0,
        compute_ms=30.0,
        deadline_misses=0,
        max_backlog_ms=2.0,
        resources=ResourceUsage(rss_mb=50.0, threads=2, vram_mb=None),
        model_padding_samples=0,
        chunks_yielded=(samples_seen + CONFIG.native_chunk_samples - 1)
        // CONFIG.native_chunk_samples,
        source_samples_offered=request.pcm.samples,
        source_samples_consumed=source_consumed,
        tail_samples_offered=request.stream.tail_padding_samples,
        tail_samples_consumed=tail_consumed,
        endpoint_reason=reason,
        native_endpoint=native,
        encoder_frame=encoder_frame,
        encoder_time_seconds=encoder_time,
        event_origin=origin,
        endpoint_sample=samples_seen if endpoint_sample is None and native else endpoint_sample,
        endpoint_latency_ms=endpoint_latency_ms,
        authoritative=authoritative,
        observed_events=observed_events,
    )


def _validator_manifest() -> SimpleNamespace:
    return SimpleNamespace(adapter_config=CONFIG)


def _validate(request: TranscribeRequest, event: ParakeetCppFinalEvent) -> None:
    _validate_parakeet_cpp_final(
        _validator_manifest(),  # type: ignore[arg-type]
        request,
        event,
        expected_seq=2,
        last_samples=0,
        last_elapsed=0.0,
    )


def _serialized_event(
    request: TranscribeRequest,
    case: SimpleNamespace,
) -> tuple[dict[str, object], ParakeetCppFinalEvent]:
    payload = _parakeet_cpp_final_payload(
        request,
        case,
        CONFIG,
        partial_count=2,
    )
    event = parse_response(encode_message(payload))
    assert isinstance(event, ParakeetCppFinalEvent)
    return payload, event


def test_schema_v8_selects_the_distinct_probability_free_v4_wire_contract(
    tmp_path: Path,
):
    manifest = SimpleNamespace(
        schema_version=8,
        adapter=PARAKEET_CPP_ADAPTER,
        adapter_config=CONFIG,
    )
    request = _request(tmp_path)
    payload, event = _serialized_event(request, _case(request))

    assert worker_wire_protocol_version(manifest) == 4
    assert supervisor_wire_protocol_version(manifest) == 4
    assert request.as_dict()["v"] == 4
    assert payload["v"] == 4
    assert payload["type"] == "parakeet_cpp_final"
    assert "endpoint_probability" not in payload
    assert payload["observed_events"] == [
        {
            "event_type": "eou",
            "event_origin": "feed",
            "observed_sample": 16_640,
            "encoder_frame": 13,
            "encoder_time_seconds": 1.04,
        }
    ]
    assert event.protocol_version == PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION


def test_authoritative_feed_eou_is_serialized_and_independently_validated(
    tmp_path: Path,
):
    request = _request(tmp_path)
    payload, event = _serialized_event(request, _case(request))

    assert payload["source_samples_offered"] == 16_000
    assert payload["source_samples_consumed"] == 16_000
    assert payload["tail_samples_consumed"] == 640
    assert payload["endpoint_sample"] == 16_640
    assert payload["endpoint_latency_ms"] == 40.0
    assert payload["authoritative"] is True
    _validate(request, event)

    with pytest.raises(WorkerError, match="worker_protocol"):
        _validate(request, replace(event, authoritative=False))


def test_eob_before_complete_source_eou_cannot_shadow_the_terminal(
    tmp_path: Path,
):
    request = _request(tmp_path)
    observations = (
        ParakeetCppObservedEvent("eob", "feed", 16_640, 12, 0.96),
        ParakeetCppObservedEvent("eou", "feed", 16_640, 13, 1.04),
    )

    payload, event = _serialized_event(
        request,
        _case(request, observed_events=observations),
    )

    assert [item["event_type"] for item in payload["observed_events"]] == [
        "eob",
        "eou",
    ]
    assert payload["endpoint_reason"] == "eou"
    assert payload["encoder_frame"] == 13
    _validate(request, event)


def test_impossible_one_sample_feed_boundary_fails_worker_and_supervisor(
    tmp_path: Path,
):
    request = _request(tmp_path)
    impossible = ParakeetCppObservedEvent("eou", "feed", 16_001, 12, 0.96)
    impossible_case = _case(
        request,
        tail_consumed=1,
        encoder_frame=12,
        encoder_time=0.96,
        endpoint_latency_ms=0.0625,
        observed_events=(impossible,),
    )

    with pytest.raises(ProtocolError):
        _parakeet_cpp_final_payload(
            request,
            impossible_case,
            CONFIG,
            partial_count=2,
        )

    _payload, valid_event = _serialized_event(request, _case(request))
    forged = replace(
        valid_event,
        samples_seen=16_001,
        chunks=13,
        tail_samples_consumed=1,
        observed_events=(impossible,),
        encoder_frame=12,
        encoder_time_seconds=0.96,
        endpoint_sample=16_001,
        endpoint_latency_ms=0.0625,
    )
    with pytest.raises(WorkerError, match="worker_protocol"):
        _validate(request, forged)


def test_near_time_duplicate_and_feed_after_finalize_fail_closed(
    tmp_path: Path,
):
    request = _request(tmp_path, tail_samples=8_000)
    duplicate_events = (
        ParakeetCppObservedEvent("eob", "feed", 16_640, 13, 1.0399995),
        ParakeetCppObservedEvent("eob", "feed", 17_920, 13, 1.04),
    )
    tail_case = _case(
        request,
        tail_consumed=8_000,
        reason="tail_exhausted",
        origin="none",
        native=False,
        encoder_frame=None,
        encoder_time=None,
        endpoint_sample=None,
        endpoint_latency_ms=None,
        authoritative=False,
        observed_events=duplicate_events,
    )
    with pytest.raises(ProtocolError):
        _parakeet_cpp_final_payload(
            request,
            tail_case,
            CONFIG,
            partial_count=2,
        )

    valid_observation = (
        ParakeetCppObservedEvent("eob", "feed", 16_640, 13, 1.04),
    )
    _payload, valid_event = _serialized_event(
        request,
        _case(
            request,
            tail_consumed=8_000,
            reason="tail_exhausted",
            origin="none",
            native=False,
            encoder_frame=None,
            encoder_time=None,
            endpoint_sample=None,
            endpoint_latency_ms=None,
            authoritative=False,
            observed_events=valid_observation,
        ),
    )
    with pytest.raises(WorkerError, match="worker_protocol"):
        _validate(request, replace(valid_event, observed_events=duplicate_events))

    feed_after_finalize = (
        ParakeetCppObservedEvent("eob", "finalize", 24_000, 13, 1.04),
        ParakeetCppObservedEvent("eou", "feed", 24_000, 14, 1.12),
    )
    with pytest.raises(ProtocolError):
        _parakeet_cpp_final_payload(
            request,
            _case(
                request,
                tail_consumed=8_000,
                reason="tail_exhausted",
                origin="none",
                native=False,
                encoder_frame=None,
                encoder_time=None,
                endpoint_sample=None,
                endpoint_latency_ms=None,
                authoritative=False,
                observed_events=feed_after_finalize,
            ),
            CONFIG,
            partial_count=2,
        )


def test_early_eou_and_eob_are_telemetry_and_full_input_is_consumed(
    tmp_path: Path,
):
    request = _request(tmp_path, tail_samples=8_000)
    observations = (
        ParakeetCppObservedEvent("eou", "feed", 2_560, 2, 0.16),
        ParakeetCppObservedEvent("eob", "feed", 5_120, 4, 0.32),
        ParakeetCppObservedEvent("eou", "feed", 16_640, 13, 1.04),
        ParakeetCppObservedEvent("eob", "finalize", 24_000, 14, 1.12),
    )
    payload, event = _serialized_event(
        request,
        _case(
            request,
            tail_consumed=8_000,
            reason="tail_exhausted",
            origin="none",
            native=False,
            encoder_frame=None,
            encoder_time=None,
            endpoint_sample=None,
            endpoint_latency_ms=None,
            authoritative=False,
            observed_events=observations,
        ),
    )

    assert payload["source_samples_consumed"] == 16_000
    assert payload["tail_samples_consumed"] == 8_000
    assert [item["event_type"] for item in payload["observed_events"]] == [
        "eou",
        "eob",
        "eou",
        "eob",
    ]
    assert payload["native_endpoint"] is False
    assert payload["event_origin"] == "none"
    assert payload["authoritative"] is False
    assert payload["endpoint_latency_ms"] is None
    _validate(request, event)


def test_later_eou_cannot_revalidate_an_early_first_turn(tmp_path: Path):
    request = _request(tmp_path)
    observations = (
        ParakeetCppObservedEvent("eou", "feed", 2_560, 2, 0.16),
        ParakeetCppObservedEvent("eou", "feed", 16_640, 13, 1.04),
    )
    promoted = _case(request, observed_events=observations)

    with pytest.raises(ProtocolError):
        _parakeet_cpp_final_payload(
            request,
            promoted,
            CONFIG,
            partial_count=2,
        )

    _payload, valid_event = _serialized_event(request, _case(request))
    with pytest.raises(WorkerError, match="worker_protocol"):
        _validate(request, replace(valid_event, observed_events=observations))


def test_feed_encoder_time_cannot_exceed_the_offered_sample_boundary(
    tmp_path: Path,
):
    request = _request(tmp_path)
    invalid_observation = ParakeetCppObservedEvent(
        "eou",
        "feed",
        16_640,
        14,
        1.12,
    )
    invalid_case = _case(request, observed_events=(invalid_observation,))

    with pytest.raises(ProtocolError):
        _parakeet_cpp_final_payload(
            request,
            invalid_case,
            CONFIG,
            partial_count=2,
        )

    _payload, valid_event = _serialized_event(request, _case(request))
    invalid_event = replace(
        valid_event,
        observed_events=(invalid_observation,),
    )
    with pytest.raises(WorkerError, match="worker_protocol"):
        _validate(request, invalid_event)


def test_finalize_may_flush_past_capture_time_but_requires_complete_input(
    tmp_path: Path,
):
    request = _request(tmp_path, tail_samples=0)
    finalize_observation = ParakeetCppObservedEvent(
        "eou",
        "finalize",
        16_000,
        13,
        1.04,
    )
    complete = _case(
        request,
        tail_consumed=0,
        reason="tail_exhausted",
        origin="none",
        native=False,
        encoder_frame=None,
        encoder_time=None,
        endpoint_sample=None,
        endpoint_latency_ms=None,
        authoritative=False,
        observed_events=(finalize_observation,),
    )
    payload, event = _serialized_event(request, complete)

    assert payload["encoder_time_seconds"] is None
    assert payload["endpoint_sample"] is None
    assert payload["observed_events"][0]["encoder_time_seconds"] == 1.04
    assert (
        payload["observed_events"][0]["encoder_time_seconds"]
        > payload["observed_events"][0]["observed_sample"] / 16_000
    )
    assert payload["authoritative"] is False
    _validate(request, event)

    incomplete_observation = ParakeetCppObservedEvent(
        "eou",
        "finalize",
        12_800,
        10,
        0.8,
    )
    incomplete = _case(
        request,
        source_consumed=12_800,
        tail_consumed=0,
        reason="tail_exhausted",
        origin="none",
        native=False,
        encoder_frame=None,
        encoder_time=None,
        endpoint_sample=None,
        endpoint_latency_ms=None,
        authoritative=False,
        observed_events=(incomplete_observation,),
    )
    with pytest.raises(ProtocolError):
        _parakeet_cpp_final_payload(
            request,
            incomplete,
            CONFIG,
            partial_count=2,
        )

    invalid_event = replace(
        event,
        samples_seen=12_800,
        chunks=10,
        source_samples_consumed=12_800,
        observed_events=(incomplete_observation,),
    )
    with pytest.raises(WorkerError, match="worker_protocol"):
        _validate(request, invalid_event)


def _write_file(path: Path, payload: bytes = b"x") -> Path:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o600)
    return path.resolve()


def _launch_fixture(tmp_path: Path) -> tuple[WorkerManifest, SimpleNamespace, Path]:
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    native_root = tmp_path / "native"
    model_root = tmp_path / "model"
    native_root.mkdir(mode=0o700)
    model_root.mkdir(mode=0o700)

    filenames = {**NATIVE_FILENAMES, **MODEL_FILENAMES}
    artifacts: list[BoundArtifact] = []
    for name in PARAKEET_CPP_ARTIFACT_NAMES:
        filename = filenames[name]
        root = native_root if name in NATIVE_FILENAMES else model_root
        path = _write_file(root / filename)
        artifacts.append(
            BoundArtifact(
                name=name,
                path=path,
                sha256="b" * 64,
                size_bytes=1,
            )
        )

    bundle_root = tmp_path / "source-bundle"
    bundle_root.mkdir(mode=0o700)
    bundle_worker = _write_file(
        bundle_root / "tools" / "streaming_stt" / "worker.py",
        b"# worker\n",
    )
    bundle_manifest = _write_file(bundle_root / "source-bundle.json", b"{}\n")
    source_bundle = SimpleNamespace(
        root=bundle_root.resolve(),
        worker_path=bundle_worker,
        manifest_path=bundle_manifest,
        tree_sha256="c" * 64,
        files=(object(),),
    )

    python_path = _write_file(tmp_path / "venv" / "bin" / "python")
    worker_path = _write_file(tmp_path / "bound-worker.py")
    manifest_path = _write_file(tmp_path / "worker-manifest.json", b"{}\n")
    manifest = WorkerManifest(
        path=manifest_path,
        digest="d" * 64,
        schema_version=8,
        model_id="parakeet-cpp-cpu",
        adapter=PARAKEET_CPP_ADAPTER,
        python=BoundFile(python_path, "e" * 64, 1),
        worker=BoundFile(worker_path, "f" * 64, 1),
        artifacts=tuple(artifacts),
        limits=WorkerLimits(startup_timeout_sec=5.0, case_timeout_sec=10.0),
        adapter_config=CONFIG,
    )
    artifact_names = tuple(artifact.name for artifact in manifest.artifacts)
    assert artifact_names == PARAKEET_CPP_ARTIFACT_NAMES
    assert set(PARAKEET_CPP_ARTIFACT_NAMES) == {
        *NATIVE_FILENAMES,
        *MODEL_FILENAMES,
    }
    return manifest, source_bundle, scratch.resolve()


def _mount_pairs(command: list[str], option: str) -> list[tuple[str, str]]:
    return [
        (command[index + 1], command[index + 2])
        for index, value in enumerate(command)
        if value == option
    ]


def test_cpu_bwrap_has_closed_roots_no_network_or_gpu_and_one_writable_bind(
    tmp_path: Path,
):
    manifest, source_bundle, scratch = _launch_fixture(tmp_path)
    command = _parakeet_cpp_bwrap_command(
        manifest,
        scratch,
        source_bundle,  # type: ignore[arg-type]
        bundle_descriptor=41,
        worker_descriptor=42,
        python_descriptor=43,
        system_ro_paths=_NEMOTRON_CORE_RO_PATHS,
    )

    assert command[:13] == [
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
    ]
    assert command.count("--unshare-net") == 1
    assert "--share-net" not in command
    assert "--dev-bind" not in command
    assert "--dev-bind-try" not in command
    assert all("nvidia" not in value.lower() for value in command)
    assert all("cuda" not in value.lower() for value in command)

    assert _mount_pairs(command, "--bind") == [(str(scratch), str(scratch))]
    ro_pairs = _mount_pairs(command, "--ro-bind")
    artifacts = manifest.artifact_by_name
    native_root = artifacts["libparakeet"].path.parent
    model_root = artifacts["model-gguf"].path.parent
    assert native_root != model_root
    assert not native_root.is_relative_to(model_root)
    assert not model_root.is_relative_to(native_root)
    assert set(os.listdir(native_root)) == set(NATIVE_FILENAMES.values())
    assert set(os.listdir(model_root)) == set(MODEL_FILENAMES.values())
    assert (str(native_root), str(native_root)) in ro_pairs
    assert (str(model_root), str(model_root)) in ro_pairs
    assert ("/", "/") not in ro_pairs
    assert not any(
        str(artifact.path) == source
        for artifact in manifest.artifacts
        for source, _destination in ro_pairs
    )

    separator = command.index("--")
    assert command[separator - 4 : separator] == [
        "--chdir",
        str(scratch),
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
        str(scratch),
        "--source-bundle-manifest",
        "/proc/self/fd/41/source-bundle.json",
        "--source-bundle-sha256",
        source_bundle.tree_sha256,
        "--source-bundle-fd",
        "41",
    ]


def test_cpu_bwrap_rejects_a_non_closed_native_artifact_root(tmp_path: Path):
    manifest, source_bundle, scratch = _launch_fixture(tmp_path)
    _write_file(manifest.artifact_by_name["libparakeet"].path.parent / "extra.so")

    with pytest.raises(WorkerError, match="worker_prerequisite"):
        _parakeet_cpp_bwrap_command(
            manifest,
            scratch,
            source_bundle,  # type: ignore[arg-type]
            bundle_descriptor=41,
            worker_descriptor=42,
            python_descriptor=43,
            system_ro_paths=_NEMOTRON_CORE_RO_PATHS,
        )


def test_cpu_systemd_scope_has_exact_smaller_hard_limits_and_pinned_bwrap():
    command = _parakeet_cpp_systemd_scope_command(
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
        "--description=speaker-parakeet-cpp-worker",
        "--property",
        "MemoryHigh=2147483648",
        "--property",
        "MemoryMax=3221225472",
        "--property",
        "MemorySwapMax=0",
        "--property",
        "CPUQuota=100%",
        "--property",
        "TasksMax=64",
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


def _write_cpp_cgroup_tree(
    root: Path,
    *,
    overrides: dict[str, str] | None = None,
) -> bytes:
    root.mkdir()
    (root / "cgroup.controllers").write_text(
        "cpuset cpu io memory pids\n",
        encoding="ascii",
    )
    relative = Path(
        "user.slice/user-1000.slice/user@1000.service/app.slice/run-r5678.scope"
    )
    scope = root / relative
    scope.mkdir(parents=True)
    values = {
        "memory.high": "2147483648",
        "memory.max": "3221225472",
        "memory.swap.max": "0",
        "memory.oom.group": "1",
        "cpu.max": "100000 100000",
        "pids.max": "64",
    }
    values.update(overrides or {})
    for name, value in values.items():
        (scope / name).write_text(value + "\n", encoding="ascii")
    return f"0::/{relative}\n".encode("ascii")


def test_cpu_cgroup_verifier_returns_only_stable_exact_evidence(tmp_path: Path):
    root = tmp_path / "cgroup"
    proc_payload = _write_cpp_cgroup_tree(root)

    evidence = _verify_parakeet_cpp_cgroup_evidence(
        proc_payload,
        cgroup_root=root,
    )

    assert evidence == {
        "kind": "systemd-user-scope-cgroup-v2",
        "memory_high_bytes": 2_147_483_648,
        "memory_max_bytes": 3_221_225_472,
        "memory_swap_max_bytes": 0,
        "cpu_quota_percent": 100,
        "tasks_max": 64,
        "oom_policy": "kill",
        "verified": True,
    }
    assert "run-r5678.scope" not in repr(evidence)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("memory.high", "max"),
        ("memory.max", "3221225473"),
        ("memory.swap.max", "1"),
        ("memory.oom.group", "0"),
        ("cpu.max", "200000 100000"),
        ("pids.max", "65"),
    ],
)
def test_cpu_cgroup_verifier_rejects_every_relaxed_effective_limit(
    tmp_path: Path,
    name: str,
    value: str,
):
    root = tmp_path / "cgroup"
    proc_payload = _write_cpp_cgroup_tree(root, overrides={name: value})

    with pytest.raises(WorkerError, match="worker_prerequisite"):
        _verify_parakeet_cpp_cgroup_evidence(proc_payload, cgroup_root=root)


def test_source_bundle_contains_the_exact_parakeet_cpp_adapter(tmp_path: Path):
    assert WORKER_SOURCE_FILES.count(ADAPTER_SOURCE) == 1

    bundle = stage_worker_source_bundle(REPO_ROOT, tmp_path / "bundle")
    bound = bundle.file_by_path[ADAPTER_SOURCE]
    staged = bundle.root / ADAPTER_SOURCE

    assert staged.is_file()
    assert bound.size_bytes == staged.stat().st_size
    assert bound.sha256 == hashlib.sha256(staged.read_bytes()).hexdigest()
    assert staged.read_bytes() == (REPO_ROOT / ADAPTER_SOURCE).read_bytes()


def test_schema_v8_launch_environment_is_cpu_only_offline_and_single_threaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest, source_bundle, scratch = _launch_fixture(tmp_path)
    runtime_directory = tmp_path / "user-runtime"
    runtime_directory.mkdir(mode=0o700)
    captured: dict[str, object] = {}
    worker = StreamingWorker(manifest, scratch, source_bundle)  # type: ignore[arg-type]

    monkeypatch.setattr(worker, "_verify_bound_files", lambda: None)
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._core_system_ro_paths",
        lambda: (Path("/usr"),),
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._nemotron_device_nodes",
        lambda: (_ for _ in ()).throw(AssertionError("GPU probe")),
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._nemotron_system_ro_paths",
        lambda: (_ for _ in ()).throw(AssertionError("GPU mount")),
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._verified_user_runtime_directory",
        lambda: runtime_directory,
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._open_bwrap_descriptor",
        lambda: os.open("/dev/null", os.O_RDONLY),
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._open_systemd_run_descriptor",
        lambda: os.open("/dev/null", os.O_RDONLY),
    )
    monkeypatch.setattr(
        worker,
        "_open_staged_worker_descriptor",
        lambda: os.open("/dev/null", os.O_RDONLY),
    )
    monkeypatch.setattr(
        worker,
        "_open_bound_python_descriptor",
        lambda: os.open("/dev/null", os.O_RDONLY),
    )

    def rejecting_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        raise OSError

    worker._popen_factory = rejecting_popen  # noqa: SLF001 - launch seam
    with pytest.raises(WorkerError, match="worker_start"):
        worker.start()

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert environment["PARAKEET_DEVICE"] == "cpu"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert "CUDA_DEVICE_ORDER" not in environment
    assert "PATH" not in environment
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
    assert environment["MKL_NUM_THREADS"] == "1"
    assert environment["NUMEXPR_NUM_THREADS"] == "1"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["XDG_RUNTIME_DIR"] == str(runtime_directory)
    assert "DBUS_SESSION_BUS_ADDRESS" not in environment
    assert "INVOCATION_ID" not in environment
    assert kwargs["start_new_session"] is True
    assert kwargs["close_fds"] is True
    assert len(kwargs["pass_fds"]) == 5
    assert "shell" not in kwargs
    assert worker._bundle_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._worker_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._python_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._bwrap_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._systemd_run_descriptor == -1  # noqa: SLF001
