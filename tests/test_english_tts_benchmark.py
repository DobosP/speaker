"""Deterministic TTS timing/privacy/provision tests; no native model or audio I/O."""

import hashlib
import io
import json
from types import SimpleNamespace
import tarfile

import numpy as np
import pytest

from tools import english_tts_benchmark as bench


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_callback_ttfa_and_full_latency_are_distinct_and_aggregate_only():
    clock = Clock()
    private_text = "Synthetic private sentinel"

    def generate(text, callback):
        assert text == private_text
        clock.advance(0.2)
        assert callback(np.zeros(2, dtype=np.float32), 0.2) == 1
        clock.advance(0.1)
        assert callback(np.array([0.1, -0.2], dtype=np.float32), 1.0) == 1
        clock.advance(0.3)
        return SimpleNamespace(
            samples=np.array([0.1, -0.2], dtype=np.float32), sample_rate=2
        )

    report = bench.benchmark_model(generate, [private_text], repeats=3, clock=clock)
    assert report["generation_count"] == 3
    assert report["first_callback_seconds"]["p50"] == pytest.approx(0.2)
    assert report["first_nonzero_pcm_callback_seconds"]["p50"] == pytest.approx(0.3)
    assert report["full_synthesis_seconds"]["p50"] == pytest.approx(0.6)
    assert report["rtf"]["p50"] == pytest.approx(0.6)
    assert report["callback_chunks"]["p50"] == 2
    assert report["generations_with_callback_before_return"] == 3
    assert report["invalid_waveforms"] == 0
    assert report["all_zero_waveforms"] == 0
    assert not report["quality_authority"]
    assert not report["audibility_authority"]
    assert private_text not in json.dumps(report)


def test_whole_clip_model_does_not_fabricate_a_callback_time():
    def generate(text, callback):
        return SimpleNamespace(
            samples=np.array([0.2], dtype=np.float32), sample_rate=24000
        )

    report = bench.benchmark_model(generate, ["synthetic"], repeats=1)
    assert report["first_callback_seconds"] is None
    assert report["first_nonzero_pcm_callback_seconds"] is None
    assert report["generations_without_callback"] == 1
    assert report["generations_with_callback_before_return"] == 0


@pytest.mark.parametrize(
    "samples,rate,invalid,zero,clipped",
    [
        ([], 24000, 1, 0, 0),
        ([float("nan")], 24000, 1, 0, 0),
        ([0.1], 0, 1, 0, 0),
        ([0.0, 0.0], 24000, 0, 1, 0),
        ([1.0, -1.2, 0.4], 24000, 0, 0, 2),
    ],
)
def test_waveform_checks_count_failures_without_retaining_samples(
    samples, rate, invalid, zero, clipped
):
    report = bench.benchmark_model(
        lambda text, callback: SimpleNamespace(
            samples=np.array(samples), sample_rate=rate
        ),
        ["synthetic"],
        repeats=1,
    )
    assert report["invalid_waveforms"] == invalid
    assert report["all_zero_waveforms"] == zero
    assert report["samples_at_or_above_full_scale"] == clipped
    assert "samples" not in report


def test_provider_failure_never_stringifies_private_exception():
    class HostileError(Exception):
        def __str__(self):
            raise AssertionError("must not stringify provider failure")

    def generate(text, callback):
        raise HostileError("private provider content")

    with pytest.raises(bench.BenchmarkError, match="^synthesis_failed$"):
        bench.benchmark_model(generate, ["synthetic"], repeats=1)


@pytest.mark.parametrize("repeat", [0, 11, True])
def test_repetition_bounds(repeat):
    with pytest.raises(bench.BenchmarkError, match="^repeats_invalid$"):
        bench.benchmark_model(lambda *args: None, ["synthetic"], repeats=repeat)


def test_manifest_binding_contains_no_private_text_file_or_labels(tmp_path):
    manifest = {
        "clips": [
            {
                "file": "PRIVATE-FILE.wav",
                "text": "PRIVATE-TEXT",
                "role": "round_trip",
                "group": "PRIVATE-GROUP",
            }
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    texts, receipt = bench.load_owner_manifest(path)
    assert texts == ["PRIVATE-TEXT"]
    assert (
        receipt["manifest_sha256"]
        == hashlib.sha256(bench.canonical_bytes(manifest)).hexdigest()
    )
    assert receipt["reference_count"] == 1
    assert not receipt["source_audio_consumed"]
    assert not receipt["disjoint_holdout"]
    assert "PRIVATE" not in json.dumps(receipt)
    # The source WAV need not exist: this TTS evaluation never reads it.


def test_manifest_text_bound_is_fixed_and_content_free(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"clips": [{"text": "x" * (bench.MAX_TEXT_BYTES + 1)}]}))
    with pytest.raises(bench.BenchmarkError, match="^manifest_text_limit$"):
        bench.load_owner_manifest(path)


def test_inventory_rejects_symlink_and_detects_content_identity(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"synthetic model")
    first = bench.artifact_inventory(tmp_path)
    (tmp_path / "model.onnx").write_bytes(b"changed model")
    assert bench.artifact_inventory(tmp_path) != first
    (tmp_path / "linked").symlink_to(tmp_path / "model.onnx")
    with pytest.raises(bench.BenchmarkError, match="^artifact_not_regular$"):
        bench.artifact_inventory(tmp_path)


def test_report_is_private_and_no_clobber(tmp_path):
    path = tmp_path / "report.json"
    bench._private_write(path, {"count": 37})
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(bench.BenchmarkError, match="^report_write_failed$"):
        bench._private_write(path, {"count": 0})
    assert json.loads(path.read_bytes()) == {"count": 37}


def _candidate_archive(tmp_path, *, link=False):
    archive = tmp_path / "candidate.tar.bz2"
    with tarfile.open(archive, "w:bz2") as stream:
        member = tarfile.TarInfo("package/model.int8.onnx")
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = "/unrelated"
            stream.addfile(member)
        else:
            payload = b"fake native model bytes"
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))
    return archive


@pytest.mark.parametrize("link", [False, True])
def test_provision_uses_bounded_extractor_and_rejects_archive_links(
    tmp_path, monkeypatch, link
):
    source = _candidate_archive(tmp_path, link=link)
    candidate = {
        "archive": source.name,
        "archive_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "max_archive_bytes": 100000,
        "version": "synthetic",
        "weights_license": "Apache-2.0",
        "license_source": "public",
    }
    monkeypatch.setattr(
        bench,
        "load_catalog",
        lambda: {"candidates": {"kitten_nano_v0_8_int8": candidate}},
    )
    monkeypatch.setattr(
        bench,
        "_download",
        lambda candidate, target: target.write_bytes(source.read_bytes()),
    )
    root = tmp_path / "provision"
    if link:
        with pytest.raises(bench.BenchmarkError, match="^archive_unsafe$"):
            bench.provision_candidate("kitten_nano_v0_8_int8", root)
        assert not (root / "kitten_nano_v0_8_int8").exists()
    else:
        receipt = bench.provision_candidate("kitten_nano_v0_8_int8", root)
        assert receipt["artifacts"][0]["file"] == "model.int8.onnx"
        assert receipt["artifacts"][0]["bytes"] == len(b"fake native model bytes")
        assert bench.provision_candidate("kitten_nano_v0_8_int8", root) == receipt
        (root / "kitten_nano_v0_8_int8" / "model.int8.onnx").write_bytes(b"tampered")
        with pytest.raises(bench.BenchmarkError, match="^provision_changed$"):
            bench.provision_candidate("kitten_nano_v0_8_int8", root)


def test_catalog_contains_public_native_candidates_only():
    candidates = bench.load_catalog()["candidates"]
    assert set(candidates) == {"kitten_nano_v0_8_int8", "supertonic3_int8"}
    for candidate in candidates.values():
        assert len(candidate["archive_sha256"]) == 64
        assert candidate["url"].startswith(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
        )


def _payload(root="unused"):
    return {
        "model_id": "kitten_nano_v0_8_int8",
        "model_root": root,
        "texts": ["PRIVATE-TEXT"],
        "threads": 2,
        "repeats": 1,
        "sid": 0,
        "steps": 8,
    }


def _fake_report():
    metrics = bench.benchmark_model(
        lambda text, callback: SimpleNamespace(
            samples=np.array([0.1]), sample_rate=24000
        ),
        ["PRIVATE-TEXT"],
        repeats=1,
    )
    return {
        "status": "ok",
        "model_id": "kitten_nano_v0_8_int8",
        "model_kind": "kitten",
        "runtime": {
            "sherpa_onnx": "1.13.3",
            "provider": "cpu",
            "configured_inference_threads": 2,
            "speaker_id": 0,
            "generation_steps": 8,
            "speed": 1.0,
            "postprocessing": "none",
        },
        "execution": bench._execution_receipt(2, 600),
        "model_artifacts_sha256": "a" * 64,
        "model_artifact_bytes": 27,
        "model_load_seconds": 0.1,
        "peak_process_rss_kib": 100,
        "metrics": metrics,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update({"private_text": "PRIVATE-TEXT"}),
        lambda r: r["metrics"]["full_synthesis_seconds"].update({"PRIVATE-LABEL": 1}),
        lambda r: r["metrics"].update({"quality_authority": True}),
        lambda r: r["metrics"].update({"generation_count": 999}),
        lambda r: r["runtime"].update({"provider": "cuda"}),
    ],
)
def test_worker_protocol_rejects_private_or_inconsistent_fields(mutation):
    report = _fake_report()
    mutation(report)
    with pytest.raises(bench.BenchmarkError, match="^tts_worker_protocol_failed$"):
        bench._validate_worker_report(report, _payload())


def test_worker_protocol_accepts_exact_aggregate_shape():
    bench._validate_worker_report(_fake_report(), _payload())


@pytest.mark.parametrize(
    "key,value", [("repeats", 0), ("threads", True), ("sid", -1), ("steps", 17)]
)
def test_request_is_rejected_before_model_loading(tmp_path, monkeypatch, key, value):
    entered = []
    monkeypatch.setattr(bench, "_load_native", lambda *args: entered.append(True))
    payload = _payload(str(tmp_path))
    payload[key] = value
    with pytest.raises(bench.BenchmarkError, match="^worker_input_invalid$"):
        bench.worker(payload)
    assert not entered


def test_worker_failure_is_fixed_even_when_stdout_contains_private_values(monkeypatch):
    class FakeProcess:
        returncode = 2

        def communicate(self, payload, timeout):
            assert b"PRIVATE-TEXT" in payload
            return b'{"status":"failed","code":"PRIVATE-NATIVE-ERROR"}', None

    monkeypatch.setattr(
        bench.subprocess, "Popen", lambda *args, **kwargs: FakeProcess()
    )
    with pytest.raises(bench.BenchmarkError, match="^tts_worker_failed$"):
        bench.run_isolated(_payload())


def test_model_worker_timeout_kills_exact_process_group_and_waits(monkeypatch):
    calls = []

    class FakeProcess:
        pid = 123456

        def poll(self):
            return None

        def communicate(self, payload=None, timeout=None):
            calls.append((payload, timeout))
            if len(calls) <= 2:
                raise bench.subprocess.TimeoutExpired("synthetic", timeout)
            return b"", None

    killed = []
    monkeypatch.setattr(
        bench.subprocess, "Popen", lambda *args, **kwargs: FakeProcess()
    )
    monkeypatch.setattr(
        bench.os, "killpg", lambda pid, signal: killed.append((pid, signal))
    )
    with pytest.raises(bench.BenchmarkError, match="^tts_worker_timeout$"):
        bench.run_isolated(_payload(), timeout=1)
    assert killed == [(123456, bench.signal.SIGTERM), (123456, bench.signal.SIGKILL)]
    assert calls[1:] == [(None, 1), (None, 5)]


def test_parent_interrupt_retires_worker_before_propagating(monkeypatch):
    calls = []

    class FakeProcess:
        pid = 123456

        def poll(self):
            return None

        def communicate(self, payload=None, timeout=None):
            calls.append((payload, timeout))
            if len(calls) == 1:
                raise KeyboardInterrupt()
            return b"", None

    killed = []
    monkeypatch.setattr(
        bench.subprocess, "Popen", lambda *args, **kwargs: FakeProcess()
    )
    monkeypatch.setattr(
        bench.os, "killpg", lambda pid, signal: killed.append((pid, signal))
    )
    with pytest.raises(KeyboardInterrupt):
        bench.run_isolated(_payload())
    assert killed == [(123456, bench.signal.SIGTERM)]
    assert calls[1] == (None, 1)


def test_named_candidate_identity_rejects_changed_bytes_before_native_load(
    tmp_path, monkeypatch
):
    (tmp_path / "model.int8.onnx").write_bytes(b"unadmitted bytes")
    entered = []
    monkeypatch.setattr(bench, "_load_native", lambda *args: entered.append(True))
    with pytest.raises(bench.BenchmarkError, match="^model_identity_mismatch$"):
        bench.worker(_payload(str(tmp_path)))
    assert not entered


def test_candidate_inventory_binding_is_exact(monkeypatch):
    inventory = [{"file": "model.int8.onnx", "bytes": 7, "sha256": "a" * 64}]
    binding = hashlib.sha256(bench.canonical_bytes(inventory)).hexdigest()
    monkeypatch.setattr(
        bench,
        "load_catalog",
        lambda: {
            "candidates": {
                "kitten_nano_v0_8_int8": {"artifact_inventory_sha256": binding}
            }
        },
    )
    bench._verify_model_identity("kitten_nano_v0_8_int8", inventory)
    inventory[0]["bytes"] = 8
    with pytest.raises(bench.BenchmarkError, match="^model_identity_mismatch$"):
        bench._verify_model_identity("kitten_nano_v0_8_int8", inventory)


def test_baseline_requires_named_model_voice_and_token_bindings(monkeypatch):
    expected = {
        "model.int8.onnx": {"bytes": 1, "sha256": "a" * 64},
        "voices.bin": {"bytes": 2, "sha256": "b" * 64},
        "tokens.txt": {"bytes": 3, "sha256": "c" * 64},
    }
    monkeypatch.setattr(
        bench,
        "load_catalog",
        lambda: {"candidates": {}, "baseline_kokoro_identity": expected},
    )
    inventory = [{"file": name, **value} for name, value in expected.items()]
    bench._verify_model_identity("kokoro_v1_1", inventory)
    inventory.pop()
    with pytest.raises(bench.BenchmarkError, match="^model_identity_mismatch$"):
        bench._verify_model_identity("kokoro_v1_1", inventory)


@pytest.mark.parametrize(
    "counter",
    [
        "invalid_waveforms",
        "all_zero_waveforms",
        "samples_total",
        "invalid_callback_chunks",
    ],
)
@pytest.mark.parametrize("bad", [{"PRIVATE_SENTINEL": 1}, True, -1, 0.5])
def test_scalar_metric_counters_reject_private_keys_and_nonintegers(counter, bad):
    report = _fake_report()
    report["metrics"][counter] = bad
    with pytest.raises(bench.BenchmarkError, match="^tts_worker_protocol_failed$"):
        bench._validate_worker_report(report, _payload())


def test_clipped_sample_count_cannot_exceed_all_samples():
    report = _fake_report()
    report["metrics"]["samples_at_or_above_full_scale"] = 2
    with pytest.raises(bench.BenchmarkError, match="^tts_worker_protocol_failed$"):
        bench._validate_worker_report(report, _payload())


@pytest.mark.parametrize("bad_chunk", [np.array([np.nan]), np.array([[0.1, 0.2]])])
def test_invalid_callback_is_not_first_pcm_even_when_final_waveform_is_clean(bad_chunk):
    clock = Clock()

    def generate(text, callback):
        clock.advance(0.1)
        callback(bad_chunk, 0.5)
        clock.advance(0.1)
        return SimpleNamespace(samples=np.array([0.1]), sample_rate=24000)

    report = bench.benchmark_model(generate, ["synthetic"], repeats=1, clock=clock)
    assert report["invalid_callback_chunks"] == 1
    assert report["invalid_waveforms"] == 1
    assert report["first_callback_seconds"] is None
    assert report["first_nonzero_pcm_callback_seconds"] is None


def test_worker_environment_is_allowlisted_and_offline(monkeypatch):
    monkeypatch.setattr(
        bench.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "PRIVATE_TOKEN": "PRIVATE_SENTINEL",
            "HTTPS_PROXY": "PRIVATE_SENTINEL",
            "PYTHONPATH": "PRIVATE_SENTINEL",
            "HOME": "PRIVATE_SENTINEL",
        },
    )
    environment = bench._worker_environment()
    assert environment["PATH"] == "/usr/bin"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["HF_HUB_OFFLINE"] == environment["TRANSFORMERS_OFFLINE"] == "1"
    assert "PRIVATE_SENTINEL" not in str(environment)
    assert not {"PRIVATE_TOKEN", "HTTPS_PROXY", "PYTHONPATH", "HOME"} & set(environment)


def _mock_guards(monkeypatch, allowed):
    mask = set(allowed)

    def set_affinity(pid, selected):
        nonlocal mask
        assert pid == 0
        mask = set(selected)

    monkeypatch.setattr(bench.os, "sched_getaffinity", lambda pid: mask)
    monkeypatch.setattr(bench.os, "sched_setaffinity", set_affinity)
    limits = []
    monkeypatch.setattr(
        bench.resource,
        "setrlimit",
        lambda resource, limit: limits.append((resource, limit)),
    )
    monkeypatch.setattr(
        bench.os, "environ", {"PATH": "/usr/bin", "PRIVATE_TOKEN": "PRIVATE_SENTINEL"}
    )
    for method in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(bench.socket.socket, method, lambda *args: None)
    monkeypatch.setattr(bench.socket, "create_connection", lambda *args: None)
    return limits


def test_worker_guards_apply_exact_cpu_limits_and_python_network_hooks(monkeypatch):
    limits = _mock_guards(monkeypatch, {4, 5, 6, 7})
    receipt = bench._apply_worker_guards(2, 600)
    assert bench.os.sched_getaffinity(0) == {4, 5}
    assert limits == [
        (bench.resource.RLIMIT_CPU, (600, 601)),
        (bench.resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3)),
        (bench.resource.RLIMIT_CORE, (0, 0)),
    ]
    assert receipt == bench._execution_receipt(2, 600)
    assert receipt["network_python_hooks"] is True
    assert receipt["os_network_isolated"] is False
    assert "PRIVATE_TOKEN" not in bench.os.environ
    for method in ("connect", "connect_ex", "sendto"):
        with pytest.raises(
            bench.BenchmarkError, match="^worker_python_network_denied$"
        ):
            getattr(bench.socket.socket, method)(None, "private address")
    with pytest.raises(bench.BenchmarkError, match="^worker_python_network_denied$"):
        bench.socket.create_connection("private address")


def test_worker_refuses_missing_cpu_budget_before_native_work(monkeypatch):
    limits = _mock_guards(monkeypatch, {4})
    with pytest.raises(bench.BenchmarkError, match="^worker_cpu_affinity_unavailable$"):
        bench._apply_worker_guards(2, 600)
    assert not limits


def test_execution_receipt_cannot_claim_os_network_isolation():
    report = _fake_report()
    report["execution"]["os_network_isolated"] = True
    with pytest.raises(bench.BenchmarkError, match="^tts_worker_protocol_failed$"):
        bench._validate_worker_report(report, _payload())
