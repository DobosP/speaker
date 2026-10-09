"""Generated private-fixture pairs only: no owner recording/native model."""

from __future__ import annotations

import ctypes
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from core.diagnostic_bundle import (
    CaptureCoordinate,
    DiagnosticTrack,
    SynchronizedDiagnosticBundle,
    validate_manifest,
)
from tools import localvqe_recorded_eval as evaluate


@pytest.fixture
def bundle(tmp_path):
    def create(*, mismatched=False):
        directory = tmp_path / ("mismatch" if mismatched else "valid")
        directory.mkdir(mode=0o700)
        manifest = directory / "fixture.diagnostic.json"
        writer = SynchronizedDiagnosticBundle(
            {role: directory / (role.value + ".wav") for role in DiagnosticTrack},
            directory / "timeline.jsonl",
            manifest,
            16000,
            queue_max=32,
            flush_sec=0.01,
        )
        for index in range(8):
            tracks = {
                role: np.full(1600, 0.1 + 0.05 * ordinal, dtype=np.float32)
                for ordinal, role in enumerate(DiagnosticTrack)
            }
            if mismatched and index in {0, 1}:
                tracks[DiagnosticTrack.PLAYBACK_REFERENCE_READER_SNAPSHOT] = np.full(
                    1599 if index == 0 else 1601, 0.25, dtype=np.float32
                )
            coordinate = CaptureCoordinate(
                sequence=index,
                sample_rate_hz=48000,
                captured_started_at=index * 0.1,
                captured_at=(index + 1) * 0.1,
                capture_epoch=0,
                source_generation=0,
                capture_generation=0,
                source_sample_start=index * 4800,
                source_sample_end=(index + 1) * 4800,
            )
            assert writer.write_frame(tracks, coordinate) is not None
        writer.close(clean_shutdown=True)
        assert validate_manifest(manifest)
        return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()

    return create


def test_complete_pair_preserves_after_host_roles_and_every_frame(bundle):
    manifest, digest = bundle()
    pair = evaluate.load_pair(manifest, digest)
    assert pair.microphone.dtype == pair.reference.dtype == np.float32
    assert pair.microphone.shape == pair.reference.shape == (12800,)
    assert pair.ranges == tuple(
        (index * 1600, (index + 1) * 1600) for index in range(8)
    )
    assert set(pair.receipt) == {
        "manifest_sha256",
        "timeline_sha256",
        "pre_gain_sha256",
        "reader_reference_sha256",
        "samples",
        "frame_count",
        "sample_rate_hz",
        "minimum_frame_samples",
        "maximum_frame_samples",
        "capture_source_groups",
        "capture_gaps",
    }
    assert (
        pair.receipt["capture_gaps"] == 0 and pair.receipt["capture_source_groups"] == 1
    )
    assert np.allclose(pair.microphone, 0.1, atol=1 / 32768)
    assert np.allclose(pair.reference, 0.25, atol=1 / 32768)


def test_individually_valid_tracks_cannot_be_flattened_into_a_pair(bundle):
    manifest, digest = bundle(mismatched=True)
    # Full bundle validity permits unequal track lengths per capture call.
    # This paired evaluator requires exact matching ranges per original frame.
    assert validate_manifest(manifest)
    with pytest.raises(evaluate.base.EvalError):
        evaluate.load_pair(manifest, digest)


def test_unknown_manifest_digest_refuses_before_bundle_reads(bundle, monkeypatch):
    manifest, _digest = bundle()
    called = []
    monkeypatch.setattr(evaluate, "validate_manifest", lambda _path: called.append(1))
    with pytest.raises(evaluate.base.EvalError):
        evaluate.load_pair(manifest, "0" * 64)
    assert called == []


def test_mutated_wave_and_incomplete_manifest_refuse(bundle):
    manifest, digest = bundle()
    value = json.loads(manifest.read_text())
    reference = manifest.parent / value["tracks"][evaluate.ROLES[1]]["file"]
    with reference.open("r+b") as stream:
        stream.seek(50)
        stream.write(b"changed")
    with pytest.raises(evaluate.base.EvalError):
        evaluate.load_pair(manifest, digest)
    value["complete"] = False
    manifest.write_text(json.dumps(value))
    with pytest.raises(evaluate.base.EvalError):
        evaluate.load_pair(manifest, hashlib.sha256(manifest.read_bytes()).hexdigest())


@pytest.mark.parametrize(
    "change", ["gap", "group", "sequence", "source_jump", "range", "oversized"]
)
def test_pair_geometry_and_source_continuity_are_not_inferred(
    bundle, monkeypatch, change
):
    manifest, _digest = bundle()
    value = json.loads(manifest.read_text())
    timeline = manifest.parent / value["timeline"]["file"]
    events = [json.loads(line) for line in timeline.read_bytes().splitlines()]
    frames = [event for event in events if event.get("kind") == "frame"]
    target = frames[3]
    if change == "gap":
        target["coordinate"]["gap_reason"] = "media_backpressure"
    elif change == "group":
        target["coordinate"]["capture_generation"] += 1
    elif change == "sequence":
        target["coordinate"]["sequence"] += 1
    elif change == "source_jump":
        target["coordinate"]["source_sample_start"] += 1
    elif change == "range":
        target["track_ranges"][evaluate.ROLES[1]]["sample_start"] += 1
    elif change == "oversized":
        value["tracks"][evaluate.ROLES[0]]["bytes"] = 100 * 1024**2
    raw = b"".join(json.dumps(event).encode() + b"\n" for event in events)
    timeline.write_bytes(raw)
    value["timeline"]["sha256"] = hashlib.sha256(raw).hexdigest()
    value["timeline"]["bytes"] = len(raw)
    manifest.write_text(json.dumps(value))
    # Exercise the evaluator's stricter paired boundary independently of the
    # existing bundle validator, which normally refuses these mutations too.
    monkeypatch.setattr(evaluate, "validate_manifest", lambda _path: True)
    with pytest.raises(evaluate.base.EvalError):
        evaluate.load_pair(manifest, hashlib.sha256(manifest.read_bytes()).hexdigest())


def test_buffered_native_hops_preserve_original_variable_capture_geometry():
    calls = []

    def process(_ctx, near, far, count, out):
        assert count == 256
        assert abs(far[0] - 0.25) < 1e-6
        ctypes.memmove(out, near, count * 4)
        calls.append(count)
        return 0

    adapter = evaluate.BufferedLocalVqe.__new__(evaluate.BufferedLocalVqe)
    adapter.ctx = 1
    adapter.library = SimpleNamespace(localvqe_process_frame_f32=process)
    adapter.near = adapter.far = np.zeros(0, dtype=np.float32)
    near = np.linspace(-0.5, 0.5, 12800, dtype=np.float32)
    far = np.full(12800, 0.25, dtype=np.float32)
    outputs = [
        adapter.process(near[start : start + 1600], far[start : start + 1600])
        for start in range(0, len(near), 1600)
    ]
    assert [len(item) for item in outputs] == [1536, 1536, 1536, 1792] * 2
    np.testing.assert_array_equal(np.concatenate(outputs), near)
    assert len(calls) == 50 and len(adapter.near) == len(adapter.far) == 0


class Passthrough:
    def __init__(self):
        self.lengths = []
        self.closed = False

    def process(self, near, far):
        self.lengths.append(len(near))
        return near.copy()

    def close(self):
        self.closed = True


def _result(bundle, monkeypatch):
    manifest, digest = bundle()
    pair = evaluate.load_pair(manifest, digest)
    adapter = Passthrough()
    result = evaluate.benchmark_pair(
        {}, "nlms-default", pair, adapter_factory=lambda *_: adapter
    )
    monkeypatch.setattr(evaluate.sys, "platform", "darwin")
    result.update(
        {
            "bindings_sha256": "a" * 64,
            "peak_rss_kib": None,
            "cpu_affinity_applied": False,
            "posix_resource_limits_applied": False,
            "thread_affinity": None,
        }
    )
    return pair, result, adapter


def test_recorded_metrics_are_energy_only_and_have_no_transcript_or_near_truth(
    bundle, monkeypatch
):
    pair, result, adapter = _result(bundle, monkeypatch)
    evaluate.validate_result(result, "nlms-default", pair, "a" * 64)
    assert adapter.lengths == [1600] * 8 and adapter.closed
    assert result["metrics"]["input_output_power_change_db"] == 0
    assert result["metrics"]["reference_active_frame_fraction"] == 1
    assert not any(
        "near" in key or "erle" in key or "text" in key for key in result["metrics"]
    )


@pytest.mark.parametrize(
    "change",
    [
        "text",
        "array",
        "bool_cpu",
        "nonfinite",
        "wrong_digest",
        "wrong_geometry",
        "wrong_frame_count",
        "invalid_fraction",
    ],
)
def test_parent_accepts_only_bound_scalar_recorded_results(bundle, monkeypatch, change):
    pair, result, _adapter = _result(bundle, monkeypatch)
    if change == "text":
        result["transcript"] = "private fake"
    elif change == "array":
        result["metrics"]["output_peak"] = [1, 2]
    elif change == "bool_cpu":
        result["process_cpu_seconds"] = True
    elif change == "nonfinite":
        result["metrics"]["output_rms"] = float("inf")
    elif change == "wrong_digest":
        result["input_binding"] = "b" * 64
    elif change == "wrong_geometry":
        result["samples"] -= 1
    elif change == "wrong_frame_count":
        result["frame_count"] += 1
    elif change == "invalid_fraction":
        result["metrics"]["reference_active_frame_fraction"] = 2
    with pytest.raises(evaluate.base.EvalError):
        evaluate.validate_result(result, "nlms-default", pair, "a" * 64)


def test_failed_processing_closes_and_preserves_the_real_failure_phase(bundle):
    manifest, digest = bundle()
    pair = evaluate.load_pair(manifest, digest)

    class Broken(Passthrough):
        def process(self, near, far):
            raise RuntimeError("private fake exception")

    adapter = Broken()
    phase = []
    with pytest.raises(RuntimeError):
        evaluate.benchmark_pair(
            {},
            "nlms-default",
            pair,
            adapter_factory=lambda *_: adapter,
            phase=phase.append,
        )
    assert adapter.closed and phase[-1] == "frames"


def test_cli_error_protocol_never_echoes_private_paths(capsys):
    assert evaluate.main(["--manifest", "/private/fake"]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "ok": False,
        "error": "localvqe_recorded_evaluation_failed",
    }
