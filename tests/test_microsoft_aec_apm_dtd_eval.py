from __future__ import annotations

import base64
import copy
from dataclasses import replace
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import types
import weakref

import numpy as np
import pytest

from tools import microsoft_aec_apm_dtd_eval as subject
from tools.prepare_microsoft_aec_fixture import (
    LoadedMicrosoftAecBundle,
    MicrosoftAecArtifact,
    MicrosoftAecCase,
)


_DIGESTS = tuple(f"{index:064x}" for index in range(1, 12))


class _TrackingProcessor:
    always_on = True
    suppresses_noise = True
    suppresses_nearend = True

    def __init__(self) -> None:
        self.blocks: list[tuple[float, float, float, float]] = []
        self.reset_calls = 0

    def process_16k(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        self.blocks.append(
            (
                float(np.max(np.abs(near))),
                float(np.max(np.abs(far))),
                float(near[-1]),
                float(far[-1]),
            )
        )
        factor = np.float32(0.92 if not np.any(far) else 0.55)
        return np.asarray(near * factor, dtype=np.float32)

    def reset(self) -> None:
        self.reset_calls += 1


class _Factory:
    def __init__(self, processor_type: type = _TrackingProcessor) -> None:
        self.processor_type = processor_type
        self.instances: list[object] = []

    def __call__(self, _config) -> object:
        processor = self.processor_type()
        self.instances.append(processor)
        return processor


def _artifact(index: int, role: str, samples: int) -> MicrosoftAecArtifact:
    return MicrosoftAecArtifact(
        name=f"private-{index:02d}-{role}.wav",
        role=role,
        size_bytes=44 + samples * 2,
        sha256=_DIGESTS[index + 1],
        samples=samples,
        snapshot=(index, samples),
    )


def _snapshot(
    tmp_path: Path,
    *,
    samples: int = 159_999,
) -> tuple[subject._BundleSnapshot, subject._CaseAudio]:
    rng = np.random.default_rng(20260809)
    far = np.asarray(0.06 * rng.standard_normal(samples), dtype=np.float32)
    echo = np.zeros(samples, dtype=np.float32)
    echo[80:] = np.float32(0.45) * far[:-80]
    time = np.arange(samples, dtype=np.float64) / subject.SAMPLE_RATE_HZ
    near = np.asarray(0.16 * np.sin(2.0 * np.pi * 223.0 * time), dtype=np.float32)
    scale = np.float32("0.5")
    mixture = np.asarray(echo + near * scale, dtype=np.float32)
    case = MicrosoftAecCase(
        case_id="private-aec-case-00",
        ordinal=0,
        rank_sha256=_DIGESTS[0],
        nearend_scale="0.5",
        ser="0",
        is_farend_nonlinear=False,
        is_farend_noisy=False,
        is_nearend_noisy=False,
        split="test",
        artifacts=tuple(
            _artifact(index, role, samples)
            for index, role in enumerate(subject.SIGNAL_ROLES)
        ),
    )
    bundle_root = tmp_path / "owner-private-audio"
    bundle_root.mkdir(mode=0o700, exist_ok=True)
    bundle = LoadedMicrosoftAecBundle(
        root=bundle_root,
        fixture_id="private-test-fixture",
        production_evidence=False,
        lock_recipe_sha256=_DIGESTS[5],
        manifest_sha256=_DIGESTS[6],
        receipt_sha256=_DIGESTS[7],
        source_contract_sha256=_DIGESTS[8],
        cases=(case,),
        identities=((1, 2, 3),),
    )
    tracks = {
        "echo_signal": echo,
        "farend_speech": far,
        "nearend_mic_signal": mixture,
        "nearend_speech": near,
    }
    for track in tracks.values():
        track.setflags(write=False)
    return subject._BundleSnapshot(bundle=bundle), subject._CaseAudio(
        case=case,
        tracks=tracks,
    )


def _install_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> subject._BundleSnapshot:
    snapshot, case_audio = _snapshot(tmp_path)
    monkeypatch.setattr(subject, "_load_bundle_snapshot", lambda _path: snapshot)
    monkeypatch.setattr(
        subject,
        "_read_case_audio",
        lambda _bundle, _case: case_audio,
    )
    monkeypatch.setattr(subject, "verify_microsoft_aec_bundle", lambda _bundle: None)
    return snapshot


def _fast_case(
    case,
    _config,
    _factory,
    totals: subject._ReplayTotals,
    *,
    production_geometry: bool,
    require_production_apm: bool,
) -> None:
    del production_geometry, require_production_apm
    samples = int(next(iter(case.tracks.values())).size)
    padding = (-samples) % subject.BLOCK_SAMPLES
    totals.source_samples += samples * 3
    totals.padding_samples += padding * 3
    totals.processed_samples += (samples + padding) * 3
    totals.near_projection_db.append(0.0)
    totals.near_cosine.append(1.0)
    totals.near_si_sdr.append(120.0)
    totals.echo_erle.append(6.0)
    totals.echo_energy += 4.0
    totals.residual_energy += 1.0
    totals.double_projection_db.append(-1.0)
    totals.double_cosine.append(0.8)
    totals.double_si_sdr.append(2.0)
    totals.double_interference.append(3.0)
    totals.echo_prefix_frames += 100
    totals.eligible_active_frames += 80
    totals.detected_active_frames += 0
    totals.d_score_echo.extend([0.0] * 100)
    totals.d_score_double.extend([0.0] * 80)
    totals.echo_false_cut_cases += 1


def _install_fast_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> subject._BundleSnapshot:
    snapshot = _install_snapshot(tmp_path, monkeypatch)
    monkeypatch.setattr(subject, "_evaluate_case", _fast_case)
    return snapshot


def _resign(report: dict) -> None:
    unsigned = dict(report)
    unsigned.pop("binding_sha256")
    report["binding_sha256"] = hashlib.sha256(
        subject._canonical_json(unsigned)
    ).hexdigest()


def test_replay_uses_exact_phase_order_state_carry_tail_and_source_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _install_snapshot(tmp_path, monkeypatch)
    factory = _Factory()
    guard_calls = 0

    def guard() -> None:
        nonlocal guard_calls
        guard_calls += 1

    report = subject.run_microsoft_aec_apm_dtd_eval(
        snapshot.bundle.root,
        run_guard=guard,
        _processor_factory=factory,
    )

    assert len(factory.instances) == 2
    near, continuous = factory.instances
    assert isinstance(near, _TrackingProcessor)
    assert isinstance(continuous, _TrackingProcessor)
    assert len(near.blocks) == 100
    assert len(continuous.blocks) == 200
    assert near.reset_calls == 1
    assert continuous.reset_calls == 1
    assert all(block[1] == 0.0 for block in near.blocks)
    assert all(block[1] > 0.0 for block in continuous.blocks[:99])
    assert max(block[0] for block in continuous.blocks[100:]) > max(
        block[0] for block in continuous.blocks[:100]
    )
    assert near.blocks[-1][2:] == (0.0, 0.0)
    assert continuous.blocks[99][2:] == (0.0, 0.0)
    assert continuous.blocks[-1][2:] == (0.0, 0.0)
    assert report["protocol"]["phase_order"] == [
        "near-only-fresh",
        "echo-prefix-fresh",
        "double-talk-same-state",
    ]
    assert report["protocol"]["near_target"] == "nearend-scale-f32-v1"
    assert report["coverage"] == {
        "cases": 1,
        "clipped_samples": 0,
        "dropped_samples": 0,
        "nonfinite_samples": 0,
        "padding_samples": 3,
        "phase_evaluations": 3,
        "processed_samples": 480_000,
        "scored_samples": 479_997,
        "source_complete": True,
        "source_samples": 479_997,
        "tracks": 4,
    }
    assert report["runtime"]["production_evidence"] is False
    assert report["runtime"]["livekit"] is None
    assert report["evidence"]["apm_component_replay"] is False
    assert report["evidence"]["oracle_gated_dtd_component_replay"] is True
    assert report["dtd"]["eligible_active_frames"] == 100
    assert report["dtd"]["d_score_double_talk"]["count"] == 100
    assert guard_calls >= 3


def test_sequential_replay_releases_case_audio_before_decoding_next(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, _case_audio = _snapshot(tmp_path)
    first = original.bundle.cases[0]
    second = replace(
        first,
        case_id="private-aec-case-01",
        ordinal=1,
        rank_sha256="e" * 64,
        artifacts=tuple(
            replace(
                artifact,
                name=artifact.name.replace("00", "01"),
                sha256=f"{index + 20:064x}",
            )
            for index, artifact in enumerate(first.artifacts)
        ),
    )
    snapshot = subject._BundleSnapshot(
        bundle=replace(original.bundle, cases=(first, second))
    )
    monkeypatch.setattr(subject, "_load_bundle_snapshot", lambda _path: snapshot)
    monkeypatch.setattr(subject, "verify_microsoft_aec_bundle", lambda _bundle: None)
    monkeypatch.setattr(subject, "_evaluate_case", _fast_case)
    previous: weakref.ReferenceType[np.ndarray] | None = None
    reads = 0

    def read(_bundle, case):
        nonlocal previous, reads
        gc.collect()
        if previous is not None:
            assert previous() is None
        samples = case.artifacts[0].samples
        track = np.ones(samples, dtype=np.float32)
        previous = weakref.ref(track)
        reads += 1
        return subject._CaseAudio(
            case=case,
            tracks={role: track for role in subject.SIGNAL_ROLES},
        )

    monkeypatch.setattr(subject, "_read_case_audio", read)
    report = subject.run_microsoft_aec_apm_dtd_eval(
        snapshot.bundle.root,
        _processor_factory=_Factory(),
    )
    gc.collect()

    assert reads == 2
    assert previous is not None and previous() is None
    assert report["coverage"]["cases"] == 2


def test_scaled_near_uses_one_exact_float32_decimal_conversion(tmp_path: Path) -> None:
    _snapshot_value, case = _snapshot(tmp_path)

    target = subject._scaled_near(case)

    expected = np.asarray(case.tracks["nearend_speech"], dtype=np.float32) * np.float32(
        "0.5"
    )
    assert target.dtype == np.float32
    assert np.array_equal(target, expected)


def test_oracle_inactive_frames_learn_echo_and_age_sustain_without_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _snapshot_value, original = _snapshot(tmp_path)
    samples = next(iter(original.tracks.values())).size
    near = np.zeros(samples, dtype=np.float32)
    time = np.arange(10 * subject.BLOCK_SAMPLES, dtype=np.float64)
    active = np.asarray(
        0.2 * np.sin(2.0 * np.pi * 223.0 * time / subject.SAMPLE_RATE_HZ),
        dtype=np.float32,
    )
    near[20 * subject.BLOCK_SAMPLES : 30 * subject.BLOCK_SAMPLES] = active
    tracks = dict(original.tracks)
    tracks["nearend_speech"] = near
    tracks["nearend_mic_signal"] = np.asarray(
        tracks["echo_signal"] + near * np.float32("0.5"),
        dtype=np.float32,
    )
    case = subject._CaseAudio(case=original.case, tracks=tracks)
    looks_calls = 0
    observe_calls = 0
    sustain_calls = 0
    real_observe = subject.AdaptiveDTD.observe_echo
    real_sustain = subject.BargeSustain.update

    def looks(engine, _processed, _raw):
        nonlocal looks_calls
        looks_calls += 1
        engine._dtd.last_D = 0.0
        return False

    def observe(dtd, raw_rms, resid_rms, incoherent):
        nonlocal observe_calls
        observe_calls += 1
        return real_observe(dtd, raw_rms, resid_rms, incoherent)

    def sustain(instance, eligible):
        nonlocal sustain_calls
        sustain_calls += 1
        return real_sustain(instance, eligible)

    monkeypatch.setattr(subject.SherpaOnnxEngine, "_looks_like_user", looks)
    monkeypatch.setattr(subject.AdaptiveDTD, "observe_echo", observe)
    monkeypatch.setattr(subject.BargeSustain, "update", sustain)
    totals = subject._ReplayTotals()

    subject._evaluate_case(
        case,
        subject._configuration_snapshot().config,
        _Factory(),
        totals,
        production_geometry=False,
        require_production_apm=False,
    )

    assert totals.eligible_active_frames == 10
    assert len(totals.d_score_double) == 10
    assert looks_calls == 110  # 100 echo diagnostics + 10 oracle-active DT frames.
    assert observe_calls == 90
    assert sustain_calls == 200


def test_sparse_oracle_activity_ages_out_sustain_eligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _snapshot_value, original = _snapshot(tmp_path)
    samples = next(iter(original.tracks.values())).size
    near = np.zeros(samples, dtype=np.float32)
    frame_time = np.arange(subject.BLOCK_SAMPLES, dtype=np.float64)
    active_frame = np.asarray(
        0.2 * np.sin(2.0 * np.pi * 223.0 * frame_time / subject.SAMPLE_RATE_HZ),
        dtype=np.float32,
    )
    for frame_index in range(5, 100, 10):
        start = frame_index * subject.BLOCK_SAMPLES
        stop = min(start + subject.BLOCK_SAMPLES, samples)
        near[start:stop] = active_frame[: stop - start]
    tracks = dict(original.tracks)
    tracks["nearend_speech"] = near
    tracks["nearend_mic_signal"] = np.asarray(
        tracks["echo_signal"] + near * np.float32("0.5"),
        dtype=np.float32,
    )
    case = subject._CaseAudio(case=original.case, tracks=tracks)
    gate_calls = 0

    def looks(engine, _processed, _raw):
        nonlocal gate_calls
        gate_calls += 1
        active_decision = gate_calls > 100
        engine._dtd.last_D = 10.0 if active_decision else 0.0
        return active_decision

    monkeypatch.setattr(subject.SherpaOnnxEngine, "_looks_like_user", looks)
    totals = subject._ReplayTotals()
    subject._evaluate_case(
        case,
        subject._configuration_snapshot().config,
        _Factory(),
        totals,
        production_geometry=False,
        require_production_apm=False,
    )

    assert gate_calls == 110
    assert totals.eligible_active_frames == 10
    assert totals.detected_active_frames == 10
    assert totals.detected_cases == 0
    assert totals.cut_latency_ms == []


def test_active_sample_mask_clips_tail_and_metrics_ignore_inactive_content() -> None:
    samples = 159_999
    target = np.zeros(samples, dtype=np.float32)
    active_start = 90 * subject.BLOCK_SAMPLES
    time = np.arange(samples - active_start, dtype=np.float64)
    target[active_start:] = np.asarray(
        0.2 * np.sin(2.0 * np.pi * 191.0 * time / subject.SAMPLE_RATE_HZ),
        dtype=np.float32,
    )
    oracle, mask = subject._activity_masks(target)

    assert oracle.active_frames == 10
    assert int(np.count_nonzero(mask)) == 10 * subject.BLOCK_SAMPLES - 1
    assert mask.shape == target.shape
    assert mask[-1]

    phase = np.arange(samples, dtype=np.float64)
    mixture = np.asarray(target + 0.04 * np.cos(phase * 0.013), dtype=np.float32)
    processed = np.asarray(target + 0.01 * np.sin(phase * 0.017), dtype=np.float32)
    baseline = subject._double_talk_metrics(processed, mixture, target, mask)
    changed_mixture = mixture.copy()
    changed_output = processed.copy()
    changed_mixture[~mask] = np.float32(0.91)
    changed_output[~mask] = np.float32(-0.87)

    assert subject._double_talk_metrics(
        changed_output,
        changed_mixture,
        target,
        mask,
    ) == pytest.approx(baseline)


def test_zero_output_is_a_bounded_catastrophic_result_not_a_failed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _install_snapshot(tmp_path, monkeypatch)

    class Zero(_TrackingProcessor):
        def process_16k(self, near, far):
            del near, far
            return np.zeros(subject.BLOCK_SAMPLES, dtype=np.float32)

    report = subject.run_microsoft_aec_apm_dtd_eval(
        snapshot.bundle.root,
        _processor_factory=_Factory(Zero),
    )

    for section in ("near_retention", "double_talk"):
        assert report[section]["projection_gain_db"]["min"] == -120.0
        assert report[section]["absolute_cosine"]["min"] == 0.0
        assert report[section]["si_sdr_db"]["min"] == -120.0
    assert report["echo_only"]["erle_db"]["min"] == 120.0


@pytest.mark.parametrize("bad_kind", ("short", "integer", "nan"))
def test_bad_processor_output_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_kind: str,
) -> None:
    snapshot = _install_snapshot(tmp_path, monkeypatch)

    class Bad(_TrackingProcessor):
        def process_16k(self, near, far):
            del far
            if bad_kind == "short":
                return np.asarray(near[:-1], dtype=np.float32)
            if bad_kind == "integer":
                return np.zeros(subject.BLOCK_SAMPLES, dtype=np.int16)
            result = np.asarray(near, dtype=np.float32).copy()
            result[0] = np.nan
            return result

    with pytest.raises(subject.MicrosoftAecApmDtdEvalError):
        subject.run_microsoft_aec_apm_dtd_eval(
            snapshot.bundle.root,
            _processor_factory=_Factory(Bad),
        )


def test_report_is_closed_aggregate_private_and_cross_checked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _install_fast_run(tmp_path, monkeypatch)
    report = subject.run_microsoft_aec_apm_dtd_eval(
        snapshot.bundle.root,
        _processor_factory=_Factory(),
    )
    encoded = subject._canonical_json(report).decode("ascii")

    assert set(report) == {
        "binding_sha256",
        "coverage",
        "double_talk",
        "dtd",
        "echo_only",
        "evaluator",
        "evidence",
        "fixture",
        "kind",
        "near_retention",
        "protocol",
        "runtime",
        "schema_version",
    }
    assert "private-aec-case-00" not in encoded
    assert "private-test-fixture" not in encoded
    assert str(snapshot.bundle.root) not in encoded
    assert ".wav" not in encoded
    assert report["dtd"]["echo_false_cut_cases"] == 1
    assert report["dtd"]["echo_false_cut_rate"] == 1.0
    assert "echo_false_cut_frames" not in report["dtd"]

    extra = copy.deepcopy(report)
    extra["case_rows"] = []
    _resign(extra)
    with pytest.raises(subject.MicrosoftAecApmDtdEvalError):
        subject._validate_report(extra, bundle=snapshot)

    bad_math = copy.deepcopy(report)
    bad_math["coverage"]["scored_samples"] += 1
    _resign(bad_math)
    with pytest.raises(subject.MicrosoftAecApmDtdEvalError):
        subject._validate_report(bad_math, bundle=snapshot)

    bad_summary = copy.deepcopy(report)
    bad_summary["near_retention"]["absolute_cosine"]["count"] = 2
    _resign(bad_summary)
    with pytest.raises(subject.MicrosoftAecApmDtdEvalError):
        subject._validate_report(bad_summary, bundle=snapshot)


@pytest.mark.parametrize("surface", ("fixture", "coverage", "config", "runtime"))
def test_report_rebinds_to_retained_inputs_not_only_self_consistency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    snapshot = _install_fast_run(tmp_path, monkeypatch)
    report = subject.run_microsoft_aec_apm_dtd_eval(
        snapshot.bundle.root,
        _processor_factory=_Factory(),
    )
    forged = copy.deepcopy(report)
    closure = subject._execution_closure()
    configuration = subject._configuration_snapshot()
    runtime = subject._runtime_binding(injected=True)
    if surface == "fixture":
        forged["fixture"]["receipt_sha256"] = "f" * 64
    elif surface == "coverage":
        forged["coverage"]["source_samples"] += 3
        forged["coverage"]["scored_samples"] += 3
        forged["coverage"]["processed_samples"] += 3
    elif surface == "config":
        forged["runtime"]["configuration"]["values"]["dtd_k"] += 1.0
        values = forged["runtime"]["configuration"]["values"]
        forged["runtime"]["configuration"]["sherpa_sha256"] = hashlib.sha256(
            subject._canonical_json(values)
        ).hexdigest()
    else:
        forged["runtime"].update(
            {
                "implementation": "livekit.rtc.AudioProcessingModule",
                "livekit": {
                    "content_set_sha256": "1" * 64,
                    "distribution": "livekit",
                    "file_count": 1,
                    "maximum_file_bytes": 1,
                    "record_sha256": "2" * 64,
                    "record_size_bytes": 1,
                    "total_bytes": 1,
                    "version": "1.1.14",
                },
                "production_evidence": True,
            }
        )
        forged["evidence"]["apm_component_replay"] = True
    _resign(forged)

    subject._validate_report(forged, closure=closure)
    with pytest.raises(subject.MicrosoftAecApmDtdEvalError):
        subject._validate_report(
            forged,
            bundle=snapshot,
            closure=closure,
            configuration=configuration,
            runtime=runtime,
        )


@pytest.mark.parametrize("surface", ("bundle", "closure", "config", "runtime", "guard"))
def test_post_replay_drift_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    snapshot = _install_fast_run(tmp_path, monkeypatch)
    output_parent = tmp_path / "private-output"
    output_parent.mkdir(mode=0o700)
    output = output_parent / "aggregate.json"
    guard_calls = 0

    def guard() -> None:
        nonlocal guard_calls
        guard_calls += 1
        if surface == "guard" and guard_calls >= 2:
            raise RuntimeError("private blocker detail")

    if surface == "bundle":
        calls = 0

        def verify(_bundle) -> None:
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise RuntimeError("private bundle mutation")

        monkeypatch.setattr(subject, "verify_microsoft_aec_bundle", verify)
    elif surface == "closure":
        real = subject._execution_closure
        calls = 0

        def closure():
            nonlocal calls
            calls += 1
            value = real()
            return replace(value, sha256="0" * 64) if calls >= 3 else value

        monkeypatch.setattr(subject, "_execution_closure", closure)
    elif surface == "config":
        real = subject._configuration_snapshot
        calls = 0

        def configuration():
            nonlocal calls
            calls += 1
            value = real()
            return replace(value, source_sha256="0" * 64) if calls >= 3 else value

        monkeypatch.setattr(subject, "_configuration_snapshot", configuration)
    elif surface == "runtime":
        real = subject._runtime_binding
        calls = 0

        def runtime(*, injected: bool, retained=None):
            nonlocal calls
            calls += 1
            value = real(injected=injected, retained=retained)
            if calls < 3:
                return value
            changed = dict(value.binding)
            changed["implementation"] = "drifted-test-double"
            return replace(value, binding=changed)

        monkeypatch.setattr(subject, "_runtime_binding", runtime)

    with pytest.raises(subject.MicrosoftAecApmDtdEvalError):
        subject.run_microsoft_aec_apm_dtd_eval(
            snapshot.bundle.root,
            output_path=output,
            run_guard=guard,
            _processor_factory=_Factory(),
        )
    assert not output.exists()


def test_terminal_private_publication_and_no_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _install_fast_run(tmp_path, monkeypatch)
    parent = tmp_path / "private-output"
    parent.mkdir(mode=0o700)
    output = parent / "aggregate.json"

    report = subject.run_microsoft_aec_apm_dtd_eval(
        snapshot.bundle.root,
        output_path=output,
        _processor_factory=_Factory(),
    )

    assert output.read_bytes() == subject._canonical_json(report, newline=True)
    metadata = output.lstat()
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_nlink == 1
    with pytest.raises(subject.MicrosoftAecApmDtdEvalError):
        subject.run_microsoft_aec_apm_dtd_eval(
            snapshot.bundle.root,
            output_path=output,
            _processor_factory=_Factory(),
        )


@pytest.mark.parametrize(
    "destination",
    ("equal_bundle", "inside_bundle", "git_worktree"),
)
def test_output_rejects_bundle_mutation_and_git_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    destination: str,
) -> None:
    snapshot = _install_fast_run(tmp_path, monkeypatch)
    if destination == "equal_bundle":
        output = snapshot.bundle.root
    elif destination == "inside_bundle":
        output = snapshot.bundle.root / "aggregate.json"
    else:
        repository = tmp_path / "private-repository"
        repository.mkdir(mode=0o700)
        (repository / ".git").mkdir(mode=0o700)
        (repository / ".git" / "HEAD").write_text(
            "ref: refs/heads/main\n",
            encoding="ascii",
        )
        output = repository / "aggregate.json"

    with pytest.raises(subject.MicrosoftAecApmDtdEvalError):
        subject.run_microsoft_aec_apm_dtd_eval(
            snapshot.bundle.root,
            output_path=output,
            _processor_factory=_Factory(),
        )
    if destination == "equal_bundle":
        assert output.is_dir() and not tuple(output.iterdir())
    else:
        assert not output.exists()


def test_prelink_rebind_rejects_renamed_output_parent(tmp_path: Path) -> None:
    parent = tmp_path / "private-output"
    parent.mkdir(mode=0o700)
    moved = tmp_path / "moved-output"
    output = parent / "aggregate.json"

    def replace_parent() -> None:
        parent.rename(moved)
        parent.mkdir(mode=0o700)

    with pytest.raises(subject.MicrosoftAecApmDtdEvalError):
        subject._publish_report(
            output,
            {"safe": True},
            commit_guard=replace_parent,
            state=subject._ReportCommitState(),
        )
    assert not output.exists()
    assert not (moved / output.name).exists()


def test_post_guard_staged_file_mutation_never_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "private-output"
    parent.mkdir(mode=0o700)
    output = parent / "aggregate.json"
    real_verify = subject._verify_staged_report
    calls = 0

    def verify(descriptor: int, encoded: bytes, digest: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            os.fchmod(descriptor, 0o640)
        real_verify(descriptor, encoded, digest)

    monkeypatch.setattr(subject, "_verify_staged_report", verify)
    with pytest.raises(subject.MicrosoftAecApmDtdEvalError):
        subject._publish_report(
            output,
            {"safe": True},
            commit_guard=lambda: None,
            state=subject._ReportCommitState(),
        )
    assert calls == 2
    assert not output.exists()


def test_publication_lifecycle_is_prelink_abort_and_postlink_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "private-output"
    parent.mkdir(mode=0o700)
    before = parent / "before.json"
    state = subject._ReportCommitState()

    def interrupted() -> None:
        raise subject._LifecycleSignal(signal_number)

    signal_number = int(getattr(subject.signal, "SIGTERM", 15))
    with pytest.raises(subject._LifecycleSignal):
        subject._publish_report(
            before,
            {"safe": True},
            commit_guard=interrupted,
            state=state,
        )
    assert not before.exists()
    assert state.committed is False

    after = parent / "after.json"
    state = subject._ReportCommitState()
    real_fsync = subject.os.fsync
    calls = 0

    def fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise subject._LifecycleSignal(signal_number)
        real_fsync(descriptor)

    monkeypatch.setattr(subject.os, "fsync", fsync)
    digest = subject._publish_report(
        after,
        {"safe": True},
        commit_guard=lambda: None,
        state=state,
    )
    expected = subject._canonical_json({"safe": True}, newline=True)
    assert after.read_bytes() == expected
    assert digest == hashlib.sha256(expected).hexdigest()
    assert state.committed is True


def test_ambiguous_link_recovery_fsyncs_directory_before_returning_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "private-output"
    parent.mkdir(mode=0o700)
    output = parent / "aggregate.json"
    state = subject._ReportCommitState()
    real_link = subject.os.link
    real_fsync = subject.os.fsync
    fsynced_directory = False
    signal_number = int(getattr(subject.signal, "SIGTERM", 15))

    def link(*args, **kwargs) -> None:
        real_link(*args, **kwargs)
        raise subject._LifecycleSignal(signal_number)

    def fsync(descriptor: int) -> None:
        nonlocal fsynced_directory
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            fsynced_directory = True
        real_fsync(descriptor)

    monkeypatch.setattr(subject.os, "link", link)
    monkeypatch.setattr(subject.os, "fsync", fsync)
    digest = subject._publish_report(
        output,
        {"safe": True},
        commit_guard=lambda: None,
        state=state,
    )

    encoded = subject._canonical_json({"safe": True}, newline=True)
    assert output.read_bytes() == encoded
    assert digest == hashlib.sha256(encoded).hexdigest()
    assert state.committed is True
    assert fsynced_directory is True


def test_run_returns_retained_report_after_terminal_commit_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _install_fast_run(tmp_path, monkeypatch)
    parent = tmp_path / "private-output"
    parent.mkdir(mode=0o700)
    output = parent / "aggregate.json"
    signal_number = int(getattr(subject.signal, "SIGTERM", 15))

    def publish(path, report, *, commit_guard, state) -> str:
        commit_guard()
        encoded = subject._canonical_json(report, newline=True)
        path.write_bytes(encoded)
        path.chmod(0o600)
        state.committed = True
        state.digest = hashlib.sha256(encoded).hexdigest()
        raise subject._LifecycleSignal(signal_number)

    monkeypatch.setattr(subject, "_publish_report", publish)
    report = subject.run_microsoft_aec_apm_dtd_eval(
        snapshot.bundle.root,
        output_path=output,
        _processor_factory=_Factory(),
    )

    assert report["binding_sha256"]
    assert output.exists()


def _record_hash(raw: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode(
        "ascii"
    ).rstrip("=")


def _drop_livekit_modules() -> None:
    for name in tuple(sys.modules):
        if name == "livekit" or name.startswith("livekit."):
            sys.modules.pop(name, None)
    subject.importlib.invalidate_caches()


@pytest.fixture
def clean_livekit_execution_surface():
    _drop_livekit_modules()
    yield
    _drop_livekit_modules()


def _install_fake_livekit_distribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    namespace_resources: bool = False,
) -> tuple[Path, dict[str, bytes]]:
    root = tmp_path / "site-packages"
    members = {
        "livekit/rtc/__init__.py": (
            b"from .apm import AudioProcessingModule\n"
            b"from .audio_frame import AudioFrame\n"
            b"from .version import __version__\n"
        ),
        "livekit/rtc/_proto/__init__.py": b'"""Synthetic protobuf package."""\n',
        "livekit/rtc/_proto/ffi_pb2.py": (
            b"from types import SimpleNamespace\n"
            b"class _Message:\n"
            b"    def SerializeToString(self): return b''\n"
            b"    def ParseFromString(self, _raw): return 0\n"
            b"class FfiRequest(_Message):\n"
            b"    def __init__(self):\n"
            b"        self.new_apm = SimpleNamespace()\n"
            b"        self.apm_process_stream = SimpleNamespace()\n"
            b"        self.apm_process_reverse_stream = SimpleNamespace()\n"
            b"        self.apm_set_stream_delay = SimpleNamespace()\n"
            b"class FfiResponse(_Message):\n"
            b"    def __init__(self):\n"
            b"        handle = SimpleNamespace(id=1)\n"
            b"        self.new_apm = SimpleNamespace(\n"
            b"            apm=SimpleNamespace(handle=handle)\n"
            b"        )\n"
            b"        self.apm_process_stream = SimpleNamespace(error='')\n"
            b"        self.apm_process_reverse_stream = SimpleNamespace(error='')\n"
            b"        self.apm_set_stream_delay = SimpleNamespace(error='')\n"
            b"class FfiEvent(_Message): pass\n"
            b"class _LogLevel:\n"
            b"    LOG_ERROR = 0\n"
            b"    def Name(self, value): return str(value)\n"
            b"    def Value(self, _name): return self.LOG_ERROR\n"
            b"LogLevel = _LogLevel()\n"
        ),
        "livekit/rtc/_utils.py": (
            b"import ctypes\n"
            b"class classproperty:\n"
            b"    def __init__(self, function):\n"
            b"        self.f = classmethod(function)\n"
            b"    def __get__(self, *args):\n"
            b"        return self.f.__get__(*args)()\n"
            b"def _ensure_compatible_buffer(data): return data\n"
            b"def get_address(data):\n"
            b"    if isinstance(data, bytearray):\n"
            b"        return ctypes.addressof(ctypes.c_char.from_buffer(data))\n"
            b"    return 0\n"
        ),
        "livekit/rtc/version.py": b"__version__ = '1.1.14'\n",
        "livekit/rtc/_ffi_client.py": (
            b"import atexit\n"
            b"import ctypes\n"
            b"import importlib.resources\n"
            b"import logging\n"
            b"import os\n"
            b"import platform\n"
            b"import signal\n"
            b"import sys\n"
            b"import threading\n"
            b"from contextlib import ExitStack\n"
            b"from ._proto import ffi_pb2 as proto_ffi\n"
            b"from ._utils import classproperty\n"
            b"from .version import __version__\n"
            b"logger = logging.getLogger('synthetic-livekit')\n"
            b"_resource_files = ExitStack()\n"
            b"atexit.register(_resource_files.close)\n"
            b"def _lib_name(): return 'liblivekit_ffi.so'\n"
            b"def get_ffi_lib():\n"
            b"    override = os.environ.get('LIVEKIT_LIB_PATH', '').strip()\n"
            b"    if override:\n"
            b"        path = override\n"
            b"    else:\n"
            b"        path = importlib.resources.files(\n"
            b"            'livekit.rtc.resources'\n"
            b"        ).joinpath('liblivekit_ffi.so')\n"
            b"    library = ctypes.CDLL(None)\n"
            b"    library._name = str(path)\n"
            b"    def initialize(*_args): return None\n"
            b"    def request(*_args): return 1\n"
            b"    def drop_handle(*_args): return True\n"
            b"    def dispose(*_args): return None\n"
            b"    library.livekit_ffi_initialize = initialize\n"
            b"    library.livekit_ffi_request = request\n"
            b"    library.livekit_ffi_drop_handle = drop_handle\n"
            b"    library.livekit_ffi_dispose = dispose\n"
            b"    return library\n"
            b"ffi_cb_fnc = ctypes.CFUNCTYPE(\n"
            b"    None, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t\n"
            b")\n"
            b"INVALID_HANDLE = 0\n"
            b"class FfiHandle:\n"
            b"    def __init__(self, handle):\n"
            b"        self.handle = handle\n"
            b"        self._disposed = False\n"
            b"    def __del__(self): self.dispose()\n"
            b"    def dispose(self):\n"
            b"        if self.handle != INVALID_HANDLE and not self._disposed:\n"
            b"            self._disposed = True\n"
            b"            FfiClient.instance._ffi_lib.livekit_ffi_drop_handle(\n"
            b"                ctypes.c_uint64(self.handle)\n"
            b"            )\n"
            b"class FfiQueue:\n"
            b"    def __init__(self):\n"
            b"        self._lock = threading.RLock()\n"
            b"        self._items = []\n"
            b"    def put(self, item): self._items.append(item)\n"
            b"@ctypes.CFUNCTYPE(\n"
            b"    None, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t\n"
            b")\n"
            b"def ffi_event_callback(_data, _size): return None\n"
            b"def to_python_level(_level): return None\n"
            b"class FfiClient:\n"
            b"    _instance = None\n"
            b"    @classproperty\n"
            b"    def instance(cls):\n"
            b"        if cls._instance is None:\n"
            b"            cls._instance = FfiClient()\n"
            b"        return cls._instance\n"
            b"    def __init__(self):\n"
            b"        self._lock = threading.RLock()\n"
            b"        self._queue = FfiQueue()\n"
            b"        self._ffi_lib = get_ffi_lib()\n"
            b"        self._ffi_lib.livekit_ffi_initialize.argtypes = [\n"
            b"            ffi_cb_fnc, ctypes.c_bool, ctypes.c_char_p, ctypes.c_char_p\n"
            b"        ]\n"
            b"        self._ffi_lib.livekit_ffi_initialize.restype = ctypes.c_int\n"
            b"        self._ffi_lib.livekit_ffi_request.argtypes = [ctypes.c_void_p]\n"
            b"        self._ffi_lib.livekit_ffi_request.restype = ctypes.c_uint64\n"
            b"        self._ffi_lib.livekit_ffi_drop_handle.argtypes = [\n"
            b"            ctypes.c_uint64\n"
            b"        ]\n"
            b"        self._ffi_lib.livekit_ffi_drop_handle.restype = ctypes.c_bool\n"
            b"        self._ffi_lib.livekit_ffi_dispose.argtypes = []\n"
            b"        self._ffi_lib.livekit_ffi_dispose.restype = None\n"
            b"        self._ffi_lib.livekit_ffi_initialize(\n"
            b"            ffi_event_callback, True, b'python', __version__.encode()\n"
            b"        )\n"
            b"    @property\n"
            b"    def queue(self): return self._queue\n"
            b"    def request(self, _request):\n"
            b"        response = proto_ffi.FfiResponse()\n"
            b"        FfiHandle(1).dispose()\n"
            b"        return response\n"
        ),
        "livekit/rtc/apm.py": (
            b"from ._ffi_client import FfiClient, FfiHandle\n"
            b"from ._proto import ffi_pb2 as proto_ffi\n"
            b"from ._utils import get_address\n"
            b"from .audio_frame import AudioFrame\n"
            b"class AudioProcessingModule:\n"
            b"    construction_count = 0\n"
            b"    def __init__(self, **_kwargs):\n"
            b"        type(self).construction_count += 1\n"
            b"        request = proto_ffi.FfiRequest()\n"
            b"        response = FfiClient.instance.request(request)\n"
            b"        self._ffi_handle = FfiHandle(\n"
            b"            response.new_apm.apm.handle.id\n"
            b"        )\n"
            b"    def process_stream(self, data):\n"
            b"        request = proto_ffi.FfiRequest()\n"
            b"        request.data_ptr = get_address(data._data)\n"
            b"        FfiClient.instance.request(request)\n"
            b"    def process_reverse_stream(self, data):\n"
            b"        request = proto_ffi.FfiRequest()\n"
            b"        request.data_ptr = get_address(data._data)\n"
            b"        FfiClient.instance.request(request)\n"
            b"    def set_stream_delay_ms(self, _delay):\n"
            b"        FfiClient.instance.request(proto_ffi.FfiRequest())\n"
        ),
        "livekit/rtc/audio_frame.py": (
            b"import ctypes\n"
            b"from ._ffi_client import FfiHandle\n"
            b"from ._utils import _ensure_compatible_buffer, get_address\n"
            b"class AudioFrame:\n"
            b"    def __init__(\n"
            b"        self, data, sample_rate, num_channels, samples_per_channel\n"
            b"    ):\n"
            b"        self._data = _ensure_compatible_buffer(data)\n"
            b"        self._sample_rate = sample_rate\n"
            b"        self._num_channels = num_channels\n"
            b"        self._samples_per_channel = samples_per_channel\n"
            b"    @property\n"
            b"    def data(self): return memoryview(self._data).cast('h')\n"
            b"    @property\n"
            b"    def sample_rate(self): return self._sample_rate\n"
            b"    @property\n"
            b"    def num_channels(self): return self._num_channels\n"
        ),
        "livekit/rtc/resources/liblivekit_ffi.so": b"synthetic-native-bytes",
    }
    if not namespace_resources:
        members["livekit/rtc/resources/__init__.py"] = (
            b'"""Verified synthetic resource package."""\n'
        )
    for relative, raw in members.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    dist_info = root / "livekit-1.1.14.dist-info"
    dist_info.mkdir()
    record_relative = "livekit-1.1.14.dist-info/RECORD"
    record_rows = [
        f"{relative},{_record_hash(raw)},{len(raw)}"
        for relative, raw in sorted(members.items())
    ]
    record_rows.append(f"{record_relative},,")
    (root / record_relative).write_text(
        "\n".join(record_rows) + "\n",
        encoding="utf-8",
    )

    class Distribution:
        version = "1.1.14"
        files = tuple(
            PurePosixPath(relative) for relative in (*members, record_relative)
        )

        @staticmethod
        def locate_file(relative) -> Path:
            return root / str(relative)

    isolated_path = [str(root)]
    for item in sys.path:
        base = Path(item or Path.cwd())
        if not (base / "livekit").exists():
            isolated_path.append(item)
    monkeypatch.setattr(subject.sys, "path", isolated_path)
    monkeypatch.setattr(
        subject.importlib_metadata,
        "distribution",
        lambda name: Distribution() if name == "livekit" else None,
    )
    monkeypatch.delenv("LIVEKIT_LIB_PATH", raising=False)
    subject.importlib.invalidate_caches()
    return root, members


def test_clean_process_eager_local_imports_equal_the_bound_closure() -> None:
    script = """
import json
from pathlib import Path
import sys

root = Path.cwd().resolve()
import tools.microsoft_aec_apm_dtd_eval

observed = set()
for module in tuple(sys.modules.values()):
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str):
        continue
    try:
        relative = Path(raw_path).resolve(strict=True).relative_to(root).as_posix()
    except (OSError, RuntimeError, ValueError):
        continue
    if relative.endswith(".py") and relative.split("/", 1)[0] in {
        "always_on_agent", "core", "tools"
    }:
        observed.add(relative)
print(json.dumps(sorted(observed), separators=(",", ":")))
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert tuple(json.loads(completed.stdout)) == subject._EAGER_LOCAL_IMPORT_FILES
    assert subject._CLOSURE_FILES == tuple(
        sorted(
            (
                *subject._EAGER_LOCAL_IMPORT_FILES,
                "core/engines/_apm.py",
                "requirements-remote.txt",
            )
        )
    )


def test_cli_returns_success_after_committed_report_when_terminal_io_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_parent = tmp_path / "private-output"
    output_parent.mkdir(mode=0o700)
    output = output_parent / "aggregate.json"
    binding = "a" * 64

    def run(
        _bundle,
        *,
        output_path,
        run_guard,
        _processor_factory,
        _commit_state,
    ):
        assert callable(run_guard)
        assert _processor_factory is None
        assert output_path == output
        output.write_bytes(b"committed\n")
        _commit_state.committed = True
        _commit_state.digest = "b" * 64
        return {"binding_sha256": binding}

    signal_calls = 0

    def install_handler(_signum, _handler):
        nonlocal signal_calls
        signal_calls += 1
        if signal_calls > 2:
            raise OSError("late restore failure")
        return object()

    monkeypatch.setattr(subject, "_run_microsoft_aec_apm_dtd_eval", run)
    monkeypatch.setattr(subject.signal, "signal", install_handler)
    monkeypatch.setattr(
        subject.os,
        "write",
        lambda _descriptor, _raw: (_ for _ in ()).throw(OSError("stdout failed")),
    )

    assert subject.main(["--bundle", str(tmp_path), "--output", str(output)]) == 0
    assert output.read_bytes() == b"committed\n"
    assert signal_calls == 4


def test_cli_shared_state_closes_post_runner_pre_terminal_signal_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_parent = tmp_path / "private-output"
    output_parent.mkdir(mode=0o700)
    output = output_parent / "aggregate.json"
    signal_number = int(getattr(subject.signal, "SIGTERM", 15))

    def run(
        _bundle,
        *,
        output_path,
        run_guard,
        _processor_factory,
        _commit_state,
    ):
        del run_guard, _processor_factory
        assert output_path == output
        output.write_bytes(b"committed\n")
        _commit_state.committed = True
        _commit_state.digest = "c" * 64
        return {"binding_sha256": "a" * 64}

    def interrupted(_value, *, newline=False):
        del newline
        raise subject._LifecycleSignal(signal_number)

    monkeypatch.setattr(subject, "_run_microsoft_aec_apm_dtd_eval", run)
    monkeypatch.setattr(subject, "_canonical_json", interrupted)

    assert subject.main(["--bundle", str(tmp_path), "--output", str(output)]) == 0
    assert output.read_bytes() == b"committed\n"


def test_runtime_closure_hashes_every_record_member_and_detects_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_livekit_execution_surface,
) -> None:
    del clean_livekit_execution_surface
    root, members = _install_fake_livekit_distribution(tmp_path, monkeypatch)

    snapshot = subject._runtime_binding(injected=False)

    livekit = snapshot.binding["livekit"]
    assert livekit["version"] == "1.1.14"
    assert livekit["file_count"] == len(members)
    assert livekit["total_bytes"] == sum(map(len, members.values()))
    assert len(snapshot.files) == len(members)
    assert all(item.identity for item in snapshot.files)
    assert tuple(item.name for item in snapshot.modules) == (
        "livekit",
        "livekit.rtc",
        "livekit.rtc._ffi_client",
        "livekit.rtc._proto",
        "livekit.rtc._proto.ffi_pb2",
        "livekit.rtc._utils",
        "livekit.rtc.apm",
        "livekit.rtc.audio_frame",
        "livekit.rtc.resources",
        "livekit.rtc.version",
    )
    assert snapshot.library_state.initialization_attempted is True
    library = snapshot.library_state.cdll
    client = snapshot.library_state.client
    assert client is not None
    assert library is not None
    assert snapshot.library_state.queue is client._queue
    assert snapshot.library_state.native_handle == library._handle
    assert len(snapshot.library_state.native_functions) == 4
    assert len(snapshot.library_state.native_signatures) == 4
    assert snapshot.library_state.resolved_name == str(
        root / "livekit/rtc/resources/liblivekit_ffi.so"
    )
    assert (
        Path(snapshot.library_state.resolved_name).read_bytes()
        == members["livekit/rtc/resources/liblivekit_ffi.so"]
    )
    apm_class = sys.modules["livekit.rtc"].AudioProcessingModule
    assert apm_class.construction_count == 0
    loaded_before = subject._loaded_livekit_names()
    processor = apm_class()
    assert processor._ffi_handle.handle == 1
    assert apm_class.construction_count == 1
    assert subject._loaded_livekit_names() == loaded_before

    rebound = subject._runtime_binding(injected=False, retained=snapshot)
    assert rebound == snapshot
    assert rebound.execution_identity == snapshot.execution_identity
    assert subject._runtime_binding(injected=False, retained=snapshot) == snapshot
    assert snapshot.library_state.client is client
    assert snapshot.library_state.cdll is library
    assert not (root / "livekit/rtc/__pycache__").exists()

    native = root / "livekit/rtc/resources/liblivekit_ffi.so"
    raw = bytearray(native.read_bytes())
    raw[0] ^= 1
    native.write_bytes(raw)
    with pytest.raises(subject.MicrosoftAecApmDtdEvalError):
        subject._runtime_binding(injected=False, retained=snapshot)


def test_runtime_binds_exact_one_location_resource_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_livekit_execution_surface,
) -> None:
    del clean_livekit_execution_surface
    root, members = _install_fake_livekit_distribution(
        tmp_path,
        monkeypatch,
        namespace_resources=True,
    )

    snapshot = subject._runtime_binding(injected=False)
    resources = sys.modules["livekit.rtc.resources"]
    assert resources.__spec__.origin is None
    assert (
        type(resources.__spec__.loader) is subject.importlib_machinery.NamespaceLoader
    )
    assert tuple(Path(item) for item in resources.__path__) == (
        root / "livekit/rtc/resources",
    )
    retained_module = next(
        item for item in snapshot.modules if item.name == "livekit.rtc.resources"
    )
    assert retained_module.module_path_identity == id(resources.__path__)
    assert retained_module.spec_locations_identity == id(
        resources.__spec__.submodule_search_locations
    )

    loaded_before = subject._loaded_livekit_names()
    processor = sys.modules["livekit.rtc"].AudioProcessingModule()
    assert processor._ffi_handle.handle == 1
    assert (
        Path(snapshot.library_state.resolved_name).read_bytes()
        == members["livekit/rtc/resources/liblivekit_ffi.so"]
    )
    assert subject._loaded_livekit_names() == loaded_before
    assert subject._runtime_binding(injected=False, retained=snapshot) == snapshot
    assert subject._runtime_binding(injected=False, retained=snapshot) == snapshot


@pytest.mark.parametrize("override", (" ", "\t", "override.so"))
def test_runtime_rejects_livekit_library_override_at_initial_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_livekit_execution_surface,
    override: str,
) -> None:
    del clean_livekit_execution_surface
    _install_fake_livekit_distribution(tmp_path, monkeypatch)
    monkeypatch.setenv("LIVEKIT_LIB_PATH", override)

    with pytest.raises(subject.MicrosoftAecApmDtdEvalError) as raised:
        subject._runtime_binding(injected=False)
    assert raised.value.args == ()
    assert subject._loaded_livekit_names() == ()


def test_runtime_accepts_empty_livekit_library_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_livekit_execution_surface,
) -> None:
    del clean_livekit_execution_surface
    _install_fake_livekit_distribution(tmp_path, monkeypatch)
    monkeypatch.setenv("LIVEKIT_LIB_PATH", "")

    snapshot = subject._runtime_binding(injected=False)

    assert snapshot.binding["production_evidence"] is True


def test_runtime_cleans_initial_imports_after_mid_bind_environment_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_livekit_execution_surface,
) -> None:
    del clean_livekit_execution_surface
    _install_fake_livekit_distribution(tmp_path, monkeypatch)
    validate_tree = subject._validate_livekit_tree
    calls = 0

    def mutate_after_pre_import_validation(*args, **kwargs):
        nonlocal calls
        validate_tree(*args, **kwargs)
        calls += 1
        if calls == 1:
            monkeypatch.setenv("LIVEKIT_LIB_PATH", str(tmp_path / "late.so"))

    monkeypatch.setattr(
        subject,
        "_validate_livekit_tree",
        mutate_after_pre_import_validation,
    )
    with pytest.raises(subject.MicrosoftAecApmDtdEvalError):
        subject._runtime_binding(injected=False)
    assert calls == 1
    assert subject._loaded_livekit_names() == ()

    monkeypatch.delenv("LIVEKIT_LIB_PATH")
    assert (
        subject._runtime_binding(injected=False).binding["production_evidence"] is True
    )


def test_runtime_rejects_livekit_library_override_after_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_livekit_execution_surface,
) -> None:
    del clean_livekit_execution_surface
    _install_fake_livekit_distribution(tmp_path, monkeypatch)
    snapshot = subject._runtime_binding(injected=False)
    monkeypatch.setenv("LIVEKIT_LIB_PATH", str(tmp_path / "override.so"))

    with pytest.raises(subject.MicrosoftAecApmDtdEvalError) as raised:
        subject._runtime_binding(injected=False, retained=snapshot)
    assert raised.value.args == ()


@pytest.mark.parametrize(
    "mutation",
    (
        "dual_ffi_client",
        "ffi_client_request",
        "apm_get_address",
        "proto_request",
        "proto_request_init",
        "proto_parser",
        "proto_simple_namespace",
    ),
)
def test_run_guard_cannot_mutate_bound_apm_execution_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_livekit_execution_surface,
    mutation: str,
) -> None:
    del clean_livekit_execution_surface
    bundle = _install_snapshot(tmp_path, monkeypatch)
    _install_fake_livekit_distribution(tmp_path, monkeypatch)
    closure = subject._execution_closure()
    configuration = subject._configuration_snapshot()
    runtime = subject._runtime_binding(injected=False)
    apm_module = sys.modules["livekit.rtc.apm"]
    ffi_module = sys.modules["livekit.rtc._ffi_client"]
    proto_module = sys.modules["livekit.rtc._proto.ffi_pb2"]
    assert runtime.library_state.client is ffi_module.FfiClient._instance
    assert apm_module.AudioProcessingModule.construction_count == 0
    guard_calls = 0

    def guard() -> None:
        nonlocal guard_calls
        guard_calls += 1
        if mutation == "dual_ffi_client":
            original = ffi_module.FfiClient
            replacement = type(
                "FfiClient",
                (),
                {
                    "__module__": "livekit.rtc._ffi_client",
                    "_instance": original._instance,
                    "instance": vars(original)["instance"],
                    "__init__": vars(original)["__init__"],
                    "queue": vars(original)["queue"],
                    "request": vars(original)["request"],
                },
            )
            ffi_module.FfiClient = replacement
            apm_module.FfiClient = replacement
        elif mutation == "ffi_client_request":

            def replacement_request(self, request):
                del self, request
                return None

            replacement_request.__module__ = "livekit.rtc._ffi_client"
            ffi_module.FfiClient.request = replacement_request
        elif mutation == "apm_get_address":

            def replacement_address(_data):
                return 0

            replacement_address.__module__ = "livekit.rtc._utils"
            apm_module.get_address = replacement_address
        elif mutation == "proto_request":
            proto_module.FfiRequest = type("FfiRequest", (), {})
        elif mutation == "proto_request_init":

            def replacement_init(self):
                self.replaced = True

            replacement_init.__module__ = "livekit.rtc._proto.ffi_pb2"
            proto_module.FfiRequest.__init__ = replacement_init
        elif mutation == "proto_parser":

            def replacement_parser(self, raw):
                del self, raw
                return -1

            replacement_parser.__module__ = "livekit.rtc._proto.ffi_pb2"
            proto_module.FfiResponse.ParseFromString = replacement_parser
        else:
            proto_module.SimpleNamespace = type("ReplacementNamespace", (), {})

    with pytest.raises(subject.MicrosoftAecApmDtdEvalError):
        subject._verify_bound_state(
            bundle,
            closure,
            configuration,
            runtime,
            injected=False,
            run_guard=guard,
        )
    assert guard_calls == 1
    assert apm_module.AudioProcessingModule.construction_count == 0


def test_proto_direct_globals_remain_stable_on_normal_rebind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_livekit_execution_surface,
) -> None:
    del clean_livekit_execution_surface
    _install_fake_livekit_distribution(tmp_path, monkeypatch)
    snapshot = subject._runtime_binding(injected=False)
    proto_module = sys.modules["livekit.rtc._proto.ffi_pb2"]
    namespace = proto_module.SimpleNamespace

    request = proto_module.FfiRequest()
    response = proto_module.FfiResponse()
    response.ParseFromString(request.SerializeToString())
    sys.modules["livekit.rtc"].AudioProcessingModule()
    rebound = subject._runtime_binding(injected=False, retained=snapshot)

    assert rebound == snapshot
    assert rebound.execution_identity == snapshot.execution_identity
    assert proto_module.SimpleNamespace is namespace


def test_eager_native_initialization_preserves_lifecycle_and_has_no_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_livekit_execution_surface,
) -> None:
    del clean_livekit_execution_surface
    _install_fake_livekit_distribution(tmp_path, monkeypatch)
    bounded_read = subject.read_regular_bounded
    signal_number = int(getattr(subject.signal, "SIGTERM", 15))

    def interrupt_after_initialization(path, *args, **kwargs):
        ffi_module = sys.modules.get("livekit.rtc._ffi_client")
        initialized = (
            ffi_module is not None
            and vars(ffi_module.FfiClient).get("_instance") is not None
        )
        if Path(path).name in subject._LIVEKIT_NATIVE_NAMES and initialized:
            raise subject._LifecycleSignal(signal_number)
        return bounded_read(path, *args, **kwargs)

    monkeypatch.setattr(subject, "read_regular_bounded", interrupt_after_initialization)
    with pytest.raises(subject._LifecycleSignal) as raised:
        subject._runtime_binding(injected=False)
    assert raised.value.signum == signal_number
    assert subject._loaded_livekit_names()
    ffi_module = sys.modules["livekit.rtc._ffi_client"]
    assert ffi_module.FfiClient._instance is not None
    assert sys.modules["livekit.rtc.apm"].AudioProcessingModule.construction_count == 0


@pytest.mark.parametrize(
    "missing_error",
    (
        subject.importlib_metadata.PackageNotFoundError("livekit"),
        ImportError("synthetic missing livekit"),
    ),
)
def test_missing_livekit_is_normalized_detail_free(
    monkeypatch: pytest.MonkeyPatch,
    clean_livekit_execution_surface,
    missing_error: BaseException,
) -> None:
    del clean_livekit_execution_surface
    assert subject._loaded_livekit_names() == ()

    def missing(_name):
        raise missing_error

    monkeypatch.setattr(subject.importlib_metadata, "distribution", missing)
    with pytest.raises(subject.MicrosoftAecApmDtdEvalError) as raised:
        subject._runtime_binding(injected=False)
    assert raised.value.args == ()


@pytest.mark.parametrize(
    "attack",
    (
        "preloaded",
        "namespace_path",
        "pyc",
        "module_replacement",
        "spec_origin",
        "export_replacement",
        "resource_path",
        "cdll_name",
        "ffi_client_instance",
        "native_request",
    ),
)
def test_runtime_rejects_unbound_or_interposed_execution_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_livekit_execution_surface,
    attack: str,
) -> None:
    del clean_livekit_execution_surface
    root, _members = _install_fake_livekit_distribution(tmp_path, monkeypatch)
    retained = None
    if attack == "preloaded":
        sys.modules["livekit"] = types.ModuleType("livekit")
    elif attack == "namespace_path":
        interposed = tmp_path / "interposed"
        (interposed / "livekit").mkdir(parents=True)
        monkeypatch.setattr(subject.sys, "path", [str(interposed), *sys.path])
        subject.importlib.invalidate_caches()
    elif attack == "pyc":
        cache = root / "livekit/rtc/__pycache__"
        cache.mkdir()
        (cache / "apm.cpython-312.pyc").write_bytes(b"unbound bytecode")
    else:
        retained = subject._runtime_binding(injected=False)
        if attack == "module_replacement":
            sys.modules["livekit.rtc.apm"] = types.ModuleType("livekit.rtc.apm")
        elif attack == "spec_origin":
            module = sys.modules["livekit.rtc.apm"]
            module.__spec__.origin = str(tmp_path / "interposed-apm.py")
        elif attack == "resource_path":
            resource_module = sys.modules["livekit.rtc.resources"]
            resource_module.__path__ = [str(tmp_path / "interposed-resources")]
        elif attack in {"cdll_name", "ffi_client_instance", "native_request"}:
            sys.modules["livekit.rtc"].AudioProcessingModule()
            subject._runtime_binding(injected=False, retained=retained)
            if attack == "cdll_name":
                interposed = tmp_path / "interposed-native.so"
                interposed.write_bytes(b"interposed")
                retained.library_state.cdll._name = str(interposed)
            elif attack == "ffi_client_instance":
                ffi_client = sys.modules["livekit.rtc._ffi_client"].FfiClient
                replacement = object.__new__(ffi_client)
                replacement._ffi_lib = retained.library_state.cdll
                ffi_client._instance = replacement
            else:
                retained.library_state.cdll.livekit_ffi_request = lambda *_args: 0
        else:
            rtc = sys.modules["livekit.rtc"]
            apm = sys.modules["livekit.rtc.apm"]
            original = apm.AudioProcessingModule
            replacement = type(
                "AudioProcessingModule",
                (),
                {
                    "__module__": "livekit.rtc.apm",
                    **{
                        name: vars(original)[name]
                        for name in (
                            "__init__",
                            "process_stream",
                            "process_reverse_stream",
                            "set_stream_delay_ms",
                        )
                    },
                },
            )
            rtc.AudioProcessingModule = replacement
            apm.AudioProcessingModule = replacement

    with pytest.raises(subject.MicrosoftAecApmDtdEvalError):
        subject._runtime_binding(injected=False, retained=retained)


def test_current_unpinned_livekit_cannot_be_production_runtime() -> None:
    try:
        installed = subject.importlib_metadata.version("livekit")
    except subject.importlib_metadata.PackageNotFoundError:
        installed = None
    required = subject._required_livekit_version()
    if installed == required:
        snapshot = subject._runtime_binding(injected=False)
        assert snapshot.binding["production_evidence"] is True
    else:
        with pytest.raises(subject.MicrosoftAecApmDtdEvalError):
            subject._runtime_binding(injected=False)


@pytest.mark.skipif(
    importlib.util.find_spec("livekit") is None,
    reason="livekit is not installed",
)
def test_tiny_installed_livekit_apm_synthetic_smoke_is_non_production() -> None:
    from core.engines._apm import _WebRTCAPM

    assert (
        subject.importlib_metadata.version("livekit")
        != subject._required_livekit_version()
    )
    processor = _WebRTCAPM(
        echo_cancellation=True,
        noise_suppression=False,
        high_pass_filter=False,
        gain_control=False,
    )
    near = np.zeros(subject.BLOCK_SAMPLES, dtype=np.float32)
    far = np.zeros_like(near)

    output = processor.process(near, far)

    assert output.shape == near.shape
    assert output.dtype == np.float32
    assert np.all(np.isfinite(output))
