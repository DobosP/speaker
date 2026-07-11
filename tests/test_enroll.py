"""Unit tests for the speaker enrollment flow (core.enroll).

Pure / injected fakes -- no microphone, no sherpa-onnx, no model files. The
recorder and the gate are both injected, so ``run_enrollment`` exercises the
real averaging + persistence + self-check logic with synthetic audio.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from core.enroll import (
    CaptureResolution,
    Enrollment,
    EnrollmentCaptureError,
    EnrollmentFrontend,
    average_embeddings,
    enroll_from_recordings,
    enrollment_matches_frontend,
    enrollment_matches_model,
    l2_normalize,
    load_enrollment,
    make_enrollment_frontend_provenance,
    run_enrollment,
    save_enrollment,
    verify_required_os_echo_route,
)
from core.engines.speaker_gate import SpeakerGate

USER = [1.0, 0.0, 0.0]


def _gate(embed):
    """A SpeakerGate whose embed() returns ``embed`` regardless of input."""
    return SpeakerGate(threshold=0.5, embed_fn=lambda samples, sr: embed)


# --- pure embedding math -----------------------------------------------------


def test_l2_normalize_unit_and_zero():
    assert l2_normalize([3.0, 0.0, 0.0]) == [1.0, 0.0, 0.0]
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]  # zero vector unchanged


def test_average_embeddings_normalizes_each_then_renormalizes():
    out = average_embeddings([[2.0, 0.0], [0.0, 5.0]])
    # Each normalized to a unit axis; mean is the diagonal, renormalized.
    assert pytest.approx(out[0], abs=1e-9) == 0.7071067811865476
    assert pytest.approx(out[1], abs=1e-9) == 0.7071067811865476


def test_average_embeddings_rejects_empty_and_mismatched():
    with pytest.raises(ValueError):
        average_embeddings([])
    with pytest.raises(ValueError):
        average_embeddings([[]])
    with pytest.raises(ValueError):
        average_embeddings([[1.0, 0.0], [1.0]])


# --- persistence -------------------------------------------------------------


def test_save_load_round_trip(tmp_path):
    path = tmp_path / "sub" / "enroll.json"  # parent dir created on save
    frontend = make_enrollment_frontend_provenance(
        {"sample_rate": 16000, "denoise_model": "/m/gtcrn.onnx"},
        input_agc=None,
        idle_apm=None,
        denoiser=object(),
        apm_owns_ns=False,
    )
    enr = Enrollment(
        model="/m/spk.onnx",
        embedding=[0.6, 0.8],
        sample_rate=16000,
        passes=3,
        frontend=frontend,
    )
    save_enrollment(str(path), enr)
    loaded = load_enrollment(str(path))
    assert loaded.model == "/m/spk.onnx"
    assert loaded.embedding == [0.6, 0.8]
    assert loaded.sample_rate == 16000
    assert loaded.passes == 3
    assert loaded.dim == 2
    assert loaded.frontend == frontend


def test_load_enrollment_rejects_missing_embedding(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"model": "/m/spk.onnx", "embedding": []}))
    with pytest.raises(ValueError):
        load_enrollment(str(path))


def test_enrollment_matches_model():
    enr = Enrollment(model="/m/spk.onnx", embedding=USER)
    assert enrollment_matches_model(enr, "/m/spk.onnx") is True
    assert enrollment_matches_model(enr, "/m/other.onnx") is False
    # Empty recorded model (hand-written / legacy) is trusted.
    assert enrollment_matches_model(Enrollment(model="", embedding=USER), "/m/x.onnx") is True


def test_frontend_fingerprint_is_stable_and_tracks_active_stages():
    cfg = {
        "sample_rate": 16000,
        "input_agc_target_rms": 0.12,
        "denoise_model": "/m/gtcrn.onnx",
    }
    first = make_enrollment_frontend_provenance(
        cfg,
        input_agc=object(),
        idle_apm=None,
        denoiser=object(),
        apm_owns_ns=False,
    )
    same = make_enrollment_frontend_provenance(
        dict(reversed(list(cfg.items()))),
        input_agc=object(),
        idle_apm=None,
        denoiser=object(),
        apm_owns_ns=False,
    )
    changed = make_enrollment_frontend_provenance(
        {**cfg, "input_agc_target_rms": 0.2},
        input_agc=object(),
        idle_apm=None,
        denoiser=object(),
        apm_owns_ns=False,
    )
    assert first == same
    assert first.fingerprint.startswith("sha256:")
    assert first.fingerprint != changed.fingerprint
    assert first.summary == "input-agc-capped -> gtcrn"
    assert first.raw_baseline is False


def test_frontend_fingerprint_tracks_resolved_capture_not_ambient_calibration():
    from core.audio_frontend import InputAGC

    agc = InputAGC(noise_floor_rms=0.011)
    cfg = {"sample_rate": 16000, "input_agc": True}
    capture = CaptureResolution(
        route="default:PipeWire:echo-cancel-source",
        capture_sample_rate=48000,
        model_sample_rate=16000,
        resampler="soxr",
        voice_comm="pipewire-echo-cancel:capture=ec-source; playback=ec-sink",
        input_agc_noise_floor_rms=0.011,
    )

    def provenance(resolution):
        return make_enrollment_frontend_provenance(
            cfg,
            input_agc=agc,
            idle_apm=None,
            denoiser=None,
            apm_owns_ns=False,
            capture=resolution,
        )

    baseline = provenance(capture)
    assert provenance(CaptureResolution(**{**capture.__dict__, "route": "other"})) != baseline
    assert provenance(CaptureResolution(**{**capture.__dict__, "resampler": "scipy"})) != baseline
    assert provenance(CaptureResolution(**{
        **capture.__dict__, "input_agc_noise_floor_rms": 0.02
    })) == baseline
    assert provenance(CaptureResolution(**{
        **capture.__dict__, "input_agc_noise_floor_rms": 0.0114
    })) == baseline


def test_pre_cap_agc_fingerprints_require_reenrollment():
    from core.audio_frontend import InputAGC

    agc = InputAGC(noise_floor_rms=0.011)
    cfg = {"sample_rate": 16000, "input_agc": True}
    capture = CaptureResolution(
        route="default:PipeWire:echo-cancel-source",
        capture_sample_rate=48000,
        model_sample_rate=16000,
        resampler="soxr",
        voice_comm="pipewire-echo-cancel:capture=ec-source; playback=ec-sink",
        input_agc_noise_floor_rms=0.011,
    )

    def provenance(resolution):
        return make_enrollment_frontend_provenance(
            cfg,
            input_agc=agc,
            idle_apm=None,
            denoiser=None,
            apm_owns_ns=False,
            capture=resolution,
        )

    active = provenance(capture)
    assert active.version == 4
    assert active.compatible_fingerprints == frozenset()
    floor_bucket = capture.descriptor()["input_agc_noise_floor_db_3"]
    shared_legacy_descriptor = {
        "capture": {
            **{
                key: value
                for key, value in capture.descriptor().items()
                if key != "input_agc_noise_floor_db_3"
            },
            "resampler_quality": "HQ",
            "block_sec": 0.1,
        },
        "gain": {
            "kind": "input_agc",
            "target_rms": 0.12,
            "max_gain": 12.0,
            "rise": 0.08,
            "fall": 0.4,
        },
        "idle_apm": {"active": False},
        "denoise": {"active": False},
    }
    for legacy_version in (2, 3):
        legacy_descriptor = {
            **shared_legacy_descriptor,
            "schema": legacy_version,
        }
        if legacy_version == 2:
            legacy_descriptor["capture"] = {
                **legacy_descriptor["capture"],
                "input_agc_noise_floor_db_3": floor_bucket,
            }
            legacy_descriptor["gain"] = {
                **legacy_descriptor["gain"],
                "noise_floor_db_3": floor_bucket,
            }
        canonical = json.dumps(
            legacy_descriptor, sort_keys=True, separators=(",", ":")
        )
        legacy_hash = (
            "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        )
        legacy_frontend = type(active)(
            version=legacy_version,
            fingerprint=legacy_hash,
            summary="input-agc",
            raw_baseline=False,
        )
        saved = Enrollment(
            model="/m/spk.onnx", embedding=USER, frontend=legacy_frontend
        )

        assert legacy_hash not in active.compatible_fingerprints
        assert enrollment_matches_frontend(saved, active) is False


def test_v4_migrates_v2_v3_only_when_input_agc_was_absent():
    cfg = {"sample_rate": 16000, "denoise_model": "/m/gtcrn.onnx"}
    active = make_enrollment_frontend_provenance(
        cfg,
        input_agc=None,
        idle_apm=None,
        denoiser=object(),
        apm_owns_ns=False,
    )
    changed = make_enrollment_frontend_provenance(
        {**cfg, "denoise_model": "/m/other.onnx"},
        input_agc=None,
        idle_apm=None,
        denoiser=object(),
        apm_owns_ns=False,
    )
    descriptor = {
        "capture": {
            "route": "unresolved:None",
            "capture_sample_rate": 16000,
            "model_sample_rate": 16000,
            "resampler": "identity",
            "voice_comm": "none",
            "resampler_quality": "HQ",
            "block_sec": 0.1,
        },
        "gain": {"kind": "static", "gain": 1.0},
        "idle_apm": {"active": False},
        "denoise": {"active": True, "model": "/m/gtcrn.onnx"},
    }

    assert active.version == 4
    assert len(active.compatible_fingerprints) == 2
    for legacy_version in (2, 3):
        legacy_descriptor = {**descriptor, "schema": legacy_version}
        canonical = json.dumps(
            legacy_descriptor, sort_keys=True, separators=(",", ":")
        )
        legacy_hash = (
            "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        )
        saved = Enrollment(
            model="/m/spk.onnx",
            embedding=USER,
            frontend=type(active)(
                version=legacy_version,
                fingerprint=legacy_hash,
                summary=active.summary,
                raw_baseline=False,
            ),
        )

        assert legacy_hash in active.compatible_fingerprints
        assert enrollment_matches_frontend(saved, active) is True
        assert enrollment_matches_frontend(saved, changed) is False


def test_default_pipewire_node_identity_distinguishes_raw_and_ec_routes():
    from core.readiness import PipeWireState

    raw = PipeWireState(
        default_source="alsa_input.raw", default_sink="alsa_output.raw"
    )
    ec = PipeWireState(
        default_source="echo-cancel-source", default_sink="echo-cancel-sink"
    )
    cfg = {"input_device": "pipewire"}
    raw_id = verify_required_os_echo_route(
        cfg, platform="linux", pipewire_probe=lambda: raw
    )
    ec_id = verify_required_os_echo_route(
        cfg, platform="linux", pipewire_probe=lambda: ec
    )
    assert raw_id != ec_id
    assert "alsa_input.raw" in raw_id
    assert "echo-cancel-source" in ec_id


def test_required_linux_os_echo_route_fails_closed_when_unverifiable():
    with pytest.raises(EnrollmentCaptureError, match="not verifiable"):
        verify_required_os_echo_route(
            {
                "barge_in_enabled": True,
                "barge_word_cut_enabled": True,
                "aec_enabled": False,
            },
            platform="linux",
            pipewire_probe=lambda: None,
        )


def test_legacy_enrollment_matches_only_raw_frontend():
    legacy = Enrollment(model="/m/spk.onnx", embedding=USER)
    raw = make_enrollment_frontend_provenance(
        {"sample_rate": 16000},
        input_agc=None,
        idle_apm=None,
        denoiser=None,
        apm_owns_ns=False,
    )
    denoised = make_enrollment_frontend_provenance(
        {"sample_rate": 16000, "denoise_model": "/m/gtcrn.onnx"},
        input_agc=None,
        idle_apm=None,
        denoiser=object(),
        apm_owns_ns=False,
    )
    gained = make_enrollment_frontend_provenance(
        {"sample_rate": 16000, "input_gain": 2.0},
        input_agc=None,
        idle_apm=None,
        denoiser=None,
        apm_owns_ns=False,
    )
    assert enrollment_matches_frontend(legacy, raw) is True
    assert enrollment_matches_frontend(legacy, denoised) is False
    assert enrollment_matches_frontend(legacy, gained) is False


def test_versioned_enrollment_requires_exact_frontend_fingerprint():
    raw = make_enrollment_frontend_provenance(
        {"sample_rate": 16000},
        input_agc=None,
        idle_apm=None,
        denoiser=None,
        apm_owns_ns=False,
    )
    denoised = make_enrollment_frontend_provenance(
        {"sample_rate": 16000, "denoise_model": "/m/gtcrn.onnx"},
        input_agc=None,
        idle_apm=None,
        denoiser=object(),
        apm_owns_ns=False,
    )
    saved = Enrollment(model="/m/spk.onnx", embedding=USER, frontend=denoised)
    assert enrollment_matches_frontend(saved, denoised) is True
    assert enrollment_matches_frontend(saved, raw) is False


# --- blockwise enrollment front end -----------------------------------------


class _FakeAGC:
    def __init__(self, scale: float = 2.0):
        self.scale = scale
        self.blocks = []

    def process(self, block):
        import numpy as np

        a = np.asarray(block, dtype="float32")
        self.blocks.append(a.copy())
        return a * self.scale


class _FakeDenoiser:
    def __init__(self, offset: float = 1.0):
        self.offset = offset
        self.blocks = []

    def process_16k(self, block):
        import numpy as np

        a = np.asarray(block, dtype="float32")
        self.blocks.append(a.copy())
        return a + self.offset


def _frontend(*, config=None, agc=None, apm=None, denoiser=None, owns_ns=False, **kw):
    cfg = {"sample_rate": 10, "block_sec": 0.2, **(config or {})}
    provenance = make_enrollment_frontend_provenance(
        cfg,
        input_agc=agc,
        idle_apm=apm,
        denoiser=denoiser,
        apm_owns_ns=owns_ns,
    )
    return EnrollmentFrontend(
        sample_rate=10,
        block_sec=0.2,
        input_agc=agc,
        idle_apm=apm,
        denoiser=denoiser,
        apm_owns_ns=owns_ns,
        provenance=provenance,
        **kw,
    )


def test_frontend_agc_precedes_static_gain_and_denoises_blockwise():
    import numpy as np

    agc = _FakeAGC(scale=2.0)
    denoiser = _FakeDenoiser(offset=1.0)
    front = _frontend(agc=agc, denoiser=denoiser, input_gain=9.0)
    captured = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype="float32")

    out = front.process(captured, capture_sample_rate=10)

    # 0.2 s at 10 Hz -> [2, 2, 1] sample blocks. AGC wins over the deliberately
    # huge static gain, then every resulting block goes through denoise.
    assert [b.size for b in agc.blocks] == [2, 2, 1]
    assert [b.size for b in denoiser.blocks] == [2, 2, 1]
    np.testing.assert_allclose(out, captured * 2.0 + 1.0)


def test_frontend_disabled_is_byte_identical():
    import numpy as np

    front = _frontend(input_gain=1.0)
    captured = np.array([0.1, -0.2, 0.3, -0.4, 0.5], dtype="float32")
    np.testing.assert_array_equal(
        front.process(captured, capture_sample_rate=10), captured
    )


def test_frontend_always_on_apm_gets_zero_far_and_owns_ns():
    import numpy as np

    class _FakeAPM:
        suppresses_noise = True

        def __init__(self):
            self.far = []

        def process_16k(self, near, far):
            self.far.append(np.asarray(far).copy())
            return np.asarray(near, dtype="float32") * 0.5

    apm = _FakeAPM()
    denoiser = _FakeDenoiser()
    front = _frontend(apm=apm, denoiser=denoiser, owns_ns=True)
    captured = np.ones(5, dtype="float32")

    out = front.process(captured, capture_sample_rate=10)

    assert [f.size for f in apm.far] == [2, 2, 1]
    assert all(np.count_nonzero(f) == 0 for f in apm.far)
    assert denoiser.blocks == []  # APM owns NS; live capture skips double-NS.
    np.testing.assert_allclose(out, captured * 0.5)


# --- enroll_from_recordings --------------------------------------------------


def test_enroll_from_recordings_averages_and_records_provenance():
    gate = _gate([3.0, 0.0, 0.0])  # un-normalized; averaging must normalize it
    enr = enroll_from_recordings(
        gate, [[0.0], [0.0]], model_path="/m/spk.onnx", sample_rate=16000
    )
    assert enr is not None
    assert enr.passes == 2
    assert enr.embedding == [1.0, 0.0, 0.0]
    assert enr.model.endswith("spk.onnx")
    assert enr.frontend is not None
    assert enr.frontend.raw_baseline is True


def test_enroll_from_recordings_returns_none_when_no_usable_embedding():
    gate = _gate(None)  # model can't embed any recording
    assert enroll_from_recordings(gate, [[0.0]], model_path="/m/spk.onnx") is None


# --- run_enrollment (CLI orchestration) --------------------------------------


def _config(tmp_path, **sherpa):
    base = {
        "speaker_embedding_model": "/m/spk.onnx",
        "speaker_enroll_embedding": str(tmp_path / "enroll.json"),
        "sample_rate": 16000,
    }
    base.update(sherpa)
    return {"sherpa": base}


def test_run_enrollment_saves_embedding_and_wires_config(tmp_path):
    config = _config(tmp_path)
    config_path = tmp_path / "config.local.json"
    msgs: list[str] = []
    code = run_enrollment(
        config,
        passes=3,
        seconds=1.0,
        config_path=str(config_path),
        recorder=lambda secs: [0.1, 0.2, 0.3],
        gate=_gate(USER),
        out=msgs.append,
    )
    assert code == 0
    # The averaged embedding landed on disk.
    enr = load_enrollment(str(tmp_path / "enroll.json"))
    assert enr.embedding == USER
    assert enr.passes == 3
    assert enr.frontend is not None
    assert enr.frontend.raw_baseline is True
    # The machine-local config now points at the model + the saved embedding.
    written = json.loads(config_path.read_text())
    assert written["sherpa"]["speaker_embedding_model"].endswith("spk.onnx")
    assert written["sherpa"]["speaker_enroll_embedding"].endswith("enroll.json")
    # Self-check reported a clean (high-similarity) reference, no warning.
    text = "\n".join(msgs)
    assert "Enrolled from 3 clip(s)" in text
    assert "WARNING" not in text


def test_run_enrollment_prints_exact_nondefault_device_selectors(tmp_path):
    messages: list[str] = []
    code = run_enrollment(
        _config(
            tmp_path,
            input_device="pipewire",
            output_device="pipewire",
        ),
        passes=1,
        seconds=1.0,
        config_path=str(tmp_path / "config.local.json"),
        recorder=lambda _secs: [0.1, 0.2, 0.3],
        gate=_gate(USER),
        out=messages.append,
    )

    assert code == 0
    assert (
        "Now run:  python -m core --engine sherpa "
        "--input-device pipewire --output-device pipewire"
    ) in "\n".join(messages)


def test_injected_raw_recorder_is_not_labeled_as_configured_denoise(tmp_path):
    frontend = _frontend(denoiser=_FakeDenoiser())
    code = run_enrollment(
        _config(tmp_path, denoise_enabled=True, denoise_model="/m/gtcrn.onnx"),
        passes=1,
        config_path=str(tmp_path / "c.json"),
        recorder=lambda _secs: [0.1, 0.2, 0.3],
        gate=_gate(USER),
        frontend=frontend,
        out=lambda _line: None,
    )

    assert code == 0
    saved = load_enrollment(str(tmp_path / "enroll.json"))
    assert saved.frontend is not None
    assert saved.frontend.raw_baseline is True
    assert saved.frontend.summary == "raw baseline"
    assert saved.frontend.fingerprint != frontend.provenance.fingerprint


def test_run_enrollment_without_model_is_actionable(tmp_path):
    config = {"sherpa": {}}  # no speaker_embedding_model
    msgs: list[str] = []
    code = run_enrollment(
        config, config_path=str(tmp_path / "c.json"),
        recorder=lambda secs: [0.0], gate=_gate(USER), out=msgs.append,
    )
    assert code == 2
    assert "setup_models" in "\n".join(msgs)


def test_run_enrollment_reports_failure_when_no_embedding(tmp_path):
    config = _config(tmp_path)
    msgs: list[str] = []
    code = run_enrollment(
        config, passes=2, config_path=str(tmp_path / "c.json"),
        recorder=lambda secs: [0.0], gate=_gate(None), out=msgs.append,
    )
    assert code == 3
    assert "Enrollment failed" in "\n".join(msgs)


def test_run_enrollment_production_recorder_receives_built_frontend(monkeypatch, tmp_path):
    import numpy as np

    provenance = make_enrollment_frontend_provenance(
        {"sample_rate": 16000},
        input_agc=None,
        idle_apm=None,
        denoiser=None,
        apm_owns_ns=False,
    )
    frontend = EnrollmentFrontend(provenance=provenance)
    seen = {}

    def _record_once(seconds, sample_rate, **kwargs):
        seen.update(seconds=seconds, sample_rate=sample_rate, **kwargs)
        kwargs["frontend"].bind_capture(
            CaptureResolution(
                route="test-production-mic",
                capture_sample_rate=sample_rate,
                model_sample_rate=sample_rate,
                resampler="identity",
            )
        )
        return np.concatenate([
            np.zeros(3200, dtype="float32"),
            np.random.default_rng(7).normal(0.0, 0.15, 9600).astype("float32"),
            np.zeros(3200, dtype="float32"),
        ])

    monkeypatch.setattr("core.enroll.record_once", _record_once)
    code = run_enrollment(
        _config(tmp_path),
        passes=1,
        seconds=1.0,
        config_path=str(tmp_path / "c.json"),
        gate=_gate(USER),
        frontend=frontend,
        out=lambda _line: None,
    )

    assert code == 0
    assert seen["frontend"] is frontend
    assert seen["sample_rate"] == 16000
    assert seen["require_voice_evidence"] is True


def test_production_enrollment_rejects_silence_even_if_embedder_returns_vector(
    monkeypatch, tmp_path
):
    import sys
    import types

    import numpy as np

    provenance = make_enrollment_frontend_provenance(
        {"sample_rate": 16000},
        input_agc=None,
        idle_apm=None,
        denoiser=None,
        apm_owns_ns=False,
    )
    frontend = EnrollmentFrontend(provenance=provenance)

    near_zero = np.concatenate([
        np.zeros(3200, dtype="float32"),
        np.full(9600, 1e-6, dtype="float32"),
        np.zeros(3200, dtype="float32"),
    ])
    fake_sd = types.SimpleNamespace(
        rec=lambda frames, **_kwargs: near_zero[:frames, None],
        wait=lambda: None,
        check_input_settings=lambda **_kwargs: None,
        query_devices=lambda *_args, **_kwargs: {
            "name": "test-production-mic",
            "hostapi": 0,
            "default_samplerate": 16000,
        },
        query_hostapis=lambda _index: {"name": "PipeWire"},
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(
        "core.enroll.verify_required_os_echo_route",
        lambda _config, **_kwargs: "none",
    )
    messages = []

    code = run_enrollment(
        _config(tmp_path),
        passes=1,
        seconds=1.0,
        config_path=str(tmp_path / "c.json"),
        gate=_gate(USER),
        frontend=frontend,
        out=messages.append,
    )

    assert code == 3
    assert not (tmp_path / "enroll.json").exists()
    assert "pre-gain voice" in "\n".join(messages)


def test_production_enrollment_rejects_noise_after_prior_clip_raised_agc_gain(
    monkeypatch, tmp_path
):
    import sys
    import types

    import numpy as np

    from core.audio_frontend import InputAGC

    agc = InputAGC(noise_floor_rms=0.004)
    config = {
        "sample_rate": 16000,
        "block_sec": 0.1,
        "input_agc": True,
        "input_agc_noise_floor_rms": 0.004,
    }
    provenance = make_enrollment_frontend_provenance(
        config,
        input_agc=agc,
        idle_apm=None,
        denoiser=None,
        apm_owns_ns=False,
    )
    frontend = EnrollmentFrontend(
        sample_rate=16000,
        block_sec=0.1,
        input_agc=agc,
        provenance=provenance,
        config=config,
    )
    rng = np.random.default_rng(11)
    clips = iter([
        rng.normal(0.0, 0.02, 64000).astype("float32"),
        rng.normal(0.0, 0.0015, 64000).astype("float32"),
    ])
    fake_sd = types.SimpleNamespace(
        rec=lambda _frames, **_kwargs: next(clips)[:, None],
        wait=lambda: None,
        check_input_settings=lambda **_kwargs: None,
        query_devices=lambda *_args, **_kwargs: {
            "name": "test-production-mic",
            "hostapi": 0,
            "default_samplerate": 16000,
        },
        query_hostapis=lambda _index: {"name": "PipeWire"},
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(
        "core.enroll.verify_required_os_echo_route",
        lambda _config, **_kwargs: "none",
    )
    messages = []

    code = run_enrollment(
        _config(tmp_path, input_agc=True),
        passes=2,
        seconds=4.0,
        config_path=str(tmp_path / "c.json"),
        gate=_gate(USER),
        frontend=frontend,
        out=messages.append,
    )

    assert agc.gain > 5.0  # pass 1 would amplify pass-2 noise above the old gate
    assert code == 3
    assert not (tmp_path / "enroll.json").exists()
    assert "pre-gain voice" in "\n".join(messages)


def test_calibrated_measured_ambient_admits_quiet_voice_below_agc_minimum(
    monkeypatch,
):
    import numpy as np

    from core.audio_frontend import InputAGC
    from core.enroll import _has_enrollment_voice_evidence, record_once

    agc = InputAGC(noise_floor_rms=0.004)
    config = {
        "sample_rate": 16000,
        "block_sec": 0.1,
        "input_agc": True,
        "input_agc_noise_floor_rms": 0.004,
    }
    frontend = EnrollmentFrontend(
        sample_rate=16000,
        block_sec=0.1,
        input_agc=agc,
        provenance=make_enrollment_frontend_provenance(
            config,
            input_agc=agc,
            idle_apm=None,
            denoiser=None,
            apm_owns_ns=False,
        ),
        config=config,
    )
    rng = np.random.default_rng(20260711)
    ambient = rng.normal(0.0, 0.0001, 16000).astype("float32")
    calibration = frontend.calibrate(ambient, 16000)

    assert calibration["ambient_rms"] == pytest.approx(0.0001, rel=0.2)
    assert agc.noise_floor_rms == 0.004  # runtime AGC clamp is unchanged
    assert frontend.measured_pre_gain_ambient_rms == calibration["ambient_rms"]

    clip = rng.normal(0.0, 0.0001, 16000).astype("float32")
    t = np.arange(9600, dtype="float64") / 16000.0
    clip[3200:12800] += (0.0012 * np.sin(2.0 * np.pi * 200.0 * t)).astype(
        "float32"
    )
    admitted, voiced_sec = _has_enrollment_voice_evidence(
        clip,
        16000,
        ambient_floor_rms=frontend.measured_pre_gain_ambient_rms,
    )
    old_admitted, _ = _has_enrollment_voice_evidence(
        clip,
        16000,
        ambient_floor_rms=agc.noise_floor_rms,
    )
    assert admitted is True and voiced_sec >= 0.35
    assert old_admitted is False

    resolution = CaptureResolution(
        route="test-production-mic",
        capture_sample_rate=16000,
        model_sample_rate=16000,
        resampler="identity",
        input_agc_noise_floor_rms=agc.noise_floor_rms,
    )
    monkeypatch.setattr(
        "core.enroll._capture_raw_once",
        lambda *_args, **_kwargs: (clip, resolution),
    )
    recorded = record_once(
        1.0,
        sample_rate=16000,
        frontend=frontend,
        os_echo_mode="none",
        require_voice_evidence=True,
    )
    assert recorded.size > 0
    assert resolution.descriptor()["input_agc_noise_floor_db_3"] == -48
    raw_ambient_resolution = CaptureResolution(
        route=resolution.route,
        capture_sample_rate=resolution.capture_sample_rate,
        model_sample_rate=resolution.model_sample_rate,
        resampler=resolution.resampler,
        input_agc_noise_floor_rms=calibration["ambient_rms"],
    )
    raw_ambient_provenance = make_enrollment_frontend_provenance(
        config,
        input_agc=agc,
        idle_apm=None,
        denoiser=None,
        apm_owns_ns=False,
        capture=raw_ambient_resolution,
    )
    # Admission still uses the raw measured ambient, while reusable capture-chain
    # provenance intentionally ignores that volatile per-run operating point.
    assert frontend.provenance == raw_ambient_provenance


def test_measured_ambient_rejects_muted_bed_and_short_loud_transient():
    import numpy as np

    from core.enroll import _has_enrollment_voice_evidence

    muted = np.zeros(16000, dtype="float32")
    steady_bed = np.full(16000, 0.0001, dtype="float32")
    transient = steady_bed.copy()
    transient[3200:6400] = 0.02  # 0.2 seconds, below the sustain minimum

    for clip, expected_voiced in (
        (muted, 0.0),
        (steady_bed, 0.0),
        (transient, 0.2),
    ):
        admitted, voiced_sec = _has_enrollment_voice_evidence(
            clip,
            16000,
            ambient_floor_rms=0.0001,
        )
        assert admitted is False
        assert voiced_sec == pytest.approx(expected_voiced)


def test_measured_near_zero_ambient_keeps_absolute_silence_guard():
    import numpy as np

    from core.enroll import _has_enrollment_voice_evidence

    admitted, voiced_sec = _has_enrollment_voice_evidence(
        np.full(16000, 5e-5, dtype="float32"),
        16000,
        ambient_floor_rms=1e-6,
    )
    assert admitted is False
    assert voiced_sec == 0.0


def test_uncalibrated_agc_keeps_configured_floor_instead_of_dynamic_fallback(
    monkeypatch,
):
    import numpy as np

    from core.audio_frontend import InputAGC
    from core.enroll import (
        EnrollmentVoiceError,
        _has_enrollment_voice_evidence,
        record_once,
    )

    agc = InputAGC(noise_floor_rms=0.004)
    frontend = EnrollmentFrontend(
        sample_rate=16000,
        input_agc=agc,
        provenance=make_enrollment_frontend_provenance(
            {"sample_rate": 16000, "input_agc": True},
            input_agc=agc,
            idle_apm=None,
            denoiser=None,
            apm_owns_ns=False,
        ),
    )
    clip = np.zeros(16000, dtype="float32")
    t = np.arange(9600, dtype="float64") / 16000.0
    clip[3200:12800] = (0.0014 * np.sin(2.0 * np.pi * 200.0 * t)).astype(
        "float32"
    )
    dynamic_admitted, _ = _has_enrollment_voice_evidence(
        clip,
        16000,
        ambient_floor_rms=None,
    )
    assert dynamic_admitted is True
    assert frontend.measured_pre_gain_ambient_rms is None

    resolution = CaptureResolution(
        route="test-production-mic",
        capture_sample_rate=16000,
        model_sample_rate=16000,
        resampler="identity",
    )
    monkeypatch.setattr(
        "core.enroll._capture_raw_once",
        lambda *_args, **_kwargs: (clip, resolution),
    )
    with pytest.raises(EnrollmentVoiceError, match="pre-gain voice"):
        record_once(
            1.0,
            sample_rate=16000,
            frontend=frontend,
            os_echo_mode="none",
            require_voice_evidence=True,
        )


def test_production_enrollment_rejects_out_of_model_band_capture_energy(
    monkeypatch, tmp_path
):
    import sys
    import types

    import numpy as np

    provenance = make_enrollment_frontend_provenance(
        {"sample_rate": 16000},
        input_agc=None,
        idle_apm=None,
        denoiser=None,
        apm_owns_ns=False,
    )
    frontend = EnrollmentFrontend(sample_rate=16000, provenance=provenance)

    def rec(frames, *, samplerate, **_kwargs):
        t = np.arange(frames, dtype="float64") / float(samplerate)
        tone = 0.02 * np.sin(2.0 * np.pi * 12000.0 * t)
        return tone.astype("float32")[:, None]

    fake_sd = types.SimpleNamespace(
        rec=rec,
        wait=lambda: None,
        check_input_settings=lambda **_kwargs: None,
        query_devices=lambda *_args, **_kwargs: {
            "name": "48k-test-mic",
            "hostapi": 0,
            "default_samplerate": 48000,
        },
        query_hostapis=lambda _index: {"name": "PipeWire"},
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(
        "core.enroll.verify_required_os_echo_route",
        lambda _config, **_kwargs: "none",
    )
    messages = []

    code = run_enrollment(
        _config(tmp_path, capture_samplerate=48000),
        passes=1,
        seconds=1.0,
        config_path=str(tmp_path / "c.json"),
        gate=_gate(USER),
        frontend=frontend,
        out=messages.append,
    )

    assert code == 3
    assert not (tmp_path / "enroll.json").exists()
    assert "pre-gain voice" in "\n".join(messages)


# --- record_once: pin capture_samplerate (the AT2020 self-mute fix) -----------


def test_capture_fallback_reverifies_actual_route_before_labeling_ec(
    monkeypatch,
):
    import sys
    import types

    import numpy as np

    fake_sd = types.SimpleNamespace()

    def rec(frames, *, device=None, **kwargs):
        if device == "configured-ec":
            raise RuntimeError("configured route disappeared")
        return np.zeros((frames, 1), dtype="float32")

    fake_sd.rec = rec
    fake_sd.wait = lambda: None
    fake_sd.check_input_settings = lambda **kwargs: None
    fake_sd.query_devices = lambda *args, **kwargs: {
        "name": "default-input",
        "hostapi": 0,
        "default_samplerate": 48000,
    }
    fake_sd.query_hostapis = lambda _index: {"name": "PipeWire"}
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    checked = []

    def verify(config, **kwargs):
        checked.append(config.get("input_device"))
        if config.get("input_device") is None:
            raise EnrollmentCaptureError("raw fallback is not EC-routed")
        return "pipewire-echo-cancel:configured"

    monkeypatch.setattr("core.enroll.verify_required_os_echo_route", verify)
    from core.enroll import _capture_raw_once

    with pytest.raises(EnrollmentCaptureError, match="raw fallback"):
        _capture_raw_once(
            0.1,
            16000,
            device="configured-ec",
            os_echo_mode="pipewire-echo-cancel:configured",
            platform="linux",
            route_config={
                "barge_in_enabled": True,
                "barge_word_cut_enabled": True,
                "aec_enabled": False,
            },
        )
    assert checked == [None]


def test_record_once_pins_capture_samplerate_and_never_probes(monkeypatch):
    """With capture_samplerate set, record_once opens ONLY at that rate (never the
    16000 probe that self-mutes the AT2020). Regression for the enrollment bug
    where probing 16000 captured near-silence -> a garbage reference."""
    import sys
    import types

    import numpy as np

    opened_rates = []

    fake_sd = types.SimpleNamespace()
    fake_sd.PortAudioError = type("PortAudioError", (Exception,), {})

    def rec(frames, samplerate=None, channels=1, dtype="float32", device=None):
        opened_rates.append(int(samplerate))
        return np.zeros((frames, 1), dtype="float32")

    fake_sd.rec = rec
    fake_sd.wait = lambda: None
    fake_sd.query_devices = lambda *a, **k: {"default_samplerate": 48000}
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    from core.enroll import record_once

    record_once(1.0, sample_rate=16000, capture_samplerate=44100)
    assert opened_rates == [44100]  # pinned, no 16000 probe, no 48000 fallback


def test_record_once_legacy_probe_when_unpinned(monkeypatch):
    # capture_samplerate=0 keeps the old probe-then-fallback behaviour.
    import sys
    import types

    import numpy as np

    opened = []
    fake_sd = types.SimpleNamespace()
    fake_sd.PortAudioError = type("PortAudioError", (Exception,), {})

    def rec(frames, samplerate=None, channels=1, dtype="float32", device=None):
        opened.append(int(samplerate))
        if int(samplerate) == 16000:
            raise fake_sd.PortAudioError("reject")  # the AT2020 rejects 16000
        return np.zeros((frames, 1), dtype="float32")

    fake_sd.rec = rec
    fake_sd.wait = lambda: None
    fake_sd.query_devices = lambda *a, **k: {"default_samplerate": 44100}
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    from core.enroll import record_once

    record_once(1.0, sample_rate=16000, capture_samplerate=0)
    assert opened == [16000, 44100]  # probed 16000 (rejected) then fell back


def test_record_once_routes_selected_device_audio_through_injected_frontend(monkeypatch):
    import sys
    import types

    import numpy as np

    fake_sd = types.SimpleNamespace()
    fake_sd.PortAudioError = type("PortAudioError", (Exception,), {})
    captured = np.ones((16000, 1), dtype="float32")
    fake_sd.rec = lambda frames, **kwargs: captured[:frames]
    fake_sd.wait = lambda: None
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    class _Frontend:
        def __init__(self):
            self.calls = []

        def process(self, samples, capture_sample_rate):
            self.calls.append((np.asarray(samples).copy(), capture_sample_rate))
            return np.asarray(samples, dtype="float32") * 0.5

    frontend = _Frontend()
    from core.enroll import record_once

    out = record_once(1.0, sample_rate=16000, device=7, frontend=frontend)

    assert len(frontend.calls) == 1
    assert frontend.calls[0][1] == 16000
    np.testing.assert_array_equal(out, np.full(16000, 0.5, dtype="float32"))


def test_vad_trim_cuts_silent_head_and_tail():
    import numpy as np

    from core.enroll import _vad_trim
    sr = 16000
    clip = np.concatenate([
        np.zeros(sr, dtype="float32"),
        (np.random.default_rng(0).standard_normal(sr) * 0.3).astype("float32"),
        np.zeros(sr, dtype="float32"),
    ])
    trimmed = _vad_trim(clip, sr)
    assert clip.size == 3 * sr
    assert sr * 0.9 < trimmed.size < sr * 1.6   # ~the 1s voiced middle + small pad
    # all-silence stays unchanged (no voiced region to trim to)
    assert _vad_trim(np.zeros(sr, dtype="float32"), sr).size == sr
