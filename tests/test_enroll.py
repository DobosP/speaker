"""Unit tests for the speaker enrollment flow (core.enroll).

Pure / injected fakes -- no microphone, no sherpa-onnx, no model files. The
recorder and the gate are both injected, so ``run_enrollment`` exercises the
real averaging + persistence + self-check logic with synthetic audio.
"""
from __future__ import annotations

import json

import pytest

from core.enroll import (
    Enrollment,
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
    assert first.summary == "input-agc -> gtcrn"
    assert first.raw_baseline is False


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
        return [0.1, 0.2, 0.3]

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


# --- record_once: pin capture_samplerate (the AT2020 self-mute fix) -----------


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
