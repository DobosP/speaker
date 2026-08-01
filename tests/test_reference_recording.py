"""Playback-reference recording (record_playback_reference): the .ref.wav stays
FRAME-ALIGNED with the mic WAV so an open-speaker barge run replays faithfully.
Pure -- exercises the accumulator/frame helpers directly, no audio device."""
from __future__ import annotations

import json
import wave

import numpy as np
import pytest

from core.diagnostic_bundle import DiagnosticTrack, validate_manifest
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from core.media_session import CapturedBlock


class _RecRec:
    """Capture what the reference recorder is handed, per frame."""

    def __init__(self):
        self.frames: list = []
        self.path = "x.ref.wav"

    def write(self, s) -> None:
        self.frames.append(np.asarray(s, dtype="float32").reshape(-1).copy())


def _eng() -> SherpaOnnxEngine:
    eng = SherpaOnnxEngine(SherpaConfig())   # sample_rate 16000
    eng._ref_recorder = _RecRec()
    eng._ref_accum = np.zeros(0, dtype="float32")
    return eng


def test_silence_when_nothing_played():
    eng = _eng()
    eng._write_reference_frame(1600)
    f = eng._ref_recorder.frames[-1]
    assert f.shape[0] == 1600 and np.all(f == 0.0)   # idle -> silence, still aligned


def test_played_block_then_aligned_pop():
    eng = _eng()
    eng._accumulate_reference(np.ones(1600, dtype="float32"), 16000)
    eng._write_reference_frame(1600)
    f = eng._ref_recorder.frames[-1]
    assert f.shape[0] == 1600 and np.allclose(f, 1.0)


def test_partial_play_is_silence_padded_to_frame():
    eng = _eng()
    eng._accumulate_reference(np.ones(1000, dtype="float32"), 16000)
    eng._write_reference_frame(1600)
    f = eng._ref_recorder.frames[-1]
    assert f.shape[0] == 1600
    assert np.allclose(f[:1000], 1.0) and np.all(f[1000:] == 0.0)


def test_play_rate_is_resampled_to_16k():
    eng = _eng()
    eng._accumulate_reference(np.ones(2205, dtype="float32"), 22050)   # ~0.1s @ 22050
    assert abs(eng._ref_accum.shape[0] - 1600) <= 4                    # -> ~1600 @ 16k


def test_alignment_holds_across_frames():
    # Every mic frame must produce exactly one ref frame of the same length.
    eng = _eng()
    for i in range(10):
        if i in (3, 4, 5):                       # "speaking" for 3 frames
            eng._accumulate_reference(np.full(1600, 0.5, dtype="float32"), 16000)
        eng._write_reference_frame(1600)
    assert len(eng._ref_recorder.frames) == 10
    assert all(f.shape[0] == 1600 for f in eng._ref_recorder.frames)


def test_config_field_parses_and_defaults_off():
    assert SherpaConfig().record_playback_reference is False
    assert SherpaConfig.from_dict({"record_playback_reference": True}).record_playback_reference is True
    assert SherpaConfig().record_pre_dsp_reference is False
    assert (
        SherpaConfig.from_dict({"record_pre_dsp_reference": True})
        .record_pre_dsp_reference
        is True
    )
    assert SherpaOnnxEngine(SherpaConfig())._ref_recorder is None   # off until start()+flag


def test_pre_dsp_post_and_playback_frames_share_one_coordinate():
    eng = SherpaOnnxEngine(SherpaConfig())
    eng._recorder = _RecRec()
    eng._pre_dsp_recorder = _RecRec()
    eng._ref_recorder = _RecRec()
    eng._ref_accum = np.zeros(0, dtype="float32")

    eng._write_recording_frame(
        np.array([0.1, 0.2], dtype="float32"),
        np.array([0.4, 0.5, 0.6], dtype="float32"),
    )

    pre = eng._pre_dsp_recorder.frames[-1]
    post = eng._recorder.frames[-1]
    ref = eng._ref_recorder.frames[-1]
    assert pre.shape == post.shape == ref.shape == (3,)
    np.testing.assert_allclose(pre, [0.1, 0.2, 0.0])
    np.testing.assert_allclose(post, [0.4, 0.5, 0.6])
    np.testing.assert_allclose(ref, 0.0)


def test_disabled_endpoint_diagnostics_ignore_dormant_nonfinite_knobs(
    tmp_path,
):
    eng = SherpaOnnxEngine(
        SherpaConfig(
            endpoint_enabled=False,
            endpoint_min_silence_sec=float("nan"),
            record_pre_dsp_reference=True,
            record_playback_reference=True,
        )
    )
    eng.set_record_path(str(tmp_path / "dormant.wav"))

    eng._open_recorders()
    try:
        bundle = eng._diagnostic_bundle
        assert bundle is not None
        replay = bundle.endpoint_replay_config
        assert replay.enabled is False
        assert replay.min_silence_sec == 0.5
    finally:
        eng._close_recorders(log_completed=False)


def test_actual_aligned_wavs_finalize_to_equal_frame_counts(tmp_path):
    eng = SherpaOnnxEngine(
        SherpaConfig(
            record_pre_dsp_reference=True,
            record_playback_reference=True,
        )
    )
    paths = {
        "post": tmp_path / "run.wav",
        "pre": tmp_path / "run.pre-dsp.wav",
        "asr": tmp_path / "run.asr.wav",
        "ref": tmp_path / "run.ref.wav",
        "final": tmp_path / "run.final-input.f32le",
        "timeline": tmp_path / "run.timeline.jsonl",
        "manifest": tmp_path / "run.diagnostic.json",
    }
    eng.set_record_path(str(paths["post"]))
    eng._open_recorders()

    bundle = eng._diagnostic_bundle
    assert bundle is not None
    assert bundle.path_by_role[DiagnosticTrack.MODEL_GATE_PCM] == paths["post"]
    assert bundle.path_by_role[DiagnosticTrack.MODEL_PRE_GAIN_TAP] == paths["pre"]
    assert bundle.path_by_role[DiagnosticTrack.MODEL_ASR_TAP] == paths["asr"]
    assert (
        bundle.path_by_role[DiagnosticTrack.PLAYBACK_REFERENCE_READER_SNAPSHOT]
        == paths["ref"]
    )

    def captured(sequence: int, start: int, count: int) -> CapturedBlock:
        return CapturedBlock(
            pcm=np.zeros(count, dtype="float32"),
            sample_rate_hz=16000,
            sequence=sequence,
            captured_started_at=10.0 + start / 16000,
            captured_at=10.0 + (start + count) / 16000,
            capture_epoch=1,
            source_generation=1,
            capture_generation=1,
            source_sample_start=start,
            source_sample_end=start + count,
        )

    eng._write_recording_frame(
        np.full(800, 0.1, dtype="float32"),
        np.full(800, 0.2, dtype="float32"),
        asr_input_samples=np.full(800, 0.3, dtype="float32"),
        captured=captured(1, 0, 800),
    )
    eng._write_recording_frame(
        np.full(1200, -0.1, dtype="float32"),
        np.full(1200, -0.2, dtype="float32"),
        asr_input_samples=np.full(1200, -0.3, dtype="float32"),
        captured=captured(2, 800, 1200),
    )
    eng._close_recorders(log_completed=True)

    counts = []
    first_samples = []
    for key in ("post", "pre", "asr", "ref"):
        path = paths[key]
        with wave.open(str(path), "rb") as recording:
            assert recording.getframerate() == 16000
            counts.append(recording.getnframes())
            first_samples.append(
                int.from_bytes(recording.readframes(1), "little", signed=True)
            )
    assert counts == [2000, 2000, 2000, 2000]
    assert first_samples == [6553, 3276, 9830, 0]
    assert paths["final"].read_bytes() == b""
    assert paths["final"].stat().st_mode & 0o777 == 0o600
    payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert payload["final_model_input"]["file"] == paths["final"].name
    assert payload["final_model_input"]["input_count"] == 0
    assert validate_manifest(paths["manifest"])
    assert eng._recorder is None
    assert eng._pre_dsp_recorder is None
    assert eng._ref_recorder is None
    assert eng._diagnostic_bundle is None


def test_malformed_diagnostic_coordinate_never_breaks_capture_and_fails_closed(
    tmp_path,
):
    eng = SherpaOnnxEngine(
        SherpaConfig(
            record_pre_dsp_reference=True,
            record_playback_reference=True,
        )
    )
    eng.set_record_path(str(tmp_path / "bad-coordinate.wav"))
    eng._open_recorders()
    malformed = CapturedBlock(
        pcm=np.zeros(8, dtype="float32"),
        sample_rate_hz=16000,
        sequence=1,
        captured_started_at=1.0,
        captured_at=1.001,
        capture_epoch=1,
        source_generation=1,
        capture_generation=1,
        source_sample_start=0,
        source_sample_end=0,
    )

    assert (
        eng._write_recording_frame(
            np.zeros(8, dtype="float32"),
            np.zeros(8, dtype="float32"),
            asr_input_samples=np.zeros(8, dtype="float32"),
            captured=malformed,
        )
        is None
    )
    eng._close_recorders(log_completed=True)

    assert eng.diagnostic_status == {
        "status": "incomplete",
        "failure_codes": ["capture_integration_error"],
    }
    assert not validate_manifest(tmp_path / "bad-coordinate.diagnostic.json")


def test_native_rate_reference_snapshot_preserves_its_independent_cursor(
    tmp_path,
):
    eng = SherpaOnnxEngine(
        SherpaConfig(
            record_pre_dsp_reference=True,
            record_playback_reference=True,
        )
    )
    record_path = tmp_path / "native-rate.wav"
    eng.set_record_path(str(record_path))
    eng._open_recorders()
    # A stateful 48 kHz -> 16 kHz resampler may emit 1,468 samples for a native
    # 4,800-sample block even though the reader-time reference snapshot is 1,600.
    # The reference remains the unmodified 1,600-sample reader snapshot on its
    # own cursor; the 1,468-sample pre-gain tap and later DSP tracks are separate.
    output_count = 1468
    captured = CapturedBlock(
        pcm=np.zeros(4800, dtype="float32"),
        sample_rate_hz=48000,
        sequence=1,
        captured_started_at=1.0,
        captured_at=1.1,
        capture_epoch=1,
        source_generation=1,
        capture_generation=1,
        source_sample_start=0,
        source_sample_end=4800,
    )
    span = eng._write_recording_frame(
        np.zeros(output_count, dtype="float32"),
        np.zeros(output_count, dtype="float32"),
        asr_input_samples=np.zeros(output_count, dtype="float32"),
        captured_reference=np.ones(1600, dtype="float32"),
        captured=captured,
    )
    assert span is not None
    eng._close_recorders(log_completed=True)

    manifest = tmp_path / "native-rate.diagnostic.json"
    assert validate_manifest(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["provenance"]["playback_reference_preserves_reader_snapshot"] is True
    with wave.open(str(tmp_path / "native-rate.ref.wav"), "rb") as recording:
        assert recording.getnframes() == 1600


def test_sidecar_open_failure_closes_primary_recorder(monkeypatch, tmp_path):
    opened = []

    class _OwnedRecorder:
        seconds = 0.0

        def __init__(self, path):
            self.path = path
            self.closed = False

        def close(self):
            self.closed = True

    def _recorder(path, _sample_rate):
        if opened:
            raise OSError("sidecar unavailable")
        owned = _OwnedRecorder(path)
        opened.append(owned)
        return owned

    monkeypatch.setattr("core.recorder.WavRecorder", _recorder)
    eng = SherpaOnnxEngine(SherpaConfig(record_pre_dsp_reference=True))
    eng.set_record_path(str(tmp_path / "run.wav"))

    with pytest.raises(OSError, match="sidecar unavailable"):
        eng._open_recorders()

    assert opened[0].closed is True
    assert eng._recorder is None
    assert eng._pre_dsp_recorder is None


def test_far_ref_path_records_the_aec_reference():
    # AEC on: the reference recorder reads the FarEndRing (true-playback-aligned,
    # delay 0) -- the exact far-end the canceller reads -- so a replay can recover
    # the LIVE aec_ref_delay_ms.
    from core.engines._aec import FarEndRing

    eng = SherpaOnnxEngine(SherpaConfig())
    eng._ref_recorder = _RecRec()
    eng._far_ref = FarEndRing()
    eng._far_ref.push(np.full(1600, 0.3, dtype="float32"))   # 0.1s of played far-end
    eng._write_reference_frame(1600)
    f = eng._ref_recorder.frames[-1]
    assert f.shape[0] == 1600 and np.allclose(f, 0.3)        # most-recent played (delay 0)


def test_far_ref_silence_when_nothing_played():
    from core.engines._aec import FarEndRing

    eng = SherpaOnnxEngine(SherpaConfig())
    eng._ref_recorder = _RecRec()
    eng._far_ref = FarEndRing()                              # empty ring -> zeros
    eng._write_reference_frame(1600)
    f = eng._ref_recorder.frames[-1]
    assert f.shape[0] == 1600 and np.all(f == 0.0)
