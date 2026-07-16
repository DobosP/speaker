"""Playback-reference recording (record_playback_reference): the .ref.wav stays
FRAME-ALIGNED with the mic WAV so an open-speaker barge run replays faithfully.
Pure -- exercises the accumulator/frame helpers directly, no audio device."""
from __future__ import annotations

import wave

import numpy as np
import pytest

from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine


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
        "ref": tmp_path / "run.ref.wav",
    }
    eng.set_record_path(str(paths["post"]))
    eng._open_recorders()

    assert eng._recorder.path == str(paths["post"])
    assert eng._pre_dsp_recorder.path == str(paths["pre"])
    assert eng._ref_recorder.path == str(paths["ref"])

    eng._write_recording_frame(
        np.full(800, 0.1, dtype="float32"),
        np.full(800, 0.2, dtype="float32"),
    )
    eng._write_recording_frame(
        np.full(1200, -0.1, dtype="float32"),
        np.full(1200, -0.2, dtype="float32"),
    )
    eng._close_recorders(log_completed=False)

    counts = []
    for path in paths.values():
        with wave.open(str(path), "rb") as recording:
            assert recording.getframerate() == 16000
            counts.append(recording.getnframes())
    assert counts == [2000, 2000, 2000]
    assert eng._recorder is None
    assert eng._pre_dsp_recorder is None
    assert eng._ref_recorder is None


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
