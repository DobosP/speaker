"""Playback-reference recording (record_playback_reference): the .ref.wav stays
FRAME-ALIGNED with the mic WAV so an open-speaker barge run replays faithfully.
Pure -- exercises the accumulator/frame helpers directly, no audio device."""
from __future__ import annotations

import numpy as np

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
    assert SherpaOnnxEngine(SherpaConfig())._ref_recorder is None   # off until start()+flag


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
