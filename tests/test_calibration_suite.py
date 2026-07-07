"""Unit tests for the mic calibration recording suite (tools/calibration_suite).

No microphone, no models: exercises the pure logic (WER, preset overrides,
ranking) and the capture front-end + WAV round-trip over a synthetic signal, so
the suite stays a CI gate.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools import calibration_suite as cs


def test_word_error_rate_basic():
    assert cs.word_error_rate("the quick brown fox", "the quick brown fox") == 0.0
    # one substitution out of four words
    assert cs.word_error_rate("the quick brown fox", "the quick brown dog") == 0.25
    # punctuation + casing are normalized away
    assert cs.word_error_rate("Hello, world!", "hello world") == 0.0
    # empty reference -> undefined
    assert cs.word_error_rate("", "anything") is None


def test_word_error_rate_insertions_and_deletions():
    assert cs.word_error_rate("a b c", "a c") == 0.333      # deletion (rounded to 3dp)
    assert cs.word_error_rate("a b c", "a b b c") == 0.333  # insertion


def test_preset_config_overrides_minimal():
    raw = cs.CalibrationPreset(name="raw", blurb="")
    assert raw.config_overrides() == {}  # baseline touches nothing

    agc = cs.CalibrationPreset(name="x", blurb="", input_agc=True, input_calibrate=True)
    assert agc.config_overrides() == {"input_agc": True, "input_calibrate": True}

    vc = cs.CalibrationPreset(name="x", blurb="", capture_voice_comm=True, input_gain=4.0)
    ov = vc.config_overrides()
    assert ov["capture_voice_comm"] is True and ov["input_gain"] == 4.0


def test_default_presets_are_distinct_and_five():
    names = [p.name for p in cs.DEFAULT_PRESETS]
    assert len(names) == 5
    assert len(set(names)) == 5
    # Round-2 defaults: raw control + the Teams-style DSP candidates.
    assert names == ["raw", "voice_comm", "denoise", "apm", "voice_comm_denoise"]
    # Round-1 presets stay reachable via --presets.
    all_names = {p.name for p in cs.ALL_PRESETS}
    assert {"agc_calibrated", "gain_boost", "voice_comm_agc"} <= all_names


def test_apm_and_denoise_config_overrides():
    apm = next(p for p in cs.ALL_PRESETS if p.name == "apm")
    ov = apm.config_overrides()
    assert ov == {"aec_enabled": True, "aec_backend": "apm",
                  "apm_always_on": True, "apm_gain_control": True}

    den = next(p for p in cs.ALL_PRESETS if p.name == "denoise")
    assert den.config_overrides() == {"denoise_enabled": True}

    combo = next(p for p in cs.ALL_PRESETS if p.name == "voice_comm_denoise")
    assert combo.config_overrides() == {"capture_voice_comm": True, "denoise_enabled": True}


def test_select_presets_subset_and_unknown():
    chosen = cs._select_presets("raw,voice_comm")
    assert [p.name for p in chosen] == ["raw", "voice_comm"]
    assert len(cs._select_presets("")) == 5  # empty -> the default sweep
    # round-1 presets remain selectable even though they left the default sweep
    assert [p.name for p in cs._select_presets("gain_boost")] == ["gain_boost"]
    with pytest.raises(SystemExit):
        cs._select_presets("nope")


def test_frontend_denoise_missing_model_fails_open():
    # No denoise_model in the cfg -> build_denoiser returns None -> the stage is
    # skipped, a warning is recorded, and processing is a passthrough (raw).
    preset = cs.CalibrationPreset(name="d", blurb="", denoise=True)
    fe = cs.CaptureFrontEnd(preset, capture_sr=16000, sherpa_cfg={})
    assert fe._denoiser is None
    assert any("denoise" in w.lower() or "gtcrn" in w.lower() for w in fe.warnings)
    sig = (0.1 * np.sin(2 * np.pi * 300 * np.arange(1600) / 16000)).astype("float32")
    out = fe.process(sig)
    assert np.allclose(out, sig)  # byte-identical passthrough


def test_frontend_apm_fails_open_or_processes():
    # With livekit installed the APM engages (output differs / may be shorter due
    # to 10 ms framing warm-up); without it the build fails open to passthrough
    # with a warning. Both are correct -- the sweep must never crash.
    preset = cs.CalibrationPreset(name="a", blurb="", apm=True)
    fe = cs.CaptureFrontEnd(preset, capture_sr=16000, sherpa_cfg={})
    sig = (0.1 * np.sin(2 * np.pi * 300 * np.arange(3200) / 16000)).astype("float32")
    out = fe.process(sig)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    if fe._aec is None:
        assert fe.warnings  # fail-open must be loud
        assert np.allclose(out, sig)


def test_looped_chunk_wraps_and_handles_empty():
    data = np.arange(5, dtype="float32")
    chunk, pos = cs._looped_chunk(data, 3, 4)  # 3,4 -> wrap -> 0,1
    assert chunk.tolist() == [3.0, 4.0, 0.0, 1.0]
    assert pos == 2
    # longer than the clip: wraps repeatedly
    chunk, pos = cs._looped_chunk(data, 0, 12)
    assert chunk.tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]
    assert pos == 2
    # empty reference -> silence, position pinned at 0
    chunk, pos = cs._looped_chunk(np.zeros(0, "float32"), 0, 8)
    assert chunk.size == 8 and not chunk.any() and pos == 0


def test_frontend_gain_boost_raises_level_without_hard_clip():
    quiet = (0.02 * np.sin(2 * np.pi * 300 * np.arange(16000) / 16000)).astype("float32")
    preset = cs.CalibrationPreset(name="g", blurb="", input_gain=4.0)
    fe = cs.CaptureFrontEnd(preset, capture_sr=16000)
    out = np.concatenate([fe.process(quiet[i:i + 1600]) for i in range(0, quiet.size, 1600)])
    # louder than the input, but the soft-knee keeps it inside [-1, 1]
    assert cs._est_snr_db is not None
    assert float(np.sqrt(np.mean(out ** 2))) > float(np.sqrt(np.mean(quiet ** 2)))
    assert float(np.max(np.abs(out))) <= 1.0


def test_frontend_resamples_to_16k():
    preset = cs.CalibrationPreset(name="raw", blurb="")
    fe = cs.CaptureFrontEnd(preset, capture_sr=48000)  # 48k -> 16k = /3
    block = np.zeros(4800, dtype="float32")  # 0.1 s at 48 kHz
    out = fe.process(block)
    # ~1/3 the samples (allow FIR warm-up slack on the first block)
    assert 1200 <= out.size <= 1800


def test_wav_round_trip(tmp_path):
    sig = (0.3 * np.sin(2 * np.pi * 440 * np.arange(8000) / 16000)).astype("float32")
    p = tmp_path / "clip.wav"
    cs.write_wav(str(p), sig, 16000)
    import wave

    w = wave.open(str(p), "rb")
    assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
    back = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype("float32") / 32768.0
    w.close()
    assert back.size == sig.size
    assert np.max(np.abs(back - sig)) < 1e-3  # int16 quantization only


def test_ranking_prefers_lower_wer():
    def mk(name, wer, rms=0.12, clip=0.0):
        p = cs.CalibrationPreset(name=name, blurb="")
        return cs.RecordResult(preset=p, samples=np.zeros(0, "float32"), capture_sr=16000,
                               calibration=None, metrics={"rms": rms, "clip_pct": clip}, wer=wer)

    ranked = cs.rank_results([mk("bad", 0.5), mk("good", 0.1), mk("mid", 0.3)])
    assert [r.preset.name for r in ranked] == ["good", "mid", "bad"]


def test_ranking_none_wer_sorts_last():
    def mk(name, wer):
        p = cs.CalibrationPreset(name=name, blurb="")
        return cs.RecordResult(preset=p, samples=np.zeros(0, "float32"), capture_sr=16000,
                               calibration=None, metrics={"rms": 0.12, "clip_pct": 0.0}, wer=wer)

    ranked = cs.rank_results([mk("unscored", None), mk("scored", 0.4)])
    assert ranked[0].preset.name == "scored"


def test_ranking_uses_confidence_when_phrase_not_read():
    # Free speech: every WER ~1.0 -> rank by ASR confidence (avg_logprob, closer
    # to 0 = cleaner), NOT by the meaningless WER.
    def mk(name, conf, wer=1.0):
        p = cs.CalibrationPreset(name=name, blurb="")
        return cs.RecordResult(preset=p, samples=np.zeros(0, "float32"), capture_sr=16000,
                               calibration=None, metrics={"rms": 0.1, "clip_pct": 0.0},
                               wer=wer, asr_confidence=conf)

    results = [mk("noisy", -0.9), mk("clean", -0.2), mk("mid", -0.5)]
    assert not cs.phrase_was_read(results)
    assert [r.preset.name for r in cs.rank_results(results)] == ["clean", "mid", "noisy"]


def test_phrase_was_read_switches_metric():
    def mk(name, conf, wer):
        p = cs.CalibrationPreset(name=name, blurb="")
        return cs.RecordResult(preset=p, samples=np.zeros(0, "float32"), capture_sr=16000,
                               calibration=None, metrics={"rms": 0.1, "clip_pct": 0.0},
                               wer=wer, asr_confidence=conf)

    # One clip closely matches the phrase -> WER is trusted and wins over confidence.
    results = [mk("read_low_wer", -0.9, 0.05), mk("free_high_conf", -0.1, 1.0)]
    assert cs.phrase_was_read(results)
    assert cs.rank_results(results)[0].preset.name == "read_low_wer"


def test_metrics_and_synthetic_capture_shapes():
    sig = cs._synthetic_capture(1.0, 16000)
    assert sig.dtype == np.float32 and sig.size == 16000
    m = cs._audio_metrics(sig, 16000, None)
    assert "rms" in m and "clip_pct" in m and m["duration_s"] == 1.0


def test_loudness_normalize_matches_quiet_and_loud_to_same_level():
    tone = np.sin(2 * np.pi * 300 * np.arange(16000) / 16000).astype("float32")
    quiet = 0.01 * tone
    loud = 0.30 * tone

    def voiced_ref(x):
        w = 320
        n = (x.size // w) * w
        e = np.sqrt((x[:n].reshape(-1, w).astype("float64") ** 2).mean(axis=1))
        return float(np.percentile(e[e > 1e-6], 90))

    nq = cs.loudness_normalize(quiet, 0.12)
    nl = cs.loudness_normalize(loud, 0.12)
    # both land near the same voiced level despite a 30x input gap
    assert voiced_ref(nq) == pytest.approx(0.12, rel=0.15)
    assert voiced_ref(nl) == pytest.approx(0.12, rel=0.15)
    assert float(np.max(np.abs(nq))) <= 1.0 and float(np.max(np.abs(nl))) <= 1.0


def test_loudness_normalize_silence_is_safe():
    assert cs.loudness_normalize(np.zeros(16000, "float32"), 0.12).size == 16000
