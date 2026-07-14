from __future__ import annotations

import os
import wave

import numpy as np
import pytest

from core.contract import is_stop_command
from tools.autotest import audio as audio_mod
from tools.autotest import clips as clips_mod
from tools.autotest.audio import _with_lead_in, normalize_synth_injection
from tools.autotest.clips import _DEFAULT_SCRIPT, _SYNTH_SPEED_BY_ROLE


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def test_synth_injection_normalizes_dense_speech_to_target_rms():
    samples = np.tile(np.array([-0.20, -0.10, 0.10, 0.20], dtype=np.float32), 200)

    normalized = normalize_synth_injection(samples)

    assert _rms(normalized) == pytest.approx(0.12, abs=1e-6)
    assert float(np.max(np.abs(normalized))) <= 0.80
    assert not np.shares_memory(normalized, samples)


def test_synth_injection_caps_sparse_impulse_without_forcing_target_rms():
    samples = np.zeros(1000, dtype=np.float32)
    samples[0] = 1.0

    normalized = normalize_synth_injection(samples)

    assert float(np.max(np.abs(normalized))) == pytest.approx(0.80, abs=1e-6)
    assert _rms(normalized) < 0.12


def test_synth_injection_leaves_silence_unchanged():
    samples = np.zeros(128, dtype=np.float32)

    normalized = normalize_synth_injection(samples)

    np.testing.assert_array_equal(normalized, samples)


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
def test_synth_injection_rejects_non_finite_audio(bad):
    with pytest.raises(RuntimeError, match="non-finite audio"):
        normalize_synth_injection(np.array([0.0, bad], dtype=np.float32))


def test_silent_delay_command_is_canonical_and_long_enough_for_word_cut():
    assert _DEFAULT_SCRIPT["command"] == ["quiet"]
    assert _SYNTH_SPEED_BY_ROLE == {"command": 0.8}
    assert is_stop_command(_DEFAULT_SCRIPT["command"][0])
    assert is_stop_command("quiet")


def test_only_synthetic_command_uses_the_slow_validation_pace(
    tmp_path, monkeypatch
):
    rendered = []

    def _render(text, path, *, sherpa_cfg, speed):
        rendered.append((text, speed, sherpa_cfg))
        return 1.0

    monkeypatch.setattr(clips_mod.audio, "synth_to_wav", _render)
    by_role = clips_mod.synth_clips(str(tmp_path), {"sentinel": True})

    assert set(by_role) == set(_DEFAULT_SCRIPT)
    speed_by_text = {text: speed for text, speed, _cfg in rendered}
    assert speed_by_text["quiet"] == 0.8
    assert all(
        speed == 1.0
        for text, speed, _cfg in rendered
        if text != "quiet"
    )
    assert all(cfg == {"sentinel": True} for _text, _speed, cfg in rendered)


def test_autotest_synthesis_requests_deterministic_vits(tmp_path, monkeypatch):
    from core.engines import _sherpa_models

    seen = {}

    class _Tts:
        def generate(self, text, *, sid, speed):
            seen["generate"] = (text, sid, speed)
            return type(
                "Audio",
                (),
                {
                    "samples": np.full(1600, 0.1, dtype="float32"),
                    "sample_rate": 16000,
                },
            )()

    def build(config, *, deterministic_vits=False):
        seen["config"] = config
        seen["deterministic_vits"] = deterministic_vits
        return _Tts()

    monkeypatch.setattr(_sherpa_models, "build_tts", build)
    out = tmp_path / "command.wav"

    duration = audio_mod.synth_to_wav(
        "quiet", str(out), sherpa_cfg={}, speed=0.8
    )

    assert duration == pytest.approx(0.1)
    assert out.is_file()
    assert seen["deterministic_vits"] is True
    assert seen["generate"] == ("quiet", 0, 0.8)


def test_injection_padding_keeps_stream_alive_after_short_command(tmp_path):
    source = tmp_path / "command.wav"
    speech = np.full(100, 1234, dtype="<i2")
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        wav.writeframes(speech.tobytes())

    padded = _with_lead_in(
        str(source),
        lead_in_ms=100,
        trailing_silence_ms=500,
    )
    try:
        with wave.open(padded, "rb") as wav:
            actual = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        assert actual.size == 700
        np.testing.assert_array_equal(actual[:100], 0)
        np.testing.assert_array_equal(actual[100:200], speech)
        np.testing.assert_array_equal(actual[200:], 0)
    finally:
        os.remove(padded)
