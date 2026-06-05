"""Tier-3 live-output smoke: the REAL sherpa TTS synthesizes a sentence and it
PLAYS through the real audio device, without erroring or hanging.

This is the ONLY test that makes sound. It is double-gated so it never runs by
accident:
  1. ``@pytest.mark.live_output`` -- declared in pytest.ini.
  2. the conftest skip unless ``SPEAKER_LIVE=1`` is in the env.
A bare ``pytest``, CI's tests.yml, and the unit/fast stages therefore never play
audio. Run it on purpose with ``python tools/run_tests.py live`` (which preflights
models+audio, then sets SPEAKER_LIVE=1).

It pins the one thing fixtures cannot: real TTS weights -> the real output device
actually produce sound and the playback call RETURNS (no wedge). Richer scenario
validation (latency suites, barge-in, real_usage shutdown-hang) lives in the
``tools.live_session`` / ``tools.real_usage`` CLIs, run by hand.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live_output


def test_real_tts_plays_through_the_speakers(monkeypatch):
    sd = pytest.importorskip("sounddevice")
    pytest.importorskip("sherpa_onnx")
    import numpy as np

    # Use the machine's real model paths (the hermetic default disables them).
    monkeypatch.setenv("SPEAKER_NO_LOCAL_CONFIG", "0")
    from core.config import apply_device_profile, load_config
    from core.engines._sherpa_models import build_tts
    from core.engines.sherpa import SherpaConfig

    cfg = load_config()
    cfg = apply_device_profile(cfg, cfg.get("device", "desktop"))
    scfg = SherpaConfig.from_dict(cfg.get("sherpa", {}))
    if not getattr(scfg, "tts_model", "") or not os.path.exists(scfg.tts_model):
        pytest.skip("sherpa TTS model not configured (set paths in config.local.json)")

    tts = build_tts(scfg)
    assert tts is not None, "build_tts returned None despite a configured tts_model"

    audio = tts.generate(
        "Testing one two three. The live output tier is working.",
        sid=int(getattr(scfg, "tts_speaker_id", 0) or 0),
        speed=float(getattr(scfg, "tts_speed", 1.0) or 1.0),
    )
    samples = np.asarray(audio.samples, dtype="float32")
    sr = int(audio.sample_rate)
    assert samples.size > 0 and sr > 0, "real TTS produced no audio"

    # Actually play it on the real device and block until it finishes. A wedged
    # output device makes sd.wait() hang -> the pytest-timeout (60s) fails the
    # test instead of letting it run forever.
    sd.play(samples, sr)
    try:
        sd.wait()
    finally:
        sd.stop()
