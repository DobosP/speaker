"""Recorded-session replay regression (P5 audio-calibration follow-up).

Freezes the clean utterance from the live self-interruption test
(docs/audio_calibration.md) as a small .npy fixture and replays it through the
REAL sherpa recognizer via FileReplayEngine -- no live mic, no sound card. This
guards the real recorded-audio ASR pipeline and the barge-in callback path the
live session exercised (run-20260529-174902: a barge-in fired right after the
reply started, at 17:52:16.543).

It does NOT (and cannot) reproduce the *acoustic* self-interruption, which needs
live speaker->mic playback; that behaviour is guarded model-free in
tests/test_barge_in_suppression.py (the calibrated 6 dB output margin).

Marked ``recorded`` + ``slow``: it needs the sherpa ASR models, so it SKIPS
cleanly where they aren't installed/configured (e.g. the logic-only CI) and
loads them (~seconds) where they are -- keeping it out of the fast dev loop.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.recorded, pytest.mark.slow, pytest.mark.real_model]

FIXTURE = Path(__file__).parent / "fixture_audio" / "recorded_ocean_utterance.npy"


def _sherpa_config_or_skip():
    pytest.importorskip("sherpa_onnx")
    from core.config import apply_device_profile, load_config
    from core.engines.sherpa import SherpaConfig

    cfg = load_config()
    cfg = apply_device_profile(cfg, cfg.get("device", "desktop"))
    scfg = SherpaConfig.from_dict(cfg.get("sherpa", {}))
    if not getattr(scfg, "asr_encoder", "") or not os.path.exists(scfg.asr_encoder):
        pytest.skip("sherpa ASR models not configured (set paths in config.local.json)")
    if not FIXTURE.exists():
        pytest.skip(f"missing replay fixture {FIXTURE}")
    return scfg


def test_recorded_utterance_replays_through_real_asr(monkeypatch):
    """Replay the frozen real-audio clip: the real recognizer transcribes it and
    a barge-in routed through the engine reaches the runtime's handler -- a
    regression on the recorded ASR + barge-in path, with no live devices."""
    import numpy as np

    # This test deliberately uses the machine-local sherpa models, so opt back
    # into the config.local.json overlay that the hermetic-test default
    # (conftest.py: SPEAKER_NO_LOCAL_CONFIG=1) disables.
    monkeypatch.setenv("SPEAKER_NO_LOCAL_CONFIG", "0")
    scfg = _sherpa_config_or_skip()

    from core.engine import EngineCallbacks
    from core.engines.file_replay import FileReplayEngine

    clip = np.load(FIXTURE).astype("float32")
    finals: list[str] = []
    barge_ins: list[int] = []

    engine = FileReplayEngine(scfg)
    engine.start(
        EngineCallbacks(
            on_final=lambda t: finals.append(t),
            on_barge_in=lambda: barge_ins.append(1),
        )
    )
    try:
        engine.replay_samples(clip, 16000)
        # A barge-in routed through the engine must reach the on_barge_in
        # handler -- the cancellation path the live session hit at 17:52:16.543.
        engine.barge_in()
    finally:
        engine.stop()

    joined = " ".join(finals).lower()
    assert "ocean" in joined, f"asr finals were: {finals!r}"
    assert "ocean" in engine.last_final.lower(), f"last_final was: {engine.last_final!r}"
    assert barge_ins, "engine.barge_in() did not reach the on_barge_in callback"
