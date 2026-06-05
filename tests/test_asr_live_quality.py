"""Tier-2 real-model ASR quality: the real recognizer replays a frozen clip and
its offline FINAL stays anchored to what the streaming partials heard.

``test_replay_recorded.py`` checks only that the final *contains* the topic word.
This file pins the stronger property the 2026-06 SenseVoice second-pass bug
violated on short clips: the offline final must NOT diverge into unrelated text
-- it carries the content the streaming partials already surfaced. It also pins
that the streaming path actually emits partials before the final.

real_model tier (real weights, no sound card): self-skips when the sherpa ASR
models aren't on disk, so it is safe to collect anywhere. Run with
``python tools/run_tests.py real_model``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.real_model, pytest.mark.recorded, pytest.mark.slow]

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


def _replay(monkeypatch):
    """Replay the frozen ocean clip through the REAL recognizer, collecting the
    streaming partials, the finals, and the engine's last_final."""
    import numpy as np

    # This test uses the machine-local sherpa models, so opt back into the
    # config.local.json overlay the hermetic default (conftest) disables.
    monkeypatch.setenv("SPEAKER_NO_LOCAL_CONFIG", "0")
    scfg = _sherpa_config_or_skip()

    from core.engine import EngineCallbacks
    from core.engines.file_replay import FileReplayEngine

    clip = np.load(FIXTURE).astype("float32")
    partials: list[str] = []
    finals: list[str] = []
    engine = FileReplayEngine(scfg)
    engine.start(
        EngineCallbacks(
            on_partial=lambda t: partials.append(t),
            on_final=lambda t: finals.append(t),
        )
    )
    try:
        engine.replay_samples(clip, 16000)
    finally:
        engine.stop()
    return partials, finals, engine.last_final


def test_recorded_clip_emits_streaming_partials_then_a_final(monkeypatch):
    partials, finals, last = _replay(monkeypatch)
    assert any(p.strip() for p in partials), (
        f"no non-empty streaming partial -- the streaming recognizer is silent: {partials!r}"
    )
    assert finals and (last or finals[-1]).strip(), f"no non-empty final; finals={finals!r}"


def test_final_stays_anchored_to_partials_no_second_pass_hallucination(monkeypatch):
    partials, finals, last = _replay(monkeypatch)
    final = (last or (finals[-1] if finals else "")).lower()
    # The topic survives the streaming->final handoff (the clip is about the ocean).
    assert "ocean" in final, f"final lost the expected content: {final!r}"
    assert any("ocean" in p.lower() for p in partials), (
        f"'ocean' never appeared in a streaming partial: {partials!r}"
    )
    # Stronger anchor: most of the final's words were already heard in some
    # partial -- the offline second pass refines, it does not hallucinate a
    # different sentence (the tiny-clip divergence bug). Lenient threshold so
    # legitimate casing/punctuation/word fixes don't flake it.
    partial_words: set[str] = set()
    for p in partials:
        partial_words.update(w.strip(".,!?").lower() for w in p.split())
    final_words = [w.strip(".,!?").lower() for w in final.split() if w.strip(".,!?").isalpha()]
    assert final_words, f"final had no alphabetic words: {final!r}"
    overlap = sum(1 for w in final_words if w in partial_words) / len(final_words)
    assert overlap >= 0.4, (
        f"final diverged from the streaming partials (word overlap {overlap:.2f} < 0.40) "
        f"-- possible second-pass hallucination. final={final!r}"
    )
