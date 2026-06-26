"""Lock-in gate for the 2026-06-21 audio shipment (APM + fluid TTS).

The fluidity defaults and the open_speaker APM stack are the difference between
"sounds good" and "self-interrupts / choppy". Nothing pinned them, so a stray
config or dataclass edit could silently revert the whole shipment with a green
suite. These Tier-0 tests read the COMMITTED config.json + the dataclass defaults
(no audio, no models, no config.local.json) so a regression fails CI.
"""
from __future__ import annotations

import json
import os

from core.config import apply_device_profile
from core.engines.sherpa import SherpaConfig


def _committed_config() -> dict:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "config.json"), encoding="utf-8") as fh:
        return json.load(fh)


def test_shipped_fluidity_default_is_on():
    # tts_target_rms=0.0 was the "not fluid" defect; config.json is the ONLY place
    # that pins it ON (a clean clone is byte-identical except this field).
    assert _committed_config()["sherpa"]["tts_target_rms"] == 0.12


def test_declick_and_fade_dataclass_defaults_ship_on():
    # These ship implicitly via the dataclass (absent from config.json), so a
    # future dataclass edit dropping de-click / fade would otherwise go unnoticed.
    c = SherpaConfig()
    assert c.tts_declick is True
    assert c.tts_declick_threshold == 0.22
    assert c.barge_fade_ms == 4.0


def test_open_speaker_profile_keeps_the_apm_stack():
    # open_speaker is the reproducible open-speaker barge-in config; its APM keys
    # must stay ON or a clean clone loses the only AEC-capable profile.
    sh = _committed_config()["device_profiles"]["open_speaker"]["sherpa"]
    assert sh["aec_enabled"] is True
    assert sh["aec_backend"] == "apm"
    assert sh["apm_always_on"] is True
    assert sh["apm_noise_suppression"] is True
    assert sh["aec_auto_delay"] is True


def test_open_speaker_apm_survives_the_deep_merge():
    # The override must actually reach the sherpa config SherpaConfig is built from
    # (not just sit in the profile block), AND base siblings must survive.
    cfg = _committed_config()
    merged = apply_device_profile(cfg, "open_speaker")["sherpa"]
    assert merged["aec_backend"] == "apm"
    assert merged["apm_always_on"] is True
    assert merged["sample_rate"] == 16000          # base sibling survived
    assert SherpaConfig.from_dict(merged).aec_backend == "apm"


def test_base_aec_stays_off_so_clean_clones_are_byte_identical():
    # The shipment is opt-in: the BASE config must keep AEC off so an auto-selected
    # box (or a clean clone) doesn't silently run the unvalidated live capture path.
    assert _committed_config()["sherpa"]["aec_enabled"] is False


def test_target_rms_gt0_couples_wholeclip_only_on_first_sentence():
    """Backlog item (d): tts_target_rms>0 forces whole-clip synthesis on the FIRST
    sentence of a session (no prior gain established), but from the second sentence
    onward the feed-forward carry enables the streaming callback path.
    tts_target_rms=0 takes the streaming path unconditionally.

    This pins the loudness-norm -> first-sentence-whole-clip coupling so a future
    change that accidentally makes target_rms always-streaming (without the carry
    mechanism) is caught by CI."""
    import numpy as np
    from core.engines.sherpa import SherpaOnnxEngine

    sr = 16000
    t = np.arange(int(sr * 0.25)) / sr
    samples = (0.3 * np.sqrt(2) * np.sin(2 * np.pi * 220 * t)).astype("float32")

    class _Track:
        sample_rate = sr
        callback_used: bool | None = None

        def generate(self, text, sid=0, speed=1.0, callback=None):
            self.callback_used = callback is not None
            if callback is not None:
                callback(samples.copy(), 1.0)
            return type("A", (), {"samples": samples.copy(), "sample_rate": sr})()

    tts = _Track()

    # target_rms>0, no prior carry: WHOLE-CLIP on sentence 1.
    eng = SherpaOnnxEngine(SherpaConfig(tts_target_rms=0.12, tts_declick=False))
    eng._tts = tts
    eng._synthesize("sentence one", [].append)
    assert tts.callback_used is False, "first sentence must use whole-clip path"
    assert eng._tts_normalize_gain is not None, "carry must be set after first sentence"

    # Sentence 2: carry established -> STREAMING.
    tts.callback_used = None
    eng._synthesize("sentence two", [].append)
    assert tts.callback_used is True, "second sentence must use streaming path"

    # target_rms=0: always streaming, no carry involved.
    tts2 = _Track()
    eng2 = SherpaOnnxEngine(SherpaConfig(tts_target_rms=0.0, tts_declick=False))
    eng2._tts = tts2
    eng2._synthesize("any sentence", [].append)
    assert tts2.callback_used is True, "target_rms=0 must always use streaming path"
    assert eng2._tts_normalize_gain is None, "target_rms=0 path must not touch the carry"
