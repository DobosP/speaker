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
