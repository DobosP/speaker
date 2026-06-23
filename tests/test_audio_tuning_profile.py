"""audio-bargein-1: the barge/audio front-end params are per-device-profile
tunable, and the shipped defaults are byte-identical (no behaviour change until a
profile actually overrides one).

Before this, block_sec (0.1) and the coherence FFT nperseg (256) were hard-coded
literals, so a phone ran the identical DSP budget as a 4090. They are now
SherpaConfig fields alongside the ones that were already configurable
(aec_filter_taps, dtd_k, coherence_ring_ms), so device_profiles[*].sherpa can
retune them. This pins the mechanism; the per-profile *values* are tuned later
against a live open-speaker A/B.

Tier 0: dataclass defaults + the pure config merge. No audio, no models.
"""
from __future__ import annotations

from core.config import apply_device_profile
from core.engines.echo_coherence import EchoCoherenceDetector
from core.engines.sherpa import SherpaConfig


def test_audio_tuning_defaults_are_unchanged():
    """The shipped defaults must not move -- promoting a literal to a config field
    is only safe if the default equals the old literal (byte-identical behaviour)."""
    c = SherpaConfig()
    assert c.block_sec == 0.1            # was the hard-coded capture block size
    assert c.coherence_nperseg == 256    # was the detector's built-in default
    # The ones that were already fields (pinned so the audio-tuning surface is
    # documented in one place):
    assert c.aec_filter_taps == 512
    assert c.dtd_k == 5.0
    assert c.coherence_ring_ms == 600.0
    assert c.coherence_max_delay_ms == 400.0


def test_sherpa_config_carries_overrides_to_the_engine():
    c = SherpaConfig(block_sec=0.05, coherence_nperseg=128, aec_filter_taps=256, dtd_k=4.0)
    assert c.block_sec == 0.05
    assert c.coherence_nperseg == 128
    assert c.aec_filter_taps == 256
    assert c.dtd_k == 4.0


def test_audio_params_are_device_profile_overridable():
    """A device_profile's sherpa block can now retune the audio front-end (not
    just thread counts) -- the deep-merge carries it through to the sherpa config
    SherpaConfig is built from."""
    config = {
        "sherpa": {"sample_rate": 16000, "block_sec": 0.1, "coherence_nperseg": 256},
        "device_profiles": {
            "phone_lite": {
                "sherpa": {"block_sec": 0.05, "coherence_nperseg": 128, "dtd_k": 4.5}
            }
        },
    }
    merged = apply_device_profile(config, "phone_lite")["sherpa"]
    assert merged["block_sec"] == 0.05          # profile override landed
    assert merged["coherence_nperseg"] == 128
    assert merged["dtd_k"] == 4.5
    assert merged["sample_rate"] == 16000       # base sibling survived the merge


def test_coherence_detector_honours_configured_nperseg():
    """The wiring target: the detector actually uses the nperseg it's given."""
    det = EchoCoherenceDetector(16000, nperseg=128)
    assert det.nperseg == 128


def test_playback_fifo_default_and_capable_profile_override():
    """Playback FIFO depth: 1.0 default (unchanged), capable profiles raise it to
    1.5 for whole-clip-synth headroom under load; phone profiles keep 1.0."""
    import json

    assert SherpaConfig().playback_fifo_sec == 1.0  # default unmoved
    config = json.load(open("config.json"))
    assert config["sherpa"]["playback_fifo_sec"] == 1.0  # base unchanged
    for profile in ("desktop", "desktop_gpu_4090", "macbook_m_series"):
        merged = apply_device_profile(config, profile)["sherpa"]
        assert merged["playback_fifo_sec"] == 1.5, profile
    # Weak profiles inherit the lean 1.0 base (no override).
    for profile in ("phone", "phone_lite"):
        merged = apply_device_profile(config, profile)["sherpa"]
        assert merged.get("playback_fifo_sec", 1.0) == 1.0, profile
