"""Pure regressions for VAD-backed live ASR segment ownership."""
from __future__ import annotations

import numpy as np

from core.engines._asr_segment import ASRSegment
from core.engines._speech_evidence import (
    PreGainCaptureDomain,
    SpeechEvidenceDisposition,
    build_speech_evidence_profile,
)


def _segment(
    *,
    vad: bool = True,
    pre: float = 0.8,
    maximum: float = 22.0,
    evidence: bool = False,
    sample_rate: int = 100,
):
    return ASRSegment(
        sample_rate=sample_rate,
        pre_roll_sec=pre,
        max_utterance_sec=maximum,
        vad_available=vad,
        block_sec=0.1,
        speech_evidence_required=evidence,
    )


def _evidence_profile():
    sample_rate = 1000
    t = np.arange(20, dtype="float64") / sample_rate
    ambient_frame = (0.014142 * np.sin(2 * np.pi * 350 * t)).astype("float32")
    ambient = np.tile(ambient_frame, 10)
    profile = build_speech_evidence_profile(
        {"ambient_rms": 0.01, "clipping_fraction": 0.0},
        [ambient],
        domain=PreGainCaptureDomain(
            route="test",
            capture_sample_rate=sample_rate,
            model_sample_rate=sample_rate,
            resampler="identity",
            voice_comm="none",
        ),
        calibration_generation=3,
        sample_rate=sample_rate,
    )
    assert profile is not None
    return profile


def _evidence_voice(frames: int):
    sample_rate = 1000
    t = np.arange(20, dtype="float64") / sample_rate
    frame = (
        0.08 * np.sin(2 * np.pi * 100 * t)
        + 0.04 * np.sin(2 * np.pi * 200 * t + 0.2)
    ).astype("float32")
    return np.tile(frame, frames)


def test_idle_audio_is_bounded_to_preroll_then_complete_speech_is_owned():
    s = _segment()
    idle = np.zeros(10, dtype="float32")
    for i in range(90):  # 9 s idle used to make a short final look ~10 s long
        s.observe_vad(False, i * 0.1)
        s.append(idle)
    assert s.samples == 80  # only 0.8 s model lookback survives

    speech = np.full(10, 0.5, dtype="float32")
    for i in range(4):
        s.observe_vad(True, 9.0 + i * 0.1)
        s.append(speech)
    for i in range(3):
        s.observe_vad(False, 9.4 + i * 0.1)
        s.append(idle)

    primary, _ = s.arrays()
    assert primary.size == 80 + 40 + 30
    assert np.count_nonzero(primary == 0.5) == 40
    assert s.final_admitted is True
    assert 0.39 <= s.speech_duration_sec <= 0.41


def test_fifteen_second_utterance_keeps_its_head_under_rule3_sized_bound():
    s = _segment(maximum=22.4)
    head = np.full(10, 0.91, dtype="float32")
    s.observe_vad(True, 0.0)
    s.append(head)
    for i in range(149):
        s.observe_vad(True, 0.1 + i * 0.1)
        s.append(np.full(10, 0.2, dtype="float32"))
    primary, _ = s.arrays()
    assert primary.size == 1500
    np.testing.assert_array_equal(primary[:10], head)


def test_configured_vad_rejects_decoder_text_when_no_speech_was_seen():
    s = _segment(vad=True)
    s.observe_vad(False, 0.0)
    s.append(np.zeros(10, dtype="float32"))
    s.observe_text(0.1)  # e.g. the live idle hallucination raw 'AND'
    assert s.final_admitted is False
    assert s.early_endpoint_allowed is False
    assert s.last_text_at is None


def test_abandoned_vad_episode_expires_only_without_current_epoch_text():
    s = _segment(vad=True)
    s.observe_vad(True, 1.0)
    s.append(np.ones(10, dtype="float32"))
    s.observe_vad(False, 1.1)

    assert not s.abandoned_without_text(1.19, quiet_limit_sec=0.2)
    assert s.abandoned_without_text(1.21, quiet_limit_sec=0.2)

    s.observe_text(1.3)
    assert not s.abandoned_without_text(9.0, quiet_limit_sec=0.2)


def test_word_cut_prefix_uses_its_own_endpoint_contract_not_abandon_reset():
    s = _segment(vad=True)
    s.prepend(
        [np.ones(10, dtype="float32")],
        speech_at=1.0,
        speech_end_at=1.0,
        offline_recovery_authorized=True,
    )
    s.observe_vad(False, 1.1)

    assert not s.abandoned_without_text(9.0, quiet_limit_sec=0.2)


def test_missing_vad_fails_open_for_final_but_disables_early_endpoint():
    s = _segment(vad=False)
    s.append(np.ones(10, dtype="float32"))
    s.observe_text(1.0)
    assert s.final_admitted is True
    assert s.early_endpoint_allowed is False


def test_missing_vad_keeps_audio_before_delayed_first_partial():
    s = _segment(vad=False, pre=0.8, maximum=3.0)
    blocks = [np.full(10, i, dtype="float32") for i in range(20)]
    for block in blocks:  # two seconds before the decoder emits any text
        s.append(block)
    s.observe_text(2.0)
    primary, _ = s.arrays()
    assert primary.size == 200
    np.testing.assert_array_equal(primary[:10], blocks[0])
    assert s.speech_duration_sec is None  # finalizer uses owned-array duration


def test_vad_activity_not_decoder_stability_controls_early_endpoint_clock():
    s = _segment(vad=True)
    s.observe_vad(True, 1.0)
    s.observe_text(1.0)
    # A stable decoder hypothesis for a whole second is still ACTIVE speech.
    s.observe_vad(True, 2.0)
    assert s.trailing_silence(2.0) == 0.0
    assert s.early_endpoint_allowed is False
    # Only a VAD quiet transition opens the semantic early-endpoint window.
    s.observe_vad(False, 2.1)
    assert 0.59 <= s.trailing_silence(2.6) <= 0.61
    assert s.early_endpoint_allowed is True


def test_pause_learning_sample_is_emitted_only_when_speech_resumes():
    s = _segment(vad=True)
    assert s.observe_vad(True, 1.0) is None
    assert s.observe_vad(False, 1.2) is None
    assert s.observe_vad(False, 1.5) is None
    pause = s.observe_vad(True, 1.7)
    assert 0.49 <= pause <= 0.51


def test_confirmed_barge_pcm_can_be_prepended_and_marks_speech():
    s = _segment(vad=True)
    s.append(np.zeros(10, dtype="float32"))
    n = s.prepend(
        [np.full(10, 0.3, dtype="float32"), np.full(10, 0.4, dtype="float32")],
        speech_at=3.0,
        speech_end_at=3.2,
    )
    primary, alternate = s.arrays()
    assert n == 20
    np.testing.assert_allclose(primary[:20], np.r_[np.full(10, 0.3), np.full(10, 0.4)])
    assert alternate is not None and alternate.size == 20
    assert s.final_admitted is True
    assert 0.29 <= s.speech_duration_sec <= 0.31


def test_segment_scopes_sustained_evidence_to_current_vad_epoch():
    s = _segment(evidence=True, sample_rate=1000)
    profile = _evidence_profile()
    energetic = _evidence_voice(6)

    s.observe_vad(False, 0.0)
    s.observe_pre_gain_model_pcm(
        energetic,
        profile,
        capture_generation=8,
    )
    assert s.speech_evidence_snapshot().disposition is (
        SpeechEvidenceDisposition.UNAVAILABLE
    )

    s.observe_vad(True, 0.1)
    s.observe_pre_gain_model_pcm(
        energetic,
        profile,
        capture_generation=8,
    )
    snapshot = s.speech_evidence_snapshot()
    assert snapshot.disposition is SpeechEvidenceDisposition.SATISFIED
    assert snapshot.capture_generation == 8


def test_segment_missing_profile_abstains_but_short_profile_rejects():
    s = _segment(evidence=True, sample_rate=1000)
    s.observe_vad(True, 0.0)
    s.observe_pre_gain_model_pcm(
        _evidence_voice(2),
        None,
        capture_generation=1,
    )
    missing = s.speech_evidence_snapshot()
    assert missing.disposition is SpeechEvidenceDisposition.UNAVAILABLE
    assert missing.admitted is True

    s.reset()
    profile = _evidence_profile()
    s.observe_vad(True, 1.0)
    s.observe_pre_gain_model_pcm(
        _evidence_voice(2),
        profile,
        capture_generation=1,
    )
    short = s.speech_evidence_snapshot()
    assert short.disposition is SpeechEvidenceDisposition.INSUFFICIENT
    assert short.admitted is False


def test_segment_reset_clears_evidence_and_word_cut_prepend_bypasses():
    s = _segment(evidence=True, sample_rate=1000)
    profile = _evidence_profile()
    s.observe_vad(True, 0.0)
    s.observe_pre_gain_model_pcm(
        _evidence_voice(6),
        profile,
        capture_generation=4,
    )
    assert s.speech_evidence_snapshot().disposition is (
        SpeechEvidenceDisposition.SATISFIED
    )

    s.reset()
    assert s.speech_evidence_snapshot().disposition is (
        SpeechEvidenceDisposition.UNAVAILABLE
    )
    s.prepend([np.ones(10, dtype="float32")], speech_at=1.0)
    bypassed = s.speech_evidence_snapshot()
    assert bypassed.disposition is SpeechEvidenceDisposition.BYPASSED
    assert bypassed.reason == "word_cut_handoff"


def test_vad_flicker_after_onset_does_not_hide_energetic_epoch_frames():
    s = _segment(evidence=True, sample_rate=1000)
    profile = _evidence_profile()
    voice = _evidence_voice(3)

    s.observe_vad(True, 0.0)
    s.observe_pre_gain_model_pcm(voice, profile, capture_generation=9)
    s.observe_vad(False, 0.1)
    s.observe_pre_gain_model_pcm(voice, profile, capture_generation=9)

    snapshot = s.speech_evidence_snapshot()
    assert snapshot.qualified_frames == 6
    assert snapshot.longest_qualified_run == 6
    assert snapshot.disposition is SpeechEvidenceDisposition.SATISFIED


def test_confirmed_barge_adopts_bounded_pcm_without_waiting_for_vad():
    s = _segment(evidence=True, sample_rate=1000)
    idle_primary = np.full(20, 0.1, dtype="float32")
    idle_alternate = np.full(20, 0.15, dtype="float32")
    s.append(idle_primary, idle_alternate)
    primary = np.full(40, 0.2, dtype="float32")
    alternate = np.full(40, 0.3, dtype="float32")

    adopted = s.adopt_confirmed_barge_handoff(
        [primary],
        [alternate],
        speech_at=1.0,
        speech_end_at=1.04,
    )

    assert adopted == 40
    assert s.speech_seen is True
    owned_primary, owned_alternate = s.arrays()
    np.testing.assert_array_equal(
        owned_primary,
        np.concatenate([idle_primary, primary]),
    )
    np.testing.assert_array_equal(
        owned_alternate,
        np.concatenate([idle_alternate, alternate]),
    )
    snapshot = s.speech_evidence_snapshot()
    assert snapshot.disposition is SpeechEvidenceDisposition.BYPASSED
    assert snapshot.reason == "confirmed_barge_handoff"
