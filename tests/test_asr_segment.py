"""Pure regressions for VAD-backed live ASR segment ownership."""
from __future__ import annotations

import numpy as np

from core.engines._asr_segment import ASRSegment


def _segment(*, vad: bool = True, pre: float = 0.8, maximum: float = 22.0):
    return ASRSegment(
        sample_rate=100,
        pre_roll_sec=pre,
        max_utterance_sec=maximum,
        vad_available=vad,
        block_sec=0.1,
    )


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
