"""Unit tests for the speaker-ID barge-in gate decision logic.

The embedding function is injected, so these run with no sherpa-onnx and no
model files — they verify the *gate policy*, not the embedding model.
"""

from __future__ import annotations

from core.engines.speaker_gate import SpeakerGate, cosine_similarity

USER = [1.0, 0.0, 0.0, 0.0]
USER_NOISY = [0.92, 0.1, 0.05, 0.0]  # same speaker, slightly different
ASSISTANT_TTS = [0.0, 1.0, 0.0, 0.0]  # a different "voice"


def _gate_returning(vec, threshold=0.5):
    gate = SpeakerGate(threshold=threshold, embed_fn=lambda samples, sr: vec)
    return gate


def test_cosine_basics():
    assert cosine_similarity(USER, USER) == 1.0
    assert cosine_similarity(USER, ASSISTANT_TTS) == 0.0
    assert cosine_similarity([], [1.0]) == 0.0


def test_unenrolled_gate_fails_open():
    # No enrollment -> never block barge-in.
    gate = _gate_returning(ASSISTANT_TTS)
    assert not gate.is_enrolled
    assert gate.accept([0.0], 16000) is True


def test_enrolled_user_voice_is_accepted_as_barge_in():
    gate = _gate_returning(USER_NOISY)
    gate.enroll_embedding(USER)
    assert gate.accept([0.0], 16000) is True


def test_assistant_voice_is_rejected():
    gate = _gate_returning(ASSISTANT_TTS)
    gate.enroll_embedding(USER)
    # Different speaker -> not a real barge-in -> blocked.
    assert gate.accept([0.0], 16000) is False


def test_threshold_boundary():
    # Build a vector with a known cosine to USER and check threshold behavior.
    partial = [0.6, 0.8, 0.0, 0.0]  # cosine with USER = 0.6
    assert abs(cosine_similarity(partial, USER) - 0.6) < 1e-9

    strict = _gate_returning(partial, threshold=0.7)
    strict.enroll_embedding(USER)
    assert strict.accept([0.0], 16000) is False  # 0.6 < 0.7

    lax = _gate_returning(partial, threshold=0.5)
    lax.enroll_embedding(USER)
    assert lax.accept([0.0], 16000) is True  # 0.6 >= 0.5


def test_enroll_via_embed_fn():
    gate = SpeakerGate(threshold=0.5, embed_fn=lambda samples, sr: USER)
    assert gate.enroll([0.1, 0.2], 16000) is True
    assert gate.is_enrolled
    assert gate.accept([0.0], 16000) is True  # same embed_fn -> matches itself


def test_unusable_embedding_fails_open():
    gate = SpeakerGate(threshold=0.5, embed_fn=lambda samples, sr: None)
    gate.enroll_embedding(USER)
    assert gate.accept([0.0], 16000) is True  # None embedding -> don't block


def test_embed_returns_raw_vector_without_enrolling():
    # enroll_from_recordings relies on embed() exposing the per-recording vector
    # without mutating the enrolled reference.
    gate = SpeakerGate(threshold=0.5, embed_fn=lambda samples, sr: USER)
    assert list(gate.embed([0.1, 0.2], 16000)) == USER
    assert not gate.is_enrolled
