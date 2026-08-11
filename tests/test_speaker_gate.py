"""Unit tests for the speaker-ID barge-in gate decision logic.

The embedding function is injected, so these run with no sherpa-onnx and no
model files — they verify the *gate policy*, not the embedding model.
"""

from __future__ import annotations

from dataclasses import replace
import threading
import time

import pytest

from core.engines.speaker_gate import (
    RuntimeSpeakerInferencePermit,
    RuntimeSpeakerInferencePermitSnapshot,
    SpeakerGate,
    cosine_similarity,
)

USER = [1.0, 0.0, 0.0, 0.0]
USER_NOISY = [0.92, 0.1, 0.05, 0.0]  # same speaker, slightly different
ASSISTANT_TTS = [0.0, 1.0, 0.0, 0.0]  # a different "voice"


def _gate_returning(vec, threshold=0.5):
    gate = SpeakerGate(threshold=threshold, embed_fn=lambda samples, sr: vec)
    return gate


def _call_runtime_try_path(gate, method_name, permit):
    if method_name == "try_similarity_batch":
        return gate.try_similarity_batch(
            ((1.0,),),
            16_000,
            runtime_permit=permit,
        )
    return getattr(gate, method_name)(
        (1.0,),
        16_000,
        runtime_permit=permit,
    )


def test_runtime_speaker_inference_permit_tracks_exact_lease_lifecycle():
    permit = RuntimeSpeakerInferencePermit()

    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=False,
        acquisitions=0,
        releases=0,
    )

    first = permit.try_acquire()
    assert first is not None
    assert permit.is_current(first)
    assert permit.try_acquire() is None
    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=True,
        acquisitions=1,
        releases=0,
    )

    assert permit.release(first)
    assert not permit.is_current(first)
    assert not permit.release(first)
    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=False,
        acquisitions=1,
        releases=1,
    )

    second = permit.try_acquire()
    assert second is not None
    assert second is not first
    assert not permit.release(first)
    assert permit.is_current(second)
    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=True,
        acquisitions=2,
        releases=1,
    )
    assert permit.release(second)


def test_runtime_speaker_inference_permit_rejects_forged_and_cross_leases():
    permit = RuntimeSpeakerInferencePermit()
    other_permit = RuntimeSpeakerInferencePermit()
    lease = permit.try_acquire()
    other_lease = other_permit.try_acquire()
    assert lease is not None
    assert other_lease is not None

    forged = replace(lease)
    assert forged is not lease
    assert not permit.is_current(forged)
    assert not permit.release(forged)
    assert not permit.is_current(other_lease)
    assert not permit.release(other_lease)
    assert not other_permit.is_current(lease)
    assert not other_permit.release(lease)

    assert permit.is_current(lease)
    assert other_permit.is_current(other_lease)
    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=True,
        acquisitions=1,
        releases=0,
    )
    assert other_permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=True,
        acquisitions=1,
        releases=0,
    )
    assert permit.release(lease)
    assert other_permit.release(other_lease)


def test_runtime_speaker_inference_permit_serializes_distinct_gates():
    calls: list[str] = []
    permit = RuntimeSpeakerInferencePermit()
    second = SpeakerGate(
        embed_fn=lambda _samples, _sample_rate: calls.append("second") or USER
    )

    def first_embed(_samples, _sample_rate):
        calls.append("first")
        assert (
            second.try_embed(
                (2.0,),
                16_000,
                runtime_permit=permit,
            )
            is None
        )
        return USER

    first = SpeakerGate(embed_fn=first_embed)

    assert first.try_embed((1.0,), 16_000, runtime_permit=permit) == USER
    assert calls == ["first"]
    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=False,
        acquisitions=1,
        releases=1,
    )


@pytest.mark.parametrize(
    "method_name",
    [
        "try_similarity",
        "try_similarity_batch",
        "try_verification_similarity",
        "try_embed",
        "try_enroll",
    ],
)
def test_runtime_try_paths_abstain_when_process_permit_is_busy(method_name):
    calls: list[tuple[float, ...]] = []

    def embed(samples, _sample_rate):
        calls.append(tuple(float(sample) for sample in samples))
        return USER

    permit = RuntimeSpeakerInferencePermit()
    gate = SpeakerGate(embed_fn=embed)
    gate.enroll_embedding(USER)
    lease = permit.try_acquire()
    assert lease is not None

    assert _call_runtime_try_path(gate, method_name, permit) is None
    assert calls == []
    assert permit.is_current(lease)
    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=True,
        acquisitions=1,
        releases=0,
    )
    assert permit.release(lease)


@pytest.mark.parametrize(
    "method_name",
    [
        "try_similarity",
        "try_similarity_batch",
        "try_verification_similarity",
        "try_embed",
        "try_enroll",
    ],
)
def test_runtime_try_paths_release_permit_when_gate_lock_is_busy(method_name):
    calls = 0

    def embed(_samples, _sample_rate):
        nonlocal calls
        calls += 1
        return USER

    permit = RuntimeSpeakerInferencePermit()
    gate = SpeakerGate(embed_fn=embed)
    gate.enroll_embedding(USER)
    assert gate._inference_lock.acquire(blocking=False)
    try:
        assert _call_runtime_try_path(gate, method_name, permit) is None
    finally:
        gate._inference_lock.release()

    assert calls == 0
    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=False,
        acquisitions=1,
        releases=1,
    )


@pytest.mark.parametrize("method_name", ["try_similarity_batch", "try_enroll"])
def test_runtime_try_paths_release_permit_after_embedding_error(method_name):
    def embed(_samples, _sample_rate):
        raise RuntimeError("injected runtime embedding failure")

    permit = RuntimeSpeakerInferencePermit()
    gate = SpeakerGate(embed_fn=embed)
    gate.enroll_embedding(USER)

    with pytest.raises(RuntimeError, match="injected runtime embedding failure"):
        _call_runtime_try_path(gate, method_name, permit)

    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=False,
        acquisitions=1,
        releases=1,
    )
    assert gate._inference_lock.acquire(blocking=False)
    gate._inference_lock.release()


def test_similarity_batch_releases_owned_permit_when_gate_lock_acquire_raises():
    class RaisingLock:
        def acquire(self, *, blocking):
            assert blocking is False
            raise RuntimeError("injected gate-lock acquisition failure")

        def release(self):
            raise AssertionError("an unacquired gate lock must not be released")

    permit = RuntimeSpeakerInferencePermit()
    gate = SpeakerGate(embed_fn=lambda _samples, _sample_rate: USER)
    gate.enroll_embedding(USER)
    gate._inference_lock = RaisingLock()

    with pytest.raises(RuntimeError, match="gate-lock acquisition failure"):
        gate.try_similarity_batch(
            ((1.0,),),
            16_000,
            runtime_permit=permit,
        )

    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=False,
        acquisitions=1,
        releases=1,
    )


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


def test_clear_enrollment_restores_fail_open_state():
    gate = _gate_returning(ASSISTANT_TTS)
    gate.enroll_embedding(USER)
    assert not gate.accept([0.0], 16000)
    gate.clear_enrollment()
    assert not gate.is_enrolled
    assert gate.accept([0.0], 16000)


def test_clear_during_embedding_makes_inflight_decision_fail_open():
    embedding_started = threading.Event()
    release_embedding = threading.Event()

    def embed(_samples, _sample_rate):
        embedding_started.set()
        if not release_embedding.wait(timeout=2.0):
            raise RuntimeError("test did not release embedding")
        return ASSISTANT_TTS

    gate = SpeakerGate(threshold=0.5, embed_fn=embed)
    gate.enroll_embedding(USER)
    accepted: list[bool] = []
    worker = threading.Thread(
        target=lambda: accepted.append(gate.accept([0.0], 16000))
    )

    worker.start()
    assert embedding_started.wait(timeout=2.0)
    gate.clear_enrollment()
    release_embedding.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert accepted == [True]
    assert not gate.is_enrolled


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


def test_similarity_batch_scores_at_most_two_clips_under_one_current_receipt():
    embedded: list[tuple[float, ...]] = []

    def embed(samples, _sample_rate):
        clip = tuple(float(value) for value in samples)
        embedded.append(clip)
        return USER if clip[0] > 0.0 else ASSISTANT_TTS

    gate = SpeakerGate(threshold=0.5, embed_fn=embed)
    gate.enroll_embedding(USER)

    one = gate.try_similarity_batch(((1.0,),), 16_000)
    two = gate.try_similarity_batch(((2.0,), (-1.0,)), 16_000)

    assert one is not None
    assert one.similarities == (1.0,)
    assert gate.authority_is_current(one.authority)
    assert two is not None
    assert two.similarities == (1.0, 0.0)
    assert gate.authority_is_current(two.authority)
    assert embedded == [(1.0,), (2.0,), (-1.0,)]


@pytest.mark.parametrize("clips", [(), ((1.0,), (2.0,), (3.0,))])
def test_similarity_batch_rejects_zero_or_more_than_two_clips(clips):
    gate = _gate_returning(USER)
    gate.enroll_embedding(USER)

    with pytest.raises(ValueError, match="one or two"):
        gate.try_similarity_batch(clips, 16_000)


def test_similarity_batch_busy_abstains_without_running_embedding():
    calls = 0

    def embed(_samples, _sample_rate):
        nonlocal calls
        calls += 1
        return USER

    gate = SpeakerGate(threshold=0.5, embed_fn=embed)
    gate.enroll_embedding(USER)
    assert gate._inference_lock.acquire(blocking=False)
    try:
        assert gate.try_similarity_batch(((1.0,), (2.0,)), 16_000) is None
    finally:
        gate._inference_lock.release()
    assert calls == 0


def test_similarity_batch_unenrolled_receipt_runs_no_embedding():
    calls = 0

    def embed(_samples, _sample_rate):
        nonlocal calls
        calls += 1
        return USER

    gate = SpeakerGate(threshold=0.5, embed_fn=embed)

    receipt = gate.try_similarity_batch(((1.0,), (2.0,)), 16_000)

    assert receipt is not None
    assert receipt.similarities == (0.0, 0.0)
    assert receipt.authority.enrolled is False
    assert gate.authority_is_current(receipt.authority)
    assert calls == 0


def test_similarity_batch_embedding_error_releases_inference_lock():
    gate = SpeakerGate(
        threshold=0.5,
        embed_fn=lambda _samples, _sample_rate: (_ for _ in ()).throw(
            RuntimeError("injected batch failure")
        ),
    )
    gate.enroll_embedding(USER)

    with pytest.raises(RuntimeError, match="injected batch failure"):
        gate.try_similarity_batch(((1.0,),), 16_000)

    gate.set_embed_fn(lambda _samples, _sample_rate: USER)
    receipt = gate.try_similarity_batch(((1.0,),), 16_000)
    assert receipt is not None
    assert receipt.similarities == (1.0,)


@pytest.mark.parametrize(
    "mutation",
    ["model", "enrollment", "policy"],
)
def test_similarity_batch_receipt_is_stale_after_authority_mutation(mutation):
    gate = _gate_returning(USER)
    gate.enroll_embedding(USER)
    receipt = gate.try_similarity_batch(((1.0,),), 16_000)
    assert receipt is not None
    assert gate.authority_is_current(receipt.authority)

    if mutation == "model":
        gate.set_embed_fn(lambda _samples, _sample_rate: USER)
    elif mutation == "enrollment":
        gate.enroll_embedding(USER)
    else:
        gate.threshold = gate.threshold

    assert not gate.authority_is_current(receipt.authority)


def test_similarity_batch_receipt_cannot_cross_gate_instances():
    first = _gate_returning(USER)
    second = _gate_returning(USER)
    first.enroll_embedding(USER)
    second.enroll_embedding(USER)
    receipt = first.try_similarity_batch(((1.0,),), 16_000)

    assert receipt is not None
    assert first.authority_is_current(receipt.authority)
    assert not second.authority_is_current(receipt.authority)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_generation", False),
        ("enrollment_generation", True),
        ("policy_generation", False),
        ("enrolled", 1),
    ],
)
def test_similarity_batch_authority_rejects_coercible_field_types(field, value):
    gate = _gate_returning(USER)
    gate.enroll_embedding(USER)
    receipt = gate.try_similarity_batch(((1.0,),), 16_000)
    assert receipt is not None

    forged = replace(receipt.authority, **{field: value})

    assert not gate.authority_is_current(forged)


def test_similarity_batch_enrollment_and_policy_aba_during_inference_is_stale():
    embedding_started = threading.Event()
    release_embedding = threading.Event()

    def embed(_samples, _sample_rate):
        embedding_started.set()
        assert release_embedding.wait(timeout=2.0)
        return USER

    gate = SpeakerGate(threshold=0.5, embed_fn=embed)
    gate.enroll_embedding(USER)
    receipts = []
    worker = threading.Thread(
        target=lambda: receipts.append(gate.try_similarity_batch(((1.0,),), 16_000))
    )
    worker.start()
    assert embedding_started.wait(timeout=2.0)
    gate.clear_enrollment()
    gate.enroll_embedding(USER)
    gate.threshold = gate.threshold
    release_embedding.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert len(receipts) == 1
    assert receipts[0] is not None
    assert receipts[0].similarities == (0.0,)
    assert not gate.authority_is_current(receipts[0].authority)


def test_authority_recheck_linearizes_before_concurrent_enrollment_mutation():
    gate = _gate_returning(USER)
    gate.enroll_embedding(USER)
    authority = gate.authority_state()
    results: list[bool] = []
    mutation_done = threading.Event()

    gate._policy_lock.acquire()
    checker = threading.Thread(
        target=lambda: results.append(gate.authority_is_current(authority))
    )
    checker.start()
    deadline = time.monotonic() + 2.0
    while gate._enrollment_lock.acquire(blocking=False):
        gate._enrollment_lock.release()
        assert time.monotonic() < deadline
        time.sleep(0.001)

    def mutate_enrollment() -> None:
        gate.enroll_embedding(USER)
        mutation_done.set()

    mutator = threading.Thread(target=mutate_enrollment)
    mutator.start()
    assert not mutation_done.wait(timeout=0.05)
    gate._policy_lock.release()
    checker.join(timeout=2.0)
    mutator.join(timeout=2.0)

    assert not checker.is_alive()
    assert not mutator.is_alive()
    assert results == [True]
    assert mutation_done.is_set()
    assert not gate.authority_is_current(authority)


def test_authority_recheck_observes_policy_mutation_while_waiting_for_snapshot():
    gate = _gate_returning(USER)
    gate.enroll_embedding(USER)
    authority = gate.authority_state()
    results: list[bool] = []

    gate._enrollment_lock.acquire()
    checker = threading.Thread(
        target=lambda: results.append(gate.authority_is_current(authority))
    )
    checker.start()
    gate.threshold = 0.75
    gate._enrollment_lock.release()
    checker.join(timeout=2.0)

    assert not checker.is_alive()
    assert results == [False]


def test_loudness_admits_logic():
    from core.engines.speaker_gate import loudness_admits
    assert loudness_admits(0.1, 0.01, margin_db=0) is True       # disabled -> abstain True
    assert loudness_admits(0.1, 0.0, margin_db=10) is True        # no floor yet -> abstain True
    assert loudness_admits(0.1, 0.01, margin_db=10) is True       # 20 dB over >= 10 -> admit
    assert loudness_admits(0.012, 0.01, margin_db=10) is False    # ~1.5 dB over < 10 -> reject
    assert loudness_admits(0.0, 0.01, margin_db=10) is False      # silent -> reject
