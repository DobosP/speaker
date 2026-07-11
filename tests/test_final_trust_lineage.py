"""Typed engine-final trust from Sherpa through the runtime event boundary.

Admission and owner verification are deliberately different facts.  A final may
be admitted so the assistant remains usable when identity is disabled or broken,
but only an exact enrolled-speaker acceptance over unmixed final speech may mark
the runtime turn as owner verified.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from always_on_agent.events import EventKind
from core.engine import EngineCallbacks
from core.engines._asr_segment import ASRSegment
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from core.engines.speaker_gate import SpeakerGate
from core.llm import EchoLLM
from core.runtime import VoiceRuntime


USER = [1.0, 0.0]
OTHER = [0.0, 1.0]


def _trust_types():
    # Kept inside the tests so the pre-implementation suite collects and reports
    # exact failing test names instead of aborting module collection.
    from core.engine import FinalTranscript, OwnerVerification

    return FinalTranscript, OwnerVerification


def _gate(embedding, *, threshold: float = 0.5) -> SpeakerGate:
    gate = SpeakerGate(
        threshold=threshold,
        embed_fn=lambda _samples, _sample_rate: embedding,
    )
    gate.enroll_embedding(USER)
    return gate


def _wire_typed_runtime(engine: SherpaOnnxEngine):
    FinalTranscript, _OwnerVerification = _trust_types()
    runtime = VoiceRuntime(engine, EchoLLM(reply="unused"))
    published = []
    typed_results = []
    legacy_text = []

    # Stop at the runtime event seam: publishing onto the live bus would consume
    # the event immediately and turn this trust-contract test into an LLM test.
    runtime.bus.publish = published.append

    def on_final_result(result):
        assert isinstance(result, FinalTranscript)
        typed_results.append(result)
        runtime._on_final_result(result)

    engine._cb = EngineCallbacks(
        on_final=legacy_text.append,
        on_final_result=on_final_result,
    )
    engine._final_recognizer = None
    engine._final_above_floor = lambda _samples: True
    return typed_results, legacy_text, published


def _dispatch(
    engine: SherpaOnnxEngine,
    samples,
    text: str = "owner command",
    **finalizer_kwargs,
):
    engine._finalize_and_dispatch(
        np.asarray(samples, dtype="float32"),
        text,
        time.perf_counter(),
        **finalizer_kwargs,
    )


def _final_events(published):
    return [event for event in published if event.kind is EventKind.STT_FINAL]


def test_final_transcript_defaults_fail_closed():
    FinalTranscript, OwnerVerification = _trust_types()

    result = FinalTranscript("hello")

    assert result.text == "hello"
    assert result.owner_verification is OwnerVerification.UNKNOWN
    assert result.origin == "unknown"
    assert OwnerVerification.REJECTED is not OwnerVerification.UNKNOWN
    assert OwnerVerification.REJECTED is not OwnerVerification.VERIFIED


def test_exact_final_speaker_accept_reaches_runtime_as_verified_live_audio():
    _FinalTranscript, OwnerVerification = _trust_types()
    engine = SherpaOnnxEngine(SherpaConfig(speaker_gate_input=True))
    engine._speaker_gate = _gate(USER)
    typed, legacy, published = _wire_typed_runtime(engine)

    _dispatch(engine, np.full(3200, 0.2, dtype="float32"))

    assert legacy == []
    assert len(typed) == 1
    assert typed[0].owner_verification is OwnerVerification.VERIFIED
    assert typed[0].origin == "live_audio"
    [event] = _final_events(published)
    assert event.payload["owner_verified"] is True
    assert event.payload["origin"] == "live_audio"


@pytest.mark.parametrize(
    "case",
    [
        "gate_off",
        "unavailable",
        "embed_none",
        "embed_error",
        "loudness_rescue",
    ],
)
def test_fail_open_or_rescued_final_is_admitted_but_never_owner_verified(case):
    _FinalTranscript, OwnerVerification = _trust_types()
    samples = np.full(3200, 0.2, dtype="float32")

    if case == "gate_off":
        engine = SherpaOnnxEngine(SherpaConfig(speaker_gate_input=False))
        engine._speaker_gate = _gate(USER)
    elif case == "unavailable":
        engine = SherpaOnnxEngine(SherpaConfig(speaker_gate_input=True))
        engine._speaker_gate = None
    elif case == "embed_none":
        # SpeakerGate.accept() historically returns True here as a usability
        # fail-open.  Typed trust must not confuse that admission with a real
        # finite cosine match against the enrolled snapshot.
        engine = SherpaOnnxEngine(SherpaConfig(speaker_gate_input=True))
        engine._speaker_gate = SpeakerGate(
            threshold=0.5,
            embed_fn=lambda _samples, _sample_rate: None,
        )
        engine._speaker_gate.enroll_embedding(USER)
    elif case == "embed_error":
        def broken(_samples, _sample_rate):
            raise RuntimeError("speaker backend failed")

        engine = SherpaOnnxEngine(SherpaConfig(speaker_gate_input=True))
        engine._speaker_gate = SpeakerGate(threshold=0.5, embed_fn=broken)
        engine._speaker_gate.enroll_embedding(USER)
    else:
        engine = SherpaOnnxEngine(
            SherpaConfig(
                speaker_gate_input=True,
                input_loudness_margin_db=10.0,
            )
        )
        engine._speaker_gate = _gate(OTHER)
        engine._ambient_rms = 0.01
        samples = np.full(3200, 0.5, dtype="float32")

    typed, legacy, published = _wire_typed_runtime(engine)
    _dispatch(engine, samples)

    assert legacy == []
    assert len(typed) == 1  # admitted for usability, not identity-attested
    expected = (
        OwnerVerification.REJECTED
        if case == "loudness_rescue"
        else OwnerVerification.UNKNOWN
    )
    assert typed[0].owner_verification is expected
    [event] = _final_events(published)
    assert event.payload["owner_verified"] is False


def test_double_talk_point30_barge_acceptance_is_not_final_owner_verification():
    _FinalTranscript, OwnerVerification = _trust_types()
    similarity = 0.387
    double_talk_embedding = [similarity, math.sqrt(1.0 - similarity**2)]
    engine = SherpaOnnxEngine(
        SherpaConfig(
            barge_word_cut_enabled=True,
            barge_word_cut_require_speaker=True,
            barge_word_cut_speaker_min_sec=0.0,
            barge_word_cut_speaker_threshold=0.30,
            speaker_gate_input=False,
        )
    )
    engine._speaker_gate = _gate(double_talk_embedding)
    engine._speaker_gate_warmed = True
    owner_pcm = np.full(1600, 0.2, dtype="float32")
    engine._append_word_cut_candidate(owner_pcm)
    assert engine._word_cut_speaker_decision() == "accept"
    assert engine._promote_word_cut_candidate(
        object(),
        None,
        None,
        reason="cut",
        text="",
    )
    pending = []
    engine._splice_word_cut_preroll(pending)
    segment = np.concatenate(pending)
    typed, legacy, published = _wire_typed_runtime(engine)

    _dispatch(engine, segment)

    assert legacy == []
    assert len(typed) == 1
    assert typed[0].owner_verification is OwnerVerification.UNKNOWN
    [event] = _final_events(published)
    assert event.payload["owner_verified"] is False


def test_other_speaker_post_cut_continuation_cannot_inherit_owner_attestation():
    _FinalTranscript, OwnerVerification = _trust_types()
    owner_pcm = np.full(1600, 0.2, dtype="float32")
    other_pcm = np.full(1600, 0.7, dtype="float32")

    def embedding(samples, _sample_rate):
        audio = np.asarray(samples, dtype="float32")
        # The owner-only cut window is a clean accept.  The aggregate embedding
        # also accepts (the dilution failure this regression guards), while an
        # independently checked continuation is clearly another speaker.
        if audio.size and np.all(audio == np.float32(0.7)):
            return OTHER
        return USER

    gate = SpeakerGate(threshold=0.5, embed_fn=embedding)
    gate.enroll_embedding(USER)
    engine = SherpaOnnxEngine(
        SherpaConfig(
            barge_word_cut_enabled=True,
            barge_word_cut_require_speaker=True,
            barge_word_cut_speaker_min_sec=0.0,
            speaker_gate_input=True,
        )
    )
    engine._speaker_gate = gate
    engine._speaker_gate_warmed = True
    engine._append_word_cut_candidate(owner_pcm)
    assert engine._word_cut_speaker_decision() == "accept"
    assert engine._promote_word_cut_candidate(
        object(),
        None,
        None,
        reason="cut",
        text="",
    )
    pending = []
    engine._splice_word_cut_preroll(pending)
    mixed_segment = np.concatenate([*pending, other_pcm])
    typed, legacy, published = _wire_typed_runtime(engine)

    _dispatch(engine, mixed_segment, owner_lineage_intact=False)

    assert legacy == []
    # Dropping the mixed final is safe.  If it is admitted for a non-action
    # response, neither the typed result nor the runtime event may be verified.
    assert not typed or typed[0].owner_verification is not OwnerVerification.VERIFIED
    for event in _final_events(published):
        assert event.payload["owner_verified"] is False


def test_asr_segment_demotes_word_cut_lineage_on_fresh_voiced_continuation():
    owner_pcm = np.full(1600, 0.2, dtype="float32")
    segment = ASRSegment(
        sample_rate=16000,
        pre_roll_sec=0.8,
        max_utterance_sec=20.8,
        vad_available=True,
        block_sec=0.1,
    )
    segment.prepend(
        [owner_pcm],
        speech_at=1.0,
        speech_end_at=1.0,
        offline_recovery_authorized=True,
    )

    assert segment.owner_lineage_intact is True
    segment.observe_vad(False, 1.1)  # endpoint silence is not a new speaker
    assert segment.owner_lineage_intact is True
    segment.observe_vad(True, 1.2)  # fresh voiced PCM was not cut-time attested
    assert segment.owner_lineage_intact is False
