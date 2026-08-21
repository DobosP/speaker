"""Tests for gating normal ASR finals on speaker identity (input gating).

These exercise SherpaOnnxEngine._should_act_on_final and _enroll_speaker_gate
directly with an injected gate -- no sherpa-onnx, no models, no audio device.
The capture loop's threaded I/O is out of scope; the decision logic is not.
"""
from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from core.engine import OwnerVerification
from core.engines._kws_speaker_inference_owner import (
    KwsSpeakerInferenceOutcome,
    try_claim_kws_speaker_inference_owner,
)
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from core.engines.speaker_gate import (
    SpeakerGate,
    runtime_speaker_inference_permit,
)

USER = [1.0, 0.0, 0.0]
OTHER = [0.0, 1.0, 0.0]


def _capture():
    from core.enroll import CaptureResolution

    return CaptureResolution(
        route="test-mic",
        capture_sample_rate=16000,
        model_sample_rate=16000,
        resampler="identity",
    )


def _gate(embed, *, enrolled_to=USER):
    g = SpeakerGate(threshold=0.5, embed_fn=lambda samples, sr: embed)
    if enrolled_to is not None:
        g.enroll_embedding(enrolled_to)
    return g


def _engine(*, gate_input=True, gate=None):
    eng = SherpaOnnxEngine(SherpaConfig(speaker_gate_input=gate_input))
    eng._speaker_gate = gate
    return eng


class _ReturnedOwnedSpeakerTask:
    def __init__(self, permit, lease) -> None:
        self.permit = permit
        self.lease = lease
        self.reap_calls = 0

    def try_reap(self) -> bool:
        self.reap_calls += 1
        assert self.reap_calls == 1
        assert self.permit.release(self.lease)
        return True


def _build_with_observed_speaker_allocation(monkeypatch, config):
    import core.engines.sherpa as sherpa_module

    for name in (
        "build_recognizer",
        "build_final_recognizer",
        "build_final_verifier",
        "build_vad",
        "build_tts",
        "build_denoiser",
        "build_keyword_spotter",
        "build_punctuation",
    ):
        monkeypatch.setattr(sherpa_module, name, lambda _config: None)
    monkeypatch.setattr(
        sherpa_module,
        "build_aec",
        lambda _config, **_kwargs: None,
    )

    calls = []
    allocated_gate = object()

    def build_gate(model, **kwargs):
        calls.append((model, kwargs))
        return allocated_gate

    monkeypatch.setattr(sherpa_module, "sherpa_speaker_gate", build_gate)
    engine = SherpaOnnxEngine(config)
    engine._build()
    return engine, allocated_gate, calls


def test_inactive_configured_speaker_model_does_not_allocate_gate(monkeypatch):
    engine, _allocated_gate, calls = _build_with_observed_speaker_allocation(
        monkeypatch,
        SherpaConfig(
            speaker_embedding_model="/missing/spk.onnx",
            speaker_enroll_embedding="/missing/enroll.json",
            barge_in_enabled=True,
            barge_word_cut_enabled=True,
            barge_word_cut_require_speaker=False,
            aec_enabled=False,
            coherence_barge_in_enabled=False,
        ),
    )

    assert calls == []
    assert engine._speaker_gate is None


def test_existing_enrollment_allocates_speaker_gate(monkeypatch, tmp_path):
    from core.enroll import Enrollment, save_enrollment

    model = tmp_path / "spk.onnx"
    model.touch()
    enrollment = tmp_path / "enroll.json"
    save_enrollment(
        str(enrollment),
        Enrollment(model=str(model), embedding=USER),
    )
    engine, allocated_gate, calls = _build_with_observed_speaker_allocation(
        monkeypatch,
        SherpaConfig(
            speaker_embedding_model=str(model),
            speaker_enroll_embedding=str(enrollment),
            coherence_barge_in_enabled=False,
        ),
    )

    assert [model_path for model_path, _kwargs in calls] == [str(model)]
    assert engine._speaker_gate is allocated_gate


def test_existing_wav_enrollment_allocates_speaker_gate(monkeypatch, tmp_path):
    model = tmp_path / "spk.onnx"
    model.touch()
    enrollment = tmp_path / "enroll.wav"
    enrollment.touch()
    engine, allocated_gate, calls = _build_with_observed_speaker_allocation(
        monkeypatch,
        SherpaConfig(
            speaker_embedding_model=str(model),
            speaker_enroll_wav=str(enrollment),
            coherence_barge_in_enabled=False,
        ),
    )

    assert [model_path for model_path, _kwargs in calls] == [str(model)]
    assert engine._speaker_gate is allocated_gate


def test_session_mask_preserves_files_and_prevents_speaker_gate_allocation(
    monkeypatch, tmp_path
):
    from core.config import apply_no_speaker_enrollment

    model = tmp_path / "spk.onnx"
    model.write_bytes(b"model-sentinel")
    embedding = tmp_path / "enroll.json"
    embedding.write_bytes(b"embedding-sentinel")
    wav = tmp_path / "enroll.wav"
    wav.write_bytes(b"wav-sentinel")
    effective = apply_no_speaker_enrollment(
        {
            "sherpa": {
                "speaker_embedding_model": str(model),
                "speaker_enroll_embedding": str(embedding),
                "speaker_enroll_wav": str(wav),
                "barge_in_enabled": True,
                "barge_word_cut_enabled": True,
                "barge_word_cut_require_speaker": False,
                "aec_enabled": False,
                "coherence_barge_in_enabled": False,
            }
        }
    )

    engine, _allocated_gate, calls = _build_with_observed_speaker_allocation(
        monkeypatch,
        SherpaConfig(**effective["sherpa"]),
    )

    assert calls == []
    assert engine._speaker_gate is None
    assert model.read_bytes() == b"model-sentinel"
    assert embedding.read_bytes() == b"embedding-sentinel"
    assert wav.read_bytes() == b"wav-sentinel"


def test_active_os_word_cut_speaker_filter_allocates_gate_without_enrollment(
    monkeypatch, tmp_path
):
    model = tmp_path / "spk.onnx"
    model.touch()
    engine, allocated_gate, calls = _build_with_observed_speaker_allocation(
        monkeypatch,
        SherpaConfig(
            speaker_embedding_model=str(model),
            barge_in_enabled=True,
            barge_word_cut_enabled=True,
            barge_word_cut_require_speaker=True,
            aec_enabled=False,
            coherence_barge_in_enabled=False,
        ),
    )

    assert [model_path for model_path, _kwargs in calls] == [str(model)]
    assert engine._speaker_gate is allocated_gate


def test_required_word_cut_incompatible_enrollment_fails_start_policy():
    engine = SherpaOnnxEngine(
        SherpaConfig(
            barge_in_enabled=True,
            barge_word_cut_enabled=True,
            barge_word_cut_require_speaker=True,
            aec_enabled=False,
        )
    )
    engine._speaker_gate = _gate(USER, enrolled_to=None)

    with pytest.raises(
        RuntimeError,
        match="active word-cut requires a compatible speaker enrollment",
    ):
        engine._require_word_cut_speaker_authority()


def test_in_app_aec_keeps_unenrolled_speaker_model_inactive(monkeypatch, tmp_path):
    model = tmp_path / "spk.onnx"
    model.touch()
    engine, _allocated_gate, calls = _build_with_observed_speaker_allocation(
        monkeypatch,
        SherpaConfig(
            speaker_embedding_model=str(model),
            barge_in_enabled=True,
            barge_word_cut_enabled=True,
            barge_word_cut_require_speaker=True,
            aec_enabled=True,
            coherence_barge_in_enabled=False,
        ),
    )

    assert calls == []
    assert engine._speaker_gate is None


def test_rebuild_clears_gate_after_enrollment_reference_disappears(
    monkeypatch, tmp_path
):
    from core.enroll import Enrollment, save_enrollment

    model = tmp_path / "spk.onnx"
    model.touch()
    enrollment = tmp_path / "enroll.json"
    save_enrollment(
        str(enrollment),
        Enrollment(model=str(model), embedding=USER),
    )
    engine, allocated_gate, calls = _build_with_observed_speaker_allocation(
        monkeypatch,
        SherpaConfig(
            speaker_embedding_model=str(model),
            speaker_enroll_embedding=str(enrollment),
            coherence_barge_in_enabled=False,
        ),
    )
    assert engine._speaker_gate is allocated_gate

    enrollment.unlink()
    engine._build()

    assert len(calls) == 1
    assert engine._speaker_gate is None
    assert engine._speaker_gate_warmed is False


def test_input_gating_disabled_acts_even_with_rejecting_gate():
    eng = _engine(gate_input=False, gate=_gate(OTHER))  # gate would reject
    assert eng._should_act_on_final([0.0]) is True


def test_no_gate_fails_open():
    assert _engine(gate=None)._should_act_on_final([0.0]) is True


def test_unenrolled_input_gate_fails_open():
    eng = _engine(gate=_gate(OTHER, enrolled_to=None))
    assert not eng._speaker_gate.is_enrolled
    assert eng._should_act_on_final([0.0]) is True


def test_enrolled_user_final_is_acted_on():
    assert _engine(gate=_gate(USER))._should_act_on_final([0.0]) is True


def test_final_speaker_verification_stays_unknown_while_process_permit_is_held():
    embed_calls = []
    gate = SpeakerGate(
        threshold=0.5,
        embed_fn=lambda _samples, _sample_rate: embed_calls.append("embed") or USER,
    )
    gate.enroll_embedding(USER)
    eng = _engine(gate=gate)
    permit = runtime_speaker_inference_permit()
    lease = permit.try_acquire()
    assert lease is not None
    try:
        decision = eng._speaker_decision_for_final(
            np.full(1_600, 0.25, dtype="float32")
        )
    finally:
        assert permit.release(lease)

    assert decision.admitted
    assert decision.verification is OwnerVerification.UNKNOWN
    assert embed_calls == []


def test_final_missing_try_seam_never_uses_blocking_legacy_or_mints_verified():
    class _LegacyGate:
        is_enrolled = True
        threshold = 0.5

        def verification_similarity(self, _samples, _sample_rate):
            raise AssertionError("blocking final similarity fallback was called")

        def accept(self, _samples, _sample_rate):
            raise AssertionError("blocking final admission fallback was called")

    eng = _engine(gate=_LegacyGate())

    decision = eng._speaker_decision_for_final(np.full(1_600, 0.25, dtype="float32"))

    assert decision.admitted
    assert decision.verification is OwnerVerification.UNKNOWN


def test_final_reaps_returned_owned_kws_task_before_nonblocking_similarity():
    embed_calls = []
    gate = SpeakerGate(
        threshold=0.5,
        embed_fn=lambda _samples, _sample_rate: embed_calls.append("embed") or USER,
    )
    gate.enroll_embedding(USER)
    eng = _engine(gate=gate)
    permit = runtime_speaker_inference_permit()
    lease = permit.try_acquire()
    assert lease is not None
    returned = _ReturnedOwnedSpeakerTask(permit, lease)
    eng._kws_speaker_inference_owner = returned

    decision = eng._speaker_decision_for_final(np.full(1_600, 0.25, dtype="float32"))

    assert decision.admitted
    assert decision.verification is OwnerVerification.VERIFIED
    assert embed_calls == ["embed"]
    assert returned.reap_calls == 1
    assert eng._kws_speaker_inference_owner is None
    assert not permit.snapshot().active


def test_final_reaps_returned_cross_engine_process_owner_before_similarity():
    class _InlineThread:
        def __init__(self, *, target, args, name, daemon) -> None:
            self.target = target
            self.args = args
            self.name = name
            self.daemon = daemon
            self.alive = False

        def start(self) -> None:
            self.alive = True
            try:
                self.target(*self.args)
            finally:
                self.alive = False

        def join(self, _timeout=None) -> None:
            pass

        def is_alive(self) -> bool:
            return self.alive

    foreign_gate = SpeakerGate(
        threshold=0.5,
        embed_fn=lambda _samples, _sample_rate: USER,
    )
    foreign_gate.enroll_embedding(USER)
    clip = np.full(160, 0.25, dtype="float32")
    clip.setflags(write=False)
    owner = try_claim_kws_speaker_inference_owner(
        foreign_gate,
        (clip,),
        16_000,
        deadline=time.monotonic() + 5.0,
        thread_factory=lambda **kwargs: _InlineThread(**kwargs),
    )
    assert owner is not None
    owner.start()
    assert owner.abandon(KwsSpeakerInferenceOutcome.STOPPED)
    assert owner.snapshot().worker_returned
    assert not owner.snapshot().reaped

    final_embed_calls = []
    final_gate = SpeakerGate(
        threshold=0.5,
        embed_fn=lambda _samples, _sample_rate: (
            final_embed_calls.append("embed") or USER
        ),
    )
    final_gate.enroll_embedding(USER)
    eng = _engine(gate=final_gate)
    try:
        decision = eng._speaker_decision_for_final(
            np.full(1_600, 0.25, dtype="float32")
        )
        reaped_by_final = owner.snapshot().reaped
    finally:
        owner.abandon(KwsSpeakerInferenceOutcome.STOPPED)
        owner.try_reap()

    assert decision.admitted
    assert decision.verification is OwnerVerification.VERIFIED
    assert final_embed_calls == ["embed"]
    assert reaped_by_final
    assert eng._kws_speaker_inference_owner is None
    assert not runtime_speaker_inference_permit().snapshot().active


def test_speaker_warm_stays_cold_while_process_permit_is_held() -> None:
    embed_calls = []
    gate = SpeakerGate(
        threshold=0.5,
        embed_fn=lambda _samples, _sample_rate: embed_calls.append("embed") or USER,
    )
    gate.enroll_embedding(USER)
    eng = _engine(gate=None)
    eng._replace_speaker_gate(gate)
    permit = runtime_speaker_inference_permit()
    lease = permit.try_acquire()
    assert lease is not None
    try:
        assert not eng._warm_speaker_gate()
    finally:
        assert permit.release(lease)

    assert not eng._speaker_gate_warmed
    assert embed_calls == []


def test_speaker_warm_reaps_returned_owned_kws_task_before_try_embed() -> None:
    embed_calls = []
    gate = SpeakerGate(
        threshold=0.5,
        embed_fn=lambda _samples, _sample_rate: embed_calls.append("embed") or USER,
    )
    eng = _engine(gate=None)
    eng._replace_speaker_gate(gate)
    permit = runtime_speaker_inference_permit()
    lease = permit.try_acquire()
    assert lease is not None
    returned = _ReturnedOwnedSpeakerTask(permit, lease)
    eng._kws_speaker_inference_owner = returned

    assert eng._warm_speaker_gate()

    assert eng._speaker_gate_warmed
    assert embed_calls == ["embed"]
    assert returned.reap_calls == 1
    assert eng._kws_speaker_inference_owner is None
    assert not permit.snapshot().active


def test_live_speaker_embedding_uses_same_voiced_envelope_as_enrollment():
    import numpy as np

    captured = []
    gate = SpeakerGate(
        threshold=0.5,
        embed_fn=lambda samples, sr: captured.append(np.asarray(samples)) or USER,
    )
    gate.enroll_embedding(USER)
    eng = _engine(gate=gate)
    clip = np.r_[
        np.zeros(16000, dtype="float32"),
        np.full(8000, 0.2, dtype="float32"),
        np.zeros(16000, dtype="float32"),
    ]

    assert eng._should_act_on_final(clip) is True
    assert len(captured) == 1
    from core.enroll import _vad_trim

    np.testing.assert_array_equal(captured[0], _vad_trim(clip, 16000))


def test_enrolled_other_voice_final_is_dropped():
    assert _engine(gate=_gate(OTHER))._should_act_on_final([0.0]) is False


def test_capture_try_similarity_never_overlaps_or_waits_for_final_inference():
    import threading

    entered = threading.Event()
    release = threading.Event()
    calls = []

    def embedding(_samples, _sr):
        calls.append("enter")
        entered.set()
        assert release.wait(1.0)
        calls.append("exit")
        return USER

    gate = SpeakerGate(threshold=0.5, embed_fn=embedding)
    gate.enroll_embedding(USER)
    result = []
    worker = threading.Thread(
        target=lambda: result.append(gate.similarity([0.2] * 1600, 16000))
    )
    worker.start()
    assert entered.wait(1.0)

    # The capture-thread seam abstains immediately instead of racing the shared
    # extractor or blocking behind the async final worker.
    assert gate.try_similarity([0.2] * 1600, 16000) is None
    assert calls == ["enter"]

    release.set()
    worker.join(1.0)
    assert not worker.is_alive()
    assert result == [1.0]
    assert calls == ["enter", "exit"]


def test_broken_speaker_embedder_fails_open_instead_of_dropping_turn():
    def broken(_samples, _sr):
        raise RuntimeError("embedding backend failed")

    gate = SpeakerGate(threshold=0.5, embed_fn=broken)
    gate.enroll_embedding(USER)
    assert _engine(gate=gate)._should_act_on_final([0.2] * 1600) is True


# --- _enroll_speaker_gate: load the persisted embedding into the gate --------


def test_enrollment_load_is_deferred_until_capture_resolution(tmp_path, caplog):
    import logging

    from core.enroll import Enrollment, save_enrollment

    model = "/m/spk.onnx"
    path = tmp_path / "enroll.json"
    save_enrollment(str(path), Enrollment(model=model, embedding=USER))
    eng = SherpaOnnxEngine(
        SherpaConfig(speaker_embedding_model=model, speaker_enroll_embedding=str(path))
    )
    eng._speaker_gate = SpeakerGate(threshold=0.5, embed_fn=lambda s, sr: None)

    with caplog.at_level(logging.WARNING, logger="speaker.sherpa"):
        eng._enroll_speaker_gate()

    assert not eng._speaker_gate.is_enrolled
    assert "deferred until the capture route/rate" in caplog.text


def test_legacy_wav_enrollment_stays_unavailable_while_process_permit_is_held(
    monkeypatch,
) -> None:
    embed_calls = []
    gate = SpeakerGate(
        threshold=0.5,
        embed_fn=lambda _samples, _sample_rate: embed_calls.append("embed") or USER,
    )
    gate.enroll_embedding(OTHER)
    assert gate.is_enrolled
    eng = SherpaOnnxEngine(
        SherpaConfig(
            speaker_embedding_model="/m/spk.onnx",
            speaker_enroll_wav="/m/legacy.wav",
        )
    )
    eng._speaker_gate = gate
    eng._capture_resolution = _capture()
    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(
            read_wave=lambda _path: (
                np.full(1_600, 0.25, dtype="float32"),
                16_000,
            )
        ),
    )
    permit = runtime_speaker_inference_permit()
    lease = permit.try_acquire()
    assert lease is not None
    try:
        eng._enroll_speaker_gate()
    finally:
        assert permit.release(lease)

    assert not gate.is_enrolled
    assert embed_calls == []


def test_legacy_wav_enrollment_reaps_returned_owned_kws_task_before_try_embed(
    monkeypatch,
) -> None:
    embed_calls = []
    gate = SpeakerGate(
        threshold=0.5,
        embed_fn=lambda _samples, _sample_rate: embed_calls.append("embed") or USER,
    )
    gate.enroll_embedding(OTHER)
    before = SpeakerGate.authority_state(gate)
    eng = SherpaOnnxEngine(
        SherpaConfig(
            speaker_embedding_model="/m/spk.onnx",
            speaker_enroll_wav="/m/legacy.wav",
        )
    )
    eng._speaker_gate = gate
    eng._capture_resolution = _capture()
    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(
            read_wave=lambda _path: (
                np.full(1_600, 0.25, dtype="float32"),
                16_000,
            )
        ),
    )
    permit = runtime_speaker_inference_permit()
    lease = permit.try_acquire()
    assert lease is not None
    returned = _ReturnedOwnedSpeakerTask(permit, lease)
    eng._kws_speaker_inference_owner = returned

    eng._enroll_speaker_gate()

    after = SpeakerGate.authority_state(gate)
    assert gate.is_enrolled
    assert after.enrollment_generation > before.enrollment_generation
    assert embed_calls == ["embed"]
    assert returned.reap_calls == 1
    assert eng._kws_speaker_inference_owner is None
    assert not permit.snapshot().active


def test_enroll_speaker_gate_loads_matching_embedding(tmp_path):
    from core.enroll import Enrollment, save_enrollment

    model = "/m/spk.onnx"
    path = tmp_path / "enroll.json"
    save_enrollment(str(path), Enrollment(model=model, embedding=USER))
    eng = SherpaOnnxEngine(
        SherpaConfig(speaker_embedding_model=model, speaker_enroll_embedding=str(path))
    )
    eng._speaker_gate = SpeakerGate(threshold=0.5, embed_fn=lambda s, sr: None)
    eng._capture_resolution = _capture()
    eng._enroll_speaker_gate()
    assert eng._speaker_gate.is_enrolled


def test_enroll_speaker_gate_ignores_mismatched_model(tmp_path):
    from core.enroll import Enrollment, save_enrollment

    path = tmp_path / "enroll.json"
    save_enrollment(str(path), Enrollment(model="/m/OTHER.onnx", embedding=USER))
    eng = SherpaOnnxEngine(
        SherpaConfig(speaker_embedding_model="/m/spk.onnx", speaker_enroll_embedding=str(path))
    )
    eng._speaker_gate = SpeakerGate(threshold=0.5, embed_fn=lambda s, sr: None)
    eng._capture_resolution = _capture()
    eng._enroll_speaker_gate()
    assert not eng._speaker_gate.is_enrolled


def test_enroll_speaker_gate_loads_matching_frontend_embedding(tmp_path):
    from core.enroll import (
        Enrollment,
        make_enrollment_frontend_provenance,
        save_enrollment,
    )

    model = "/m/spk.onnx"
    path = tmp_path / "enroll.json"
    cfg = SherpaConfig(
        speaker_embedding_model=model,
        speaker_enroll_embedding=str(path),
        denoise_enabled=True,
        denoise_model="/m/gtcrn.onnx",
    )
    eng = SherpaOnnxEngine(cfg)
    eng._denoiser = object()  # actual built processor; no model needed
    frontend = make_enrollment_frontend_provenance(
        cfg,
        input_agc=eng._input_agc,
        idle_apm=None,
        denoiser=eng._denoiser,
        apm_owns_ns=False,
        capture=_capture(),
    )
    save_enrollment(
        str(path), Enrollment(model=model, embedding=USER, frontend=frontend)
    )
    eng._speaker_gate = SpeakerGate(threshold=0.5, embed_fn=lambda s, sr: None)
    eng._capture_resolution = _capture()

    eng._enroll_speaker_gate()

    assert eng._speaker_gate.is_enrolled


def test_enroll_speaker_gate_frontend_mismatch_fails_open(tmp_path, caplog):
    import logging

    from core.enroll import Enrollment, save_enrollment

    model = "/m/spk.onnx"
    path = tmp_path / "legacy-enroll.json"
    # Legacy raw enrollment: no front-end provenance.
    save_enrollment(str(path), Enrollment(model=model, embedding=USER))
    eng = SherpaOnnxEngine(
        SherpaConfig(
            speaker_embedding_model=model,
            speaker_enroll_embedding=str(path),
            denoise_enabled=True,
            denoise_model="/m/gtcrn.onnx",
        )
    )
    eng._denoiser = object()  # runtime now sees a denoised speaker-ID domain
    eng._speaker_gate = SpeakerGate(threshold=0.5, embed_fn=lambda s, sr: None)
    eng._capture_resolution = _capture()

    with caplog.at_level(logging.WARNING, logger="speaker.sherpa"):
        eng._enroll_speaker_gate()

    assert not eng._speaker_gate.is_enrolled  # fail OPEN: no stale rejecting gate
    assert "does not match the active speaker-ID capture front end" in caplog.text
    assert "FAILS OPEN" in caplog.text
    assert "python -m core --enroll" in caplog.text


# --- loudness gate: rescue a loud near-field user when identity dips -----------


def test_loudness_rescue_admits_loud_user_when_identity_dips():
    import numpy as np

    eng = SherpaOnnxEngine(SherpaConfig(speaker_gate_input=True, input_loudness_margin_db=10.0))
    eng._speaker_gate = _gate(OTHER)   # identity REJECTS (embedding dipped / mismatch)
    eng._ambient_rms = 0.01
    loud = np.full(160, 0.5, dtype="float32")    # ~34 dB above the ambient floor
    assert eng._should_act_on_final(loud) is True   # rescued by loudness
    quiet = np.full(160, 0.012, dtype="float32")  # ~1.6 dB above floor < 10 dB margin
    assert eng._should_act_on_final(quiet) is False  # not loud enough -> dropped


def test_loudness_off_is_identity_only():
    import numpy as np

    eng = SherpaOnnxEngine(SherpaConfig(speaker_gate_input=True, input_loudness_margin_db=0.0))
    eng._speaker_gate = _gate(OTHER)
    eng._ambient_rms = 0.01
    loud = np.full(160, 0.5, dtype="float32")
    assert eng._should_act_on_final(loud) is False  # margin off -> identity-only -> dropped


def test_loudness_never_overrides_an_accepting_identity():
    import numpy as np

    eng = SherpaOnnxEngine(SherpaConfig(speaker_gate_input=True, input_loudness_margin_db=10.0))
    eng._speaker_gate = _gate(USER)  # identity ACCEPTS
    eng._ambient_rms = 0.01
    assert eng._should_act_on_final(np.zeros(160, dtype="float32")) is True  # accepted regardless of loudness


# --- L1 echo-floor gate on the FINAL-dispatch path ---------------------------
# Drops a final whose level sits at/near the device's LEARNED echo/quiet floor
# (the assistant's own residual echo / ambient noise transcribed into words) --
# the root fix for the open-speaker self-interrupt cascade (run-20260608-181250).


def test_final_floor_gate_off_by_default_admits_everything():
    import numpy as np

    eng = SherpaOnnxEngine(SherpaConfig())  # final_floor_margin_db defaults to 0.0
    eng._ambient_rms = 0.01
    eng._playback_floor_rms = 0.012
    # Disabled -> abstains (True) regardless of level, even near-silence.
    assert eng._final_above_floor(np.full(160, 0.001, dtype="float32")) is True


def test_final_floor_gate_fails_open_until_a_floor_is_learned():
    import numpy as np

    eng = SherpaOnnxEngine(SherpaConfig(final_floor_margin_db=6.0))
    # Cold start: no floor learned yet (both 0.0) -> never drop the first real turn.
    assert eng._ambient_rms == 0.0 and eng._playback_floor_rms == 0.0
    assert eng._final_above_floor(np.full(160, 0.004, dtype="float32")) is True


def test_final_floor_gate_drops_echo_passes_speech_against_playback_floor():
    import numpy as np

    eng = SherpaOnnxEngine(SherpaConfig(final_floor_margin_db=6.0))
    # Windows-style residual echo floor learned DURING playback; quiet floor lower.
    eng._ambient_rms = 0.001
    eng._playback_floor_rms = 0.012     # echo sits here -> the gate keys off max()
    # Echo-borne final (~the playback floor, e.g. the 'BEING'/'THIRTEEN' garbage):
    # < 6 dB above 0.012 -> dropped.
    assert eng._final_above_floor(np.full(160, 0.008, dtype="float32")) is False
    assert eng._final_above_floor(np.full(160, 0.018, dtype="float32")) is False
    # Real speech is many dB above the floor -> passes.
    assert eng._final_above_floor(np.full(160, 0.3, dtype="float32")) is True
    # A loud talk-over (barge) final also passes.
    assert eng._final_above_floor(np.full(160, 0.5, dtype="float32")) is True


def test_final_floor_gate_uses_the_louder_of_quiet_and_playback_floor():
    import numpy as np

    eng = SherpaOnnxEngine(SherpaConfig(final_floor_margin_db=6.0))
    # Quiet floor higher than the playback floor (noisy room, good AEC): the gate
    # must use the MAX so a final near the quiet floor is still treated as ambient.
    eng._ambient_rms = 0.02
    eng._playback_floor_rms = 0.002
    assert eng._final_above_floor(np.full(160, 0.025, dtype="float32")) is False  # ~1.9 dB < 6
    assert eng._final_above_floor(np.full(160, 0.2, dtype="float32")) is True


# --- L2 post-speaking refractory ---------------------------------------------


def test_post_speaking_refractory_active_right_after_speaking_clears():
    import time

    eng = SherpaOnnxEngine(SherpaConfig(barge_in_refractory_sec=0.5))
    now = time.monotonic()
    eng._last_speaking_end = now            # just stopped speaking
    assert eng._in_post_speaking_refractory(now) is True
    assert eng._in_post_speaking_refractory(now + 0.4) is True
    assert eng._in_post_speaking_refractory(now + 0.6) is False  # window expired


def test_post_speaking_refractory_disabled_when_zero():
    import time

    eng = SherpaOnnxEngine(SherpaConfig(barge_in_refractory_sec=0.0))
    now = time.monotonic()
    eng._last_speaking_end = now
    assert eng._in_post_speaking_refractory(now) is False  # off-switch parity


def test_post_speaking_refractory_inert_before_any_speech():
    import time

    eng = SherpaOnnxEngine(SherpaConfig(barge_in_refractory_sec=0.5))
    # _last_speaking_end is still its 0.0 init -> the deadline (0.5) is far in the
    # past relative to a real monotonic clock, so the refractory is inert.
    assert eng._in_post_speaking_refractory(time.monotonic()) is False
