"""Tests for gating normal ASR finals on speaker identity (input gating).

These exercise SherpaOnnxEngine._should_act_on_final and _enroll_speaker_gate
directly with an injected gate -- no sherpa-onnx, no models, no audio device.
The capture loop's threaded I/O is out of scope; the decision logic is not.
"""
from __future__ import annotations

from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from core.engines.speaker_gate import SpeakerGate

USER = [1.0, 0.0, 0.0]
OTHER = [0.0, 1.0, 0.0]


def _gate(embed, *, enrolled_to=USER):
    g = SpeakerGate(threshold=0.5, embed_fn=lambda samples, sr: embed)
    if enrolled_to is not None:
        g.enroll_embedding(enrolled_to)
    return g


def _engine(*, gate_input=True, gate=None):
    eng = SherpaOnnxEngine(SherpaConfig(speaker_gate_input=gate_input))
    eng._speaker_gate = gate
    return eng


def test_input_gating_disabled_acts_even_with_rejecting_gate():
    eng = _engine(gate_input=False, gate=_gate(OTHER))  # gate would reject
    assert eng._should_act_on_final([0.0]) is True


def test_no_gate_fails_open():
    assert _engine(gate=None)._should_act_on_final([0.0]) is True


def test_unenrolled_gate_fails_open():
    eng = _engine(gate=_gate(OTHER, enrolled_to=None))
    assert not eng._speaker_gate.is_enrolled
    assert eng._should_act_on_final([0.0]) is True


def test_enrolled_user_final_is_acted_on():
    assert _engine(gate=_gate(USER))._should_act_on_final([0.0]) is True


def test_enrolled_other_voice_final_is_dropped():
    assert _engine(gate=_gate(OTHER))._should_act_on_final([0.0]) is False


# --- _enroll_speaker_gate: load the persisted embedding into the gate --------


def test_enroll_speaker_gate_loads_matching_embedding(tmp_path):
    from core.enroll import Enrollment, save_enrollment

    model = "/m/spk.onnx"
    path = tmp_path / "enroll.json"
    save_enrollment(str(path), Enrollment(model=model, embedding=USER))
    eng = SherpaOnnxEngine(
        SherpaConfig(speaker_embedding_model=model, speaker_enroll_embedding=str(path))
    )
    eng._speaker_gate = SpeakerGate(threshold=0.5, embed_fn=lambda s, sr: None)
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
    eng._enroll_speaker_gate()
    assert not eng._speaker_gate.is_enrolled


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
