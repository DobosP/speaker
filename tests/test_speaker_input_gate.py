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


def test_unenrolled_input_gate_fails_open():
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
