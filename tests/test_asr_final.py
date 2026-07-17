"""Two-pass ASR: optional SenseVoice/Whisper/NeMo offline FINAL recognition.

These tests pin the wiring/fallback logic with fakes -- no models/audio.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from core.engine import EngineCallbacks
from core.engines._sherpa_models import (
    build_final_recognizer,
    build_final_verifier,
)
from core.engines._asr_segment import ASRSegment
from core.engines.sherpa import (
    SherpaConfig,
    SherpaOnnxEngine,
    _resolve_final_transcript,
    _transcribe_final_text,
)


class _FakeStream:
    def __init__(self, text):
        self.result = SimpleNamespace(text=text)

    def accept_waveform(self, sr, a):
        pass


class _FakeOffline:
    """A stand-in OfflineRecognizer: returns a fixed text for any audio."""

    def __init__(self, text):
        self._text = text

    def create_stream(self):
        return _FakeStream(self._text)

    def decode_stream(self, stream):
        pass


class _FakeVerifier:
    def __init__(self, text="", *, error=None):
        self._text = text
        self._error = error
        self.calls = []

    def transcribe(self, samples, sample_rate):
        self.calls.append((samples, sample_rate))
        if self._error is not None:
            raise self._error
        return SimpleNamespace(text=self._text)


def _engine(**sherpa):
    return SherpaOnnxEngine(SherpaConfig.from_dict(sherpa))


# --- build_final_recognizer (fail-open) ---------------------------------------


def test_build_final_recognizer_none_without_backend():
    assert build_final_recognizer(SherpaConfig.from_dict({})) is None


def test_build_final_recognizer_none_when_model_missing():
    cfg = SherpaConfig.from_dict({"asr_final_backend": "sense_voice",
                                  "asr_final_model": "/does/not/exist.onnx"})
    assert build_final_recognizer(cfg) is None  # graceful, no crash


def test_build_final_recognizer_wires_nemo_transducer_contract(monkeypatch):
    import os

    import sherpa_onnx

    captured: dict = {}

    def _fake(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(sherpa_onnx.OfflineRecognizer, "from_transducer", _fake)
    monkeypatch.setattr(os.path, "exists", lambda _path: True)
    cfg = SherpaConfig.from_dict(
        {
            "sample_rate": 16000,
            "provider": "cpu",
            "asr_num_threads": 3,
            "asr_final_backend": "nemo_transducer",
            "asr_final_model": "/model/encoder.int8.onnx",
            "asr_final_decoder": "/model/decoder.int8.onnx",
            "asr_final_joiner": "/model/joiner.int8.onnx",
            "asr_final_tokens": "/model/tokens.txt",
        }
    )

    assert build_final_recognizer(cfg) is not None
    assert captured == {
        "encoder": "/model/encoder.int8.onnx",
        "decoder": "/model/decoder.int8.onnx",
        "joiner": "/model/joiner.int8.onnx",
        "tokens": "/model/tokens.txt",
        "num_threads": 3,
        "sample_rate": 16000,
        "feature_dim": 80,
        "decoding_method": "greedy_search",
        "max_active_paths": 4,
        "provider": "cpu",
        "model_type": "nemo_transducer",
    }


def test_build_final_verifier_is_opt_in_and_requires_supported_local_model(
    monkeypatch,
    tmp_path,
):
    import core.engines._faster_whisper as faster_whisper_adapter

    built = []

    class _Recognizer:
        def __init__(self, model_path):
            built.append(model_path)

    monkeypatch.setattr(
        faster_whisper_adapter,
        "FasterWhisperEndpointRecognizer",
        _Recognizer,
    )

    assert build_final_verifier(SherpaConfig()) is None
    verifier = build_final_verifier(
        SherpaConfig(
            asr_final_verifier_backend="faster_whisper",
            asr_final_verifier_model=str(tmp_path),
        )
    )
    assert isinstance(verifier, _Recognizer)
    assert built == [str(tmp_path)]
    assert build_final_verifier(
        SherpaConfig(
            asr_final_verifier_backend="remote_api",
            asr_final_verifier_model=str(tmp_path),
        )
    ) is None


def test_build_final_verifier_fails_open_for_nonlocal_model_identifier():
    assert build_final_verifier(
        SherpaConfig(
            asr_final_verifier_backend="faster_whisper",
            asr_final_verifier_model="org/model-that-is-not-a-local-directory",
        )
    ) is None


# --- _final_transcribe (second pass vs fallback) ------------------------------


def test_final_transcribe_uses_second_pass_when_present():
    eng = _engine()
    eng._final_recognizer = _FakeOffline("Hey, are you listening to me.")
    out = eng._final_transcribe(np.ones(16000, dtype="float32"), "HEY IRIC LISTENING TO ME")
    assert out == "Hey, are you listening to me."  # the clean second-pass text


def test_final_decision_exposes_raw_hypotheses_and_preserves_selected_text():
    cfg = SherpaConfig()
    offline = _FakeOffline("  are you there  ")
    segment = np.ones(2 * 16000, dtype="float32")

    decision = _resolve_final_transcript(
        cfg,
        offline,
        None,
        segment,
        "Ario der",
    )

    assert decision.streaming_raw == "Ario der"
    assert decision.offline_raw == "  are you there  "
    assert decision.selected == "are you there"
    assert _transcribe_final_text(
        cfg,
        offline,
        None,
        segment,
        "Ario der",
    ) == decision.selected


def test_final_decision_hypotheses_are_excluded_from_repr():
    decision = _resolve_final_transcript(
        SherpaConfig(),
        _FakeOffline("private offline hypothesis"),
        None,
        np.ones(16000, dtype="float32"),
        "private streaming hypothesis",
    )

    rendered = repr(decision)
    assert "private streaming hypothesis" not in rendered
    assert "private offline hypothesis" not in rendered
    assert decision.selected not in rendered


def test_exact_offline_verifier_consensus_changes_baseline_on_identical_pcm():
    accepted = {}

    class _Offline:
        def create_stream(self):
            stream = _FakeStream("  private corrected phrase  ")

            def accept(sample_rate, samples):
                accepted["offline"] = (samples, sample_rate)

            stream.accept_waveform = accept
            return stream

        def decode_stream(self, stream):
            del stream

    verifier = _FakeVerifier("private corrected phrase")
    segment = np.ones(2 * 16000, dtype="float32")

    decision = _resolve_final_transcript(
        SherpaConfig(),
        _Offline(),
        None,
        segment,
        "unrelated private streaming words",
        final_verifier=verifier,
    )

    verifier_pcm, verifier_rate = verifier.calls[0]
    offline_pcm, offline_rate = accepted["offline"]
    assert verifier_pcm is offline_pcm
    assert verifier_pcm is segment
    assert verifier_rate == offline_rate == 16000
    assert decision.selected == "private corrected phrase"
    assert decision.verifier_outcome == "consensus"
    assert decision.verifier_support == 2
    assert decision.verifier_changed is True
    rendered = repr(decision)
    assert "private corrected phrase" not in rendered
    assert "unrelated private streaming words" not in rendered


def test_nemo_exact_offline_verifier_quorum_changes_established_streaming_final():
    decision = _resolve_final_transcript(
        SherpaConfig(asr_final_backend="nemo_transducer"),
        _FakeOffline("are you there"),
        None,
        np.ones(2 * 16000, dtype="float32"),
        "Ario der",
        final_verifier=_FakeVerifier("are you there"),
    )

    assert decision.selected == "are you there"
    assert decision.verifier_outcome == "consensus"
    assert decision.verifier_support == 2
    assert decision.verifier_changed is True


@pytest.mark.parametrize(
    ("verifier", "expected_outcome", "expected_support"),
    [
        (None, "unavailable", 0),
        (_FakeVerifier("different words"), "tie", 1),
        (_FakeVerifier(""), "empty", 0),
        (_FakeVerifier(error=RuntimeError("private verifier failure")), "error", 0),
    ],
)
def test_nemo_offline_is_evidence_only_without_exact_quorum(
    verifier,
    expected_outcome,
    expected_support,
):
    decision = _resolve_final_transcript(
        SherpaConfig(asr_final_backend="nemo_transducer"),
        _FakeOffline("are you there"),
        None,
        np.ones(2 * 16000, dtype="float32"),
        "Ario der",
        final_verifier=verifier,
    )

    assert decision.offline_outcome == "decoded"
    assert decision.selected == "Ario der"
    assert decision.verifier_outcome == expected_outcome
    assert decision.verifier_support == expected_support
    assert decision.verifier_changed is False
    assert "private verifier failure" not in repr(decision)


@pytest.mark.parametrize(
    ("verifier", "expected_outcome", "expected_support"),
    [
        (_FakeVerifier(""), "empty", 0),
        (_FakeVerifier("independent unique hypothesis"), "tie", 1),
        (_FakeVerifier(error=RuntimeError("private model failure")), "error", 0),
    ],
)
def test_verifier_empty_error_or_nonconsensus_preserves_baseline(
    verifier,
    expected_outcome,
    expected_support,
):
    decision = _resolve_final_transcript(
        SherpaConfig(),
        None,
        None,
        np.ones(16000, dtype="float32"),
        "keep baseline",
        final_verifier=verifier,
    )

    assert decision.selected == "Keep baseline"
    assert decision.verifier_outcome == expected_outcome
    assert decision.verifier_support == expected_support
    assert decision.verifier_changed is False
    assert "private model failure" not in repr(decision)


def test_two_decoded_empty_models_veto_one_word_noncontrol_final():
    decision = _resolve_final_transcript(
        SherpaConfig(),
        _FakeOffline(""),
        None,
        np.ones(16000, dtype="float32"),
        "ARTIFACT",
        final_verifier=_FakeVerifier(""),
    )

    assert decision.selected == ""
    assert decision.offline_outcome == "empty"
    assert decision.verifier_outcome == "empty_veto"
    assert decision.verifier_support == 2
    assert decision.verifier_changed is True


def test_two_decoded_empty_models_cannot_veto_stop():
    decision = _resolve_final_transcript(
        SherpaConfig(),
        _FakeOffline(""),
        None,
        np.ones(16000, dtype="float32"),
        "STOP",
        final_verifier=_FakeVerifier(""),
    )

    assert decision.selected == "Stop"
    assert decision.verifier_outcome == "control_guard"
    assert decision.verifier_support == 2
    assert decision.verifier_changed is False


def test_nemo_two_model_consensus_recovers_attested_long_stop():
    decision = _resolve_final_transcript(
        SherpaConfig(asr_final_backend="nemo_transducer"),
        _FakeOffline("Stop speaking."),
        None,
        np.ones(2 * 16000, dtype="float32"),
        "DON'T PLAY SPEAK",
        final_verifier=_FakeVerifier("stop speaking"),
        speech_sec=1.5,
    )

    assert decision.selected == "Stop speaking."
    assert decision.verifier_outcome == "attested_control"
    assert decision.verifier_support == 2
    assert decision.verifier_changed is True


@pytest.mark.parametrize(
    ("backend", "verifier_text", "speech_sec"),
    [
        ("nemo_transducer", "different words", 1.5),
        ("nemo_transducer", "stop speaking", None),
        ("nemo_transducer", "stop speaking", 1.2),
        ("nemo_transducer", "stop speaking", 2.1),
        ("whisper", "stop speaking", 1.5),
    ],
)
def test_attested_nemo_control_requires_two_exact_votes_and_owned_timing(
    backend,
    verifier_text,
    speech_sec,
):
    decision = _resolve_final_transcript(
        SherpaConfig(asr_final_backend=backend),
        _FakeOffline("Stop speaking."),
        None,
        np.ones(2 * 16000, dtype="float32"),
        "DON'T PLAY SPEAK",
        final_verifier=_FakeVerifier(verifier_text),
        speech_sec=speech_sec,
    )

    assert decision.selected == "Don't play speak"
    assert decision.verifier_outcome != "attested_control"
    assert decision.verifier_changed is False


def test_live_decode_error_circuit_breaks_verifier_and_emits_one_metric():
    engine = _engine()
    verifier = _FakeVerifier(error=RuntimeError("private model failure"))
    metrics = []
    finals = []
    engine._final_verifier = verifier
    engine._final_above_floor = lambda _samples: True
    engine._cb = EngineCallbacks(
        on_metric=lambda name, *_args, **_kwargs: metrics.append(name),
        on_final=finals.append,
    )

    engine._finalize_and_dispatch(
        np.ones(16000, dtype="float32"),
        "keep baseline",
        time.perf_counter(),
    )

    assert engine._final_verifier is None
    assert finals == ["Keep baseline"]
    assert metrics.count("asr_final_verifier_disabled_after_decode_error") == 1
    assert len(verifier.calls) == 1

    engine._finalize_and_dispatch(
        np.ones(16000, dtype="float32"),
        "next baseline",
        time.perf_counter(),
    )
    assert finals[-1] == "Next baseline"
    assert len(verifier.calls) == 1
    assert metrics.count("asr_final_verifier_disabled_after_decode_error") == 1


def test_nemo_verifier_circuit_break_never_promotes_offline_fallback():
    engine = SherpaOnnxEngine(SherpaConfig(asr_final_backend="nemo_transducer"))
    verifier = _FakeVerifier(error=RuntimeError("private model failure"))
    finals = []
    engine._final_recognizer = _FakeOffline("are you there")
    engine._final_verifier = verifier
    engine._final_above_floor = lambda _samples: True
    engine._cb = EngineCallbacks(on_final=finals.append)

    engine._finalize_and_dispatch(
        np.ones(2 * 16000, dtype="float32"),
        "Ario der",
        time.perf_counter(),
    )
    assert engine._final_verifier is None
    assert finals == ["Ario der"]

    engine._finalize_and_dispatch(
        np.ones(2 * 16000, dtype="float32"),
        "keep next baseline",
        time.perf_counter(),
    )
    assert finals[-1] == "Keep next baseline"
    assert len(verifier.calls) == 1


def test_verifier_cannot_manufacture_turn_from_empty_streaming_without_authority():
    segment = np.ones(16000, dtype="float32")
    offline = _FakeOffline("ordinary recovered phrase")

    blocked = _resolve_final_transcript(
        SherpaConfig(),
        offline,
        None,
        segment,
        "",
        final_verifier=_FakeVerifier("ordinary recovered phrase"),
    )
    authorized = _resolve_final_transcript(
        SherpaConfig(),
        offline,
        None,
        segment,
        "",
        final_verifier=_FakeVerifier("ordinary recovered phrase"),
        allow_empty_streaming=True,
    )

    assert blocked.selected == ""
    assert blocked.verifier_outcome == "empty_streaming_guard"
    assert blocked.verifier_support == 2
    assert blocked.verifier_changed is False
    assert authorized.selected == "ordinary recovered phrase"
    assert authorized.verifier_outcome == "consensus"


def test_single_verifier_vote_has_no_quorum_and_preserves_empty_baseline():
    decision = _resolve_final_transcript(
        SherpaConfig(),
        None,
        None,
        np.ones(16000, dtype="float32"),
        "",
        final_verifier=_FakeVerifier("unsupported recovered phrase"),
    )

    assert decision.selected == ""
    assert decision.verifier_outcome == "no_quorum"
    assert decision.verifier_support == 1
    assert decision.verifier_changed is False


def test_final_decision_keeps_empty_streaming_fail_closed():
    cfg = SherpaConfig()
    segment = np.ones(16000, dtype="float32")

    decision = _resolve_final_transcript(
        cfg,
        _FakeOffline("Stop speaking."),
        None,
        segment,
        "",
    )

    assert decision.streaming_raw == ""
    assert decision.offline_raw == "Stop speaking."
    assert decision.selected == ""
    assert _transcribe_final_text(
        cfg,
        _FakeOffline("Stop speaking."),
        None,
        segment,
        "",
    ) == ""


def test_final_transcribe_falls_back_without_second_pass():
    eng = _engine()
    assert eng._final_recognizer is None
    # falls back to _postprocess_final (casing) of the streaming raw
    assert eng._final_transcribe(None, "hello world") == "Hello world"


def test_final_transcribe_empty_second_pass_falls_back():
    eng = _engine()
    eng._final_recognizer = _FakeOffline("")  # empty result -> use the streaming final
    assert eng._final_transcribe(np.ones(16000, dtype="float32"), "hello world") == "Hello world"


def test_final_transcribe_empty_streaming_fails_closed_by_default():
    eng = _engine()
    eng._final_recognizer = _FakeOffline("Stop speaking.")
    seg = np.ones(16000, dtype="float32")

    assert eng._final_transcribe(seg, "") == ""
    assert (
        eng._final_transcribe(seg, "", allow_empty_streaming=True)
        == "Stop speaking."
    )


def test_final_transcribe_min_sec_skips_short_utterance():
    eng = _engine(asr_final_min_sec=2.0)
    eng._final_recognizer = _FakeOffline("Hi there.")
    # 0.5s of audio < the 2.0s floor -> skip the second pass, fall back.
    out = eng._final_transcribe(np.ones(8000, dtype="float32"), "hi there")
    assert out != "Hi there."
    # ...but a long-enough utterance DOES use it.
    long_out = eng._final_transcribe(np.ones(2 * 16000, dtype="float32"), "hi there")
    assert long_out == "Hi there."


def test_final_transcribe_recovers_from_second_pass_error():
    class _Boom:
        def create_stream(self):
            raise RuntimeError("model exploded")

    eng = _engine()
    eng._final_recognizer = _Boom()
    # a second-pass failure must never lose the turn -> the streaming final stands.
    assert eng._final_transcribe(np.ones(16000, dtype="float32"), "hello world") == "Hello world"


def test_committed_config_defaults_to_sense_voice_at_standard_path():
    # The shipped default is the two-pass at the standard setup_models location:
    # present -> active; absent -> build_final_recognizer returns None (streaming,
    # byte-identical). Don't regress this default.
    import json
    from pathlib import Path

    c = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())["sherpa"]
    assert c["asr_final_backend"] == "sense_voice"
    assert c["asr_final_model"].endswith("sense_voice/model.int8.onnx")
    assert c["asr_final_tokens"].endswith("sense_voice/tokens.txt")


# --- L3: short-clip 2nd-pass hallucination rejected via the agreement guard ----


def test_final_transcribe_rejects_short_hallucination():
    # The Windows cascade trigger (run-20260608-181250): a short open-speaker echo
    # clip the streaming pass heard as 'BEING' and the SenseVoice 2nd pass HALLUCINATED
    # into 'I.'. _final_transcribe must route through agreement_guard (short clip, no
    # shared content token) and keep the streaming final, NOT emit the invented 'I.'.
    eng = _engine()
    eng._final_recognizer = _FakeOffline("I.")
    seg = np.ones(int(0.4 * 16000), dtype="float32")  # 0.4s < short_sec -> guarded
    out = eng._final_transcribe(seg, "BEING")
    assert out != "I."
    assert "eing" in out.lower()  # the (post-processed) streaming final stands


def test_final_transcribe_keeps_long_garbled_correction():
    # The legit case the 2nd pass exists for: a real, longer utterance the streaming
    # pass mangled ('Ario der' -> 'are you there'), near-zero token overlap but
    # high phrase similarity. The guard must not regress it.
    eng = _engine()
    eng._final_recognizer = _FakeOffline("are you there")
    seg = np.ones(2 * 16000, dtype="float32")  # 2.0s -> not short
    assert eng._final_transcribe(seg, "Ario der") == "are you there"


def test_final_transcribe_uses_vad_speech_duration_not_idle_padded_pcm_length():
    eng = _engine()
    eng._final_recognizer = _FakeOffline("are you there")
    # The owned clip may contain pre-roll/tail, but only 0.4 s was real speech.
    # Treat it as a short clip so the low-overlap second pass cannot take the
    # long-utterance escape hatch merely because padding made the array 2 s.
    seg = np.ones(2 * 16000, dtype="float32")
    out = eng._final_transcribe(seg, "Ario der", speech_sec=0.4)
    assert out == "Ario der"


# --- exact owner-attested short interrupt repair -----------------------------


def test_sense_voice_recovers_attested_short_cancel_with_owned_speech_duration():
    eng = _engine(asr_final_backend="sense_voice")
    eng._final_recognizer = _FakeOffline("Cancel that.")
    # Owned PCM includes pre-roll/tail and is deliberately long. Independent
    # live-segment timing proves the spoken command itself was short.
    seg = np.ones(2 * 16000, dtype="float32")
    assert (
        eng._final_transcribe(seg, "CASTLE DEATH", speech_sec=0.6)
        == "Cancel that."
    )


def test_attested_short_repair_requires_explicit_owned_speech_duration():
    eng = _engine(asr_final_backend="sense_voice")
    eng._final_recognizer = _FakeOffline("Cancel that.")
    seg = np.ones(int(0.6 * 16000), dtype="float32")
    assert eng._final_transcribe(seg, "CASTLE DEATH") == "Castle death"


def test_attested_short_repair_accepts_confirmed_word_cut_owned_timing():
    segment = ASRSegment(
        sample_rate=16000,
        pre_roll_sec=0.8,
        max_utterance_sec=3.0,
        vad_available=True,
        block_sec=0.1,
    )
    # Word-cut confirmation owns this playback-time user PCM before a post-cut
    # VAD observation. Its bounded timestamps are valid speech timing, not a
    # padded-array-duration fallback.
    segment.prepend(
        [np.ones(1600, dtype="float32") for _ in range(3)],
        speech_at=4.0,
        speech_end_at=4.2,
    )
    owned, _ = segment.arrays()
    assert segment.speech_duration_sec is not None

    eng = _engine(asr_final_backend="sense_voice")
    eng._final_recognizer = _FakeOffline("Cancel that.")
    assert (
        eng._final_transcribe(
            owned,
            "CASTLE DEATH",
            speech_sec=segment.speech_duration_sec,
        )
        == "Cancel that."
    )


def test_attested_short_repair_requires_sense_voice_backend():
    eng = _engine(asr_final_backend="whisper")
    eng._final_recognizer = _FakeOffline("Cancel that.")
    seg = np.ones(16000, dtype="float32")
    assert (
        eng._final_transcribe(seg, "CASTLE DEATH", speech_sec=0.6)
        == "Castle death"
    )


def test_attested_short_repair_rejects_long_vad_utterance_at_boundary():
    eng = _engine(asr_final_backend="sense_voice")
    eng._final_recognizer = _FakeOffline("Cancel that.")
    seg = np.ones(2 * 16000, dtype="float32")
    assert (
        eng._final_transcribe(seg, "CASTLE DEATH", speech_sec=1.2)
        == "Castle death"
    )


@pytest.mark.parametrize("speech_sec", [1.4, 1.5, 1.9, 2.0])
def test_sense_voice_recovers_attested_long_stop_with_owned_speech_duration(
    speech_sec,
):
    eng = _engine(asr_final_backend="sense_voice")
    eng._final_recognizer = _FakeOffline("Stop speaking.")
    seg = np.ones(2 * 16000, dtype="float32")
    assert (
        eng._final_transcribe(seg, "DON'T PLAY SPEAK", speech_sec=speech_sec)
        == "Stop speaking."
    )


def test_attested_long_stop_repair_rejects_missing_or_short_owned_duration():
    eng = _engine(asr_final_backend="sense_voice")
    eng._final_recognizer = _FakeOffline("Stop speaking.")
    seg = np.ones(2 * 16000, dtype="float32")
    assert eng._final_transcribe(seg, "DON'T PLAY SPEAK") == "Don't play speak"
    assert (
        eng._final_transcribe(seg, "DON'T PLAY SPEAK", speech_sec=0.9)
        == "Don't play speak"
    )


@pytest.mark.parametrize("speech_sec", [-1.0, 0.0, 1.2, 1.399, 2.001, 30.0])
def test_attested_long_stop_repair_rejects_out_of_range_owned_duration(
    speech_sec,
):
    eng = _engine(asr_final_backend="sense_voice")
    eng._final_recognizer = _FakeOffline("Stop speaking.")
    seg = np.ones(2 * 16000, dtype="float32")
    assert (
        eng._final_transcribe(seg, "DON'T PLAY SPEAK", speech_sec=speech_sec)
        == "Don't play speak"
    )


@pytest.mark.parametrize("speech_sec", [float("nan"), float("inf"), float("-inf")])
def test_attested_long_stop_repair_rejects_nonfinite_owned_duration(speech_sec):
    eng = _engine(asr_final_backend="sense_voice")
    eng._final_recognizer = _FakeOffline("Stop speaking.")
    seg = np.ones(2 * 16000, dtype="float32")
    assert (
        eng._final_transcribe(seg, "DON'T PLAY SPEAK", speech_sec=speech_sec)
        == "Don't play speak"
    )


def test_attested_long_stop_repair_requires_sense_voice_backend():
    eng = _engine(asr_final_backend="whisper")
    eng._final_recognizer = _FakeOffline("Stop speaking.")
    seg = np.ones(2 * 16000, dtype="float32")
    assert (
        eng._final_transcribe(seg, "DON'T PLAY SPEAK", speech_sec=1.9)
        == "Don't play speak"
    )


def test_attested_short_repair_rejects_unlisted_rewrites():
    eng = _engine(asr_final_backend="sense_voice")
    seg = np.ones(16000, dtype="float32")

    eng._final_recognizer = _FakeOffline("Cancel that.")
    assert eng._final_transcribe(seg, "BEING", speech_sec=0.6) == "Being"

    eng._final_recognizer = _FakeOffline("Okay.")
    assert eng._final_transcribe(seg, "CASTLE DEATH", speech_sec=0.6) == "Castle death"

    eng._final_recognizer = _FakeOffline("Cancel that.")
    assert (
        eng._final_transcribe(seg, "CASTLE 123 DEATH", speech_sec=0.6)
        == "Castle 123 death"
    )

    eng._final_recognizer = _FakeOffline("Cancel that system.")
    assert eng._final_transcribe(seg, "CASTLE DEATH", speech_sec=0.6) == "Castle death"

    eng._final_recognizer = _FakeOffline("Cancel that 系统.")
    assert eng._final_transcribe(seg, "CASTLE DEATH", speech_sec=0.6) == "Castle death"

    eng._final_recognizer = _FakeOffline("Cancel that.")
    out = eng._final_transcribe(seg, "系统 CASTLE DEATH", speech_sec=0.6)
    assert out != "Cancel that." and "系统" in out


def test_no_vad_segment_uses_owned_pcm_duration_for_second_pass():
    segment = ASRSegment(
        sample_rate=16000,
        pre_roll_sec=0.8,
        max_utterance_sec=3.0,
        vad_available=False,
        block_sec=0.1,
    )
    for _ in range(20):  # speech began two seconds before the first partial
        segment.append(np.ones(1600, dtype="float32"))
    segment.observe_text(2.0)
    owned, _ = segment.arrays()

    eng = _engine(asr_final_min_sec=0.5)
    eng._final_recognizer = _FakeOffline("Garbled stream cleaned.")
    assert segment.speech_duration_sec is None
    assert eng._final_transcribe(
        owned,
        "garbled stream",
        speech_sec=segment.speech_duration_sec,
    ) == "Garbled stream cleaned."


def test_config_parses_final_fields():
    c = SherpaConfig.from_dict({
        "asr_final_backend": "sense_voice", "asr_final_model": "/m.onnx",
        "asr_final_tokens": "/t.txt", "asr_final_decoder": "/d.onnx",
        "asr_final_joiner": "/j.onnx", "asr_final_use_itn": False,
        "asr_final_min_sec": 0.5,
        "asr_final_preroll_sec": 0.6,
        "asr_final_verifier_backend": "faster_whisper",
        "asr_final_verifier_model": "/cached/model",
    })
    assert c.asr_final_backend == "sense_voice" and c.asr_final_model == "/m.onnx"
    assert c.asr_final_decoder == "/d.onnx" and c.asr_final_joiner == "/j.onnx"
    assert c.asr_final_use_itn is False and c.asr_final_min_sec == 0.5
    assert c.asr_final_preroll_sec == 0.6
    assert c.asr_final_verifier_backend == "faster_whisper"
    assert c.asr_final_verifier_model == "/cached/model"


def _capture_sense_voice(monkeypatch):
    """Patch from_sense_voice to capture kwargs + make the model 'exist'."""
    import os

    import sherpa_onnx

    captured: dict = {}

    def _fake(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(sherpa_onnx.OfflineRecognizer, "from_sense_voice", _fake)
    monkeypatch.setattr(os.path, "exists", lambda _p: True)
    return captured


def test_build_final_recognizer_wires_homophone_replacement(monkeypatch):
    # The hr_* / rule_fsts fields are the ONLY contextual biasing that reaches the
    # SenseVoice second-pass final (asr_hotwords biases only the streaming pass).
    captured = _capture_sense_voice(monkeypatch)
    cfg = SherpaConfig.from_dict({
        "asr_final_backend": "sense_voice",
        "asr_final_model": "/fake/model.onnx",
        "asr_final_hr_dict_dir": "/hr/dict",
        "asr_final_hr_lexicon": "/hr/lexicon.txt",
        "asr_final_rule_fsts": "/rules.fst",
    })
    assert build_final_recognizer(cfg) is not None
    assert captured["hr_dict_dir"] == "/hr/dict"
    assert captured["hr_lexicon"] == "/hr/lexicon.txt"
    assert captured["rule_fsts"] == "/rules.fst"
    assert "hr_rule_fsts" not in captured       # empty field -> not passed


def test_build_final_recognizer_omits_hr_when_unset(monkeypatch):
    # Byte-identical when unconfigured: none of the hr_/rule keys are passed.
    captured = _capture_sense_voice(monkeypatch)
    cfg = SherpaConfig.from_dict({
        "asr_final_backend": "sense_voice", "asr_final_model": "/fake/model.onnx",
    })
    assert build_final_recognizer(cfg) is not None
    for k in ("hr_dict_dir", "hr_lexicon", "hr_rule_fsts", "rule_fsts"):
        assert k not in captured
