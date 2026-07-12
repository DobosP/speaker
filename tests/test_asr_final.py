"""Two-pass ASR: the optional offline second-pass FINAL recognizer (SenseVoice/
Whisper) that re-transcribes the endpointed utterance for the text that reaches
the LLM. These pin the wiring/fallback logic with fakes -- no models/audio."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.engines._sherpa_models import build_final_recognizer
from core.engines._asr_segment import ASRSegment
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine


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


def _engine(**sherpa):
    return SherpaOnnxEngine(SherpaConfig.from_dict(sherpa))


# --- build_final_recognizer (fail-open) ---------------------------------------


def test_build_final_recognizer_none_without_backend():
    assert build_final_recognizer(SherpaConfig.from_dict({})) is None


def test_build_final_recognizer_none_when_model_missing():
    cfg = SherpaConfig.from_dict({"asr_final_backend": "sense_voice",
                                  "asr_final_model": "/does/not/exist.onnx"})
    assert build_final_recognizer(cfg) is None  # graceful, no crash


# --- _final_transcribe (second pass vs fallback) ------------------------------


def test_final_transcribe_uses_second_pass_when_present():
    eng = _engine()
    eng._final_recognizer = _FakeOffline("Hey, are you listening to me.")
    out = eng._final_transcribe(np.ones(16000, dtype="float32"), "HEY IRIC LISTENING TO ME")
    assert out == "Hey, are you listening to me."  # the clean second-pass text


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
        "asr_final_tokens": "/t.txt", "asr_final_use_itn": False, "asr_final_min_sec": 0.5,
        "asr_final_preroll_sec": 0.6,
    })
    assert c.asr_final_backend == "sense_voice" and c.asr_final_model == "/m.onnx"
    assert c.asr_final_use_itn is False and c.asr_final_min_sec == 0.5
    assert c.asr_final_preroll_sec == 0.6


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
