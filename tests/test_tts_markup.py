"""Tests for the opt-in TTS expressive-markup capability (emotion + voice
diversity). Two layers:

* the pure helpers in ``core.tts_markup`` (parse + resolve) -- exhaustive, no deps;
* the engine wiring -- ``SherpaOnnxEngine.speak() -> _synthesize() -> generate(sid,
  speed)`` -- exercised on a bare instance (``object.__new__``) with a fake TTS, so
  the directive path is covered without starting threads, models, or a sound card.

The contract: default OFF is byte-identical (no parsing, static sid/speed); when
on, a leading ``[emotion:.. voice:.. rate:..]`` tag is stripped from the spoken
text and mapped to a per-utterance ``(sid, speed)``, fail-soft and clamped.
"""
from __future__ import annotations

import queue
import threading

import numpy as np
import pytest

from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from core.persona import build_system_prompt
from core.tts_markup import (
    build_markup_guidance,
    parse_tts_markup,
    resolve_tts_params,
)


# --- parse_tts_markup -----------------------------------------------------

def test_parse_no_tag_is_passthrough():
    assert parse_tts_markup("Just a normal sentence.") == ("Just a normal sentence.", {})


def test_parse_empty_and_none_safe():
    assert parse_tts_markup("") == ("", {})
    assert parse_tts_markup(None) == (None, {})  # type: ignore[arg-type]


def test_parse_extracts_and_strips_leading_tag():
    text, d = parse_tts_markup("[emotion:calm voice:warm] Here is a gentle line.")
    assert text == "Here is a gentle line."
    assert d == {"emotion": "calm", "voice": "warm"}


def test_parse_accepts_equals_and_commas():
    text, d = parse_tts_markup("[voice=narrator, rate=1.05] And the exciting part!")
    assert text == "And the exciting part!"
    assert d == {"voice": "narrator", "rate": "1.05"}


def test_parse_footnote_bracket_is_not_a_tag():
    # A leading bracket with NONE of the known keys is ordinary text, untouched.
    assert parse_tts_markup("[1] First reference.") == ("[1] First reference.", {})
    assert parse_tts_markup("[see note] hello") == ("[see note] hello", {})


def test_parse_only_keeps_known_keys():
    text, d = parse_tts_markup("[voice:warm bogus:x emotion:sad] hi")
    assert text == "hi"
    assert d == {"voice": "warm", "emotion": "sad"}  # 'bogus' dropped


def test_parse_tag_only_emission():
    text, d = parse_tts_markup("[emotion:calm]")
    assert text == ""
    assert d == {"emotion": "calm"}


def test_parse_only_consumes_one_leading_tag():
    # A bracket later in the sentence is real text and is left alone.
    text, d = parse_tts_markup("[voice:warm] the array is a[3] not a tag")
    assert text == "the array is a[3] not a tag"
    assert d == {"voice": "warm"}


# --- resolve_tts_params ---------------------------------------------------

def test_resolve_no_directives_returns_defaults():
    assert resolve_tts_params(None, default_sid=3, default_speed=1.0) == (3, 1.0)
    assert resolve_tts_params({}, default_sid=3, default_speed=1.0) == (3, 1.0)


def test_resolve_voice_name_maps_to_sid():
    sid, speed = resolve_tts_params(
        {"voice": "warm"}, default_sid=0, default_speed=1.0,
        voice_map={"warm": 7}, num_speakers=103)
    assert sid == 7 and speed == 1.0


def test_resolve_unknown_voice_falls_back_to_default_sid():
    sid, _ = resolve_tts_params(
        {"voice": "nope"}, default_sid=2, default_speed=1.0, voice_map={"warm": 7})
    assert sid == 2


def test_resolve_emotion_multiplies_speed():
    _, speed = resolve_tts_params(
        {"emotion": "calm"}, default_sid=0, default_speed=1.0,
        emotion_speed_map={"calm": 0.9})
    assert speed == pytest.approx(0.9)


def test_resolve_rate_multiplies_and_combines_with_emotion():
    _, speed = resolve_tts_params(
        {"emotion": "excited", "rate": "1.1"}, default_sid=0, default_speed=1.0,
        emotion_speed_map={"excited": 1.1})
    assert speed == pytest.approx(1.1 * 1.1)


def test_resolve_speed_clamped_to_band():
    _, fast = resolve_tts_params(
        {"rate": "50"}, default_sid=0, default_speed=1.0, speed_max=2.0)
    assert fast == 2.0
    _, slow = resolve_tts_params(
        {"rate": "0.01"}, default_sid=0, default_speed=1.0, speed_min=0.5)
    assert slow == 0.5


def test_resolve_out_of_range_sid_falls_back():
    sid, _ = resolve_tts_params(
        {"voice": "ghost"}, default_sid=4, default_speed=1.0,
        voice_map={"ghost": 999}, num_speakers=103)
    assert sid == 4  # 999 >= num_speakers -> reject, keep default


def test_resolve_nonnumeric_rate_ignored():
    _, speed = resolve_tts_params(
        {"rate": "fast"}, default_sid=0, default_speed=1.0)
    assert speed == 1.0


# --- engine wiring (no threads/models/sound card) -------------------------

class _FakeAudio:
    def __init__(self, n, sr):
        self.samples = np.full(n, 0.1, dtype="float32")
        self.sample_rate = sr


class _FakeTTS:
    """Records the (text, sid, speed) of each generate() and supports the
    streaming callback path the engine prefers."""

    sample_rate = 24000
    num_speakers = 103

    def __init__(self):
        self.calls = []

    def generate(self, text, sid=0, speed=1.0, callback=None):
        self.calls.append((text, sid, speed))
        if callback is not None:  # streaming path
            callback(np.full(2400, 0.1, dtype="float32"))
            return _FakeAudio(0, self.sample_rate)
        return _FakeAudio(2400, self.sample_rate)


def _bare_engine(config: SherpaConfig) -> SherpaOnnxEngine:
    """A SherpaOnnxEngine with ONLY the attrs _synthesize/speak touch -- no
    __init__ (which would build models + threads)."""
    eng = object.__new__(SherpaOnnxEngine)
    eng.config = config
    eng._tts = _FakeTTS()
    eng._tts_lock = threading.Lock()
    eng._stop_speaking = threading.Event()
    eng._speak_gen = 0
    eng._tts_can_stream = True
    eng._play_q = queue.Queue()
    eng._tts_level_gain_db = None
    return eng


def test_synthesize_applies_directives_streaming_path():
    cfg = SherpaConfig(
        tts_markup=True, tts_declick=False, tts_target_rms=0.0, tts_output_leveler=False,
        tts_speaker_voices={"warm": 7}, tts_emotion_speed_map={"calm": 0.9})
    eng = _bare_engine(cfg)
    written = []
    eng._synthesize("hello", written.append, gen=0,
                    directives={"voice": "warm", "emotion": "calm"})
    assert eng._tts.calls == [("hello", 7, pytest.approx(0.9))]
    assert written  # the streaming callback fed at least one block


def test_synthesize_without_directives_uses_config_defaults():
    cfg = SherpaConfig(
        tts_markup=True, tts_declick=False, tts_target_rms=0.0, tts_output_leveler=False,
        tts_speaker_id=5, tts_speed=1.2)
    eng = _bare_engine(cfg)
    eng._synthesize("hello", lambda s: None, gen=0, directives=None)
    assert eng._tts.calls == [("hello", 5, pytest.approx(1.2))]


def test_speak_strips_tag_and_enqueues_directives():
    cfg = SherpaConfig(tts_markup=True, tts_speaker_voices={"warm": 7})
    eng = _bare_engine(cfg)
    eng.speak("[voice:warm emotion:calm] Hello there.")
    text, on_done, gen, directives = eng._play_q.get_nowait()
    assert text == "Hello there."
    assert directives == {"voice": "warm", "emotion": "calm"}


def test_speak_markup_off_is_passthrough():
    cfg = SherpaConfig(tts_markup=False)
    eng = _bare_engine(cfg)
    eng.speak("[voice:warm] Hello there.")  # tag NOT parsed when off
    text, on_done, gen, directives = eng._play_q.get_nowait()
    assert text == "[voice:warm] Hello there."
    assert directives is None


def test_speak_tag_only_emission_is_dropped():
    cfg = SherpaConfig(tts_markup=True, tts_emotion_speed_map={"calm": 0.9})
    eng = _bare_engine(cfg)
    done = []
    eng.speak("[emotion:calm]", on_done=lambda: done.append(True))
    assert eng._play_q.empty()  # nothing to say
    assert done == [True]       # on_done still fired


# --- system-prompt guidance (opt-in) --------------------------------------

def test_markup_guidance_empty_without_options():
    assert build_markup_guidance([], []) == ""
    assert build_markup_guidance(None, None) == ""


def test_markup_guidance_lists_voices_and_emotions():
    g = build_markup_guidance(["warm", "narrator"], ["calm", "excited"])
    assert "warm" in g and "narrator" in g
    assert "calm" in g and "excited" in g
    assert "[emotion:calm voice:warm]" in g  # a usable example


def test_build_system_prompt_appends_guidance_only_when_given():
    guidance = build_markup_guidance(["warm"], ["calm"])
    base = build_system_prompt(persona=None, web_enabled=False)
    assert "Expressive voice" not in base  # default never leaks the tag grammar
    with_g = build_system_prompt(
        persona=None, web_enabled=False, markup_guidance=guidance)
    assert with_g == base + "\n\n" + guidance  # appended verbatim, nothing else


def test_runtime_threads_markup_guidance_from_engine_config():
    from core.engines.scripted import ScriptedEngine
    from core.llm import EchoLLM
    from core.runtime import VoiceRuntime

    engine = ScriptedEngine()
    engine.config = SherpaConfig(  # duck-typed: runtime reads engine.config
        tts_markup=True, tts_speaker_voices={"warm": 7},
        tts_emotion_speed_map={"calm": 0.9})
    rt = VoiceRuntime(engine, EchoLLM(reply="hi"))
    assert "Expressive voice" in rt._system_prompt
    assert "warm" in rt._system_prompt and "calm" in rt._system_prompt


def test_runtime_no_guidance_when_markup_off():
    from core.engines.scripted import ScriptedEngine
    from core.llm import EchoLLM
    from core.runtime import VoiceRuntime

    rt = VoiceRuntime(ScriptedEngine(), EchoLLM(reply="hi"))  # no .config
    assert "Expressive voice" not in rt._system_prompt
