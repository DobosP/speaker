"""Tests for the opt-in TTS expressive-markup capability (emotion + voice
diversity). Two layers:

* the pure helpers in ``core.tts_markup`` (parse + resolve) -- exhaustive, no deps;
* the engine wiring -- ``SherpaOnnxEngine.speak() -> _synthesize() -> generate(sid,
  speed)`` -- exercised on a bare instance (``object.__new__``) with a fake TTS, so
  the directive path is covered without starting threads, models, or a sound card.

The contract: markup OFF is byte-identical (no parsing, static sid/speed); when
on, a leading ``[emotion:.. voice:.. rate:..]`` tag is stripped from the spoken
text and resolved fail-soft. The shipped speaker lock keeps ``sid`` fixed while
emotion/rate may change speed; an explicit opt-out retains named-voice mapping.
"""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

import numpy as np
import pytest

from core.engine import SpeechStyle
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from core.persona import build_system_prompt
from core.tts_markup import (
    PreparedSpeech,
    ReplySpeechId,
    ReplyVoiceContinuity,
    build_markup_guidance,
    parse_tts_markup,
    prepare_speech_style,
    resolve_tts_params,
)


# --- parse_tts_markup -----------------------------------------------------

def test_shipped_config_and_typed_default_lock_physical_speaker():
    committed = json.loads(
        (Path(__file__).resolve().parents[1] / "config.json").read_text(
            encoding="utf-8"
        )
    )

    assert SherpaConfig().tts_lock_speaker_id is True
    assert committed["sherpa"]["tts_lock_speaker_id"] is True

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


def test_parse_accepts_configured_keyless_voice_and_emotion_variants():
    text, d = parse_tts_markup(
        "[gentle voice:soft] Yes.",
        voices=["soft", "warm"],
        emotions=["gentle", "calm"],
    )
    assert text == "Yes."
    assert d == {"voice": "soft", "emotion": "gentle"}

    text, d = parse_tts_markup(
        "[warm voice] Once upon a time.",
        voices=["soft", "warm"],
        emotions=["gentle", "calm"],
    )
    assert text == "Once upon a time."
    assert d == {"voice": "warm"}

    text, d = parse_tts_markup(
        "[voice warm rate 1.05] Once.",
        voices=["soft", "warm"],
        emotions=["gentle", "calm"],
    )
    assert text == "Once."
    assert d == {"voice": "warm", "rate": "1.05"}


def test_parse_footnote_bracket_is_not_a_tag():
    # A leading bracket with NONE of the known keys is ordinary text, untouched.
    assert parse_tts_markup("[1] First reference.") == ("[1] First reference.", {})
    assert parse_tts_markup("[see note] hello") == ("[see note] hello", {})


@pytest.mark.parametrize(
    ("raw", "voices"),
    [
        ("[tag:story] Here is the first sequence.", []),
        ("[tag:narrator] The moon orbits Earth.", []),
        ("[narrator:deep] Once upon a time.", ["narrator"]),
        ("[role:story] A role alias.", []),
        ("[speaker:deep] A speaker alias.", []),
        ("[style:dramatic] A style alias.", []),
        ("[tone:warm] A tone alias.", []),
        ("[warm:deep] A configured-name alias.", ["warm"]),
    ],
)
def test_parse_strips_unsupported_model_control_tags_without_applying_style(
    raw,
    voices,
):
    text, directives = parse_tts_markup(raw, voices=voices)

    assert not text.startswith("[")
    assert directives == {}


@pytest.mark.parametrize(
    "raw",
    [
        "[citation needed] This claim needs a source.",
        "[citation:needed] This is listener-visible notation.",
        "[chapter:one] This is an unknown listener namespace.",
    ],
)
def test_parse_preserves_listener_brackets_outside_the_control_vocabulary(raw):
    assert parse_tts_markup(raw) == (raw, {})


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


# --- reply-scoped voice continuity ---------------------------------------

def test_reply_voice_persists_but_emotion_and_rate_stay_sentence_local():
    continuity = ReplyVoiceContinuity(
        voices=["warm", "deep"],
        emotions=["calm", "excited"],
    )
    reply = ReplySpeechId("reply-a", 7)

    first = continuity.prepare(
        "[voice:warm emotion:calm rate:0.9] First.",
        reply=reply,
    )
    assert first.text == "First."
    assert first.style == SpeechStyle("warm", "calm", 0.9)

    second = continuity.prepare("Second.", reply=reply)
    assert second.style == SpeechStyle(voice="warm")

    emotional = continuity.prepare(
        "[emotion:excited rate:1.1] Third.",
        reply=reply,
    )
    assert emotional.style == SpeechStyle("warm", "excited", 1.1)
    assert continuity.prepare("Fourth.", reply=reply).style == SpeechStyle(
        voice="warm"
    )


def test_locked_speaker_sanitizes_voice_but_keeps_fragment_expression():
    continuity = ReplyVoiceContinuity(
        voices=["warm", "deep"],
        emotions=["calm", "excited"],
        lock_speaker_id=True,
    )
    first_reply = ReplySpeechId("reply-a", 7)
    second_reply = ReplySpeechId("reply-b", 8)

    first = continuity.prepare(
        "[voice:warm emotion:calm rate:0.9] First.",
        reply=first_reply,
    )
    later = continuity.prepare("Second.", reply=first_reply)
    other = continuity.prepare(
        "[voice:deep emotion:excited] Third.",
        reply=second_reply,
    )

    assert first == PreparedSpeech(
        "First.", SpeechStyle(emotion="calm", rate=0.9)
    )
    assert later == PreparedSpeech("Second.", None)
    assert other == PreparedSpeech("Third.", SpeechStyle(emotion="excited"))


def test_later_explicit_voice_switches_only_future_fragments():
    continuity = ReplyVoiceContinuity(voices=["warm", "deep"])
    reply = ReplySpeechId("reply-a", 3)

    queued_before_switch = continuity.prepare("[voice:warm] One.", reply=reply)
    inherited_before_switch = continuity.prepare("Two.", reply=reply)
    switch = continuity.prepare("[voice:deep] Three.", reply=reply)
    inherited_after_switch = continuity.prepare("Four.", reply=reply)

    assert [
        item.style.voice if item.style is not None else None
        for item in (
            queued_before_switch,
            inherited_before_switch,
            switch,
            inherited_after_switch,
        )
    ] == ["warm", "warm", "deep", "deep"]
    # PreparedSpeech is an immutable queue snapshot: the later switch cannot
    # retroactively mutate fragments already admitted with warm.
    assert queued_before_switch.style == SpeechStyle(voice="warm")


def test_reply_voices_are_isolated_and_close_by_exact_epoch_or_task():
    continuity = ReplyVoiceContinuity(voices=["warm", "deep"])
    a_old = ReplySpeechId("reply-a", 1)
    a_new = ReplySpeechId("reply-a", 2)
    b = ReplySpeechId("reply-b", 1)

    continuity.prepare("[voice:warm] A1.", reply=a_old)
    continuity.prepare("[voice:deep] B1.", reply=b)
    continuity.prepare("[voice:deep] A2.", reply=a_new)
    assert continuity.prepare("A1 next.", reply=a_old).style == SpeechStyle(
        voice="warm"
    )
    assert continuity.prepare("B next.", reply=b).style == SpeechStyle(voice="deep")

    continuity.close(a_old)
    assert continuity.prepare("A1 reset.", reply=a_old).style is None
    assert continuity.prepare("A2 stays.", reply=a_new).style == SpeechStyle(
        voice="deep"
    )
    assert continuity.prepare("B stays.", reply=b).style == SpeechStyle(voice="deep")

    continuity.close_task("reply-a")
    assert continuity.prepare("A2 reset.", reply=a_new).style is None
    assert continuity.prepare("B still stays.", reply=b).style == SpeechStyle(
        voice="deep"
    )


def test_unknown_voice_is_ignored_without_destroying_active_lease():
    continuity = ReplyVoiceContinuity(voices=["warm"])
    reply = ReplySpeechId("reply", 1)
    continuity.prepare("[voice:warm] One.", reply=reply)

    unknown = continuity.prepare("[voice:not-configured] Two.", reply=reply)
    assert unknown.text == "Two."
    assert unknown.style == SpeechStyle(voice="warm")
    assert continuity.prepare("Three.", reply=reply).style == SpeechStyle(
        voice="warm"
    )


def test_unsupported_control_tag_is_silent_and_does_not_mutate_voice_lease():
    continuity = ReplyVoiceContinuity(voices=["lively", "narrator"])
    reply = ReplySpeechId("reply", 1)
    continuity.prepare("[voice:lively] One.", reply=reply)

    malformed = continuity.prepare(
        "[narrator:deep] That sounds lovely.",
        reply=reply,
    )
    alias = continuity.prepare("[tag:story] Two.", reply=reply)

    assert malformed.text == "That sounds lovely."
    assert malformed.style == SpeechStyle(voice="lively")
    assert alias.text == "Two."
    assert alias.style == SpeechStyle(voice="lively")


def test_later_unsupported_control_tag_is_removed_from_one_engine_fragment():
    prepared = prepare_speech_style(
        "First sentence. [tag:story] Second sentence.",
        style=SpeechStyle(voice="warm"),
        voices=["warm"],
    )

    assert prepared.text == "First sentence. Second sentence."
    assert prepared.style == SpeechStyle(voice="warm")


def test_auxiliary_unscoped_style_neither_inherits_nor_mutates_reply():
    continuity = ReplyVoiceContinuity(voices=["warm", "deep"])
    reply = ReplySpeechId("reply", 1)
    continuity.prepare("[voice:warm] Answer one.", reply=reply)

    assert continuity.prepare("Please wait.", reply=None).style is None
    aux_explicit = continuity.prepare("[voice:deep] Please wait.", reply=None)
    assert aux_explicit.style == SpeechStyle(voice="deep")
    assert continuity.prepare("Answer two.", reply=reply).style == SpeechStyle(
        voice="warm"
    )


# --- resolve_tts_params ---------------------------------------------------

def test_resolve_no_directives_returns_defaults():
    assert resolve_tts_params(None, default_sid=3, default_speed=1.0) == (3, 1.0)
    assert resolve_tts_params({}, default_sid=3, default_speed=1.0) == (3, 1.0)


def test_resolve_voice_name_maps_to_sid():
    sid, speed = resolve_tts_params(
        {"voice": "warm"}, default_sid=0, default_speed=1.0,
        voice_map={"warm": 7}, num_speakers=103)
    assert sid == 7 and speed == 1.0


def test_resolve_locked_speaker_ignores_voice_but_keeps_emotion_and_rate():
    sid, speed = resolve_tts_params(
        {"voice": "warm", "emotion": "calm", "rate": "1.1"},
        default_sid=3,
        default_speed=1.0,
        voice_map={"warm": 7},
        emotion_speed_map={"calm": 0.9},
        num_speakers=103,
        lock_speaker_id=True,
    )
    assert sid == 3
    assert speed == pytest.approx(0.99)


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
    eng._playback_stopping = threading.Event()
    eng._speak_gen = 0
    eng._gen_lock = threading.Lock()
    eng._receipt_lock = threading.RLock()
    eng._tts_can_stream = True
    eng._play_q = queue.Queue()
    eng._tts_level_gain_db = None
    return eng


def test_synthesize_applies_directives_streaming_path():
    cfg = SherpaConfig(
        tts_markup=True, tts_declick=False, tts_target_rms=0.0, tts_output_leveler=False,
        tts_lock_speaker_id=False,
        tts_speaker_voices={"warm": 7}, tts_emotion_speed_map={"calm": 0.9})
    eng = _bare_engine(cfg)
    written = []
    eng._synthesize("hello", written.append, gen=0,
                    directives={"voice": "warm", "emotion": "calm"})
    assert eng._tts.calls == [("hello", 7, pytest.approx(0.9))]
    assert written  # the streaming callback fed at least one block


def test_synthesize_strips_unparsed_bracket_directive_before_tts():
    cfg = SherpaConfig(
        tts_markup=True, tts_declick=False, tts_target_rms=0.0, tts_output_leveler=False,
        tts_lock_speaker_id=False,
        tts_speaker_voices={"warm": 7}, tts_emotion_speed_map={"gentle": 0.92})
    eng = _bare_engine(cfg)
    written = []
    eng._synthesize("[warm voice] Once upon a time.", written.append, gen=0)
    assert eng._tts.calls == [("Once upon a time.", 7, pytest.approx(1.0))]
    assert written


def test_synthesize_merges_final_guard_directives_with_existing_ones():
    cfg = SherpaConfig(
        tts_markup=True, tts_declick=False, tts_target_rms=0.0, tts_output_leveler=False,
        tts_lock_speaker_id=False,
        tts_speaker_voices={"warm": 7, "soft": 3}, tts_emotion_speed_map={"gentle": 0.92})
    eng = _bare_engine(cfg)
    eng._synthesize(
        "[gentle voice:soft] Yes.",
        lambda _samples: None,
        gen=0,
        directives={"voice": "warm"},
    )
    assert eng._tts.calls == [("Yes.", 7, pytest.approx(0.92))]


def test_synthesize_locks_physical_speaker_across_different_reply_styles():
    cfg = SherpaConfig(
        tts_markup=True,
        tts_lock_speaker_id=True,
        tts_speaker_id=5,
        tts_speed=1.0,
        tts_speaker_voices={"warm": 7, "deep": 9},
        tts_emotion_speed_map={"calm": 0.9, "excited": 1.1},
        tts_declick=False,
        tts_target_rms=0.0,
        tts_output_leveler=False,
    )
    eng = _bare_engine(cfg)

    eng._synthesize(
        "first reply",
        lambda _samples: None,
        gen=0,
        directives={"voice": "warm", "emotion": "calm"},
    )
    eng._synthesize(
        "second reply",
        lambda _samples: None,
        gen=0,
        directives={"voice": "deep", "emotion": "excited"},
    )

    assert eng._tts.calls == [
        ("first reply", 5, pytest.approx(0.9)),
        ("second reply", 5, pytest.approx(1.1)),
    ]


def test_synthesize_lowpass_streams_and_attenuates_hf():
    """tts_output_lowpass_hz>0 streams through a causal per-chunk filter and
    attenuates high-frequency energy."""
    class _BrightTTS:
        sample_rate = 24000
        num_speakers = 103

        def __init__(self):
            self.callback_used = None

        def generate(self, text, sid=0, speed=1.0, callback=None):
            self.callback_used = callback is not None
            t = np.arange(int(self.sample_rate * 0.3)) / self.sample_rate
            tone = (0.3 * np.sin(2 * np.pi * 10000 * t)).astype("float32")  # 10 kHz
            if callback is not None:
                for chunk in np.array_split(tone, 4):
                    callback(chunk, 1.0)
                return _FakeAudio(0, self.sample_rate)
            return type("A", (), {"samples": tone, "sample_rate": self.sample_rate})()

    cfg = SherpaConfig(tts_declick=False, tts_target_rms=0.0, tts_output_leveler=False,
                       tts_output_lowpass_hz=3000.0)
    eng = _bare_engine(cfg)
    eng._tts = _BrightTTS()
    written = []
    eng._synthesize("hi", written.append, gen=0)
    assert eng._tts.callback_used is True
    out = np.concatenate(written) if written else np.zeros(0)
    t = np.arange(out.size) / eng._tts.sample_rate
    raw_mag = float((2.0 / out.size) * abs(np.dot(
        0.3 * np.sin(2 * np.pi * 10000 * t), np.sin(2 * np.pi * 10000 * t)
    )))
    out_mag = float((2.0 / out.size) * abs(np.dot(out, np.sin(2 * np.pi * 10000 * t))))
    assert out_mag < 0.12 * raw_mag


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
    text, on_done, gen, directives, ticket = eng._play_q.get_nowait()
    assert text == "Hello there."
    assert directives == {"voice": "warm", "emotion": "calm"}
    assert ticket is None


def test_sherpa_enqueues_typed_style_without_textual_markup():
    cfg = SherpaConfig(
        tts_markup=True,
        tts_speaker_voices={"warm": 7},
        tts_emotion_speed_map={"calm": 0.9},
    )
    eng = _bare_engine(cfg)
    eng._submit_playback(
        "Inherited voice.",
        on_done=None,
        ticket=None,
        style=SpeechStyle("warm", "calm", 1.05),
    )

    text, on_done, gen, directives, ticket = eng._play_q.get_nowait()
    assert text == "Inherited voice."
    assert directives == {"voice": "warm", "emotion": "calm", "rate": 1.05}
    assert ticket is None


def test_speak_strips_live_keyless_voice_variant():
    cfg = SherpaConfig(tts_markup=True, tts_speaker_voices={"warm": 7})
    eng = _bare_engine(cfg)
    eng.speak("[warm voice] Once upon a time.")
    text, on_done, gen, directives, ticket = eng._play_q.get_nowait()
    assert text == "Once upon a time."
    assert directives == {"voice": "warm"}
    assert ticket is None


@pytest.mark.parametrize(
    "raw",
    [
        "[tag:story] Here is the first sequence.",
        "[tag:narrator] The moon orbits Earth.",
        "[narrator:deep] Once upon a time.",
    ],
)
def test_sherpa_does_not_enqueue_unsupported_control_tag_text(raw):
    cfg = SherpaConfig(
        tts_markup=True,
        tts_speaker_voices={"narrator": 7},
    )
    eng = _bare_engine(cfg)

    eng.speak(raw)

    text, _on_done, _gen, directives, ticket = eng._play_q.get_nowait()
    assert not text.startswith("[")
    assert directives is None
    assert ticket is None


def test_synthesize_final_guard_strips_unsupported_control_tag():
    cfg = SherpaConfig(
        tts_markup=True,
        tts_declick=False,
        tts_target_rms=0.0,
        tts_output_leveler=False,
    )
    eng = _bare_engine(cfg)

    eng._synthesize("[tag:story] The visible line.", lambda _samples: None, gen=0)

    assert eng._tts.calls == [("The visible line.", 0, pytest.approx(1.0))]


def test_speak_markup_off_is_passthrough():
    cfg = SherpaConfig(tts_markup=False)
    eng = _bare_engine(cfg)
    eng.speak("[voice:warm] Hello there.")  # tag NOT parsed when off
    text, on_done, gen, directives, ticket = eng._play_q.get_nowait()
    assert text == "[voice:warm] Hello there."
    assert directives is None
    assert ticket is None

    eng.speak("[tag:story] Listener-visible while markup is off.")
    text, _on_done, _gen, directives, ticket = eng._play_q.get_nowait()
    assert text == "[tag:story] Listener-visible while markup is off."
    assert directives is None
    assert ticket is None


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
    g = build_markup_guidance(
        ["warm", "narrator"],
        ["calm", "excited"],
        voice_continuity=True,
    )
    assert "warm" in g and "narrator" in g
    assert "calm" in g and "excited" in g
    assert "[emotion:calm voice:warm]" in g  # a usable example
    assert "voice choice persists" in g
    assert "one tag per reply" in g
    assert "never put tags between sentences" in g
    assert "Emotion and rate apply to" in g


def test_markup_guidance_hides_named_voices_when_speaker_is_locked():
    guidance = build_markup_guidance(
        ["warm", "narrator"],
        ["calm"],
        voice_continuity=True,
        lock_speaker_id=True,
    )

    assert "physical speaker voice is fixed" in guidance
    assert "Never emit a voice directive" in guidance
    assert "warm" not in guidance and "narrator" not in guidance
    assert "[emotion:calm]" in guidance
    assert "voice choice persists" not in guidance


def test_locked_speaker_without_emotions_has_no_empty_markup_example():
    guidance = build_markup_guidance(
        ["warm"],
        [],
        voice_continuity=True,
        lock_speaker_id=True,
    )

    assert guidance == (
        "The physical speaker voice is fixed for this session. Never emit a "
        "voice directive or a voice name."
    )
    assert "[]" not in guidance


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
        tts_markup=True, tts_lock_speaker_id=False,
        tts_speaker_voices={"warm": 7},
        tts_emotion_speed_map={"calm": 0.9})
    rt = VoiceRuntime(engine, EchoLLM(reply="hi"))
    assert "Expressive voice" in rt._system_prompt
    assert "warm" in rt._system_prompt and "calm" in rt._system_prompt


def test_runtime_prompt_enforces_shipped_speaker_lock():
    from core.engines.scripted import ScriptedEngine
    from core.llm import EchoLLM
    from core.runtime import VoiceRuntime

    engine = ScriptedEngine()
    engine.config = SherpaConfig(
        tts_markup=True,
        tts_lock_speaker_id=True,
        tts_speaker_voices={"warm": 7},
        tts_emotion_speed_map={"calm": 0.9},
    )
    runtime = VoiceRuntime(engine, EchoLLM(reply="hi"))

    assert "physical speaker voice is fixed" in runtime._system_prompt
    assert "warm" not in runtime._system_prompt
    assert "calm" in runtime._system_prompt


def test_runtime_no_guidance_when_markup_off():
    from core.engines.scripted import ScriptedEngine
    from core.llm import EchoLLM
    from core.runtime import VoiceRuntime

    rt = VoiceRuntime(ScriptedEngine(), EchoLLM(reply="hi"))  # no .config
    assert "Expressive voice" not in rt._system_prompt


def test_runtime_preserves_raw_markup_for_legacy_speak_override():
    from always_on_agent.events import AgentEvent, EventKind
    from core.engine import PlaybackCapabilities
    from core.engines.scripted import ScriptedEngine
    from core.llm import EchoLLM
    from core.runtime import VoiceRuntime

    class _LegacyMarkupEngine(ScriptedEngine):
        def __init__(self):
            super().__init__()
            self.config = SherpaConfig(
                tts_markup=True,
                tts_speaker_voices={"warm": 7},
            )
            self.received = []

        def speak(self, text, on_done=None):
            self.received.append(text)
            super().speak(text, on_done)

        @property
        def playback_capabilities(self):
            # Even an invalid/custom hints-without-tracking combination must
            # not make runtime strip the only directive carrier before speak().
            return PlaybackCapabilities(speech_style_hints=True)

    engine = _LegacyMarkupEngine()
    assert engine.playback_capabilities.speech_style_hints
    assert not engine.playback_capabilities.tracked_terminal
    runtime = VoiceRuntime(engine, EchoLLM(reply="unused"))
    runtime.start(run_bus=False)
    try:
        runtime.bus.publish(
            AgentEvent(
                EventKind.TTS_REQUEST,
                {"text": "[voice:warm] Legacy fragment."},
            )
        )
        runtime.bus.drain()

        assert engine.received == ["[voice:warm] Legacy fragment."]
        assert "voice choice persists" not in runtime._system_prompt
    finally:
        runtime.stop()
