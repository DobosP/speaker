from __future__ import annotations

from copy import deepcopy
from threading import Event

import pytest

from core.contract import stream_sentences
from core.speech_chunking import (
    SpeechChunker,
    SpeechChunkingConfig,
    apply_speech_latency,
)


def chunks(parts, mode="fast", **kwargs):
    chunker = SpeechChunker(SpeechChunkingConfig(mode=mode, **kwargs))
    result = []
    for part in parts:
        result.extend(chunker.feed(part))
    tail = chunker.finish()
    if tail:
        result.append(tail)
    return result


@pytest.mark.parametrize(
    "text",
    [
        "A short reply. Another one!\nLast line",
        "Version 3.14 works. ",
        "  First\n\n Second\r\nThird! ",
        "[emotion:calm] Hello, world. More.",
    ],
)
def test_normal_mode_keeps_shared_sentence_contract_for_every_split(text):
    for split in range(len(text) + 1):
        parts = (text[:split], text[split:])
        assert chunks(parts, mode="normal") == stream_sentences(parts)


def test_first_clause_ready_before_long_sentence_or_model_completion():
    chunker = SpeechChunker(SpeechChunkingConfig(mode="fast"))
    prefix = "There is a clear way to approach this problem, "
    assert chunker.feed(prefix) == []  # safe continuation lookahead
    assert chunker.feed("start ") == [prefix.strip()]
    assert chunker.feed(
        "with the simplest case, and then check the remaining details. Next sentence. "
    ) == [
        "start with the simplest case, and then check the remaining details.",
        "Next sentence.",
    ]
    assert chunker.finish() == ""


def test_unpunctuated_first_sentence_has_one_word_boundary_only():
    text = " ".join([f"word{chr(97 + i % 26)}" for i in range(70)])
    out = chunks((word + " " for word in text.split()), max_words=12)
    assert len(out) == 2
    assert len(out[0].split()) == 12
    assert " ".join(out) == text


@pytest.mark.parametrize(
    "text",
    [
        "[emotion:calm] Here is a sufficiently long introduction, followed by the rest.",
        "The quoted words are 'this is a very long sentence, inside the quotation'.",
        'The quoted words are "this is a very long sentence, inside the quotation".',
        "The array looks like [some very long content, with more values] today.",
        "An explanation with (a deliberately long parenthetical clause, with details) follows.",
        "A response containing `a deliberately long code example, with arguments` follows.",
        "A long enough opening with a possible tag, [emotion:calm] must not activate it.",
    ],
)
def test_protected_first_sentence_is_not_split_early_at_any_token_boundary(text):
    for split in range(len(text) + 1):
        out = chunks((text[:split], text[split:]))
        assert out == stream_sentences((text[:split], text[split:]))


def test_numeric_punctuation_and_contractions_stay_intact():
    text = "It's a total of 1,234.56 units at 12:30, and the remaining description follows."
    out = chunks(iter(text))
    assert out == [text]
    text = "It's a long enough neutral opening clause, and now the remainder follows."
    assert chunks(iter(text)) == [
        "It's a long enough neutral opening clause,",
        "and now the remainder follows.",
    ]


def test_earliest_complete_sentence_wins_over_later_clause():
    text = "Short first. This is a much longer second sentence, with details. "
    assert chunks([text]) == stream_sentences([text])


def test_long_unbroken_token_is_not_cut_or_lost():
    text = "x" * 4096
    assert chunks(iter(text)) == [text]


def test_selection_is_pure_and_preserves_models_capabilities_and_streaming():
    config = {
        "tts": {"streaming": False},
        "llm": {"fast_model": "existing"},
        "capabilities": {"tools": True},
    }
    before = deepcopy(config)
    selected, policy = apply_speech_latency(config, "fast")
    assert config == before
    assert policy.mode == "fast"
    assert (
        selected["llm"] == config["llm"]
        and selected["capabilities"] == config["capabilities"]
    )
    assert selected["tts"]["streaming"] is False
    rolled, policy = apply_speech_latency(selected, "normal")
    assert policy.mode == "normal" and rolled["tts"]["speech_latency"] == "normal"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mode="unknown"),
        dict(min_chars=True),
        dict(min_chars=0),
        dict(max_words=0),
        dict(max_words=81),
    ],
)
def test_configuration_bounds_are_strict(kwargs):
    with pytest.raises(ValueError):
        SpeechChunkingConfig(**kwargs)


def test_fast_fragment_is_delivered_before_provider_completion_and_keeps_full_answer():
    from core.capabilities import _stream_and_speak

    first = "There is a clear way to approach this problem,"
    spoken, ready = [], []

    def provider():
        yield first + " start "
        assert spoken == [first]
        yield "with the simplest case. Next sentence. Tail"

    text, cancelled = _stream_and_speak(
        provider(),
        Event(),
        spoken.append,
        chunking=SpeechChunkingConfig(mode="fast"),
        on_first_text=lambda: ready.append(True),
    )
    assert text == first + " start with the simplest case. Next sentence. Tail"
    assert cancelled is False
    assert spoken == [first, "start with the simplest case.", "Next sentence.", "Tail"]
    assert ready == [True]


@pytest.mark.parametrize("cancel_at", ["observer", "first_fragment"])
def test_cancellation_prevents_following_fragments_and_closes_provider(cancel_at):
    from core.capabilities import _stream_and_speak

    cancel = Event()
    closed, spoken = [], []
    first = "There is a clear way to approach this problem,"

    def provider():
        try:
            yield first + " start here. Another sentence. Tail"
            pytest.fail("cancelled provider consumed again")
        finally:
            closed.append(True)

    def ready():
        if cancel_at == "observer":
            cancel.set()

    def emit(value):
        spoken.append(value)
        cancel.set()

    _, cancelled = _stream_and_speak(
        provider(),
        cancel,
        emit,
        chunking=SpeechChunkingConfig(mode="fast"),
        on_first_text=ready,
    )
    assert cancelled is True
    assert spoken == ([] if cancel_at == "observer" else [first])
    assert closed == [True]


@pytest.mark.parametrize("failure_at", ["observer", "emitter"])
def test_callback_exception_closes_entered_provider(failure_at):
    from core.capabilities import _stream_and_speak

    closed = []

    def provider():
        try:
            yield "There is a clear way to approach this problem, start here. "
        finally:
            closed.append(True)

    def fail(*_):
        raise RuntimeError("consumer failed")

    with pytest.raises(RuntimeError, match="consumer failed"):
        _stream_and_speak(
            provider(),
            Event(),
            fail if failure_at == "emitter" else lambda _: None,
            chunking=SpeechChunkingConfig(mode="fast"),
            on_first_text=fail if failure_at == "observer" else None,
        )
    assert closed == [True]


def test_provider_failure_after_first_clause_never_starts_fallback():
    from always_on_agent.capabilities import CapabilityRegistry
    from core.capabilities import attach_llm_capabilities

    class Broken:
        def stream(self, *_args, **_kwargs):
            yield "There is a clear way to approach this problem, start "
            raise RuntimeError("stream failed after early speech")

    class Fallback:
        def stream(self, *_args, **_kwargs):
            pytest.fail("speech must not replay through fallback")

    registry = attach_llm_capabilities(
        CapabilityRegistry(),
        Fallback(),
        fast_llm=Broken(),
        speech_chunking=SpeechChunkingConfig(mode="fast"),
    )
    spoken = []
    result = registry.invoke(
        "assistant.answer", "what time is it", {"emit_speech": spoken.append}
    )
    assert result.ok is False
    assert spoken == ["There is a clear way to approach this problem,"]


@pytest.mark.parametrize("streaming", [False, True])
def test_runtime_applies_fast_policy_only_when_streaming(streaming):
    from core.engines.scripted import ScriptedEngine
    from core.runtime import VoiceRuntime

    text = (
        "There is a clear way to approach this problem, start with the simplest case."
    )

    class Model:
        def stream(self, *_args, **_kwargs):
            yield text

    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        Model(),
        stream_tts=streaming,
        speech_chunking=SpeechChunkingConfig(mode="fast"),
    )
    try:
        runtime.start(run_bus=False)
        engine.final("tell me how to approach this problem")
        assert runtime.wait_idle()
        assert engine.spoken == (
            [
                "There is a clear way to approach this problem,",
                "start with the simplest case.",
            ]
            if streaming
            else [text]
        )
    finally:
        runtime.stop()


def test_doctor_launcher_share_selection_without_loading_models(tmp_path, monkeypatch):
    import json
    from tools import doctor, live_launcher

    config = {
        "device": "safe",
        "device_profiles": {"safe": {"sherpa": {"aec_enabled": False}}},
        "tts": {"streaming": True},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    seen = []
    monkeypatch.setattr(
        doctor, "run_runtime_checks", lambda value, **_: seen.append(value) or []
    )
    checks = doctor.run_all(
        config, device="safe", speech_latency="fast", config_root=tmp_path
    )
    selected = live_launcher._selected_live_config(
        tmp_path, "safe", speech_latency="fast"
    )
    assert selected.speech_latency == seen[0]["tts"]["speech_latency"] == "fast"
    assert any(c.name == "speech latency policy" for c in checks)
    assert "speech_latency" not in config["tts"]


@pytest.mark.parametrize(
    "args",
    [
        ["--speech-latency", "unknown"],
        ["--speech-latency", "normal", "--speech-latency", "fast"],
        ["--guided-stt-capture", "--speech-latency", "fast"],
    ],
)
def test_invalid_or_capture_mode_stops_at_launcher_parser(args):
    from tools.live_launcher import _parse_live_arguments

    with pytest.raises(SystemExit):
        _parse_live_arguments(args)
