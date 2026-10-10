"""Actual producer/finalizer observations with public text and fake model clocks."""

from __future__ import annotations

import json
import io
import logging
from threading import Event

import pytest

from always_on_agent.capabilities import CapabilityRegistry
from core.capabilities import attach_llm_capabilities
from core.metrics import (
    ASR_FINAL,
    LLM_FIRST_TOKEN,
    LLM_REQUESTED,
    MetricsRecorder,
    TTS_TEXT_READY,
)
from core.routing import FAST, MAIN
from core.speech_chunking import SpeechChunkingConfig


class _Clock:
    value = 1.0

    def __call__(self):
        return self.value


class _Recorder(MetricsRecorder):
    def __init__(self, clock):
        super().__init__(clock=clock)
        self.ready_attempts = 0

    def mark(self, stage, **kwargs):
        if stage == TTS_TEXT_READY:
            self.ready_attempts += 1
        super().mark(stage, **kwargs)


class _Model:
    chunks = (
        (3.0, "A sufficiently long public opening clause, "),
        (4.0, "with a clear continuation"),
        (5.0, " that ends here. "),
        (6.0, "Another public sentence."),
    )

    def __init__(
        self, clock, recorder, *, failure=None, cancel=None, before_first=None
    ):
        self.clock, self.recorder = clock, recorder
        self.failure, self.cancel, self.before_first = failure, cancel, before_first
        self.calls = []
        self.closed = False

    def stream(self, prompt, **kwargs):
        # This is a regular method returning a generator: capture the actual
        # provider API call, separately from the first lazy iterator step.
        self.calls.append(
            (self.clock.value, self.recorder.records()[-1].stamps.get(LLM_REQUESTED))
        )
        if self.failure == "dispatch":
            raise RuntimeError("public injected provider dispatch failure")

        def pieces():
            try:
                if self.before_first is not None:
                    self.before_first()
                if self.failure == "empty":
                    self.clock.value = 7.0
                    return
                for index, (at, text) in enumerate(self.chunks):
                    self.clock.value = at
                    yield text
                    if index == 0:
                        if self.failure == "unready_partial":
                            raise RuntimeError(
                                "public injected provider iterator failure"
                            )
                        if self.cancel is not None:
                            self.cancel.set()
                self.clock.value = 7.0
            finally:
                self.closed = True

        return pieces()


class _Router:
    def __init__(self, tier):
        self.tier = tier

    def choose(self, query, context):
        return self.tier


def _setup(*, capability, streaming, speech_mode, tier=FAST, failure=None, cancel=None):
    clock = _Clock()
    recorder = _Recorder(clock)
    recorder.mark(ASR_FINAL)
    token = recorder.current_turn_token()
    main = _Model(clock, recorder, failure=failure, cancel=cancel)
    fast = _Model(clock, recorder)
    # Failure tests keep one provider so the existing cross-tier retry policy
    # cannot conceal which entered provider produced an observation.
    registry = attach_llm_capabilities(
        CapabilityRegistry(),
        main,
        fast_llm=fast if failure is None and cancel is None else None,
        router=_Router(tier),
        recorder=recorder,
        speech_chunking=SpeechChunkingConfig(mode=speech_mode),
    )
    emitted = []
    context = {"metadata": {"metrics_turn_token": token}}
    if streaming:
        context["emit_speech"] = lambda text: emitted.append((clock.value, text))
    if cancel is not None:
        context["cancel_event"] = cancel
    clock.value = 2.0
    return registry, recorder, clock, main, fast, context, emitted


@pytest.mark.parametrize(
    "capability,tier",
    [
        ("assistant.answer", FAST),
        ("assistant.answer", MAIN),
        ("research.local", FAST),
    ],
)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("speech_mode", ["normal", "fast"])
def test_real_producers_measure_dispatch_and_first_speakable_text(
    capability, tier, streaming, speech_mode
):
    registry, recorder, _clock, main, fast, context, emitted = _setup(
        capability=capability,
        streaming=streaming,
        speech_mode=speech_mode,
        tier=tier,
    )
    result = registry.invoke(capability, "Tell me about a public lantern.", context)
    assert (
        result.ok
        and result.text == "".join(text for _at, text in _Model.chunks).strip()
    )
    selected = main if capability == "research.local" or tier == MAIN else fast
    other = fast if selected is main else main
    assert selected.calls == [(2.0, 2.0)] and other.calls == []
    record = recorder.records()[0]
    assert record.stamps[LLM_REQUESTED] == 2.0
    assert record.stamps[LLM_FIRST_TOKEN] == 3.0
    expected = 7.0 if not streaming else 4.0 if speech_mode == "fast" else 5.0
    assert record.stamps[TTS_TEXT_READY] == expected
    assert recorder.ready_attempts == 1
    if streaming:
        assert emitted[0][0] == expected
        assert len(emitted) == (3 if speech_mode == "fast" else 2)
    else:
        assert emitted == []
    assert selected.closed


@pytest.mark.parametrize("capability", ["assistant.answer", "research.local"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("failure", ["dispatch", "unready_partial", "empty"])
def test_failure_or_empty_provider_cannot_invent_ready_text(
    capability, streaming, failure
):
    registry, recorder, _clock, main, _fast, context, emitted = _setup(
        capability=capability,
        streaming=streaming,
        speech_mode="fast",
        failure=failure,
    )
    result = registry.invoke(capability, "Tell me about a public lantern.", context)
    assert result.ok is (failure == "empty")
    assert main.calls == [(2.0, 2.0)]
    assert TTS_TEXT_READY not in recorder.records()[0].stamps
    assert recorder.ready_attempts == 0 and emitted == []
    if failure != "dispatch":
        assert main.closed


@pytest.mark.parametrize("capability", ["assistant.answer", "research.local"])
@pytest.mark.parametrize("streaming", [False, True])
def test_cancellation_before_ready_text_closes_source_without_ready_observation(
    capability, streaming
):
    cancel = Event()
    registry, recorder, _clock, main, _fast, context, emitted = _setup(
        capability=capability,
        streaming=streaming,
        speech_mode="fast",
        cancel=cancel,
    )
    result = registry.invoke(capability, "Tell me about a public lantern.", context)
    assert result.ok and result.data["cancelled"] is True
    assert main.closed and emitted == []
    assert TTS_TEXT_READY not in recorder.records()[0].stamps
    assert recorder.ready_attempts == 0


@pytest.mark.parametrize("capability", ["assistant.answer", "research.local"])
@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"metrics_turn_token": None},
        {"metrics_turn_token": True},
        {"metrics_turn_token": "1"},
        {"metrics_turn_token": 1.0},
    ],
)
def test_missing_or_invalid_captured_token_does_not_bind_new_stage_to_current(
    capability, metadata
):
    registry, recorder, _clock, _main, _fast, context, emitted = _setup(
        capability=capability,
        streaming=True,
        speech_mode="fast",
    )
    context.clear()
    if metadata is not None:
        context["metadata"] = metadata
    context["emit_speech"] = lambda text: emitted.append(text)
    result = registry.invoke(capability, "Tell me about a public lantern.", context)
    assert result.ok and emitted
    record = recorder.records()[0]
    assert LLM_REQUESTED not in record.stamps
    assert TTS_TEXT_READY not in record.stamps
    if metadata is None or metadata == {}:
        assert record.stamps[LLM_FIRST_TOKEN] == 3.0  # inherited legacy fallback


@pytest.mark.parametrize("capability", ["assistant.answer", "research.local"])
@pytest.mark.parametrize("replace_before_entry", [False, True])
def test_captured_stale_token_never_marks_successor_turn(
    capability, replace_before_entry
):
    registry, recorder, clock, main, fast, context, _emitted = _setup(
        capability=capability,
        streaming=True,
        speech_mode="fast",
    )

    def replace():
        clock.value = 2.5
        recorder.mark(ASR_FINAL)

    if replace_before_entry:
        replace()
        clock.value = 2.6
    else:
        main.before_first = fast.before_first = replace
    result = registry.invoke(capability, "Tell me about a public lantern.", context)
    assert result.ok
    successor = recorder.records()[-1]
    assert LLM_REQUESTED not in successor.stamps
    assert LLM_FIRST_TOKEN not in successor.stamps
    assert TTS_TEXT_READY not in successor.stamps
    assert all(value is None for value in successor.stage_breakdown().values())


@pytest.mark.parametrize("capability", ["assistant.answer", "research.local"])
def test_cancel_at_first_actual_emit_preserves_only_that_ready_observation(capability):
    registry, recorder, clock, main, fast, context, emitted = _setup(
        capability=capability,
        streaming=True,
        speech_mode="fast",
    )
    cancel = Event()
    context["cancel_event"] = cancel

    def first_emit(text):
        emitted.append((clock.value, text))
        cancel.set()

    context["emit_speech"] = first_emit
    result = registry.invoke(capability, "Tell me about a public lantern.", context)
    selected = main if capability == "research.local" else fast
    assert result.ok and result.data["cancelled"] is True
    assert selected.closed and len(emitted) == 1
    assert emitted[0][0] == 4.0
    assert recorder.ready_attempts == 1
    assert recorder.records()[0].stamps[TTS_TEXT_READY] == 4.0


@pytest.mark.parametrize("break_stage_snapshot", [False, True])
def test_actual_console_finalizer_exports_new_summary_or_preserves_legacy_fallback(
    tmp_path,
    monkeypatch,
    break_stage_snapshot,
):
    import core.app as app

    config = tmp_path / "public-config.json"
    config.write_text(
        json.dumps(
            {
                "memory": {"backend": "inmemory"},
                "warm_on_start": False,
                "llm": {"input_gate": False},
                "tts": {"speech_latency": "fast"},
                "device_profiles": {"public-test-cpu": {"llm": {"input_gate": False}}},
                "screen_capture": {"enabled": False},
                "visual_memory": {"enabled": False},
            }
        )
    )
    monkeypatch.setattr(app, "_load_config", lambda: json.loads(config.read_text()))
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("SPEAKER_RUN_LOG_DIR", str(log_dir))
    monkeypatch.setenv("SPEAKER_KEEP_RUNS", "0")
    monkeypatch.setenv("SPEAKER_NO_LOCAL_CONFIG", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO("Tell me about a public lantern.\n"))
    if break_stage_snapshot:

        def failed_snapshot(_self):
            raise RuntimeError("public injected stage snapshot failure")

        monkeypatch.setattr(MetricsRecorder, "stage_breakdowns", failed_snapshot)
    logger = logging.getLogger("speaker")
    prior = list(logger.handlers), logger.level, logger.propagate
    try:
        assert (
            app.main(
                [
                    "--session",
                    "console",
                    "--llm",
                    "echo",
                    "--stream-tts",
                    "--device",
                    "public-test-cpu",
                ]
            )
            == 0
        )
    finally:
        logger.handlers, logger.level, logger.propagate = prior
    reports = list(log_dir.glob("*.summary.json"))
    assert len(reports) == 1
    data = json.loads(reports[0].read_text())
    assert data["turns"] and "first_audio_latency" in data["turns"][0]
    stages = data["latency_stage_breakdown"]
    if break_stage_snapshot:
        assert stages["turns"] == [] and stages["aggregates"] == {}
    else:
        assert stages["aggregates"]["model_request_to_first_token"]["n"] >= 1
        assert stages["aggregates"]["first_token_to_speakable_text"]["n"] >= 1
    assert "Tell me about" not in json.dumps(stages)
