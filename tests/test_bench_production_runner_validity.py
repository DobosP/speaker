from __future__ import annotations

import io
import logging
import struct
from types import SimpleNamespace

import pytest

from core.llm import EchoLLM, OllamaLLM
from core.metrics import ASR_FINAL, LLM_FIRST_TOKEN, SPEECH_END, TTS_FIRST_AUDIO
from tools.bench import runner
from tools.conversation_eval.provenance import LOCAL_OLLAMA_HEADERS


def _case(index: int) -> SimpleNamespace:
    audio = b"".join(
        struct.pack("<f", value)
        for value in (index + 0.1, index + 0.2, index + 0.3, index + 0.4)
    )
    return SimpleNamespace(
        case_id=f"case-{index}",
        audio_bytes=audio,
        expected_text=f"reference-{index}",
        assertion="transcript",
        commands=(),
        forbidden_commands=(),
    )


def _config() -> dict:
    return {
        "llm": {
            "backend": "ollama",
            "main_model": "main:test",
            "fast_model": "fast:test",
        },
        "sherpa": {
            "asr_encoder": "encoder",
            "asr_tokens": "tokens",
            "vad_model": "vad",
            "tts_model": "tts",
            "tts_tokens": "tts-tokens",
        },
    }


def _run(cases, config, **overrides):
    arguments = {
        "device_profile": "desktop_gpu_4090",
        "final_stt_profile": "sense-voice",
        "final_stt_profile_sha256": "1" * 64,
        "final_stt_profile_schema_version": 1,
        "production_config_sha256": "2" * 64,
        "execution_config_sha256": "3" * 64,
        "replay_safety_schema_version": 1,
        "replay_safety_sha256": "4" * 64,
        "active_artifacts_sha256": "5" * 64,
        "active_artifact_roots": 5,
        "active_artifact_files": 5,
        "active_artifact_bytes": 100,
        "source_revision": "6" * 40,
        "runtime_identities": {},
        "ollama_identity_sha256": "7" * 64,
        "ollama_role_contracts": {},
        "timeout": 9.0,
    }
    arguments.update(overrides)
    return runner.run_production(
        SimpleNamespace(cases=tuple(cases)),
        config,
        **arguments,
    )


def _install_runtime_fakes(
    monkeypatch,
    *,
    tts_completed: tuple[bool, ...],
    warm_completed: tuple[bool, ...] | None = None,
    private_log_canary: str | None = None,
):
    import core.app as core_app
    import core.engines.file_replay as file_replay
    import core.llm_factory as llm_factory
    import core.routing as routing

    events: list[tuple] = []
    runtimes = []
    engines = []
    llm_arguments = []
    router_ids = []
    warm_results = warm_completed or tuple(True for _ in tts_completed)

    class WarmReady:
        def __init__(self, runtime, index):
            self.runtime = runtime
            self.index = index

        def wait(self, *, timeout):
            events.append(("warm_wait", self.index, timeout))
            result = warm_results[self.index]
            self.runtime.warmed = result
            return result

    class Engine:
        def __init__(self, config):
            self.config = config
            self.index = len(engines)
            self.finals: list[str] = []
            self.spoken: list[str] = []
            self.last_final = ""
            self.runtime = None
            engines.append(self)

        def replay_samples(self, samples, sample_rate):
            events.append(("replay", self.index))
            assert self.runtime is runtimes[self.index]
            assert self.runtime.warmed is True
            assert self.runtime.session_state == []
            assert samples.shape == (4,)
            assert sample_rate == 16_000
            self.runtime.session_state.append(f"row-{self.index}")
            metrics = self.runtime.metrics
            metrics.mark(SPEECH_END)
            metrics.mark(ASR_FINAL)
            metrics.mark(LLM_FIRST_TOKEN)
            # Admission alone is deliberately insufficient evidence: both rows
            # append to spoken, while only the first completes the TTS boundary.
            self.spoken.append(f"admitted-answer-{self.index}")
            if tts_completed[self.index]:
                metrics.mark(TTS_FIRST_AUDIO)
            self.finals.append(f"final-{self.index}")
            if private_log_canary is not None:
                logging.getLogger("speaker.runtime").info(private_log_canary)

    class Runtime:
        def __init__(self, engine, index):
            from core.metrics import MetricsRecorder

            self.index = index
            self.engine = engine
            self.metrics = MetricsRecorder()
            self.session_state: list[str] = []
            self.warmed = False
            self.warm_ready = WarmReady(self, index)
            self.stopped = False
            engine.runtime = self

        def start(self, *, run_bus):
            events.append(("start", self.index, run_bus))

        def wait_idle(self, *, timeout):
            events.append(("idle", self.index, timeout))
            return True

        def stop(self):
            events.append(("stop", self.index))
            self.stopped = True

    def build_llms(args, config):
        llm_arguments.append(args)
        assert config is not None
        headers = dict(args.ollama_client_headers)
        return (
            OllamaLLM("main:test", client=object(), client_headers=headers),
            OllamaLLM("fast:test", client=object(), client_headers=headers),
        )

    def build_runtime(config, **kwargs):
        index = len(runtimes)
        events.append(("build_runtime", index))
        assert config is not None
        assert type(kwargs["llm"]) is OllamaLLM
        assert type(kwargs["fast_llm"]) is OllamaLLM
        assert kwargs["router"] is router_ids[index]
        runtime = Runtime(kwargs["engine"], index)
        runtimes.append(runtime)
        return runtime

    def build_router(config):
        router = object()
        router_ids.append(router)
        events.append(("build_router", len(router_ids) - 1))
        assert config is not None
        return router

    monkeypatch.setattr(file_replay, "FileReplayEngine", Engine)
    monkeypatch.setattr(llm_factory, "build_llms", build_llms)
    monkeypatch.setattr(core_app, "build_runtime", build_runtime)
    monkeypatch.setattr(routing, "build_router", build_router)
    return SimpleNamespace(
        events=events,
        runtimes=runtimes,
        engines=engines,
        llm_arguments=llm_arguments,
        router_ids=router_ids,
    )


def test_rows_get_fresh_runtime_wait_for_warmup_and_require_completed_tts(
    monkeypatch,
):
    observed = _install_runtime_fakes(
        monkeypatch,
        tts_completed=(True, False),
    )
    after_build = []

    def mark_after_build():
        after_build.append(True)
        observed.events.append(("after_build", 0))

    result = _run(
        (_case(0), _case(1)),
        _config(),
        after_build=mark_after_build,
    )

    assert len(observed.llm_arguments) == 1
    assert observed.llm_arguments[0].ollama_client_headers == LOCAL_OLLAMA_HEADERS
    assert len(observed.runtimes) == len(observed.engines) == 2
    assert observed.runtimes[0] is not observed.runtimes[1]
    assert observed.engines[0] is not observed.engines[1]
    assert all(runtime.stopped for runtime in observed.runtimes)
    assert [runtime.session_state for runtime in observed.runtimes] == [
        ["row-0"],
        ["row-1"],
    ]
    assert len(observed.router_ids) == 2
    assert after_build == [True]
    assert [sample.transcript for sample in result.samples] == ["final-0", "final-1"]
    assert [sample.responded for sample in result.samples] == [True, False]
    for index in range(2):
        warm_position = observed.events.index(("warm_wait", index, 9.0))
        replay_position = observed.events.index(("replay", index))
        assert warm_position < replay_position
    assert observed.events.index(("warm_wait", 0, 9.0)) < observed.events.index(
        ("after_build", 0)
    ) < observed.events.index(("replay", 0))


def test_warmup_timeout_fails_before_timing_or_replay_and_stops_runtime(monkeypatch):
    observed = _install_runtime_fakes(
        monkeypatch,
        tts_completed=(True,),
        warm_completed=(False,),
    )

    with pytest.raises(runner.ProductionBenchError):
        _run((_case(0),), _config())

    assert not any(event[0] == "replay" for event in observed.events)
    assert observed.runtimes[0].stopped is True


@pytest.mark.parametrize("invalid_pair", ("missing_fast", "wrong_main", "headers"))
def test_runner_rejects_non_exact_local_ollama_pair(monkeypatch, invalid_pair):
    import core.llm_factory as llm_factory

    def build_llms(args, _config):
        headers = None if invalid_pair == "headers" else args.ollama_client_headers
        main = OllamaLLM("main:test", client=object(), client_headers=headers)
        fast = OllamaLLM("fast:test", client=object(), client_headers=headers)
        if invalid_pair == "missing_fast":
            fast = None
        elif invalid_pair == "wrong_main":
            main = EchoLLM()
        return main, fast

    monkeypatch.setattr(llm_factory, "build_llms", build_llms)

    with pytest.raises(runner.ProductionBenchError):
        _run((_case(0),), _config())


def test_private_speaker_logs_are_suppressed_and_handler_state_is_restored(
    monkeypatch,
):
    private_canary = "PRIVATE_TRANSCRIPT_LOG_CANARY"
    _install_runtime_fakes(
        monkeypatch,
        tts_completed=(True,),
        private_log_canary=private_canary,
    )
    target = logging.getLogger("speaker.runtime")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    previous_level = target.level
    previous_propagate = target.propagate
    previous_disabled = target.disabled
    previous_threshold = logging.root.manager.disable
    filters_before = tuple(handler.filters)
    target.addHandler(handler)
    target.setLevel(logging.INFO)
    target.propagate = False
    target.disabled = False
    try:
        target.info("BEFORE_GUARD_VISIBLE")
        _run((_case(0),), _config())
        target.info("AFTER_GUARD_VISIBLE")
        handler.flush()
        captured = stream.getvalue()
        assert "BEFORE_GUARD_VISIBLE" in captured
        assert private_canary not in captured
        assert "AFTER_GUARD_VISIBLE" in captured
        assert tuple(handler.filters) == filters_before
        assert logging.root.manager.disable == previous_threshold
    finally:
        target.removeHandler(handler)
        target.setLevel(previous_level)
        target.propagate = previous_propagate
        target.disabled = previous_disabled
