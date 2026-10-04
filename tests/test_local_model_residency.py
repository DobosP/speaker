"""Opt-in warm/residency policies preserve the full local capability plane."""

from __future__ import annotations

from argparse import Namespace
import math
import threading

import pytest

from core.addressing import LLMAddressingClassifier
from core.engines.scripted import ScriptedEngine
from core.llm import HedgeLLM
from core.llm_factory import build_llms
from core.runtime import VoiceRuntime


class RecordingLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, *, system=None, images=None):
        self.calls.append((prompt, images))
        return "ok"

    def stream(self, prompt, *, system=None, images=None):
        yield self.generate(prompt, system=system, images=images)


def _args():
    return Namespace(llm="ollama", model=None, fast_model=None)


def _config(**overrides):
    return {
        "llm": {
            "backend": "ollama",
            "main_model": "main-large",
            "fast_model": "fast-small",
            "keep_alive": -1,
            "options": {"num_ctx": 4096, "num_predict": 512},
            **overrides,
        }
    }


def test_all_default_preserves_complete_warm_plan_and_fast_is_explicit():
    for policy, expected in (("all", (1, 1)), ("fast", (0, 1))):
        main, fast = RecordingLLM(), RecordingLLM()
        runtime = VoiceRuntime(
            ScriptedEngine(), main, fast_llm=fast, warm_start_policy=policy
        )
        runtime._warm()
        assert (len(main.calls), len(fast.calls)) == expected
        assert runtime.warm_ready.is_set()


@pytest.mark.parametrize("policy", [None, "", "Fast", " fast ", "none", True, 1, []])
def test_warm_policy_is_strict_before_engine_access(policy):
    class Untouched:
        @property
        def playback_capabilities(self):
            raise AssertionError("engine must remain untouched")

    with pytest.raises(ValueError, match="warm_start_policy"):
        VoiceRuntime(Untouched(), warm_start_policy=policy)


def test_fast_plan_keeps_media_and_shared_or_sole_tier_warm():
    media = []

    class Engine(ScriptedEngine):
        def warm(self):
            media.append(True)

    for has_fast in (False, True):
        model = RecordingLLM()
        runtime = VoiceRuntime(
            Engine(),
            model,
            fast_llm=model if has_fast else None,
            warm_start_policy="fast",
        )
        runtime._warm()
        assert len(model.calls) == 1
    assert media == [True, True]


def test_fast_warm_ready_is_not_blocked_by_unused_main_and_remains_one_worker():
    main, fast = RecordingLLM(), RecordingLLM()

    class Engine(ScriptedEngine):
        def warm(self):
            self.warm_thread = threading.get_ident()

    engine = Engine()
    runtime = VoiceRuntime(
        engine, main, fast_llm=fast, warm_on_start=True, warm_start_policy="fast"
    )
    runtime.start(run_bus=False)
    try:
        assert runtime.warm_ready.wait(timeout=2)
        assert main.calls == []
        assert len(fast.calls) == 1
        assert engine.warm_thread != threading.get_ident()
    finally:
        runtime.stop()


def test_cold_main_research_and_multimodal_capabilities_remain_usable():
    main, fast = RecordingLLM(), RecordingLLM()
    runtime = VoiceRuntime(
        ScriptedEngine(), main, fast_llm=fast, warm_start_policy="fast"
    )
    names = runtime.supervisor.capabilities.names()
    runtime._warm()
    assert main.calls == []
    assert runtime.supervisor.capabilities.names() == names
    assert runtime.supervisor.capabilities.invoke(
        "research.local", "compare synthetic choices", {"previous_steps": []}
    ).ok
    assert main.calls
    main.calls.clear()
    assert runtime.supervisor.capabilities.invoke(
        "assistant.answer", "describe synthetic frame", {"images": [b"synthetic-image"]}
    ).ok
    assert main.calls[-1][1] == [b"synthetic-image"]
    assert runtime.supervisor.capabilities.names() == names


def test_fast_plan_never_warms_cloud_hybrid_or_unused_local_main():
    local_main, fast_local, cloud = RecordingLLM(), RecordingLLM(), RecordingLLM()
    runtime = VoiceRuntime(
        ScriptedEngine(),
        HedgeLLM(local=local_main, cloud=cloud),
        fast_llm=fast_local,
        warm_start_policy="fast",
    )
    runtime._warm()
    assert len(fast_local.calls) == 1
    assert local_main.calls == cloud.calls == []
    fallback = VoiceRuntime(
        ScriptedEngine(),
        HedgeLLM(local=local_main, cloud=cloud),
        warm_start_policy="fast",
    )
    fallback._warm()
    assert len(local_main.calls) == 1
    assert cloud.calls == []


def test_bound_main_gate_is_not_indirectly_warmed_under_fast_policy():
    main, fast = RecordingLLM(), RecordingLLM()
    addressing = LLMAddressingClassifier(main)
    runtime = VoiceRuntime(
        ScriptedEngine(),
        main,
        fast_llm=fast,
        addressing=addressing,
        warm_start_policy="fast",
    )
    runtime._warm()
    assert main.calls == []
    assert len(fast.calls) == 1
    addressing.classify("synthetic ordinary utterance")
    assert main.calls  # The gate remains callable on demand.


def test_role_keepalive_preserves_options_caps_and_forwards_to_exact_local_requests():
    main, fast = build_llms(_args(), _config(main_keep_alive="60s", fast_keep_alive=-1))
    assert main._keep_alive == "60s" and fast._keep_alive == -1
    assert main._options == fast._options == {"num_ctx": 4096, "num_predict": 512}

    class Client:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return {"message": {"content": "ok"}}

    for llm, expected in ((main, "60s"), (fast, -1)):
        client = Client()
        llm._client = client
        assert llm.generate("synthetic request") == "ok"
        assert client.calls[-1]["keep_alive"] == expected
        assert client.calls[-1]["options"] == {"num_ctx": 4096, "num_predict": 512}


@pytest.mark.parametrize(
    "overrides,expected_main,expected_fast",
    [
        ({}, -1, -1),
        ({"main_keep_alive": "60s"}, "60s", -1),
        ({"fast_keep_alive": 0}, -1, 0),
        ({"main_keep_alive": None}, None, -1),
    ],
)
def test_role_retention_falls_back_independently(
    overrides, expected_main, expected_fast
):
    main, fast = build_llms(_args(), _config(**overrides))
    assert main._keep_alive == expected_main
    assert fast._keep_alive == expected_fast


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, -1),
        ({"main_keep_alive": "60s", "fast_keep_alive": -1}, -1),
        ({"main_keep_alive": "60s"}, -1),
        ({"fast_keep_alive": "10m"}, "10m"),
    ],
)
def test_shared_ollama_model_uses_fast_residency_without_collapsing_legacy_clients(
    overrides, expected
):
    main, fast = build_llms(
        _args(), _config(main_model="shared", fast_model="shared", **overrides)
    )
    assert main is not fast
    assert main._keep_alive == fast._keep_alive == expected


@pytest.mark.parametrize("key", ["main_keep_alive", "fast_keep_alive"])
@pytest.mark.parametrize("bad", [True, [], {}, "", math.inf, math.nan])
def test_invalid_new_retention_values_refuse_before_client_construction(
    monkeypatch, key, bad
):
    calls = []
    monkeypatch.setattr(
        "core.llm_factory.OllamaLLM", lambda **kwargs: calls.append(kwargs)
    )
    with pytest.raises(ValueError, match="role keep_alive"):
        build_llms(_args(), _config(**{key: bad}))
    assert calls == []


def test_runtime_builder_forwards_fast_policy_without_changing_caps(monkeypatch):
    from core import app
    from always_on_agent.events import Mode
    from always_on_agent.memory import SessionMemory

    monkeypatch.setattr(app, "_build_memory", lambda *_args: SessionMemory())
    main, fast = RecordingLLM(), RecordingLLM()
    runtime = app.build_runtime(
        {"warm_start_policy": "fast"},
        engine=ScriptedEngine(),
        llm=main,
        fast_llm=fast,
        router=None,
        start_mode=Mode.ASSISTANT,
    )
    runtime._warm()
    assert runtime._warm_start_policy == "fast"
    assert main.calls == [] and len(fast.calls) == 1
    assert {"assistant.answer", "research.local"} <= set(
        runtime.supervisor.capabilities.names()
    )
