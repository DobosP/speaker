"""Multimodal LLM client, two-model capability routing, and CPU thread tuning.

All fakes -- no Ollama, no sherpa-onnx, no GPU required.
"""
from __future__ import annotations

from typing import Iterator, Optional, Sequence

from always_on_agent.capabilities import CapabilityRegistry
from core.capabilities import attach_llm_capabilities
from core.engines.sherpa import SherpaConfig, _auto_threads
from core.llm import EchoLLM, OllamaLLM


class FakeOllamaClient:
    """Records the kwargs passed to ``chat`` and returns a canned reply."""

    def __init__(self, reply: str = "ok"):
        self.reply = reply
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([{"message": {"content": self.reply}}])
        return {"message": {"content": self.reply}}


class RecordingLLM:
    """LLMClient fake that tags its output so we can see which model ran."""

    def __init__(self, tag: str):
        self.tag = tag
        self.prompts: list[str] = []

    def generate(self, prompt, *, system=None, images=None) -> str:
        self.prompts.append(prompt)
        return f"[{self.tag}] {prompt}"

    def stream(self, prompt, *, system=None, images=None) -> Iterator[str]:
        yield self.generate(prompt, system=system, images=images)


# --- multimodal LLM client ---------------------------------------------------


def test_ollama_attaches_images_to_user_message():
    client = FakeOllamaClient()
    llm = OllamaLLM(model="gemma3:12b", client=client)
    llm.generate("what is this", images=["/tmp/a.png", b"rawbytes"])
    msg = client.calls[0]["messages"][-1]
    assert msg["role"] == "user"
    assert msg["images"] == ["/tmp/a.png", b"rawbytes"]


def test_ollama_omits_images_key_when_none():
    client = FakeOllamaClient()
    OllamaLLM(model="gemma3:4b", client=client).generate("hello")
    assert "images" not in client.calls[0]["messages"][-1]


def test_ollama_passes_options_through():
    client = FakeOllamaClient()
    llm = OllamaLLM(model="gemma3:12b", client=client, options={"num_ctx": 4096})
    llm.generate("hi")
    assert client.calls[0]["options"] == {"num_ctx": 4096}


def test_ollama_stream_yields_pieces():
    client = FakeOllamaClient(reply="streamed")
    out = list(OllamaLLM(client=client).stream("hi"))
    assert out == ["streamed"]
    assert client.calls[0]["stream"] is True


def test_echo_llm_tolerates_images():
    out = EchoLLM().generate("look", images=["/tmp/x.png"])
    assert "1 image" in out


# --- two-model routing -------------------------------------------------------


def test_assistant_uses_fast_model_research_uses_main():
    main = RecordingLLM("main")
    fast = RecordingLLM("fast")
    registry = attach_llm_capabilities(CapabilityRegistry(), main, fast_llm=fast)

    answer = registry.invoke("assistant.answer", "what time is it")
    assert answer.text.startswith("[fast]")
    assert fast.prompts and not main.prompts

    research = registry.invoke("research.local", "compare options", {"previous_steps": []})
    assert research.text.startswith("[main]")
    assert main.prompts


def test_single_model_when_no_fast_llm():
    main = RecordingLLM("main")
    registry = attach_llm_capabilities(CapabilityRegistry(), main)
    assert registry.invoke("assistant.answer", "hi").text.startswith("[main]")


# --- CPU thread tuning -------------------------------------------------------


def test_auto_threads_is_clamped():
    assert 2 <= _auto_threads() <= 4


def test_thread_resolution_prefers_explicit_overrides():
    cfg = SherpaConfig(num_threads=6, asr_num_threads=3, tts_num_threads=8)
    assert cfg.base_threads == 6
    assert cfg.resolved_asr_threads == 3
    assert cfg.resolved_tts_threads == 8


def test_threads_fall_back_to_base_then_auto():
    base = SherpaConfig(num_threads=5)
    assert base.resolved_asr_threads == 5 and base.resolved_tts_threads == 5

    auto = SherpaConfig()  # all zero -> auto-detect
    assert auto.resolved_asr_threads == _auto_threads()


def test_provider_defaults_to_cpu_and_from_dict_reads_new_fields():
    cfg = SherpaConfig.from_dict(
        {"provider": "cpu", "asr_num_threads": 2, "tts_num_threads": 4, "unknown": "x"}
    )
    assert cfg.provider == "cpu"
    assert cfg.resolved_asr_threads == 2
    assert cfg.resolved_tts_threads == 4
