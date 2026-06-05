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
        self.images: list = []  # the images= each call received (None when text-only)

    def generate(self, prompt, *, system=None, images=None) -> str:
        self.prompts.append(prompt)
        self.images.append(images)
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


# --- visual context: a frame reaches the model through the CAPABILITY --------
#
# test_multimodal_e2e pins the LLM-chain forwarding; these pin the layer above
# it -- the assistant capability actually reading an image off the turn and
# forwarding it, routing it to the main/multimodal tier, and the VoiceRuntime
# set_current_frame() feed that makes a host machine's frame ambient.

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32  # a token PNG header -- enough for a fake


def test_per_turn_image_forwards_to_model_and_forces_main_tier():
    main = RecordingLLM("main")
    fast = RecordingLLM("fast")
    registry = attach_llm_capabilities(CapabilityRegistry(), main, fast_llm=fast)
    # "what is this" alone routes to fast; an attached image forces the main tier.
    answer = registry.invoke("assistant.answer", "what is this", {"images": [_PNG]})
    assert answer.text.startswith("[main]")          # routed to the multimodal main tier
    assert main.images and main.images[-1] == [_PNG]  # the image bytes reached the model
    assert not fast.prompts                            # the fast (non-multimodal) tier never ran


def test_text_only_turn_carries_no_images():
    main = RecordingLLM("main")
    fast = RecordingLLM("fast")
    registry = attach_llm_capabilities(CapabilityRegistry(), main, fast_llm=fast)
    registry.invoke("assistant.answer", "what time is it")
    assert fast.images == [None]   # answered on fast tier, no images attached
    assert not main.prompts


def test_ambient_image_provider_feeds_turns_then_clears():
    main = RecordingLLM("main")
    fast = RecordingLLM("fast")
    box = {"frame": _PNG}
    registry = attach_llm_capabilities(
        CapabilityRegistry(), main, fast_llm=fast,
        image_provider=lambda: ([box["frame"]] if box["frame"] is not None else None),
    )
    # With a frame set, even a simple turn goes to main carrying the frame.
    registry.invoke("assistant.answer", "what time is it")
    assert main.images and main.images[-1] == [_PNG]
    # Clear the frame -> back to text-only on the fast tier.
    main.prompts.clear(); main.images.clear()
    box["frame"] = None
    registry.invoke("assistant.answer", "what time is it")
    assert not main.prompts and fast.prompts


def test_per_turn_image_overrides_ambient_provider():
    main = RecordingLLM("main")
    registry = attach_llm_capabilities(
        CapabilityRegistry(), main, image_provider=lambda: [b"AMBIENT"],
    )
    registry.invoke("assistant.answer", "what is this", {"images": [_PNG]})
    assert main.images[-1] == [_PNG]  # the per-turn image wins over the ambient frame


def test_runtime_set_current_frame_feeds_the_capability():
    from always_on_agent.events import Mode
    from core.engines.scripted import ScriptedEngine
    from core.runtime import VoiceRuntime

    main = RecordingLLM("main")
    rt = VoiceRuntime(ScriptedEngine(), main, start_mode=Mode.ASSISTANT)
    rt.set_current_frame(_PNG)
    rt.supervisor.capabilities.invoke("assistant.answer", "what is this", {})
    assert main.images and main.images[-1] == [_PNG]  # the host frame reached the model
    # Clearing it returns to text-only.
    main.prompts.clear(); main.images.clear()
    rt.clear_current_frame()
    rt.supervisor.capabilities.invoke("assistant.answer", "what is this", {})
    assert main.images[-1] is None


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
