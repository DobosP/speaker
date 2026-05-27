"""Device profiles (desktop/phone) and the on-device llama.cpp LLM client.

All fakes -- no llama.cpp native lib, no Ollama, no models required.
"""
from __future__ import annotations

import argparse

from core.app import _apply_device_profile, _build_llms, _load_config
from core.llm import LlamaCppLLM, OllamaLLM


class FakeLlama:
    """Stands in for llama_cpp.Llama; records create_chat_completion calls."""

    def __init__(self, reply: str = "hi"):
        self.reply = reply
        self.calls: list[dict] = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([{"choices": [{"delta": {"content": self.reply}}]}])
        return {"choices": [{"message": {"content": self.reply}}]}


def _args(**kw) -> argparse.Namespace:
    base = dict(llm="ollama", model=None, fast_model=None)
    base.update(kw)
    return argparse.Namespace(**base)


# --- device profile merge ----------------------------------------------------


def test_profile_overrides_are_shallow_merged_per_section():
    config = {
        "llm": {"backend": "ollama", "host": "keep", "main_model": "gemma3:12b"},
        "sherpa": {"num_threads": 0, "sample_rate": 16000},
        "device_profiles": {
            "phone": {
                "llm": {"backend": "llamacpp", "main_model_path": "m.gguf"},
                "sherpa": {"num_threads": 2},
            }
        },
    }
    phone = _apply_device_profile(config, "phone")
    assert phone["llm"]["backend"] == "llamacpp"
    assert phone["llm"]["host"] == "keep"  # untouched base field survives
    assert phone["llm"]["main_model_path"] == "m.gguf"
    assert phone["sherpa"]["num_threads"] == 2
    assert phone["sherpa"]["sample_rate"] == 16000
    # base config is not mutated
    assert config["llm"]["backend"] == "ollama"


def test_unknown_device_is_noop():
    config = {"llm": {"backend": "ollama"}}
    assert _apply_device_profile(config, "watch") is config


def test_shipped_config_phone_profile_uses_small_gemma_on_llamacpp():
    config = _apply_device_profile(_load_config(), "phone")
    llm_cfg = config["llm"]
    assert llm_cfg["backend"] == "llamacpp"
    assert "gemma-3-4b" in llm_cfg["main_model_path"]
    assert "gemma-3-1b" in llm_cfg["fast_model_path"]
    # STT/TTS threads dialed down for a phone CPU
    assert config["sherpa"]["num_threads"] == 2


def test_shipped_config_desktop_stays_on_ollama():
    config = _apply_device_profile(_load_config(), "desktop")
    assert config["llm"]["backend"] == "ollama"
    assert config["llm"]["main_model"] == "gemma3:12b"


# --- backend selection in _build_llms ---------------------------------------


def test_build_llms_phone_profile_builds_llamacpp_clients():
    config = _apply_device_profile(_load_config(), "phone")
    main, fast = _build_llms(_args(), config)
    assert isinstance(main, LlamaCppLLM) and isinstance(fast, LlamaCppLLM)
    assert main.model_path.endswith("gemma-3-4b-it-Q4_K_M.gguf")
    assert fast.model_path.endswith("gemma-3-1b-it-Q4_K_M.gguf")
    assert main.n_ctx == 2048 and main.n_gpu_layers == 0


def test_build_llms_desktop_profile_builds_ollama_clients():
    config = _apply_device_profile(_load_config(), "desktop")
    main, fast = _build_llms(_args(), config)
    assert isinstance(main, OllamaLLM) and isinstance(fast, OllamaLLM)
    assert main.model == "gemma3:12b" and fast.model == "gemma3:4b"


def test_llamacpp_backend_requires_main_path():
    config = {"llm": {"backend": "llamacpp"}}
    try:
        _build_llms(_args(), config)
        assert False, "expected SystemExit"
    except SystemExit:
        pass


# --- on-device LLM client ----------------------------------------------------


def test_llamacpp_generate_uses_chat_completion():
    fake = FakeLlama(reply="answer")
    llm = LlamaCppLLM("m.gguf", client=fake, options={"temperature": 0.5})
    out = llm.generate("question", system="be brief")
    assert out == "answer"
    call = fake.calls[0]
    assert call["temperature"] == 0.5
    assert call["messages"][0] == {"role": "system", "content": "be brief"}
    assert call["messages"][-1] == {"role": "user", "content": "question"}


def test_llamacpp_stream_yields_content_deltas():
    fake = FakeLlama(reply="streamed")
    assert list(LlamaCppLLM("m.gguf", client=fake).stream("hi")) == ["streamed"]
    assert fake.calls[0]["stream"] is True


def test_llamacpp_formats_multimodal_content_for_images():
    fake = FakeLlama()
    LlamaCppLLM("m.gguf", client=fake).generate("describe", images=["/tmp/a.png", b"raw"])
    content = fake.calls[0]["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["image_url"]["url"] == "/tmp/a.png"
    assert content[2]["image_url"]["url"].startswith("data:image/png;base64,")
