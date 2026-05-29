"""Device profiles (desktop/phone) and the on-device llama.cpp LLM client.

All fakes -- no llama.cpp native lib, no Ollama, no models required.
"""
from __future__ import annotations

import argparse

from core.app import _apply_device_profile, _build_llms, _load_config
from core.config import apply_device_profile, deep_merge, load_config
from core.llm import LlamaCppLLM, OllamaLLM
from core.llm_factory import build_llms


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


# --- recursive deep-merge (cross-platform-2) ---------------------------------


def test_deep_merge_recurses_and_preserves_siblings():
    """A nested override updates one leaf and keeps its siblings at every
    level (the fix for the shallow merge that stranded sibling keys)."""
    base = {
        "llm": {
            "backend": "ollama",
            "options": {"num_ctx": 4096, "num_predict": 384},
            "cloud": {"enabled": False, "strategy": "hedge", "model": "x"},
        },
    }
    merged = deep_merge(base, {"llm": {"cloud": {"enabled": True}}})
    assert merged["llm"]["cloud"]["enabled"] is True       # leaf changed
    assert merged["llm"]["cloud"]["strategy"] == "hedge"   # sibling leaf survived
    assert merged["llm"]["cloud"]["model"] == "x"          # sibling leaf survived
    assert merged["llm"]["backend"] == "ollama"            # sibling subsection survived
    assert merged["llm"]["options"]["num_ctx"] == 4096     # untouched subsection survived
    # Inputs are not mutated.
    assert base["llm"]["cloud"]["enabled"] is False


def test_deep_merge_replaces_non_dict_and_dict_vs_scalar():
    """Scalars/lists replace wholesale; a dict replacing a scalar (or vice
    versa) also replaces -- matching the pre-deep-merge behaviour."""
    assert deep_merge({"x": 1, "y": [1, 2]}, {"x": 9, "y": [9]}) == {"x": 9, "y": [9]}
    assert deep_merge({"a": 1}, {"a": {"b": 2}}) == {"a": {"b": 2}}
    assert deep_merge({"a": {"b": 2}}, {"a": 1}) == {"a": 1}


def test_deep_merge_replaces_opaque_options_bag_wholesale():
    """``llm.options`` is a backend-specific opaque bag: an override REPLACES it
    wholesale (not merged), so switching backend can't inherit the base
    backend's params. Base Ollama ``num_ctx`` must NOT survive into a llamacpp
    profile's options (it would TypeError when spread into create_chat_completion)."""
    base = {"llm": {"backend": "ollama", "options": {"num_ctx": 4096}}}
    over = {"llm": {"backend": "llamacpp", "options": {"temperature": 0.7}}}
    merged = deep_merge(base, over)
    assert merged["llm"]["options"] == {"temperature": 0.7}
    assert "num_ctx" not in merged["llm"]["options"]
    assert merged["llm"]["backend"] == "llamacpp"  # sibling still merged normally


def test_shipped_phone_profile_options_drop_ollama_only_keys():
    """Regression for the deep-merge fix: the SHIPPED phone profile (llamacpp)
    must not inherit the base Ollama-only ``num_ctx`` in llm.options, which
    would crash the first on-device generation."""
    config = _apply_device_profile(_load_config(), "phone")
    assert "num_ctx" not in config["llm"].get("options", {})


def test_profile_overriding_one_nested_key_keeps_cloud_siblings():
    """A device_profile that flips only ``llm.cloud.enabled`` must keep the
    sibling ``cloud_providers`` / ``cloud_chains`` -- the shallow merge used to
    replace the whole ``llm`` (or ``llm.cloud``) sub-dict and silently disable
    the configured cloud tier (cross-platform-2)."""
    config = {
        "llm": {
            "backend": "ollama",
            "main_model": "gemma3:12b",
            "cloud": {"enabled": False, "strategy": "hedge", "timeout_s": 20},
            "cloud_providers": {"openrouter": {"model": "m", "host": "US"}},
            "cloud_chains": {"public": ["openrouter"]},
        },
        "device_profiles": {
            "desktop_gpu_4090": {
                "llm": {"cloud": {"enabled": True}},
            }
        },
    }
    merged = apply_device_profile(config, "desktop_gpu_4090")
    llm = merged["llm"]
    assert llm["cloud"]["enabled"] is True            # the override landed
    assert llm["cloud"]["strategy"] == "hedge"        # sibling leaf in cloud survived
    assert llm["cloud"]["timeout_s"] == 20            # sibling leaf in cloud survived
    # The siblings of ``cloud`` -- the whole reason the cloud tier works -- are
    # not stranded.
    assert llm["cloud_providers"] == {"openrouter": {"model": "m", "host": "US"}}
    assert llm["cloud_chains"] == {"public": ["openrouter"]}
    assert llm["backend"] == "ollama"
    assert llm["main_model"] == "gemma3:12b"
    # Base config is not mutated.
    assert config["llm"]["cloud"]["enabled"] is False


def test_profile_non_nested_override_still_replaces_whole_section_value():
    """A non-nested override (scalar / new key) behaves exactly as before: the
    value is taken from the profile, base siblings in the same section survive."""
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
    phone = apply_device_profile(config, "phone")
    assert phone["llm"]["backend"] == "llamacpp"          # scalar replaced
    assert phone["llm"]["host"] == "keep"                 # base sibling survived
    assert phone["llm"]["main_model_path"] == "m.gguf"    # new key added
    assert phone["llm"]["main_model"] == "gemma3:12b"     # base sibling survived
    assert phone["sherpa"]["num_threads"] == 2
    assert phone["sherpa"]["sample_rate"] == 16000


def test_local_config_overlay_deep_merges_nested_keys(tmp_path, monkeypatch):
    """``config.local.json`` overriding a nested key keeps base siblings, same
    as device profiles (both go through deep_merge)."""
    import json

    (tmp_path / "config.json").write_text(json.dumps({
        "llm": {"backend": "ollama", "cloud": {"enabled": False, "strategy": "hedge"}},
    }), encoding="utf-8")
    (tmp_path / "config.local.json").write_text(json.dumps({
        "llm": {"cloud": {"enabled": True}},
    }), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SPEAKER_NO_LOCAL_CONFIG", raising=False)
    merged = load_config()
    assert merged["llm"]["cloud"]["enabled"] is True
    assert merged["llm"]["cloud"]["strategy"] == "hedge"   # sibling preserved
    assert merged["llm"]["backend"] == "ollama"            # sibling preserved


# --- public-module / remote-worker import surface ----------------------------


def test_public_modules_and_aliases_resolve():
    """The new public modules expose the factory + config transforms, and the
    historical private aliases still resolve to the same callables."""
    from core.app import _apply_device_profile as app_apply
    from core.app import _build_llms as app_build
    from core.app import _load_config as app_load

    # core.app re-exports the moved logic for back-compat callers.
    assert app_apply is apply_device_profile
    assert app_load is load_config
    assert app_build is build_llms


def test_remote_worker_imports_from_public_modules():
    """remote/worker.py resolves _load_config/_apply_device_profile/build_llms
    from the public modules (core.config / core.llm_factory), not core.app
    internals."""
    import remote.worker  # noqa: F401 - import must not raise

    # The names the worker now depends on live on the public modules.
    from core.config import apply_device_profile as cfg_apply
    from core.config import load_config as cfg_load
    from core.llm_factory import build_llms as factory_build

    assert callable(cfg_apply) and callable(cfg_load) and callable(factory_build)


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
