"""llm-inference-3: on-device output is bounded.

A device_profile's ``llm.options`` is written in Ollama's vocabulary
(``num_ctx`` / ``num_predict``), but ``LlamaCppLLM`` drives llama.cpp's
``create_chat_completion``, whose output cap is ``max_tokens`` and whose context
size is a CONSTRUCTOR arg. Without translation the cap was silently ignored and
the model generated to the context limit on the weakest hardware. These pin the
translation (no models -- the llama_cpp import is lazy; a fake client records the
request kwargs)."""
from __future__ import annotations

from core.llm import (
    LLAMACPP_PINNED_VERSION,
    LlamaCppLLM,
    _normalize_llamacpp_options,
    _resolve_kv_cache_type,
)


def _add_verified_abort_symbols(module) -> None:
    """Make a constructor fake satisfy the audited production import gate."""

    module.__version__ = LLAMACPP_PINNED_VERSION
    module.ggml_abort_callback = lambda callback: callback
    module.llama_set_abort_callback = lambda *_args: None
    module.llama_get_memory = lambda _ctx: object()
    module.llama_memory_clear = lambda *_args: None


# --- the pure translation -----------------------------------------------------


def test_num_predict_becomes_max_tokens():
    assert _normalize_llamacpp_options({"num_predict": 256}) == {"max_tokens": 256}


def test_num_ctx_and_keep_alive_dropped():
    out = _normalize_llamacpp_options({"num_ctx": 2048, "keep_alive": "5m", "temperature": 0.7})
    assert out == {"temperature": 0.7}  # constructor/daemon-only keys gone


def test_explicit_max_tokens_wins_over_num_predict():
    out = _normalize_llamacpp_options({"num_predict": 999, "max_tokens": 128})
    assert out == {"max_tokens": 128}


def test_empty_and_none_are_safe():
    assert _normalize_llamacpp_options(None) == {}
    assert _normalize_llamacpp_options({}) == {}


def test_unrelated_options_pass_through():
    out = _normalize_llamacpp_options({"temperature": 0.5, "top_p": 0.9})
    assert out == {"temperature": 0.5, "top_p": 0.9}


# --- end to end through the client --------------------------------------------


class _RecordingClient:
    """Stand-in llama_cpp.Llama: records the per-request kwargs it receives."""

    def __init__(self):
        self.calls: list[dict] = []

    def create_chat_completion(self, *, messages, stream=False, **opts):
        self.calls.append(opts)
        if stream:
            return iter([{"choices": [{"delta": {"content": "hi"}}]}])
        return {"choices": [{"message": {"content": "hi"}}]}


def test_stream_sends_max_tokens_not_ollama_keys():
    client = _RecordingClient()
    llm = LlamaCppLLM(
        "/x.gguf",
        options={"num_predict": 256, "num_ctx": 2048, "temperature": 0.7},
        client=client,
    )
    out = "".join(llm.stream("hello"))
    assert out == "hi"
    opts = client.calls[-1]
    assert opts["max_tokens"] == 256          # the cap actually reaches llama.cpp
    assert "num_predict" not in opts          # Ollama output key translated away
    assert "num_ctx" not in opts              # constructor-only key not sent per request
    assert opts["temperature"] == 0.7


def test_generate_sends_max_tokens_not_ollama_keys():
    client = _RecordingClient()
    llm = LlamaCppLLM("/x.gguf", options={"num_predict": 200}, client=client)
    assert llm.generate("hello") == "hi"
    assert client.calls[-1]["max_tokens"] == 200


def test_comment_keys_dropped():
    # config.json's '_'-prefixed human comments must never reach the model API.
    out = _normalize_llamacpp_options({"_num_predict_comment": "doc", "num_predict": 256})
    assert out == {"max_tokens": 256}


def test_numeric_string_and_float_caps_coerce():
    assert _normalize_llamacpp_options({"num_predict": "256"}) == {"max_tokens": 256}
    assert _normalize_llamacpp_options({"num_predict": 256.0}) == {"max_tokens": 256}


class _StrictClient:
    """Mirrors the REAL llama_cpp.Llama.create_chat_completion: a fixed keyword
    signature with NO **kwargs, so a stray option key raises TypeError exactly
    as the library would on-device (the lenient **kwargs fakes hide this)."""

    def __init__(self):
        self.calls: list[dict] = []

    def create_chat_completion(self, *, messages, stream=False, temperature=None,
                               top_p=None, max_tokens=None):
        self.calls.append({"max_tokens": max_tokens, "temperature": temperature})
        if stream:
            return iter([{"choices": [{"delta": {"content": "hi"}}]}])
        return {"choices": [{"message": {"content": "hi"}}]}


# --- llm-inference-9: KV-cache quantization plumbing --------------------------


def test_resolve_kv_cache_type():
    assert _resolve_kv_cache_type("q8_0") == 8       # friendly name -> ggml int
    assert _resolve_kv_cache_type("f16") == 1
    assert _resolve_kv_cache_type(8) == 8            # raw int passes through
    assert _resolve_kv_cache_type("nonsense") is None  # typo -> default (None)
    assert _resolve_kv_cache_type(None) is None


def test_kv_cache_types_resolved_at_construction():
    llm = LlamaCppLLM("/x.gguf", type_k="q8_0", type_v="q8_0", client=object())
    assert llm.type_k == 8 and llm.type_v == 8
    plain = LlamaCppLLM("/x.gguf", client=object())
    assert plain.type_k is None and plain.type_v is None


def test_ensure_forwards_kv_cache_types_to_llama(monkeypatch):
    import sys
    import types

    captured: dict = {}

    class _FakeLlama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_mod = types.ModuleType("llama_cpp")
    fake_mod.Llama = _FakeLlama
    _add_verified_abort_symbols(fake_mod)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_mod)

    LlamaCppLLM(
        "/x.gguf",
        type_k="q8_0",
        type_v="q8_0",
        n_threads=2,
        n_threads_batch=3,
    )._ensure()  # no client -> builds
    assert captured["type_k"] == 8 and captured["type_v"] == 8
    assert captured["model_path"] == "/x.gguf"
    assert captured["n_threads"] == 2 and captured["n_threads_batch"] == 3

    captured.clear()
    LlamaCppLLM("/x.gguf")._ensure()
    assert "type_k" not in captured and "type_v" not in captured  # default: not forwarded


def test_ensure_degrades_when_llama_lacks_kv_quant_kwargs(monkeypatch):
    import sys
    import types

    seen: list = []

    class _OldLlama:
        def __init__(self, **kwargs):
            seen.append(kwargs)
            if "type_k" in kwargs or "type_v" in kwargs:
                raise TypeError("__init__() got an unexpected keyword argument 'type_k'")

    fake_mod = types.ModuleType("llama_cpp")
    fake_mod.Llama = _OldLlama
    _add_verified_abort_symbols(fake_mod)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_mod)

    # An old lib must NOT crash the first turn -- it degrades to the f16 default.
    LlamaCppLLM("/x.gguf", type_k="q8_0", type_v="q8_0")._ensure()
    assert len(seen) == 2                                   # first attempt + the retry
    assert "type_k" not in seen[-1] and "type_v" not in seen[-1]  # retried without KV-quant


def test_shipped_on_device_profiles_drive_a_strict_client_without_typeerror():
    # Regression for the _num_predict_comment leak: the phone/phone_lite merged
    # options must drive the REAL (strict-signature) client cleanly.
    import json
    from pathlib import Path

    from core.config import apply_device_profile

    base = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
    for tier, cap in (("phone", 384), ("phone_lite", 256)):
        opts = apply_device_profile(base, tier)["llm"]["options"]
        client = _StrictClient()
        llm = LlamaCppLLM("/x.gguf", options=opts, client=client)
        out = "".join(llm.stream("hello"))   # would TypeError if a stray key leaked
        assert out == "hi"
        assert client.calls[-1]["max_tokens"] == cap
