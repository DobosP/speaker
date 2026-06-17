"""llm-inference-3: on-device output is bounded.

A device_profile's ``llm.options`` is written in Ollama's vocabulary
(``num_ctx`` / ``num_predict``), but ``LlamaCppLLM`` drives llama.cpp's
``create_chat_completion``, whose output cap is ``max_tokens`` and whose context
size is a CONSTRUCTOR arg. Without translation the cap was silently ignored and
the model generated to the context limit on the weakest hardware. These pin the
translation (no models -- the llama_cpp import is lazy; a fake client records the
request kwargs)."""
from __future__ import annotations

from core.llm import LlamaCppLLM, _normalize_llamacpp_options


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
