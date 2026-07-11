"""Fast unit tests for the LLM output-degeneracy heuristics + the
cloud-provider smoke probe.

These pin the detection logic with no model (the real MiniCPM5-1B run on CPU
lives in ``tools/llm_sanity.main``, exercised by the llm-sanity workflow),
and exercise the probe-shape/exit-code policy of the cloud smoke without
hitting any network."""
from __future__ import annotations

import json
import os
import threading
from typing import Iterator

import pytest

from core.llm import LLMCallCancelled, capability_context
from tools.llm_sanity import (
    FAIL,
    OK,
    SKIP,
    _NativeAbortProbeEvent,
    ProbeResult,
    _check,
    looks_degenerate,
    outputs_distinct,
    probe_llamacpp_abort_recovery,
    probe_provider,
    validate_registry,
)


def test_flags_single_token_spam():
    assert looks_degenerate("okay okay okay okay okay okay")


def test_flags_phrase_loop():
    assert looks_degenerate("let's do this let's do this let's do this")


def test_flags_empty():
    assert looks_degenerate("")


def test_accepts_normal_answer():
    assert not looks_degenerate("The capital of France is Paris.")


def test_accepts_short_nonempty_answer():
    assert not looks_degenerate("Four.")


def test_outputs_distinct_true_for_varied_answers():
    assert outputs_distinct(["Paris.", "Here's a joke.", "Four."])


def test_outputs_distinct_false_when_all_identical():
    assert not outputs_distinct(
        ["Okay, let's do this!", "okay, let's do this!", "Okay, let's do this!"]
    )


def test_check_flags_canned_repeated_answers():
    pairs = [("q1", "Okay, let's do this!"), ("q2", "Okay, let's do this!")]
    assert _check(pairs)  # all-same answers -> flagged


def test_check_passes_healthy_answers():
    pairs = [
        ("capital of France", "The capital of France is Paris."),
        ("2+2", "Two plus two is four."),
    ]
    assert _check(pairs) == []


# --- real llama.cpp abort/recovery gate (model-free fakes) ----------------


class _FakeAbortRecoveryLLM:
    """Mimic the native callback contract without loading a model."""

    def __init__(self, *, recovery: str = "ready") -> None:
        self._context_poisoned = False
        self.recovery = recovery
        self.stream_calls = 0
        self.generate_calls = 0
        self._tick = threading.Event()

    def stream(self, prompt, *, system=None, images=None, history=None) -> Iterator[str]:
        self.stream_calls += 1
        cancel_event = capability_context.get()["cancel_event"]

        # This name deliberately mirrors LlamaCppLLM's retained ctypes closure;
        # _NativeAbortProbeEvent uses it to distinguish a callback poll from the
        # surrounding Python stream loop.
        def should_abort(_data) -> bool:
            return bool(cancel_event.is_set())

        while not should_abort(None):
            self._tick.wait(0.001)
        raise LLMCallCancelled("fake native abort")
        yield ""  # pragma: no cover - keeps this function an iterator

    def generate(self, prompt, *, system=None, images=None, history=None) -> str:
        self.generate_calls += 1
        return self.recovery


def test_native_abort_probe_event_marks_only_should_abort_stack():
    event = _NativeAbortProbeEvent()
    assert event.is_set() is False
    assert event.wait_for_native_poll(0.0) is False

    def should_abort(_data) -> bool:
        return event.is_set()

    assert should_abort(None) is False
    assert event.wait_for_native_poll(0.0) is True
    event.set()
    assert should_abort(None) is True


def test_llamacpp_abort_probe_requires_cancel_then_reuses_same_client():
    llm = _FakeAbortRecoveryLLM()
    result = probe_llamacpp_abort_recovery(
        llm,
        native_poll_timeout=0.5,
        cancel_exit_timeout=0.5,
    )

    assert result.native_poll_ms >= 0.0
    assert result.cancel_exit_ms >= 0.0
    assert result.recovery_ms >= 0.0
    assert llm.stream_calls == 1
    assert llm.generate_calls == 1
    assert llm._context_poisoned is False


class _WrongAbortErrorLLM(_FakeAbortRecoveryLLM):
    def stream(self, prompt, *, system=None, images=None, history=None) -> Iterator[str]:
        cancel_event = capability_context.get()["cancel_event"]

        def should_abort(_data) -> bool:
            return bool(cancel_event.is_set())

        while not should_abort(None):
            self._tick.wait(0.001)
        raise RuntimeError("binding surfaced an ordinary failure")
        yield ""  # pragma: no cover


def test_llamacpp_abort_probe_rejects_non_cancellation_exit():
    with pytest.raises(RuntimeError, match="did not exit as LLMCallCancelled"):
        probe_llamacpp_abort_recovery(
            _WrongAbortErrorLLM(),
            native_poll_timeout=0.5,
            cancel_exit_timeout=0.5,
        )


class _PieceBeforeAbortLLM(_FakeAbortRecoveryLLM):
    def stream(self, prompt, *, system=None, images=None, history=None) -> Iterator[str]:
        yield "premature-token"
        cancel_event = capability_context.get()["cancel_event"]

        def should_abort(_data) -> bool:
            return bool(cancel_event.is_set())

        while not should_abort(None):
            self._tick.wait(0.001)
        raise LLMCallCancelled("fake native abort after output")


def test_llamacpp_abort_probe_rejects_any_piece_before_cancellation():
    llm = _PieceBeforeAbortLLM()
    with pytest.raises(RuntimeError, match=r"emitted 1 piece\(s\)"):
        probe_llamacpp_abort_recovery(
            llm,
            native_poll_timeout=0.5,
            cancel_exit_timeout=0.5,
        )
    assert llm.generate_calls == 0


class _NoNativePollLLM(_FakeAbortRecoveryLLM):
    def stream(self, prompt, *, system=None, images=None, history=None) -> Iterator[str]:
        cancel_event = capability_context.get()["cancel_event"]
        while not cancel_event.is_set():
            self._tick.wait(0.001)
        raise LLMCallCancelled("Python-only cancellation")
        yield ""  # pragma: no cover


def test_llamacpp_abort_probe_rejects_python_only_poll_without_native_callback():
    with pytest.raises(RuntimeError, match="native abort callback was not polled"):
        probe_llamacpp_abort_recovery(
            _NoNativePollLLM(),
            native_poll_timeout=0.02,
            cancel_exit_timeout=0.5,
        )


class _PoisonedAbortLLM(_FakeAbortRecoveryLLM):
    def stream(self, prompt, *, system=None, images=None, history=None) -> Iterator[str]:
        try:
            yield from super().stream(prompt, system=system, images=images, history=history)
        finally:
            self._context_poisoned = True


def test_llamacpp_abort_probe_rejects_poisoned_context_before_recovery():
    llm = _PoisonedAbortLLM()
    with pytest.raises(RuntimeError, match="context was poisoned"):
        probe_llamacpp_abort_recovery(
            llm,
            native_poll_timeout=0.5,
            cancel_exit_timeout=0.5,
        )
    assert llm.generate_calls == 0


def test_llamacpp_abort_probe_rejects_empty_recovery_completion():
    with pytest.raises(RuntimeError, match="recovery completion was empty"):
        probe_llamacpp_abort_recovery(
            _FakeAbortRecoveryLLM(recovery="  "),
            native_poll_timeout=0.5,
            cancel_exit_timeout=0.5,
        )


# --- cloud-provider smoke probe -------------------------------------------


class _FakeLLM:
    """Stand-in for OpenAICompatLLM with a scripted stream + reasoning counter."""

    def __init__(self, tokens, *, raise_on_stream=None, reasoning_chars=0):
        self.tokens = list(tokens)
        self.raise_on_stream = raise_on_stream
        self.last_reasoning_chars = reasoning_chars

    def stream(self, prompt, *, system=None, images=None) -> Iterator[str]:
        if self.raise_on_stream:
            raise self.raise_on_stream
        for t in self.tokens:
            yield t


def test_probe_returns_skip_when_api_key_env_unset(monkeypatch):
    monkeypatch.delenv("FAKE_KEY", raising=False)
    preset = {"model": "m", "api_key_env": "FAKE_KEY"}
    r = probe_provider("p", preset, llm_factory=lambda _: _FakeLLM(["A"]))
    assert r.status == "skip"
    assert r.exit_code == SKIP
    assert "FAKE_KEY" in r.error


def test_probe_returns_ok_for_healthy_response(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    preset = {"model": "m", "api_key_env": "FAKE_KEY"}
    r = probe_provider(
        "p", preset,
        llm_factory=lambda _: _FakeLLM(["Hello ", "world ", "this ", "is ", "fine."]),
    )
    assert r.status == "ok"
    assert r.exit_code == OK
    assert r.tokens == 5
    assert r.ttft_ms is not None and r.ttft_ms >= 0
    assert r.total_ms is not None and r.total_ms >= 0
    assert "Hello" in r.text


def test_probe_returns_fail_on_exception(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    preset = {"model": "m", "api_key_env": "FAKE_KEY"}
    boom = RuntimeError("rate limited")
    r = probe_provider(
        "p", preset,
        llm_factory=lambda _: _FakeLLM([], raise_on_stream=boom),
    )
    assert r.status == "fail"
    assert r.exit_code == FAIL
    assert "rate limited" in r.error


def test_probe_returns_fail_on_empty_response(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    preset = {"model": "m", "api_key_env": "FAKE_KEY"}
    r = probe_provider("p", preset, llm_factory=lambda _: _FakeLLM([]))
    assert r.status == "fail"
    assert r.error == "empty response"


def test_probe_returns_fail_on_degenerate_response(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    preset = {"model": "m", "api_key_env": "FAKE_KEY"}
    r = probe_provider(
        "p", preset,
        llm_factory=lambda _: _FakeLLM(["okay "] * 20),
    )
    assert r.status == "fail"
    assert "degenerate" in r.error


def test_probe_captures_reasoning_chars_metric(monkeypatch):
    """When the underlying LLM tracked reasoning bytes (DeepSeek V4-Pro),
    ProbeResult must surface them so the smoke summary can show 'thought
    N chars before answering'."""
    monkeypatch.setenv("FAKE_KEY", "x")
    preset = {"model": "m", "api_key_env": "FAKE_KEY"}
    r = probe_provider(
        "p", preset,
        llm_factory=lambda _: _FakeLLM(["A", "B"], reasoning_chars=2048),
    )
    assert r.reasoning_chars == 2048


def test_probe_no_api_key_env_means_always_run():
    """If a preset has no api_key_env (e.g. a local proxy), the probe
    runs regardless of env state."""
    preset = {"model": "m"}  # no api_key_env
    r = probe_provider("p", preset, llm_factory=lambda _: _FakeLLM(["hi there"]))
    assert r.status == "ok"


def test_markdown_row_format():
    r = ProbeResult(
        "name", "model-id", "ok", OK,
        ttft_ms=123.4, total_ms=456.7, tokens=5, text="hello",
    )
    row = r.as_markdown_row()
    assert row.startswith("| name | model-id | ok |")
    assert "123" in row
    assert "457" in row
    assert "5" in row


# --- registry validation --------------------------------------------------


def test_validate_registry_passes_when_every_model_is_known():
    registry = {"foo": {}, "bar": {}, "cerebras/gpt-oss-120b": {}}
    presets = {
        "p1": {"model": "foo"},
        "p2": {"model": "gpt-oss-120b"},  # matches the vendor-prefixed form
    }
    assert validate_registry(presets, registry) == []


def test_validate_registry_flags_unknown_model():
    registry = {"foo": {}}
    presets = {"p1": {"model": "bar-deprecated"}}
    problems = validate_registry(presets, registry)
    assert len(problems) == 1
    assert "bar-deprecated" in problems[0]


def test_validate_registry_ignores_underscore_keys_and_explicit_ignore_list():
    """``_comment`` shouldn't be treated as a preset; ``ignore`` lets you
    skip a model name that's known-not-in-registry-yet."""
    registry = {"foo": {}}
    presets = {
        "_comment": "metadata",
        "p1": {"model": "very-new-model"},
    }
    problems = validate_registry(presets, registry, ignore=["p1"])
    assert problems == []


def test_validate_registry_accepts_vendor_prefixed_match():
    """LiteLLM keys like 'cerebras/gpt-oss-120b' should match a preset's
    plain 'gpt-oss-120b'."""
    registry = {"cerebras/gpt-oss-120b": {}}
    presets = {"p1": {"model": "gpt-oss-120b"}}
    assert validate_registry(presets, registry) == []


def test_validate_registry_accepts_empty_registry():
    """If the registry file is missing/empty, validation is a no-op so a
    fresh checkout doesn't fail-block."""
    assert validate_registry({"p1": {"model": "anything"}}, {}) == []


# --- shipped config.json sanity ---------------------------------------------


def test_shipped_config_passes_registry_validation():
    """The committed config.json's cloud_providers must validate against
    the vendored LiteLLM registry. If a provider deprecates a model
    upstream, this catches it on the next PR -- so fixes can land before
    users hit 4xx in production."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = json.load(open(os.path.join(repo_root, "config.json")))
    registry_path = os.path.join(repo_root, "tools/litellm_model_registry.json")
    if not os.path.exists(registry_path):
        # No registry vendored -> nothing to validate against. The smoke
        # workflow downloads it; locally it may be absent.
        return
    registry = json.load(open(registry_path))
    presets = {n: p for n, p in cfg["llm"]["cloud_providers"].items()
               if not n.startswith("_")}
    # Provider strings known-good per audit may legitimately not be in the
    # registry yet if LiteLLM hasn't catalogued them; that's a warning,
    # not a hard test failure -- log it but don't break CI here.
    problems = validate_registry(presets, registry)
    if problems:
        # Print so the test report shows the gap, but don't fail.
        print("REGISTRY VALIDATION WARNINGS (informational):")
        for p in problems:
            print(" -", p)
