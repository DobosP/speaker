"""Tests for the Open Interpreter action brain.

A fake interpreter is injected directly (brain._interpreter) so these tests run
without open-interpreter installed -- the real OI chunk/approval behavior is
verified manually on a machine, per the plan.
"""
import pytest

from utils.agent_brain import (
    AgentBrain,
    AgentBrainConfig,
    SAFE,
    BLOCKED,
    NEEDS_CONFIRM,
)


class FakeInterpreter:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.messages = ["seed"]
        self.auto_run = False

    def chat(self, message, stream=True, display=False):
        for chunk in self._chunks:
            yield chunk


def _brain(chunks, **cfg):
    config = AgentBrainConfig(
        allowlist=cfg.pop("allowlist", ("^ls", "^pwd")),
        denylist=cfg.pop("denylist", ("rm -rf",)),
        **cfg,
    )
    brain = AgentBrain(config)
    brain._interpreter = FakeInterpreter(chunks)  # bypass lazy OI import
    return brain


def _msg(content, **extra):
    chunk = {"role": "assistant", "type": "message", "content": content}
    chunk.update(extra)
    return chunk


def _confirm(code, language="shell"):
    return {
        "role": "computer",
        "type": "confirmation",
        "format": "execution",
        "content": {"code": code, "language": language},
    }


def _console(output):
    return {"role": "computer", "type": "console", "format": "output", "content": output}


def _spoken(events):
    return " ".join(e.text for e in events if e.kind == "speak")


def test_classify_allow_deny_needsconfirm():
    brain = _brain([])
    assert brain.classify("ls -la") == SAFE
    assert brain.classify("rm -rf /") == BLOCKED
    assert brain.classify("python3 do_something()") == NEEDS_CONFIRM


def test_stream_speaks_message_and_console_output():
    chunks = [_msg("Listing "), _msg("your files."), _msg("", end=True), _console("a.txt\nb.txt")]
    events = list(_brain(chunks).stream_run("list files"))
    assert "Listing your files." in _spoken(events)
    assert any(e.kind == "result" and "a.txt" in e.text for e in events)


def test_safe_confirmation_runs_without_refusal():
    events = list(_brain([_confirm("ls -la"), _console("a.txt")]).stream_run("list"))
    assert any(e.kind == "confirm" and e.text == SAFE for e in events)
    assert "won't run" not in _spoken(events)
    assert any(e.kind == "result" and "a.txt" in e.text for e in events)


def test_blocked_confirmation_refuses():
    events = list(_brain([_confirm("rm -rf /")]).stream_run("wipe disk"))
    assert any(e.kind == "confirm" and e.text == BLOCKED for e in events)
    assert "won't run" in _spoken(events)


def test_needs_confirm_auto_safe_refuses():
    brain = _brain([_confirm("python3 -c 'print(1)'")], confirm_mode="auto_safe")
    assert "won't run" in _spoken(list(brain.stream_run("run code")))


def test_needs_confirm_always_ask_invokes_callback_and_runs():
    chunks = [_confirm("python3 -c 'print(1)'"), _console("1")]
    brain = _brain(chunks, confirm_mode="always_ask")
    calls = []

    def on_confirm(code, language):
        calls.append((code, language))
        return True

    events = list(brain.stream_run("run code", on_confirm=on_confirm))
    assert calls and calls[0][0] == "python3 -c 'print(1)'"
    assert "won't run" not in _spoken(events)


def test_needs_confirm_always_ask_denied_refuses():
    brain = _brain([_confirm("python3 -c 'print(1)'")], confirm_mode="always_ask")
    events = list(brain.stream_run("run code", on_confirm=lambda c, l: False))
    assert "won't run" in _spoken(events)


def test_cancel_stops_stream_early():
    state = {"first": True}

    def should_cancel():
        if state["first"]:
            state["first"] = False
            return False
        return True

    chunks = [_msg("one. "), _msg("two. "), _msg("three.")]
    events = list(_brain(chunks).stream_run("count", should_cancel=should_cancel))
    assert "three" not in _spoken(events)


def test_max_output_chars_truncates():
    brain = _brain([_console("x" * 5000)], max_output_chars=100)
    results = [e.text for e in brain.stream_run("dump") if e.kind == "result"]
    assert results and len(results[0]) < 200 and "truncated" in results[0]


def test_missing_open_interpreter_raises_runtimeerror():
    # No fake injected -> lazy import path. OI is not installed here, so
    # _ensure_interpreter raises RuntimeError (surfaced on first iteration).
    brain = AgentBrain(AgentBrainConfig())
    with pytest.raises(RuntimeError):
        list(brain.stream_run("do something"))
