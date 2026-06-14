"""Tests for the ported Open Interpreter action brain (core/agent.py).

A fake interpreter is injected (brain._interpreter) so these run without
open-interpreter installed -- the real OI chunk/approval behavior is verified
manually on a machine, per the plan.
"""
from threading import Event

from always_on_agent.capabilities import CapabilityRegistry
from always_on_agent.origin import Origin

from core.agent import (
    SAFE,
    BLOCKED,
    NEEDS_CONFIRM,
    AgentBrain,
    AgentBrainConfig,
    _is_local_model,
    attach_agent_capability,
)
from core.agent_os import detect_display_server, os_mode_preflight

# An owner-verified live-audio turn -- the only origin allowed to drive an action
# (security chokepoint). Capability tests that exercise the BRAIN pass this so the
# action gate admits them; the gate itself is tested separately below.
_OWNER_CTX = {"owner_verified": True, "origin": Origin.LIVE_AUDIO}


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
    brain._interpreter = FakeInterpreter(chunks)
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


# -- classification / model selection ----------------------------------------

def test_classify_allow_deny_needsconfirm():
    brain = _brain([])
    assert brain.classify("ls -la") == SAFE
    assert brain.classify("rm -rf /") == BLOCKED
    assert brain.classify("python3 do_something()") == NEEDS_CONFIRM


def test_is_local_model():
    assert _is_local_model("ollama/gemma3:latest")
    assert not _is_local_model("gpt-4o")


def test_local_only_blocks_cloud_model():
    brain = _brain([], model="gpt-4o", local_only=True)
    try:
        brain._effective_model()
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "local_only" in str(exc)


def test_local_only_falls_back_to_local():
    brain = _brain([], model="gpt-4o", local_only=True, local_fallback_model="ollama/gemma3")
    assert brain._effective_model() == "ollama/gemma3"


# -- streaming behavior -------------------------------------------------------

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


def test_needs_confirm_declined_in_auto_safe():
    events = list(_brain([_confirm("python3 x()")], confirm_mode="auto_safe").stream_run("run"))
    assert any(e.kind == "confirm" and e.text == NEEDS_CONFIRM for e in events)
    assert "won't run" in _spoken(events)


# -- capability wrapper (the new core seam) ----------------------------------

def _registry(chunks, **cfg):
    """Registry whose command.stage runs a brain backed by a fake interpreter."""
    registry = CapabilityRegistry()
    config = AgentBrainConfig(
        allowlist=cfg.pop("allowlist", ("^ls", "^pwd")),
        denylist=cfg.pop("denylist", ("rm -rf",)),
        **cfg,
    )
    attach_agent_capability(registry, config, brain=_brain(chunks, **cfg))
    return registry


def test_capability_registers_under_command_stage():
    registry = _registry([])
    assert "command.stage" in registry.names()


def test_capability_returns_spoken_text():
    registry = _registry([_msg("Done listing."), _msg("", end=True), _console("a.txt")])
    result = registry.invoke("command.stage", "list files", {"cancel_event": Event(), **_OWNER_CTX})
    assert result.ok
    assert result.data.get("executed") is True
    assert "Done listing." in result.text and "a.txt" in result.text


def test_capability_empty_instruction_errors():
    registry = _registry([])
    result = registry.invoke("command.stage", "  ", {})
    assert not result.ok
    assert "empty" in result.error


def test_capability_cancellation_stops_stream():
    registry = _registry([_msg("part one.", end=True), _msg("part two", end=True)])
    cancel = Event()
    cancel.set()  # cancelled before it starts
    result = registry.invoke("command.stage", "talk", {"cancel_event": cancel, **_OWNER_CTX})
    assert result.ok
    assert result.text == "Done."  # nothing spoken because cancelled immediately


# -- P0 security: the action-trust chokepoint + classify hardening -------------

def test_capability_blocked_without_owner_verification():
    # Default require_owner_verified=True: a turn with no owner-verified signal
    # (ambient/leaked audio, recalled/web/screen text) is REFUSED -- the brain never
    # runs. Fail-closed.
    registry = _registry([_msg("Done.", end=True)])
    result = registry.invoke("command.stage", "delete my files", {"cancel_event": Event()})
    assert result.ok
    assert result.data.get("executed") is False
    assert result.data.get("blocked") == "owner_verification"


def test_capability_blocked_for_untrusted_origin_even_if_verified():
    # An action whose lineage touched screen/web/memory is blocked regardless.
    registry = _registry([_msg("Done.", end=True)])
    ctx = {"cancel_event": Event(), "owner_verified": True, "origin": Origin.SCREEN}
    result = registry.invoke("command.stage", "click delete", ctx)
    assert result.data.get("blocked") == "owner_verification"


def test_capability_opt_out_runs_without_verification():
    # require_owner_verified=False restores the legacy (unverified) behavior knowingly.
    registry = _registry([_msg("ran.", end=True)], require_owner_verified=False)
    result = registry.invoke("command.stage", "do it", {"cancel_event": Event()})
    assert result.ok and result.data.get("executed") is True


def test_classify_never_auto_safe_blocks_allowlisted_code_exec():
    # The auto-RCE bypass: a broad allowlist that matches `python -c` must NOT
    # auto-run it -- the built-in never-auto-safe set downgrades it to needs-confirm.
    brain = _brain([], allowlist=("^python3?\\s+-c\\b", "^ls"), denylist=("rm -rf",))
    assert brain.classify('python -c "import shutil; shutil.rmtree(\'/x\')"') == NEEDS_CONFIRM
    assert brain.classify("ls; python -c 'x'") == NEEDS_CONFIRM   # chaining
    assert brain.classify("ls -la > /tmp/x") == NEEDS_CONFIRM     # redirect
    assert brain.classify("ls\nchmod 777 /etc/passwd") == NEEDS_CONFIRM  # newline chaining
    assert brain.classify("ls -la") == SAFE                       # still auto-safe
    assert brain.classify("rm -rf ~") == BLOCKED                  # denylist precedence


def test_config_allowlist_has_no_arbitrary_code_or_file_read():
    # Regression guard: the shipped agent_brain allowlist must never re-add a code-exec
    # or arbitrary-file-read entry to the auto-SAFE (no-confirm) tier.
    import json
    import os

    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config.json")))
    allow = cfg.get("agent_brain", {}).get("allowlist", [])
    banned = ("python", " -c", "cat", "head", "tail", "echo", "eval", "exec")
    for entry in allow:
        assert not any(b in entry for b in banned), f"dangerous auto-SAFE allowlist entry: {entry}"


# -- config + os preflight ----------------------------------------------------

def test_config_from_dict_coerces_lists():
    cfg = AgentBrainConfig.from_dict(
        {"model": "ollama/x", "allowlist": ["^ls"], "denylist": ["rm -rf"], "unknown_key": 1}
    )
    assert cfg.model == "ollama/x"
    assert cfg.allowlist == ("^ls",)
    assert cfg.denylist == ("rm -rf",)


def test_detect_display_server():
    assert detect_display_server(env={}, platform="darwin") == "macos"
    assert detect_display_server(env={"WAYLAND_DISPLAY": "wayland-0"}, platform="linux") == "wayland"
    assert detect_display_server(env={"DISPLAY": ":0"}, platform="linux") == "x11"
    assert detect_display_server(env={}, platform="linux") == "headless"


def test_os_preflight_headless_warns():
    warnings = os_mode_preflight(env={}, platform="linux", which=lambda _x: None)
    assert any("headless" in w for w in warnings)
