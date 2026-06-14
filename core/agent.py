"""Action brain backed by Open Interpreter (voice -> machine control).

This module is the SINGLE place the Open Interpreter (`interpreter`) API is
touched. A version bump, an API change, or a switch to subprocess isolation
only changes this file -- the rest of the assistant talks to the brain through
the small ``AgentBrainEvent`` stream below and never imports ``interpreter`` directly.

Open Interpreter is an OPTIONAL dependency: it is imported lazily inside
``_ensure_interpreter`` so the base voice assistant runs (and its test suite
passes) without it installed. Install on the target machine with
``pip install open-interpreter==0.4.3`` (see ``requirements-agent.txt``).

Ported from the legacy ``utils/agent_brain.py`` onto the new ``core`` runtime.
``attach_agent_capability`` wires it into the ``always_on_agent`` capability
registry under ``command.stage`` (the name command-mode already routes to), so
spoken commands run through the brain with no planner/supervisor changes.
"""
from __future__ import annotations

import builtins
import contextlib
import os
import re
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from always_on_agent.capabilities import CapabilityRegistry, CapabilityResult
from always_on_agent.origin import Origin, should_block_action


@dataclass
class AgentBrainConfig:
    """Configuration for the action brain (mirrors config.json ``agent_brain``)."""

    model: str = "ollama/gemma3:latest"          # LiteLLM model string
    api_base: Optional[str] = "http://localhost:11434"
    api_key_env: Optional[str] = None            # name of env var holding the key
    offline: bool = True
    local_only: bool = True                      # forbid cloud models when True
    local_fallback_model: Optional[str] = None   # used if a cloud model is blocked
    vision_model: Optional[str] = None           # preferred model for OS-mode screen vision
    os_mode: bool = False                         # desktop GUI control (OI "OS mode")
    web_tool: str = "oi"                          # reserved: oi | browser_use | ui_tars
    screenshot_dir: Optional[str] = None
    auto_run: bool = False
    confirm_mode: str = "auto_safe"               # auto_safe | always_ask
    allowlist: tuple[str, ...] = ()
    denylist: tuple[str, ...] = ()
    max_output_chars: int = 2000
    # SECURITY (default ON): refuse to take ANY action unless the turn is
    # owner-verified live audio (always_on_agent.origin chokepoint). Fail-closed:
    # with no owner-verified signal on the turn the whole action capability is
    # blocked -- so command-mode machine control cannot be driven by ambient/leaked
    # audio or by recalled/web/screen text. Set False to restore the legacy
    # (unverified) behavior, knowingly.
    require_owner_verified: bool = True

    @classmethod
    def from_dict(cls, data: dict | None) -> "AgentBrainConfig":
        """Build a config from a config.json ``agent_brain`` block (tolerant)."""
        from dataclasses import fields as _fields

        data = data or {}
        allowed = {f.name for f in _fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in allowed}
        for seq in ("allowlist", "denylist"):
            if seq in kwargs and kwargs[seq] is not None:
                kwargs[seq] = tuple(kwargs[seq])
        return cls(**kwargs)


@dataclass
class AgentBrainEvent:
    """Normalized event emitted while the brain works a task.

    NOTE: named ``AgentBrainEvent`` to avoid colliding with the PUBLIC
    cross-platform contract ``always_on_agent.events.AgentBrainEvent`` (the
    shell<->core seam shared with the Dart/mobile port). This type is
    module-private to the Open Interpreter integration.
    """

    kind: str            # speak | code | confirm | result | error
    text: str = ""
    code: str = ""
    language: str = ""


# classify() verdicts
SAFE = "safe"
NEEDS_CONFIRM = "needs_confirm"
BLOCKED = "blocked"

# Built-in, NON-overridable patterns that can NEVER be auto-SAFE (no-confirm),
# even if a misconfigured allowlist matches them -- they downgrade an allowlist hit
# to NEEDS_CONFIRM (so in auto_safe mode they don't run, and in always_ask mode the
# human must approve). This is the defense-in-depth that closes the
# allowlist-auto-RCE bypass: arbitrary code execution (interpreter inline -c/-e,
# eval/exec/system/subprocess), shell chaining/substitution, redirection, AND any
# command carrying a newline/control char (a one-line status command never does --
# a multi-line payload is a chained second command) can never silently run. The
# configurable denylist (-> BLOCKED) still takes precedence.
_NEVER_AUTO_SAFE = [
    re.compile(p, re.I) for p in (
        r"\bpython[0-9.]*\b\s+.*-c\b", r"\b(?:node|ruby|perl|php|deno|bun)\b.*\s-e\b",
        r"\b(?:bash|sh|zsh|fish|pwsh|powershell)\b.*\s-c\b", r"-c\s*['\"]",
        r"\beval\b", r"\bexec\b", r"os\.system", r"\bsubprocess\b", r"__import__",
        r"\bcompile\s*\(", r"\bpickle\b", r"\bsource\b", r"\bxargs\b",
        r"[;&|]", r"\$\(", r"`", r"[<>]",            # shell chaining / substitution / redirect
        r"[\n\r\f\v\x00]",                            # newline/control-char command chaining
    )
]

# affirmative answers fed to Open Interpreter's execution prompt
_OI_YES = "y"
_OI_NO = "n"

# LiteLLM model-string prefixes that run on-device (no cloud round-trip)
_LOCAL_MODEL_PREFIXES = ("ollama/", "ollama_chat/", "local/")


def _is_local_model(model: str) -> bool:
    return (model or "").lower().startswith(_LOCAL_MODEL_PREFIXES)


_SENTENCE_END = re.compile(r"[.!?](\s|$)")


def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END.search(text.strip()[-2:])) or len(text) > 160


def _truncate(text: str, limit: int) -> str:
    if limit and len(text) > limit:
        return text[:limit].rstrip() + " ... (truncated)"
    return text


class AgentBrain:
    """Wraps Open Interpreter and yields a normalized event stream."""

    def __init__(self, config: AgentBrainConfig):
        self.config = config
        self._interpreter = None
        self._allow_res = [re.compile(p, re.I) for p in config.allowlist]
        self._deny_res = [re.compile(p, re.I) for p in config.denylist]

    # -- model selection (hybrid local/cloud) --------------------------------
    def _effective_model(self) -> str:
        """Resolve the model to use, enforcing the local_only guard."""
        model = self.config.model
        if self.config.os_mode and self.config.vision_model:
            model = self.config.vision_model
        if self.config.local_only and not _is_local_model(model):
            fallback = self.config.local_fallback_model
            if fallback and _is_local_model(fallback):
                print(
                    f"[agent] local_only=true: cloud model '{model}' blocked; "
                    f"using local '{fallback}'."
                )
                return fallback
            raise RuntimeError(
                f"local_only=true forbids cloud model '{model}'. Set local_only "
                f"to false to allow cloud, or use an 'ollama/...' model."
            )
        return model

    # -- lazy Open Interpreter construction ----------------------------------
    def _ensure_interpreter(self):
        if self._interpreter is not None:
            return self._interpreter
        try:
            from interpreter import OpenInterpreter  # type: ignore
        except Exception as exc:  # pragma: no cover - needs OI installed
            raise RuntimeError(
                "open-interpreter is not installed. Install with "
                "`pip install open-interpreter==0.4.3` (see requirements-agent.txt)."
            ) from exc

        oi = OpenInterpreter()
        model = self._effective_model()
        oi.llm.model = model
        if self.config.api_base and _is_local_model(model):
            oi.llm.api_base = self.config.api_base
        if self.config.api_key_env:
            key = os.environ.get(self.config.api_key_env)
            if key:
                oi.llm.api_key = key
        oi.offline = bool(self.config.offline) and _is_local_model(model)
        oi.auto_run = bool(self.config.auto_run)
        if self.config.os_mode:
            from .agent_os import os_mode_preflight

            for warning in os_mode_preflight():
                print(f"[agent][os] {warning}")
            with contextlib.suppress(Exception):
                oi.os = True
        else:
            with contextlib.suppress(Exception):
                oi.os = False
        with contextlib.suppress(Exception):
            oi.verbose = False
        self._interpreter = oi
        return oi

    # -- safety classification ------------------------------------------------
    def classify(self, code: str) -> str:
        text = code or ""
        for rx in self._deny_res:
            if rx.search(text):
                return BLOCKED
        # An allowlist hit can ONLY be auto-SAFE if it also clears the built-in
        # never-auto-safe set -- so e.g. `python -c "<anything>"`, `ls; rm ...`,
        # `cat x > y`, a newline-chained second command, or an eval/exec/subprocess
        # payload never auto-runs even if a broad allowlist pattern matches it
        # (closes the auto-RCE bypass).
        for rx in self._allow_res:
            if rx.search(text):
                if any(bad.search(text) for bad in _NEVER_AUTO_SAFE):
                    return NEEDS_CONFIRM
                return SAFE
        return NEEDS_CONFIRM

    def decide(
        self,
        verdict: str,
        code: str,
        language: str,
        on_confirm: Optional[Callable[[str, str], bool]],
    ) -> bool:
        """Map a classification verdict to an allow/deny decision."""
        if verdict == BLOCKED:
            return False
        if verdict == SAFE:
            return True
        # NEEDS_CONFIRM
        if self.config.confirm_mode == "auto_safe":
            return False  # Phase 1: only explicitly-safe commands run
        if on_confirm is None:
            return False
        return bool(on_confirm(code, language))

    # -- streaming run --------------------------------------------------------
    def stream_run(
        self,
        instruction: str,
        should_cancel: Optional[Callable[[], bool]] = None,
        on_confirm: Optional[Callable[[str, str], bool]] = None,
    ) -> Iterator[AgentBrainEvent]:
        """Run ``instruction`` through Open Interpreter, yielding AgentBrainEvents."""
        should_cancel = should_cancel or (lambda: False)
        oi = self._ensure_interpreter()

        buffer: list[str] = []
        last_code = {"code": "", "language": ""}

        def flush() -> Iterator[AgentBrainEvent]:
            text = "".join(buffer).strip()
            buffer.clear()
            if text:
                yield AgentBrainEvent("speak", text=text)

        with self._auto_answer(last_code, on_confirm):
            try:
                for chunk in oi.chat(instruction, stream=True, display=False):
                    if should_cancel():
                        yield from flush()
                        return
                    if not isinstance(chunk, dict):
                        continue
                    ctype = chunk.get("type")
                    role = chunk.get("role")

                    if role == "assistant" and ctype == "message":
                        content = chunk.get("content")
                        if content and not chunk.get("start") and not chunk.get("end"):
                            buffer.append(str(content))
                            if _ends_sentence("".join(buffer)):
                                yield from flush()
                        elif chunk.get("end"):
                            yield from flush()
                        continue

                    if ctype == "code":
                        content = chunk.get("content")
                        if content and not chunk.get("start") and not chunk.get("end"):
                            last_code["code"] += str(content)
                        fmt = chunk.get("format")
                        if fmt:
                            last_code["language"] = str(fmt)
                        continue

                    if ctype == "confirmation":
                        content = chunk.get("content") or {}
                        code = content.get("code", "") if isinstance(content, dict) else ""
                        language = content.get("language", "") if isinstance(content, dict) else ""
                        verdict = self.classify(code)
                        yield AgentBrainEvent("confirm", code=code, language=language, text=verdict)
                        if not self.decide(verdict, code, language, on_confirm):
                            yield AgentBrainEvent("speak", text="I won't run that.")
                            self._reset()
                            return
                        last_code["code"] = ""
                        continue

                    if role == "computer" and ctype == "console":
                        if chunk.get("format") == "output":
                            out = chunk.get("content")
                            if isinstance(out, str) and out.strip():
                                yield AgentBrainEvent(
                                    "result",
                                    text=_truncate(out.strip(), self.config.max_output_chars),
                                )
                        continue
                yield from flush()
            except Exception as exc:  # defensive boundary around OI
                yield AgentBrainEvent("error", text=str(exc))

    @contextlib.contextmanager
    def _auto_answer(self, last_code: dict, on_confirm):
        """Temporarily answer OI's interactive 'run this code?' prompt."""
        if self.config.auto_run:
            yield
            return
        original = builtins.input

        def shim(prompt: str = "") -> str:
            code = last_code.get("code", "")
            language = last_code.get("language", "")
            verdict = self.classify(code)
            return _OI_YES if self.decide(verdict, code, language, on_confirm) else _OI_NO

        builtins.input = shim
        try:
            yield
        finally:
            builtins.input = original

    def _reset(self):
        with contextlib.suppress(Exception):
            if self._interpreter is not None:
                self._interpreter.messages = []


def attach_agent_capability(
    registry: CapabilityRegistry,
    config: AgentBrainConfig,
    *,
    capability_name: str = "command.stage",
    brain: "AgentBrain | None" = None,
) -> CapabilityRegistry:
    """Register the action brain as a capability provider.

    Overrides ``command.stage`` (the name command-mode already plans to) with a
    provider that runs the spoken instruction through the brain and returns the
    joined spoken output. Cancellation is honored via ``context['cancel_event']``
    so barge-in stops machine-control mid-task, like every other capability.

    ``brain`` may be supplied for testing (with a fake interpreter injected).
    """
    brain = brain or AgentBrain(config)

    def provider(query: str, context: dict[str, object]) -> CapabilityResult:
        instruction = (query or "").strip()
        if not instruction:
            return CapabilityResult(False, "", error="empty instruction")

        # SECURITY chokepoint (always_on_agent.origin): machine control may run ONLY
        # from owner-verified live audio. Fail-closed -- without an explicit
        # owner-verified signal on the turn the action is refused, so ambient/leaked
        # audio or recalled/web/screen-derived text can never drive a real action.
        # The runtime supplies context['origin'] + context['owner_verified'] from
        # the speaker-ID gate; disable knowingly via agent_brain.require_owner_verified.
        if config.require_owner_verified:
            origin = context.get("origin", Origin.UNKNOWN)
            owner_verified = context.get("owner_verified", False)
            if should_block_action(origin, owner_verified=owner_verified):
                return CapabilityResult(
                    True,
                    "I can't take that action without verified-owner authorization.",
                    data={"executed": False, "blocked": "owner_verification"},
                )

        cancel = context.get("cancel_event")
        should_cancel = (lambda: cancel.is_set()) if cancel is not None else None  # type: ignore[union-attr]

        spoken: list[str] = []
        last_error = ""
        stream = brain.stream_run(instruction, should_cancel=should_cancel)
        try:
            for event in stream:
                if should_cancel and should_cancel():
                    break
                if event.kind in ("speak", "result") and event.text:
                    spoken.append(event.text)
                elif event.kind == "error":
                    last_error = event.text
        finally:
            stream.close()  # restore agent_brain's stdin shim even on early break

        text = " ".join(spoken).strip()
        if not text and last_error:
            return CapabilityResult(
                True,
                "Sorry, I couldn't complete that action.",
                data={"executed": False, "error": last_error},
            )
        return CapabilityResult(True, text or "Done.", data={"executed": True})

    registry.register(capability_name, provider)
    return registry
