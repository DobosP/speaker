"""
Action brain backed by Open Interpreter.

This module is the SINGLE place the Open Interpreter (`interpreter`) API is
touched. A version bump, an API change, or a switch to subprocess isolation
only changes this file -- the rest of the assistant talks to the brain through
the small ``AgentEvent`` stream below and never imports ``interpreter`` directly.

Open Interpreter is an OPTIONAL dependency: it is imported lazily inside
``_ensure_interpreter`` so the base voice assistant runs (and its test suite
passes) without it installed. Install on the target machine with
``pip install open-interpreter==0.4.3`` (see requirements-agent.txt).
"""
from __future__ import annotations

import builtins
import contextlib
import os
import re
from dataclasses import dataclass
from typing import Callable, Iterator, Optional


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
    os_mode: bool = False                        # desktop GUI control (OI "OS mode")
    web_tool: str = "oi"                         # reserved: oi | browser_use | ui_tars
    screenshot_dir: Optional[str] = None
    auto_run: bool = False
    confirm_mode: str = "auto_safe"              # auto_safe | always_ask
    allowlist: tuple[str, ...] = ()
    denylist: tuple[str, ...] = ()
    max_output_chars: int = 2000


@dataclass
class AgentEvent:
    """Normalized event emitted while the brain works a task."""

    kind: str            # speak | code | confirm | result | error
    text: str = ""
    code: str = ""
    language: str = ""


# classify() verdicts
SAFE = "safe"
NEEDS_CONFIRM = "needs_confirm"
BLOCKED = "blocked"

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
        """Resolve the model to use, enforcing the local_only guard.

        In OS mode a configured vision_model is preferred (screen understanding
        needs vision). If local_only is set and the chosen model is a cloud
        model, fall back to a local model when one is configured, else raise.
        """
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
        # local models may stay offline; a cloud model needs network access
        oi.offline = bool(self.config.offline) and _is_local_model(model)
        # We always drive approval ourselves (chunk gate + stdin shim), so OI's
        # own auto_run stays off unless the operator explicitly opts in.
        oi.auto_run = bool(self.config.auto_run)
        if self.config.os_mode:
            from utils.agent_os import os_mode_preflight

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
        for rx in self._allow_res:
            if rx.search(text):
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
    ) -> Iterator[AgentEvent]:
        """Run ``instruction`` through Open Interpreter, yielding AgentEvents.

        Approval is enforced two ways so the same logic works across OI builds:
          * the documented ``type == "confirmation"`` chunk (newer API / fakes), and
          * a scoped stdin shim that auto-answers OI's interactive execution
            prompt (real 0.4.x, which reads ``input()`` when ``auto_run`` is off).
        Both converge on :meth:`decide`.
        """
        should_cancel = should_cancel or (lambda: False)
        oi = self._ensure_interpreter()

        buffer: list[str] = []
        last_code = {"code": "", "language": ""}

        def flush() -> Iterator[AgentEvent]:
            text = "".join(buffer).strip()
            buffer.clear()
            if text:
                yield AgentEvent("speak", text=text)

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
                        yield AgentEvent("confirm", code=code, language=language, text=verdict)
                        if not self.decide(verdict, code, language, on_confirm):
                            yield AgentEvent("speak", text="I won't run that.")
                            self._reset()
                            return
                        last_code["code"] = ""
                        continue

                    if role == "computer" and ctype == "console":
                        if chunk.get("format") == "output":
                            out = chunk.get("content")
                            if isinstance(out, str) and out.strip():
                                yield AgentEvent(
                                    "result",
                                    text=_truncate(out.strip(), self.config.max_output_chars),
                                )
                        continue
                yield from flush()
            except Exception as exc:  # defensive boundary around OI
                yield AgentEvent("error", text=str(exc))

    @contextlib.contextmanager
    def _auto_answer(self, last_code: dict, on_confirm):
        """Temporarily answer OI's interactive 'run this code?' prompt.

        Real Open Interpreter calls ``input()`` for execution approval when
        ``auto_run`` is off. We answer it from :meth:`decide` using the code
        seen so far, so the voice flow never blocks on a terminal prompt.
        Scoped to the chat() call to limit the global ``input`` patch.
        """
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
