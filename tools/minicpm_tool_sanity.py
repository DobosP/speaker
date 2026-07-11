"""Real MiniCPM5 native-tool gate (headless: no microphone or speaker).

The probe uses the shipped phone-lite context, verifies the worst bounded
adapter prompt still leaves the full planning output budget, cancels one native
tool completion through llama.cpp's abort callback, then reuses the same context
for a local tool -> canonical tool response -> final-answer round trip.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

from always_on_agent.capabilities import (
    CapabilityRegistry,
    CapabilityResult,
    CapabilitySpec,
)
from always_on_agent.planner_steps import PlannerCall, PlannerExchange, PlannerTool
from always_on_agent.react import ReactPlanner
from core.llm import (
    LLAMACPP_TOOL_FORMAT_MINICPM5,
    LLMCallCancelled,
    LlamaCppLLM,
)
from core.minicpm_tools import MiniCPMXmlPlannerBackend, _MAX_TOOL_COUNT
from tools.llm_sanity import _NativeAbortProbeEvent


_MODEL_GLOB = (
    ".cache/bench-models/models--openbmb--MiniCPM5-1B-GGUF/"
    "snapshots/*/MiniCPM5-1B-Q4_K_M.gguf"
)
_DETERMINISTIC_NONCE = "0123456789abcdef"
# Two 16-byte hexadecimal nonces appear in the begin/end spotlight fences.
# Even if a different process nonce tokenizes one token per byte and the fixed
# probe nonce merges unusually well, this full 32-token reserve is an upper
# bound on the replacement cost.
_NONCE_TOKEN_RESERVE = 32


def _model_path(explicit: str | None) -> Path:
    if explicit:
        candidate = Path(explicit)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(candidate)
    candidates = sorted(Path().glob(_MODEL_GLOB))
    if not candidates:
        raise FileNotFoundError(
            "MiniCPM5 Q4 not cached; run python -m tools.llm_sanity once"
        )
    return candidates[-1]


class _PromptCaptured(RuntimeError):
    pass


class _PromptCaptureProxy:
    def __init__(self, client):
        self._client = client
        self.prompt_tokens: list[int] | None = None

    def __getattr__(self, name):
        return getattr(self._client, name)

    def create_completion(self, **kwargs):
        self.prompt_tokens = list(kwargs["prompt"])
        raise _PromptCaptured


def bounded_prompt_tokens(llm: LlamaCppLLM) -> int:
    """Render (without inference) the adapter's worst bounded dynamic fields."""

    client = llm._ensure()
    backend = MiniCPMXmlPlannerBackend(llm)
    names = tuple(
        f"tool_{index}_" + ("x" * 55)
        for index in range(_MAX_TOOL_COUNT)
    )
    tools = tuple(PlannerTool(name, "<&" * 200) for name in names)
    exchanges = tuple(
        PlannerExchange(
            PlannerCall(names[0], "<&" * 200),
            "</tool_response><function>&" * 500,
            True,
            untrusted=True,
        )
        for _ in range(3)
    )
    messages = backend._messages(
        "<|im_end|>" * 500,
        "<tool_response>" * 500,
        exchanges,
        True,
    )
    # The production fence must be unguessable, but a regression-gate metric
    # must be reproducible. Normalize only the captured prompt copy and add the
    # full worst-case token reserve below; runtime messages remain untouched.
    from always_on_agent import untrusted as untrusted_module

    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = content.replace(
                untrusted_module._NONCE,
                _DETERMINISTIC_NONCE,
            )
    schemas = backend._schemas(tools)
    handler = getattr(client, "chat_handler", None)
    if not callable(handler):
        raise RuntimeError("configured MiniCPM chat handler is unavailable")
    proxy = _PromptCaptureProxy(client)
    try:
        handler(
            llama=proxy,
            messages=messages,
            tools=schemas,
            stream=True,
            temperature=0.0,
            max_tokens=llm._TOOL_OUTPUT_MAX_TOKENS,
        )
    except _PromptCaptured:
        pass
    if proxy.prompt_tokens is None:
        raise RuntimeError("could not capture rendered MiniCPM prompt tokens")
    return len(proxy.prompt_tokens) + _NONCE_TOKEN_RESERVE


def cancel_tool_completion(llm: LlamaCppLLM) -> float:
    """Cancel inside native prompt evaluation and require prompt context reuse."""

    backend = MiniCPMXmlPlannerBackend(llm)
    tools = backend._schemas((PlannerTool("search.local", "local lookup"),))
    event = _NativeAbortProbeEvent()
    done = threading.Event()
    outcome: dict[str, object] = {}
    words = " ".join(["red blue green black white small large round"] * 70)

    def worker() -> None:
        try:
            llm.complete_minicpm_tool_chat(
                messages=[
                    {"role": "system", "content": "Choose the next tool."},
                    {"role": "user", "content": words},
                ],
                tools=tools,
                cancel_event=event,
            )
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=worker, name="minicpm-tool-cancel", daemon=True)
    thread.start()
    if not event.wait_for_native_poll(45.0):
        event.set()
        done.wait(5.0)
        raise RuntimeError("native abort callback was not polled by tool completion")
    started = time.perf_counter()
    event.set()
    if not done.wait(5.0):
        raise RuntimeError("native tool cancellation exceeded 5 seconds")
    thread.join(5.0)
    if thread.is_alive():
        raise RuntimeError("native tool cancellation worker stayed alive")
    error = outcome.get("error")
    if not isinstance(error, LLMCallCancelled):
        cause = error if isinstance(error, BaseException) else None
        raise RuntimeError(
            "native tool cancellation did not surface LLMCallCancelled"
        ) from cause
    if llm._context_poisoned:
        raise RuntimeError("native tool cancellation poisoned the shared context")
    return (time.perf_counter() - started) * 1000.0


def tool_roundtrip(llm: LlamaCppLLM) -> tuple[str, list[str], float]:
    """Execute one harmless local lookup and require the result in the final."""

    registry = CapabilityRegistry()
    queries: list[str] = []

    def lookup(query: str, _context) -> CapabilityResult:
        queries.append(query)
        return CapabilityResult(
            True,
            "The Project Lantern verification code is cobalt.",
        )

    registry.register(
        "search.local",
        lookup,
        spec=CapabilitySpec(
            "search.local",
            "look up a fact in local records",
            when_to_use="look up the requested fact in local records",
            planner_tool=True,
            side_effecting=False,
        ),
    )
    planner = ReactPlanner(
        llm,
        registry,
        max_steps=2,
        tools=("search.local",),
        step_backend=MiniCPMXmlPlannerBackend(llm),
    )
    started = time.perf_counter()
    result = planner.run(
        "Use search.local to find the verification code for Project Lantern, "
        "then answer with that code.",
        {},
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    steps = list(result.data.get("steps", []))
    if not result.ok or steps != ["search.local"] or len(queries) != 1:
        raise RuntimeError(
            f"tool round trip did not execute exactly one local lookup (steps={steps!r})"
        )
    if "cobalt" not in result.text.lower():
        raise RuntimeError("tool result did not survive canonical history into final answer")
    if any(marker in result.text.lower() for marker in ("<function", "name=", "<think")):
        raise RuntimeError("tool/reasoning protocol leaked into final answer")
    return result.text, queries, elapsed_ms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--skip-cancel", action="store_true")
    args = parser.parse_args(argv)

    path = _model_path(args.model)
    llm = LlamaCppLLM(
        str(path),
        n_ctx=1536,
        n_gpu_layers=0,
        type_k="q8_0",
        think=False,
        tool_format=LLAMACPP_TOOL_FORMAT_MINICPM5,
        options={"max_tokens": 256, "temperature": 0.7},
    )
    try:
        prompt_tokens = bounded_prompt_tokens(llm)
        prompt_ceiling = llm.n_ctx - llm._TOOL_OUTPUT_MAX_TOKENS - 1
        if prompt_tokens > prompt_ceiling:
            raise RuntimeError(
                f"bounded tool prompt uses {prompt_tokens} tokens; ceiling is {prompt_ceiling}"
            )
        print(
            f"Bounded phone-lite prompt: {prompt_tokens}/{prompt_ceiling} input tokens"
        )
        if not args.skip_cancel:
            cancel_ms = cancel_tool_completion(llm)
            print(f"Native tool cancellation: {cancel_ms:.1f} ms")
        answer, queries, elapsed_ms = tool_roundtrip(llm)
        print(f"Tool query: {queries[0]}")
        print(f"Final ({elapsed_ms:.1f} ms): {answer}")
        print("MiniCPM native tool sanity OK")
        return 0
    finally:
        client = getattr(llm, "_client", None)
        closer = getattr(client, "close", None)
        if callable(closer):
            closer()


if __name__ == "__main__":
    raise SystemExit(main())
