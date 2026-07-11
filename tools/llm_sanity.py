"""Real-model output sanity check + per-provider cloud-LLM smoke probe.

Two modes coexist:

- **Local fast-GGUF sanity** (legacy ``main()``): downloads MiniCPM5-1B Q4
  GGUF and runs a few prompts on CPU, asserting the answers are distinct and
  non-degenerate. Guards against the "Gemma answering the same canned line"
  bug. Run by ``.github/workflows/llm-sanity.yml``.

- **Cloud-provider smoke probe** (``--provider`` / ``--all`` / ``--smoke``):
  sends one short prompt to each ``llm.cloud_providers`` entry in config and
  reports TTFT + total time + token count. Catches provider drift (deprecated
  model ids, broken endpoints) before users do. Run weekly by
  ``.github/workflows/llm-cloud-smoke.yml``. Also validates each preset's
  ``model`` against the vendored LiteLLM registry at
  ``tools/litellm_model_registry.json`` to catch upstream deprecations.

The pure heuristics (``looks_degenerate`` / ``outputs_distinct``) are
unit-tested in ``tests/test_llm_sanity.py`` with no model. Provider-probe
shape + exit codes are unit-tested with a stub ``OpenAICompatLLM``.

Note: the local-sanity path exercises the Python ``llama.cpp`` runtime, not
the mobile ``flutter_gemma`` runtime (which can't run on a stock CI runner).
The mobile glue is guarded separately by ``mobile/test/llm_glue_test.dart``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional

_PROMPTS = [
    "What is the capital of France?",
    "Tell me a short joke.",
    "What is two plus two?",
]

_SYSTEM = "You are a concise, friendly assistant. Answer in one or two short sentences."

# Long enough to cross llama.cpp's default 512-token decode batch without
# approaching this tool's 2k context limit.  Repetition is intentional: the
# answer is discarded, and familiar single-token words make the lower bound
# much less tokenizer-dependent than prose would be.
_ABORT_PROBE_PROMPT = (
    "Read every word before replying. "
    + " ".join(["red blue green black white small large round"] * 112)
)
_LOCAL_MAX_TOKENS = 96
_ABORT_POLL_TIMEOUT_SEC = 45.0
_ABORT_EXIT_TIMEOUT_SEC = 5.0

# Single short prompt for the cloud smoke probe -- we want round-trip + a
# real token, not a quality check. Reasoning models will preamble; that's OK
# because we only check non-empty + non-degenerate.
_PROBE_PROMPT = "Reply briefly: what is 2+2?"


def looks_degenerate(text: str, *, min_words: int = 4) -> bool:
    """True if ``text`` looks stuck: empty, one token spammed, or a short cycle."""
    words = text.split()
    if len(words) < min_words:
        return len(words) == 0  # very short is only degenerate if empty
    # One token dominates ("okay okay okay ...").
    if Counter(words).most_common(1)[0][1] / len(words) > 0.5:
        return True
    # A short phrase repeats back-to-back ("let's do this let's do this ...").
    for cycle in range(1, 7):
        if len(words) < cycle * 3:
            continue
        seg = words[:cycle]
        reps, i = 1, cycle
        while i + cycle <= len(words) and words[i:i + cycle] == seg:
            reps += 1
            i += cycle
        if reps >= 3:
            return True
    return False


def outputs_distinct(outputs: list[str]) -> bool:
    """True if not all answers are identical (catches 'same reply to everything')."""
    return len({o.strip().lower() for o in outputs}) > 1


def _check(pairs: list[tuple[str, str]]) -> list[str]:
    """Return human-readable problems (empty list == healthy)."""
    problems = []
    for prompt, out in pairs:
        lowered = out.lower()
        if looks_degenerate(out):
            problems.append(f"degenerate output for {prompt!r}: {out!r}")
        if any(marker in lowered for marker in ("<think", "</think", "<|thought_")):
            problems.append(f"reasoning markup leaked for {prompt!r}")
        if prompt == "What is the capital of France?":
            has_paris = bool(re.search(r"\bparis\b", lowered))
            negates_paris = bool(
                re.search(r"\b(?:not|isn't|isnt|never)\b.{0,24}\bparis\b", lowered)
                or re.search(r"\bparis\b\s+(?:is|was)\s+not\b", lowered)
            )
            if not has_paris or negates_paris:
                problems.append("capital probe did not answer Paris")
        if prompt == "What is two plus two?":
            has_four = bool(re.search(r"\b(?:four|4)\b", lowered))
            negates_four = bool(
                re.search(r"\b(?:not|isn't|isnt|never)\b.{0,24}\b(?:four|4)\b", lowered)
                or re.search(r"\b(?:four|4)\b\s+(?:is|was)\s+not\b", lowered)
            )
            if not has_four or negates_four:
                problems.append("arithmetic probe did not answer four")
    if len(pairs) > 1 and not outputs_distinct([o for _, o in pairs]):
        problems.append("all prompts produced the same answer")
    return problems


@dataclass(frozen=True)
class NativeAbortProbeResult:
    """Timings from the real llama.cpp abort-and-reuse sanity gate."""

    native_poll_ms: float
    cancel_exit_ms: float
    recovery_ms: float


class _NativeAbortProbeEvent:
    """Event that distinguishes native abort polls from Python-side polls.

    ``LlamaCppLLM``'s retained ctypes callback is a closure named
    ``should_abort``.  Its callback and the Python stream loop both inspect the
    same event, so a plain ``Event`` cannot prove that native prefill actually
    reached the callback.  Walking this tiny stack only during the sanity job
    provides that proof without adding a production-only hook to ``core.llm``.
    """

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._native_polled = threading.Event()

    def is_set(self) -> bool:
        frame = sys._getframe(1)
        try:
            while frame is not None:
                if frame.f_code.co_name == "should_abort":
                    self._native_polled.set()
                    break
                frame = frame.f_back
        finally:
            # Avoid retaining worker frames (and therefore the model/client)
            # through a reference cycle after this callback returns.
            del frame
        return self._cancelled.is_set()

    def set(self) -> None:
        self._cancelled.set()

    def wait_for_native_poll(self, timeout: float) -> bool:
        return self._native_polled.wait(timeout=max(0.0, timeout))

    @property
    def native_polled(self) -> bool:
        return self._native_polled.is_set()


def probe_llamacpp_abort_recovery(
    llm,
    *,
    native_poll_timeout: float = _ABORT_POLL_TIMEOUT_SEC,
    cancel_exit_timeout: float = _ABORT_EXIT_TIMEOUT_SEC,
) -> NativeAbortProbeResult:
    """Require native stream cancellation and immediate same-context reuse.

    The first-call timeout includes lazy model loading.  Once the audited
    native callback has polled, cancellation must retire the worker within the
    much smaller exit bound as :class:`LLMCallCancelled`.  The discarded stream
    deliberately follows the voice agent's production barge-in path.
    """
    from core.llm import LLMCallCancelled, capability_context

    cancel_event = _NativeAbortProbeEvent()
    done = threading.Event()
    outcome: dict[str, object] = {"pieces": 0}

    def consume_cancelled_stream() -> None:
        token = capability_context.set({"cancel_event": cancel_event})
        try:
            for _piece in llm.stream(_ABORT_PROBE_PROMPT, system=_SYSTEM):
                outcome["pieces"] = int(outcome["pieces"]) + 1
        except BaseException as exc:
            outcome["error"] = exc
        else:
            outcome["error"] = None
        finally:
            capability_context.reset(token)
            done.set()

    started_at = time.perf_counter()
    worker = threading.Thread(
        target=consume_cancelled_stream,
        name="llamacpp-native-abort-sanity",
        daemon=True,
    )
    worker.start()

    poll_deadline = started_at + max(0.0, native_poll_timeout)
    while not cancel_event.native_polled:
        remaining = poll_deadline - time.perf_counter()
        if remaining <= 0.0 or done.wait(timeout=min(0.01, remaining)):
            break

    if not cancel_event.native_polled:
        # Best-effort retirement also keeps an early-failure/unit-test worker
        # from leaking after the gate has diagnosed the missing native poll.
        cancel_event.set()
        stopped = done.wait(timeout=max(0.0, cancel_exit_timeout))
        if not stopped:
            raise RuntimeError(
                "native abort callback was not polled and the stream worker "
                f"did not stop within {cancel_exit_timeout:.2f}s"
            )
        worker.join(timeout=max(0.0, cancel_exit_timeout))
        if worker.is_alive():
            raise RuntimeError("native abort probe worker stayed alive after teardown")
        error = outcome.get("error")
        if error is not None:
            raise RuntimeError(
                "native abort callback was not polled before stream failure "
                f"({type(error).__name__})"
            ) from error
        raise RuntimeError("native abort callback was not polled before stream completion")

    native_polled_at = time.perf_counter()
    cancel_event.set()
    cancel_started_at = time.perf_counter()
    if not done.wait(timeout=max(0.0, cancel_exit_timeout)):
        raise RuntimeError(
            "native-cancelled stream worker did not stop within "
            f"{cancel_exit_timeout:.2f}s"
        )
    worker.join(timeout=max(0.0, cancel_exit_timeout))
    if worker.is_alive():
        raise RuntimeError("native-cancelled stream worker stayed alive after teardown")
    cancel_finished_at = time.perf_counter()

    error = outcome.get("error")
    if not isinstance(error, LLMCallCancelled):
        actual = "normal completion" if error is None else type(error).__name__
        raise RuntimeError(
            "native-cancelled stream did not exit as LLMCallCancelled "
            f"(got {actual})"
        ) from error
    pieces_seen = int(outcome["pieces"])
    if pieces_seen:
        raise RuntimeError(
            "native abort was not pre-token: stream emitted "
            f"{pieces_seen} piece(s) before cancellation"
        )
    if getattr(llm, "_context_poisoned", None) is not False:
        raise RuntimeError("llama.cpp context was poisoned or could not be inspected")

    recovery_started_at = time.perf_counter()
    recovered = str(
        llm.generate(
            "Reply with only the word ready.",
            system="This is a health check. Follow the request exactly.",
        )
        or ""
    ).strip()
    recovery_finished_at = time.perf_counter()
    if not recovered:
        raise RuntimeError("same-context recovery completion was empty")
    if int(getattr(llm, "last_reasoning_blocks", 0) or 0):
        raise RuntimeError(
            "same-context recovery needed the reasoning-tag fallback; "
            "no-think template control is not effective"
        )
    if hasattr(llm, "last_finish_reason") and llm.last_finish_reason != "stop":
        raise RuntimeError(
            "same-context recovery did not finish naturally "
            f"(finish_reason={llm.last_finish_reason!r})"
        )
    if getattr(llm, "_context_poisoned", None) is not False:
        raise RuntimeError("llama.cpp context became poisoned during recovery")

    return NativeAbortProbeResult(
        native_poll_ms=(native_polled_at - started_at) * 1000.0,
        cancel_exit_ms=(cancel_finished_at - cancel_started_at) * 1000.0,
        recovery_ms=(recovery_finished_at - recovery_started_at) * 1000.0,
    )


# --- cloud-provider smoke -------------------------------------------------


# Exit codes for the per-provider probe.
OK = 0
FAIL = 1
SKIP = 2


@dataclass
class ProbeResult:
    """Outcome of one provider probe. ``status`` is ``"ok"`` / ``"fail"`` /
    ``"skip"``; ``exit_code`` is the corresponding integer for CI."""

    name: str
    model: str
    status: str
    exit_code: int
    ttft_ms: Optional[float] = None
    total_ms: Optional[float] = None
    tokens: int = 0
    text: str = ""
    error: str = ""
    reasoning_chars: int = 0

    def as_markdown_row(self) -> str:
        ttft = f"{self.ttft_ms:.0f}" if self.ttft_ms is not None else "-"
        tot = f"{self.total_ms:.0f}" if self.total_ms is not None else "-"
        note = self.error or (self.text[:60].replace("\n", " ") if self.text else "")
        return (
            f"| {self.name} | {self.model} | {self.status} | "
            f"{ttft} | {tot} | {self.tokens} | {note} |"
        )


def probe_provider(
    name: str,
    preset: dict,
    *,
    prompt: str = _PROBE_PROMPT,
    timeout: float = 15.0,
    llm_factory=None,
) -> ProbeResult:
    """Send one short prompt to ``preset`` and return a structured outcome.

    ``status`` is ``"skip"`` when the preset's ``api_key_env`` env var is
    missing (the most common CI case), ``"fail"`` on any error / empty
    response / degenerate output, otherwise ``"ok"``. TTFT is the time
    between the first ``stream()`` call and the first non-empty token --
    which for reasoning models (DeepSeek V4-Pro, Groq gpt-oss-120b) may
    arrive on the reasoning channel before any ``content`` is produced.

    ``llm_factory`` is injectable for tests; defaults to building a real
    :class:`core.llm.OpenAICompatLLM`.
    """
    model = preset.get("model", "?")
    api_key_env = preset.get("api_key_env")
    if api_key_env and not os.environ.get(api_key_env):
        return ProbeResult(name, model, "skip", SKIP, error=f"env var {api_key_env} unset")

    if llm_factory is None:
        from core.llm import OpenAICompatLLM
        llm = OpenAICompatLLM(
            model=model,
            base_url=preset.get("base_url"),
            api_key_env=api_key_env,
            options=preset.get("options"),
            profile=preset.get("profile"),
            timeout=timeout,
        )
    else:
        llm = llm_factory(preset)

    start = time.perf_counter()
    first_token_at: Optional[float] = None
    pieces: list[str] = []
    tokens = 0
    try:
        for piece in llm.stream(prompt, system=_SYSTEM):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            pieces.append(piece)
            tokens += 1
            # cap to avoid runaway reasoning chains burning quota in a smoke probe.
            if tokens > 200:
                break
    except Exception as exc:  # network / 4xx / 5xx / SDK error
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return ProbeResult(
            name, model, "fail", FAIL,
            total_ms=elapsed_ms, error=f"{type(exc).__name__}: {exc}",
        )

    end = time.perf_counter()
    total_ms = (end - start) * 1000.0
    ttft_ms = (first_token_at - start) * 1000.0 if first_token_at else None
    text = "".join(pieces).strip()
    reasoning_chars = int(getattr(llm, "last_reasoning_chars", 0) or 0)

    if not text:
        return ProbeResult(
            name, model, "fail", FAIL,
            ttft_ms=ttft_ms, total_ms=total_ms, tokens=tokens,
            error="empty response", reasoning_chars=reasoning_chars,
        )
    if looks_degenerate(text):
        return ProbeResult(
            name, model, "fail", FAIL,
            ttft_ms=ttft_ms, total_ms=total_ms, tokens=tokens, text=text,
            error="degenerate output", reasoning_chars=reasoning_chars,
        )
    return ProbeResult(
        name, model, "ok", OK,
        ttft_ms=ttft_ms, total_ms=total_ms, tokens=tokens, text=text,
        reasoning_chars=reasoning_chars,
    )


def validate_registry(
    presets: dict, registry: dict, *, ignore: Iterable[str] = ()
) -> list[str]:
    """Return a list of problems when a preset's ``model`` isn't in the LiteLLM
    registry. Empty list == every model id is recognized upstream.

    Empty registry is a no-op (returns []) so a fresh checkout without the
    vendored ``tools/litellm_model_registry.json`` doesn't fail-block.

    LiteLLM keys some entries with a ``<provider>/<model>`` prefix (e.g.
    ``cerebras/gpt-oss-120b``); we accept both the raw model id and any
    provider-prefixed form.
    """
    if not registry:
        return []
    problems: list[str] = []
    ignore_set = set(ignore)
    for name, preset in presets.items():
        if name.startswith("_") or name in ignore_set:
            continue
        if not isinstance(preset, dict):
            continue
        model = preset.get("model")
        if not model:
            continue
        if model in registry:
            continue
        # Accept any "<vendor>/<model>" form (LiteLLM frequently keys this way).
        prefixed_hits = [
            key for key in registry
            if key.endswith("/" + model) or key == model
        ]
        if not prefixed_hits:
            problems.append(
                f"{name}: model {model!r} not found in LiteLLM registry "
                f"(may be deprecated upstream)"
            )
    return problems


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_registry(path: str) -> dict:
    """Load the vendored LiteLLM model registry. Returns {} if missing so the
    smoke job degrades gracefully instead of failing on a fresh checkout."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _format_table(results: list[ProbeResult]) -> str:
    header = (
        "| Provider | Model | Status | TTFT (ms) | Total (ms) | Tokens | Note |\n"
        "|---|---|---|---:|---:|---:|---|"
    )
    rows = "\n".join(r.as_markdown_row() for r in results)
    return f"{header}\n{rows}"


def _emit_github_summary(text: str) -> None:
    """If running under GH Actions, append to $GITHUB_STEP_SUMMARY."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text)
            fh.write("\n")
    except OSError:
        pass


def _run_cloud_probe(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    presets = (cfg.get("llm") or {}).get("cloud_providers") or {}
    # Filter underscore keys (e.g. _comment).
    presets = {n: p for n, p in presets.items() if not n.startswith("_")}
    if not presets:
        print(f"No cloud_providers configured in {args.config}", file=sys.stderr)
        return FAIL

    # Registry validation up front -- catches deprecations even when no keys
    # are set (so the smoke job warns on a fresh checkout).
    registry = _load_registry(args.registry)
    if registry:
        ignored = set((args.ignore_registry or "").split(",")) - {""}
        problems = validate_registry(presets, registry, ignore=ignored)
        for problem in problems:
            print(f"REGISTRY WARNING: {problem}", file=sys.stderr)

    targets: dict[str, dict]
    if args.provider:
        if args.provider not in presets:
            print(
                f"Unknown provider {args.provider!r}; "
                f"available: {sorted(presets)}",
                file=sys.stderr,
            )
            return FAIL
        targets = {args.provider: presets[args.provider]}
    else:  # --all / --smoke
        targets = presets

    results: list[ProbeResult] = []
    for name in sorted(targets):
        result = probe_provider(name, targets[name], timeout=args.timeout)
        results.append(result)
        # Per-provider line to stdout for the developer running locally.
        print(
            f"[{result.status:4}] {result.name:30} {result.model:35} "
            f"ttft={result.ttft_ms or '-':>5} ms tot={result.total_ms or '-':>5} ms "
            f"tokens={result.tokens:4} {result.error or result.text[:60]!r}"
        )

    table = _format_table(results)
    print()
    print(table)
    _emit_github_summary(table)

    # Exit-code policy:
    #   --provider X    -> exit code = that provider's exit code (0/1/2)
    #   --all           -> max of individual codes (any fail -> 1, all skip -> 2)
    #   --smoke         -> 0 unless a KEY-SET probe failed (skips don't fail CI)
    if args.provider:
        return results[0].exit_code
    if args.smoke:
        return FAIL if any(r.status == "fail" for r in results) else OK
    return max(r.exit_code for r in results)


# --- main -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LLM sanity checks: local-GGUF degeneration + cloud-provider smoke.",
    )
    parser.add_argument("--cache-dir", default=".cache/bench-models",
                        help="legacy local-sanity model cache dir")
    parser.add_argument("--models-manifest", default=None,
                        help="legacy local-sanity manifest path")

    cloud = parser.add_argument_group("cloud-provider smoke")
    cloud.add_argument("--provider", default=None,
                       help="probe a single preset by name (from llm.cloud_providers)")
    cloud.add_argument("--all", action="store_true",
                       help="probe every preset; exit code = max(individual)")
    cloud.add_argument("--smoke", action="store_true",
                       help="like --all but exit 0 unless a set-key probe failed")
    cloud.add_argument("--config", default="config.json",
                       help="path to config.json (default: ./config.json)")
    cloud.add_argument("--registry", default="tools/litellm_model_registry.json",
                       help="path to the vendored LiteLLM model registry")
    cloud.add_argument("--ignore-registry", default="",
                       help="comma-separated preset names to skip in registry validation "
                            "(e.g. for very-new models LiteLLM hasn't catalogued yet)")
    cloud.add_argument("--timeout", type=float, default=15.0,
                       help="per-provider timeout in seconds")

    args = parser.parse_args(argv)

    if args.provider or args.all or args.smoke:
        return _run_cloud_probe(args)

    # Legacy local-sanity path: download MiniCPM5 Q4 + run distinctness checks.
    from huggingface_hub import hf_hub_download

    from core.llm import LlamaCppLLM
    from tools.bench.models import load_manifest

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    coords = load_manifest(args.models_manifest)["fast_gguf"]
    gguf = hf_hub_download(
        repo_id=coords["repo"], filename=coords["file"], cache_dir=args.cache_dir, token=token
    )
    # One shared context is intentional: the gate below proves that a native
    # prefill abort leaves this exact client reusable before quality prompts
    # run.  The explicit output cap bounds every completion in this job.
    llm = LlamaCppLLM(
        gguf,
        n_ctx=2048,
        # Fixed functional baseline: keep host topology/oversubscription out of
        # this correctness gate. The profile-aware perf workflow owns tuning.
        n_threads=2,
        n_threads_batch=2,
        n_gpu_layers=0,
        options={"max_tokens": _LOCAL_MAX_TOKENS},
    )

    try:
        abort_probe = probe_llamacpp_abort_recovery(llm)
    except Exception as exc:
        print(
            "LLM SANITY FAILED: native abort/recovery gate: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return FAIL
    print(
        "Native abort/recovery OK: "
        f"poll={abort_probe.native_poll_ms:.1f}ms "
        f"cancel={abort_probe.cancel_exit_ms:.1f}ms "
        f"recovery={abort_probe.recovery_ms:.1f}ms"
    )

    pairs: list[tuple[str, str]] = []
    template_problems: list[str] = []
    for prompt in _PROMPTS:
        started_at = time.perf_counter()
        first_at: float | None = None
        pieces: list[str] = []
        for piece in llm.stream(prompt, system=_SYSTEM):
            if first_at is None:
                first_at = time.perf_counter()
            pieces.append(piece)
        finished_at = time.perf_counter()
        out = "".join(pieces).strip()
        ttft_ms = (
            (first_at - started_at) * 1000.0
            if first_at is not None
            else float("inf")
        )
        print(
            f"\n=== {prompt} "
            f"[ttft={ttft_ms:.1f}ms total={(finished_at - started_at) * 1000.0:.1f}ms]"
            f"\n{out}"
        )
        pairs.append((prompt, out))
        if int(getattr(llm, "last_reasoning_blocks", 0) or 0):
            template_problems.append(
                f"no-think template failed for {prompt!r}; output fallback was needed"
            )
        if getattr(llm, "last_finish_reason", None) != "stop":
            template_problems.append(
                f"answer did not finish naturally for {prompt!r}: "
                f"{getattr(llm, 'last_finish_reason', None)!r}"
            )

    problems = template_problems + _check(pairs)
    if problems:
        print("\nLLM SANITY FAILED:")
        for problem in problems:
            print(" -", problem)
        return FAIL
    print("\nLLM sanity OK: direct complete answers, no reasoning fallback.")
    return OK


if __name__ == "__main__":
    raise SystemExit(main())
