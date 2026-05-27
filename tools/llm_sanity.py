"""Real-model output sanity check that runs on a (GitHub) CPU.

The mobile bug — Gemma answering the same canned line and looping regardless of
input — is the kind of degeneration this guards against. The pure heuristics
(``looks_degenerate`` / ``outputs_distinct``) are unit-tested in
``tests/test_llm_sanity.py`` with no model; ``main`` downloads the real fast
Gemma 1B GGUF and runs a few prompts on CPU, asserting the answers are distinct
and non-degenerate (run in CI by ``.github/workflows/llm-sanity.yml``).

Note: this exercises the Python ``llama.cpp`` path, not the mobile
``flutter_gemma`` runtime (which can't run on a stock CI runner). It validates
that the model + prompt produce sane output on CPU; the flutter_gemma glue is
guarded separately by the Dart unit test (``mobile/test/llm_glue_test.dart``).
"""
from __future__ import annotations

from collections import Counter

_PROMPTS = [
    "What is the capital of France?",
    "Tell me a short joke.",
    "What is two plus two?",
]

_SYSTEM = "You are a concise, friendly assistant. Answer in one or two short sentences."


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
        if looks_degenerate(out):
            problems.append(f"degenerate output for {prompt!r}: {out!r}")
    if len(pairs) > 1 and not outputs_distinct([o for _, o in pairs]):
        problems.append("all prompts produced the same answer")
    return problems


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Real-model LLM output sanity check.")
    parser.add_argument("--cache-dir", default=".cache/bench-models")
    parser.add_argument("--models-manifest", default=None)
    args = parser.parse_args(argv)

    from huggingface_hub import hf_hub_download

    from core.llm import LlamaCppLLM
    from tools.bench.models import load_manifest

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    coords = load_manifest(args.models_manifest)["fast_gguf"]
    gguf = hf_hub_download(
        repo_id=coords["repo"], filename=coords["file"], cache_dir=args.cache_dir, token=token
    )
    llm = LlamaCppLLM(gguf, n_ctx=2048, n_gpu_layers=0)

    pairs: list[tuple[str, str]] = []
    for prompt in _PROMPTS:
        out = "".join(llm.stream(prompt, system=_SYSTEM)).strip()
        print(f"\n=== {prompt}\n{out}")
        pairs.append((prompt, out))

    problems = _check(pairs)
    if problems:
        print("\nLLM SANITY FAILED:")
        for problem in problems:
            print(" -", problem)
        return 1
    print("\nLLM sanity OK: distinct, non-degenerate answers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
