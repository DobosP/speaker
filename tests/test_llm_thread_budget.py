"""llm-inference-2: the on-device LLM's CPU threads are budgeted against the
always-on audio path so a llama.cpp generation can't starve the real-time
capture/VAD/barge-in loop.

Tier 0: constructs LlamaCppLLM clients (the llama_cpp import is lazy -- only in
``_ensure`` at first generate -- so construction needs no native lib) and reads
the resolved ``n_threads``. No models, no audio.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.llm import LlamaCppLLM
from core.llm_factory import _auto_llm_threads, build_llms


@pytest.mark.parametrize(
    "cores,expected",
    [
        (1, 1),    # tiny host: never below 1
        (2, 1),    # 2 cores: reserve 2 -> clamp to 1
        (4, 2),    # quad-core phone: 2 for the LLM, 2 reserved
        (6, 4),    # hexa-core phone: 4 for the LLM
        (8, 6),
        (16, 14),
    ],
)
def test_auto_llm_threads_reserves_headroom(cores, expected):
    assert _auto_llm_threads(cores) == expected


def test_auto_llm_threads_always_at_least_one():
    # Even with an absurd reserve, the LLM gets >= 1 thread.
    assert _auto_llm_threads(2, reserve=99) == 1


def _args(**kw) -> SimpleNamespace:
    base = dict(llm="llamacpp", model=None, fast_model=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_llamacpp_unset_threads_are_auto_budgeted():
    """No llm.n_threads -> the LLM is budgeted (NOT None / all-cores), so the
    capture/barge path keeps headroom."""
    config = {
        "llm": {
            "backend": "llamacpp",
            "main_model_path": "main.gguf",
            "fast_model_path": "fast.gguf",
            "n_ctx": 2048,
        }
    }
    main, fast = build_llms(_args(), config)
    assert isinstance(main, LlamaCppLLM) and isinstance(fast, LlamaCppLLM)
    assert main.n_threads == _auto_llm_threads()
    assert main.n_threads >= 1
    # fast tier shares the same budget (sequential with main, not concurrent).
    assert fast.n_threads == _auto_llm_threads()


def test_explicit_n_threads_wins_over_auto_budget():
    config = {
        "llm": {
            "backend": "llamacpp",
            "main_model_path": "main.gguf",
            "n_threads": 3,
        }
    }
    main, _ = build_llms(_args(), config)
    assert main.n_threads == 3
