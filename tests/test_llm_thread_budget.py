"""llm-inference-2: the on-device LLM's CPU threads are budgeted against the
always-on audio path so a llama.cpp generation can't starve the real-time
capture/VAD/barge-in loop.

Tier 0: constructs LlamaCppLLM clients (the llama_cpp import is lazy -- only in
``_ensure`` at first generate -- so construction needs no native lib) and reads
the resolved ``n_threads``. No models, no audio.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from core.llm import LlamaCppLLM
from core.llm_factory import _auto_llm_threads, build_llms
from core.llm_threads import (
    LlamaCppThreadPair,
    auto_llm_batch_threads,
    available_cpu_count,
    resolve_llamacpp_thread_pair,
)


@pytest.mark.parametrize(
    "cores,expected",
    [
        (1, 1),    # tiny host: never below 1
        (2, 1),    # 2 cores: reserve 2 -> clamp to 1
        (4, 2),    # quad-core phone: 2 for the LLM, 2 reserved
        (6, 4),    # hexa-core phone: 4 for the LLM
        (8, 4),    # compact-model ceiling preserves extra voice headroom
        (16, 4),
        (32, 4),   # never treat hybrid logical CPUs as homogeneous workers
    ],
)
def test_auto_llm_threads_reserves_headroom(cores, expected):
    assert _auto_llm_threads(cores) == expected


def test_auto_llm_threads_always_at_least_one():
    # Even with an absurd reserve, the LLM gets >= 1 thread.
    assert _auto_llm_threads(2, reserve=99) == 1


def test_available_cpu_count_prefers_process_aware_probe(monkeypatch):
    monkeypatch.setattr(os, "process_cpu_count", lambda: 5, raising=False)
    monkeypatch.setattr(
        os,
        "sched_getaffinity",
        lambda _pid: set(range(7)),
        raising=False,
    )
    monkeypatch.setattr(os, "cpu_count", lambda: 99)

    assert available_cpu_count() == 5


def test_available_cpu_count_uses_affinity_before_host_count(monkeypatch):
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.setattr(
        os,
        "sched_getaffinity",
        lambda _pid: {2, 3, 4},
        raising=False,
    )
    monkeypatch.setattr(os, "cpu_count", lambda: 99)

    assert available_cpu_count() == 3


def test_available_cpu_count_uses_tighter_affinity_than_process_override(monkeypatch):
    monkeypatch.setattr(os, "process_cpu_count", lambda: 32, raising=False)
    monkeypatch.setattr(
        os,
        "sched_getaffinity",
        lambda _pid: set(range(4)),
        raising=False,
    )
    monkeypatch.setattr(os, "cpu_count", lambda: 64)

    assert available_cpu_count() == 4


@pytest.mark.parametrize(("host_count", "expected"), [(6, 6), (None, 1)])
def test_available_cpu_count_portable_fallback(monkeypatch, host_count, expected):
    monkeypatch.delattr(os, "process_cpu_count", raising=False)

    def unavailable_affinity(_pid):
        raise OSError("not supported")

    monkeypatch.setattr(os, "sched_getaffinity", unavailable_affinity, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: host_count)

    assert available_cpu_count() == expected


@pytest.mark.parametrize("process_result", [None, NotImplementedError])
def test_available_cpu_count_ignores_unavailable_process_probe(
    monkeypatch,
    process_result,
):
    def process_count():
        if isinstance(process_result, type) and issubclass(process_result, Exception):
            raise process_result
        return process_result

    monkeypatch.setattr(os, "process_cpu_count", process_count, raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(), raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 7)

    assert available_cpu_count() == 7


@pytest.mark.parametrize(
    ("cores", "generation", "batch"),
    [
        (2, 1, 1),
        (4, 2, 2),
        (6, 4, 4),
        (8, 4, 4),
        (9, 4, 5),
        (12, 4, 8),
        (32, 4, 8),
    ],
)
def test_auto_batch_threads_expand_only_with_generation_sized_headroom(
    cores,
    generation,
    batch,
):
    assert auto_llm_batch_threads(cores, generation) == batch


def test_thread_pair_auto_bounds_both_phases():
    pair = resolve_llamacpp_thread_pair(available_cpus=32)

    assert pair.n_threads == 4
    assert pair.n_threads_batch == 8
    assert pair.available_cpus == 32


def test_thread_pair_partial_and_full_overrides():
    assert resolve_llamacpp_thread_pair(3, available_cpus=8).n_threads_batch == 3
    pair = resolve_llamacpp_thread_pair(None, 5, available_cpus=8)
    assert (pair.n_threads, pair.n_threads_batch) == (4, 5)
    pair = resolve_llamacpp_thread_pair(2, 6, available_cpus=8)
    assert (pair.n_threads, pair.n_threads_batch) == (2, 6)


@pytest.mark.parametrize("value", [True, False, 0, -1, 2.0, "2"])
@pytest.mark.parametrize("field", ["generation", "batch"])
def test_thread_pair_rejects_non_positive_non_integer_values(value, field):
    kwargs = {"n_threads" if field == "generation" else "n_threads_batch": value}

    with pytest.raises(ValueError, match="positive integer"):
        resolve_llamacpp_thread_pair(**kwargs, available_cpus=8)


def test_provider_boundary_resolves_pair_before_model_loading():
    llm = LlamaCppLLM("not-loaded.gguf", n_threads=2)

    assert (llm.n_threads, llm.n_threads_batch) == (2, 2)
    assert (llm._n_threads_source, llm._n_threads_batch_source) == (
        "explicit",
        "paired",
    )
    with pytest.raises(ValueError, match="llm.n_threads_batch"):
        LlamaCppLLM("not-loaded.gguf", n_threads_batch=0)


def test_provider_records_auto_and_explicit_batch_sources():
    auto = LlamaCppLLM("not-loaded.gguf")
    mixed = LlamaCppLLM("not-loaded.gguf", n_threads_batch=3)

    assert (auto._n_threads_source, auto._n_threads_batch_source) == (
        "auto",
        "auto",
    )
    assert (mixed._n_threads_source, mixed._n_threads_batch_source) == (
        "auto",
        "explicit",
    )


def test_provider_warns_when_explicit_pair_exceeds_headroom_ceiling(caplog):
    LlamaCppLLM("not-loaded.gguf", n_threads=10_000)

    assert "voice-headroom ceiling" in caplog.text


def test_provider_warns_when_explicit_pair_equals_all_visible_cpus(
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(
        "core.llm.resolve_llamacpp_thread_pair",
        lambda *_args: LlamaCppThreadPair(4, 4, 4),
    )

    LlamaCppLLM("not-loaded.gguf", n_threads=4)

    assert "voice-headroom ceiling" in caplog.text


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
    expected = resolve_llamacpp_thread_pair()
    assert isinstance(main, LlamaCppLLM) and isinstance(fast, LlamaCppLLM)
    assert main.n_threads == expected.n_threads == _auto_llm_threads()
    assert main.n_threads_batch == expected.n_threads_batch
    assert main.n_threads >= 1
    # fast tier shares the same budget (sequential with main, not concurrent).
    assert fast.n_threads == expected.n_threads
    assert fast.n_threads_batch == expected.n_threads_batch


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
    assert main.n_threads_batch == 3


def test_explicit_batch_threads_win_over_paired_default():
    config = {
        "llm": {
            "backend": "llamacpp",
            "main_model_path": "main.gguf",
            "n_threads": 3,
            "n_threads_batch": 4,
        }
    }

    main, _ = build_llms(_args(), config)
    assert (main.n_threads, main.n_threads_batch) == (3, 4)
