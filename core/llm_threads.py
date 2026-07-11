"""Bounded CPU thread-pair policy for in-process llama.cpp voice models."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

DEFAULT_LLM_THREAD_RESERVE = 2
DEFAULT_LLM_THREAD_CAP = 4
DEFAULT_LLM_BATCH_THREAD_CAP = 8


@dataclass(frozen=True)
class LlamaCppThreadPair:
    """Effective generation and prompt/batch CPU thread counts."""

    n_threads: int
    n_threads_batch: int
    available_cpus: int


def available_cpu_count() -> int:
    """Return the tightest CPU count visible to the calling thread.

    Python 3.13's ``process_cpu_count`` can be operator-overridden, while Linux
    and Android ``sched_getaffinity`` exposes the current thread's cpuset. When
    both exist, use the smaller valid bound. Other platforms retain the portable
    ``cpu_count`` fallback.
    """

    bounded_counts: list[int] = []
    process_count = getattr(os, "process_cpu_count", None)
    if callable(process_count):
        try:
            count = process_count()
        except (NotImplementedError, OSError, TypeError, ValueError):
            count = None
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            bounded_counts.append(count)

    affinity = getattr(os, "sched_getaffinity", None)
    if callable(affinity):
        try:
            count = len(affinity(0))
        except (NotImplementedError, OSError, TypeError, ValueError):
            count = 0
        if count > 0:
            bounded_counts.append(count)

    if bounded_counts:
        return min(bounded_counts)

    count = os.cpu_count()
    return (
        count
        if isinstance(count, int) and not isinstance(count, bool) and count > 0
        else 1
    )


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def auto_llm_threads(
    cores: Optional[int] = None,
    *,
    reserve: int = DEFAULT_LLM_THREAD_RESERVE,
    cap: int = DEFAULT_LLM_THREAD_CAP,
) -> int:
    """Bound compact-model CPU work while preserving voice-path headroom.

    Keep at least ``reserve`` logical CPUs available where the topology permits,
    and never auto-admit more than ``cap`` workers. Explicit profile values are
    resolved separately and remain authoritative.
    """

    available = (
        available_cpu_count()
        if cores is None
        else _positive_int(cores, name="available CPU count")
    )
    if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve < 0:
        raise ValueError(
            f"thread reserve must be a non-negative integer, got {reserve!r}"
        )
    maximum = _positive_int(cap, name="thread cap")
    return max(1, min(maximum, available - reserve))


def auto_llm_batch_threads(
    cores: int,
    generation_threads: int,
    *,
    reserve: int = DEFAULT_LLM_THREAD_RESERVE,
    cap: int = DEFAULT_LLM_BATCH_THREAD_CAP,
) -> int:
    """Allow faster prompt work only when a generation-sized reserve remains."""

    available = _positive_int(cores, name="available CPU count")
    generation = _positive_int(
        generation_threads,
        name="generation thread count",
    )
    if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve < 0:
        raise ValueError(
            f"thread reserve must be a non-negative integer, got {reserve!r}"
        )
    maximum = _positive_int(cap, name="batch thread cap")
    headroom = max(reserve, generation)
    return max(generation, min(maximum, available - headroom))


def resolve_llamacpp_thread_pair(
    n_threads: object = None,
    n_threads_batch: object = None,
    *,
    available_cpus: Optional[int] = None,
) -> LlamaCppThreadPair:
    """Resolve one explicit, positive pair before loading the native model.

    An omitted generation count uses the bounded voice default. With a fully
    automatic pair, prompt work may rise to eight threads only when at least a
    generation-sized CPU reserve remains; smaller/phone topologies stay paired.
    When generation is explicit and batch is omitted, batch follows it. Both
    rules prevent the Python binding's independent all-CPU default.
    """

    available = (
        available_cpu_count()
        if available_cpus is None
        else _positive_int(available_cpus, name="available CPU count")
    )
    generation = (
        auto_llm_threads(available)
        if n_threads is None
        else _positive_int(n_threads, name="llm.n_threads")
    )
    if n_threads_batch is None:
        batch = (
            auto_llm_batch_threads(available, generation)
            if n_threads is None
            else generation
        )
    else:
        batch = _positive_int(n_threads_batch, name="llm.n_threads_batch")
    return LlamaCppThreadPair(
        n_threads=generation,
        n_threads_batch=batch,
        available_cpus=available,
    )
