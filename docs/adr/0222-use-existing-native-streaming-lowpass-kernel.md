# ADR-0222: Use the existing native streaming low-pass kernel

Date: 2026-10-04
Status: accepted

## Decision

Use SciPy's compiled single-section float64 SOS kernel for StreamingLowpass when
available. Preserve the original scalar recurrence as the dependency-free/error
fallback. Convert output once to float32 and publish both state values only after
successful result validation. Preserve empty/bypass/reset/nonfinite behavior and
per-utterance state ownership. Do not add a Rust extension for this measured seam.

## Context / why

The owner requested less resource use, including consideration of small Rust
modules. Profiling found Python's per-sample low-pass recurrence measurable, while
inference already executes in native libraries. Existing SciPy supplies the needed
kernel without a new build/ABI dependency. The initial lfilter prototype changed
some float64 state bits and was rejected. One-section sosfilt reproduced output
and state exactly across the tested rate/cutoff/Q and chunk-partition matrix.

## Consequences

The implemented kernel measured p50 101.04 us/p95 131.59 us versus scalar p50
871.26 us/p95 1002.68 us on 300 synthetic 2400-sample/100-ms chunks: about 8.62x
faster for this CPU kernel. Tracemalloc transient peak increased from 10,157 to
40,779 bytes (about 30 KiB per measured chunk). The cold SciPy import was excluded;
SciPy is already an optional audio/DSP dependency with existing native paths.
139 targeted regressions passed, including exact float32 output and float64 state,
partitioning, reset, failures, bypass, nonfinite behavior and APM/DTD adjacency.

These are synthetic kernel measurements, not whole-assistant CPU/RSS, phone,
acoustic quality, audibility or live barge-in evidence. Ordinary mathematical/state
behavior is preserved in the tested matrix; a different library/platform still
needs its deterministic gate. A future Rust PCM owner/shared pool requires a
separate measured scheduling/allocation bottleneck and the ADR-0214 architecture
criteria; this optimization does not move the control plane or callback ownership.
