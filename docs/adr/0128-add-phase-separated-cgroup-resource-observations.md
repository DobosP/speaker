# ADR-0128: Add phase-separated cgroup resource observations

Date: 2026-08-05
Status: accepted

Refines: ADR-0099, ADR-0119

## Decision

Add supervisor-owned, observation-only resource snapshots for the Parakeet
Realtime EOU and parakeet.cpp benchmark workers already running in verified
systemd-user cgroup-v2 scopes. Pin the verified scope internally, recheck the
worker's membership and exact hard limits at every sample, and expose no PID or
scope path. Record exactly three ordered controller boundaries: `startup` after
an accepted Ready event, `first_use` after the first fully validated terminal,
and `resident` after all requested cases but immediately before shutdown. Use
controller-monotonic elapsed time plus bounded read-only `memory.current`,
`memory.peak`, and `cpu.stat` observations. Bind the exact completed-case count
into the resident boundary and fail the scoped benchmark report closed if the
three observations are not complete and internally monotonic.

Keep adapter-reported Ready/final resource samples, `max_reported_rss_mb`, and
adapter-internal `model_load_ms` unchanged and semantically separate. Label OS
cache control as `uncontrolled`, record that no cache eviction was performed,
and make no cache-state claim. Never describe these observations as cold or
warm. Do not add a substitute for workers without a verified dedicated scope,
and do not define a resource threshold, comparison verdict, quality decision,
promotion, runtime/default, tool, device, or live authority here.

## Context / why

The existing worker reports sample one process at Ready and final boundaries;
they explicitly do not report a process peak. Existing cgroup evidence proves
hard limits before Ready but discards the scope identity and observes no usage.
Adapter `model_load_ms` also starts inside adapter construction, excluding the
supervisor launch, interpreter startup, cgroup migration, receipt checks, and
Ready IPC. Those values cannot answer how whole-scope resource use differs at
startup, first use, and the end of a requested evaluation.

The Linux cgroup-v2 contract defines `memory.current` as current memory for the
cgroup and descendants, `memory.peak` as a cumulative scope peak, and the
required `cpu.stat` time counters as cumulative microseconds. Therefore
`memory.current` may fall and includes cgroup-charged file/page cache;
`memory.peak` is not a phase-local peak; and only differences between ordered
CPU snapshots describe intervals. The host and page-cache state are not
controlled, so sequential runs cannot support causal cache comparisons. A
directory descriptor, bounded reads through EOF, membership rechecks, and
counter monotonicity prevent a moved/replaced scope or partial multi-line read
from becoming valid evidence. The report explicitly labels the separate cgroup
file reads as sequential rather than atomic.

## Consequences

Applicable isolated reports now distinguish controller-observed startup,
first-use, and post-suite resident boundaries without changing the worker wire
protocol, adapter behavior, model selection, or application runtime. The
resident sample is a precise pre-shutdown observation point, not an idle
plateau, leak test, process-lifetime result, post-exit emptiness proof, or VRAM
measurement. Actual values remain nondeterministic; deterministic headless
tests cover parsing, ordering, privacy, identity/migration failure, cumulative
counter validation, and legacy report compatibility without a model, GPU, or
audio device.

Zipformer and the other legacy adapters have no comparable dedicated cgroup and
retain their previous report shape. First-use cost is corpus-content dependent,
and OS cache, host contention, thermals, architectures, and device classes
remain uncontrolled. A later predeclared policy and real multi-host runs must
establish resource thresholds or comparative acceptance; ADR-0127's
four-source coverage gate remains non-promotional.
