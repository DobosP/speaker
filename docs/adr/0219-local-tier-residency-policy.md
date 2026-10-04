# ADR-0219: Keep heavy local models available without compulsory startup residency

Date: 2026-10-04
Status: accepted

## Decision

Add explicit `warm_start_policy="fast"` alongside the unchanged `all` default.
Warm media first on the existing single worker, then only the chosen local fast tier
(or the sole local answering tier). Keep main/vision/research, addressing and cleanup
available on demand. Avoid indirectly warming a different model through helper owners.
Add per-role Ollama `main_keep_alive` and `fast_keep_alive`, each falling back to the
existing global `keep_alive`. If both roles address the same daemon model, use fast
retention for both without collapsing their existing client identities.

## Context / why

Startup previously generated on both fast and main, pinning a large vision model
even when every ordinary voice turn used the small tier. The owner requested a smaller
footprint while preserving functionality. This changes residency and cold-start tradeoffs,
not model routing, context, generation limits, tool permissions or local/cloud policy.
GGUF construction retains its existing shared-context lifecycle; these expiration settings
are Ollama-specific.

## Consequences

The main tier can be cold for its first complex/image turn. Expiration applies when
this runtime next requests that daemon model; it does not evict an existing externally
pinned model. Fast-only warm readiness describes its selected warm plan. Native-free
regressions cover usable cold research/images, sole/shared models, helper ownership,
cloud exclusion and actual request forwarding. The affected gate passed 218 tests.
Synthetic delayed/buffered loaders illustrate skipped work only; they are not measurements
of native Ollama RAM savings. See the explicit modes in ADR-0218.
