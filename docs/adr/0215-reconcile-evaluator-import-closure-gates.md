# ADR-0215: Reconcile current evaluator import-closure gates

Date: 2026-10-04
Status: accepted

## Decision

Bind prospective AMI natural-turn and Microsoft AEC evaluator source closures to their actual
clean-process eager imports, including core/engines/_kws_speaker_inference_owner.py and
core/kws_contract.py. AMI's current closure contains 58 files rather than 56. Preserve exact
source hashing and fail-closed validation; do not rewrite earlier locks, receipts or reports.
Treat them as evidence for their original bound implementation, not a new current execution.

In the Parakeet repository-lock identity test, verify public repository bytes and digest,
stage a distinct 0600 private copy and load that copy through the existing private loader.
Assert the returned path, mode and digest and preserve repository bytes/mode. Do not relax
private-input guards or depend on permissions that Git does not preserve at checkout.

## Context / why

The full headless review run exposed three genuine existing gate failures on both the repaired
tree and untouched dab5e15. Two clean-import tests showed these exact KWS modules were missing
from persisted source lists. The public lock's size and digest passed, but the test attempted
to read the checkout's 0664 file through a loader requiring a private 0600 runtime input.
Those checks must describe the current execution and appropriate input domain accurately.

Other provisioning failures are environmental: this execution environment places Git markers
in /tmp and /home/dobo/work. Strict private-artifact tests reject their generated model fixtures
there by design. A paired hermetic test-fixture run outside those markers passes all 36 Anyreach
cases on both current and original source. No marker, artifact boundary or retained evidence
was weakened to make that test pass; task source and reports remain in the prescribed worktree
and scratch area. Test-generated fixtures contain synthetic payloads only.

## Consequences

The focused six cases and complete three affected test modules pass 6 and 187 tests; scoped
Ruff and whitespace are clean. New provenance binds the current source closure. No native
model, real corpus, physical microphone, private recording or quality/default/promotion gate
ran, and previous published artifacts were neither edited nor deleted.

A CI-style full suite requires a framework fixture root outside Git markers for these privacy
tests. A failed run using default /tmp is not a green full-suite receipt. Per-change and full
verification results are recorded separately in STATUS.md and WORKLOG.md.
