# ADR-0223: Run GitHub Actions on demand

Date: 2026-10-04
Status: accepted (explicit owner request)

## Decision

Run GitHub Actions only through `workflow_dispatch`, when the owner asks or a
specific change needs a remote validation, audit, build, or deployment. Remove
push, pull-request, label, and recurring schedule triggers. Preserve the useful
jobs, dispatch inputs, credentials handling, and existing test/security gates.
A workflow may be dispatched when concretely needed; no periodic or speculative
runs are authorized. This owner decision supersedes earlier automatic-trigger
preferences only; application safety and correctness requirements still apply.

Real-model benchmarks, cloud probes, APK publishing, and secret scans remain
manual controls. Existing provider/model/scanner compatibility limits still apply.

## Context / why

The owner did not intend to run CI continuously and requested that GitHub jobs
run "only when mentioned by me or they are really needed, not all the time."
Existing automatic workflows produced runs on ordinary pushes, PRs, and timers.
Moving the trigger to manual dispatch implements that request without deleting
useful checks or changing billing/spending controls.

## Consequences

Start the relevant workflow from Actions > Run workflow, or through the GitHub
CLI/API, with the intended branch/tag and existing inputs. A deliberate remote
run still consumes Actions resources and remains subject to the current budget.
Older feature branches must incorporate this trigger policy before publication;
otherwise their older YAML can still start a run. Restore automatic triggers
only after a new owner decision. Application, release, credential, and security
gates continue to apply when a manual job is run.
