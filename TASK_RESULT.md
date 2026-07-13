# Task Result — MiniCPM v5 release integration

Valid until: this branch lands or its implementation changes — then treat as history.

Branch: `integration/minicpm-v5-release`

Status: identity, install, memory, injected barge, and enrollment-promotion lanes integrated; final combined gates pending.

## Outcome

- The canonical desktop MiniCPM Q8 alias, source blob, quantization, template,
  and parameters are now one production contract reused by provisioning,
  startup, doctor, live-session preflight, Docker verification, and evaluation
  (ADR-0062).
- Fresh installs now require SciPy/soxr and the selected SenseVoice, GTCRN,
  Kokoro, and speaker-ID artifacts before atomically publishing local config.
  Setup failures propagate; skipped models are incomplete; deferred Ollama
  checks can report only `BASE READY`, never full `READY` (ADR-0063).
- Phone llama.cpp Q4 behavior is unchanged.
- The clean no-device barge profile no longer applies physical echo, level,
  confirmation, or denoising gates. It retains real VAD/sustain/cancel/FIFO
  control; failures and incomplete scenario coverage now return nonzero
  (ADR-0064).

## Exact branch evidence

- Identity branch `fix/minicpm-provision-identity` (`HEAD` before this
  cherry-pick): full `3714 passed, 31 skipped, 9 warnings`; focused identity
  `196 passed`; shared readiness/startup/live/import/Docker `321 passed`; APM
  `6 passed`; deterministic conversation `42/42`, all fourteen scenarios 3/3.
- Fresh-install branch `fix/fresh-install-readiness` (`599d125`): full
  `3692 passed, 31 skipped, 9 warnings in 73.79s`; installer/readiness focused
  `121 passed in 0.39s`; strict recorded/APM `9/6 passed`.
- Both branch suites used injected or device-free boundaries. Neither lane ran
  a model download, real Ollama/model check, network request, microphone, or
  speaker validation.

## Integration verification

- Combined focused identity/fresh tests: `549 passed in 9.70s`.
- Combined full suite: `3728 passed, 31 skipped, 9 warnings in 76.08s`.
- Strict archived recorded-owner replay: `9 passed in 53.66s`; fake streams
  were used and no physical audio hardware was opened.
- APM/DTD regression: `6 passed in 0.69s`.
- Deterministic conversation: `42/42`; every one of the fourteen scenarios
  passed 3/3 and semantic, coverage, A/B, provenance, and warmup gates were
  all true.
- Injected-profile focused tests: `133 passed in 1.86s`.
- Full suite after the profile and exact-coverage changes: `3737 passed,
  31 skipped, 9 warnings in 80.18s`.
- Strict archived recorded-owner replay after the shared profile change:
  `9 passed in 65.78s`; fake streams, no physical hardware.
- Reproduced baseline `203334`: cuts `1/2, 2/2, 1/2`, but old CLI exit 0.
- Honest intermediate `204845`: cuts `1/2` in all three repeats and exit 1,
  isolating inherited GTCRN as the remaining clean-profile mismatch.
- Controlled no-denoise `205151`: `2/2` cuts, zero self-interrupts, exit 0.
- Final exact gate `205351`: all three repetitions full-duplex `ok`, each `2/2`
  cuts, zero self-interrupts; aggregate `6/6`, exit 0. No hardware opened.
- Memory provenance branch `ed548d1`: focused `179 passed`; full `3718 passed,
  31 skipped, 9 warnings`; strict recorded/APM `9/6`; deterministic conversation
  `42/42`; independent stop-ship review green. Integration-focused memory tests
  then passed `180/180`.
- `git diff --check`: clean.

## Remaining validation

A real installed MiniCPM alias, Docker image, fresh-install download, ordinary
doctor `READY`, production-hybrid Ollama A/B, and live bare-speaker behavior are
not validated by this integration. The injected result does not validate GTCRN,
physical echo, current-room short Stop, v5 identity, or audible stop quality.

## Enrollment promotion outcome

Implemented a device-free, fail-closed workflow for activating an explicitly
accepted isolated v5 speaker enrollment. Preparation schema v2 binds paths plus
full metadata and SHA-256 lineage for the primary config, empty reservation,
historical v4, and backup. Promotion uses descriptor-bound reads, validates
strict live-v5 provenance and model/rate/dimension compatibility, and requires
all six protected config/enrollment objects to be path- and inode-disjoint. It
publishes an independent candidate-derived mode-600 accepted copy, safely adopts
only an exact private orphan, then atomically changes only the primary pointer.

Strict file/directory fsyncs and a stable advisory lock make outcomes
commit-aware for cooperating promoters: 0 means the primary pointer committed;
2 means refused before this invocation committed accepted/config state; 3 means
the accepted copy is confirmed durable while the original pointer is confirmed
inactive; and 4 means the filesystem result is ambiguous and needs inspection.
The lock cannot exclude a non-cooperating writer. Historical v4, its backup, the
isolated candidate, and unrelated config values remain unchanged.

## Files changed

- `tools/promote_enrollment.py` — promotion API/CLI and atomic staging/adoption.
- `tools/prepare_enrollment.py` / `core/enroll.py` — schema-v2 lineage capture
  and pre-capture validation.
- `tests/test_promote_enrollment.py` — deterministic success, refusal, race,
  retry, provenance, secrecy, and commit-point coverage.
- `tests/test_prepare_enrollment.py` — schema-v2 preparation lineage coverage.
- `docs/adr/0066-promote-accepted-v5-enrollment-without-replacing-v4.md` —
  decision and failure semantics.
- `STATUS.md` — current implemented promotion state and focused evidence.
- `docs/agent-map.md` / `docs/agent-testing.md` — task routing and exact gate.
- `docs/2026-07-12-v5-bare-speaker-acceptance.md` — post-acceptance command and
  exit handling.

## Commands run and exact results

```text
/home/dobo/work/speaker/.venv/bin/python -m pytest \
  tests/test_promote_enrollment.py tests/test_prepare_enrollment.py -q
107 passed

/home/dobo/work/speaker/.venv/bin/python -m pytest \
  tests/test_promote_enrollment.py tests/test_prepare_enrollment.py \
  tests/test_enroll.py -q
146 passed

/home/dobo/work/speaker/.venv/bin/python -m pytest tests -q
3765 passed, 31 skipped, 9 warnings

/home/dobo/work/speaker/.venv/bin/python -m pytest \
  tests/test_apm_double_talk.py -q
6 passed

git diff --check
PASS (no output)
```

The focused suite is device-free and used temporary directories only. No real
config, enrollment, audio device, model, log corpus, or network service was
opened or modified.

## Risks / manual review

- `--accept-live-gate` records an explicit operator assertion; it does not
  manufacture or independently grade physical bare-speaker evidence.
- Accepted-copy publication and config activation are necessarily two ordered
  filesystem commits. Exit 3 makes the confirmed safe intermediate state
  retryable; exit 4 prevents uncertain durability or replacement from appearing
  as a refusal.
- The advisory config lock protects cooperating promotion commands only. A
  process that ignores it remains outside the lock guarantee.
- The focused, full-repository, and APM headless gates are green. Real-model and
  live audio gates were not run; the orchestrator must run the final combined
  integration gates before landing.

## Merge recommendation

Focused promotion/preparation tests and whitespace are green. Review the new
CLI contract and ADR, then run the combined full landing gate. Do not claim live
v5 or bare-speaker acceptance until the operator actually completes it.
