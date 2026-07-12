# Task Result — MiniCPM integration and deterministic injected barge gate

Valid until: this branch lands or its implementation changes — then treat as history.

Branch: `fix/inject-barge-gate-determinism`

Status: control-plane lanes integrated; injected barge gate fixed and headless-green.

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
- `git diff --check`: clean.

## Remaining validation

A real installed MiniCPM alias, Docker image, fresh-install download, ordinary
doctor `READY`, production-hybrid Ollama A/B, and live bare-speaker behavior are
not validated by this integration. The injected result does not validate GTCRN,
physical echo, current-room short Stop, v5 identity, or audible stop quality.
