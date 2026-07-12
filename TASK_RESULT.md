# Task Result — MiniCPM identity and fresh-install readiness integration

Valid until: this integration branch lands or its implementation changes — then treat as history.

Branch: `integration/minicpm-v5-release`

Status: both control-plane lanes integrated; combined headless suite green.

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
- `git diff --check`: clean.
- Later release gates remain pending.

## Remaining validation

A real installed MiniCPM alias, Docker image, fresh-install download, ordinary
doctor `READY`, production-hybrid Ollama A/B, and live bare-speaker behavior are
not validated by this integration. Do not claim them from the headless evidence.
