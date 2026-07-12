# ADR-0062: Pin MiniCPM Q8 provisioning and readiness identity

Date: 2026-07-12
Status: accepted

## Decision

Define one production-owned identity contract for the desktop Ollama
MiniCPM5-1B tier: the exact `minicpm5-1b:q8` alias, official OpenBMB Q8 source,
pinned full model-layer digest, `Q8_0` quantization, canonical ChatML template,
and exact parameter set. Provisioning reports success only after inspecting
both the created alias and source against that contract. Native startup,
doctor, live-session preflight, Docker verification, and conversation
evaluation reuse the same pure verifier. Missing, renamed, unreachable, or
mismatched identities fail closed with the canonical setup command.

Keep presence-only readiness for unrelated Ollama models. Keep phone-class
llama.cpp Q4 under its existing file, template, and abort-ABI gates; do not
assert or infer a Q4 digest from the desktop Q8 artifact.

## Context / why

Ollama alias-name presence and a successful `ollama create` do not prove which
weights, quantization, template, or parameters will serve requests. The setup
helper previously accepted arbitrary source, alias, and Modelfile overrides,
then printed `ready` without inspecting the result. Shared runtime readiness
checked only installed names. The conversation evaluator had the only deep
identity verifier, but it depended on setup-owned constants and could redefine
the expected alias, source, or Modelfile per call.

Duplicating that verifier across setup and readiness was rejected because its
blob, template, and parameter rules could drift. Treating an effective-config
digest as sufficient was also rejected: a freshly computed digest proves only
that some configuration was observed, not that it is the supported one.
Readiness must inspect the same configured Ollama host used by generation.

## Consequences

The canonical verifier performs no process, network, daemon, model, or audio
access; callers inject `show` and own transport construction. Unit tests use
only fake list/show responses. Setup validates the committed Modelfile, runs
the exact pull/create commands, and verifies afterward; a failed verification
leaves the alias available for diagnosis but exits nonzero and never says
`ready`. Arbitrary MiniCPM aliases and extra behavior-changing parameters fail.

Production readiness creates one host-bound Ollama client for list and show,
so doctor and native startup cannot inspect a different default daemon. The
Docker console flow retains its container-local create sequence and adds a
read-only `--verify-only` step through the speaker container. This decision
does not claim a real-daemon, real-model, Docker-build, or live-audio run; those
remain separate validation when the relevant environment is available. The
device-free implementation gate passed 196 focused identity tests plus 321
shared readiness, startup, live-session, import, and Docker-entrypoint tests.
