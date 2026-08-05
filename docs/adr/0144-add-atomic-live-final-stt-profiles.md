# ADR-0144: Add atomic live final-STT profiles

Date: 2026-08-05
Status: accepted

## Decision

Add two closed, versioned, session-scoped final-STT profiles named
`sense-voice` and `parakeet-faster-whisper`. Each profile replaces the complete
final recognizer, independent verifier, final-selection resource, and bounded
finalizer policy tuple after device-profile resolution and before readiness.
Expose the same selection through the existing `./live.sh`, core, doctor, and
recorded-evaluation paths. `./live.sh` must pass one requested profile to both
doctor and the core child; selection is in memory only and never rewrites
machine configuration. A missing, malformed, mixed, or incomplete profile must
fail before the microphone opens. Applying either profile marks its final
components required: readiness constructs and decodes the selected final, while
the shared live/replay builders reject recognizer construction failure and
eagerly warm any required verifier instead of degrading the named profile.

Core accepts profile selection only for local Sherpa or replay STT sessions;
console, transport, and enrollment paths reject it instead of recording a
profile identity for a model that never ran.

Keep absence of the option behaviorally unchanged. Keep the legacy
backend-only `--asr-final` research override, but make it mutually exclusive
with an atomic profile and do not treat it as valid profile A/B evidence.
Recorded comparison requires an explicit baseline/candidate profile pair and
rejects mixing that pair with field-level `--set` overrides. Bind each private
run to a safe profile name and canonical profile digest; retain the evaluator's
full effective-config and active-model artifact digests.

`parakeet-faster-whisper` continues to use Parakeet Unified English as the
offline final and Faster-Whisper Small only as its independent endpoint
verifier, with one CTranslate2 worker and one requested CPU thread. Do not add
Faster-Whisper as a direct final backend or count one model twice as independent
evidence. ADR-0080's exact-consensus and protected-control policy and ADR-0133's
authority demotion for nontrivial rewrites remain unchanged.

## Context / why

The machine's ambient configuration already selects Parakeet plus
Faster-Whisper. The documented verifier override therefore produces the same
effective configuration as its supposed baseline: a nominal A/B can compare a
candidate with itself. Conversely, changing only `asr_final_backend` can retain
artifact paths from whichever family setup wrote last, pairing a backend with
the wrong model files. Skipping the local overlay is not a usable control on
this machine because the committed template does not contain its active
streaming model paths.

The EdAcc result prioritizes Faster-Whisper for further testing but measured it
as a complete-PCM final-only comparator against Zipformer. It does not prove
that the production Parakeet/verifier selector beats SenseVoice, nor does it
provide live capture, endpoint, latency, command-safety, or device evidence.
Making Faster-Whisper the direct live final would cross both the ADR-0102
benchmark-only boundary and the evidence required by ADR-0133.

## Consequences

An owner can run the same physical script through the single supported entry
with `./live.sh --final-stt-profile sense-voice` and then
`./live.sh --final-stt-profile parakeet-faster-whisper`. A labelled exact-input
corpus can compare those same two complete profiles without depending on
ambient final-ASR selection. Both paths remain local-only and preserve the
existing synchronized private recording bundle.

The committed profiles use the standard setup locations. A machine without the
selected artifacts fails readiness rather than degrading to streaming or a
partial pair. Device-specific streaming ASR, provider, and general thread
budgets remain outside the final-profile replacement. A custom model layout
must replace the opaque profile map as a whole; partial recursive profile
merges are rejected.

Pre-landing readiness constructed and decoded the exact SenseVoice and
Parakeet finals and warmed the exact Faster-Whisper verifier with CUDA FP16.
That supplies no microphone, speaker, WER, latency, natural-conversation,
command, multi-voice, or adoption result. The normal configured path and
defaults remain unchanged. Promotion still requires the same-corpus recorded
gate, broader command/noise and conversation sources, and a fresh
owner-reviewed open-speaker A/B.
