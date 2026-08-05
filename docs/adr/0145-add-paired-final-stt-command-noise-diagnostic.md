# ADR-0145: Add paired final-STT command/noise diagnostic

Date: 2026-08-06
Status: accepted

## Decision

Extend the exact public command/noise production-final evaluator with one
opt-in, ordered profile comparison: `sense-voice` control followed by
`parakeet-faster-whisper` candidate. Resolve both complete final-STT profiles
independently from one device-effective base, bind both configuration and
active-model closures plus source/runtime identities before either replay,
run the same loaded corpus sequentially, and publish one aggregate-only
schema-v3 report only after both cells complete. SenseVoice must construct a
required final recognizer with no verifier and account for every verifier
outcome as unavailable. The Parakeet cell must construct its required final,
warm Faster-Whisper on CUDA FP16, and attest at least one completed verifier
decode.

Require the unchanged streaming aggregates to match. Report a mismatch as
`inconclusive_streaming_control_mismatch`, while retaining operational exit 0
when both cells and atomic publication completed. Never select a winner or
authorize runtime-default promotion from this diagnostic. Preserve the
existing no-profile schema-v2 path and all persisted configuration, setup,
core, and `./live.sh` defaults.

Retain SenseVoice as the control for the owner live A/B and reject promotion
of the Parakeet/Faster-Whisper profile on the command/noise result below.

## Context / why

ADR-0118 measured the ambient configured Parakeet/Faster-Whisper selector, but
it did not compare that selector with the newly atomic SenseVoice profile.
ADR-0144 prevented nominal self-comparisons and mixed artifact families, yet
its pre-landing real-model checks established readiness rather than relative
command quality. A fixed same-PCM comparison was required before asking the
owner to spend time on physical A/B recordings.

The exact run used 57 locked Speech Commands v0.02/ECCC v1.2 cases, one repeat,
and the `desktop_gpu_4090` device profile. Both cells produced 58 terminal
decisions and byte-equivalent aggregate streaming controls. SenseVoice's
selected view retained 17/26 command hits, 1/31 monitored negative-case false
activations, 0.8095 target precision, and 0.8936 WER. Its clean GSC result was
16/20 commands at 0.30 WER; noisy ECCC was 1/6 commands, 1/21 false
activations, and 1.3333 WER.

The Parakeet/Faster-Whisper selected view reached 19/26 commands, 5/31 false
activations, 0.7308 target precision, and 0.8085 WER. Its clean GSC result
improved to 18/20 at 0.10 WER, but noisy ECCC remained 1/6 at 1.3333 WER while
false activations rose to 5/21. The candidate therefore recovered two clean
commands and reduced aggregate WER, but added four noisy false activations
without one additional noisy-command hit. That is the wrong trade for an
action-capable voice agent.

The mode-0600, single-link, 48,704-byte private schema-v3 report has SHA-256
`63b72344f2ac87579fb855d5883e02ee3a9537fc7ad193ca6e736f87320397ae`.
Its committed path-free aggregate is
`docs/evidence/2026-08-06-final-stt-profile-command-ab.json`.

## Consequences

The exact production-selector command gate can now compare the two complete
live-selectable profiles without ambient configuration or field-level mixing.
Both model closures must be present before baseline replay, each native engine
is stopped before the next cell, and candidate failure publishes nothing.
Schema-v2 configured-path reports remain valid historical evidence.

No default changes. This result begins after PCM, supplies neither VAD-owned
speech duration nor live offline-recovery authorization, and runs profiles
sequentially. It is not capture, endpoint, native-reader, AEC, open-speaker
barge-in, enrollment, multi-voice authority, tool, TTS, natural-conversation,
latency, resource-acceptance, or live evidence. The next decision gate remains
fresh owner target/negative recordings followed by the ordered bare-speaker
`./live.sh` profile A/B.
