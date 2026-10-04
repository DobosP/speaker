# ADR-0218: Add explicit capability-preserving performance modes

Date: 2026-10-04
Status: accepted

## Decision

Add startup/session `current`, `responsive` and `compact` selections through the
same pure resource policy in core, doctor and live.sh. Current is an exact no-op.
Responsive selects the bound English INT8 Zipformer tuple and installed LibriTTS
VITS; compact selects the same streaming tuple and Kitten Nano 0.8 INT8. Both
request two native ASR/TTS threads and use ADR-0219's fast-only warm/residency policy.
Apply device -> performance -> explicit final-STT -> speaker-identity override.
Verify the complete file/directory checksum contract before applying any preset;
record path-free mode identity/digest. Reject uncertainty, mixed model families,
incompatible active BPE hotword context and corrupted/missing assets.

## Context / why

The owner requested concrete resource modes and newer measured models while
preserving functionality. ADR-0216 supplies development corpus tradeoffs and
ADR-0217 supplies the explicit Kitten backend. A model/resource preset must not
silently widen cloud/raw-audio egress or remove tools, memory, vision, speaker
policy, endpoint rules, DSP, final recognition/verifier, context or generation caps.
The selected physical voice changes with its model, while expressive voice policy
remains intact and unsupported voice IDs retain existing clamping behavior.
The ASR/TTS/LLM implementations already execute native inference; model residency
and data staging provide larger footprint gains than rewriting the control plane.

## Consequences

Defaults and persisted local setup remain unchanged. The owner can select a
session mode without an active-session model swap. The existing CUDA-required
final verifier stays configured unless the owner explicitly chooses a different
complete final-STT profile. Per-engine thread requests are not a total CPU budget.
Current profiles are portable relative paths to locally provisioned assets;
asset preflight performs bounded filesystem reads before capture, never in an
audio callback. Snapshot/path/inventory consistency checks cover preflight,
not filesystem mutations after verification/native model opening.

24 deterministic policy tests cover preserved capabilities/expressive policy,
mode/hash/root binding, corrupt assets, mutation/path/inventory races, hotword
incompatibility and doctor/live selection parity. The previous wider focused gate
passed 329; combined integration verification is recorded in STATUS/WORKLOG.
Native CPU ASR/TTS pairs ran locally on silence/public text, with separate model
bindings and no final recognizer/verifier/LLM/DSP/audio device. Development quality,
phone thermals/latency, actual audibility and bare-speaker A/B remain open.
See [the mode guide](../performance_modes.md); mobile budgets are ADR-0221.
