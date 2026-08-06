# ADR-0152: Add session-only speaker identity-off mode

Date: 2026-08-06
Status: accepted

## Decision

Expose `--no-speaker-enrollment` only on a non-enrollment local session and
through the existing `./live.sh` physical entry. The launcher forwards that
selector exactly once to both the shared doctor and core child. After device
and atomic final-STT profile selection, all three paths use one pure effective-
config transform that clears only `speaker_enroll_embedding` and
`speaker_enroll_wav` in memory. It does not mutate or probe an enrollment file,
and it preserves the configured speaker model, input gate, word-cut policy,
thresholds, tools, and defaults.

The transform reuses ADR-0137's speaker-identity activation predicate. If the
selected no-in-app-AEC word-cut path explicitly requires speaker identity, it
rejects the identity-off session; the launcher does so before Ollama or the
host echo-control route is prepared. It never converts a multi-voice policy to
identity-free operation. An inert require-speaker value remains unchanged.

A successful session records only
`speaker_identity_policy=off_for_session` in the private run summary. Startup
and doctor output state that identity and speaker verification are unavailable
for the process while configured references and files remain unchanged.
Omitting the selector restores configured behavior on the next process.

This is an identity downgrade, not a safe or private mode. Any audible speaker
may converse and invoke read tools such as vault search, and may cause web
egress. Existing direct-live-plus-confirmation routes for reminders and
allowlisted apps also remain available. The selector never mints owner trust;
verified-owner machine control remains governed by the existing authority
configuration. ADR-0133's final-selection demotion and every capability's
authority contract remain unchanged.

ADR-0072 lexical interruption semantics also remain unchanged: an identity-
free, verified echo-cancel route may accept a novel exact stop/cancel or at
least four novel non-own words, while a stop-class control ambiguous with
current or recent TTS still needs speaker authority. A barge-in cut grants no
owner or tool authority.

## Context / why

The laptop has a valid persisted enrollment. Under ADR-0137, the existing
reference correctly activates speaker identity even though generic word-cut is
identity-optional. That made the planned owner STT/barge-in A/B incapable of
proving enrollment-free behavior. Editing, moving, or deleting the enrollment
would be stateful and could damage the multi-voice setup; changing
`speaker_gate_input` or `barge_word_cut_require_speaker` would weaken unrelated
policy and could hide an unsafe configuration conflict. A separate live script
would also violate the single-entry contract.

The session-only effective-config layer provides reproducible evidence while
preserving the installed enrollment for later multi-voice use. The explicit
warning is required because disabling identity does not disable the assistant's
read and confirmed low-risk tool surfaces.

## Consequences

The matched SenseVoice and Parakeet/Faster-Whisper physical A/B can now run
with enrollment retained on disk but provably unused by the process. Doctor,
readiness, and engine construction see the same masked references, and the
speaker model remains unallocated. A following normal session sees the
configured references again.

Headless tests cover both final-STT profiles, pure non-mutation, active versus
inert multi-voice policy, pre-host rejection, doctor/core parity, safe summary
identity, persisted-byte preservation, process isolation, and zero gate
allocation. Existing owner-authority, final-trust-lineage, lexical word-cut,
and APM/DTD suites remain the regression boundary. No live result, STT quality
claim, model/default promotion, or authority change follows until the owner
completes and reviews the recorded physical A/B.
