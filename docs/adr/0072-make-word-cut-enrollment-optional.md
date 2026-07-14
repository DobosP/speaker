# ADR-0072: Make word-cut enrollment optional

Date: 2026-07-14
Status: accepted

## Decision

Make single-voice word-cut independent of speaker enrollment. Default
`barge_word_cut_require_speaker` to false in both the runtime and committed
configuration, and interpret an omitted key as false in engine and readiness
paths. On a verified echo-cancel route, identity-free authority still requires
either a typed canonical STOP-class control or at least
`max(4, barge_word_cut_min_words)` novel non-own words. No local value may let
zero-to-three generic words, empty PCM, silence, or generic recognized own echo
cut, seed normal ASR, or acquire continuation authority.

Retain `barge_word_cut_require_speaker=true` as an explicit multi-voice filter
for generic overrides. That mode keeps the compatible-enrollment readiness
requirement, warm bounded speaker comparison, purpose-specific thresholds,
ambiguity/retry behavior, and the same hard lexical floor. Novel exact
Stop/cancel remains an open fail-safe control in either mode. Keep normal-final
`speaker_gate_input`, enrollment
capture/provenance, isolated preparation, and accepted-candidate promotion
unchanged. A playback-time cut never grants owner verification or
side-effecting/action tool authority; those remain separate typed gates.

Preserve ADR-0042's narrower fail-closed rule independently of that optional
filter. A STOP-class control or attested STOP repair that reads like current or
recent own TTS must still obtain compatible enrolled-speaker authority before
cutting, even when `barge_word_cut_require_speaker=false`. Missing, cold, busy,
rejected, or errored speaker authority abstains. Novel canonical STOP-class
controls remain enrollment-free; this ambiguity guard does not make ordinary
exact Stop enrollment-dependent. ADR-0042 therefore remains accepted.

This supersedes ADR-0045's mandatory speaker-acceptance half while retaining and
extending its four-word floor to identity-free mode. It also supersedes
ADR-0071's statement that enrollment-off must remain only an in-memory
diagnostic. ADR-0071's rejected-v5 verdict, bounded scalar evidence, cleanup,
and upstream physical diagnosis remain facts.

## Context / why

The owner chose enrollment as an optional control for rooms with multiple
voices, not a prerequisite for ordinary single-user interruption. The prior
production default coupled two separate safeguards: lexical evidence against
garbled echo and biometric identity against another speaker. That coupling made
a compatible embedding and warmed speaker model a startup requirement even
when no competing voice was present.

The 2026-07-14 physical A/B does not justify treating enrollment as the barge
mechanism. Exact Stop failed both with enrollment enabled and disabled; the
enrollment-off run showed playback-time VAD at zero and never started the energy
fallback. The unresolved live blocker is therefore before identity acceptance.
Conversely, ADR-0045's silent false cuts prove that identity-free mode must not
restore zero-word audio authority or a configurable one-to-three-word floor.

## Consequences

- Active word-cut can start without a speaker model or enrollment when the
  optional filter is false; doctor reports speaker ID as advisory.
- Novel exact canonical controls remain an open bounded authority in either
  mode; four-or-more-word novel overrides are identity-free only when the
  optional filter is off. Generic own text and zero-to-three fragments remain red.
- Current/recent-own-TTS ambiguous STOP controls and attested repairs retain
  ADR-0042's enrolled-speaker tie-breaker regardless of the optional generic
  filter; unavailable authority fails closed.
- Setting the optional filter true still fails readiness/startup without a
  compatible enrollment and retains the enrolled-speaker rejection path for
  generic overrides; novel exact Stop/cancel remains the fail-safe exception.
- Normal-final identity gating and all biometric files/workflows remain present;
  this decision does not delete, promote, or rewrite enrollment state.
- Existing owner-verification and sensitive-action policy is unchanged. An
  identity-free word-cut never mints `VERIFIED`; only the unchanged normal-final
  identity gate can independently do so.
- Physical bare-speaker barge-in remains unvalidated. The next live run must
  diagnose capture, OS echo cancellation, calibration, VAD/energy, denoise, and
  decoder feed with the optional word-cut filter off, then demonstrate prompt
  exact Stop and no self-cut before wider acceptance.
