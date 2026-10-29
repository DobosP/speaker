# ADR-0256: Corroborate Qwen dialogue edits without changing control authority

Date: 2026-10-29
Status: accepted for software implementation; paired recorded qualification pending

## Context

ADR-0255's CPU Qwen primary can recognize the recordings better while the
selected agent transcript still falls back to streaming. Exact whole-phrase
agreement discards a correction when the independent verifier supports that
correction but makes a separate error elsewhere. Raw-model WER does not prove
that a less constrained selector is safe or improves selected WER.

## Decision

Retain `verify_asr_consensus` and its exact whole-transcript voting, tie,
empty-veto and control behavior. Only the explicitly selected `qwen_cpu`
primary with a distinct `sense_voice` or `parakeet_tdt_cpu` verifier can use a
separate `corroborated_primary` dialogue fallback after `tie` or `no_quorum`.
This is per-edit acoustic corroboration, not whole-transcript consensus.

Return one complete existing Qwen rendering, never a splice. Every Qwen change
must occur as the identical contiguous token replacement/insertion span at the
identical streaming-source indices in the independent verifier. On unchanged
spans, the actual streaming recognizer supplies the second vote. Separate
verifier errors may remain only where streaming already supports Qwen; an
adjoining broader verifier edit cannot support a narrower primary change.

Use the existing Unicode-preserving exact tokenization, not the English WER
normalizer, word bags, subsequences, edit-distance thresholds or hints. A
minimum Levenshtein alignment is usable only when its global optimal path is
unique; path counts saturate at two. Any ambiguity, primary deletion (including
within a mixed edit), order-only replacement or cross-span token migration
abstains. A verifier deletion can describe only a separate verifier error;
absence never supplies a primary correction vote.

Bound each input to 8192 characters before tokenization and 128 tokens before
quadratic alignment; exceedance abstains, never crops. Streaming and primary
must each contain at least three tokens, so one/two-token fragments cannot
use this partial-phrase expansion path. Longer text can still be brief or
ambient; acoustic and conversation admission remain necessary, with no noise
robustness claim. All sources must be decoded and
nonempty, and all three original full renderings must have no controller-owned
control class. Invalid/unsupported inputs silently preserve the established
result without exposing text or making a healthy verifier fail.

Preserve STOP, confirm/deny and mode-switch fences, empty-streaming recovery,
minimum-duration checks, the historical exact NeMo/Faster-Whisper repair pair,
streaming partial/control ownership, capture revision/cancellation, original
PCM and CPU-only two-thread model execution. No new STOP repair or default
profile change follows. A changed final keeps `verifier_changed=True`,
`origin=unknown`, unknown owner verification and no direct-instruction/action
authority. The new source/outcome label is aggregate-safe and distinct from
exact consensus in production diagnostic/evaluation vocabularies.

Audio admission is independent of instruction trust. `_on_final_result` now
requests the existing conversation-cue gate even when ASR correction downgraded
its origin. A typed audio room fragment cannot cancel a reply or claim a new
input generation by becoming `unknown`. The legacy console string callback
retains its semantics; dictation/meeting and bounded heard-question followups
still use their existing admission rules. Echo, speaker and self-TTS rejection
remain in place. No new public trust flag or stronger origin is introduced.

## Verification and limits

The initial six-file model-free helper/selection/trust/admission/cancellation
gate passed 477 tests in 2.31 seconds before the final short-fragment regression
and additional command-trust cases. Those checks cover exact per-edit positive
support, independent disjoint verifier errors, same-family/unavailable voters,
whole-quorum precedence, controls, ambiguity, deletions, migration, Unicode,
symbols, numeric mismatches, bounds and silent failure. They also cover typed
corrected dialogue losing action trust and unknown-origin room audio retaining
conversation admission. This is not final gate or recorded-quality evidence.

Paired before/after composed CPU replay on Mic6 and Owner37 is required with
unchanged assets/configuration/corpora, both lexical/normalized errors, selected
STOP/false-control/drop counts, computation latency and resource use. No heldout
quality, actual speech-end/agent/TTS/DAC latency, microphone, live barge-in,
phone or other operating-system validation follows from these software tests.
The existing missed STOP remains a separate safety/quality problem; this
fallback cannot change its control meaning to improve a benchmark.

The final focused model-free gate passed **856 tests,1 skipped in5.42s** across
helper, exact consensus, production final selection, trust lineage, conversation
admission, cancellation, recorded/diagnostic/production/guided schemas and
profile tests. An initial expanded run was853passed/1skipped/3failed: the source
vocabulary fixture still supplied only five labels after a sixth was added,
and two new recording-mode tests wrongly expected no downstream addressing
call. Updating those test expectations preserved runtime behavior; the same
complete focused invocation then passed. Scoped Ruff E9/F63/F7/F82, AST parsing
and whitespace checks pass. The native benchmark owner serialized these checks
after its before-run workers were reaped. Parent owns independent final review,
the full logic suite, paired recorded verdict and any eventual landing.
