# ADR-0256: Corroborate Qwen dialogue edits without changing control authority

Date: 2026-10-29
Status: accepted for software and development recording qualification; STOP/live qualification remains open

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

## Full-gate race-fixture repair

The parent-owned first full CI-style run on7a9c58d completed **12737 passed,
40 skipped,9 warnings and1 failure in207.60s**. The only failure was the known
`test_bound_file_hash_rejects_in_place_mutation_during_streaming_read` race:
it did not raise the expected production error. Its precise cause was not
captured. Source review found two fixture hazards: its real same-size2MiB
rewrite could share the initial filesystem timestamp tick, and its process-wide
`os.read` hook could fire on an unrelated background read before the target's
own snapshot began. All selection cases in that run passed; the full run
remains failed history.

Repair that test only by scoping the hook to the captured target device/inode
before the actual read. Unrelated reads retain their original results/errors
and cannot increment the target counter or cause its mutation; an explicit
unrelated-FD probe verifies the target remains unchanged. Retain the real byte
rewrite after the first target streaming read, preserve its recorded atime and
explicitly set mtime one second beyond its own initial mtime. Assert the
timestamp actually differs and the rewritten bytes are present. The production
bounded reader, metadata identity checks, expected `BoundedReadError`, deadlines
and resource/safety policy are unchanged. This removes two scheduling/clock
assumptions without sleeping, excluding the case or retrying unchanged code
until success. The final exact isolated and related manifest receipts follow.
Parent will run the full gate on the clean successor; benchmarked7a9c58d runtime
bytes remain unchanged by this fixture repair.

Final fixture verification: the exact isolated case passed1 in0.15s; complete
streaming and extra-Kyutai manifest files passed114 in1.42s. Scoped Ruff,
whitespace and the38-file documentation check pass. Independent static review
accepted both target-descriptor scoping and the explicit owned mtime change.

## Final paired qualification and landing evidence

The declared four-cell selector comparison passed: actual FileReplay on BEFORE
`fa59d9d03c9e6ef9be9a29ef9b1fff605403d876` and AFTER
`7a9c58d10585ec438c9b78c0345bdbec0c793e0f`, benchmark
`899201c3b01020808c86dd7239fedfdecfffa938`, two separate repeats per corpus.
Same profile/config/assets/scorer/original PCM and CPU mask[0,2]/thread2 bind
both phases. All raw score fields/control counts and endpoint/empty accounting
match; three corroborated primary decisions replace three ties per repeat.
Owner37 selected legacy22/212→18/212, normalized21/213→17/213 and exact26/37→29/37;
Mic6 remains legacy3/25,normalized4/25. No label tuning or further voter trial
followed. Retain the optional CPU profile and unchanged defaults.

STOP is not fixed: Mic6 selected0/1 and Owner37 1/2 per repeat, one guarded miss
per corpus. Benign false controls remain0; no confirm/deny/mode positive coverage
or noise-negative acoustic proof follows. The selector gain cannot authorize
live/default/control promotion. Typed runtime/engine tests independently prove
admission and action-authority fences; FileReplay scores alone do not.

Owner complete-call p50/p95 compute2.607/4.280→2.473/3.213s, peakworkerRSS
1.850→1.850decimalGB, RTF1.008→0.946. Timings are nonexclusive-host after-PCM
compute, not causal acceleration, real speech-end response or STOP delivery.
Both original endpoint leases closed in every cell, children exited0/reaped,
network isolation/no GPU/audio-device exposure/resource bounds passed. The
[compact packet](../evidence/qwen-selection-comparison-2026-10-29.json) is exactly
14,043bytes, SHA `a82ed9672d855337da90d74b0b3cad4e10628fbb1c07185c0185feb5135e9c76`.
It preserves172clip calls across four cells without raw voice/reference/hypothesis
text. Full raw score equality is aggregate parity, not transcript byte identity.
The research branch retains measured899201c and docs-onlyb92f496 reproducibility
sources; original recordings and baseline model assets remain unchanged.

After the test-only fixture repair, the parent-owned full CI-style gate on
`54104f68014feb1619466bbd7cbe9f18582d465c` passed **12738 tests,40skipped,
9warnings in218.52s**, exit0. Command: low-priority single-thread environment,
private tmpfs/tmp+pseudo/dev+writable task-tree `bwrap`, canonical `.venv`
`python -B -m pytest tests -q --tb=short -p no:cacheprovider
--ignore=tests/test_livekit_audio.py --ignore=tests/test_livekit_engine.py
--basetemp=/tmp/speaker-qwen-selection-fullgate`.
Two LiveKit ignores are the documented CI exclusions; no failing case was
excluded. Measured7a9 runtime/core/tools/config bytes equal tested54104; final
successor documentation/evidence changes no runtime or tests. Earlier failed
full gate and exact successful receipts remain in private `logs/runs/qwen-selection-20261029/`.
Private synthetic full-suite fixture trees vanished with their namespaces;
original logs/audio/model keepers and unmerged research work were not pruned.
No microphone, doctor, live speaker, phone or other operating-system validation
ran. Linuxx86_64 remains the qualified Qwen binary tuple.
