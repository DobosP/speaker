# ADR-0196: Add a receipt-bound NOTSOFAR-1 natural-overlap evaluator

Date: 2026-08-13
Status: accepted
Refines: ADR-0098, ADR-0109, ADR-0159, ADR-0162, ADR-0194
Supersedes: none

## Decision

Keep public matrix v5, the isolated 18-case NOTSOFAR-1 fixture and results,
the exact retained ADR-0194 bundle, its creation-time
`overlap_metric=not-implemented` fields, and every recognizer/runtime/default
unchanged. Add one separate receipt-bound aggregate evaluator:

- module `tools.notsofar1_natural_overlap_eval`;
- lock
  `tools/streaming_stt/notsofar1-natural-overlap-evaluator-v1.lock.json`; and
- report kind `notsofar1-natural-overlap-stt-eval-v1`.

The self-digested evaluator lock admits only ADR-0194's settled production
manifest SHA-256
`898e291c713154e3cbd78ba9c38fca6fed08284edace98ac615ebdd37265da7b`
and terminal-receipt SHA-256
`29b5d0f1f11f2ac56dee92a7d5ab6ef02a18c4d695a0b9b981b67dde3a24e2ba`.
It fixes one worker, one repeat, and exactly 14 sequential requests in
window-index order, with anonymous `channel-a` before `channel-b` in every one
of seven windows. A request must consume its complete source and produce one
typed final accepted by the selected worker adapter; an early or malformed
terminal is not scored.

Score each channel's single hypothesis against both private window references
with the existing `two-utterance-min-order-wer-v1` metric. That metric compares
the two whole-reference concatenation orders, retains the lower integer edit
cost, and selects the first order on a tie. `aggregate_accuracy` pools
substitutions, insertions, deletions, reference words, hypothesis words, and
errors before deriving the custom WER. `channel_accuracy` publishes the same
integer accounting separately for anonymous channels A and B. `paired_device`
publishes only bounded aggregates:
comparable windows, exact normalized-hypothesis agreement windows, selected-
order agreement windows, absolute word-error difference, and absolute
hypothesis-word-count difference. Do not publish repeat stability: the fixed
contract has one repeat.

This custom minimum-order result is not ordinary WER, ORC-WER, tcpWER, tcORC,
diarization, speaker attribution, endpoint truth, or conversational latency.
It is a development-only diagnostic for how one single-output recognizer
represents two concurrent utterances on two synchronized original channels.
The report carries no baseline/candidate roles, winner, delta, threshold,
qualification, promotion, or runtime/default authority.

Bind the report to the exact fixture manifest, terminal receipt, source
contract, raw evaluator lock and lock recipe, selected stream configuration,
worker manifest, runtime and model receipts, staged worker-source closure,
generic evaluator binding, and this wrapper's clean-import execution closure.
Reopen and compare the bundle, evaluator lock, wrapper closure, worker
manifest/receipts/binding, and source bundle before execution, after worker
shutdown, and immediately before return or publication. Retain the existing
single host lock, one-worker supervisor, private scratch teardown, lifecycle
handling, and canonical no-clobber mode-0600/single-link publication seams.

The closed aggregate report must contain no references or fragments,
hypotheses, per-window/case rows, speaker roles, raw device identifiers, local
paths, or winner language. Its only channel names are the evaluator-owned
anonymous A/B labels. Report validation recomputes integer identities,
denominators, derived WERs, channel-to-aggregate equality, paired bounds,
configuration/report bindings, and aggregate privacy. The public command emits
only a detail-free failure object or a successful report digest and evaluation
count; it never emits model text.

## Context / why

ADR-0194 intentionally stopped after strictly materializing seven natural-
overlap windows because the evaluator could not be bound until the production
manifest and terminal-receipt digests settled. Its custom two-reference shape
cannot enter the ordinary single-reference schema-v2 evaluator, and rewriting
the retained manifest to add a metric would invalidate already audited private
evidence. An independent evaluator lock is therefore the narrow follow-on
boundary.

ADR-0159's 18-case NOTSOFAR-1 result contains only isolated speech. ADR-0162's
PriMock57 overlap diagnostic has two clean stems plus one synthetic mix for
each case. The ADR-0194 bundle instead has two synchronized original far-field
channels observing the same natural two-speaker overlaps. Keeping both
channels anonymous and scoring each against both utterances preserves that
useful acoustic comparison without inventing device identity, speaker
attribution, or an ordinary-WER denominator.

Three repeats would triple model work while adding only repeat-stability
evidence, which is not needed for this bounded first diagnostic. One fixed
repeat minimizes resource use and still exercises all 14 immutable inputs.
Any later repeat-stability, standard speaker-attributed overlap metric, or
comparative candidate design requires a new lock and decision.

## Consequences

- An exact receipt-bound streaming worker can now evaluate every retained
  natural-overlap channel without exposing private references or hypotheses.
- A successful report describes only this seven-window after-PCM slice. It
  cannot select a model, change a default, establish endpoint/latency/device
  behavior, or replace owner bare-speaker and natural-conversation validation.
- The ADR-0194 bundle remains byte-valid and continues to record
  `overlap_metric=not-implemented` as its creation-time fact; ADR-0196's
  independent lock supplies the later evaluator method.
- No download, cloud/provider call, audio-device action, or paid service is in
  the evaluator. Real model execution is a separate explicitly authorized,
  resource-capped action.

## Verification

The final low-priority fake/static gate passes 111 focused tests in 2.03
seconds (2.25 seconds wall; 62,868 KiB maximum RSS). The adjacent seven-file
evaluator, preparer, shared metric, PriMock overlap, supervisor,
runtime-receipt, and APM/DTD gate passes 440 tests in 9.79 seconds (10.10
seconds wall; 137,960 KiB maximum RSS) with one thread per numeric runtime,
`nice -n 19`, `ionice -c 3`, and fresh explicit outside-Git basetemps. Fake and
static security review is GO. No test invoked a network, model, GPU, audio
device, microphone, cloud/provider, or paid-service path.

The already retained owner-private bundle passes the new exact evaluator
preflight: seven windows, 14 PCM inputs, manifest
`898e291c713154e3cbd78ba9c38fca6fed08284edace98ac615ebdd37265da7b`,
receipt `29b5d0f1f11f2ac56dee92a7d5ab6ef02a18c4d695a0b9b981b67dde3a24e2ba`,
source contract
`c1bb8c6c10e2abff7bb4f7a57948dbb34a12c2d36384b66b304d26f1103b57cc`,
evaluator-lock recipe
`e65f70c42e2568f914ffb4cc4e29c37fcba6fdf9ff291c89aebb0471ffa66ec1`,
and wrapper closure
`9215d56c73c1be349fb10b08dad8f097142d744f08b9b4e2df0aa6fe9c221fb3`.
Preflight took 0.37 seconds wall (0.34 user, 0.03 system) with 49,604 KiB
maximum RSS.

On 2026-08-14, the separately authorized retained Faster-Whisper Small
CUDA/FP16 diagnostic completed all 14 source-complete burst evaluations across
seven windows, two anonymous synchronized channels, and one repeat. The custom
`two-utterance-min-order-wer-v1` micro-aggregate is .4230769231: 77 errors over
182 reference-word opportunities, comprising 70 deletions, 7 substitutions,
and zero insertions, with 112 hypothesis words. Anonymous channel A/B are
.4505494505/.3956043956. All seven paired windows were comparable and both
channels reported the same selected reference order in 7/7, while normalized
hypotheses agreed exactly in 1/7; summed absolute per-window word-error and
hypothesis-word-count differences are 9 and 8.

Preserve the canonical mode-0600/single-link 7,946-byte report SHA-256
`18ec9f583190752bd683f07341e8205d9b511e6823081b5173405a8fe0df45bd`.
Its report binding is
`16e46fc5bf9824fecb2ae2e9c020ee86fbb99a8df9f2a8d6a81b2da406b17125`;
worker manifest/runtime-receipt/model-receipt SHA-256 values are respectively
`ed323aca7868b9279f5a8fd66bb0fb8c8b76a882385a0803ffa9e61570adcbca`,
`85dfc56712f9db4997836329c4fdb2b1e56a265ac9a54a4d5859f74e2ea309a2`,
and `28d63ead1b37e6241dbca196155dfc988ea61220e7484ad64ca60b15e9f7c132`.
The independent read-only closure rederived the canonical report/config
bindings, raw/recipe lock digests, wrapper closure, current generic evaluator
file set, worker-source bundle `8407a284…47f48`, manifest-derived worker
binding, evaluator binding, metric arithmetic, coverage schema, and forbidden-
content shape without rerunning the model or opening private PCM/reference
text. The retained manifest/receipt files were hash-checked only and still
match ADR-0194's exact pins.

The 14/14 source-complete count is execution/source accounting, not corpus or
turn coverage. This deletion-dominated custom result describes only the fixed
single-output after-PCM slice; it does not establish ordinary WER, ORC-WER,
tcpWER, tcORC, diarization, utterance order, endpoint behavior, latency, device
identity, held-out quality, qualification, promotion, runtime/default, or live
behavior. The report contains no cgroup/resource measurement observations, so the
collected launch unit and zero-compute-PID check remain separate operational
cleanup evidence rather than report-attested resource evidence.

Frozen module/test/lock evidence is respectively:

- `tools/notsofar1_natural_overlap_eval.py`: 82,129 bytes, SHA-256
  `0534e19e5a6db936196b539a8010fd4320cb3e7edaca6385dacb47f5ad119578`;
- `tests/test_notsofar1_natural_overlap_eval.py`: 59,644 bytes, SHA-256
  `d04ba7430dc2f31b80f3bfd8badc1e699c9f1068dfc0cfa30576cee8cb916a56`;
  and
- evaluator lock: 1,620 bytes, SHA-256
  `1e036ba048eedcdbfb2196083c4268c478c34af560cff8e0a10a54ec15076c62`.

No quality verdict, promotion, runtime/default change, microphone/audio-device
action, or live validation occurred.
