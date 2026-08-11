# ADR-0185: Bound the KWS speaker-inference lifecycle

Date: 2026-08-11
Status: accepted

## Decision

Carry forward every phrase, label, timestamp, capture/source/route/playback,
speaker-generation, per-word acceptance, final-claim, and typed STOP fence from
ADR-0183, plus every ledger-object, phrase-copy, geometry-validation, and
callback-time Python-reference bound from ADR-0184. Change the execution
lifecycle for speaker scoring of an exact own-TTS-ambiguous KWS hit and add the
necessary process-wide nonblocking admission fence to every live runtime
speaker-inference path; change no scoring, threshold, or authority policy.

Permit exactly one unfinished KWS speaker-inference task process-wide, counting
both may-start and running state. Acquire the exact process-wide runtime permit
before publishing or starting that task. Admit no waiting successor,
replacement, or retry while it may still own the permit, worker, or native
call. The worker payload contains only the exact gate, one or two already-owned
word clips, their sample rate, and opaque lifecycle-ticket metadata. It contains
no engine, callback, expected speaker authority, KWS control authority, native
KWS stream, submitted-feed array, or ledger. The capture owner retains all
expected control/speaker authority and callback state outside the task.

Before each of at most two synchronous native embedding calls, one per word,
the worker must enter a lifecycle-Condition-linearized `enter_native_step` for
the exact ticket. That admission is the timeout/stop versus native-entry
ordering point: abandonment that wins first forbids the call, while a step that
wins admission may finish after a later timeout or stop but cannot make its
result deliverable. Recheck the same admission between the first and second
word. An Event observation alone is not sufficient because it leaves a
check-to-call race.

The worker may publish only a raw immutable `SpeakerSimilarityBatchReceipt`, an
explicit error outcome, or a busy outcome. It must clear its gate and owned-clip
references before publishing that outcome and never performs threshold/status
reduction. The waiting capture owner applies the unchanged per-word threshold
policy, checks the exact deadline, revalidates the complete ADR-0183 control and
speaker authority, installs the same playback KWS claim, and invokes the
already-captured typed STOP callback through the existing capture-effect
barrier. The worker never invokes an external callback or claims playback
authority.

Pin `_KWS_SPEAKER_INFERENCE_TIMEOUT_SEC` to `0.050`: one code-owned 50 ms
total action deadline begins before task claim and the same absolute deadline is
rechecked at pre-start/native-step admission, result capture/reduction and
playback claim, and final callback admission. Expiry fails closed and latches the
current capture epoch as described below. This is a headless safety/liveness
bound, not speaker-model latency calibration or live timing evidence.

Linearize task admission, native-step admission, result publication, timeout,
and shutdown against the exact ticket under one lifecycle Condition. If timeout
or engine stop wins before capture claims the result, mark that task permanently
abandoned, wake its waiter, and discard every later result. Do not cancel,
replace, detach, or retry an admitted native call. Stop does not join that
worker. A direct rebuild and a later public start must refuse before mutating
shared engine/model state while the process-wide task may start or its worker/
native call remains. The worker clears its gate/clip locals before reporting
return. Only the reaper may release the process permit and registry slot, after
it observes worker return, proves the thread dead, performs a zero-time join,
and clears any remaining owner-held payload/result references. A later rebuild/
start may then proceed. An exception from worker start with ambiguous liveness
retains the may-start task and permit until the lifecycle can prove absence or
reap the worker; it never authorizes a replacement. A current-epoch timeout,
worker/error outcome, or malformed/nonfinite receipt latches KWS speaker
inference unavailable for the rest of that capture epoch; stale, reject, defer,
and busy evidence abstains without setting that latch.

Use the same process-wide nonblocking runtime permit for KWS tasks, exact final
speaker verification, speaker warm-up, and legacy-WAV runtime enrollment. Busy
final verification yields `UNKNOWN` and cannot mint `VERIFIED`; busy warm-up
remains cold; busy legacy-WAV enrollment first revokes any old enrollment and
remains unavailable/unenrolled. Before any of these live try seams, an engine
may opportunistically reap a proven-returned exact process owner, including one
created by another engine, but never waits for or reaps a live worker. Keep
word-cut speaker scoring synchronous at its existing nonblocking
`try_similarity` path. A legacy/custom live final or word-cut gate without the
corresponding process-permit try seam abstains instead of using a blocking
fallback: final remains `UNKNOWN` and word-cut remains unavailable. Stable idle
KWS, playback-novel labels, and non-ambiguous exact playback STOP/WAIT hits
create no task or worker and retain their existing callback/effect eligibility.
Add no public configuration, threshold, model, label, callback, effect, route,
or default.

This ADR supersedes only ADR-0183's synchronous capture-owner inference and
capture-processor-shutdown clauses, and ADR-0184's instruction not to move KWS
speaker inference. It carries every other ADR-0183 and ADR-0184 decision
forward, including the rule that an entered native call is not Python-
preemptible.

## Context / why

ADR-0183 bounded the mailbox, retained PCM, and native call count, but its one
or two embeddings ran synchronously on the capture processor. One native call
that did not return could therefore outlive the bounded mailbox and delay that
processor's shutdown. ADR-0184 reduced Python object/copy/callback retention but
explicitly left this liveness question open.

Python 3.12 documents that [`Future.cancel()` cannot cancel a call that is
already running](https://docs.python.org/3.12/library/concurrent.futures.html).
The sherpa-onnx public [speaker-extractor C API exposes create, readiness,
compute, and destroy](https://k2-fsa.github.io/sherpa/onnx/c-api/html/c-api_8h_source.html)
but no per-compute cancellation/RunOptions seam. ONNX Runtime itself exposes
[`RunOptions.terminate`](https://onnxruntime.ai/docs/api/python/api_summary.html),
but that is not exposed by the pinned sherpa speaker wrapper. Therefore this
decision treats an admitted native call as non-preemptible. Replacing a timed-
out worker or constructing a new executor on each restart would turn one stuck
call into unbounded threads, work items, model contention, and PCM retention. A
single non-replaceable task makes the failure mode conservative and finite:
later ambiguous hits abstain while it is busy, and restart refuses rather than
creating another native owner.

A read-only static inspection of the installed CPython 3.12/Linux x86_64
sherpa-onnx 1.13.3 wheel found its `SpeakerEmbeddingExtractor.compute` pybind
branch bracketing the C++ compute call with `PyEval_SaveThread` and
`PyEval_RestoreThread`. This is exact-wheel/ABI evidence that this wrapper branch
releases the GIL; no extractor, model, compute, device, or live path ran. It does
not prove native return, cancellation, internal thread/storage bounds, buffer
release, or behavior of another build.

Returning only the raw immutable speaker receipt to the capture owner preserves
ADR-0183's authority linearization and ADR-0184's callback-time reference
boundary. Letting the worker reduce policy or callback directly was rejected
because it would need to retain and reconstruct expected speaker,
capture-effect, callback-capture, acoustic-terminal, and playback-claim
authority after the original capture stack had left.

## Consequences

- A normally completing ambiguous hit keeps its existing per-word result and
  typed STOP behavior, with one bounded handoff/wait around the native calls.
  Busy, timeout, error, stale, short, rejected, or otherwise uncertain evidence
  still abstains.
- At most one Python KWS speaker worker and one unfinished KWS speaker task may
  exist across the process. Timeout and stop release the capture waiter but may
  retain that worker, its old gate, and bounded owned PCM until native code
  returns. No late result may callback.
- While that exact process permit remains held, other live final, warm-up,
  legacy-WAV enrollment, and word-cut inference attempts do not wait or start a
  second native operation. They retain their conservative busy outcomes; a
  proven-returned local or cross-engine owner may be reaped first without wait.
- A retained worker deliberately blocks engine rebuild/restart. If native code
  never returns, process replacement remains the recovery boundary; this ADR
  does not claim native cancellation.
- This does not bound offline enrollment tooling or native/model internals.
  Word-cut remains at its existing synchronous nonblocking-acquisition seam;
  native code may create its own threads or retain storage.
- Deterministic headless fakes can prove request/thread bounds, atomic result-
  versus-abandon ordering, stop wakeup, stale-result rejection, restart fences,
  no late callback, and Python PCM release after a fake native return. They
  cannot prove GIL behavior. The separate exact-wheel static evidence above
  establishes only the installed compute wrapper's GIL release; neither source
  proves native return, buffer release, latency, model correctness, or behavior
  with a microphone/open speaker.
- ADR-0183's paired owner bare-speaker A/B and ADR-0184's hostile finite-
  configuration byte-cap follow-up remain pending.

## Verification

- Deterministic headless coverage passed `14` isolated owner tests in 0.19 s,
  `167` isolated virtual-engine tests in 1.55 s, and the exact documented
  six-file lifecycle command with `460` tests in 4.60 s. These cover process-
  wide one-task admission, permit-before-start, start/reap failure retention,
  no replacement, Condition-linearized per-word entry and first-terminal
  ordering, stop wake/no-join, late-result and callback fencing, capture-owned
  reduction/authority, current-epoch latch classification, callback-thread
  preservation, local/cross-engine reap, mutation refusal/recovery, payload
  exclusion, exact Python-reference release, shared live try seams, and zero-
  task idle/novel/non-ambiguous KWS.
- The carried ten-file authority/resource command passed `643` tests in 5.91 s;
  the adjacent owner/activation command passed `290` tests with two pytest
  warnings in 3.46 s, plus the existing exit-line `swigvarlink`
  `DeprecationWarning`; the exact complete-Sherpa command passed `538` tests in
  5.34 s; and APM/DTD passed `6` tests in 1.08 s.
- Final SHA-256 is
  `d3a4015b306cea183ae59cbe4345d2f4be8e06545c399f7860fa6c1ec4fbfffc`
  for the owner,
  `ff168dfa636dbeb4f4ed283c76d3aff990d9168f1f1b9faadf1b6bee80d88b45`
  for `speaker_gate.py`, and
  `9669c92d458e4defb93de1bf5eb3349c454110e33ee431a65fcdbdb9e3f09dfe`
  for `sherpa.py`. Scoped full lint passed for eight non-Sherpa files plus
  Sherpa F821/F822/F823; whole-nine-file lint retained exactly Sherpa's 23
  inherited findings. Nine-file AST parse, Ruff AST equivalence, diff check,
  exact 100-line STATUS, and changed-line formatter overlap are green; remaining
  formatter debt matches `main` with zero task-line overlap. Independent
  adversarial review at the frozen hashes is GO.
- No model, microphone, audio device, GPU, network, latency, quality, or live
  path is part of this headless lifecycle proof. The paired owner bare-speaker
  A/B remains separate and pending.
