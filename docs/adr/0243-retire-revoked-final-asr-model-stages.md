# ADR-0243: Retire revoked final-ASR model stages

Date: 2026-10-10
Status: accepted

## Decision

Pass the live engine's existing exact-work currentness predicate into final-text
resolution. Check it before/after punctuation and each offline-recognizer or
verifier boundary. Use a private cancellation unwind, separate from model errors,
that follows the existing stale-final diagnostic return. Once revocation is
observed, admit no later model work, fail-open selection or verifier-health change.
Keep all healthy selection, native model/resource ownership, agreement/consensus,
endpoint, floor/speaker, raw-audio and direct-live/action contracts unchanged.

## Context / why

ADR-0107 already bounds and exactly retires async final work, but the final
selector previously checked ownership only before and after its complete model
pipeline. A deterministic production-finalizer probe on baseline 1bed9e2 revoked
the current media item during punctuation: offline create/accept/decode plus the
independent verifier still ran. Final callbacks were correctly fenced, yet obsolete
work occupied the sole final worker and could delay a successor or shutdown. An
obsolete verifier error could also reach the session-health circuit breaker before
the later final-currentness check.

Use `_capture_callback_is_current`, which already reads the exact thread-local
media cancellation event, capture epoch, decode identity and capture-effect lease.
Do not invent elapsed-time heuristics or a new concurrent inference owner. A
private `_FinalAsrCancelled` unwinds through punctuation/offline/verifier error
handlers; it cannot masquerade as unavailable/failed recognition or trigger a
fallback. The finalizer catches it as stale work. Currentness is also rechecked
before an error decision reaches the verifier-health mutation.

## Consequences

This is cooperative stage admission. It does not interrupt or destroy an entered
native call. Cancellation can race immediately after a successful checkpoint;
that entered operation must return before its worker/resources settle. Exact
stage finish and resource release remain on the original final worker, with no
extra thread, queue or retry. Unknown/blocked native cleanup retains its existing
fence. An opaque custom `_final_transcribe` override remains responsible for its
own internal work; existing surrounding callback fences still apply.

The optional helper predicate defaults to absent. FileReplay and standalone
helper callers retain their established selection path and outcome vocabulary;
recorded/default text quality and provenance are unchanged. No default model,
acoustic threshold, sample rate, ASR window, endpoint timing, speaker/control
policy, raw egress boundary or native ABI changed. Healthy live work uses the
same PCM object in the offline and verifier models as before.

The [compact synthetic receipt](../evidence/asr-final-cancellation-2026-10-10.json)
binds pinned project helper source and current helper hashes. It verifies 48
healthy exact decision cases against baseline Git source. At punctuation/create/
accept/decode revocation, later fake model calls fall 4/3/2/1 to zero. With an
explicit fake verifier allocating 1MiB, cancellation after offline decode lowers
traced peak 1050248 to 2096 bytes and paired p50 242.811 to 21.153us. These numbers
measure that fake workload only; they do not estimate native ASR, RSS, physical
voice latency or whole-agent speed. Healthy no-cost fake resolver p50 was 67.376
versus 70.944us; its lambda/Event predicate is not full-engine timing. Four BLAS/
OpenMP thread environment requests were one; CPU isolation was not established.

New tests exercise revocation during every model boundary, normal and error return,
no fallback/health mutation, exact stage retirement during blocked calls, no early
resource release or successor entry, same-worker stream release, sole-worker
successor progress, foreign-scope isolation and healthy exact decisions. Broader
ASR, trust/provenance, capture/streaming, diagnostics, barge and replay regression:
**914 passed, 1 optional-model skip** in 22.76s. The optional native-model test
skipped because worktree assets were absent. No native inference, model download,
microphone/device, doctor, private recording or live test ran. Acoustic, Windows/
macOS hardware and native blocked-call timing remain owner-authorized gates.

Reproduce the small project-only benchmark with one fresh output path:

```sh
SPEAKER_TEST_LOG=0 SPEAKER_NO_LOCAL_CONFIG=1 SPEAKER_LIVE=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ionice -c 3 nice -n 19 /home/dobo/work/speaker/.venv/bin/python -B -m tools.bench_asr_final_cancellation --output <fresh-json>
```
