# ADR-0234: Isolate desktop output failure and admit explicit recovery

Date: 2026-10-10
Status: accepted

## Decision

Give ordinary Sherpa desktop output its own fixed lifecycle and exact native
cleanup owner. A playback-worker exception fences output, revokes its speech
generation, hard-drops queued PCM and fails active/queued/new tracked fragments
exactly once. Keep capture, finalization and receipt dispatch alive. Preserve the
trusted virtual-route proof-loss behavior as fatal to its complete route.

Expose `OutputState` and additive `EngineCallbacks.on_output_state(code)` with
fixed codes only; publish fixed log messages as well. Supply an explicit,
nonblocking `engine.recover_output()` API. Never replay a failed fragment or
implicitly open a new player in response to each rejected sentence.

| State | Meaning |
|---|---|
| `ready` | Ordinary output admission is available. |
| `unavailable` | Output is fenced while native cleanup/capture reset acknowledgement is pending. |
| `recoverable` | Native cleanup and capture reset acknowledgement succeeded; recovery still verifies old worker exit and idle model ownership. |
| `poisoned` | Native open/close or worker launch is uncertain; explicit recovery refuses. |
| `stopped` | The whole engine/session was deliberately stopped. |

Route ordinary failure, idle release, full stop and retained-resource output
teardown through one `OutputCleanupOwner` for the exact handle. Each caller waits
at most one second for the cleanup task. Native stop and close each run at most
once; close is still attempted after a stop error. Timeout, failure, interrupted
wait or ambiguous task launch latches uncertainty and retains the task/handle.
A late native return cannot upgrade a timed-out owner to clean. No callback or
native driver call runs under the output-handle lock.

Recovery requires a healthy capture session, no capture-resource hold, exact old
playback worker termination, successful exact native cleanup, and a nonblocking
acquisition of the existing TTS model lock. It admits one new worker while the
receipt/admission lock excludes concurrent callers. A thrown/ambiguous launch
poisons output and retains its exact thread. Existing start guards are preserved,
and a new start also refuses unsettled output cleanup or an inconclusive native
open. Native import, BaseException and unexpected early worker exit all fence
output and fail ownership instead of leaving a dead worker advertised as ready.
General native restart/preemption is not introduced.

## Context / why

Previously a TTS, DSP or output exception cleared the shared `_running` event,
which also stopped the capture/ASR loop. The assistant became unresponsive even
when microphone input was healthy. Reopening output immediately would risk
native stream overlap, discarded speech replay or own-TTS transcription. Merely
removing `_running.clear()` would also leave new requests in an unconsumed queue.
The existing teardown call sites swallowed native errors and could forget whether
an exact stream had actually closed, so their ownership must be shared before
recovery can be truthful.

## Consequences

New tracked requests receive immediate failed receipts while output is fenced;
legacy completion callbacks still fire for discarded failure-queue work. Failed
receipts carry no safe spoken prefix. Prior completed receipts remain completed.
Speaker/markup/DSP/model choices, input authority, controls and ordinary healthy
barge-in behavior remain unchanged.

If native silence is uncertain, retain the existing speaking/echo quarantine.
Capture and control processing remain alive, but ordinary ASR is withheld by the
existing playback branch until exact cleanup proves silence. STOP still revokes
the generation and clears queued PCM; it cannot claim that an uncertain native
player became silent. Mark quarantine before detaching an ordinary handle, and
retain the exact cleanup-owner identity after the shared pointer is detached.
After native cleanup succeeds, queue an epoch-bound reset request. The capture
owner consumes it at a safe frame boundary after the prior frame/effect retires,
resets AEC/detectors, acknowledges it and only then clears playback state/reference
and resumes ordinary ASR. No output worker resets mutable capture DSP or waits
for its completion. Direct reset is allowed only when existing ownership checks
prove every capture owner/effect idle (pure playback/stopped-engine paths).
A failed reset poisons output and retains quarantine. A constructor that never
returned a handle poisons output;
no PCM was admitted there, so it need not retain a fictitious audible quarantine.

An embedding/operator can inspect and explicitly attempt recovery:

```python
print(runtime.engine.output_state.value)
recovered = runtime.engine.recover_output()  # immediate True/False; never waits for native cleanup
```

`on_output_state` reports advisory fixed transition codes without exception text
or transcripts. Concurrent transitions can supersede a delayed callback, so an
observer re-reads `engine.output_state` before display or recovery; a callback is
never native-cleanup authority. Fixed engine logs expose failure even without an
observer. `core/runtime.py`
is unchanged here; wiring a richer runtime observer or a shell recovery command
is separate integration. `recoverable` can be published while the old worker is
finishing its epilogue; a concurrent recovery attempt safely returns False until
that exact worker is dead. Poisoned cleanup requires process restart. Replacing
an uncertain engine with another instance is not qualified by this decision.

The bound is per engine/session, not a process-global or cross-engine claim.
At most one closer owns an exact native stream. A wedged closer remains retained
instead of authorizing another one; already closed owner metadata is replaced
when a later stream is admitted. These are ownership/resource bounds, not an
RSS or native latency benchmark. A native TTS call that never returns is not
forcibly preempted or detected by a new synthesis deadline in this change.

Headless scoped gate: **940 passed, 2 optional-model skips** in 12.02 seconds.
It covers the pure closer, threaded output recovery, existing playback/media
session/fake duplex/model build/virtual route/capture integration, DSP, barge,
receipts/history, markup and leveler suites. New tests exercise failed/queued/new
receipts, once-only close under concurrent callers and stop/failure, timeout/late
return, interrupted wait, ambiguous cleanup/recovery launch, observer failure,
explicit recovery with busy admission/model locks, and fresh completed playback
afterward. SystemExit/KeyboardInterrupt, failed output import, unexpected sentinel
exit and idle-detach/blocked-stop regressions keep output fenced. A production
capture worker using injected silent input and fake ASR continues feeding ASR
after clean output failure. A held-process AEC barrier proves reset cannot race
capture DSP and occurs on the capture owner's thread before quarantine release. No microphone, physical device, native
model inference, private recording, network or Windows/macOS audio runtime ran.
Live tests remain owner-stopped. Driver focus/unplug/late callbacks and acoustic
silence/tail behavior still need owner-authorized cross-platform validation.

Exact scoped command (models/devices are faked or self-skip):

```sh
SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 19 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_output_cleanup_owner.py tests/test_sherpa_output_recovery.py tests/test_sherpa_playback.py tests/test_sherpa_media_session.py tests/test_sherpa_duplex_runtime.py tests/test_sherpa_models.py tests/test_virtual_audio_engine.py tests/test_capture_integration.py tests/test_streaming_tts.py tests/test_tts_markup.py tests/test_tts_backend.py tests/test_audio_frontend.py tests/test_apm_double_talk.py tests/test_barge_word_cut.py tests/test_barge_confirm.py tests/test_playback_receipts.py tests/test_engine_playback_receipts.py tests/test_playback_history.py tests/test_speaker_input_gate.py tests/test_output_leveler.py -q
```

## Addendum — prospective evaluator source bindings (2026-10-10)

The unconditional Sherpa import of `core/engines/_output_cleanup.py` changed
the evaluator import closure. The full headless gate detected the missing AMI
source-list member; this was an inventory failure, with no new audio execution
or quality result. Generic capture replay, Microsoft AEC/APM/DTD, production
final-STT, EdAcc endpoint integrity and LiveKit causal endpoint now include the
helper in their prospective source hashes. AMI derives its exact list from
generic replay: its expected count is 59, and generic schema-4 async replay's
count is 33. Existing exact clean-process equality checks remain strict.

Historical locks, recipes, reports and native receipts are unchanged; the new
source identity qualifies only future evaluation runs. Media-only inventories
do not acquire unrelated conversation-admission, LLM-decision or voice-model
selection modules. Whole-agent conversation provenance already binds clean
Git revision plus configuration/model identity; the separately scoped native
factory benchmark receives its own prospective manifest.

Seven deterministic source-closure, source-membership, changed-helper digest
and async-report checks passed in 12.26 s. The exact commands and an earlier
scratch-parent setup failure are retained in `WORKLOG.md`. Scoped lint and
whitespace checks pass. No recording, native inference, microphone, doctor,
network or hardware validation ran; the owner-stopped live gates remain open.
