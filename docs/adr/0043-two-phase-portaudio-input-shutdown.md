# ADR-0043: Fence and join capture before closing PortAudio input

Date: 2026-07-11
Status: accepted

## Decision

Classify PortAudio/sounddevice `paUnanticipatedHostError` (`-9999`) as a
recoverable input condition and walk the existing bounded reopen chain.

Shut input down in two phases. First clear the engine running flag and mark the
recovering wrapper closed, which rejects new reads and prevents a late reopen
without touching the native stream. Give the capture owner one configured audio
block plus a fixed 50 ms scheduling margin to return and join; only then call
native `stop()`/`close()`. If the read remains stuck, invoke native `abort()`
through a bounded daemon helper and give the capture owner one final bounded
one-second join. Close only if that owner quiesces. If abort or the read remains
stuck, retain the wrapper/native handle rather than race `close()` against it.
Reject a later engine `start()` before clearing shutdown fences while any prior
capture, playback, final, or receipt worker is alive, or while any retained
input wrapper still owns lifecycle. Retain a receipt worker reference when its
bounded stop join times out so the next run cannot reset its shared event.

Invalidate a per-run capture epoch at the first stop instruction. Admit
recorder, KWS, barge, partial, and final effects through a tracked epoch lease;
recheck that epoch at their emission seams. Carry it through asynchronous
second-pass final work so a slow worker cannot callback after stop or restart.
If an admitted effect cannot join, retain its input, output, and recorder
resources. After playback/final worker joins, re-evaluate capture ownership and
effect leases; if they became idle during those joins, perform the deferred
input/output/recorder collection and clear the resource hold. A still-live
playback owner keeps the shared-resource hold too.

Run normal native input `stop()`/`close()` on a tracked helper with a one-second
engine budget. Atomically reserve abort/teardown ownership before selecting the
native handle. Detach only after both calls return successfully; timeout or
exception retains the handle for a later safe retry and continues to block
restart.

## Context / why

The installed current sounddevice binding reports `paUnanticipatedHostError`
as `-9999` and `paNoError` as `0`. The local wrapper instead assigned the host
error to `-9989`, and its negative regression incorrectly described `-9999` as
"no error". Live run `154451` actually ended on `-9999`, so the wrapper bypassed
recovery and let the capture loop die.

That run then recorded supervisor cancellation followed by allocator
corruption. Source inspection establishes that `SherpaOnnxEngine.stop()`
physically closed input before joining the capture thread, while the wrapper
had no active-read synchronization. A native close overlapping PortAudio's
blocking read is therefore a strong corruption hypothesis, not a proven cause:
there is no native backtrace or instrumented PortAudio trace tying the
allocator failure to that overlap.

Keeping immediate close as the normal path preserves that unsafe overlap.
Waiting without a deadline can hang shutdown on a failed driver. The phased
owner handoff removes overlap on the healthy path while retaining an explicit,
observable forced fallback for the only case that cannot quiesce normally.

## Consequences

- A normal stop may wait one input block plus 50 ms, but deterministic
  concurrency tests attest that physical close does not overlap the active
  read.
- Host error `-9999` now enters `RECOVERING`; the existing retry budget,
  fallback enumeration, state events, and first recovered block semantics stay
  unchanged.
- A stuck native read reaches the host API's explicit `abort()` after the grace
  period. Physical `close()` still never overlaps the read. If abort cannot
  quiesce it, shutdown returns bounded while retaining the daemon and handle;
  the exceptional leak is preferable to risking native heap corruption.
- A broken native `stop()` or `close()` adds at most the one-second teardown
  budget to engine stop. Timed-out, raised, or still-active native operations
  remain tracked and cannot overlap another abort/close or report success.
- Capture callbacks are epoch-fenced at their actual emission seams. A block
  admitted just before stop either finishes before teardown or keeps every
  shared audio/recording resource retained. A post-worker-join pass collects
  those resources as soon as their last admitted lease finishes.
- A fully stopped engine can be started again. A retained/stuck capture requires
  process replacement; no prior capture/playback/final/receipt worker can be
  resurrected into a duplicate owner by a new run's shared events.
- These are headless lifecycle guarantees. The ROG still needs a live rerun;
  this decision does not claim that the observed allocator corruption is fixed
  or that device-unplug recovery is hardware-validated.
