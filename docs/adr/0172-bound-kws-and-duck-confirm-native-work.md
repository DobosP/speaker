# ADR-0172: Bound KWS and duck-confirm native work

Date: 2026-08-10
Status: accepted

## Decision

Bound the native streaming work and retained PCM authorized by each existing
KWS feed and duck-confirm window. One accepted feed may call native
`decode_stream` at most 64 times. After exactly 64 decodes, make one final
`is_ready` probe: not-ready is a valid exact-budget drain, while still-ready is
exhaustion and cannot reach `get_result` or any callback.

Derive the per-feed sample cap from validated runtime configuration as
`F = ceil(sample_rate * max(block_sec, media_pcm_queue_ms / 1000))`. KWS accepts
only a non-empty feed of at most `F` samples. Invalid, empty, or oversized KWS
PCM is a controlled capture discontinuity; retire the affected capture state
and recreate the KWS stream before later PCM. A native KWS exception or decode
exhaustion on an otherwise valid feed also recreates the KWS stream and
abstains. Neither case publishes a command terminal.

At duck-confirm start, require a finite non-negative capture time and
snapshot finite, representable work limits from the existing configuration.
With `W = max(0.1, barge_confirm_window_sec)`, retain at most
`S = ceil(sample_rate * W) + F` samples independently in each primary and
alternate PCM domain, and invoke the confirm step at most
`I = ceil(W / block_sec) + 1` times. Each accepted step requires a finite time
strictly greater than the prior step and non-empty primary and alternate
domains no larger than `F`. Snapshot the clamped duck gain and non-negative
retry interval with the same window so later configuration mutation cannot
change an open window's bounds.

On invalid clock, invalid domain, prospective sample/invocation overflow,
native confirm failure, or decode exhaustion, close and discard the confirm
window before production rotates the whole decode continuity. This structural
abort is not an unconfirmed acoustic observation: it must not teach DTD, arm
retry suppression, emit an unconfirmed metric, claim a barge, publish a handoff,
or invoke another effect. Engine stop closes the same complete state.

This adds no model, keyword, threshold, default, microphone source, route,
speaker/STOP rule, callback authority, command mapping, device selection or
configuration, healthy-path recognition policy, or live claim. The cap bounds
the number of native calls after a feed; one native call or an already-entered
callback remains non-preemptible.

## Context / why

ADR-0171 kept raw KWS out of duck confirmation and made a consumed KWS terminal
close the window explicitly. Its follow-up audit found two separate unbounded
surfaces. Both KWS and confirm drained `while is_ready(stream)` under native
model control, so a faulty always-ready implementation could monopolize the
single exact decode owner indefinitely. A monotonic confirm deadline alone did
not bound calls made inside one feed, retained PCM, or repeated invocations with
a frozen/invalid caller clock.

Simply adding a loop counter was insufficient. An exact 64-decode feed still
needs a final readiness probe to distinguish successful drainage from a
requested 65th decode. Returning false on malformed KWS PCM without resetting
continuity was also unsafe because a later block could complete a keyword prefix
across the rejected gap. Treating a confirm implementation fault as ordinary
window expiry would mislabel unknown audio as echo and could teach DTD or arm
retry policy from no valid acoustic conclusion.

Wall-clock interruption was rejected. Python cannot safely preempt an
individual Sherpa/native extension call or a synchronous callback already
running on the exact owner thread. The bounded-call contract makes returned
control finite and makes exhaustion recoverable; a native call that never
returns remains an explicit limitation of the existing in-process architecture.

## Consequences

- Exactly 64 decode calls followed by a not-ready final probe remain valid;
  readiness after that probe abstains before result interpretation.
- Valid-feed KWS native faults and exhaustion recreate only KWS continuity and
  publish no command. Invalid, empty, and oversized KWS blocks become a
  controlled decoder-local capture-block gap so no later streaming decoder can
  splice across them.
- Duck-confirm has finite per-feed, per-domain cumulative, invocation, and
  strict-clock bounds derived once at window start. Boundary-equal feeds and
  totals remain valid; prospective overflow aborts before retaining or decoding
  the offending block.
- One finite, advancing deadline-crossing feed remains eligible under the
  existing decode-before-expiry order and accounts for the extra `F` of retained
  sample allowance. This slice does not change confirmation authority or
  terminal precedence.
- Confirm structural/native failure clears duck gain, generations, retained
  PCM, echo observations, handoff state, counters, and limits before the normal
  exact-owner whole-continuity recovery. Callback exceptions remain callback
  exceptions and are not reclassified as native decoder failure.
- Existing valid KWS and confirmation source selection, recognition policy,
  terminal precedence, STOP/speaker authority, and runtime callbacks are
  unchanged. Headless verification supplies no model, latency, microphone,
  audio-device, open-speaker, threshold, quality, or live evidence.

## Verification

- Focused bounded KWS/confirm headless gate (`test_barge_confirm.py`,
  `test_sherpa_media_session.py`, and `test_virtual_audio_engine.py`): 139
  passed.
- Adjacent capture/runtime gate (`test_media_session.py`,
  `test_sherpa_streaming_decode_owner.py`,
  `test_sherpa_streaming_decode_session.py`, `test_barge_word_cut.py`, and
  `test_core_runtime.py`): 258 passed; complete local-Sherpa gate: 402 passed.
- APM/DTD regression: 6 passed.
- No model, microphone, audio device, open-speaker, GPU, network, or live path
  ran.
