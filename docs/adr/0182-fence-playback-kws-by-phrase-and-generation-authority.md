# ADR-0182: Fence playback KWS by phrase and generation authority

Date: 2026-08-10
Status: accepted

## Decision

Restrict playback-time keyword-spotter effects to the shipped STOP/WAIT
interrupt floor. Pin a code-owned phone-token tuple for each phrase, make setup
reject lexicon drift from those tuples, and at engine build load the bounded
keyword file as that exact ordered six-row contract: `stop`, `stop talking`, `stop
speaking`, `be quiet`, `wait`, and `hold on`, each represented by its exact
phone-token row and collapsed `stop` or `wait` result label. During playback,
require the native decoded tokens to match exactly one row with the same closed
raw label. Missing, malformed, reordered, duplicated, custom, normalized-only,
or otherwise
unresolved bindings abstain from playback effects. The row order and phone
tokens are setup provenance, not authenticated natural-language evidence.
Threshold and boost values remain the configured values and are not changed by
this decision.

Canonicalize either accepted playback label to the typed text `stop` and
publish it only through `EngineCallbacks.on_control_stop_result`. The runtime
accepts that callback only as a typed STOP and applies the existing STOP effect
directly, without consulting the mutable command map. Do not add `wait` to the
shared STOP-command vocabulary. While the assistant is idle, preserve the
existing generic KWS callback and command-mapping behavior.

Before a playback callback, compare every shipped surface alias behind the
collapsed label against current and recent own-TTS text. If any alias reads like
the assistant's speech, abstain. In particular, `stop` covers all of `stop`,
`stop talking`, `stop speaking`, and `be quiet`; `wait` covers both `wait` and
`hold on`. Preserve ADR-0082's conservative ambiguity rule and the independent
word-cut fallback, including its existing speaker-authority machinery. KWS
does not gain speaker-based acceptance in this slice.

Bind each production KWS feed to its reader-captured capture epoch, media
generation, source generation/device identity, authority-source
generation/device identity, route proof, speaking state, speak generation, and
playback generation. Recheck that exact authority after native decode, under the
established route guard, and again while atomically claiming the callback. The
live source identity comparison is an additional
conservative check; it is not a new physical-source wall-clock barrier. Run the
re-entrant callback outside the generation lock while a short-lived claim
prevents a different playback run from starting. A playback worker must wait
for the claim, reject a stale queued sentence, and publish the new playback
generation and speaking state in one critical section. Confirmed-barge and KWS
claims remain mutually exclusive.

Require and capture the callable typed STOP callback before native playback KWS
work or claim installation. While a playback STOP claim is live, reject rather
than admit new playback submissions; a pre-claim sentence remains a valid STOP
victim, while a post-claim sentence cannot be stamped after another stop and
then cancelled by the older hit. Idle generic KWS claims do not reject response
submission and only defer the silent-to-speaking transition.

Move acoustic-turn terminalization inside the admitted capture callback
closure. If stop, capture recovery, source replacement, route loss, or a
playback transition wins before admission, emit neither a terminal nor a
callback. Release the exact KWS claim in all callback outcomes.

## Context / why

The shipped keyword file intentionally collapses distinct phrases to two
result labels. Comparing only the label to own-TTS words could suppress `stop`
yet miss the same assistant saying `be quiet`, or suppress `wait` yet miss
`hold on`. A label-only result also could not prove which configured phone row
the native decoder matched. ADR-0171 therefore required a full phrase binding
before playback-time KWS authority could be changed.

Native decode and callback delivery are separated by effectful work. In that
interval, capture recovery can replace the media/source generation and playback
can stop or begin another run. Checking only a callback epoch, or incrementing
the playback generation separately from setting speaking state, leaves an ABA
seam where a stale hit can act on a different run. Holding the generation lock
across the callback is not viable because STOP synchronously re-enters
playback/runtime control. A bounded claim plus an atomic playback-start barrier
orders those operations without granting a stale callback the later run.

Generic KWS routing is broader than a playback interrupt floor: configured
labels may enter confirmation, mode, tool, or ordinary final-text behavior.
Keeping that behavior while idle preserves the existing extension surface;
closing the playback path to the typed STOP callback prevents TTS-bearing audio
from reaching those broader effects.

## Consequences

- Shipped, exact, non-own-TTS `stop` and `wait` hits can cut only the capture,
  source, route, and playback run that produced them. Playback `wait` has the
  same STOP effect as before but cannot depend on a mutable alias map.
- Any mismatch in the six-row binding, native token result, route, capture/media
  scope, source identity, speaking state, or playback generations fails closed.
  Idle generic KWS remains available even when the playback binding is absent.
- Own-TTS ambiguity is deliberately conservative across every alias in the
  collapsed label group. A genuine simultaneous user STOP can therefore be
  ignored; the established word-cut path remains the speaker-aware fallback.
  Speaker-authorized KWS ambiguity resolution stays separate and deferred.
- Capture terminalization and callback publication now share one admission
  boundary. Losing authority cannot silently close the acoustic turn.
- The claim may briefly defer a queued playback start while the synchronous KWS
  callback runs. A playback STOP claim also rejects new speech admission until
  its synchronous effect returns; rejected text is not recorded as own-TTS.
  A callback or native call that never returns remains outside the bounded-return
  guarantees of ADR-0172.
- This extends, and does not supersede, ADR-0082, ADR-0171, or ADR-0172. It adds
  no raw-KWS stream, speaker acceptance, model, phrase, threshold, boost,
  default, capture source, route policy, device selection, or live authority.

## Verification

- The exact eight-file phrase/setup/KWS/capture/playback/runtime command recorded
  in `docs/agent-testing.md` passed **349 tests in 4.63s** with low-priority,
  single-threaded, no-cache headless settings. The adjacent media/semantic/setup/
  word-cut slice passed **218 tests in 2.30s**.
- The complete local Sherpa decode-owner command passed **449 tests in 4.52s**.
  The standard APM/DTD regression passed **6 tests in 1.00s**.
- Scoped Ruff lint passed on the five implementation files and six changed test
  files. `core/kws_contract.py` and `tests/test_kws_contract.py` are fully
  Ruff-format clean; changed regions in inherited-format-debt files are clean.
  AST parsing, `git diff --check`, and the exact 100-line STATUS check passed.
- An independent adversarial security audit is GO on the frozen production
  hashes; its focused probe passed **42 tests in 1.59s**.
- These headless receipts confer no model, microphone, audio-device,
  open-speaker, latency, quality, or live evidence. No live validation is
  claimed.
