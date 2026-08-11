# ADR-0184: Bound KWS ledger object and callback PCM lifetimes

Date: 2026-08-11
Status: superseded-by ADR-0185

## Decision

Add an independent hard ceiling of 256 retained PCM chunks to ADR-0183's
speaker-enabled KWS ledger. Apply the unchanged configuration-derived sample
window first, then discard additional oldest whole chunks as needed so the
newest submitted chunk is retained and the stable ledger contains at most 256
chunk objects. Advance `retained_start` to the first surviving sample. Do not
restart the native KWS stream merely because this object ceiling trims PCM:
the native stream already legitimately outlives the sample-window ledger, and
the existing exact timestamp/range checks make any ambiguous word span that
touches discarded PCM abstain.

Treat 256 chunks as a second conservative speaker-evidence horizon, not as a
claim that every positive finite capture geometry retains its complete
configured time window. The committed 100 ms blocks use twenty objects for the
default two-second window, and 10 ms blocks use 200. Finer fragmentation may
lose older ambiguous-speaker evidence earlier; it can cause only an abstention,
never authorization. Stable idle KWS, novel playback KWS, exact native sample
clock/adjacency, the current submitted chunk, and every ADR-0183 generation,
route, token, timestamp, speaker, and final-claim fence remain unchanged.

Copy one validated multi-word KWS phrase interval from the ledger into one
owned C-contiguous float32 array, then expose read-only per-word views into that
single root. Fill the owned interval directly instead of concatenating pieces
and then copying the concatenation again. Before allocating that output,
validate the requested span against the configuration-derived ledger window
and validate every traversed chunk's exact type, ordered contiguous
coordinates, float32/C/read-only PCM shape, and coordinate-to-size agreement.
Malformed internal geometry abstains without allocating phrase output. During
synchronous speaker preparation, normalize with `np.asarray` so an already
matching float32/C caller array can be reused rather than unconditionally
copied; retain the existing independent read-only voiced identity copy and
every unchanged scoring validation. This follows NumPy's documented
[`asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html)
normalization and explicit-output behavior without relying on object identity
for authority.

After a hit has been reduced to immutable accepted status and the native stream
has rotated, release the old stream, submitted-feed array, phrase views/root,
and decision identity arrays from the Python frame before invoking the external
typed STOP callback. This is a Python-reference lifetime boundary, not proof
that caller-owned current samples have been released, or that a native
extension, model call, or callback returns or releases its own internal storage.

Keep the PCM-byte limit configuration-derived. Do not add a new configuration
field, change the configured window or feed size, hard-cap otherwise trusted
large finite local configuration, move or retry speaker inference, or change a
model, threshold, label, callback, STOP effect, default, or authority. ADR-0184
extends ADR-0183 and supersedes no decision.

## Context / why

ADR-0183 bounded retained sample payload but used that sample count as its only
chunk-count check. With ordinary 100 ms blocks the default ledger has twenty
chunks, but a structurally valid one-sample feed could split the same 32,000
samples across 32,000 NumPy arrays and dataclass objects. The ADR-0183 audit
measured 5,768,760 bytes of shallow storage for that adversarial geometry,
approximately 5.50 MiB or 5.77 MB in decimal, even though payload bytes
remained 128,000. A separate object ceiling closes that amplification without
pretending the trusted configuration-derived byte limit has become universal.

Resetting native KWS on every object-ceiling event was rejected. At 16 kHz
one-sample feeds and a 256-object ceiling it could induce roughly 62 native
stream rotations per second, while providing no additional authorization
property: token timestamps that precede `retained_start` already fail closed.
Dropping only the oldest whole PCM chunks preserves current native/idle/novel
continuity and makes the narrower ambiguous-speaker evidence unavailable.

The same audit found two avoidable phrase-copy layers and Python PCM references
held across a re-entrant external STOP callback. Those references were bounded,
but eliminating them lowers normal peak memory and makes callback retention
independent of the phrase window.

## Consequences

- The default two-second speaker-enabled ledger remains twenty normal chunks;
  pathological fragmentation retains at most 256 chunk/array objects. The
  current chunk always survives object trimming, and ambiguous timestamps into
  an extra-trimmed prefix abstain.
- A two-word phrase owns one exact copied root plus two read-only views rather
  than separate concatenate-and-copy results. Speaker preparation may reuse
  those views synchronously but still creates independent identity PCM for the
  native embedding calls.
- No phrase, identity-decision, old-stream, or submitted-feed Python reference
  intentionally crosses the typed STOP callback boundary after acceptance.
  Caller-owned current samples and storage retained inside native/model code
  remain outside this Python-frame claim.
- Arbitrarily large trusted finite window/feed configuration and an already
  entered synchronous native embedding remain separate resource/liveness
  questions. The paired owner bare-speaker A/B required by ADR-0183 also remains
  pending.
- Headless tests can prove exact cap/eviction boundaries, unavailable-scope
  zero PCM, old-prefix abstention, one-root ownership/no-aliasing, callback-time
  weak-reference release, and unchanged KWS effects. They do not prove native
  allocator behavior, device memory, model quality, latency, or live audio.

## Verification

- The isolated virtual-engine gate passed **156 tests in 1.58s**, including
  exact 256/+1 chunk eviction, sample-window precedence, retained-tail mapping,
  forged-geometry preallocation rejection, callback weak-reference release,
  and submitted-clock closure where `sys.maxsize - 1` accepts one final
  sample, the next sample resets and abstains, and a later feed starts at a
  fresh zero origin.
- The exact ten-file KWS resource/authority gate passed **613 tests in 5.87s**.
  The exact adjacent owner/activation gate passed **280 tests with two SWIG
  deprecation warnings in 3.40s**, and the exact complete-Sherpa gate passed
  **522 tests in 5.57s**. The APM/DTD gate passed **6 tests in 1.78s**.
- Sherpa F821/F822/F823 lint, full test-file Ruff lint and format, two-file AST,
  whitespace, and the exact 100-line STATUS check are green. Whole-Sherpa Ruff
  lint retains exactly the same 23 findings as `main`. Whole-Sherpa formatter
  debt improves from 34 `main` hunks to 33 branch hunks, with zero exact
  changed-line overlap.
- Frozen `core/engines/sherpa.py` SHA-256 is
  `d56aaddb4c0285b3c82a01b6eb3a0ba13ee553efe8b1df305c6e2f1b426dd934`;
  the production diff SHA-256 is
  `74cd781ccaca50bd94798658d22c95325b9eb09ad0bba4536911d8ea8928649a`;
  and `tests/test_virtual_audio_engine.py` SHA-256 is
  `fdd3b3f0cfc459deb1233e7942115eb8ff6dc8bf2f5bde1b2ac64ffa1d3017e4`.
- No model, microphone, audio device, GPU, network, or live path ran for this
  headless resource/lifetime slice.
