# ADR-0211: Prefetch one mobile TTS clip during current playback

Date: 2026-10-04
Status: accepted

## Decision

Keep one synthesis lane and one player-operation lane in TtsPlaybackOwner. After exact current
playback starts, admit at most one following synthesis while that clip plays. Prepublish the
lookahead slot before entering even a reentrant synthesis adapter, retain it across cancellation
until the exact synthesis settles, and recheck generation/lease authority before player entry.
An observed terminal prevents new speculative prefetch. Do not admit a third synthesis until
the lookahead has been consumed or discarded. Preserve the exact cleanup and uncertainty gates
from ADR-0186/0204.

Use collision-free generated WAV names, and release a successfully returned path only after
exact successful player cleanup or if no player was admitted for that stale path. Retain the
active path on native/player uncertainty. Failed file release poisons further admission instead
of allowing retained files to grow without bound. No raw recording or log is deleted.

## Context / why

The prior pump waited for complete sentence synthesis, complete playback and player cleanup
before starting the next sentence. Its audible gap therefore included the full next synthesis
cost, even when the previous sentence provided enough time to prepare it. A single lookahead
removes that avoidable serial dependency while bounding speculative native work and generated
audio. More lookahead would waste work during interruption and increase storage/CPU pressure.

Previously filenames had only one-second precision; overlapping preparation could overwrite
the currently playing file. Time plus a process-local sequence separates concurrent requests.
Prefetch begins only after playback admission, preserving the existing single-synthesis lane
through held first synthesis, revocation and successor replies.

## Consequences

Current speech can mask the next sentence's synthesis time. This is phrase/WAV pipelining, not
native within-sentence PCM streaming, and it does not reduce the first phrase's synthesis cost.
A slow native call still retains exact lower ownership until it actually returns.

The full Flutter suite passes 243 tests; independent owner/path verification passes 26 cases.
The scoped analyzer and whitespace checks are clean. Deterministic tests cover overlap without
concurrent synthesis, at-most-one prepared clip, late enqueue, cancellation, reentrant adapters,
close, returned-file release and uncertain-player file retention.

Cleanup covers returned successful paths. Failed writes, null results and timed-out native
writers do not provide a safe release receipt; their partial/late files are outside this change.
No native plugin/model, microphone, physical phone, audible latency, CPU, battery or thermal
measurement ran. Those gates and owner open-speaker A/B remain required before claiming an
acoustic or on-device performance win.
