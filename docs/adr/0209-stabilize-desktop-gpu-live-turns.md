# ADR-0209: Stabilize desktop GPU live turns before another owner A/B

Date: 2026-08-21
Status: accepted

## Decision

Keep malformed addressing output fail-closed. For a launcher-owned Ollama
daemon, set `OLLAMA_GO_TEMPLATE=1` only in that server process. Bind the pinned
MiniCPM Q8 identity to the selected `/api/show` capabilities and require the
exact ordered set `completion`; a reused or newly started daemon that exposes
tools, thinking, a missing value, or any other set fails readiness before live
capture. Do not alias an out-of-contract label to `ACT`, and do not enable
`unsure_acts`.

On `desktop_gpu_4090`, disable the blocking whole-clip TTS output leveler while
retaining the configured target-RMS path. The discarded TTS warm clip seeds the
first live fragment's gain in the same post-DC domain as playback, through a
fresh throwaway DC blocker, and synthesis reads that seed only after acquiring
the TTS lock. Treat both partial FIFO underruns and fully empty synthesis dry
gaps as starvation when adapting FIFO lead. Count PortAudio native output
underflow separately so route/device deadline misses are not reported as slow
TTS.

Also require compatible enrolled-speaker authority for generic 4090 word-cut.
Preserve ADR-0042's novel exact STOP/cancel policy, including its own-speech
ambiguity rule, but never restore identity-free generic four-word interruption
on this profile. Resolve and warm the enrollment after the actual capture route
is known; a missing or capture-front-end-incompatible reference makes startup
fail closed. Bind pre-endpoint acoustic turns to their current synchronized
capture span, propagate that span and capture epoch through word-cut and confirm
callbacks, and publish `BARGE_DETECTED` inside the admitted capture effect
before user callback code can synchronously emit `FINAL_ABORTED`. Keep the
diagnostic schema strict.

## Context / why

The bounded 2026-08-21 desktop smoke was stopped after it failed to answer one
direct turn, produced distorted/white-noise-like playback with long gaps,
interrupted itself, and appeared stuck. Aggregate-only inspection of the two
retained private bundles, without reading transcript or PCM content, separated
four concrete causes.

The installed Ollama daemon selected the embedded GGUF thinking/tools template
instead of the alias's verified Go template, and the fast addressing model
returned an out-of-contract label. The parser correctly reduced that label to
`UNSURE`, so loosening the parser would convert arbitrary malformed room output
into speech authority. The second run admitted 56 TTS fragments but started
only eight; whole-clip synthesis repeatedly took longer than the preceding
audio, emptying the application FIFO between sentences. Synthesized clips were
not clipped, so this was not evidence against the answer model or waveform
quality. The old FIFO controller ignored thousands of fully empty dry-gap
callbacks when the partial-underrun count was small.

Generic identity-free word-cut then accepted garbled playback leakage as four
novel words. The configured enrollment did not match the active capture front
end, but that mismatch was optional under the old profile. Finally, a
pre-endpoint cancellation and shutdown could emit `FINAL_ABORTED` before the
acoustic lineage had an associated bundle span. Relaxing the diagnostic schema
would hide the missing authority rather than repair it.

## Consequences

- A launcher-owned daemon uses the verified MiniCPM template. A reused daemon
  with an incompatible selected capability set fails doctor/readiness before
  the mic opens; generic Ollama model identity digests remain unchanged.
- Once background warm-up completes, the first 4090 reply may stream with a
  post-DC feed-forward gain. Warm-up remains best-effort: speech submitted
  before it completes may safely use the existing first-sentence whole-clip
  fallback, so first-fragment latency must be measured only after warm-ready.
- Generic talk-over is unavailable until the owner repeats isolated enrollment
  on the exact active echo-cancel/capture front end. This is an intentional safe
  startup failure, not permission to disable the speaker requirement. Novel
  exact STOP/cancel remains the open playback-only fail-safe after valid startup
  and still grants no owner, application, or tool authority.
- Playback diagnostics can now distinguish application starvation from native
  output scheduling underflow. The adaptive ceiling, one-TTS-call ownership,
  FIFO/ticket ordering, actual-playback references, and cancellation generation
  fences are unchanged.
- Headless verification passes 796 affected tests, 392 adjacent lifecycle/audio
  tests, and the required 6 APM/double-talk tests. Scoped Ruff is green under the
  unchanged inherited E731/E402/F401 exclusions; the same fifteen pre-existing
  files remain formatter-red on `main` and this branch.
- No repaired live run has occurred. Owner open-speaker validation must still
  confirm addressing, first-audio latency, gap/static absence, generic and exact
  STOP barge behavior, route restoration, and a complete diagnostic bundle.
