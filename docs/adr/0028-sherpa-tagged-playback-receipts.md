# ADR-0028: Sherpa tagged playback receipts

Date: 2026-07-11
Status: accepted

## Decision
Make Sherpa receipt-capable by carrying opaque fragment ownership beside samples
inside `PlaybackFIFO`. Count post-resample samples admitted and frames copied into
the PortAudio callback under the FIFO's existing lock. Emit only lifecycle facts
from that lock into a dedicated non-real-time dispatcher; never invoke receipt
callbacks on the audio thread. Seal completion only after synthesis returns and
every admitted sample drains. On a cut, retain a bounded fade from the already
started head owner only, then expire that ownership independently if either the
sink or native synthesis stalls. Report full sanitized text only for completion;
never infer words from sample ratios. Preserve legacy behavior for subclasses that
override only `speak()` unless they explicitly implement the tracked seam.

## Context / why
Sherpa's legacy `on_done` runs after synthesis, while audio may still be queued in
the shared FIFO. Consecutive streamed utterances can coexist there, so an external
text ledger cannot atomically distinguish a PortAudio read from a concurrent flush.
Calling the runtime directly from the PortAudio callback would move locks, memory
I/O, and arbitrary consumer code onto the hard-real-time thread. A global fade is
also unsafe: it can start an unheard later fragment after a barge-in, and a fade
whose callback stalls can otherwise leave the receipt pending forever.

## Consequences
Sherpa now advertises tracked terminal, exact-start, and sample-count capabilities.
FIFO ownership linearizes completion versus interruption and preserves exact
output-domain accounting across resampling, backpressure, queue eviction, stale
generation, failure, output reopen, drain timeout, barge-in, and shutdown. A small
dispatcher adds one daemon thread and bounded polling latency; the audio callback
only appends at most start/terminal lifecycle facts per fragment. FileReplay and
LiveKit remain legacy until their sinks can provide equivalent evidence. Headless
tests prove the contract and production-worker races, but bare-speaker owner-mic,
audible fade quality, and live output-device recovery still require manual A/B.
