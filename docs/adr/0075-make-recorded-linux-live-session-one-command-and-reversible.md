# ADR-0075: Make recorded Linux live sessions one-command and reversible

Date: 2026-07-16
Status: accepted

## Decision

Add `./live.sh` as the explicit Linux/PipeWire platform launcher for a private,
recorded physical session. Keep `python -m core` and the lower-level
`./session.sh` free of host-service and default-route provisioning. The launcher
must acquire a nonblocking session lock, create a new mode-700 ignored directory
under `logs/live/`, reuse a healthy Ollama server or start one temporarily, and
reuse the canonical `echo-cancel-source`/`echo-cancel-sink` pair or load it with
the original defaults as explicit masters. It then selects both EC nodes,
requires the shared doctor to pass, and starts the production sherpa engine with
DEBUG, 16 kHz microphone recording, and a frame-aligned playback-reference
recording.

Track exact resource ownership. Restore a default only while it still points to
the node selected by this launcher, so a newer external choice is never
overwritten. Restore both safe defaults before unloading, unload only the exact
module id created by this run, and stop only the Ollama process created by this
run. If restoration cannot be verified, retain the module and return nonzero
rather than leave a default pointing at an unloaded node. A doctor failure must
stop before opening the microphone. Forward Ctrl-C/SIGTERM to the voice child,
wait for its recorder/runlog finalization, and only then clean up platform state.

Keep real-voice WAVs, aligned references, verbatim transcripts, prompts, and
Ollama startup logs local and ignored. Never stage or push them automatically.
Pass ordinary runtime options through, but reject engine, recording, enrollment,
audio-route, virtual-harness, and `open_speaker` profile overrides that would
break the launcher's evidence or cleanup contract. Do not enroll or require
enrollment: ADR-0072's optional generic word-cut filter remains unchanged.

## Context / why

The correct physical invocation previously required two terminals and a long
manual sequence: start Ollama, snapshot PipeWire defaults, load and identify an
echo-cancel module, select both nodes, run doctor, construct a private run
directory, pass recording flags, then restore and unload in the right order.
The host preflight on 2026-07-16 showed both session-only dependencies absent,
which made the safe command too difficult to reproduce and invited partial
routes, forgotten services, lost reference audio, and accidental use of the
in-app APM fallback.

Putting this lifecycle inside `core.app` would make ordinary portable startup
mutate host audio/service state and would weaken ADR-0016's side-effect-free,
fail-closed readiness boundary. A platform wrapper preserves that boundary and
is deterministic under injected `pactl`, Ollama, doctor, and child-process
fakes. The existing `session.sh` also advised committing and pushing recordings,
but new run directories are ignored and real voice is biometric data that must
remain on-device under ADR-0001/0008.

## Consequences

- The normal physical capture command on this Linux host is only `./live.sh`;
  `./live.sh --llm echo` isolates the audio path without starting Ollama.
- Every launch gets a non-reused private directory and forces both mic and
  aligned-reference evidence independently of machine-local config.
- Existing Ollama/EC resources remain running; launcher-owned resources are
  bounded to the session and cleaned up conservatively on success or failure.
- Manual `pactl`, direct `python -m core`, and `session.sh` remain available for
  diagnosis and non-Linux/platform-specific workflows, but do not acquire this
  lifecycle contract.
- A hard process kill can still bypass userspace cleanup. The next invocation
  fails closed on partial/ambiguous route state instead of stacking blindly.
- This automates preparation and evidence retention only. It does not make the
  failed physical runs pass, validate exact Stop, promote enrollment, or change
  the live-red verdict in STATUS.md.
