# ADR-0069: Bind silent delay to its virtual echo route

Date: 2026-07-13
Status: accepted

## Decision

Strengthen ADR-0058 by giving only the autonomous `delay` harness a hidden,
harness-only virtual echo-route contract. Create the contract as a regular file
owned by the current uid with mode `0600`, bind it to the creating harness
parent and its child engine, and reject missing, stale, inherited, non-regular,
wrong-owner, permissive, or parent-mismatched contracts. Do not expose a public
configuration switch that can waive native route readiness.

For each run, create two uniquely named 48 kHz mono null sinks and a pinned Pulse
`module-loopback` from the far monitor into the mic sink at the declared delay.
Keep one interactive `pw-cli` process alive as the owner of a native
`libpipewire-module-echo-cancel`: its capture stream targets the mic sink, its
playback stream targets the far sink, `buffer.play_delay` carries the same delay,
and WebRTC noise suppression/gain control stay off. Record and verify the exact
Pulse module/backing-stream IDs plus the native owner PID/uid, client ID/serial,
module handle, config digest, four node IDs/serials, ports, links, targets, groups,
route nonce, rate, mono position, and latency. Record the original defaults and
reject any change.

Create a separate uid-owned, regular, single-link, non-symlink mode-`0600` ALSA
mapping for the generated capture and playback PCMs. Bind its absolute path,
full SHA-256, and exact rendered content into the contract; require
`ALSA_CONFIG_PATH` to match and validate it before importing `sounddevice` or
discovering devices. Snapshot streams before launch and accept only the exact
newly created engine stream IDs. The custom PCM opens capture directly on the
run's echo-cancel source and playback directly on its echo-cancel sink while
each PortAudio stream is stopped; it never opens a default and moves afterward.
Start capture only after this proof; start playback only after its proof, then
re-prove duplex before granting word-cut or writing FIFO/TTS samples. Disable
capture recovery and playback restart for this private gate. Missing, ambiguous,
drifted, or stopped facts fail the child and hard-cut playback with no fade.

Synthetic delay speech cannot prove owner identity. Only after the virtual
route contract validates, derive an in-memory test profile that disables owner
speaker authorization for that child run. The profile also pins Ollama to the
loopback host, disables cloud/live routing, web search, screen capture/visual
memory, GUI actions, and window watch, forces nonpersistent in-memory
conversation memory, and keeps the agent local/offline/non-OS. Never persist
that profile, alter enrollment, or relax production or enrollment route checks.

Graph teardown begins only after the child has exited. A passing run additionally
requires stdout to reach EOF and its reader to join without error; reader failure
or a nonzero child exit is red even when the bundle and earlier markers look
green. If a child cannot be killed, or any post-child audio stream remains outside
the pre-launch inventory, retain the graph and private files rather than unloading
its target and risking a physical-default reroute. Otherwise unload in reverse
and re-probe that every owned module, exact backing/child stream ID, node, owner-
module stream, generated name is absent and both defaults are unchanged.
Topology/capture/duplex markers count only when all carry the same contract digest.

## Context / why

The old delay rig created a far sink, mic sink, and delayed loopback, but launched
the engine through generic PipeWire devices. Shared startup readiness therefore
required a separately loaded host echo-cancel route even though the harness then
moved the runtime streams away from it. Loading a host module could satisfy the
preflight without proving the route carrying test audio, falsely granting word-
cut route authority. Treating delay as exempt from readiness was also rejected:
it would turn a production safety check into an unscoped test bypass. Changing
system defaults was rejected because it would mutate unrelated desktop audio
and still would not bind the newly opened runtime streams to this run.

Names alone are not provenance. A stale or foreign process can own a similarly
named node, and the ALSA-PipeWire bridge does not expose a reliable application
PID. The contract therefore combines the retained native owner's PID/uid/client/
serial lineage, exact graph links, Pulse module facts, and before/after stream-ID
provenance, then verifies capture and lazily opened playback at the point each
becomes safety-relevant.

## Consequences

- Silent delay remains device-free and self-contained, without requiring or
  modifying the host's normal echo-cancel route or audio defaults.
- Contract validation and cleanup are part of the gate. A missing tool, module,
  node, exact stream binding, child-exit fact, digest correlation, or cleanup
  proof makes the run red rather than falling back to production devices or
  weaker readiness. An unkillable child deliberately retains its private graph
  for diagnosis.
- The validated in-memory profile lets synthesized clips exercise the virtual
  ASR/TTS/echo/control path, but does not establish enrolled-owner speaker
  authorization, current-room acoustics, audible quality, or bare-speaker barge-
  in. Those remain separate recorded-owner and physical live gates.
- Production and enrollment behavior remain fail-closed and unchanged; this
  contract is not reusable as deployment configuration or enrollment evidence.
