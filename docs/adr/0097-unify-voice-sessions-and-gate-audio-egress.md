# ADR-0097: Unify voice sessions and gate audio egress

Date: 2026-07-31
Status: accepted

## Decision

Expose one core application entrypoint, `python -m core --session`, with
`console`, `local`, `replay`, and `trusted-lan` session kinds. Keep `--engine`
as a hidden compatibility selector while launchers and current documentation
migrate. `./live.sh` remains a Linux route-provisioning wrapper and must invoke
the same `local` session; it is not another assistant runtime. Construct the
tool and authority plane exactly once through `build_runtime`, inject that
`VoiceRuntime` into a `VoiceSession`, and give the session sole teardown
ownership. Media/session adapters may transport audio and transcript events,
but may not construct an LLM, tools, or another `AgentSupervisor`.

Keep audio device-local by default. Permit encrypted raw-audio transport only
to an explicitly configured self-hosted trusted-LAN endpoint after the machine
owner records `voice_session.audio_egress=trusted_lan` through
`tools.setup_assistant`. A device profile cannot grant that permission. A room,
agent identity, or exact publisher identity is only transport configuration;
remote speech remains action-untrusted. Reject public/cloud LiveKit hosts and
unencrypted non-loopback WebSocket URLs. The canonical `trusted-lan` session
must remain unselectable until ADR-0096's concrete publisher-bound LiveKit
Agents wrapper is implemented and passes its live A/B. Retain the legacy
LiveKit worker only as an undocumented rollback until that gate passes.

This decision supersedes ADR-0001. It retains one portable core and thin
platform wrappers, but replaces the unconditional statement that raw audio can
never leave an endpoint with a default-deny, explicit trusted-LAN grant. It does
not authorize Internet/cloud audio, framework-managed tools or LLMs, recording,
or remote action authority.

## Context / why

The project had user-facing Python, Dart, legacy LiveKit, and text paths whose
media loops could drift while `core.app.build_runtime` already provided the
correct single assembly point for Obsidian, reminders, trusted apps, web,
planner, confirmation, idempotency, and playback receipts. Engine selection
also exposed a monolithic implementation detail instead of the user's intended
session topology.

The low-power voice systems reviewed in July 2026 use thin endpoints and a
stronger LAN host so microphones need not run a full speech and language stack.
That topology was impossible under ADR-0001 even though current architecture
documents already described it. Treating a participant name as owner identity,
allowing a device profile to expand egress, or immediately selecting the legacy
multi-participant engine would weaken the existing authority model. Selecting
the unfinished ADR-0096 seam would claim a production path that does not exist.

## Consequences

Console, local audio, and replay now share an explicit typed session plan and
one lifecycle owner. Setup can grant or revoke trusted-LAN audio atomically,
and run summaries record only the non-secret session/topology policy. Existing
automation can temporarily use hidden `--engine` aliases.

Trusted-LAN use is deliberately not available from the canonical entrypoint in
this change. A follow-up must pin LiveKit Agents, implement the non-escaping
publisher-bound wrapper with local VAD/STT/TTS and output-route leases, disable
parallel `/chat` authority, and pass headless plus real device A/B before the
session becomes selectable. Platform convergence and stage-separated
capture/STT/turn/TTS remain follow-up work; this decision prevents those stages
from creating another chatbot or tool plane.
