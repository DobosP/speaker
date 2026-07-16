# ADR-0076: Unify setup-enabled local tools with the normal voice agent

Date: 2026-07-16
Status: accepted

## Decision

Keep one chatbot, persona, model stack, controller, and product entry point.
Optional machine-local setup grants add capabilities to that existing registry;
they never select a separate vault, reminder, or device-tool assistant. Keep
`./live.sh` as the normal recorded Linux launch and expose capability setup
through `tools.setup_assistant` and the same options on `install.sh`.

Retain Obsidian as the default-off, bounded, read-only `vault.search` capability
from ADR-0074. Add default-off durable local reminders and an exact trusted-app
allowlist. Reminder text lives in a private SQLite store and systemd receives
only an opaque id; a standalone helper delivers through a local desktop
notification, while a running agent claims and speaks each recent delivery once.
Trusted app opening maps one setup-approved spoken alias to one validated
`.desktop` id and invokes `gtk-launch` with a fixed argument vector. Do not
accept paths, URIs, options, trailing text, shell syntax, or model-produced app
names. Unmatched commands retain the historical `command.stage` route.

Classify capability authority centrally as `none`, `direct_live`, or
`verified_owner`. Read-only listing needs no action authority. The bounded
reminder create/cancel and exact trusted-app open operations require an
unchanged direct live-audio instruction and a later direct live-audio spoken
confirmation; they do not require enrollment. Token-changing cleanup,
continuations, merged turns, post-barge response exceptions, non-live origins,
and ambient confirmations cannot retain that authority. Sensitive, destructive,
or arbitrary actions continue to require an independently verified owner or
remain unsupported. Direct-live authority must never mint `owner_verified`.

Keep every side-effecting or authority-bearing capability out of both textual
and native ReAct tool catalogs. Controller-handled mutations are not advertised
to the answering model, so a model cannot claim an action succeeded. Providers
remain the final authority boundary and recheck strict boolean provenance and
confirmation before performing an effect.

This refines ADR-0021's capability execution, ADR-0033's planner isolation,
ADR-0072's separation of enrollment from barge-in, ADR-0074's vault reader, and
ADR-0075's single live launcher; none is superseded.

## Context / why

The vault live test exposed an architectural mismatch: a private knowledge
source had become a special way to run the assistant instead of one optional
tool available to Paul's already-defined chatbot. Tool-specific live flags also
made the physical test harder to invoke and encouraged distinct personas and
execution paths. The desired interaction is the same as web search: setup
decides whether the capability exists, natural phrases select it, and the
ordinary assistant continues handling everything else.

The same session requested reminders and trusted device apps without restoring
the old assumption that every low-risk command needs speaker enrollment.
Enrollment is useful in multi-voice rooms and remains mandatory where policy
requires verified identity, but it is not the source of ordinary barge-in or a
substitute for direct instruction plus explicit confirmation. Model-planned
mutations, arbitrary app strings, or a general shell connector would make poor
STT or untrusted content an actuator input and are therefore rejected.

## Consequences

- Setup can independently enable or disable the vault and reminders and can add
  or remove exact trusted desktop aliases in mode-600 `config.local.json`.
- Every enabled capability is present in the same runtime launched by
  `./live.sh`; old tool-specific launcher flags are rejected.
- Natural vault phrases retain ADR-0074's local PRIVATE route. Reminder and app
  mutations are deterministic controller commands with a spoken readback and
  confirmation; the model cannot call them.
- Durable reminders can notify while the voice agent is down. Voice delivery
  occurs only while it is running and is claimed at most once within a bounded
  recent window.
- The initial trusted-app connector opens allowlisted desktop applications only.
  Calendar writes, messages, arbitrary commands, filesystem mutation, keyboard,
  and mouse control are not generalized by this decision.
- Headless tests validate routing, provenance, persistence, idempotency, fixed
  process arguments, and lifecycle behavior. They do not validate physical STT,
  desktop notification visibility, systemd user-session behavior, app launch,
  microphone capture, or speaker output.
