# Task Result — unified setup-enabled agent tools

Valid until: ADR-0076 or the setup/tool authority lifecycle changes — then treat as history.

## Outcome

- Obsidian remains the bounded read-only capability from ADR-0074. It is an
  optional tool of the existing chatbot/persona/model stack, not a separate
  assistant or entry point.
- `tools.setup_assistant` atomically publishes mode-600 machine-local grants for
  an Obsidian vault, reminders, and exact trusted desktop aliases. The same
  options are accepted by `install.sh`.
- Durable reminders keep text in private SQLite and pass only an opaque id to a
  fixed systemd helper. Delivery and voice claims use leases plus claim tokens;
  pending timer publication retries with bounded backoff. Desktop notification
  works while the agent is down, and a running agent announces a due reminder
  through its cancellable TTS path.
- Trusted-app v1 opens one setup-approved `.desktop` id with fixed `gtk-launch`
  argv. It accepts no shell, URI, path, option, filename, or trailing model text.
- Exact reminder/app mutations require an unchanged direct live request and a
  later direct spoken confirmation. They do not require enrollment. Sensitive
  or arbitrary actions retain the prior verified-owner path or stay unsupported.
- Side-effecting and authority-bearing capabilities are excluded from textual
  and native ReAct catalogs. Mutation tools are not advertised to the answering
  model; unmatched commands retain the historical `command.stage` path.
- `./live.sh` remains the only normal Linux physical entry. Old tool-specific
  launcher switches are rejected.

## Verification

- Full deterministic suite: `5303 passed, 31 skipped, 9 warnings`.
- Launcher/capture/setup-doctor gate: `197 passed`; APM double-talk gate:
  `6 passed`.
- No real systemd timer, desktop notification, trusted app, microphone, speaker,
  or physical STT validation ran in this implementation session.

## Machine-local setup

From the repository root:

```bash
.venv/bin/python -m tools.setup_assistant \
  --obsidian-vault /home/dobo/work/dobo-brain/paul-brain \
  --enable-reminders \
  --trust-app obsidian=obsidian.desktop
```

Setup validates only the vault directory and exact desktop id syntax; it does
not enumerate notes, launch Obsidian, create a timer, or display a notification.

## Live invocation

```bash
./live.sh
```

In that one session, try `search in my vault for speaker`, `which reminders are
set?`, `remind me to stretch in ten minutes` followed by `confirm`, and `open
obsidian?` followed by `yes`. Press Ctrl-C once after the responses. The launcher
keeps its private evidence bundle local; do not commit, push, upload, or paste it.

## Limits

The trusted-app connector opens allowlisted applications only; it does not yet
write calendars/messages, manipulate files, type, click, or run arbitrary code.
Headless tests do not improve or validate STT accuracy. Physical exact Stop and
bare-speaker barge-in remain live-red in `STATUS.md`/ADR-0072.
