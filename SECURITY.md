# Security Policy

Speaker is a local-first, always-listening voice assistant. Because it captures
microphone audio (and optionally the screen) and can reach a cloud thinking tier,
security and privacy are treated as first-class. This document explains what is
supported, how to report a vulnerability, and the invariants the project will not
break.

## Supported versions

This is an actively developed, pre-1.0 project. Security fixes land on the
`main` branch (the integration branch). There are no maintained release
branches; please report against, and expect fixes on, the latest `main`.

| Version            | Supported          |
|--------------------|--------------------|
| `main` (latest)    | :white_check_mark: |
| older commits/tags | :x:                |

## Reporting a vulnerability

**Please do not open a public GitHub issue for a security problem.**

Report privately, in order of preference:

1. **GitHub private vulnerability reporting** — the "Report a vulnerability"
   button under the repository's **Security** tab (Security Advisories). This is
   the preferred channel.
2. **Email** — pauldobos6@gmail.com, with `[SECURITY]` in the subject.

Please include: affected component (e.g. `core/`, `remote/`, `mobile/`,
`utils/memory.py`), version/commit, a description, and steps to reproduce or a
proof of concept.

**What to expect:** acknowledgement within a few days, an initial assessment of
severity and scope, and coordinated disclosure once a fix is available. This is a
solo, best-effort open-source project with no SLA, but security reports are
prioritized. Please give a reasonable window to fix before any public disclosure.

## Scope and threat model

In scope:

- The desktop Python runtime (`core/`, `always_on_agent/`).
- The remote host + thin-client path (`remote/`, `web/`) — see the note below.
- The mobile app (`mobile/`).
- The memory store (`utils/memory.py`).
- Build/CI configuration (`.github/workflows/`).

Out of scope (report upstream, not here):

- Vulnerabilities in third-party dependencies or models (`sherpa-onnx`, Silero
  VAD, Piper/VITS, Gemma weights — see [`NOTICE`](NOTICE)). Report those to the
  respective projects; we will bump versions once a fix exists.

### Known exposure: the remote token server

`remote/token_server.py` mints a LiveKit token for **any** identity/room as
written. **Put real authentication in front of `/token` before exposing it to a
network.** This is documented in [`CREDENTIALS.md`](CREDENTIALS.md) and is a
deployment responsibility, not a reportable bug in the default local config.

## Security & privacy invariants

These are project guarantees. A change that breaks one is a security regression.

### Privacy: the local/cloud boundary (`docs/target_architecture.md` §9.7)

- **Raw audio never leaves the device.** STT, TTS, VAD, speaker-ID, the
  always-on capture loop, and the fast/answering LLM tier run **on-device**.
- Only the optional, opt-in **thinking tier** (main planner, research,
  multimodal summarize) and **web search** may use the cloud.
- Only **post-ASR text, screen captures, and files explicitly given to the
  assistant** may cross the local↔cloud boundary, and only when the thinking
  tier is invoked.

A report showing raw audio, or any data outside that set, leaving the device
without invocation is a valid, high-severity security report.

### Secrets: the env-only golden rule (`CREDENTIALS.md`)

Every credential (`GIT_HUB_TOKEN`, `HUGGINGFACE_TOKEN`/`HF_TOKEN`, `LIVEKIT_*`,
`DATABASE_URL`) is read from the environment at runtime. **Never hard-code,
echo, log, commit, or write a token into a file — reference it only as `$VAR`.**
Repo tooling redacts tokens in all output (see `tools/gh_admin.py`), and a
gitleaks CI gate scans every push and PR. A committed secret, or tooling that
logs one, is a valid security report; see [`CREDENTIALS.md`](CREDENTIALS.md) for
the full policy.

## License

This project is licensed under the [MIT License](LICENSE). Security reports and
fixes are covered by the same terms.
