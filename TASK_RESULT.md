# Task Result — one-command recorded Linux live session

Valid until: ADR-0075 or the `./live.sh` lifecycle changes — then treat as history.

## Outcome

- `./live.sh` is the normal physical Linux entry point. It creates a private,
  non-reused `logs/live/<label>-<timestamp>/` bundle, starts or reuses loopback
  Ollama only for an Ollama profile, prepares or validates the canonical PipeWire echo-cancel route, runs
  the shared readiness doctor, then starts production sherpa with DEBUG, mic
  recording, and a frame-aligned playback reference.
- One host-global lock prevents different checkouts/worktrees from changing the
  same audio defaults concurrently. Exact module masters and WebRTC AEC-only
  arguments are verified before reuse; partial, stacked, duplicate, or
  ambiguous routes fail before the microphone opens.
- Cleanup restores only defaults still owned by this session, unloads only an
  exactly identified module created by the launcher, and stops only a launcher-
  owned Ollama/voice process group. Ctrl-C, SIGTERM, and SIGHUP use bounded
  finalization followed by TERM/KILL escalation when necessary.
- Runtime arguments use an explicit no-abbreviation/no-duplicate allowlist.
  Engine, recording, enrollment, raw audio-route, virtual-harness, and
  `open_speaker` overrides remain launcher-owned or rejected.
- Real voice, aligned reference, transcripts, prompts, and any Ollama logs stay
  ignored and local. Configured non-loopback Ollama endpoints are rejected and
  an ambient remote `OLLAMA_HOST` is replaced with loopback.
- Portable `python -m core` and low-level `session.sh` do not provision host
  services/default routes. Enrollment stays present and optional under
  ADR-0072; no barge-in threshold or authority behavior changed.

## Verification

- Launcher + capture + doctor integration: `195 passed`.
- Required APM/double-talk regression: `6 passed`.
- Full deterministic suite: `5205 passed, 31 skipped, 9 known warnings`.
- Shell syntax and `git diff --check`: passed.
- No physical microphone, ASR, speaker, or live-audio validation ran.

## Live invocation

From the repository root:

```bash
./live.sh --run-label vault-phrases
```

On this machine, the local private config already enables the bounded read-only
vault rooted at `~/work/dobo-brain/paul-brain`. Say `search in my vault`, `go in
my vault`, and `find in my vault`, waiting for each response; after the last,
press Ctrl-C once. The launcher
prints the private bundle directory to give a local agent for later analysis;
do not commit, push, upload, or paste its contents.

## Limits

This automates safe setup and evidence capture only. Physical exact Stop and
bare-speaker barge-in remain live-red in `STATUS.md`/ADR-0072 until a real
recorded run passes the acceptance gate.
