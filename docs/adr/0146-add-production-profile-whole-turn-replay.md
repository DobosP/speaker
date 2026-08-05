# ADR-0146: Add production-profile whole-turn replay

Date: 2026-08-06
Status: accepted

## Decision

Add an opt-in `production` mode to `python -m tools.bench` while retaining the
legacy benchmark unchanged. Require one explicit concrete device profile, one
strict retained schema-v2/v3/v4 corpus, and one atomic final-STT profile. Build
through the current LLM, router, runtime, FileReplay, final-selection, and TTS
factories from a clean Git revision; reject fake fixtures, partial corpus runs,
legacy download flags, and ambiguous mode-specific arguments.

Apply and bind a replay-only safety overlay after resolving the requested
production profile. Force the LLM transport to loopback Ollama with cloud off,
disable the optional third router model and configured external providers, and
replace durable memory with session memory. Bind and recheck the production and
execution config digests, atomic final-STT profile, corpus snapshot, descriptor-
scanned active audio artifacts, Python/provider runtime identities, clean source
revision, and main/fast Ollama blob plus effective-config contracts. Use the
same fixed local evaluation header for identity probes and inference, and admit
only exact local `OllamaLLM` main/fast clients.

Resolve every active Sherpa artifact path against the directory containing the
selected base configuration before runtime construction and before hashing the
artifact closure. This keeps task-worktree execution independent of its current
directory and guarantees the runtime receives the exact absolute paths whose
contents the evaluator binds. Reject path fields that are syntactically nonempty
but normalize to no path. Record that transform in replay-safety schema v2.

Treat corpus rows as independent samples. Construct a fresh engine, router,
runtime, metrics ledger, memory, and controller session for each row while
sharing only the bound stateless main/fast Ollama clients. Require the runtime's
configured best-effort warm phase to finish before resource marking, baselines,
or timers; suppress process logging while private case text is in flight; count
a response only after the null sink records completed TTS clip generation; and
release each row's native model cycle before continuing.

Publish only aggregate accuracy, WER components, command/silence counts,
truthfully named complete-PCM replay stages, sampled resources, safe bindings,
method, and limitations. Never publish input paths, case IDs, references,
recognized text, or assistant replies. Stage mode-0600 JSON/HTML files beneath
a mode-0700 directory, fsync and verify them, then atomically publish the absent
leaf with Linux `renameat2(RENAME_NOREPLACE)` through descriptor-relative
no-follow operations. Keep product defaults, the single `./live.sh` physical
entry, and live final-STT selection unchanged.

## Context / why

The legacy `tools.bench` path assembles its own historical Zipformer, Piper,
MiniCPM GGUF, and direct runtime combination. A green or fast legacy report
therefore says little about the shipped Ollama/profile/final-selector stack.
The first production-mode draft also shared mutable conversation state between
unrelated recordings, timed the first row while asynchronous warmup was still
running, allowed machine-local cloud configuration to survive, counted an
attempted TTS append as a response, exposed raw finals through ordinary runtime
logging, hashed artifact trees through race-prone paths, silently ignored
production arguments in fake mode, and could leave a partial report after a
write failure. Those defects made retained results misleading or private-data
unsafe. The first clean real preflight additionally showed that loading
configuration from the shared checkout while executing from a task worktree
made relative SenseVoice paths resolve against the wrong checkout; the runtime
failed closed before decoding audio. Config-root path binding closes that
evaluator-only mismatch.

A direct Faster-Whisper final is not introduced here. Existing command/noise
evidence shows excessive false activations, and its isolated EdAcc accuracy is
not comparable to SenseVoice through production selection and authority gates.
This decision creates trustworthy current-stack evidence before another model
or default change.

## Consequences

The deterministic gate can measure current selected-final accuracy and the
post-ASR/LLM/null-sink path without microphone access or external effects. Each
row excludes model/runtime construction and warmup from its latency while the
resource sampler still observes the process; repeated model construction is a
cost of state isolation. Ollama server residency and provider clients remain
shared, so this is neither a cold-start benchmark nor hard resource isolation.

Complete PCM exists before accelerated replay. The result is not capture, VAD
timing, endpoint latency, AEC, room echo, barge-in, playback, audible first
audio, tool-effect, naturalness, or live-conversation evidence. The active-
artifact closure is a bounded same-owner consistency check with before/after
rechecks, not a signed immutable provisioning receipt against a hostile local
writer. Aggregate publication currently requires Linux `renameat2`; other
platforms fail closed instead of weakening no-overwrite publication.

The execution-config digest intentionally changes with the selected config-root
path because it binds the absolute paths handed to native constructors. The
separate active-artifact digest remains path-independent and binds file/tree
contents instead.

No report may promote a final-STT profile or change a default. Owner-labelled
physical A/B through `./live.sh` remains required before any live selection,
latency, barge-in, or natural-conversation conclusion.
