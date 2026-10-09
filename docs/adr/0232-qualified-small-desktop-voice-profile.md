# ADR-0232: Bind a small desktop model to a qualified spoken prompt

Date: 2026-10-10
Status: accepted

## Decision

Expose explicit `voice_model_profile=qwen2.5-1.5b` for Ollama desktop clients.
Atomically select the measured project alias, `assistant.prompt_profile=spoken`,
and per-fast-role temperature 0, top_p .95, seed 0 and two requested native CPU
threads. Keep `current` as the configured default at this implementation stage;
default desktop selection is an integration decision after the shared resolver
and native/fake gates. The main/vision model, context/output caps, cloud scopes,
tool registry, action authority and generation cancellation remain unchanged.
A shared main/fast alias uses one requested thread budget and fast residency for
both roles, avoiding native runner reloads between decisions and answers.

Keep legacy system-prompt bytes unchanged. Spoken prompts preserve configured
identity/persona/extra instructions and expressive-TTS guidance, omit the large
irrelevant skills block, and retain truthful tool/web limits. Only complete,
unquoted capability questions get a deterministic full user-facing registry
summary filtered by the current typed egress scope. Compound/translation/quoted
requests remain normal requests; this shortcut neither executes tools nor mints
authority. Do not add a blanket instruction-copy suppression filter.

Provision the exact official Apache-2.0
[Qwen GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
revision `dd26da440ef0330c47919d1ecae0966d24022222`, Q4_K_M,
1,117,320,736 bytes, upstream SHA256
`6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`.
Ollama 0.30.6 rewrites the container on import; all 339 tensor payloads/types/
shapes and every metadata value matched the source. Bind its distinct runtime
SHA256 `098cb604ff3cc846891b7e8c00abe4f52f5c6fdc936e21e7e41f2eaf22c1c7cb`.
The existing `speaker-qwen2.5-1.5b:q4km-candidate` alias is retained to preserve
measurement identity; the suffix is not a live/owner/phone qualification claim.
Readiness verifies both exact native/Go templates, parameter multisets, native
family/precision and capabilities. Canonical config SHA256 is
`0a824a712723416708ee894c82f8e4a6823dcd608c8f0e281435c04acd24620f`.
Ollama's nondeterministic Modelfile parameter order and cache paths are not
behavior; duplicate/changed declarations and unknown directives fail closed.

## Context / why

Public MiniCPM component probes reproduced instruction recitation and wrong
exact addressing labels. Matched direct/core clients and official raw ChatML
controls did not establish a transport/template bug. Constrained output fixes
format, not semantics. A small official Qwen candidate provided correct answers
and resident decisions without routing all speech through the installed 12B
model. A minimal identity/style prompt scored 24/24 selection/confirmation
canaries versus 22/24 with the full registry prompt, with no Qwen recitations.
These finite public development canaries are not training-disjoint benchmarks.

The production factory (unchanged 4090 8192-context/512-output caps, fixed public
Iris persona, actual spoken runtime prompt) passed 32/32 exact answers including
eight new holdout questions, and 12/12 resident decisions with zero false ACT.
Short-answer median was 187–215 ms by split, max 261 ms; decision median 336 ms,
max 391 ms, after explicit warm-up. Sampled daemon plus descendants discovered
through every process thread reached 688,431,104 RSS bytes; Ollama reported
1,360,758,046 resident VRAM bytes. These are sampled/process/API measures, not
continuous peaks, reserved resources, phone results or physical first audio.
An earlier anonymous-persona run incorrectly expected the name Iris; it remains
separate unqualified identity-scoring evidence. A cold mismatched-thread run
produced unavailable decisions at the unchanged three-second bound; warming
once with matched options resolved that diagnostic, without raising the bound.

## Consequences

- `python -m tools.setup_voice_model --profile qwen2.5-1.5b` performs bounded,
  checksum-pinned exclusive asset publication and local alias import against an
  already running loopback daemon. It never changes saved defaults or routes,
  replaces a different asset/alias, starts a daemon or downloads other models.
- `python -m tools.voice_model_quality --host http://127.0.0.1:11435 --model
  speaker-qwen2.5-1.5b:q4km-candidate --factory-profile desktop_gpu_4090
  --conditions spoken --factory-holdout --output <fresh-json>` reproduces the
  fixed public factory holdout. The CPU factory option forces num_gpu=0 but is
  not yet native-qualified. No private prompt/audio/file input is accepted.
- Exact public canaries do not prove freeform quality. Earlier freeform checks
  found a one-sentence answer to a requested two-sentence story and generic
  anthropomorphic wording; no universal quality claim follows.
- CPU-only latency/decision qualification is pending; no automatic CPU/Mac/
  Windows adoption follows. Native import across other versions/platforms,
  disjoint owner questions, privacy/tool regressions, cancellation under load,
  actual acoustic admission, echo, barge-in and live latency remain open.
- Exact native-measured source and aggregate receipts are retained separately
  from later validation-only/tooling edits; private enrollment and recordings
  remain untouched. Headless gates are receipts, not live acceptance.

## Desktop entrypoint integration (2026-10-10)

`--voice-model qwen2.5-1.5b` selects the same resolved model/prompt in core,
readiness and the Linux launcher. The launcher forwards explicit main/fast model
overrides to readiness as well as core. A conflicting fast override refuses
before route/server setup. `--voice-model current` applies to the original
device/performance config; applying rollback to an already resolved Qwen object
refuses instead of reporting a false rollback. Programmatic factory/runtime and
readiness callers resolve the same marker. Selection never opens audio, starts
a server or changes saved config. The runtime passes the resolved persona to
the capability provider so the spoken capability-summary path is actually used.

The shared selection/identity/launcher/readiness/performance fake gate passes
365 tests (two inherited SWIG warnings). This is configuration consistency,
not a physical READY or another platform's native evidence.


## Installer boundary clarification (2026-10-10)

Native SDK metadata calls disable redirects and ambient proxy/auth inputs.
Parent symlinks and Modelfile syntax/control characters are refused before
filesystem/native writes. Exclusive source publication is unchanged. A portable
fail-busy installer lock spans alias recheck/import/identity and releases only
the same owned regular inode; different localhost spellings/asset roots
cooperate for the same temporary-directory namespace and daemon port/alias.
Ollama create has no compare-and-swap here: external writers or different
user/temp namespaces can still race alias publication. The lock is not a
daemon-wide atomic-create guarantee. Actual Windows execution remains open.


## Later qualification scope (2026-10-10; ADR-0235)

The CPU grouped resident pass is not CPU qualification. Actual production warm
and mixed recent-context controls exposed long answer tails and component
semantic failures; the shorter addressing candidate and20-second direct
prefill were rejected. Retain the legacy addressing policy and explicit model
selection. See [ADR-0235](0235-retain-legacy-addressing-after-compact-probe.md).
No CPU default, successful startup warm, model-semantic or live promotion follows.
