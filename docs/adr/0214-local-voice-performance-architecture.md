# ADR-0214: Repair bounded local media before choosing a whole-runtime rewrite

Date: 2026-10-04
Status: accepted

## Decision

Keep native ASR/TTS and local text-model inference behind the current typed control plane.
Prioritize a bounded platform media boundary with exact capture/playback timestamps, an actual
render reference for echo cancellation, one live streaming endpoint authority, bounded final
recognition, one text generation and one TTS synthesis lane with one prepared next clip.
Keep cancellation and exact session ownership ahead of new work. Keep inference and disk
operations outside real-time callbacks. Select language, model quantization, context/KV,
thread and acceleration budgets from disjoint quality and sustained actual-device measurements.

Use incremental replacement of measured media bottlenecks. Do not commission a wholesale
Python/Dart-to-Rust rewrite on the assumption that language alone changes native model compute.
A shared Rust/C++ media kernel with a small C ABI becomes an implementation candidate when
profiling demonstrates avoidable PCM copies, callback deadline misses or platform media drift.
The shared AgentEvent/Mode and authority contract remain the convergence boundary; this ADR
refines the performance plan without replacing ADR-0001 or ADR-0097's deployment/privacy rules.

## Context / why

The owner reports repeated self-interruption, missed speech, slow replies and stuck turns.
The review reproduced synthesis-time onset grace consuming audible echo protection, sequential
mobile sentence synthesis/playback gaps, tiny capture messages exhausting a count-only mailbox,
and language-model warm-up preceding media warm-up. ADR-0210 through ADR-0213 repair those
seams separately. Model inference already uses native libraries; no measured comparison here
establishes a benefit from replacing the orchestration language.

The [research and review](../local_voice_performance.md) compares primary-source techniques
from Google, Apple and established local runtimes, and records candidate language coverage,
licenses and the difference between vendor inference figures and full-device latency.
Multilingual Whisper, Parakeet v3 and Qwen3-ASR have Romanian support, while the current phone
Zipformer/Piper selection does not. Model language support alone does not prove this owner's
accuracy, command reliability, memory budget or physical-phone throughput.

## Consequences

The four repairs preserve local audio and established control authority. The current phone
uses phrase/WAV pipelining; zero-copy native PCM playback, effective AEC reporting, total CPU
budgeting, multilingual app integration and full Python/Dart control-plane convergence remain
implementation work requiring their own tests and device evidence. No model, threshold,
endpoint default, authority gate or cloud permission is promoted by this research decision.

Acceptance of this fully-local task requires the provisioned ASR/LLM/TTS pipeline to work
with inference networking disabled, and local failures to report unavailable without cloud
or trusted-LAN fallback. It also requires disjoint owner/public quality strata, cold/warm
end-to-end latency, owner bare-speaker desktop A/B and phone CPU/RSS/PSS/thermal/battery validation.
The published initial latency targets are engineering hypotheses rather than measured product
claims. Existing model rejections retain their exact historical scope and no prior artifact
is deleted or reclassified as fresh live evidence.
