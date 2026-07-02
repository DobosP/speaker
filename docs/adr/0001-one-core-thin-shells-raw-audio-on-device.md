# ADR-0001: One portable core + thin per-platform shells; raw audio never leaves the device

Date: 2026-05-28
Status: accepted

## Decision
Build ONE portable Python core (`core/`) with thin per-platform shells — not a
monolith, not N independent apps. All platforms share the `always_on_agent`
`AgentEvent`/`Mode` contract; the small brain is reimplemented faithfully per
runtime (Python on desktop/server, Dart on mobile). Deployment topology is
hybrid: the always-on capture loop (STT/TTS/VAD/speaker-ID + the fast LLM
tier) runs fully on-device and **raw audio never leaves the device**; the
thinking tier (main planner / research / multimodal summarize / web search)
may use cloud — only post-ASR text, files, and screenshots given to the
assistant may cross, and only when invoked.

## Context / why
iOS forbids Python background voice, so a single binary core is impossible; N
independent apps duplicate the brain and drift. The §9.7 boundary
(`docs/target_architecture.md`) preserves privacy while unblocking cloud
headroom for thinking-tier work. Why not end-to-end speech-to-speech: a
resident Moshi-class S2S model is ~24 GB — wrong for low-spec devices; the
cascaded ASR→LLM→TTS pipeline keeps a clean text-only egress seam
(re-validated by the 2026-06-16 architecture audit). Why not cloud STT/TTS:
raw voice is biometric PII; the always-on loop must survive offline.

## Consequences
- Every new capability must declare which side of the §9.7 boundary it sits
  on; anything moving raw audio off-device is rejected by design.
- The `AgentEvent` contract is the compatibility bottleneck — change it in all
  shells or none.
- Mobile convergence onto the contract (replacing the parallel Dart loop)
  remains open work; revisit when the Android app grows features.
