# Architecture audit — performance, low-spec adaptation, privacy (2026-06-16)

**Scope.** Compare this solution against alternative/SOTA architectures to find
performance headroom, with the *primary* lens being **running well on relatively
low-spec devices and adapting to device specs**, and **security & data privacy as
a hard constraint** (the `docs/target_architecture.md` §9.7 boundary). Produced by
a fan-out multi-agent audit: 8 architecture dimensions, each
*characterized-from-code → researched-against-SOTA → compared → adversarially
verified on two axes (perf-feasibility + privacy/security)*, then a cross-cutting
security audit, a low-spec-weighted roadmap, and a completeness critic.
**66 recommendations, 57 survived verification, 9 rejected.**

---

## Bottom line

1. **Your component choices are validated as best-in-class — do not replace them.**
   Every dimension's research independently confirmed it: sherpa-onnx streaming
   Zipformer-int8 + SenseVoice second pass + Piper VITS is *the* low-spec on-device
   front end; the **cascaded ASR→LLM→TTS pipeline is correctly chosen over
   end-to-end speech-to-speech** (a resident Moshi-class S2S model is ~24 GB — wrong
   for low-spec); the **threaded priority-bus + supervisor control plane** is the
   dominant 2025-26 pattern for mixed real-time work; and the **on-device-first
   privacy boundary is ahead of most open alternatives**.

2. **The real gap is adaptation, not architecture.** The system is
   *choose-a-tier-once-at-launch* and — critically — **the tier often isn't even
   chosen**: `recommend_profile.py` only *prints* a recommendation, nothing applies
   it, and `core/app.py` falls back to the heavy **`desktop`** profile. So an
   unconfigured low-spec laptop or phone **silently runs the 12B desktop tier with
   no thread budget, full second-pass ASR, and full-size TTS.** The biggest wins are
   wiring up adaptation that the codebase mostly *already has the parts for*.

3. **Privacy direction is net-positive (APPROVE WITH CONDITIONS).** No surviving
   recommendation breaches §9.7 as intended; raw audio stays on-device everywhere.
   Two systemic guardrails are mandatory before adopting the set (below), and there
   is **one confirmed, pre-existing hygiene issue that is effectively a release
   blocker** (committed biometric WAVs — below).

### Top 3 changes for low-spec devices (the audit's own ranking)
`device-adapt-1` (auto-apply the probed profile) · `llm-inference-5` (make the
phone profiles actually runnable) · `asr-tts-1` (gate the ASR second-pass/beam by
tier). All three are S/M effort and unblock everything else.

---

## Prioritized roadmap (privacy-gated, ranked by perf × low-spec benefit × 1/effort)

### Quick wins (S effort, ship first)
| ID | Change | Why it matters for low-spec |
|----|--------|------------------------------|
| **device-adapt-1** | Auto-apply the probed profile at launch; **fail-fast** (not silent no-op) on an unknown `--device`. | The single biggest out-of-the-box low-spec win — stops a weak box silently booting the `desktop`/12B tier. |
| **asr-tts-1** | Per-tier ASR policy: `phone_lite` → streaming-only finals (`asr_final_backend=''`) + `greedy_search`; keep beam+SenseVoice on desktop. Keys half-exist already. | The tier that can least afford the ~55 ms offline re-transcribe stops paying for it. |
| **llm-inference-2** | Wire `llm.n_threads` from the profile and **budget it against** sherpa's STT/TTS threads. Today no profile sets it → the LLM grabs all cores and starves the capture/AEC path. | Removes CPU oversubscription on shared-core phones; protects barge-in responsiveness. |
| **llm-inference-3** | Cap on-device `max_tokens` on phone profiles — using llama.cpp's `max_tokens` key, **not** Ollama's `num_predict` (the options dict is splatted into both; copying the wrong key no-ops). | Bounds worst-case turn latency/energy on the slowest CPUs. |
| **routing-cascade-7** | Word-boundary-compile `_ACT_MARKERS` / `_ESCALATE_MARKERS` (reuse the existing `\b` helper). | Fewer spurious escalations to the heavy tier = fewer wasted calls on weak devices. |
| **control-plane-3** | Scale watchdog stall deadlines off the rolling local-TTFT EWMA, not fixed 10 s/5 s constants. | Stops false stall warnings polluting every low-spec run bundle. |
| **device-adapt-5** | Tier-aware headroom signal: split `load_fraction()` (CPU% on llamacpp tiers, not `max(CPU,GPU)`) and sample faster (~1-2 s) during turns. | On CPU tiers the LLM *is* the CPU load — measure the real bottleneck. |
| **security-privacy-1** | Redact transcripts at the run-log **writer** + retention TTL; re-ignore `logs/runs/*.wav`; purge the 7 tracked owner WAVs. | Makes PII-free bundles the enforced default (see hygiene P0 below). |
| **cross-platform-8** | Validate `--device` names; reject unknown ones in `core.app` + `remote/worker.py`. | Stops a mistyped profile silently running heavy base config on the device that needs adaptation most. |
| **cross-platform-6** | Replace linear resampling in the LiveKit path with the local engine's anti-aliased (soxr) resampler. | Better remote ASR input quality at no cost. |
| **asr-tts-7** | Bake off Kroko-ASR (same streaming Zipformer-int8 class) as a config-only accuracy upgrade. | Lowest-risk model swap; identical footprint. |

### High-impact (M effort, biggest perf / low-spec gains)
| ID | Change | Impact |
|----|--------|--------|
| **llm-inference-5** | Make phone profiles **provisionable**: add `llama-cpp-python` extra, fetch GGUFs in `setup_models --gguf`, fail-fast with the exact acquisition command. | Foundational — the entire on-device LLM path is *inert/unrunnable as shipped*. |
| **llm-inference-1** | Reuse llama.cpp KV/prefix cache for the byte-stable system prompt across turns (retain the pre-warm KV instead of re-prefilling every turn). | **Largest single TTFT cut on the CPU phone path** — prefill dominates there. |
| **asr-tts-2** | Run the ASR second pass **asynchronously** — dispatch the streaming final to the LLM immediately, upgrade in place when SenseVoice finishes (reuse `turn_merge` off-thread machinery). | Removes ~55 ms+ from the endpoint→LLM critical path on every utterance. |
| **asr-tts-4** | Decouple loudness normalization from streaming synth so first-audio streaming works **with** the echo floor (chunk-wise limiter instead of whole-clip RMS). | First-audio latency drops to first-chunk time; biggest win on slow CPUs. Preserves the open-speaker echo floor (no headphones). |
| **audio-bargein-1** | Per-device-profile **audio-tuning block** (`block_sec`, coherence `nperseg`/`ring_ms`, AEC taps, DTD K) — today a phone runs *identical* DSP params to a 4090. | The enabling change for all low-spec audio tuning. |
| **routing-cascade-4** | Fully-local **embedding/semantic router** as the cheap disambiguator on weak profiles (capability router is OFF on every non-desktop profile today). | Sub-ms intent routing with zero LLM calls on a phone. |
| **routing-cascade-1** | Perplexity/confidence escalation axis: defer fast→main only when the fast tier's own decode is low-confidence (free signal). | Most turns resolve locally; only genuine hard cases pay the heavy/cloud cost. |
| **llm-inference-9** | KV-cache quantization (INT8 `cache-type-k/v`) + bounded context; **gate speculative/NPU decode to the high tier only** (a draft model doubles RAM — wrong for low-spec). | RAM is the wall on phones, not FLOPs. |
| **control-plane-2** | Feed `load_fraction()` into admission control: drop the task ceiling to 1 under load so two CPU-bound generations don't co-run. | Turns the static 6-cap into a load-elastic cap exactly when a weak device saturates. |
| **audio-bargein-5** | Promote the already-implemented **Smart Turn v3** prosody endpointer (8 MB int8) from OFF to a validated, per-profile default (needs a real-voice A/B harness). | Lets `endpoint_min_silence` drop, reclaiming part of the ~806 ms endpoint p50. |
| **device-adapt-2** | Thermal + battery sensing → size for the **sustained (hot) plateau**, not the cold burst. | Critical for fanless phones where sustained tok/s ≪ burst. |
| **cross-platform-2** | Auto-calibrate mobile barge-in thresholds from a short ambient/echo sample (replace the hardcoded `_bargeInRms=0.08`). | Cheap phones have the worst mics/echo; learned floors are what make open-speaker barge-in robust there. |
| **asr-tts-3** | Runtime **RTF guard**: measure decode/synth real-time-factor per turn and self-degrade when behind real-time. | Central runtime adaptation — a throttled phone currently has no feedback loop. |

### Strategic (L effort / architectural)
- **audio-bargein-9 / cross-platform-4** — extract the barge-in cascade into a pure
  `BargeInDetector` policy class (aq-1/D7) and reuse it in LiveKit + the Dart mobile
  loop, so the lowest-spec target inherits the tuned P1 logic and replay-WAV
  regressions can pin it in isolation.
- **audio-bargein-2** — wire the **already-existing** runtime AEC-delay
  auto-calibration (`measured_delay_samples`) into a slow, debounced re-align,
  removing the per-machine human `echo_probe` step (directly serves the open-speaker
  barge-in P1).
- **audio-bargein-7** — land the deferred DTLN-aec deep tier at the phone-sized
  128-unit config (currently fails open to no-AEC).
- **audio-bargein-8** — move coherence/far-ring/level-EWMA ingest off the PortAudio
  callback (real-time underrun hazard, worst on small low-spec buffers).
- **device-adapt-6** — per-turn offload of the *thinking tier only* to a nearby
  trusted host. **Mechanism fix required:** must NOT reuse the LiveKit full-handoff
  transport (it carries raw PCM); send only post-ASR text over the `/chat` channel.
- **routing-cascade-2 / device-adapt-9 / routing-cascade-6** — activate the dormant
  fail-safe `live_routing` nudge on slow profiles; calibrate specsim/recommend_profile
  thresholds from real measured runs; land aq-5 "one route decision per turn".
- **llm-inference-6** — reap timeout/cancellation for in-process llama.cpp (swap/OOM
  stalls bite hardest on low-spec).
- **security-privacy-2 / -4** — optional tiered on-device NER PII backstop
  (regex→GLiNER, OFF on phone); endpoint pinning + fail-closed allow-list for the
  cloud thinking tier.

---

## Security & data-privacy verdict — APPROVE WITH CONDITIONS

No surviving recommendation breaches §9.7 *as intended* — raw audio stays
on-device on every path, STT/TTS/VAD/speaker-ID/fast tier stay local, cloud stays
default-OFF and owner-gated. Mandatory guardrails for adopting the set:

- **Lock the cloud-enable invariant in code.** Auto-profile selection
  (`device-adapt-1`, `llm-inference-4`, `cross-platform-1/8`) is safe *only* while
  no profile enables cloud or relaxes the actuator/speaker-ID default-deny gate.
  Add an **invariant test**: no `device_profile` sets `llm.cloud.enabled=true` and
  none relaxes the gate.
- **Allowlist the new `config.local.json` write surface.** Persist only the profile
  *name* (and model paths) — never `cloud.enabled`, never gating fields. (`config.py`
  only *reads* this file today; persistence is net-new.)
- **One integrity-verified model manifest.** The provisioning fan-out
  (`llm-inference-5/7/8`, `asr-tts-7`, `audio-bargein-7`, `routing-cascade-4`,
  `security-privacy-2`) adds many downloaded artifacts → pin `{repo, revision,
  sha256}` and assert the hash post-download (currently a bare `hf_hub_download`).
- **Keep the regex+Luhn egress redactor as the unconditional last stage.** Any new
  NER/"sketch" stage (`security-privacy-2`, `routing-cascade-8`) must be additive,
  never replace the deterministic floor (fail-open risk).
- **`device-adapt-6` mechanism is forbidden as proposed** — never offload over the
  raw-PCM LiveKit path; post-ASR text only.
- **Barge-in/owner-gating cluster** (`audio-bargein-4/6/7`, `asr-tts-4`,
  `control-plane-7`, `cross-platform-2/4`) must each land behind the replay-WAV
  golden regressions and be validated on the bare Realtek speaker.

### Confirmed hygiene issue — DEFERRED to a pre-release gate (owner decision, 2026-06-17)
- **7 raw personal-voice WAVs + 21 `summary.json` + 23 `.txt` run bundles are
  git-tracked** (`logs/runs/run-2026*.wav` etc.), via a deliberate
  `!logs/runs/*.wav` un-ignore at `.gitignore:50`. Raw voice = biometric PII and
  verbatim transcripts.
- **Owner decision (D-B):** these bundles stay committed **during active
  development** — they are in use (and are the barge-in golden-regression corpus).
  The purge + transcript redaction is therefore **deferred to a pre-release gate**,
  not a current action. See [`roadmap_2026-06-17.md`](roadmap_2026-06-17.md) §DEFERRED.
- **When triggered (before public/non-owner release):** purge from history
  (filter-repo), flip the WAV un-ignore to opt-in, turn on writer-level redaction
  (`security-privacy-1`), add a `gitleaks`/PII CI gate.
- **Caveat that still applies during dev:** if the GitHub remote is *public*, these
  WAVs are already exposed regardless of local use — consider a private remote until
  the purge.
- **No API keys in the tracked TREE — but a real key IS reachable in PUBLIC
  history.** A scan of the tracked tree for key material hit only dummy strings
  in test files (`test_cloud_pii_egress.py` etc.); `CREDENTIALS.md` is env-var
  docs and `docker/.env.example` is an example. However the "rotate keys"
  recommendation **is evidence-backed**: a real Gemini key remains reachable in
  the PUBLIC repo's git history at `d32db9f` (`.env`, untracked 2026-06-10 via
  15ef939 but never purged). *Correction + status 2026-07-02:* key **rotated
  2026-07-02**; the public-history purge (D1) is **still pending, OWNER-ONLY**
  — agents must not run filter-repo/force-push.

---

## Rejected / down-scoped ideas (recorded so they're not re-proposed)

| ID | Idea | Why rejected |
|----|------|--------------|
| asr-tts-5 | Matcha-TTS tunable-step tier | `build_tts` is VITS-only; no Matcha branch exists — large rework, not a config swap. |
| asr-tts-8 | Moonshine v2 Tiny front end | Already baked off on the owner's real voice; was unreliable ("empty on half the clips"). |
| device-adapt-3 | Runtime graceful-degradation ladder | Right idea, but the rungs aren't uniformly cheap as claimed; needs scoping per rung. |
| device-adapt-4 | Gemma 3n / MatFormer as the phone main+fast pair | Model-fit story holds but feasibility/runtime support overstated — keep as a spike (`llm-inference-8`), not a swap. |
| device-adapt-8 | Self-speculative / early-exit (LayerSkip/SWIFT) | Mainstream llama.cpp/sherpa don't expose it; the rec argues itself out. |
| control-plane-5 | Event-driven cancel vs `time.sleep(0.01)` | Real but negligible perf; misframed. |
| control-plane-6 | ThreadPoolExecutor for task workers | Cost/benefit upside-down for this codebase. |
| security-privacy-3 | On-device anti-spoof/replay for speaker-ID | Direction right, but premised on speaker-ID being live-authoritative; feasibility premise doesn't hold yet. Revisit when enrollment lands. |
| cross-platform-3 | Thermal/big-cores-only throttling on-device | Python half feasible (folds into `device-adapt-2`); the Dart/Android half is not a simple change. |

---

## Coverage gaps — what this audit did *not* cover (audit-next)

The completeness critic flagged dimensions worth a follow-up pass:
- **Model lifecycle & supply-chain integrity** — how multi-GB weights reach a
  device, checksum/signature verification, cache layout, version pinning. (Ties to
  the manifest guardrail above.)
- **Power / thermal / sustained-load budget** — battery drain + duty cycle of an
  always-listening loop were not modeled (only latency/tok-s). Named a HARD design
  constraint in `target_architecture.md`.
- **Data-at-rest & key management** — no encryption/keystore/SQLCipher story for the
  memory tier or cloud API keys (D8 open).
- **Multi-user / bystander privacy** — an always-listening mic captures non-owner
  speech; remote barge-in calls a global `cancel_all()` with no session scoping.
- **Observability/telemetry privacy, memory retention, failure-mode UX, and
  multilingual breadth** — each a load-bearing subsystem not audited here.

**Unverified claims to retire with measurement:** the specsim per-device latency
table is self-labeled "modelled estimates"; **the phone/llama.cpp path has never
been measured on real hardware**; the open-speaker barge-in P1 is field-unvalidated
(zero real LLM cancellations recorded). Suggested: run
`python -m tools.bench --profile phone` (and `cpu_laptop`) on real/throttled
hardware, and a live overlapping-speech barge-in A/B on the bare speaker after
speaker-ID enrollment.

### Alternatives deliberately *not* adopted (with rationale on record)
End-to-end speech-to-speech (Moshi/Qwen2.5-Omni/Gemma-3n-audio) and full-duplex
frameworks (Pipecat/LiveKit Agents/Wyoming) remain out of scope — the cascaded
pipeline wins on low-spec memory bounding + the clean §9.7 text seam. Worth a
written **decision record** (`routing-cascade-8`) rather than leaving it implicit.
Also unexplored and worth a future spike: MLC-LLM (Vulkan/Metal on phones),
ExecuTorch/ONNX-Runtime-GenAI, NPU backends (QNN/CoreML ANE/NNAPI) — the lever most
likely to move the phone-class latency cliff — and a **personal-LAN-edge-server**
topology (home mini-PC brain + phone thin clients over mDNS/WebRTC, cloud-free).
