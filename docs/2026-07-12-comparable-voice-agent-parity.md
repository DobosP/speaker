Valid until: the next voice-runtime architecture or live-validation decision — then treat as history.

# Comparable voice-agent parity audit

This audit compares the Python/Linux voice runtime derived from `c0a26cb`, with
the follow-up behavior in ADR-0055 through ADR-0057, against current
primary-source behavior from LiveKit Agents, Pipecat, and Kyutai Moshi. It is a
feature/evidence comparison, not a claim that different
hardware, hosted services, or end-to-end speech models are interchangeable.

Primary references checked 2026-07-12:

- [LiveKit turn detection and interruption handling](https://docs.livekit.io/agents/logic/turns/)
- [Pipecat speech input and turn detection](https://docs.pipecat.ai/pipecat/learn/speech-input)
- [Pipecat Smart Turn v3](https://docs.pipecat.ai/api-reference/server/utilities/turn-detection/smart-turn-overview)
- [Pipecat user-turn strategies](https://docs.pipecat.ai/api-reference/server/utilities/turn-management/user-turn-strategies)
- [Moshi paper](https://arxiv.org/abs/2410.00037) and
  [official repository](https://github.com/kyutai-labs/moshi)

## Evidence matrix

| Capability | Comparable-project baseline | `speaker` evidence | Status |
|---|---|---|---|
| Local cascaded voice loop | LiveKit and Pipecat support VAD→STT→LLM→TTS pipelines with explicit lifecycle events. | Typed control plane in `always_on_agent/`; sherpa capture/ASR/TTS in `core/engines/`; MiniCPM fast plus Gemma main per ADR-0020. | Architecture parity; final live acceptance pending. |
| End-of-turn detection | LiveKit recommends an acoustic/semantic detector over VAD; Pipecat defaults to local Smart Turn v3 plus VAD. | Adaptive endpoint policy, learned session-pause floor, continuation merge, lexical detector, and a compatible Smart Turn v3.2 ONNX implementation in `core/endpointing.py`. | Concept parity. Smart Turn is optional/default-off and its live floor-lowering remains unvalidated. |
| Interruption / barge-in | Both frameworks stop pending bot audio when a user turn starts; Pipecat offers minimum-word strategies and LiveKit offers adaptive interruption handling. | FIFO cut, generation cancellation, four-novel-word generic floor, canonical STOP, playback state, and enrolled-speaker authority (ADR-0045/0053). | Stronger deterministic/identity policy headlessly; physical bare-speaker result remains red. |
| Backchannel and false-interruption recovery | LiveKit distinguishes adaptive/false interruptions and can resume; Pipecat separates raw VAD frames from committed user-turn strategies. | Playback-time and ordinary-turn admission are separate; `core/resume.py` and post-barge response-only handling recover bounded tails without granting tool or memory authority. | Headless parity; ear-grade continuity pending. |
| Heard-history correctness | LiveKit truncates history to the portion heard before interruption. | Terminal sink receipts, playback generations, and `TTS_REQUESTED` ownership prevent unheard text entering committed assistant history (ADR-0029/0051). | Parity with stronger deterministic receipt tests. |
| Tool/cancellation safety | Modern frameworks expose interruption-aware long-running tools. | Epoch/generation fences, bounded provider bulkhead, cancel events, no duplicate failed-provider retry, and causal trace checks. | Parity for tested local tools; remote multi-participant paths are not part of the Linux-v1 proof. |
| Noise, echo, and speaker authority | Frameworks integrate VAD and optional noise/voice cancellation; identity policy is application-specific. | PipeWire EC route, GTCRN, calibrated pre-gain evidence, DTD, speaker embedding, and current-signal AGC provenance. | Broader local acoustic policy, but fresh v5 enrollment/live validation is mandatory. |
| Model locality and role routing | Frameworks are provider-neutral and commonly mix fast and strong models. | MiniCPM5-1B Q8 handles ordinary local text/voice; Gemma remains local complex/vision; cloud is explicit opt-in (ADR-0020). | Implemented; final clean-SHA real A/B still required. |
| Deterministic evaluation | LiveKit documents a test framework; Pipecat exposes frame/metrics observers. | 14-scenario causal conversation gate, strict private recorded replay, APM/DTD tests, synthetic injection, run receipts, identity/provenance hashes, clean-revision model A/B, and the fail-closed WER/first-audio/barge policy in ADR-0055. | Strong headless coverage; final-revision and physical reruns remain pending. |
| Native end-to-end full duplex | Moshi models user and assistant audio in parallel streams and reports about 200 ms practical latency, but the official PyTorch path needs roughly 24 GB GPU memory and its bare CLI has no echo cancellation. | Cascaded local stack on a 16 GB laptop GPU; explicit STT text enables tools, privacy classification, replay grading, and MiniCPM routing. | Deliberate architectural tradeoff, not feature parity. Replacing the stack with Moshi is a separate model/product decision. |

## Current conclusion

The Linux/Python control-plane architecture includes the current mainstream
concepts: local semantic/prosodic turn seams, interruptible streaming output,
false-interruption recovery, heard-only history, cancellation fencing, hybrid
model routing, and deterministic replay. It is therefore reasonable to call the
architecture comparable.

It is not yet reasonable to call the user experience comparable or complete.
That stronger claim still requires all of the following on the final behavioral
revision:

1. A clean, provenance-valid MiniCPM-versus-Gemma production-hybrid A/B.
2. Fresh v5 enrollment on the actual PipeWire EC capture domain.
3. Bare-speaker quiet speech, low sensitivity, self-echo, generic override,
   mid-thought pause, reply-tail continuity, and STOP acceptance.
4. A final-revision autonomous delay/speaker report exercising ADR-0055's
   fail-closed WER, first-audio, self-interruption, cut-latency, error, and stuck
   checks.

Smart Turn deployment is a measured follow-up, not a substitute for those live
gates: enabling a new endpointer during the same barge-in acceptance run would
confound which change produced the result.
