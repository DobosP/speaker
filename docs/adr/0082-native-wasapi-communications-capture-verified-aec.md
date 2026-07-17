# 0082 — Native WASAPI communications capture with OS-verified AEC; stop-word floor; barge telemetry

Status: accepted 2026-07-17 (Windows box session; roadmap Phase 2, `docs/2026-07-17-performance-roadmap.md`; supersedes the ADR-0019 fail-closed *blocker* by supplying the verified API it demanded — ADR-0019's fail-closed *principle* is retained and strengthened).

## Context

ADR-0013 chose OS-level echo cancellation + text-authority word-cut as the
open-speaker barge architecture. Its Windows leg assumed
`sounddevice.WasapiSettings(communications=True)`; ADR-0019 proved that API
never existed and mandated fail-closed until "a concrete, hardware-verifiable
API that opens an AEC/NS communications stream" was provided. Research for this
ADR established a second, deeper problem: tagging a capture stream
`AudioCategory_Communications` is not evidence of anything — Windows maps
categories to driver-defined processing modes and **silently falls back** to
default processing when the driver ships no Communications APO. Even a working
category API would not have satisfied ADR-0019's bar.

## Decisions

1. **Native client (`core/engines/_wasapi_comm.py`).** A comtypes
   `IAudioClient2` capture client (default endpoint, shared mode,
   event-driven) sets `AudioClientProperties.eCategory =
   AudioCategory_Communications` *before* `GetMixFormat`/`Initialize` (the
   category changes format negotiation — MS guidance), pumps
   `IAudioCaptureClient` on a single dedicated MTA thread, and presents the
   `_RecoveringInputStream` candidate contract (blocking `.read()` →
   `(ndarray, overflowed)`, PortAudio-shaped error codes) so the existing
   capture-recovery machinery works unchanged. Delivered audio is mix-rate
   mono float32; the engine's existing resampler chain bridges to the model
   rate. All COM GUIDs/vtables were verified against win32metadata (the MS
   docs method tables are alphabetical, NOT vtable order —
   `IAudioEffectsManager` is Register/Unregister/Get/Set).
2. **The verified-route contract is the OS effects snapshot.** Post-open, the
   client queries `IAudioEffectsManager::GetAudioEffects` (Windows 11 build
   22000+) and the route is verified **only** when
   `AUDIO_EFFECT_TYPE_ACOUSTIC_ECHO_CANCELLATION` is present and ON for the
   actual stream — the Windows analogue of the Linux PipeWire echo-cancel
   node probe. `verify_required_os_echo_route`'s Windows branch now runs this
   probe (the `"wasapi-pending"` placeholder is gone); readiness'
   "OS echo-cancel route (WASAPI Communications)" check does the same;
   `_capture_voice_comm_applied` is set only when the LIVE stream carried the
   proof. No AEC effect / Win10 / probe failure ⇒ raw-capture fallback with
   the word-cut route unverified (basic ASR keeps working; word-cut fails
   closed — ADR-0019's principle, now with a real probe behind it).
3. **Measured on the dev box (2026-07-17):** Realtek mic array, Windows 11
   build 26200 — communications stream opened at 2ch float32 48 kHz with
   **AEC=ON, NS=ON, beamforming=ON** (3 effects); 500 ms of real-time capture
   verified through the pump. The ADR-0013 "Teams path" premise HOLDS on this
   hardware, now with OS-attested evidence instead of assumption.
4. **Stop-word interrupt floor (roadmap item 14).** The already-wired
   sherpa-onnx `KeywordSpotter` fast-path becomes the guaranteed interrupt
   floor: `tools/setup_models --kws` fetches a pinned zipformer KWS model and
   generates a recall-biased keywords file (stop/stop talking/stop speaking/
   be quiet/wait/hold on); "wait"/"hold on" map to the `stop` control in the
   commands map. The existing playback gate (KWS only behind a live in-app
   AEC or a verified OS echo route — fails closed) is unchanged and is the
   authority contract; an engine-side own-echo guard drops a spotted keyword
   while speaking when the currently-playing or recently-spoken sentences
   contain that word (whole-word match — the KWS analogue of ADR-0042's
   own-TTS-ambiguity rule). Deliberate bias: when the assistant's own reply
   contains a control word, a REAL user utterance of that word during
   playback is also suppressed — ambiguity fails toward not-cutting, exactly
   as ADR-0042 rules for word-cut STOP; the word-cut ASR path (with its
   speaker-authority machinery) remains the authority for those ambiguous
   cases. Committed config
   keeps KWS opt-in (keys empty); machine-local wiring only.
5. **Word-cut similarity telemetry (roadmap item 15a).** The per-reply
   word-cut funnel line now carries the speaker-similarity distribution
   (p50/p95/min/max/n) so live bundles can prove or refute the 2026-06-01
   hypothesis that owner talk-overs score ~0.15 in the echo domain and are
   silently rejected against the 0.30 accept threshold.
6. **Deferred-barge buffering (roadmap item 15b) is deliberately NOT built.**
   Feeding route-unverified playback-time audio into the next turn is
   "seeding normal ASR" under ADR-0072's fail-closed rule and sits in the
   dialogue-policy territory ADR-0071 defers until the physical pipeline is
   proven; with the native route verified on this box the discard branch is
   no longer operative here. Recorded in the backlog with the full contract
   analysis rather than landed blind.

## Consequences

- `comtypes>=1.4` becomes a Windows-only dependency.
- The Windows box can now run the ADR-0013 word-cut recipe
  (`capture_voice_comm=true`, `aec_enabled=false`) with startup verification
  instead of the ADR-0081 interim AEC3 config; the interim remains the
  documented rollback.
- v1 limits: default input endpoint only (a named selector fails the native
  path rather than risking a wrong-mic bind); Windows 10 has no effects
  framework ⇒ never verifiable ⇒ interim AEC3 stays the Win10 answer.
- The physical ADR-0013 live gate (talk-over batch, bare-"stop", silent
  control, Kokoro voice) remains owner-required; nothing here claims it.
