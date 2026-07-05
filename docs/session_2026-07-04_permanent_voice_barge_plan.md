# Permanent voice + barge plan — the honest, history-grounded version (2026-07-04)

Valid until: superseded by a live-validated decision — then treat as history.

Produced by a multi-agent study of the FULL history (9 ADRs, ~30 session docs, 15
memory notes, all live bundles) + a synthesized plan + **three independent
adversarial verifiers, all of which marked the synthesized plan `sound: False`.**
This document keeps only what survived verification.

## The hard truth (why weeks of attempts keep failing)

Open-speaker barge-in on a bare nonlinear laptop speaker is a **fundamental
physics wall**, not a tuning bug. Every acoustic approach has been tried and fails
for the same reason — the user's voice and the assistant's own echo **overlap** in
every single-mic acoustic feature on a nonlinear, clipping speaker:

| approach | tried | why it failed |
|---|---|---|
| level gate on residual | 2026-06-01 | nonlinear echo spikes overlap real-voice levels |
| coherence-primary | 06-07 | echo is ~88% incoherent too (overlaps voice ~1.0) → self-interrupt |
| AdaptiveDTD (z-fusion, residual-weighted) | 06-08, ADR-0004 | worked *only* on ALC285+DTLN where the residual kept the user; breaks under any masking canceller |
| NLMS linear AEC | 06-17 | ~0 dB ERLE on the nonlinear speaker, **diverges** → ruled out (ADR-0006) |
| DTLN deep AEC | 06-18 | +6.2 dB ERLE but a **spectral masker** — erases the near-end user from the residual |
| WebRTC APM (NS on) | 06-21, ADR-0006 | same: always-on NS suppresses the user in the residual the DTD weights 1.0 |
| word-gated duck-confirm | 07-02, ADR-0011 | correct idea, but the duck still triggers off the same broken acoustic gate |
| **this session:** raw-mic DTD + coh-veto off + loose duck + APM | 07-04 | on DTLN → **misses**; on APM → **self-interrupts** (raw mic contains the echo; the veto guard was removed) |

**The two "novel" fixes the synthesis proposed were demolished by all 3 verifiers:**
- *Self-echo "explain-away"* = AEC cancellation rebranded (mic − predicted echo IS
  a residual). A one-shot startup basis is strictly **weaker** than DTLN, which
  already failed; and it's a frozen per-session value (ADR-0012 violation in
  substance). **Rejected.**
- *Self-voice timbre fingerprint* = repeats the speaker-ID-under-echo failure
  (owner-ID scored the owner's OWN voice ~0.15). The echo is the TTS voice
  *distorted* by the same speaker, so a clean fingerprint won't match; and a
  talk-over block is a *mixture*. **Rejected.**

## What actually survives (the grounded levers)

1. **The upstream root nobody fixed: the input is broken before any DSP.** The
   built-in mic ADC is hot (+30 dB, PipeWire re-applies on resume) and **clips
   11–16% of blocks** on Linux (peak 1.09) — nonlinear distortion injected *before*
   AEC, which defeats AEC3 and inflates the echo peaks that self-interrupt. On
   Windows the opposite: capture is 30–70× too quiet. The engine only *warns*.
   **Fix the OS capture gain per device (measurable), and it must be enforced.**

2. **OS-level voice-comm capture is why Teams + your own Android app sound clean
   on this exact laptop.** The desktop opens the RAW mic (no OS/driver AEC), unlike
   the Android app (`voiceCommunication`) and Teams. Routing capture through the
   OS voice-comm path (Linux `module-echo-cancel`, Windows WASAPI communications)
   shrinks the nonlinear echo **in hardware, before any detector runs**. This is
   the single biggest lever — BUT it must **replace** the in-app APM, not stack
   (double-AEC + triple-AGC = the pumping we just saw), and it may itself suppress
   the user, so it must be *measured*, not assumed.

3. **The ADR-0011 word gate is the right hard-cut authority for open-speaker —
   but do NOT make it the SOLE authority yet.** The fatal circular dependency all
   verifiers caught: if you delete the acoustic hard-cut, the *only* way to cut is
   ≥2 clean transcribed words in the 1.5 s ducked window — and confirm-window STT
   at low mic SNR is the documented, **unproven** ceiling. Deleting hard-fire would
   trade self-interrupts (fixed) for **missed cuts** (talk-over never cuts, just
   ducks + restores). Keep the acoustic hard-cut as a scoped fallback.

4. **Voice swing:** the output is clean per-sentence but perceived loudness jumps
   because `output_leveler` targets broadband RMS (not K-weighted) and `tts_markup`
   switches voices per sentence (warm/bright/deep differ in timbre). Fix with
   **per-voice loudness offsets measured once offline** + K-weighted leveling.
   (The 7 kHz lowpass muffling is already removed.)

## Immediate cleanup (revert this session's regressions)

My 07-04 barge changes made it worse and should be reverted before anything else:
- the `coh_veto` disable under `_resid_blind` (removed the echo guard → the APM
  self-interrupt), and the loose-duck trigger (→ volume pumping). The raw-mic DTD
  re-source collapsed the fused DTD to a bare loudness gate.

## The plan (measurement-first — no more speculative DSP)

- **Phase A — stabilize:** revert the coh-veto disable + loose duck; set the OS mic
  gain so it stops clipping (Linux) / raises SNR (Windows). *Gate: a clean bundle,
  no self-interrupt, no clip warnings.*
- **Phase B — the decisive experiment:** run the OS voice-comm capture path with
  the in-app APM OFF, and **measure at the mic**: (1) is STT clean on a real
  talk-over? (2) does the confirm-window recognizer emit ≥2 words on a ducked
  talk-over? These two answers decide the entire architecture — and neither can be
  judged by the loopback autotest (physics), only a human at the bare speaker on a
  quiet box. *Gate: measured word-yield + STT WER on real recordings.*
- **Phase C — commit the architecture** that Phase B proves (OS-capture + word-gate
  with acoustic fallback), scoped to open_speaker, all thresholds bounds-only.
- **Phase D — voice:** per-voice loudness offsets + K-weighted leveler.
- **Phase E — ADR-0012 conformance + a fresh-clone live run on a second machine.**

## The uncomfortable bottom line

There is no clean acoustic silver bullet — the space is exhausted. The realistic
permanent fix is **capture-path + OS-AEC + word-gate**, and it can only be chosen
by **measuring at the real mic**. Every prior fix (including mine) looked green
headless and failed live; the loopback harness structurally cannot judge nonlinear
open-speaker echo. So the next step is a *measurement*, not another commit.
