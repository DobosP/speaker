# Barge-in, rethought: reference coherence (volume-independent, zero-setup)

**Date:** 2026-06-02
**Status:** Phase 1 shipped (`core/engines/echo_coherence.py`, wired into the
sherpa engine). Phases 2-5 are roadmap.
**Supersedes for barge-in:** the loudness/level-margin gate
(`barge_in_output_margin_db`) and the CAMPPlus speaker-ID gate, both demoted to
fallback-only. Origin: the "rethink-barge-in" multi-agent workflow + the user's
brief — *"smart, NOT loudness; at any input I say with the same volume it should
work; any user with minimum setup (a language declaration, no data-mining); the
rest set up dynamically at runtime from the environment; still reliable."*

## Why loudness was always going to fail

Without cancelling the assistant's own voice out of the mic, *some* threshold has
to separate "the user" from "the assistant's TTS leaking back into an open mic."
Loudness is the crudest such threshold — and it is level-dependent by definition,
so it either self-interrupts (the assistant's echo clears the bar — measured: a
level-only gate self-interrupted 134× in one session) or misses quiet barges. It
also can't be "the same at any volume," which is exactly what the user asked for.

The voice-identity gate (CAMPPlus embedding) was the other lever, but the
embedder is **unreliable on this user's mic/voice** — it scored the user's *own*
voice ~0.15 against an enrollment from other runs — and it imposes an enrollment
ritual the brief forbids.

## The idea: ask coherence, not loudness

The assistant always knows **exactly** what it is playing (its own TTS samples).
So instead of *"is the mic louder than playback?"* we ask the structural
question:

> **"does the mic contain sound the playback reference does *not* explain?"**

measured by the **magnitude-squared coherence** between the time-aligned TTS
reference `r` and the mic `x`, over the voiced band (300–3400 Hz):

```
C(f) = |Sxr(f)|² / ( Sxx(f) · Srr(f) )          ∈ [0, 1]
```

The decision signal is the **energy-weighted incoherent fraction**
`Σ w(f)·(1 − C(f))` over the band (`w` = normalised mic PSD). Barge-in fires when
that fraction sits a margin (`coherence_margin_delta`) **above** a runtime-learned
echo baseline `B`.

### Why this meets every requirement

- **Volume-independent — by algebra, not tuning.** `C` is invariant to scaling
  `x` or `r` (the gains cancel in the ratio), and the weights `w` are normalised,
  so they cancel a gain on `x` too. The *same utterance at any volume* yields the
  *same* incoherent fraction. The automated test asserts this across a 100×
  scaling (`tests/test_echo_coherence.py::test_decision_is_invariant_to_uniform_volume_scaling`).
- **No self-interruption — structurally.** The assistant's own TTS is, by
  definition, fully explained by the reference → incoherent fraction ≈ 0,
  *regardless of how loud it's played*. The 134-self-interrupt failure is gone by
  construction, not by a threshold (`test_echo_only_never_fires_at_any_playback_gain`).
- **Fires even when the user is quieter than the echo** — the case loudness
  fundamentally can't catch (`test_user_over_echo_fires_including_when_quieter_than_echo`,
  user at 0.6× the echo level).
- **Zero setup.** The reference is the assistant's *own output*, which the engine
  already produces — no enrollment, no voiceprint, no per-user data. The only
  user input is the language tag (which ASR/TTS already require).
- **Runtime-dynamic.** The echo **delay** is estimated continuously by
  cross-correlation (median-tracked over recent echo-only frames); the echo
  **baseline** `B` is learned online by an asymmetric EWMA on echo-only frames
  (it tracks the room's reverb/noise floor). Nothing is hand-tuned per
  environment except the one sensitivity knob below.
- **Reliable / never worse than today.** Coherence is the *primary* gate but is
  layered on the existing VAD + 0.2 s min-speech + one-barge-per-run latch +
  suppress window. When it **abstains** (no reference yet at session start, or a
  TTS silence), the legacy identity/level gate still runs — so behaviour degrades
  to "today," never below it.

## What shipped (Phase 1)

- `core/engines/echo_coherence.py` — `EchoCoherenceDetector`: reference ring
  (thread-safe), scipy cross-correlation delay estimate, Welch coherence + PSD
  over the voiced band, online baseline, `decide() -> True | False | None`. Pure
  numpy/scipy, **zero new installs**.
- `core/engines/sherpa.py` — reference tap in the playback `write()` closure;
  detector built in `_build()`; `reset()` on idle; `_looks_like_user()` consults
  coherence first and only falls back to identity/level when it abstains.
- `SherpaConfig` knobs: `coherence_barge_in_enabled` (default **True**),
  `coherence_voiced_band_hz`, `coherence_margin_delta` (default **0.12**),
  `coherence_ring_ms`, `coherence_max_delay_ms`.
- `tests/test_echo_coherence.py` — the property proofs above (no audio device).
- `tools/echo_probe.py` — now reports the live coherence calibration block.

## Calibrating on real hardware

The one sensitivity knob is **`coherence_margin_delta`** (how far above the
learned echo floor counts as user voice). Lower = more sensitive (fires on a
quieter barge, risks reverb false-fire); higher = stricter.

1. **Measure the echo floor** (assistant talks, you stay silent):
   ```
   python -m tools.echo_probe --sentences 4
   ```
   In the `coherence` block: `coherence_fired_on_own_tts` should be **0** and
   `headroom_p95` comfortably **positive**. If it fired / headroom is negative,
   raise `coherence_margin_delta`.
2. **Test a real barge** — talk over a long answer:
   ```
   python -m core --engine sherpa     # ask "tell me a long story", then talk over it
   ```
   Doesn't stop → lower `coherence_margin_delta`. Cuts itself off while you're
   silent → raise it.

## Honest limits (what this is and is not)

- It is a coherence **detector**, **not a full AEC**. No real AEC library
  (`webrtc_audio_processing`/`webrtcvad`) imports in this environment without a
  native build — re-confirmed by the workflow. The detector decides *whether*
  user voice is present (all barge-in needs; ASR is never fed during playback),
  but does **not** hand ASR an echo-free signal.
- **Nonlinear / clipping speakers at high OS volume** violate the linear
  coherence model (harmonic distortion lowers coherence even with no user). The
  learned baseline absorbs some of it; in the worst case it falls back to the
  level margin — i.e. no better than today, but never worse.
- **Reference/mic time-alignment** is the fragile dependency, especially with the
  intermittent AT2020 (USB clock drift). Drift shows up as low coherence
  (observable, not silent) and pushes toward the fallback.
- If the **OS applies post-mix DSP** after the `write()` tap, the tapped
  reference no longer matches what hits the speaker — an inherent app-level (vs
  OS/hardware) AEC limitation.
- Dropping speaker-ID from the gate means a **TV or bystander talking over the
  assistant can barge** — accepted per the brief ("no data-mining"); an optional
  "ignore the TV" soft veto is Phase 5, not in the shipped cut.

## Roadmap (Phases 2–5)

2. **Coarse echo subtraction + VAD-on-residual confirmer + true ERLE trust-switch
   + double-talk freeze.** Two independent confirmers; `ERLE = 10·log10(E{mic²}/
   E{residual²})` becomes the formal trust switch (≥20 dB → coherence-primary;
   <10 dB → widen margins / loudness fallback). Emit ERLE/coherence/delay to
   `run summary.json` for replay-regression.
3. **Adaptive timing + observability + phone-profile CPU.** `min_speech` and the
   margin self-tune on a false-barge-per-hour counter; validate the per-frame
   FFT/FIR cost inside the 0.1 s block budget under `--device phone`.
4. **Transport-agnostic shared trigger.** Factor the decision into a module both
   the sherpa and LiveKit engines call (the browser/LiveKit path already gets
   AEC3 for free → pass-through there) so every engine makes the identical
   decision. Optional NLMS residual-SNR confirmer once ERLE ≥ 20 dB.
5. **(Optional) WebRTC AEC3 backend** behind the same interface if a usable wheel
   ever lands (pure upgrade; coherence stays the always-available fallback) +
   optional default-OFF CAMPPlus post-hoc veto / DOA for "ignore the TV."
