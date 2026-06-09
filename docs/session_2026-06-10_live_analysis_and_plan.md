# Live-session analysis + fix plan — 2026-06-10

From the live run `run-20260609-234435` (self-interrupt) and `run-20260610-003800`
(post-mic-bump). Owner reported four problems; a 6-agent fan-out analyzed each
against the trace + code, synthesized a dependency-ordered plan, and an
adversarial pass verified the sequencing. **Deliverable = this plan. No code
changed yet.**

## The four problems → root causes

| # | Owner report | Root cause (confirmed in code/trace) |
|---|---|---|
| 4 | "speaker volume changing constantly" | **Per-sentence LibriTTS VITS output is un-normalized.** `_synthesize()` (`sherpa.py:1963`) streams raw chunks; `write()` (`:1795`) only resamples 22050→44100, never scales. Each sentence couples a *different* echo amplitude. |
| 1 | "interrupted itself once" | **Downstream of #4.** `_playback_floor_rms` resets to 0 per reply (`:1787`) and bootstraps on the first block; a quiet sentence → low floor → the 12 dB residual gate *admits* an onset echo → self-interrupt. `resid_floor` swung **~20–90×** (0.0002↔0.018) across the run. |
| 2 | "needed to scream and it still didn't stop" | **Two causes.** (a) Downstream of #4: a loud sentence raises the floor → real talk-over rejected. (b) A *distinct* DTD-mean-contamination bug: at the miss (`00:46:15`), `z_resid=0.00 / D=0.00 / fired=False` — the loud sentence had been absorbed into the DTD residual chart baseline, so the talk-over no longer stood out. |
| 3 | "referring to old interactions after we moved on" | **STT garble + no topic reset.** `collect_recent_turns` (`core/conversation.py:63`) returns the last 6 turns chronologically with *no topic-change detection*; after "Start again" the old "volume" topic stayed in context. Fed by garbled STT ("Start disorient", "Continuous story") that the LLM rationalizes against stale context. SenseVoice 2nd-pass is OFF (`config.local asr_final_backend=""`, though the model is on disk). |

**The causal spine:** un-normalized TTS level (#4) is the **single upstream root of
the barge instability (#1, #2a)** — the gate and the DTD charts both *learn against*
the playback echo, and that echo keeps moving, so there is no stable operating
point. The owner's instinct to keep tuning barge thresholds is calibrating to a
moving target. #2b (DTD-mean) and #3 (context/STT) are separate tracks.

## The plan (dependency-ordered — fix the source, then measure, then the rest)

**Track A — stabilize playback, then barge (do in order; measure between):**
1. **[small] Normalize per-sentence TTS output to a fixed target RMS** before the
   playback FIFO (in/after `_synthesize`/`write`, reusing the `apply_gain_soft_limit`
   pattern in `core/audio_frontend.py`). Fixes #4 directly and stabilizes the echo
   floor that #1/#2a depend on. *Validate:* a probe logging per-sentence pre/post
   RMS while replaying the run; the floor should stop swinging.
2. **[trivial — HARD GATE] Re-measure** the echo floor + DTD z-distributions on a
   *normalized-playback* recording **before touching any barge parameter.** Confirm
   `resid_floor` jitter drops to ~1–2× (AEC residual noise) and a real talk-over
   clears the gate. *Do not tune `dtd_residual_floor_margin_db`/`K` until this passes.*
3. **[medium] Fix DTD residual-chart mean contamination** so a loud later sentence
   in one reply cannot absorb a talk-over into the baseline (the `z_resid=0.00`
   miss). *Validate:* on the step-2 recording, the known talk-over frames produce
   `z_resid≫0` and `D>K`.

**Track B — STT + context (independent of barge):**
4. **[trivial] Re-enable the guarded SenseVoice 2nd-pass** — flip
   `config.local.json asr_final_backend "" → "sense_voice"` (model already on disk;
   guarded by `asr_final_min_sec=1.0` + the landed `core/asr_text.py` agreement-guard
   for short-clip hallucination). Corrects the long-utterance garble that feeds
   confabulation. *Validate:* the recorded-voice replay tests + re-run the same clips.
5. **[small] Topic-reset for the recent-conversation block** on reset phrases
   ("start again", "never mind", a new explicit request) so stale topics don't
   bleed into a new one. *Validate:* a unit test feeding stale turns + "start again"
   and asserting the block resets.

## Sequencing & the one rule
**Step 1 first, then the step-2 measurement gate.** Tuning barge thresholds on
today's 20–90× swinging floor is calibrating to noise — exactly the loop we've
been in. Track B (4, 5) can proceed in parallel since it doesn't touch the barge
path. Adversarial verdict: trustworthy + correctly sequenced; **first action =
step 1 (normalize per-sentence TTS), then measure before any barge knob.**

## The honest bigger picture
The mic gain bump helped levels (voice now `raw 0.02–0.05`) but gain scales voice
*and* echo equally — it doesn't change their ratio. Stable open-speaker barge
ultimately wants either a clean echo level (this plan: normalized playback) *and/or*
a level-independent discriminator (reference-coherence, already in the code as a
fallback — step 3 could lean on it more), or the close AT2020 mic. This plan makes
the current architecture work; it doesn't remove the open-speaker hardness.
