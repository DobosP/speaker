# Improvement backlog

Priority queue. `[ ]` = open, `[x]` = shipped (see `changelog.md`).
P0 = correctness/blocker, P1 = high value, P2 = nice-to-have.

> Refreshed 2026-06-02 during the architecture/doc unification pass. Durable
> design rationale now lives in [`docs/unified_architecture.md`](../docs/unified_architecture.md);
> this file tracks only OPEN work. The session-bootstrap helper
> (`python -m tools.session_bootstrap`) reads the OPEN P0 items below.
>
> Truth-up 2026-07-10: `STATUS.md` owns the active Linux route and verification
> state. The phase narrative below is historical context inside an open live-A/B
> item. Desktop Windows Communications capture is not wired; it fails closed
> pending a verified implementation (ADR-0019).

## P0 — correctness / blocker
- [ ] **★★★ PERMANENT open-speaker barge + voice fix (2026-07-05) — the live-test
      thread. START HERE.** Conclusion after many live tests + a 3-verifier study:
      NO clean single-mic acoustic barge fix exists (see
      `docs/session_2026-07-04_permanent_voice_barge_plan.md` + memory
      `barge-voice-no-acoustic-fix-2026-07-04`; do NOT re-try the ruled-out acoustic
      approaches). This session's barge changes were REVERTED (Phase A: coh-veto
      restored, loose duck removed) — the tree is stable/green. NEXT, in order:
      **Phase B (measure first)** — resume the live APM+clean-mic test and MEASURE:
      (1) is raw STT clean on a real talk-over? (2) does the confirm-window
      recognizer emit ≥2 words on a ducked talk-over? These decide the architecture;
      the loopback autotest cannot judge them (live human at the bare speaker only).
      Before any live run: RE-PIN the mic ADC (`amixer -c 1 sset Capture 52%` +
      `Internal Mic Boost 0`, held against PipeWire's +30 dB reset — see
      `tools/autotest/ota_setup.py`). Then **Phase C** capture path: make OS
      voice-comm capture the open_speaker default + enforce it per-OS (Linux
      module-echo-cancel; Windows remains blocked by ADR-0019) and
      the ADR-0011 word gate the hard-cut authority WITH the acoustic path as a
      scoped fallback (do NOT delete it). **Phase D** voice: per-voice loudness
      offsets (offline) + K-weighted leveler for the inter-sentence volume swing.
      All bounds-only (ADR-0012). Current machine-local state is in `STATUS.md`.
      **UPDATE 2026-07-06 (Linux live batch, run-20260706-231226):** Phase B
      (OS-capture `module-echo-cancel` + word-cut) was run live for the first
      *sustained* talk-over batch and **FAILED** — the assistant played a ~3-min
      story, the owner talked over it repeatedly, and it never cut. Bundle shows
      **ZERO multi-word ASR finals for the whole playback** (only 2 stray
      single-word `'AND'` echo fragments), so the text-authority word-cut had
      nothing to fire on. The near-end user voice did not survive capture during
      playback → ADR-0013's "clean near-end during playback" premise did **NOT**
      reproduce on a real batch (see ADR-0013 2026-07-06-evening addendum).
      Pipeline otherwise healthy (pre-playback STT clean, `clip=0.0%`, 2316 tests
      green); **NOT** a regression from the Windows session. **→ ROOT CAUSE FOUND
      + FIXED same night (branch `fix/barge-wordcut-live-diagnostics`, 2341/24
      green):** `_barge_word_cut_step` consulted `vad.is_speech_detected()` but
      NOTHING fed the VAD during playback (the word-cut branch `continue`s before
      the acoustic path's `accept_waveform`) → VAD frozen quiet → recognizer
      never fed → zero words, deterministically. Fixed (step feeds the VAD every
      block) + debounced burst reset (`barge_word_cut_reset_quiet_blocks=3`).
      Also shipped: word-cut funnel telemetry (trace/burst-reset/near-end/funnel
      lines), kill-safe WAV recording (survives SIGTERM/SIGKILL), SIGTERM→Ctrl-C
      shutdown bridge, doctor FAIL on missing module-echo-cancel, diagnose_run
      "Word-Cut Funnel (ADR-0013)" section. **NEXT — the decisive live re-run
      (needs the owner at the mic):** (1) load module-echo-cancel + set defaults
      (ADR-0013; `python -m tools.doctor` now FAILS if missing); (2) launch
      **with `--record`** (`.venv/bin/python -m core --engine sherpa
      --input-device pipewire --output-device pipewire --record` — the failed run
      had no `--record`, so no WAV existed at all; the config knob alone does
      nothing); (3) talk-over batch + bare "stop" + silent control; (4) score via
      `python -m tools.diagnose_run logs/runs/run-<id>.txt` → the Word-Cut
      Funnel section now says explicitly whether words were transcribed-and-cut,
      starved (`fed=0`), wiped by resets, or never survived the canceller
      (voiced windows vs floor) — i.e. it distinguishes tunable-vs-dead Phase B
      in one run. **→ RAN 2026-07-07 (run-20260707-002943): PASSED — first real
      word-cut CUT ever** (talk-over "I DON'T TO FEAR" → hard cut ~0.35 s,
      pre-roll answered; 5-reply funnel cuts=1 / false cuts=0 / own-echo ≈0
      words). Near-end survives the OS canceller; the 2026-07-06 failure was
      purely the VAD-starvation defect. STILL WANTED before default-on: a
      larger talk-over batch + bare-"stop" cuts + a deliberate silent-control
      story. NEW FOLLOW-UPS from the run (P2 unless noted): (a) short replies
      end before the 4-word floor fills → uncuttable except "stop"; consider a
      lower floor for short replies or stop-word emphasis; (b) playback-fed
      words are DROPPED at reply end when no cut fired → a tail talk-over loses
      its opening words; consider keeping NOVEL (non-own) words as pre-roll;
      (c) diagnose_run self-interrupt classifier stale-flags word-cut barges
      SUSPECT:NO-DTD → OVERALL FAIL (word-cut bypasses the DTD by design) —
      classifier truth-up; (d) TTS voice flips mid-story: `[voice:...]` applies
      only to the tagged first sentence (rest → sid 0) and unknown names
      (`gentle`) silently default → make the reply voice sticky + warn on
      unknown names; (e) SIGTERM→Ctrl-C bridge still not live-exercised (the
      2026-07-07 kill hit the bash wrapper; python died hard — kill-safe WAVs
      survived and are hereby VALIDATED). **OWNER-DIRECTED NEXT (2026-07-07,
      P1): STT quality on the Linux boot** — garbled raw finals all session
      ("TANE MADE LONG STORY", "CAN WE A NICE STORY ABOUT MY CAT BICKIE" — the
      latter caused an addressing INGEST miss that ate a real request). Joins
      the existing Windows STT P1; levers already on file: SenseVoice second
      pass tuning, prosody/endpoint thresholds, R14 Parakeet finalizer branch,
      and the run-20260707-002943 WAV pair as replay evidence.
- [ ] **Adopt the 2026-06-10 gap-analysis roadmap (45 verified findings, P0–P5).**
      `docs/review_2026-06-10_gap_analysis.md` — security/PII first, then real-time
      correctness (rc-2 _on_final off the audio thread DONE via turn_merge; rc-1
      wait_idle double-drain DONE; lm-3 recall sensitivity float DONE — all
      2026-06-10), then layered memory (no cross-session continuity, broken
      migration 002, dead knobs), smart routing (no quality axis, mute tier
      failure, dead KWS), cross-platform (installer omits scipy/soxr; phone
      profiles unprovisionable), remote/docs sweep. 8 owner decisions D1–D8 listed
      in the doc.
- [~] **★ WINDOWS-side self-interrupt (live 2026-06-08e, run-20260608-181250) —
      CASCADE HALF FIXED 2026-06-08f.** The runaway "two outputs back-to-back" is now
      broken on ANY device (merged `fix/echo-final-cascade` → main, 1416 green;
      handoff `docs/session_2026-06-08_echo_final_cascade_fix.md`). Three
      device-adaptive, additive, off-switchable layers — NO fixed magic-number
      thresholds (every bar RELATIVE to a LEARNED per-device floor): **L1**
      `_final_above_floor` drops a final at/near `max(_ambient_rms, _playback_floor_rms)`
      (a dB-above-LEARNED-floor margin via `loudness_admits`, never an absolute RMS),
      wired at the final-dispatch seam; fails OPEN until a floor is learned, so it only
      bites on AEC/open-speaker configs. **L2** `_in_post_speaking_refractory` (per-reply,
      stamped at both `_speaking→clear` sites) suppresses a re-fired barge on the
      cancelled echo tail; on the barge debounce ONLY, never the final seam, so a real
      barged-in request is never dropped. **L3** `agreement_guard` (core/asr_text.py)
      demotes short-clip SenseVoice hallucinations (`BEING`→`I.`) back to the streaming
      final. +30 tests; off-switch parity (`final_floor_margin_db=0`,
      `barge_in_refractory_sec=0`). **STILL OPEN — the FIRST self-interrupt itself
      (needs the mic):** the Windows AEC mis-calibration (stale pre-FIFO
      `aec_ref_delay_ms`). Run `python -m tools.echo_probe` (echo-only) on the Windows
      box → pick the ERLE-max `aec_ref_delay_ms` (don't assume 0), raise `dtd_k` if
      echo-only D approaches K=5.0; target self_interruptions=0. Replay
      run-20260608-181250.wav through echo_probe to iterate without re-talking. The
      cascade fix makes this miscalibration NON-catastrophic until it's done. Then
      live-tune `final_floor_margin_db` (6.0) / `barge_in_refractory_sec` (0.5) — watch
      the `echo_floor_rejected_final` metric in run bundles. Original analysis:
      docs/session_2026-06-08_live_session_self_interrupt.md (think=false latency +
      prosody endpointing both validated live; re-assess turn-taking after the mic fix).

## P1 — Windows live session findings (2026-07-02, owner-directed next to-dos)
> Context: live debug on the Windows/4090 box (runs 212109 → 224616) after merging
> R11/R06b + the asr-guard fix. Barge-in was rebuilt as the word-gated
> duck-then-confirm gate (ADR-0011, `feat/barge-duck-confirm`) and works live
> (talk-over + "stop" cut; pumping fixed). OWNER VERDICT ending the session:
> **audio quality still has the problem, and STT quality too** — these two are
> the explicit next to-dos.
- [ ] **★★ Audio output quality on the Windows box.** The box still synthesizes
      with Piper `en_US-libritts_r-medium` — `config.local.json` `tts_model` was
      never repointed at the adopted Kokoro (ADR-0010) because C: was 100% full
      (the `gemma3:4b` pull died mid-download; ~4 GB free after cleanup).
      **2026-07-06 update:** `python -m tools.setup_models --kokoro` now exists
      (fetches + auto-wires tts_model/tts_voices/tts_tokens/tts_data_dir/
      tts_lexicon; tests in tests/test_setup_kokoro.py). Fetch attempted this
      session but still DISK-BLOCKED: only ~1.0 GB free after temp+pip cleanup
      vs ~0.7 GB peak needed (owner declined the Docker-WSL prune for now).
      Fix: free ~1 GB more → run the one command → listen; then apply the
      voice-plan P2 items (per-device `tts_output_lowpass_hz` for the cheap
      open speaker, named-voice set). Until then "audio quality is bad" on this
      machine is EXPECTED — it is the old voice.
- [ ] **★★ STT quality on the Windows box (the conversation ceiling).** Evidence
      run-20260702-224616: raw finals "I'M NODDY" (→ "I'm not."), "NOTHING GOBOD
      BREATHED THERE AND TALKING ABOUT" (→ a fabricated cleanup), fragments
      answered. Mic captures at avg_rms ~0.0007 (30-70x below normal speech);
      digital `input_gain` (now 4.0 machine-local) cannot add SNR. Fix order:
      (1) raise the Windows OS mic level/boost toward ~-20 dBFS active median
      and verify on the next bundle WAV; (2) re-tune SenseVoice + prosody
      thresholds at the new level; (3) R14 Parakeet-TDT finalizer bakeoff;
      (4) redo speaker-ID enrollment on a quiet system (tonight's enrollment
      rejected the owner → `speaker_gate_input=false` machine-local until then);
      (5) re-enable the addressing/cleanup gates once finals are clean
      (input_gate/cleanup disabled machine-local tonight so garbled fragments
      weren't silently dropped).
- [~] **Barge follow-ups (post-ADR-0011).**
      - [x] surface `barge_in_duck` / `barge_in_confirmed` / `barge_in_unconfirmed`
        in `tools/diagnose_run.py` (2026-07-03, Linux boot): log-derived counts →
        text `--- Barge Confirm Funnel (ADR-0011) ---` (with a self-heal WARN) +
        `--json` `barge_confirm_funnel`; visible even on zero-hard-fire runs; tests
        in `tests/test_diagnose_run.py`. Adversarial-review hardening: anchored
        `_BARGE_DETECTED_PAT` to end-of-message so a confirm line whose transcript
        contains "barge-in detected" isn't miscounted (dead path, now robust).
      - [ ] consider raw-mic word-confirm + KWS hotwords in the confirm window.
      - [~] `dtd_coherence_echo_veto` default (True) vs word gate default (False):
        the interplay only bites in profiles that opt the gate ON, so the default
        pairing is coherent. DOCUMENTED, not flipped (a barge-gate change needs a
        live-mic A/B). Revisit per-profile when a profile enables the gate.
- [x] **Autotest `voice` tier un-broken for Kokoro (2026-07-03).**
      `tools/autotest/audio.py::synth_to_wav` hard-coded `cfg.model.vits.*` for the
      injected "user" clips → once Kokoro (ADR-0010) became default the native
      loader aborted ("Not a model using characters as modeling unit … --vits-lexicon").
      Now builds clip synth via the runtime's own Kokoro-aware
      `build_tts(SherpaConfig.from_dict(sherpa_cfg))`. Cable tier runs end-to-end again.
- [ ] **Autotest `voice` WER is synthetic-voice-artifact-dominated (2026-07-03).**
      Kokoro-synthesized "user" clips are OOD for the streaming zipformer ASR
      (every injected clip gets a spurious leading "And"; long clips collapse to a
      word) → the cable WER is a harness signal, NOT a human-STT measurement (real
      cable STT with real recordings ≈0.10 WER, memory `ota-stt-is-test-artifact`).
      Fix: inject **real** recordings via `--utterances DIR` (or a non-TTS user
      voice) so the tier yields a trustworthy WER; until then don't read cable WER
      as an STT verdict.
- [ ] **Barge-in cut needs a HUMAN talk-over on the Linux boot (2026-07-03).**
      `autotest voice --acoustics delay` (silent loopback): S2 self-interrupt clean
      (0 barge-ins during own reply, pass); S3 talk-over registered 0 cuts (fail),
      but CONFOUNDED by the digital-loopback caveat — loopback echo is loud and the
      config `aec_ref_delay_ms=40` mismatches the loopback's adapted ~350 ms, so the
      residual is echo-heavy and the DTD fires on echo (coherence veto correctly
      rejected 4). Not a trustworthy barge-miss signal. A real human talk-over on
      real hardware is required to verdict the cut (the loopback stress-tests the
      echo veto, not the cut). `aec_ref_delay_ms` stays echo-probe-calibrated per
      ADR-0005 (do NOT set it from a loopback run).
- [ ] **★★★ REAL-USAGE FORENSICS (2026-07-04): the STT/barge bottleneck is the
      AEC/APM pipeline, NOT the mic — partly REFUTES the "raise mic level" STT
      to-do.** Analyzed 7 real recordings (trace + an A/B replay); see
      `docs/session_2026-07-03_*` + the real-usage report artifact. Findings:
      - **`aec_ref_delay_ms` is hard-set to 40 ms (19 ms on Windows) on EVERY real
        session, but the measured echo delay is 106–220 ms (corr ≈0.15).** The
        canceller looks in the wrong place → echo not removed → always-on NS
        over-corrects → the ASR + DTD read a mangled signal. This ADR-0005
        violation is the systemic root. FIX: echo-probe-calibrate per machine
        (est ~100–220 ms here) or enable `aec_auto_delay`. Config ships static 40 ms.
      - **The raw mic is FINE.** The heartbeat `avg_rms≈0.0017` is silence-diluted
        (speech ~1.6% of a 23-min session); measured active-speech RMS is 0.045 @
        30.5 dB SNR (232506: 0.070 @ 36.8 dB) — comparable to owner enrollment.
        So digital mic gain is the WRONG lever for STT; re-check the Windows finding
        the same way (active-speech RMS, not the silence-diluted average) before
        chasing the Windows mic level.
      - **A/B PROOF:** the same streaming ASR on the same audio — live (post-APM)
        vs replay of the raw pre-APM mic (`core --engine replay`, no AEC/NS) —
        recovered a full continuous narration the live path shredded into fragments
        (`THE MERE STORY`/`ABOUT`/`MY CAT BIKI`/`MEAN` → one continuous cat-story
        sentence). The audio is intelligible; the pipeline discards it.
      - **Barge is never calibrated on real usage:** 2–5 real talk-overs
        rejected-while-speaking per Linux session (missed), vs the Windows 181250
        self-interrupt cascade (12 fires on own echo). Both downstream of the broken
        AEC. Ties to the open `_apm_owns_ns` residual P1.
      - **Cleaner fabricates** confident wrong sentences from 2–3 garbled words
        (`LIKE A QUESTION`→"And did you I could pressure in…"). Gate the LLM rewrite
        on raw length/agreement so it can't expand noise into a fabricated request.
        + TTS DC offset ~0.05 on every sentence + up to 14 underruns.
      Fix order (IMPLEMENTED 2026-07-04, branch feat/auto-calibrated-audio-pipeline,
      ADR-0012, all runtime-self-calibrating / no hard caps; suite 2290 green):
      - [x] (1) AEC ref-delay measured on-device by normalized cross-correlation
        (`AecDelayCalibrator`); aec_ref_delay_ms demoted to a seed. VALIDATED on
        run-20260702-004345 (40→~120 ms). config.local.json 40 ms override removed.
      - [x] (2) relaxed-NS ASR tap under `_apm_owns_ns` (second APM, ML NS off) feeds
        the streaming recognizer + barge-confirm AND the offline 2nd-pass decode (via
        a parallel NS-off `asr_seg` threaded through _enqueue_final/_final_worker/
        _finalize_and_dispatch); the echo-floor + speaker-ID gates keep the NS-on
        `seg`. (2nd-pass completion added after the adversarial review caught that the
        LLM-facing final was still decoded from NS-on audio.) INERT on the current
        dtln config -- **needs a live-mic A/B on the open_speaker (apm) profile** to
        confirm STT recovery + no self-interrupt regression.
      - [x] (3) cleaner anti-fabrication gate (`agreement_guard` + `rewrite_is_overreach`).
      - [x] (4) learned adaptive endpoint floor (`SessionPauseModel`, enabled in config.json).
      - [~] (5) TTS DC blocker DONE (`DCBlocker`, 0.05→0.00); the self-sizing playback
        prebuffer for underruns is DEFERRED (touches the hard-real-time audio callback,
        needs live validation) -- spec in the fix-5 design (workflow wf_eb0dff89).
      REMAINING: a live-mic session on the apm profile to A/B fixes 2 + barge-cut, then
      re-run the forensics replay (diagnose_run) to confirm the garble/fragmentation
      actually drop on real audio; implement fix 5b if underruns persist.

## P1 — voice / audio: follow-ups from the 2026-06-10 LIVE iteration (5 rounds with the owner)
> Context: docs/session_2026-06-10_capability_audit_and_fixes.md. Five live rounds
> fixed: AEC ref-delay (19→105ms calibrated, 30.3dB ERLE), DTD chart persistence,
> hold-and-merge, resume-after-interrupt, self-echo guard (acoustic + the CLEANER-
> hallucination variant), newest-input-wins supersede, noise-aware addressing +
> unsure_acts=false. Owner verdict after round 2: "barge in works properly now".
> These are the REMAINING improvements observed but not yet done:
- [ ] **Speaker-ID enrollment (OWNER, 2 minutes, biggest remaining lever for
      house-noise/other-voices).** Model present, gate FAIL-OPEN until enrolled:
      `python -m core --enroll`, read the sentence. If it then rejects the owner,
      lower `sherpa.speaker_threshold` toward 0.4 (laptop mics score 0.30-0.46
      cosine; see memory + docs/session notes). Identity-gates FINALS only (barge
      stays identity-free by design).
- [ ] **Investigate `_speaking` clearing while audio still drains (full-sentence
      echo mechanism).** Whole assistant sentences were transcribed as user turns,
      which requires ASR to hear playback — plausibly the playback epilogue's
      bounded FIFO-drain wait (`playback_fifo_sec + 0.5` deadline) expiring on long
      sentences so `_speaking` clears while sound is still playing. Measure (log
      fifo.count() at the deadline exit) and extend the wait/anchor on true drain.
      The L4 text guard now masks the symptom; this is the root.
- [ ] **Cleaner root-cause hardening:** the guards (rewrite_is_overreach + the
      own-words drop) bound the damage, but consider REMOVING assistant replies
      from the cleaner's recent-context entirely (only prior USER utterances are
      legitimate correction material) so it cannot copy the assistant's sentences
      in the first place.
- [ ] **Endpoint latency feels slow live** (endpoint_latency 1.8-3.6s/turn):
      once turn-merge is proven over a few sessions, lower
      `endpoint_min_silence_sec` 1.1 → ~0.7 on the Windows config.local.json and
      re-test mid-thought pauses (turn-merge now catches what the endpointer
      misses, so the safety margin can shrink).
- [ ] **Watchdog false "llm stuck"/"tts stuck" on held/merged/INGESTed turns**
      (review rc-5): stamp a handled_local/held metric and skip those turns in
      watchdog._check_turns; today every live bundle carries misleading
      stuck_hints.
- [ ] **STT garble is now the quality ceiling** ("conversation" for "story",
      "Skiper" for "keeper" fed weak/confabulated replies): raise mic capture
      level/gain, live-tune SenseVoice + prosody thresholds, consider the AT2020
      when docked. Garble also weakens turn-merge joins.
- [ ] **Fast-tier shallowness on contextful turns** ("That sounds lovely!" to a
      detail-laden follow-up): the roadmap P3 quality axis (route
      content-bearing follow-ups to main when recent context is rich) — see
      docs/review_2026-06-10_gap_analysis.md.

## P1 — voice / audio (migrated from session_2026-06-01 handoff)
- [x] **★ HARD REQUIREMENT (owner): open-speaker barge-in WITHOUT headphones —
      DONE + LIVE-VALIDATED 2026-06-08** on the bare ALC285 laptop mic+speaker (no
      premium mic). Owner: "barge feels good now" — a NORMAL-volume talk-over
      interrupts reliably, no shout, no self-interrupt. Solution: device-adaptive
      fused z-score double-talk detector `AdaptiveDTD` (core/engines/_dtd.py) — three
      features (raw energy / post-AEC residual / coherence) each a self-calibrated
      upward z-score from its OWN echo-only control chart; fire on the weighted SUM
      D > dimensionless K. NO fixed margin (the prior fixed-margin attempts all
      failed: self-interrupt, or rejected normal talk-over, or needed a shout). The
      decisive fix was the FIRING LOGIC, not the physics: per-frame fire
      (dtd_confirm_frames=1) + the capture-loop LEAKY integrator, because a real
      talk-over scored D=90-130 but flickered, so the old 3-consecutive rule
      discarded it. Tuned (SherpaConfig defaults): weights (raw 0.2, resid 1.0, coh
      0.0) -- z_resid is the discriminator (user voice isn't in the reference -> AEC
      can't cancel it -> lands in the residual); dtd_chart_rel_floor 0.4 (echo-leak
      precision). Commits fe0617b→5cd7f60 (coherence-primary audit) then
      71fd4ec/bbc4e01 (DTD + live tuning). Handoff:
      docs/session_2026-06-08_device_adaptive_barge_in.md. tools/echo_probe.py logs
      per-frame D for re-calibration on any machine. ref_delay stays 0 (FIFO already
      aligns the far-ref; do NOT set 260ms).

### APM-landing review findings (2026-06-21, 7-dim adversarial review; 11 confirmed 3/3) — open items
> Two P1s already FIXED + merged this session (commit on fix/apm-review-followups):
> the AEC reset()/process_16k cross-thread data race (core/engines/_aec.py lock) and
> the doctor false-fail on clean clones (tools/doctor.py active-vs-defined backend).
> The items below are the UNFIXED findings — left for a focused/live session because
> they change the hard-requirement barge gate (must validate at the mic) or are Dart
> (no SDK on the desktop box). Full review: logs of run wf_6f5da8f6 / session handoff.
- [~] **★★ P1 — DTD barge gate under `open_speaker`/`_apm_owns_ns` — CODE FIX LANDED
      2026-07-04, LIVE A/B STILL OPEN (wording refreshed 2026-07-06 after a verified
      recon caught this entry stale).** The originally-proposed fix — feed the DTD
      `resid` feature + the floor gate from a NON-NS source when the canceller blinds
      the residual — **is implemented**: `_apm_owns_ns`/`_resid_blind` are derived at
      sherpa.py ~1410 and the DTD residual feature + floor read `mic_raw` under
      `_resid_blind` (sherpa.py ~3623-3653, ~3712-3718; commits 3b994b7/55dfdb9).
      REMAINING: (a) the live-mic A/B on the APM profile that validates it (risks
      re-opening self-interrupt — cannot be judged headless); (b) the P2 double-talk
      regression below. NOTE the open-speaker barge authority is now the ADR-0013
      OS-capture + word-cut path (aec_enabled=false → the DTD never runs); this item
      only governs the in-app-APM FALLBACK path (`open_speaker` profile).
- [ ] **P2 — Mobile BargeCalibrator ambient floor is contaminated** (mobile/lib/assistant.dart):
      `observeQuiet` is fed on every `!_speaking` chunk, which includes the user's own
      request speech AND the TTS-echo tail (because `_speaking` clears before `_player.stop()`
      completes) → the learned floor drifts UP → `threshold=max(0.08, floor*2.0)` rises →
      a real talk-over gets harder to trigger over a session. Fix: gate `observeQuiet` on a
      genuinely-idle window (no ASR partial in flight + ~300-500 ms cooldown after playback)
      and add an upper clamp. Dart — validate with `flutter test barge_calibrator_test`.
- [ ] **P2 — No double-talk test at the SHIPPED APM config** (tests/test_apm.py): the only
      user-voice-survival test builds the APM with NS=FALSE (opposite of `open_speaker`).
      Add a NS=true echo+near-end mix test asserting the residual the DTD reads keeps enough
      near-end energy — this is the headless guardrail for the P1 above.
- [~] **P2/P3 — test gaps locking the APM behavior** (live-unvalidated code, tests are the
      only net; statuses refreshed 2026-07-06): (a) `aec_auto_delay` — the calibrator
      accept/reject + wiring ARE covered (tests/test_aec_seam.py:621-680,
      tests/test_denoise.py:366-407) but the single engine-level capture-loop test the
      original wording asked for (≥10 playback blocks → `_aec_ref_delay` updates /
      out-of-range ignored / no-op when off) is still missing; (b) `_apm_owns_ns`
      AND-guard derivation in `start()` still bypassed — OPEN; (c) `apm_stream_delay_ms`
      call order still untested — OPEN; (d) `tts_target_rms` streaming pin: STALE — the
      shipped behavior changed (streams once a carried gain exists, sherpa.py ~3413) and
      is pinned by tests/test_sherpa_playback.py:671-720; drop this sub-item.
- [ ] **P3 — defaults-safe / doc nits:** `barge_fade_ms=4.0` is a SECOND clean-clone default
      deviation beyond `tts_target_rms` (de-click on barge flush) — either sanction it in the
      invariant doc or default the dataclass to 0.0 and set 4.0 only in config.json.
      `aec_relaxed_margin_db` is effectively DEAD when any AEC+DTD is on (the DTD/residual-floor
      gates return first) — correct or delete its "uses aec_relaxed_margin_db instead" doc.
- [ ] **★ Stream the TTS for long answers (owner 2026-06-08).** A long story feels
      like it waits for the whole LLM answer before speaking. `_stream_and_speak`
      (core/capabilities.py) already emits sentence-by-sentence, so investigate why
      long answers feel un-streamed on the sherpa path: likely gemma4:12b main-tier
      first-token latency for a story, and/or confirm the per-sentence emit reaches
      TTS playback incrementally end-to-end. Goal: first audio after sentence 1, not
      after the whole story.
- [ ] **★ Smarter endpointing / turn-taking — don't barge in on the user's pauses
      (owner 2026-06-08).** When the user speaks with small mid-thought pauses, the
      assistant replies too early. It should use context to tell "still talking" from
      "done" and respond only with HIGH CONFIDENCE the turn is complete. Lever exists:
      Smart Turn v3 PROSODY endpoint detector on disk
      (pretrained_models/sherpa/smart_turn/), but endpoint_detector='lexical'. Switch
      to 'prosody' (needs onnxruntime+transformers), live-tune the confidence floor.
      See core/endpointing.py (adaptive confidence-tiered floor).
- [ ] **SenseVoice 2nd-pass agreement-guard (STT quality).** Re-enable sense_voice
      but accept it only when it AGREES with / clearly improves the streaming final
      (kills the short-clip hallucination 'I'->'Okay.'). New core/asr_text.py
      token-agreement helper + _final_transcribe word-count gate. (Currently reverted
      to streaming-only in config.local.json because the unguarded 2nd pass
      hallucinated.)
- [ ] **Enable + validate AEC on real hardware** (needs the mic). `config.local.json`
      → `sherpa`: `aec_enabled=true`, start `aec_backend="nlms"`. Calibrate
      `aec_ref_delay_ms` with `tools/echo_probe.py`. Confirm no self-interrupt AND a
      real interrupt still cuts through; then optionally try `aec_backend="dtln"`.
- [ ] **Extend `tools/echo_probe.py`** to print post-AEC **ERLE (dB)** and auto-suggest
      `aec_ref_delay_ms` via cross-correlation (no mic needed to write it).
- [ ] **Validate the Smart Turn v3 endpoint on hardware** (the prosody detector +
      `tools/turn_detect_check` real-voice validation tool + an adaptive
      confidence-tiered endpoint floor all LANDED on main from the voice batch below;
      what remains is the on-hardware A/B). Run
      `python -m tools.live_session --all --inject --smart-endpoint`; diff ON finals
      vs lexical/acoustic. (Default-off; on in the desktop profile.)
- [ ] **DTLN follow-ups:** smaller 256/128 size for phone profiles; clock-drift over
      long utterances; consider LiveKit AEC3 if the runtime ever moves to ≥3.11.
- [ ] **Move coherence ingest off the audio callback (real-time hardening).** The
      callback-`OutputStream` rewrite tees the played block into the AEC far ring,
      the level EWMA, AND `EchoCoherenceDetector.note_playback` from `_audio_cb`
      (the PortAudio thread). `note_playback` takes the detector lock that the
      capture thread also holds while `decide()` concatenates the reference ring —
      the only contended lock on the audio thread. Bounded + harmless at the
      default `coherence_ring_ms` (~38 KB / sub-100µs concat), but it MUST move
      off the audio thread (feed coherence from a lock-free SPSC stage drained on
      the capture/worker thread, like `FarEndRing`) before `coherence_ring_ms` is
      raised materially. Documented inline in `_audio_cb`'s docstring.

## P2 — runtime robustness (2026-07-06 codex-fleet recon; every item below adversarially verified against the code)
> Context: the 6-dimension stability recon (branch `fix/stability-recon-followups`).
> FIXED that session: EventBus discard-on-stop contract + high-water WARN
> (always_on_agent/event_bus.py + tests/test_event_bus.py) and the shutdown TTS race
> (`VoiceRuntime.stop()` `_stopping` gate — a queued TTS_REQUEST can no longer start
> speaking mid-teardown; tests/test_core_runtime.py). Remaining verified findings:
- [x] **Pending confirmations have no TTL** — DONE 2026-07-07:
      `sweep_expired_confirmations` (config `confirmation_ttl_sec`, default 180 s, 0
      disables) runs off the watchdog tick next to reap_overdue_tasks; expiry cancels
      the staged task + speaks "Confirmation expired: ..." (tests/test_confirmation_ttl.py).
- [x] **Follow-up timer + watch poller not joined/guarded on shutdown** — DONE
      2026-07-07: supervisor `_stopped` latch (shutdown() makes `_tick_followup` /
      `_schedule_followup` inert); WatchManager poller now waits on an Event, is woken +
      JOINED (bounded, self-join-safe) by shutdown() before state clears
      (tests/test_shutdown_guards.py).
- [x] **`builtins.input` shim race under concurrent tasks** — DONE 2026-07-07:
      module-level `_INPUT_SHIM_LOCK` (RLock) held across the whole `_auto_answer`
      window serializes the process-global swap; also sound because the shared
      interpreter was never safe to drive concurrently (tests/test_core_agent.py).
- [ ] **Task worker can outlive its supervisor reap** (always_on_agent/supervisor.py
      ~479-484 + tasks.py `_reap`): reap pops active_tasks + sets cancel_event but never
      joins the worker thread; bounded by daemon=True (dies at process exit) and
      documented inline as intentional — revisit if leaked workers show up in bundles.
- [x] **Unbounded queued_tasks list + runlog logging queue** — DONE 2026-07-07:
      `_queue_task` bounded admission (`max_queued_tasks=32` ctor default; drop-OLDEST
      non-continuation victim, cancelled + one spoken notice per storm); runlog queue
      bounded at 8192 with count-and-coalesce overflow ("runlog dropped N record(s)"),
      WARNING+ gets a grace put (tests/test_bounded_queues.py). (The HedgeLLM
      loser-thread join-budget leak is ALREADY tracked in the routing-polish section
      above — WARN-only observability shipped 2026-06-08d.)

## Smart routing — phase-2 audit (2026-06-08, 6-dimension fan-out: 28 findings, 19 confirmed)
> Verdict: smart routing is largely HEALTHY + fail-safe (no P0, no active §9.7
> boundary leak, every risky path double-bounded toward the more-private/conservative
> choice). Through-line = DORMANT intelligence (live_routing/cost_order/capability_router
> built+tested but off in most profiles) + one narrow PII gap. Adversarial verify
> dropped 9 plausible-but-wrong findings (e.g. "screen captures ride US cloud" is
> §9.7-authorized; several "boundary leak" claims were post-ASR-text egress, which
> the boundary permits). Landed in `feat/smart-routing-phase2-hardening` -> main.
- [x] **PII fail-safe for lowercased ASR (core/sensitivity.py).** Name+money rule was
      case-SENSITIVE; lowercase ASR ("what is john salary") slipped to PUBLIC. Added a
      case-insensitive comp-word rule (salary/wage/income/paycheck/pay stub/net worth/
      bonus) -> PRIVATE. The single fail-UNSAFE path, now closed.
- [x] **Host-aware cost ordering (core/routing.py `_preset_cost_key`).** Added host_rank
      as the OUTERMOST sort key so CN sorts after US/unknown (cost optimizes within a
      jurisdiction, not across it); fixes the CN-floats-ahead + OpenRouter-sinks-below-CN
      bugs. Latent until cost_order is enabled -> fixed before enablement.
- [x] **capability_router ON for desktop_gpu_4090 (config.json).** The shipped device
      inherited enabled=false while base 'desktop' had it on (most-capable profile, least
      routing intelligence). Mirrored desktop's block.
- [ ] **Activate dormant cost/latency levers on the CLOUD profiles + add measured evidence
      (the audit's #4, deferred).** Set `live_routing:true` (llm block) + `cloud.cost_order:true`
      on `cpu_laptop` / `phone_lite` (optionally `macbook_m_series`) where local is slow and
      cloud is on; keep desktop/4090 OFF (local 12b is fast). The cost_order fix above
      unblocks this. PAIR with a `tools/bench` (or replay) smoke that emits the chosen chain
      order + asserts cost_order lowers TTFT and the live nudge shortens the hedge under a
      high-load snapshot -- the first measured proof these levers help. Needs cloud keys to
      validate -> do on a cloud-enabled device. Files: config.json device_profiles
      (cpu_laptop/phone_lite/macbook), tools/bench/{runner,report}.py, docs/unified_architecture.md §5.
- [x] **P2 routing polish (6 of 7 shipped 2026-06-08d via the `p2-routing-polish` fan-out;
      merged to main).**
      (a) DONE -- HedgeLLM.shutdown() now WARNs (`speaker.llm.hedge`) with the survivor count
      when worker threads outlive the join budget (core/llm.py); the leak is visible in the
      run bundle instead of silent.
      (b) DONE -- WINNER_SELECT_BUDGET_FLOOR 30s -> 10s (core/llm.py:687).
      (c) DONE -- tier markers are now `\b`-anchored regexes ('show me the time' no longer
      hits 'how'; multi-word markers still match) + added 'compose'/'draft'/'write an'
      (core/routing.py `_compile_markers`). No route flips (nudges only).
      (d) DONE -- the ESCALATED (ReAct) + research.local paths now `_enrich_context` + publish
      `capability_context` (set/reset in try/finally, no cross-turn leak) so SensitivityRouterLLM
      picks the right cloud chain on those turns too (core/capabilities.py).
      (f) DONE -- LearnedRouter build-path test (backend='learned' raises RuntimeError when
      torch absent; tests/test_core_routing.py).
      (g) DONE -- doc-truth fixes (config.json cost_order comment; docs/unified_architecture.md
      Cost Order test-file citation -> tests/test_core_routing.py + host_rank note).
      (e) DEFERRED (coupled to audit #4) -- tier-aware load_fraction + shorter SystemMonitor
      cadence when live_routing on (core/sysinfo.py). The headroom signal mis-attributes CPU
      STT/TTS vs GPU LLM and lags 10s vs 1-3s turns, but it only matters once live_routing is
      enabled (the deferred #4), so do it WITH that work.

## P1 — desktop / 4090 fit
- [ ] Adopt `desktop_gpu_4090` profile on this machine (currently `device=desktop`;
      4090 profile raises `num_ctx` 4096→8192, `num_predict`→512, enables both gates).
- [ ] Measure real end-to-end ASR→LLM→TTS latency on the 4090 (`tools.bench --real`),
      calibrate `tools/specsim/specs.py` against it.
- [ ] Confirm sherpa runs the CUDA provider (config `sherpa.provider="cpu"` today) —
      evaluate GPU ASR/TTS on the 4090 vs auto-tuned CPU threads (32 logical).

## P1 — architecture / cross-platform
- [ ] **Mobile convergence onto the `AgentEvent` contract.** `mobile/lib/assistant.dart`
      is a parallel Dart loop re-deriving core behavior; align the Dart supervisor with
      the Python brain so the contract duplication disappears. See unified doc §10 / §12.
- [ ] **SQLite + sqlite-vec memory backend** for mobile (the `Memory` protocol makes it
      a drop-in for the Postgres adapter). See unified doc §6.

## Gemma 4 — ADOPTED 2026-06-05 (gemma4:12b)
- [x] **Adopted gemma4:12b** as the model (config.local.json, machine-local).
      Required updating **Ollama 0.24.0 -> 0.30.5** (0.30.4 412s on the gemma4
      manifest; 0.30.5 was the gemma4-capable release, GitHub installer since
      winget lagged). Measured head-to-head via `tools.model_probe` on the 16GB
      box: **gemma4:12b = 8.1GB VRAM (all GPU, ~7GB headroom), text 4/4, multimodal
      YES ('Red'), 256K-capable (num_ctx 8192)** -- a clean upgrade from gemma3:4b.
      gemma4:e4b = tiny (3.3GB VRAM) + 4/4 text but vision NOT wired in Ollama
      ('Please provide the image...') -> text-only. End-to-end VERIFIED: a host
      frame via `runtime.set_current_frame()` reaches gemma4 through the capability
      (image turn -> main tier -> 'Red'); text-only -> fast tier. Suite 1366 green.
      OPTIONAL next: two-tier gemma4:12b main + gemma4:e4b fast (~11.4GB, fits) for
      a real fast/main split; bump num_ctx toward 256K if memory/context needs it.

## Barge-in coherence-primary — v2 AND-gate LANDED + self-interrupt VALIDATED 2026-06-07
- [~] Coherence-on-raw-mic is the barge trigger (fe0617b); v2 (377d10e) requires a
      coherence "user" verdict to ALSO clear the post-AEC residual floor (orthogonal
      signals: AEC kills echo ENERGY not its incoherence; a real talk-over is incoherent
      AND loud). **LIVE-MEASURED on the open ALC285: 0 self-interrupts across 5 runs at
      full volume** (coherence-alone self-interrupted; echo raw-mic incoherent ~0.88
      overlaps real voice). **REMAINING (needs a human, can't be machine-tested):**
      confirm a REAL talk-over STILL fires — `python -m core --engine sherpa`, talk over
      a long reply; if missed lower `barge_in_residual_margin_db` (10.0). Re-confirm on
      the AT2020 USB mic (was unplugged this session). Handoff:
      docs/session_2026-06-07_barge_in_coherence_primary.md.
- [ ] _(superseded eval notes)_ **Evaluate + adopt Gemma 4** (Google, Apache-2.0, ~Mar 2026, actively updated).
      Ollama tags: `e2b`(7.2GB,+audio,128K), `e4b`(9.6GB,+audio,128K),
      **`12b`(7.6GB, image, 256K)** ← smaller than gemma3:12b's ~10GB, the best
      16GB fit; `26b`(18GB, MoE 3.8B active) + `31b`(20GB) too big. All multimodal
      (image); E2B/E4B/12B add native AUDIO + VIDEO in. The swap is **config-only**
      (`OllamaLLM` is model-name-agnostic; the `images=` path already works):
      `config.local.json` → `device_profiles.desktop_gpu_4090.llm`:
      `main_model`/`fast_model` → `gemma4:12b`, bump `options.num_ctx` (256K-capable).
      Harness ready: **`python -m tools.model_probe gemma4:12b gemma4:e4b --pull`**
      measures VRAM + text quality/TTFT + multimodal, vs the gemma3:4b baseline
      (4.4GB VRAM, 4/4 text, sees-image yes). BLOCKED on disk: C: had **1.8GB free**
      (need ~17GB for the 12b+e4b head-to-head). VERIFY when disk is freed:
      (a) Ollama 0.24.0 loads the gemma4 architecture (else update Ollama),
      (b) real VRAM on 16GB, (c) multimodal through the pipeline. Follow-ups:
      audio/video IN = new plumbing; mobile needs a LiteRT/.task Gemma 4 for
      flutter_gemma. See tools/model_probe.py.

## P1 — capability / testing (from 2026-06-05 test-unification pass)
- [x] **Multimodal image plumbing wired (2026-06-05).** The capability layer now
      forwards images: `core/capabilities.py::assistant()` reads `context['images']`
      (per-turn) or an ambient `image_provider()`, forwards them as
      `stream(images=…)`, forces the **main/multimodal** tier (the fast 1b can't
      see images), and floats sensitivity to PRIVATE (a screen capture never rides
      a public cloud chain, §9.7). `VoiceRuntime.set_current_frame(image)` /
      `clear_current_frame()` let a host machine feed the current frame ambiently
      to every assistant turn. Tests in `tests/test_core_multimodal.py`
      (per-turn/ambient/override/runtime + text-only-carries-none).
      A screen-capture SOURCE now exists too (`core/screen_capture.py`,
      `ScreenFrameFeed` + `build_screen_feed`, wired in `core/app.py`), **OFF by
      default** (`config.screen_capture.enabled`); when on it grabs the screen
      every `interval_sec` (mss + optional Pillow) and feeds `set_current_frame`.
      **REMAINING:** on-hardware validation (enable it on a live `--engine sherpa`
      run and confirm a frame reaches the multimodal model + the latency cost is
      acceptable), and any non-screen source (camera/app) if wanted.

## P2
- [ ] Wire `tools/swarm/harness.py perf --real` into `.github/workflows/perf.yml` parity.
- [ ] **`web.search` egress vs the floated turn sensitivity (review nit, 2026-06-08g).**
      `core/websearch.py` gates egress on the raw tool `arg` via `may_leave_device(arg)`,
      independent of `context["sensitivity"]`. Now that the ReAct planner also sees the
      recent-conversation block (2026-06-08g), it has more material it *could* phrase into
      a search arg. `_is_personal` re-classifies that exact arg so literal PII still fails
      closed, but consider ALSO gating the search arg on the turn's floated sensitivity
      (defence-in-depth). Orthogonal to the 2026-06-08g change; not a regression.

## Shipped this session (2026-06-02)
- [x] **Landed the unification refactor on `main`** (merge `d215a31`): merged
      `feat/aec-dtln` (`unified_architecture.md` + `session_bootstrap` + Windows landing
      doc) onto the diverged `origin/main`. Clean merge, full suite green
      (1283 passed, 13 skipped). Branch deleted.
- [x] Architecture/doc unification: `docs/unified_architecture.md` absorbs ~14 dated
      docs; merged docs banner-linked; stale session logs archived.
- [x] Removed dead `always_on_agent/snapshots.py`; renamed `core/agent.py` private
      `AgentEvent` → `AgentBrainEvent` (collision with the public contract).
- [x] Added `tools/session_bootstrap.py` + CLAUDE.md "Session bootstrap" section.
- [x] Relocated the accidentally-nested `social_media_activities_app/` out of the repo;
      removed the `UsersPaul` junk file; added `.gitignore` guards.

## Landed on `main` from the other machine (origin/main voice batch, 2026-06-02)
> Merged into `main` here; recorded so the next session knows this is already on `main`.
> `docs/unified_architecture.md` predates this batch and needs a refresh pass to cover it.
- [x] **SenseVoice two-pass final ASR**, shipped as the DEFAULT (run-on speech fix);
      pinned to English (was mis-detecting Chinese).
- [x] **Smart Turn v3 prosody turn-completion detector** + `tools/turn_detect_check`
      real-voice validation tool + adaptive confidence-tiered endpoint floor (~-110ms).
- [x] **Multi-signal barge-in stack:** loudness fallback (embedder-unreliable setups),
      scale-invariant reference-coherence detector (volume-independent, zero-setup),
      self-calibrating trigger margin (EWMA control chart, no per-room tuning).
- [x] **Enrollment hardening:** pin `capture_samplerate` (AT2020 self-mute fix),
      loudness rescue + VAD-trimmed enrollment; speaker-gated barge-in calibration doc.
- [x] `tools/echo_probe.py` added; `live_session` per-capability latency + denoise A/B
      + barge-in crash guard + response-quality grading.
