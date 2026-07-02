# Capability audit + the three owner complaints — 2026-06-10

> **Status (2026-07-02):** immutable dated record — the "guard.ps1 is DEAD" P0 below was fixed 2026-07-02 (the hook now blocks the work identity and direct pushes targeting `main`; see CLAUDE.md), and the public-history P0 is handled per `STATUS.md` §SECURITY (owner-accepted-deferred until republication from the organization account — not an open agent task).

Owner ask: "still has issues with the barge in, answering to incomplete
questions, and the memory of our conversation — do a review and a complex
refactor/audit to improve the capabilities." Ran the `speaker-review`
multi-agent workflow (55 agents, 45 adversarially-verified findings) AND landed
fixes for all three complaints plus the highest-leverage review findings.
Branch `fix/barge-contamination-turn-merge-topic-reset` → `main`. Full logic
suite **1514 passed / 22 skipped / 0 failed** (baseline was 1477).

## ⚠️ P0 SECURITY — OWNER ACTIONS REQUIRED (repo is PUBLIC)

The review found, and I verified by hand:

1. **A REAL Google/Gemini API key is in pushed public history** (tracked
   `.env`, since commit `d32db9f`; the OPENAI key in it is a placeholder).
   `.env` is now UNTRACKED at tip (landed this session), but the key lives in
   history forever → **ROTATE IT NOW** (aistudio.google.com → API keys).
   Rotation is required regardless of any history rewrite.
2. **Your raw voice WAVs + verbatim transcripts are public**: 52 tracked files
   under `logs/runs/` incl. 7 full-session WAVs (`.gitignore`'s
   `!logs/runs/*.wav` deliberately commits them) + the committed
   `tests/fixtures/barge_in/*.talkover.wav` clip. Voice audio is biometric PII.
3. **Decisions needed** (D1/D2 in `docs/review_2026-06-10_gap_analysis.md`):
   history-purge depth (recommended: `git filter-repo` purge of `.env` + PII
   bundles, optionally flipping the repo private first; tag/mirror the
   pre-rewrite SHA privately) and the future fixture policy (recommended:
   synthetic-only public, real voice in a private store).
4. **The `.claude/hooks/guard.ps1` hook is DEAD** — a missing `;` on line 28
   (`@{ p = 'git\s+config...'  why = ... }`) is a PowerShell parse error, so
   the whole guard fails open: none of the work-identity/SSH protections are
   active. One-character fix; the harness (correctly) refused to let the agent
   edit its own guard, so do it by hand.

## Branch → commit map (all on `main` after merge)

| Commit | What |
|---|---|
| `6de3a20` | feat(memory): topic-reset for the recent-conversation block (plan step 5) |
| `15ef939` | fix(security): untrack `.env` |
| `ed602a2` | fix(barge): DTD chart anti-contamination (plan step 3) |
| `8634e72` | feat(turn-taking): hold-and-merge final dispatch (rc-2 + rc-1) |
| `787ee04` | fix(routing): sensitivity float over the recall block (lm-3, §9.7) |
| `72c854d` | docs(review): the gap-analysis report + backlog P0 entry |

## What landed (the three complaints)

### 1. Barge-in — chart contamination fixed (plan step 3)
Root cause of BOTH recorded 2026-06-10 failures was the AdaptiveDTD's
**per-reply chart cold-restart**: warm-up re-seeded the echo baseline on
whatever played at reply onset — often the USER already objecting — so
`z_resid` pinned at 0.00 against levels 7–20× above the true floor (the
00:46:15 scream took ~3.7s to cut; the 00:45:08 normal talk-over NEVER fired),
and every missed block was then absorbed into the baseline (the miss-feedback
loop). Fix (`core/engines/_dtd.py` + `sherpa.py`, all off-switchable):
**persistent charts** across replies (`dtd_chart_persist`; per-sentence TTS
normalization makes the echo stable run-to-run — `new_run()` replaces the full
reset at both engine sites), **z-freeze** (`dtd_chart_z_freeze=3.0`: an
outlier-on-its-own-chart sample is never absorbed) with a **freeze-limit**
regime-change backstop (`dtd_chart_freeze_limit=30`), **robust lower-half
warm-up seeding** (`dtd_chart_robust_seed`), and a **VAD-quiet learning tap**
(`observe_echo`) so the charts learn from the genuinely-echo-only blocks
(decide() only ever saw VAD-speech blocks = the user-biased diet).
Validated against the committed traces via the REAL engine seam
(`tests/test_barge_contamination.py`): scream cuts in <0.5s, the 00:45
talk-over now cuts, clean echo never cuts, the 203236 turn-2/turn-3
requirements still hold, the 234435 self-interrupt replies stay silent.
**No K/margin knobs were touched** — the plan's step-2 measurement gate
(re-measure the echo floor on a normalized-playback LIVE run before tuning any
threshold) still stands as the next on-mic step.

### 2. Incomplete questions — hold-and-merge final dispatch (NEW `core/turn_merge.py`)
The live trace showed the endpoint committing fragment finals ('A running',
'Dear me', 'The fisherman', 'A long story about') and the brain answering each
one; the continuation layer can't merge because the fast tier finishes in
~0.6s. New layer at the final-dispatch seam (`turn_merge` config block, ON in
config.json): an incomplete-reading final (ends on a
conjunction/article/preposition — a deliberate superset of the endpoint list —
or a 1–2-word fragment) is HELD up to 1.2s; the next final MERGES into one
query; partials extend the hold (cap 6s); control words (yes/stop/never
mind/...) exempt; complete finals dispatch with zero added latency. ALSO moves
all final processing (addressing → cleaner → router, up to 3 blocking LLM
calls) OFF the audio capture thread (review rc-2) and fixes the
replay-harness bus double-drain (rc-1: `wait_idle` poll-only when the bus
thread runs + `EventBus.drain_once` TOCTOU + new `EventBus.idle()`).

### 3. Conversation memory — topic reset (plan step 5) + lm-3
`collect_recent_turns` now detects reset utterances ("start again", "never
mind", "new topic", RO "de la inceput", ...): the reset turn itself answers
FRESH (no recent block — the live failure replied to "Start again" with the
stale volume apology) and later turns never see pre-reset context. Bounded to
short utterances; off-switchable (`recent_context_reset_*`). Plus the §9.7
recall fix (lm-3): `memory.context_for_llm` blocks now float the turn's
sensitivity like recent-turns/images do — a precondition for ever enabling
recall (the review's P2 memory phase).

## Review deliverable

`docs/review_2026-06-10_gap_analysis.md` — 45 verified findings, per-goal
scorecard (memory/routing/cross-platform WEAK; real-time/providers ADEQUATE),
P0–P5 dependency-ordered roadmap, 8 owner decisions (D1–D8). Headlines beyond
this session's fixes: zero cross-session memory continuity in any default
config + broken migration 002; no quality/difficulty routing axis (cloud is
only a latency hedge) + mute tier failure (Ollama down = silent assistant);
installer omits scipy/soxr so fresh machines run degraded barge-in;
phone profiles unprovisionable; no cloud cost accounting.

## Environment on this box (i9-13980HX / RTX 4090 Laptop, Windows)

- SenseVoice 2nd pass is ACTIVE here (config.json ships `sense_voice`; the
  `asr_final_backend=""` disable was the LINUX box's config.local.json — plan
  step 4 applies THERE, with the agreement guard + `asr_final_min_sec` already
  landed).
- Windows config.local.json still has the stale `aec_ref_delay_ms: 19` (the
  open P0 echo_probe recalibration) and prosody endpointing ON
  (incomplete_threshold 0.5, min_silence 1.1) awaiting live tuning.
- `.venv\Scripts\python.exe` for everything; bash≠PS quirks per memory.

## Round 2 — LIVE owner test + fixes (same day)

Ran `--engine sherpa --record` live with the owner (run-20260610-124002).
Verdicts: merge/topic behavior partly good, but (a) still self-interrupts at
HIGHER speaker volume + "speaks out of turn", (b) "start again" semantics were
WRONG (owner wants RESUME of the cut reply, not a topic reset).

**Evidence + root causes from the bundle:**
- The transcript shows the assistant's own TTS coming back as USER turns
  verbatim ("Okay, let's begin. How can I help you today?") and being
  ANSWERED — the echo tail after playback ends reaches the recognizer, and at
  high volume it clears the L1 energy floor.
- `tools.echo_probe` sweep at the owner's problem volume: the stale
  `aec_ref_delay_ms=19` gives **7.3 dB** ERLE and echo-only D_p95=913 (worst of
  the whole sweep!); the true peak is **105 ms = 30.3 dB ERLE with
  self_interruptions=0**. The carried Windows-AEC-miscalibration P0 was the
  volume-dependence root. → `config.local.json` now 105 (plateau 105–135).

**Landed (branch `fix/resume-echo-guard-aec-calibration`):**
- NEW `core/resume.py` — `ResumeTracker` wired through `VoiceRuntime`:
  (a) RESUME-after-interrupt: "start again"/"continue"/"go on" (EN+RO) after a
  barge/stop cut becomes a continue-from-where-you-stopped prompt embedding the
  actually-SPOKEN tail (a second cut+continue keeps working; consumed flag; a
  new query disarms). (b) L4 SELF-ECHO GUARD: a final within `echo_window_sec`
  (3s) of playback end that reads like the just-spoken sentences (token overlap
  ≥0.75; short finals must equal the LAST sentence verbatim) is dropped before
  addressing/ingest — volume-independent, unlike the L1–L3 energy floors.
  Dataclass defaults OFF (an LLM reply often embeds the user's words verbatim —
  EchoLLM always does — so a default-on text guard eats harness repeat-queries;
  caught by test_bench); shipped config.json `resume` block opts both halves on.
- `DEFAULT_RESET_PHRASES` narrowed (owner): "start again"/"start over"/"de la
  inceput"/... removed — resume owns those; only unambiguous "never mind /
  forget it / new topic / change the subject (+RO)" reset; ordinary topic
  changes need no command at all (the model reads them from context).
- AEC calibration in machine-local config (above) — a CALIBRATION, not a
  threshold tune, so the step-2 gate is respected.

**Still watch live:** endpoint feels slow (endpoint_latency 1.8–3.6s — consider
lowering `endpoint_min_silence_sec` 1.1→0.7 once turn-merge proves itself);
watchdog still prints false "llm stuck" on merged/held turns (review rc-5,
P1); STT garble ("Skiper", "T a story about") feeds weak merges — SenseVoice
is on, but mic gain/quality is the lever.

## Rounds 3-5 — live iteration with the owner (same day)

Round-by-round: live test → bundle analysis → fix → re-test. Owner verdict
after round 2: **"the barge in works properly now."**

- **Round 3** (run-20260610-130622): phantom ECHO FINALS still derailed turns
  (the guard's 3s window missed finals that land ~4s after playback because of
  endpoint lag; garbled tails 'Leleep.'/'Loly.' aren't verbatim) and a stale
  in-flight turn spoke an old answer. → echo window 8s + char-fuzzy short-tail
  matching (difflib ≥0.6, tuned on the recorded garbles, control words exempt)
  + **newest-input-wins**: a NEW final cancels in-flight/queued turns (skips
  pending-confirm + CONTINUE add-ons). Merged `9f8ccf7`.
- **Round 4** (run-20260610-132603): "speaks at random house noise". Bundle
  smoking gun: **the LLM transcript cleaner hallucinated** — `cleaned: 'Well'
  -> 'What would you like to know about your place?'` (the assistant's own
  sentence from the cleaner's context) → phantom turns answered. → cleaner
  guards (own-words rewrite ⇒ DROP the turn; >2-word growth ⇒ keep raw —
  `core/cleanup.rewrite_is_overreach`), empty/punctuation finals dropped,
  long-final echo match reverted to EXACT overlap (fuzzy had eaten the owner's
  garbled real request "...story about my gun [cat]"), addressing prompt now
  INGESTs recognizer noise/word salad (live examples), `unsure_acts=false` on
  the enabled profiles (OWNER DECISION: UNSURE stays quiet). Merged `78e1ca9`.
- **Round 5**: launched for owner testing; session wrapped here. Bundle:
  run-20260610-135844.

**AEC calibration (machine-local config.local.json, round 2):**
`aec_ref_delay_ms` 19 → **105** (echo-probe sweep at the owner's problem
volume: 30.3 dB ERLE, self_interruptions=0; 19ms was the WORST of the sweep
at 7.3 dB / echo-only D_p95=913 — the volume-dependent self-interrupt root).

**Live-testing technique (reusable):** `.git/live_wrapper.py` (untracked) runs
`python -m core --engine sherpa --record --debug` and watches
`.git/STOP_LIVE_RUN`; creating that file raises KeyboardInterrupt in the main
thread = the graceful Ctrl-C path, so the run bundle flushes — lets an agent
session start/stop live runs cleanly. Set `SPEAKER_KEEP_RUNS=9999` or the
runlog pruner deletes committed bundles from the working tree.

**Remaining improvements (full list with rationale → `.agents/backlog.md`
"follow-ups from the 2026-06-10 LIVE iteration"):** speaker-ID enrollment
(owner, 2 min — the big remaining lever for other-voices/house-noise);
`_speaking`-clears-while-audio-drains root cause (full-sentence echo
mechanism); drop assistant replies from the cleaner context (root-cause vs
guard); endpoint_min_silence 1.1→0.7 once turn-merge proves out; watchdog
false stuck-hints on held/merged turns (rc-5); STT garble as the quality
ceiling; fast-tier shallowness on contextful turns (roadmap P3).

## Next steps (pick up here)

1. **OWNER: rotate the Gemini key + decide D1/D2 (history purge / fixture
   policy) + fix guard.ps1 line 28** — see the P0 section above.
2. **LIVE validation on the mic (plan step 2 gate + this session's fixes):**
   `python -m core --engine sherpa` — (a) confirm normalized playback stops the
   `resid_floor` swing (watch `dtd:` lines; should sit ~1–2× not 20–90×); (b)
   talk over a reply at normal volume → must cut without a shout, no
   self-interrupt (the persistent-chart fix is trace-validated but the owner
   requirement is live behavior); (c) speak with mid-thought pauses → fragments
   should merge (watch `merged held final` log lines) instead of being
   answered; (d) say "start again" → the reply must NOT resurrect the old
   topic. Only after (a) passes: tune barge knobs if still needed.
3. Windows `aec_ref_delay_ms` echo_probe recalibration (carried P0, needs mic).
4. Adopt the gap-analysis roadmap phases (P1 leftovers → P2 memory continuity
   → P3 routing quality axis → P4 cross-platform) — backlog has the pointer.
