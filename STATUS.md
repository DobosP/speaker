# Status — speaker

Single source of current truth for this repo. On any doc conflict:
STATUS.md > newest-dated ADR in `docs/adr/` > everything else (see AGENTS.md).

Last verified: 2026-07-10 (headless; ADR-0016 runtime-readiness slice: doctor,
live-session preflight, and normal native-voice startup now consume one resolved
device profile and one shared readiness contract. Normal `--engine sherpa`
fails before model/device construction when load-bearing prerequisites are red;
console/replay/enrollment remain outside that gate and `--llm echo` does not
require Ollama. Active word-cut now requires an existing VAD model and, on Linux,
a loaded AND actively routed PipeWire/Pulse echo-cancel source+sink; APM requires
LiveKit only when AEC is actually enabled. Full logic suite: 2396 passed, 30
skipped; targeted doctor/live/app: 155 passed. ADR-0015 enrollment parity also
runs the active idle capture front end blockwise, persists versioned provenance,
and makes stale model/front-end enrollment fail open; its focused suite passed
93 tests with 1 skip. ADR-0013 word-cut reliability also changed:
playback-time recognition now runs before the acoustic-reference watch, fails
closed unless OS-capture intent + runtime/VAD state agree, and uses a dedicated
stream with bounded current-burst PCM. Confirmed cuts reset+replay only candidate
user PCM into normal ASR and splice the same audio into SenseVoice/floor/speaker-ID.
The 4-word mid-playback floor remains authoritative: a novel 1–3-word reply tail is
staged until VAD-active post-playback continuation adds a word; silence/endpoint,
own speech, empty text, and stale bursts are discarded. Full logic suite 2395
passed, 30 skipped; `git diff --check` clean. **LIVE bare-speaker A/B still required**
for cut rate, false tails, tail-word continuity, and re-enrollment; no live
human-speech validation was run.) Prior: 2026-07-07 late (Linux ROG box; P2
robustness cluster landed: full
logic suite 2387 passed, 24 skipped; `git diff --check` clean. Shipped in one
commit: ① diagnose_run word-cut verdict truth-up — a word-cut-confirmed barge is
no longer stale-flagged `suspect:no-dtd` (run-level exemption keyed on the confirm
line; classification stays strict otherwise); ② staged-confirmation TTL
(`confirmation_ttl_sec`, default 180 s) swept off the watchdog tick with a spoken
"Confirmation expired: ..."; ③ shutdown guards — supervisor `_stopped` latch for the
follow-up timer + WatchManager poller Event-wake + bounded join; ④ process-global
`builtins.input` shim serialized via module lock; ⑤ bounded queues —
`queued_tasks` drop-oldest cap (32) with once-per-storm notice, runlog queue
bounded 8192 with count-and-coalesce overflow, WARNING+ grace put. Tail talk-over
pre-roll preservation DEFERRED — needs live-batch audio validation; the headless
harness can't model near-end-words-vs-own-echo separation.) Prior: 2026-07-07
(Windows workspace; merge-audit sweep: full logic
suite 2345 passed, 24 skipped; `git diff --check` clean); 2026-07-06 (Windows i9/4090 box, later the same day; full logic
suite 2316 passed, 24 skipped on branch fix/stability-recon-followups);
2026-07-06 (Linux Mint boot; 2295 passed; ADR-0013 merged to main); 2026-07-05
(fix/live-barge-dtln-and-underruns); 2026-07-04 (ADR-0012). Live: 2026-07-06
(Linux boot, evening) — Phase B open-speaker barge live talk-over batch **FAILED**
(see the block directly below).

**2026-07-10 — speaker-ID enrollment front-end parity (ADR-0015).** `--enroll`
now applies the same idle-speech stages as live capture (AGC-or-static-gain →
resample → idle APM when active → GTCRN unless APM owns NS) and stores a stable
versioned fingerprint of the stages that actually built. Runtime requires both
the embedding model and front-end provenance to match; stale or legacy-nonraw
references are ignored so gating fails open rather than rejecting the owner.
Legacy enrollment remains compatible only on the raw baseline. **Owner/live step
still required after landing:** run `python -m core --enroll` through the intended
capture route, then verify the owner is accepted; no hardware validation is
claimed by this branch.

**★★★ 2026-07-06 (LINUX BOOT, evening) — PHASE B (ADR-0013) LIVE TALK-OVER BATCH
FAILED — READ FIRST.** The first real *sustained* talk-over batch of the OS-capture +
word-cut barge ran on the bare ROG speaker (Linux Mint, PipeWire 1.0.5;
`module-echo-cancel` webrtc loaded + PipeWire defaults repointed to the virtual
nodes; `aec_enabled=false, apm_always_on=false, barge_word_cut_enabled=true,
barge_confirm_enabled=false`; LLM gemma3:12b/4b; kokoro-int8) — and **it never cut.**
The assistant played a ~3-min story, the owner talked over it repeatedly, and the
reply ran to completion. Root, straight from the bundle (`run-20260706-231226`):
**zero multi-word ASR finals for the entire playback** — only two stray single-word
`'AND'` echo fragments (correctly below the 4-word floor). The word-cut barge is a
**text authority**; it got no words during playback, so it could not fire. This is
**not** the word-count logic and **not** a regression from the Windows session (that
touched tests/docs/shutdown-gate, never the barge acoustics): everything else was
healthy — clean *pre*-playback STT, mic never clipped (`clip=0.0%`), no crash, 2316
tests green. The single failure is that **the near-end user voice did not survive
capture during playback**, i.e. ADR-0013's core premise ("clean near-end STT during
playback") did **NOT reproduce** on a real sustained talk-over here — which is
exactly the "cut-rate on a real talk-over batch" item ADR-0013 flagged as *not yet
validated*, now answered negatively for this box. **Undetermined** (no audio saved —
the run was killed, not Ctrl-C'd, so the `record_playback_reference` WAV never
flushed), among: (a) the OS webrtc canceller over-suppresses the near-end during
double-talk (classic AEC double-talk suppression → Phase B is a dead end here,
consistent with [[barge-voice-no-acoustic-fix-2026-07-04]]); (b) the 13% mic is too
quiet under double-talk (fixable via level/AGC); (c) the per-speech-burst streaming
reset + own-speech filter swallow the near-end text. **→ RESOLVED same night
(branch `fix/barge-wordcut-live-diagnostics`): a deterministic (c)-class
state-machine defect was FOUND that fully explains the zero-word outcome.**
`_barge_word_cut_step` consults `vad.is_speech_detected()` but **nothing feeds the
VAD during playback** — the word-cut branch `continue`s before the acoustic path's
`accept_waveform`, so the VAD stayed frozen at its pre-reply quiet state and the
recognizer was never fed a single block. Zero words was guaranteed, no acoustics
required. **FIXED:** the step now feeds the VAD every playback block, and the
burst reset is debounced (`barge_word_cut_reset_quiet_blocks=3` — the old 1-quiet-
block hair trigger wiped a talk-over's accumulated words on VAD flicker). Whether
(a)/(b) ALSO degrade the near-end is exactly what the next run now measures:
shipped alongside — full word-cut funnel telemetry (`word-cut trace / burst reset
/ near-end / funnel` INFO lines, one summary per reply), kill-safe WAV recording
(RIFF header patched+flushed every 2 s → audio survives SIGTERM/SIGKILL), a
SIGTERM→Ctrl-C shutdown bridge, a doctor FAIL when word-cut is configured on
Linux without `module-echo-cancel` loaded, and a "Word-Cut Funnel (ADR-0013)"
section in `tools.diagnose_run`. **Correction:** the missing WAV was also because
the run never passed `--record` (recording needs the CLI flag; the config knob
alone does nothing) — the next live run MUST launch with `--record`. Suite 2341
passed / 24 skipped. **→ LIVE-VALIDATED 2026-07-07 00:29–00:50 (run-20260707-002943,
same rig):** the fixed word-cut **CUT a real talk-over for the first time ever** —
near-end words transcribed DURING playback (`word-cut trace: "I DON'"` →
`"I DON'T TO FEAR"`), hard-cut at the 4-word floor ~0.35 s after the first trace,
and the user's full sentence survived as pre-roll and was answered. Funnel across
5 watched replies: fed=90 / skipped_quiet=164 / resets=1 / dropped_words=0 /
own_folds=1 / decode_errors=0 / **cuts=1, false cuts=0** — own-echo transcribed
≈zero words all session (the OS canceller + quiet gate are doing their job).
**Real limits found:** (1) SHORT conversational replies are effectively
uncuttable — two talk-overs reached only 2 words before the reply ended on its
own (4-word floor can't fill; bare "stop" untested tonight); (2) words fed
during playback are DROPPED at reply end unless a cut fired, so a talk-over
near the reply tail loses its opening words from the following final; (3) the
diagnose_run self-interrupt classifier stale-flags a word-cut barge as
SUSPECT:NO-DTD → OVERALL FAIL (word-cut bypasses the DTD by design) —
**FIXED 2026-07-07 late** (run-level exemption keyed on the confirm line;
tests/test_diagnose_run.py pins PASS for confirmed cuts, suspect otherwise). **Kill-safe recorder LIVE-validated** by an accidental hard
kill (wrapper PID killed → python died with no teardown): both 1209 s WAVs
still valid on disk. The SIGTERM→Ctrl-C bridge itself is unit-tested but NOT
yet live-exercised (no summary.json for this run). Also observed, separate
thread: TTS voice flips mid-story — the LLM's `[voice:...]` tag applies only to
the first sentence (rest fall back to sid 0) and unknown names (`gentle`)
silently default. Machine config stays on the word-cut recipe
(`barge_confirm_enabled=false`). **NEXT (owner-directed 2026-07-07): STT
quality** — tonight's raw finals were garbled ("TANE MADE LONG STORY", "MY CAT
BICKIE" → an addressing INGEST miss ate a real request).

**★★ 2026-07-06 (WINDOWS BOX) — STABILITY RECON + ADR-0013 WINDOWS PREP (branch
`fix/stability-recon-followups` → main, 2026-07-06).** A 6-dimension codex-fleet recon
(20 agents, high-severity findings adversarially verified) drove: (1) the
ADR-0013 word-cut state machine got its first headless net
(`tests/test_barge_word_cut.py`: 4-word floor vs garbled echo, burst reset,
no-duck invariant, guards, `_aec is None` scoping); (2) `python -m
tools.setup_models --kokoro` now EXISTS (the runtime warning referenced a
fictional flag) — Kokoro fetch on THIS box still BLOCKED on disk (~1.0 GB free
after temp+pip cleanup; needs ~0.7 GB peak — free more, then one command);
(3) `VoiceRuntime.stop()` shutdown gate — a TTS_REQUEST racing teardown can no
longer start speaking (codex-review finding) + EventBus explicit
discard-on-stop contract + backlog high-water warn; (4) doc truth-ups
(ADR-0013 → accepted + Windows addendum; audio_pipeline.md now presents
OS-capture+word-cut as the validated open-speaker path, APM as in-app
fallback; two stale backlog P1/P2 items corrected — the `_resid_blind` mic_raw
tap WAS already implemented 2026-07-04). Machine-local `config.local.json`
flipped to the ADR-0013 WINDOWS recipe (`capture_voice_comm=true,
aec_enabled=false, apm_always_on=false, barge_word_cut_enabled=true,
barge_confirm_enabled=false, input_gain 4.0→1.0, input_calibrate=true`;
backup + rollback line in the file). **The next `python -m core` live run on
this box IS the Phase-B-on-Windows measurement** — see the checklist in the
`_os_capture_note_2026_07_06` config comment.

**★★★ PHASE B OUTCOME — OS-CAPTURE + WORD-CUT BARGE (2026-07-06, ADR-0013) — READ
FIRST.** The Phase B experiment ran live on the bare laptop speaker and it WORKS
where every acoustic approach failed. Route capture through the **OS/PipeWire
echo-canceller** (`module-echo-cancel`, webrtc, NS+AGC off) **instead of** the
in-app APM (`aec_enabled=false`), and the near-end user is finally **CLEAN during
playback** — the ASR transcribes a talk-over *while the assistant speaks* (`raw
'STOP'`). Barge-in is then a **continuous no-duck WORD-CUT**: cut on ≥4 new
non-own-speech words (or a bare "stop"), no acoustic/level gate, no duck.
Why not the alternatives, proven live this session: a **LEVEL** gate can't work
(the nonlinear speaker's residual-echo bursts are as loud as the user — 6/7 false
ducks at −18 dB); the **DUCK** itself is the "volume fluctuation"/pumping the owner
heard. Three adversarial verifiers caught that garbled echo transcribes as 2-word
junk → the 4-word floor + a per-burst stream reset are the no-false-cut gates.
**VALIDATED live:** no pumping, no false cut on echo, clean near-end STT. **NOT yet
validated:** the cut-rate on a real talk-over batch (needs more live runs; loopback
can't judge nonlinear echo). Everything is OFF by default (byte-identical); enable
per ADR-0013 (load module-echo-cancel via `pactl`, config `aec_enabled=false,
apm_always_on=false, barge_word_cut_enabled=true`, launch `--input-device pipewire
--output-device pipewire`). **Capture gotcha:** keep the PipeWire source volume LOW
(~13% → ADC ~6.75 dB); it maps source volume → hardware ADC gain, so a high source
volume drives +30 dB and CLIPS (memory `capture-gain-source-volume-mechanism-2026-07-05`).
LLM this session was `gemma4:e4b` (owner pick; it fires an `llm stuck` watchdog —
the tested tier is gemma3:12b+4b). The int8 Kokoro model drops the English "-er"
vowel (robotic on answer/refers); durable TTS fix = `kokoro-en-v0_19` (deferred,
owner "test first") — memory `voice-quality-diagnosis-2026-07-05`.

**★★★ LIVE-TEST OUTCOME + PERMANENT PLAN (2026-07-05) — READ FIRST.** Multiple live
open-speaker tests + a multi-agent study (3 adversarial verifiers) concluded:
**there is NO clean single-mic acoustic fix for open-speaker barge-in** (user voice
and own-echo overlap in every acoustic feature on a nonlinear laptop speaker). This
session's barge attempts (raw-mic DTD re-source under _resid_blind, coh-veto
disable, loose duck, APM switch) all failed live — DTLN misses talk-overs, APM
SELF-INTERRUPTS, the loose duck PUMPS. Those net-negative changes were REVERTED
(Phase A: coh-veto guard restored, loose duck removed; suite green). The full
history + the survived-verification plan is in
**`docs/session_2026-07-04_permanent_voice_barge_plan.md`** (also memory
`barge-voice-no-acoustic-fix-2026-07-04`). The real levers: (1) CAPTURE PATH — the
built-in mic ADC clips at +30 dB (fix + HOLD it at 52%/7.5 dB via `amixer -c 1 sset
Capture 52%` against PipeWire's reset; the clip fell 11–16%→0% live), and OS
voice-comm AEC (Linux module-echo-cancel / Windows WASAPI communications) is why
Teams + the Android app sound clean on this laptop; (2) ADR-0011 word gate as the
open-speaker hard-cut authority but KEEP the acoustic hard-cut as a scoped fallback.
STILL SOLID from ADR-0012: fix-1 AEC-delay auto-calib (validated 40→~120 ms), fix-2
relaxed-NS ASR tap (improved live STT on APM: "tell me a story about friends" clean),
fix-3 cleaner, fix-4 endpoint, fix-5a DC + 5b self-sizing FIFO (underruns 18→4 live).
Voice muffling fixed (tts_output_lowpass_hz 7000→0). **NEXT (Phase B, next session):**
resume the measured APM+clean-mic live test — is STT clean + does it stop
self-interrupting? — then decide if OS voice-comm is still needed. Machine-local
config now: aec_backend=apm, apm_always_on=true, lowpass=0 (config.local.json,
gitignored); the mic ADC pin does NOT persist — re-pin before any live run.

## What this is

Local-first always-on voice assistant. v1 = desktop Linux; no wake word
(implicit addressing + speaker-ID); modes quiet/assistant/research/command.
Runtime is `core/` `VoiceRuntime` on sherpa-onnx — run `python -m core` (the
legacy `main.py` monolith was deleted 2026-05-26, ADR-0002). One portable core
+ thin per-platform shells; raw audio never leaves the device (§9.7, ADR-0001).

## Living docs (everything else is dated history)

- `STATUS.md` (this file) — current truth.
- `.agents/backlog.md` — **the live work queue** (open P0/P1 items).
- `CLAUDE.md` + `AGENTS.md` — behavioral rules for agents (living specs).
- `docs/unified_architecture.md` — living architecture spec (single overview).
- `docs/audio_pipeline.md` — living audio-chain companion.
- `docs/adr/` — dated decision records, append-only (`0000-template.md`).
- `docs/session_*.md` — immutable dated records, NEVER operating instructions.

## Current state (2026-07-02)

- **2026-07-07 merge-audit P1/P2:** latency policies are consumed at
  the supervisor/task boundary: `clarify` gives a short clarification prompt,
  `stream_main`/`stream_research` opt the turn into streaming TTS, and
  `silent_ingest` suppresses spoken output while preserving existing silent
  side-effect tasks. The mobile pre-partial ASR speech-start signal (P1,
  quiet-observation gate) was DEFERRED by owner decision 2026-07-07 — Dart
  changes withdrawn from this branch; revisit at the mobile stage
  (`mobile/lib/assistant.dart` TODO stands).
- **2026-07-07: ack/continuation invariant fixed** — the ack_then_think latency
  acknowledgement no longer flips `started_speaking` (new `AgentTask.ack_spoken`
  field), so an add-on spoken after the ack but before the answer MERGEs into
  one combined turn instead of queueing behind the unheard reply; merged
  continuation turns are not re-acked. Regression:
  `tests/test_core_runtime.py::test_addon_after_latency_ack_still_merges`.
- `main` = c732c14 (R11 structured-history + R06b prompt-order merged).
- PR `fix/asr-guard-test-and-kokoro-fallback` must land to restore green CI
  (fixes a deterministically-red asr-guard test + Kokoro `build_tts` fallback).
- **Open-speaker barge-in: word-gated duck-then-confirm landed 2026-07-02**
  (ADR-0011, branch `feat/barge-duck-confirm`): an acoustic trigger ducks
  playback and only real transcribed words (or "stop") hard-cut; unconfirmed
  windows self-heal and teach the DTD echo charts. Built because the Windows
  live session proved the acoustic-only dichotomy (coherence veto ON = 468/481
  correct fires vetoed → no barge; OFF = fires on own echo → self-interrupt).
  Live-validated on the Windows box: talk-over and "stop" cut; volume pumping
  fixed by chart-teaching + `_now_playing` echo filter. Prior AdaptiveDTD
  status (ADR-0004) + the 2026-06-21 APM-NS-residual P1 (backlog) still apply.
  `aec_ref_delay_ms` is echo-probe-calibrated or `aec_auto_delay` — **never
  hardcode 260 ms** (ADR-0005). Headphones remain REJECTED as a fix (D-A,
  ADR-0008).
- AEC: **WebRTC APM = production open-speaker backend** (`open_speaker`
  profile); NLMS = dependency-free base default (headset/near-field only —
  live-ruled-out for open-speaker 2026-06-17, ADR-0006); DTLN ships no models.
- Models: Ollama gemma3/gemma4 tiering (`config.json`; machine-local pins in
  `config.local.json`). Kokoro TTS adopted; SenseVoice async 2nd-pass ASR with
  fail-closed agreement guard. Memory OFF by default (owner-gated). Web search
  = 5-entry stub (real SearXNG is owner-gated infra). Speaker-ID enrolled on
  the Linux box only (per-machine).

## SECURITY (OWNER-DECIDED, DEFERRED)

Repo DobosP/speaker is PUBLIC; the (now-dead) old Gemini key is reachable in
public git history (`d32db9f:.env`). Key **rotated 2026-07-02**. Owner decision
(2026-07-02): current public history — including the committed WAVs — is
**accepted for now**; the project will be published from the organization's
GitHub account, and the history purge (D1) + WAV/PII purge (D-B, ADR-0008)
happen at that pre-release gate. Agents must NOT run filter-repo or force-push,
and must not treat the purge as an open work item.

## Git / session policy

- Fleet rule (2026-06-24, ADR-0007): commit locally on green; **never push,
  merge to main, or delete branches without Paul's explicit ask**.
- Session start: `python -m tools.session_bootstrap`; ALWAYS `git fetch` +
  `git rev-list --count main..origin/main` first — never trust pinned commits
  in souls/handoffs.
- **Gotcha:** `python -m core` prunes `logs/runs/` bundles to keep=20 —
  including COMMITTED ones (phantom git deletions; restore, don't commit the
  deletions). New `logs/runs/` files are gitignored as of 2026-07-02.

## Standard verification

```bash
python -m pytest tests -q                     # full logic suite
python -m pytest tests/test_apm_double_talk.py -q   # APM/DTD regression
git diff --check
```

(Linux box venv: `/home/dobo/work/speaker/.venv/bin/python`.)

## Next (see `.agents/backlog.md` for the live queue)

> **Boot note (2026-07-03):** items 1–2 below are **Windows-boot** conditions —
> they do NOT hold on the Linux boot (Kokoro is already active there; STT fix is a
> Windows OS mic-level change). Doing them requires booting Windows. Linux-boot
> work this session: barge confirm-funnel surfaced in `tools/diagnose_run.py`,
> the autotest `voice` tier un-broken for Kokoro (`synth_to_wav` was VITS-only →
> native abort), APM double-talk regression confirmed green. The autotest cable
> WER is synthetic-voice-artifact-dominated (Kokoro is OOD for the zipformer ASR),
> so it is NOT a human-STT verdict. See
> `docs/session_2026-07-03_linux_boot_barge_metrics_and_autotest_kokoro_fix.md`.

1. **Audio output quality on the Windows box (owner verdict 2026-07-02: "still
   has the problem").** Root cause on THIS box: `config.local.json` still
   points `tts_model` at Piper `en_US-libritts_r-medium` — the adopted Kokoro
   (ADR-0010) was never fetched here (C: was 100% full; ~4 GB free now after
   cleanup). Free disk → fetch the Kokoro package → set `tts_voices` (+
   lexicon) → listen. Backlog "Windows live findings 2026-07-02".
2. **STT quality (owner verdict 2026-07-02: still garbled).** Evidence
   run-20260702-224616: raw finals like "I'M NODDY" (→"I'm not."), "NOTHING
   GOBOD BREATHED THERE AND TALKING ABOUT"; mic captures at avg_rms ~0.0007
   (30-70x below normal speech) — digital `input_gain` 4.0 can't add SNR.
   Raise the Windows OS mic level/boost, re-tune SenseVoice + prosody
   thresholds, consider R14 (Parakeet finalizer). Speaker-ID enrollment on
   this box also needs a redo on a quiet system (tonight's rejected the owner).
   **Tool (2026-07-07): `python -m tools.calibration_suite`** records the same
   phrase under N capture calibrations into one folder, each the exact 16 kHz
   the ASR would hear; scores every clip by independent faster-whisper WER +
   reference-free confidence + audio metrics and prints the winning
   `config.local.json` block. `--listen` loudness-matches a run's clips for
   fair ear-grading; `--selftest` self-checks with no mic. **Round-1 ear grades
   (owner, this box): raw=4 > voice_comm=3.5 = gain_boost=3.5 > agc=2 =
   voice_comm_agc=2** — the boost-only InputAGC audibly pumps the floor (ruled
   out as a quality lever) and WASAPI voice-comm alone doesn't beat raw; capture
   was CLEAN on every preset (the old avg_rms 0.0007 quiet-mic condition is gone
   on this box). Round-2 default sweep now targets the remaining Teams gap:
   `denoise` (GTCRN deep NS, model fetched 523 KB), `apm` (WebRTC APM always-on:
   AEC3+NS+AGC2+HPF), `voice_comm_denoise` (OS path + ML NS — the Teams-parity
   candidate); **`--talk-over`** plays the configured TTS voice through the
   speaker while recording (double-talk test; APM presets get the played frames
   as far-end via FarEndRing + AecDelayCalibrator). Selftest: GTCRN lifted
   synthetic SNR 26.9→42.9 dB; APM AGC2 leveled 0.106→0.176 RMS.
   **ROUND-2 DONE (owner, LINUX ROG box, dedicated session, 2026-07-08) —
   `denoise` WINS; ADOPTED `sherpa.denoise_enabled=true`.** The quiet `--play`
   sweep was ear-ambiguous (presets ~equal in quiet). The **double-talk test was
   decisive**: assistant clip (`logs/kokoro_voice_audition.wav`, 104 s) played
   through the laptop speaker while the owner talked over it, 5 s/preset
   (`--talk-over <WAV>` — the full phrase is too long to speak identically in the
   window, and the configured `kokoro-int8-multi-lang-v1_1` TTS crashes synth on
   `'style_dim' does not exist in the metadata`, so feed a WAV; WER meaningless,
   EAR is the verdict). Loudness-matched `--listen` ear grade: **denoise 5
   ("considerably the best") > raw 3 > apm 2.** denoise suppressed the assistant
   echo (est_SNR 39.2 vs raw 18.9) while keeping the near-end words intact; apm
   (AEC3) cancelled most echo but *garbled* the near-end on the open speaker
   (test→taste, cat→heart, lap→lungs) — the known open-speaker AEC3 distortion.
   Grades in `calib_runs/20260708-010952/GRADES.md`. **REMAINING (owner, at the
   mic): re-enroll the voice on a quiet system** — `python -m core --enroll`
   (embedding shifts post-denoise; this box's prior enrollment was already
   rejected and needs a redo). Session-2 prep done: faster-whisper installed
   (WER now real), denoiser fetched, `module-echo-cancel` loaded as
   `ec_source`/`ec_sink` (defaults untouched) for any future `voice_comm*` pass
   via a default-source repoint, `tools/calib_round2.sh` wrapper added. Linux
   gotchas retained: `voice_comm` presets are WASAPI-only in-tool (on Linux they
   only cancel through the EC source, via default-source repoint, not
   `--input-device` which PortAudio's pipewire bridge ignores); keep the PipeWire
   source volume LOW (~13% — memory `capture-gain-source-volume-mechanism-2026-07-05`);
   use `/home/dobo/work/speaker/.venv/bin/python`. voice_comm dropped from the
   ear test (round-1 already showed it doesn't beat raw). Follow-up: the broken
   `kokoro-int8-multi-lang-v1_1` TTS (style_dim) still needs the durable fix
   (`kokoro-en-v0_19`, memory `voice-quality-diagnosis-2026-07-05`).
   **PROMOTED TO FLEET DEFAULT (owner decision 2026-07-08):** the validated
   pipeline decisions now ship in the tracked `config.json` (cross-machine, deep-
   merged so a machine's `config.local.json` still overrides): `denoise_enabled=true`
   + relative `denoise_model` path, `tts_output_leveler=true`, `tts_markup=true`
   (`asr_final_async`/`tts_declick` were already code-default true). Deliberately NOT
   promoted (stay per-machine): `input_agc` (superseded by denoise), `barge_word_cut_enabled`
   (ADR-0013, failed the sustained talk-over test), `speaker_gate_input=false` (the
   base identity gate stays `true` — do not weaken it fleet-wide), Kokoro `tts_speaker_voices`
   (model-specific). `build_denoiser` fails open, so a machine without the 523KB model
   just runs no-denoise + a warning. Guard test `test_denoise` updated to the new
   default. NOTE: the other machine (Windows boot) also still carries the broken
   Kokoro TTS mapping in ITS `config.local.json` (`tts_voices`→kokoro + VITS `tts_model`
   = the `style_dim` crash) — clear `tts_voices`/`tts_lexicon` there (or fetch
   `kokoro-en-v0_19`) before an `--enroll`/live run.
3. Open + headless: R05 routing lever, R09 dead-air, R10 cleaner guard,
   R14 Parakeet ASR branch; voice-plan P2 bundle (`setup_models --kokoro`,
   per-device roll-off, Kokoro-vs-Piper profile gate).
4. Owner-gated: D1 history purge; enable memory + persona; SearXNG infra;
   voice-set finalization by ear.

## Agent notes

- Do not delete logs unless Paul explicitly asks.
- Do not claim live hardware validation unless it actually ran.
- Never read or print secret values.
