# Status — speaker

Single source of current truth for this repo. On any doc conflict:
STATUS.md > newest-dated ADR in `docs/adr/` > everything else (see AGENTS.md).

Last verified: 2026-07-05 (Linux Mint boot; full logic suite 2295 passed, 24
skipped; branch fix/live-barge-dtln-and-underruns, merged to main). Prior:
2026-07-04 (feat/auto-calibrated-audio-pipeline, ADR-0012, 5 auto-calibrating fixes).

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
3. Open + headless: R05 routing lever, R09 dead-air, R10 cleaner guard,
   R14 Parakeet ASR branch; voice-plan P2 bundle (`setup_models --kokoro`,
   per-device roll-off, Kokoro-vs-Piper profile gate).
4. Owner-gated: D1 history purge; enable memory + persona; SearXNG infra;
   voice-set finalization by ear.

## Agent notes

- Do not delete logs unless Paul explicitly asks.
- Do not claim live hardware validation unless it actually ran.
- Never read or print secret values.
