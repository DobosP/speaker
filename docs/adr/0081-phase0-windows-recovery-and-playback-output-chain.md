# 0081 — Phase-0 Windows recovery: playback output-chain quality, dead-input diagnostics, and the interim barge fallback

Status: accepted 2026-07-17 (Windows box session; roadmap `docs/2026-07-17-performance-roadmap.md`).

## Context

The 2026-07-17 25-agent SOTA review verified that the Windows box's "lag/accuracy"
complaints were dominated by configuration rot (TEMP e4b/e4b tiers, gates off,
old Piper voice) and missing output-chain quality, not compute. Phase 0 of the
roadmap restores the box and lands the platform-neutral output-chain fixes.
Owner directive: keep the small 1B (MiniCPM) fast tier.

## Decisions

1. **Playback resampling is anti-imaging.** The producer-side `write()` path
   resamples `tts_sr→play_sr` through the capture path's soxr-backed
   `AudioResampler` (built at stream open, worker-thread only) instead of naive
   `np.interp`. A new `AudioResampler.flush()` emits the FIR tail at each
   sentence end **under the same playback ticket**, keeping tracked receipts
   exact at `round(n_in * ratio)` output-domain samples and preventing
   cross-sentence tail bleed (reply-start `reset()` drops a barged tail). The
   AEC far-ref tee inside `_audio_cb` deliberately stays linear (hard real-time,
   echo-reference only). Verified on-box: soxr 1.1.0 streams re-arm exactly
   after `last=True` + `clear()`; pinned by tests.
2. **Inter-sentence dry gaps are counted.** A fully-empty callback read while a
   sentence is still being produced (`_synth_active`) and the reply has already
   played audio is dead air the partial-read underrun counter cannot see
   (whole-clip leveler synthesis made this the dominant audible stall). Counted
   RT-safe in `_audio_cb`, reported per reply (>2 blocks → WARNING +
   `playback_dry_gap` metric). `stop_speaking()` authoritatively closes the
   window.
3. **Dead input escalates.** Five consecutive ~silent capture heartbeats
   (~10 s of `avg_rms < 1e-4`) raise one ERROR + `capture_silent_input` bundle
   metric per silent episode (recovery re-arms). A merely-too-quiet-for-speech
   mic is a different fault: `tools/calibration_suite.py` at the OS level.
4. **Run bundles aggregate latency.** `RunSummary.to_dict()` adds a `latency`
   block: p50/p95/max/n per `TurnRecord` delta; `first_audio_latency` is the
   voice-to-voice headline (target ≤1.5 s, best ~0.5 s). All four stage stamps
   already existed; only aggregation was missing.
5. **Desktop context budget unstarved.** `desktop_gpu_4090` overrides
   `recall_recent_reserve_tokens` 320→1536 — the global 320 was sized for the
   1536-ctx `phone_lite` profile and starved an 8192-ctx desktop to ~2-4
   exchanges of history.
6. **Prompt: relaxed length + runtime-only variety.** `_STYLE` extends the
   full-answer carve-out to substantive how/why questions. A new `_VARIETY`
   rule (vary openers/wording, use contractions) rides **only** in
   `build_system_prompt` — it is a multi-turn instruction, meaningless in the
   one-shot legacy `DEFAULT_SYSTEM`.
7. **Windows interim barge fallback (machine-local config).** ADR-0019 proved
   the ADR-0013 Windows recipe (voice-comm capture + word-cut) fails closed at
   startup on installed sounddevice. Until a native
   `AudioCategory_Communications` capture helper lands (roadmap Phase 2), the
   Windows box runs the calibrated AEC3 rollback (apm backend, echo-probe
   105 ms, duck-confirm 0.05/5 s). `config.local.json` documents the flip and
   the exact rollback. ADR-0008 (no headphones) unchanged.
8. **Windows suite collection fixed.** `tests/test_promote_enrollment.py` uses
   `pytest.importorskip("fcntl")` so the POSIX-only module skips on Windows
   instead of aborting collection for the whole suite.

## Consequences / evidence

- Machine-local (not committed): main=gemma4:12b + fast=minicpm5-1b:q8
  (ADR-0020 pinned identity verified via `tools.setup_minicpm`), Kokoro
  int8 24 kHz active with `tts_output_lowpass_hz=7000` (ADR-0010), doctor READY.
  gemma4:12b live-load verification deferred: WDDM-commit OOM while WSL holds
  host RAM; do not `wsl --shutdown` while owner jobs run inside.
- Windows full suite 2026-07-17: 5296 passed / 150 failed / 70 skipped — the
  150 are pre-existing Linux-lane failures (virtual-audio POSIX contracts,
  live-launcher, prepare-enrollment, reminders, acoustics), byte-identical to
  the clean upstream baseline run; zero branch-attributable failures. Linux
  remains the green verification host (STATUS 2026-07-17: 5568 passed).
- Known upstream latent bug (NOT fixed here, backlog):
  `test_repeat_without_a_previous_answer_falls_through_to_model` — clock-
  resolution race in the ADR-0068 repeat guard (`timestamp >= since` with
  coarse Windows `time.time()`), introduced with `314f8ed`; fix = monotonic
  ordinal, see backlog entry.
- Live A/B acceptance for the audible changes (Kokoro loudness/seams, dry-gap
  attribution, barge fallback behavior) still requires the Paul-assisted mic
  level fix + calibration + re-enrollment before any verdict (roadmap §4
  acceptance policy).
