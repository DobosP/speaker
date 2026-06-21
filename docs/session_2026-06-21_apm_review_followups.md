# Session 2026-06-21 — APM landing: live-validation attempt + adversarial review + P1 fixes

**Headline:** Continued the planned audio-APM validation. The headless open-speaker
`barge_stress` run was inconclusive (heavy unrelated CPU load on the box), but a
7-dimension adversarial review of the WebRTC-APM landing surfaced 11 confirmed
findings — two P1s **fixed + tested + merged**, the rest documented in the backlog.
Critically, the review's headline finding is **independently corroborated** by the
barge_stress numbers.

**Branch:** `fix/apm-review-followups` (merged to `main` at session end).

## Branch → commit map
- `a5db969` fix(audio): serialize AEC reset()/process_16k + stop doctor false-failing clean clones
- `<docs>`  docs/session(this file) + backlog + status + memory

## What landed (fixed this session)
Both P1, both verified 3/3 by independent adversarial skeptics, both headless-validated:

1. **AEC `reset()`/`process_16k` cross-thread data race** (`core/engines/_aec.py`).
   `reset()` is reachable from the event-bus thread (`CONTROL_STOP → stop_speaking`)
   and the playback worker (reply end), while `process_16k()` runs on the capture
   thread — the canceller had **no lock** (only the FarEndRing/FIFO did). A `reset()`
   landing mid-block could permanently desync a framed impl's (the WebRTC APM's)
   near/far carry buffers → the mic pairs against a shifted far-end reference for the
   rest of the session → silent ERLE collapse (the exact open-speaker failure the APM
   exists to fix). **Fix:** a `threading.Lock` serializing both entry points; the
   in-process divergence guard calls an unlocked `_do_reset()` to avoid re-entrant
   deadlock. Lock wraps a sub-millisecond numpy op on the capture thread (not the audio
   callback). Tests: concurrent reset-vs-process serialization + no-deadlock on the
   divergence path (`tests/test_aec_seam.py`).

2. **`doctor` false-fails READY on every clean clone** (`tools/doctor.py`). Now that
   the `open_speaker` profile (`aec_backend=apm`) is committed, the livekit check
   scanned **all** device_profiles and demanded `livekit` (a remote-only optional dep)
   even on boxes resolving to a non-apm profile → exit 1 / NOT READY on a healthy
   default install. **Fix:** require livekit only for the **active resolved** backend;
   a merely-**defined** apm profile is advisory (`ok=True`, never blocks READY),
   mirroring the existing `capture_voice_comm` rule. Tests: active-apm FAIL vs
   unselected-apm advisory vs no-apm-check (`tests/test_setup_doctor.py`).
   `check_audio_frontend` had **no** direct test before — that gap let this ship.

**Suite:** `2041 passed, 24 skipped` (was 2035 + 6 new). `python -m tools.doctor` → READY (exit 0).

## What was investigated but NOT fixed (see `.agents/backlog.md` → APM-landing review findings)
- **★★ P1 — DTD barge gate reads the APM-NS-suppressed residual under `open_speaker`.**
  The headline finding, and the actual reason open-speaker barge degrades under the APM.
  The June-8 DTD weights (raw 0.2, **resid 1.0**, coh 0.0) + the 12 dB residual-floor gate
  assume the post-AEC residual *contains* the near-end user (true for NLMS/DTLN). But
  `apm_always_on`+NS runs ML noise-suppression on every block and attenuates the user in
  that residual — re-opening the documented "user had to scream / 0 fired" failure. The
  2026-06-10 raw-mic fix only covers the coherence path, which is dead under the DTD branch.
  **Needs a live-mic A/B** (fixing blindly risks re-opening self-interrupts), so deferred.
- **P2** — mobile `BargeCalibrator` ambient floor contaminated by user speech + TTS-echo
  tail → threshold drifts up → erodes barge (Dart; no SDK on this box).
- **P2/P3 test gaps** — `aec_auto_delay` feedback untested; `_apm_owns_ns` AND-guard bypassed;
  `apm_stream_delay_ms`/call-order untested; `tts_target_rms`→streaming-branch unpinned.
- **P3 doc/defaults** — `barge_fade_ms=4.0` is a second clean-clone default deviation;
  `aec_relaxed_margin_db` is dead when AEC+DTD is on (misleading doc).

## The barge_stress measurement (why it's inconclusive)
Ran `tools.autotest.barge_stress` on `--device open_speaker` (injected via a temp
`config.local.json` `device` key, since restored). The **APM engaged correctly**
(`WebRTC APM: AEC3 + RES, NS, HPF, always_on=True`) and the **mic captured cleanly**
(rms 0.001–0.009, 0 clipping, 2 underruns). But raw numbers (fp 0.50 / tp 0.50, n=2/3,
one trial invalid) were measured under **~400% external CPU load** (`tesseract` + 4
`romania_scraper` processes — a different project on the same box), which destabilizes
real-time barge/TTS timing. **Not a trustworthy reading.** The talk-over rejections,
though, match the P1 DTD/NS finding above — so part of the tp drop is likely *real*, not
just contention.

## Environment on `i9-13980HX / RTX 4090 Laptop` (desktop_gpu_4090)
- `.venv/bin/python` (3.12); `python` bare is NOT on PATH. Use the venv.
- sherpa models under `pretrained_models/sherpa/` (NOT `models/`); paths pinned in
  the machine-local `config.local.json` (gitignored; locked OTA rig: `aec_backend=dtln`).
- ollama up (gemma3:4b/12b). livekit + soxr installed. **No Flutter SDK; no `gh`; no
  `GIT_HUB_TOKEN` in env** → CI status + `flutter test` must be checked elsewhere.
- The box frequently runs an unrelated heavy workload (`romania_scraper` + `tesseract`) —
  **audio/real-time tests need a quiet box to be trustworthy.** Do NOT kill those (owner's).

## Next steps (pick up here)
1. **Re-run `barge_stress` on a QUIET box** (no romania_scraper/tesseract) at `--device
   open_speaker` for a clean fp/tp reading vs the DTLN baseline (fp 0.20 / tp 1.00 / 3.02s).
   Temp-select the profile via `config.local.json` `"device":"open_speaker"` (restore after).
2. **Address the P1 DTD/NS finding** with a live mic: feed the DTD `resid` feature + floor
   from mic_raw / pre-NS tap when `_apm_owns_ns`, A/B that a real talk-over fires AND no
   self-interrupt returns. Land the double-talk regression test alongside.
3. **Re-enroll the speaker embedding** after enabling APM cleanup (`python -m core --enroll`).
4. **Mobile box:** `cd mobile && flutter test` (the 2 new test files) + `flutter analyze`;
   then the BargeCalibrator floor-contamination P2.
5. Add the deferred test guardrails (auto-delay feedback, `_apm_owns_ns` wiring, stream-delay
   order, streaming-branch pin) — pure test additions, no behavioral risk.
