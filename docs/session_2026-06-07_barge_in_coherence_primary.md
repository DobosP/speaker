# Session 2026-06-07 — Gemma 4 adoption (Linux) + barge-in coherence-primary redesign

**Branch/commit:** `feat/barge-in-coherence-primary` → merged to `main` (fix commit
`fe0617b`). Logic suite **1356 passed, 14 skipped, 0 failed**.

**Machine:** i9-13980HX (32t) / 30 GiB / RTX 4090 Laptop 16 GB, Linux. Same physical
laptop as the Windows sessions (dual-boot).

## What landed

### 1. Gemma 4 adopted on Linux (machine-local, no code change)
- Ollama updated **0.20.2 → 0.30.6** (gemma4:12b returns `412: requires a newer
  version` below 0.30.5). Pulled `gemma4:12b` (7.6 GB).
- `config.local.json` (gitignored): `gemma4:12b` as BOTH tiers, set in
  `device_profiles.desktop/.desktop_gpu_4090.llm` (the device profile is deep-merged
  LAST, so a bare top-level `llm` override is clobbered). The code default model is
  still gemma3 in `config.json`; this is a per-box override.
- Verified: `tools.model_probe gemma4:12b` → 8.06 GB VRAM (100% GPU), text 4/4,
  multimodal `sees_image: true → "Red"`. App console run answers correctly.

### 2. Barge-in: coherence-on-raw-mic is the PRIMARY trigger (commit fe0617b)
Full audit (21-agent workflow) + adversarial review (11-agent workflow) drove this.

**Root cause (audit):** the live barge trigger was a LOUDNESS gate on the post-AEC
residual (`barge_in_residual_margin_db` defaulted to 10 dB and made that branch the
live path; the owner's tuned `input_loudness_margin_db=18` was DEAD code). A level
gate has no clean operating point on a nonlinear open speaker (residual echo spikes
overlap real-voice levels). Coherence — the volume-independent discriminator — was
OFF, and when on was fed the post-AEC residual (AEC removes the correlation it keys
on, so it misread cancelled echo as "user" = the self-interrupt).

**Changes:**
- `core/engines/sherpa.py`: capture loop tees the RAW pre-AEC mic block (`mic_raw`,
  no copy — AEC/denoiser return new arrays); `_barge_in_fire_eligible(samples,
  mic_raw)` → `_looks_like_user(samples, mic_raw)`. `_looks_like_user` makes
  `EchoCoherenceDetector.decide(mic_raw)` PRIMARY (True=fire / False=reject /
  None=fall through); the residual-floor / ambient / output-margin level gates are
  retained ONLY as the None-fallback. AEC still cleans the block for ASR. No
  AEC-reference re-alignment from coherence delay.
- `core/engines/echo_coherence.py`: **warm-up baseline seeding** (control-chart
  starvation fix from the review). The chart only learned the baseline on
  below-threshold frames; persistent nonlinear echo above threshold froze
  `_baseline` at 0.5 → self-interrupt forever. The first `warmup_frames` echo-bearing
  blocks (echo-only by construction) now seed the baseline unconditionally (running
  mean) so it learns the true echo floor. Self-calibration params plumbed to config.
- `core/engines/sherpa.py` SherpaConfig: new `coherence_warmup_frames` (5),
  `coherence_sigma_k` (3.0), `coherence_baseline_alpha` (0.2), `coherence_var_alpha`
  (0.15), `coherence_provisional_baseline` (0.5) — field-tunable for live calibration.
- `core/capabilities.py`: **compute-stop fix.** `_collect`/`_stream_and_speak` close
  the token generator on a cancel (`_close_token_stream`) so the model server stops
  generating at the barge point instead of lingering until GC (the direct Ollama path
  previously only `break`-ed).
- `tools/echo_probe.py`: forward `mic_raw` through the `_looks_like_user` monkeypatch.
- Tests: barge tests rewritten for the coherence-primary contract; new warm-up /
  starvation tests; new compute-stop tests.
- `config.local.json` (gitignored): `coherence_barge_in_enabled=true`; the stale
  "hardware limit → headphones" note marked SUPERSEDED. (Code default was already
  `True`, so the redesign is on by default for every machine.)

## Next steps (pick up here)

1. **★ LIVE-VALIDATE barge-in on the open ALC285 (the real test — needs the mic).**
   The raw-mic-coherence premise on the nonlinear speaker is theoretically sound +
   unit-validated, and the demonstrated self-interrupt mechanism (chart starvation)
   is closed, but it is **UNVERIFIED on hardware**.
   - `python -m tools.echo_probe` — echo-only: confirm `self_interruptions=0`, read
     the learned baseline/effective-margin (shows what the chart calibrated to).
   - `python -m tools.interrupt_suite` — mic × strategy sweep.
   - `python -m core --engine sherpa` — talk over a long reply: it MUST cut off and
     MUST NOT self-interrupt. Tune (no code): `coherence_warmup_frames↑`,
     `coherence_provisional_baseline↑`, `coherence_confirm_frames↑`, and consider
     lowering `barge_in_min_speech_sec` (0.8) now that coherence — not a level gate —
     is the discriminator, for a snappier cut.
   - `aec_ref_delay_ms` stays 0 until a fresh echo_probe ERLE sweep on this code says
     otherwise (the audit refuted "set it to 260ms"; the reference is already teed at
     the true playback position).
2. Run-bundle git churn: `core/runlog.py` keeps `SPEAKER_KEEP_RUNS=20` but ~50 bundles
   are committed in `logs/runs/`, so every `python -m core`/test run deletes the oldest
   committed bundles and dirties the tree. Bump the keep count or trim the committed set.
3. Older open items unchanged: Smart Turn v3 endpoint on-hardware A/B; move coherence
   `note_playback` off the audio callback before raising `coherence_ring_ms`.
