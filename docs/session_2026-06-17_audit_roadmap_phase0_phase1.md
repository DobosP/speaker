# Session 2026-06-17 — architecture audit → roadmap → Phase 0 + Phase 1

**Headline:** Ran an 8-dimension fan-out architecture audit, turned it into a
phased roadmap under two owner decisions, then **landed Phase 0 and all of
Phase 1 on `main`** (plus a data-loss fix). Full logic suite green throughout.

**Branch/commit map** (all merged to `main`, pushed to origin):

| Commit | What |
|--------|------|
| `be7582e` | docs: architecture audit + low-spec/barge-in roadmap |
| `443df8a` | **Phase 0** — barge-in measured baseline scorecard + regression gate |
| `12bfe17` | fix(runlog) — never auto-prune git-tracked run bundles (data-loss fix) |
| `1412c83` | **Phase 1** device-adapt-1 + cross-platform-8 + cloud-safety guardrail |
| `ae27e6e` | **Phase 1** llm-inference-2 — budget on-device LLM threads vs the audio path |
| `f2405c2` | **Phase 1** llm-inference-5 — provision the llama.cpp tier (`setup_models --gguf`) |
| `bd2e224` | **Phase 1** audio-bargein-8 — coherence ingest off the real-time audio callback |

## What landed

**Audit + roadmap (docs).** `docs/architecture_audit_2026-06-16.md` (verdict:
component choices are best-in-class — keep sherpa Zipformer/Piper, the cascaded
pipeline, the threaded control plane; the real gap is *device adaptation*).
`docs/roadmap_2026-06-17.md` sequences the work under **D-A** (open-speaker
barge-in is a hard MUST) and **D-B** (keep `logs/runs` dev artifacts committed →
PII purge deferred to a pre-release gate).

**Phase 0 — barge-in baseline.** `tests/barge_scorecard.py` +
`tests/test_barge_scorecard.py` + committed baseline
`tests/fixtures/barge_in/phase0_scorecard.json`. Drives the real shipped chain
(DTD + residual-floor gate + BargeSustain) over the canonical traces:
**0 self-interrupts, 2/2 talk-overs cut, max latency 0.407s, meets_requirement=true.**
This is the CI regression net for the decision logic — it does **not** replace a
live open-speaker A/B (that validates acoustics).

**Phase 1 — adaptation spine + barge protection (complete).**
- `device-adapt-1` + `cross-platform-8`: `config.device="auto"` probes the host
  and applies the matching profile at launch (a GPU box still resolves to
  `desktop` — zero change); unknown `--device` fails fast. Guarded by
  `tests/test_device_profile_invariants.py` (no profile may enable
  cloud/PRC/actuator/watch/web or disable `local_only`).
- `llm-inference-2`: `_auto_llm_threads` budgets the llamacpp thread count so a
  reply can't starve the capture/VAD/barge loop.
- `llm-inference-5`: `setup_models --gguf` + `requirements-ondevice.txt` + a
  guided `SystemExit` make the on-device tier runnable.
- `audio-bargein-8`: coherence `note_playback` moved off the PortAudio callback
  to a lock-free deque drained on the capture thread (behaviour-preserving,
  proven by an equivalence test); unblocks raising `coherence_ring_ms`.

**Bonus fix (`12bfe17`):** `prune_old_runs` was deleting git-tracked run bundles
on every `python -m core` startup — silently eating the committed barge WAVs.
Now tracked bundles are a protected corpus; only untracked runs are pruned.

## Environment on i9-13980HX

- `.venv` is the minimal post-wipe venv; **use `.venv/bin/python`**.
- **scipy was pip-installed this session** (it was missing). That un-gates the
  coherence tests: `1900 passed, 28 skipped` with scipy vs `1882/30` without. A
  fresh checkout without scipy will self-skip the coherence path.
- Auto-detect resolves this box → `desktop_gpu_4090`/`desktop` (NVIDIA 16 GB).

## Next steps (pick up here)

1. **Phase 2 — barge-in robustness** (each gated on the Phase-0 scorecard **and**
   a live open-speaker A/B before merge):
   - `audio-bargein-2` — wire the EXISTING runtime AEC-delay auto-calibration
     (`EchoCoherenceDetector.measured_delay_samples`) into a slow debounced
     re-align (attacks self-interrupt at the root; removes the manual `echo_probe`).
   - `audio-bargein-1` — per-device-profile audio-tuning block (now safe to raise
     `coherence_ring_ms` after audio-bargein-8).
   - `asr-tts-4` (streaming TTS preserving the echo floor), `audio-bargein-5`
     (promote Smart Turn v3 prosody), `audio-bargein-6` (pVAD veto — needs
     speaker-ID enrolled).
2. **Owner-only to-dos** (cannot be done headless):
   - `python -m core --enroll` — enroll speaker-ID (gates the pVAD veto + the
     actuator/cloud delegation gates).
   - Live open-speaker barge-in A/B on the bare speaker (`./session.sh --record`)
     to convert the field-unvalidated P1 into a measured scorecard entry.
   - `python -m tools.bench --profile phone` on real/throttled hardware to replace
     the specsim *estimates* the audit flagged as never-measured.
3. **Deferred (pre-release gate, per D-B):** purge the committed WAVs from history,
   turn on writer-level transcript redaction, add a gitleaks/PII CI gate — only
   before the repo goes public / a non-owner build ships.

See `docs/roadmap_2026-06-17.md` for the full phased plan and
`docs/architecture_audit_2026-06-16.md` for rationale + the rejected ideas.
