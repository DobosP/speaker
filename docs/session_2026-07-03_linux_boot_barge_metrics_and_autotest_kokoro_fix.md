# Session 2026-07-03 — Linux-boot: barge-funnel diagnostics + autotest Kokoro fix

Valid until: superseded by a newer dated session doc or a STATUS change — then treat as history.

## Machine reality check (important)

The owner asked to "continue on my Windows machine," but this session ran on the
**Linux Mint 22 boot** of the same dual-boot ROG Strix G634JY laptop
(`hostnamectl` → Linux Mint 22, ASUSTeK ROG, RTX 4090 Laptop; `uname` → x86_64
GNU/Linux). It is the *same physical box* whose Windows boot produced the
2026-07-02 live findings — but the two open owner to-dos are **Windows-boot
conditions that do not hold here**:

- Audio-out to-do ("still on Piper because C: was full") — **already satisfied on
  Linux**: `config.local.json` `tts_model` points at Kokoro
  (`kokoro-int8-multi-lang-v1_1/model.int8.onnx`), voices/tokens/lexicon set,
  `tts_output_lowpass_hz=7000`. No C: drive; 307 GB free.
- STT to-do ("raise the Windows OS mic level; redo speaker-ID") — Linux uses
  PipeWire/ALSA (no Windows mic-boost slider) and speaker-ID **is** enrolled here.

Owner chose (this session): **work on the Linux boot** — land the
platform-agnostic barge follow-ups + do what testing is possible autonomously
here. The Windows-specific fixes still require the Windows boot.

## Preflight

`python -m tools.doctor` → **READY → python -m core --engine sherpa**. All sherpa
models present, PipeWire 1.0.5, speaker-ID enrolled, Ollama healthy (15 models
incl. gemma3:12b/4b), livekit APM available. Baseline logic suite green
(2243 passed, 24 skipped before changes).

## Shipped this session (Linux boot, committed locally on a branch — NOT pushed)

1. **Barge confirm-funnel surfaced in `tools/diagnose_run.py`** (the open
   "Barge follow-ups (post-ADR-0011)" backlog sub-item). The three ADR-0011
   markers — `barge_in_duck` / `barge_in_confirmed` / `barge_in_unconfirmed` —
   are log-derived (they carry no detected/REJECTED marker of their own) and
   surfaced in both the text report (`--- Barge Confirm Funnel (ADR-0011) ---`,
   with a self-heal WARN when unconfirmed/duck ≥ 0.5) and the `--json`
   (`barge_confirm_funnel` key, named for the on_metric markers). Visible even
   on runs with **zero hard-fires** (every trigger ducked then self-healed — the
   echo-pumping signature the word gate defuses). Tests added to
   `tests/test_diagnose_run.py` (funnel counts, JSON/text render, absent-without-
   the-gate, and a hardening case).
   - **Adversarial review (3-agent workflow) confirmed one low-severity, ~unreachable
     collision**: `_BARGE_DETECTED_PAT` was an unanchored substring, so a confirm
     line whose transcription literally contained the hyphenated string
     `barge-in detected` would be miscounted. Hardened by anchoring the pattern to
     end-of-message (`barge-in detected\s*$`) + a regression test. Dead path for
     real ASR (emits "barge in", no hyphen), but now robust.

2. **Fixed `tools/autotest/audio.py::synth_to_wav` for Kokoro (real harness bug).**
   It hard-coded `cfg.model.vits.*` for the injected "user" clips, so once Kokoro
   (ADR-0010) became the default TTS the native loader aborted:
   `Not a model using characters as modeling unit. Please provide --vits-lexicon`.
   The autotest **voice** tier was therefore dead on any Kokoro-configured box.
   Now it builds the clip synth through the runtime's own
   `core.engines._sherpa_models.build_tts(SherpaConfig.from_dict(sherpa_cfg))`
   (Kokoro-aware, fail-open) instead of duplicating a VITS-only path.

## Findings (no over-claim — these are autotest/synth results, not human-mic validation)

- **ADR-0011 word gate (`barge_confirm_enabled`) is OFF on the Linux boot**
  (default `False`; not set in either config here). It was a Windows
  `open_speaker` acoustic-only-dichotomy fix. The Linux boot's barge-in is the
  AdaptiveDTD path (ADR-0004), already live-validated on the ALC285. So "the last
  feature" is not the active barge path here unless the owner opts the gate on.
- **`dtd_coherence_echo_veto` defaults `True`; the word gate defaults `False`.**
  The backlog "revisit veto default (OFF where the word gate is enabled)" only
  bites in profiles that opt the gate on — the default pairing is coherent.
  Documented, **not flipped** (flipping the barge gate needs a live-mic A/B).
- **Autotest `voice --acoustics cable` now runs end-to-end → VERDICT PASS.** But
  the WER is **synthetic-voice-artifact-dominated, NOT a human-STT measurement**:
  every injected Kokoro clip carries a spurious leading `And` (clip 1 raw
  `AND WHAT IS THE CAPITAL OF FRANCE` → cleaned final `What is the capital of
  France?`, WER 0.00), and the two longest clips collapsed to `And`/`Why`
  (WER 1.0). Kokoro TTS is out-of-distribution for the streaming zipformer ASR;
  this is a harness regression signal, not evidence about human STT. Real cable
  STT with **real recordings** is ~0.10 WER (see memory `ota-stt-is-test-artifact`).
  → To get a real STT verdict here, run `--utterances DIR` with human recordings,
  or a person at the mic. This run neither confirms nor refutes the owner's
  Windows STT complaint.
- **APM double-talk regression (`tests/test_apm_double_talk.py`) already exists +
  green** (16 passed) — the P1 `_apm_owns_ns` residual-source guardrail is landed;
  no new test needed.
- **Autotest `voice --acoustics delay` (silent loopback, isolated) — barge run.**
  S2 self-interrupt: **0 barge-ins during the assistant's own reply (pass)** — the
  key safety property held. S3 talk-over: **0 cuts registered (fail)**, but that
  result is **confounded by the digital-loopback caveat**, so it is NOT a
  trustworthy barge-miss signal: the loopback echo is loud (`avg_rms≈0.08`,
  clip 3.2% while speaking) and the config `aec_ref_delay_ms=40` doesn't match the
  loopback's adapted delay (~350 ms), so the post-AEC residual stays echo-heavy and
  the DTD fires on echo (D=100k–500k, incoh 0.94–0.98). The **coherence veto
  correctly rejected 4** of those as echo; `diagnose_run` flags 1 residual suspect
  during the S3 story. → a real barge-cut verdict needs a **human talk-over** on
  real hardware (the loopback stress-tests the veto, not the cut). Bonus: this run
  **dogfooded the new funnel** — with the word gate off it is correctly ABSENT, and
  the barge-event/DTD-context rendering all work on a real echo bundle.

## Not done / needs the owner (or the Windows boot)

- Real **human-mic** barge-in + STT validation on the Linux boot (I cannot speak
  into the mic; autotest speaker/OTA mode is audible + touches the locked OTA rig,
  so it was deliberately not run).
- The two Windows-boot to-dos (Kokoro repoint on Windows `config.local.json`,
  raise Windows OS mic level, redo Windows speaker-ID) — require booting Windows.
- Optional harness follow-up: inject **real** recordings (or a non-TTS user voice)
  so the autotest voice tier yields a trustworthy WER instead of an OOD-synth one.
