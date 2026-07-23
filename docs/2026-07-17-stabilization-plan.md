# Stabilization plan — barge-in, STT transcript quality, update hygiene (2026-07-17)

The unified current plan after the ADR-0081/0082 landings and the branch
unification (both stale STT branches deleted; `main` is the only branch —
`fix/live-stt-quality` tip `371e58b` was byte-identical to landed `741e12b`
work, `feat/stt-consensus-v2` tip `08834e8` held two dev-machine symlinks).
Supersedes the phase tracking in `docs/2026-07-17-performance-roadmap.md` for
sequencing (that doc stays as the option reference); `STATUS.md` remains the
truth for what is verified. Owner intent (PROJECT_KICKOFF §1): the two
user-facing pains are "bad stop-talking" (barge) and "missing the exact
question" (STT) — this plan is scoped to exactly those plus keeping the stack
current.

## The one sequencing fact

**Every open thread funnels through the Windows OS mic level.** The capture
runs at ~6e-5 RMS ambient / ~7e-4 speech (30-70x too quiet): the tail-cut
attribution, the KWS phantom-cut rate, the ADR-0013 physical gate, calibration,
re-enrollment, and the Windows STT track are all measuring noise floor until it
is fixed. Nothing below Track A1 is trustworthy before it.

## Track A — root lever + barge verdict (owner at the machine, ~1 session)

- **A1. Fix the OS mic level** (Settings → Sound → input level/boost toward a
  ~-20 dBFS active-speech median), then run `python tools/calibration_suite.py`
  speaking — it A/B-records the presets through the real front end, scores by
  faster-whisper WER, and prints the winning `config.local.json` block. Gate
  for everything below.
- **A2. Re-enroll**: `python -m core --enroll` — required; the verified
  `wasapi-communications-aec -> gtcrn` capture domain rejects the legacy
  reference by design (ADR-0047/0056 provenance).
- **A3. Attribute the ★ tail-cut** (backlog top item — owner hears a
  deterministic cut "same spot at the end"): one `tools/echo_probe.py` run +
  one short live run, then read the bundle — the word-cut funnel line (now
  with the speaker-similarity distribution), playback receipts, dry-gap
  counter, and p50/p95 latency block landed precisely so this cut can be
  attributed (output-tail drain vs KWS phantom vs word-cut vs acoustic path)
  instead of guessed at. Fix only what the bundle proves.
- **A4. The ADR-0013 physical gate with the Kokoro voice**: sustained
  talk-over batch, bare "stop", silent control. Green → Windows barge is
  DONE (flip STATUS, write the closing ADR). Red → the funnel/similarity
  data now says exactly which stage starved — that difference is why this
  attempt is unlike the previous ones.

## Track B — STT transcript quality (runs in parallel; Linux-first)

- **B1. Build the disjoint held-out set** (STATUS's stated next step, the #1
  STT action — NOT another model swap): vault/domain terms,
  controls/near-controls, numerals, negation, silence/noise/echo, bystanders
  and multiple voices. Record through `./live.sh` so the ADR-0077 aligned
  pre-DSP/processed/reference triple is captured — frontend-vs-recognizer
  attribution is then built into every clip. Scrub per §9.7 before any
  commit (bundles stay private by default).
- **B2. A/B on that set via `tools/recorded_stt_eval.py`** (its promotion
  gate is already the policy: all clips covered, WER/CER no regression,
  keyword recall no regression, ≥1 measure improved). Candidate order —
  config-only levers first:
  1. SenseVoice second-pass parameters (the current final authority);
  2. **turn ON the landed-but-disabled consensus verifier**
     (`asr_final_verifier_backend` — Faster-Whisper Small / Parakeet
     `nemo_transducer`, ADR-0078/0080): ≥2-of-3 exact agreement flips a
     final, ties retain — measure whether quorum beats single-final;
  3. hotword/domain biasing for the vault-term misses (the engine's
     `_hotwords` seam) — the 2026-07-16 vault run scored 0/6 with clean
     capture, so domain vocabulary is the likeliest lever;
  4. endpoint prosody thresholds only if the set shows truncation-shaped
     errors.
- **B3. Windows joins after A1**: apply the calibration winner, refresh the
  `tools/real_usage` fixtures on the healthy mic (the 0/8 grade was fixture
  quality — hardening signals were green), and re-baseline
  `recorded_stt_eval` on Windows.

## Track C — barge structural work (only from Track A evidence)

- **C1. Platform reconciliation (explicit):** ADR-0071's 7-step closeout plan
  remains authoritative for the LINUX physical-barge thread (exact Stop is
  still physically red there); Windows now runs the ADR-0082
  OS-verified route and its verdict comes from A4, not from ADR-0071's
  steps. The deferred-barge buffer stays parked for route-less platforms
  only (contract analysis in backlog).
- **C2.** Decide the 2-3-word interjection cut class (roadmap item 15) FROM
  the measured similarity distributions A3/A4 produce — the 0.30/0.22 bands
  move only on data.
- **C3.** KWS floor threshold tuning likewise from live phantom/recall rates
  (floor stays conservative 0.25/0.30 until then).
- **C4. Repeat-guard clock race** (backlog P0, platform-independent, bounded):
  monotonic ordinal on `MemoryItem` compared strictly, timestamp fallback for
  foreign backends — builder-lane task, can land any time.

## Track D — update & maintenance hygiene (new; nothing existed)

- **D1. Monthly pinned-stack review** (calendar cadence, not automation-first):
  sherpa-onnx (currently 1.13.3), faster-whisper/ctranslate2/CUDA wheel trio,
  Ollama (0.32.0), Kokoro/KWS model releases. One pass = check upstream
  releases, decide bump/hold, record in backlog. First pass due 2026-08-15.
- **D2.** Enable grouped monthly Dependabot for pip as the advisory signal
  (config-only; pins stay authoritative — owner approves bumps).
- **D3.** The two P2 review cleanups when touching `tools/setup_models.py`
  next: consolidate the three tar-extraction loops (traversal guard is
  security-relevant), generalize selection-preservation into
  `wire_sherpa_paths`.

## Explicitly deferred (tracked, not lost)

Roadmap Phase 1 (endpoint 1.1→0.7 restore, instant-ack layer, memory stack
ON, SearXNG + cloud thinking tier) is the next block AFTER stabilization —
latency/UX work, unchanged in `docs/2026-07-17-performance-roadmap.md`.
Phase 3 items 18-19 (MoE main-tier swap, audio-native spike) stay parked.
