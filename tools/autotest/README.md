# `tools.autotest` — autonomous (no-human) test harness

Drives the **real** runtime end-to-end with **no person at the mic and no real
microphone**, so a session can self-verify STT / TTS / barge-in / memory and
emit a committable verdict. It sits between `tools.live_session` (needs a human)
and `--engine replay` (only re-runs *recorded* audio).

```
python -m tools.autotest all                         # memory + voice + suite + scorecard
python -m tools.autotest memory  [--llm echo|ollama] [--model gemma3:4b]
python -m tools.autotest voice   [--llm echo|ollama] [--model gemma3:4b]
python -m tools.autotest replay  [--bundle logs/runs/run-<id>.wav]
python -m tools.autotest suite                       # existing headless pytest gates
```

Use a **small** LLM — the harness tests the audio/memory plumbing, not model
intelligence. Default `ollama/gemma3:4b` (long, signal-rich replies; best for
the acoustic tiers); `--llm echo` is faster but its short replies give the AEC
too little signal and make the barge windows tiny — use it only for quick
plumbing checks.

Reports land under `logs/autotest/` (gitignored). The `voice` tier *also* drops
a normal run bundle under `logs/runs/` (the committed barge/AEC regression
corpus). Run from the repo root with the venv python: `.venv/bin/python -m
tools.autotest …`.

## Tiers

| tier | what it proves | needs |
|------|----------------|-------|
| `memory` | a fact stated in turn 1 is recalled at turn N — both the recall block is injected (plumbing) **and** the model uses it in its answer (semantic) | in-RAM `SessionMemory` (no DB); small LLM |
| `voice` | the real `sherpa` engine round-trips STT→LLM→TTS over a virtual cable; a talk-over cuts the reply; self-interrupt diagnostics | PipeWire (`pactl`/`paplay`), sherpa models, small LLM |
| `replay` | delay-independent self-interrupt (`tools.replay_barge`) + AEC ERLE/delay (`tools.aec_probe`) over a bundle | a `run-<id>.wav` with a `.ref.wav` sibling |
| `suite` | the existing headless barge/sandbox/memory pytest gates | nothing extra |

## How the `voice` tier works

1. Loads a `module-null-sink` — a virtual audio cable. **Reversible**: unloaded
   on exit; the system default sink/source is never changed.
2. Launches `python -m core --engine sherpa --input-device pipewire
   --output-device pipewire --record --debug --stream-tts` and moves **only that
   process's** playback + capture streams onto the cable
   (`pactl move-sink-input` / `move-source-output`, matched by the ALSA-PipeWire
   bridge node names — those streams carry no PID).
3. Injects TTS-synthesized "user" utterances with `paplay` and reads the
   engine's `--debug` stdout live (`[live] engine running`, `speaking:`,
   `barge-in detected`, `dropping self-echo final`).
4. Analyzes the run bundle + runs the delay-independent replay probes.

### The digital-loopback caveat (important)

The cable is a *digital* loopback: ~tens-of-ms reference delay, vs the open
speaker's ~260 ms acoustic delay (`aec_ref_delay_ms`). With the configured
260 ms the DTLN reference is misaligned, so the residual-path barge may
self-fire — an **artifact of the loopback, not the open-speaker bug**. So:

* the **verdict** gates only on the robust signals (cable carried audio,
  STT→LLM→TTS round-tripped, bundle produced);
* the **self-interrupt** result is a diagnostic, authoritative via the
  **delay-independent coherence probe** (`replay`), corroborated by AEC ERLE;
* `--calibrate` measures the loopback delay and aligns the AEC to it for a fair
  live residual-path check (per-launch delay is noisy, so it's advisory);
  `--aec-delay-ms N` forces it (P1 deep-dive).

The loopback gives **autonomous regression coverage** of the real-time path;
it does **not** replace the human-at-the-mic open-speaker validation.
