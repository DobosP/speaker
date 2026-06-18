# Session 2026-06-18 — autonomous (no-human) test harness

**Headline.** Built `tools/autotest/`, a harness that drives the **real**
runtime end-to-end with **no person at the mic and no real microphone**, so a
session can self-verify **STT / TTS / barge-in / memory** and emit a verdict.
It fills the gap between `tools.live_session` (needs a human) and `--engine
replay` (only re-plays recorded audio). Uses a **small** LLM by default
(`ollama/gemma3:4b`) — we test the audio/memory plumbing, not intelligence.

**Branch / commits.** Feature branch `feat/autonomous-test-harness` → merged to
`main`. Purely additive: new `tools/autotest/` package + one `.gitignore` line.
No existing code touched.

## What landed

`tools/autotest/` — four independent tiers (`python -m tools.autotest
<tier>`), each prints PASS/FAIL and writes JSON under `logs/autotest/`
(gitignored). See `tools/autotest/README.md`.

* **`memory`** — fact→distractors→recall over the *real* `assistant.answer`
  capability + in-RAM `SessionMemory` (no Postgres). Proves both the recall
  block is injected (plumbing) **and** the model uses it (semantic).
* **`voice`** — the novel tier. Stands up a **PipeWire virtual cable** (a
  `module-null-sink` + its monitor), launches the real `sherpa` engine routed
  onto it (the runtime's **own** streams only — system default untouched,
  reversible), and injects TTS-synthesized "user" utterances. Exercises the
  real-time capture thread, the `_audio_cb` playback FIFO, AEC, and barge-in —
  the path replay/sandbox cannot reach. Because the engine hears its own TTS
  over the loopback it is also a headless reproduction of the open-speaker
  self-interrupt condition.
* **`replay`** — wraps the delay-independent `tools.replay_barge` (coherence
  self-interrupt) + `tools.aec_probe` (ERLE/delay) over a bundle with a
  `.ref.wav`.
* **`suite`** — the existing headless barge/sandbox/memory pytest gates.

## Validated this session (gemma3:4b, desktop_gpu_4090)

* **memory** PASS — turn-1 "my favorite color is teal" → recalled at turn 5;
  answer *"You said your favorite color is teal."* (5.7 s).
* **voice** PASS (pipeline) — transcripts over the cable were accurate:
  *"What is the capital of France?" → "Paris."*; a sailor story; the planets.
  **S3 barge-in: a talk-over cut the reply (2 barges).** **S2 self-interrupt:
  delay-independent coherence = 0, AEC ERLE +19.4 dB.**
* **replay** PASS — `run-20260618-184417`: self-interrupts(remaining)=0;
  AEC best delay 20 ms, ERLE +19.4 dB.
* **suite** PASS — 56 tests. **Full logic suite: 1988 passed, 24 skipped.**

## The digital-loopback caveat (read before trusting the self-interrupt number)

The cable is a *digital* loopback: ~tens-of-ms reference delay vs the open
speaker's ~260 ms acoustic delay (`aec_ref_delay_ms`). With the configured
260 ms the DTLN reference is ~220 ms misaligned, so the **residual-path barge
can self-fire — an artifact of the loopback, not the open-speaker bug**
(`aec_probe` measures the loopback at ~20–80 ms, ERLE jumps to +19 dB once the
gemma reply gives it real signal). Therefore:

* the verdict gates only on robust signals (audio flowed + round-trip + bundle);
* the authoritative self-interrupt signal is the **delay-independent coherence
  probe** (it read 0), corroborated by ERLE;
* `--calibrate` / `--aec-delay-ms N` align the AEC to the loopback for a fair
  live residual-path check (per-launch delay is noisy → advisory).

So the harness gives **autonomous regression coverage** of the real-time path
and the coherence self-interrupt defense — it does **not** replace the
human-at-the-mic open-speaker run (still the P1's final gate).

## Environment notes (i9-13980HX / RTX 4090 Laptop)

* `python`/`pip` are NOT on PATH — use **`.venv/bin/python`** for everything.
* PipeWire is the audio server (`pactl`, `paplay`, `pw-record` present); the
  bare laptop speaker is `alsa_output.pci-0000_00_1f.3.analog-stereo`.
* PortAudio exposes only ALSA/OSS host APIs; the engine reaches PipeWire via the
  ALSA `pipewire`/`default` bridge device (`--input-device pipewire`). Those
  bridge streams carry **no** `application.process.id`, so the harness routes
  them by node name (`alsa_capture`/`alsa_playback`).
* Ollama has `gemma3:4b` (small, ~4 s warm) and `gemma3:12b`; Postgres is up but
  the memory tier uses the no-DB in-RAM backend.
* `config.local.json`: DTLN AEC on @260 ms, SenseVoice async second pass,
  `record_playback_reference=true`. The voice tier temporarily overrides
  `aec_ref_delay_ms` only with `--calibrate`/`--aec-delay-ms`, restoring the
  file verbatim afterwards.

## Next steps (pick up here)

1. **The open-speaker P1 still needs ONE human-at-the-mic run** (unchanged from
   the prior handoff): `./session.sh --llm echo` on the open speaker → confirm
   the self-interrupt is gone live, then calibrate `aec_ref_delay_ms` from the
   new `.ref.wav` via `python -m tools.aec_probe …`. The autonomous harness now
   covers the *regression* side of this headlessly.
2. **Optional: a delay-matched cable.** Insert a controlled ~260 ms latency into
   the loopback (`module-loopback latency_msec`) so the digital path matches the
   configured acoustic delay — then the *live* self-interrupt count becomes a
   fair gate too, not just the coherence probe.
3. **Optional: CI wiring.** `tools.autotest suite` is CI-safe today; the `voice`
   tier needs PipeWire + models + a small LLM, so it's a self-hosted/manual gate.
4. **Minor finding to watch:** with the loopback misaligned, one assistant story
   phrase ("jack in hand") was recognized as a *user* final and echoed back
   (`self_echo_drops=0`) — a self-echo-as-input leak the `dropping self-echo
   final` defense didn't catch. Re-check once AEC is delay-matched.
