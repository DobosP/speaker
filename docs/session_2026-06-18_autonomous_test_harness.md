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

## Enhancement (pt2): real-voice injection + over-the-air

Extended the `voice` tier so it's faithful to real-world usage (owner request):
real recorded-clip injection, WER scoring, and a pluggable acoustic path.

**New: clip sources + STT scoring.** `--utterances DIR` injects the owner's real
recordings (a `manifest.json` lists each clip's `file`/`text`/`role`); default is
the engine's synth voice. STT is scored by **WER** vs each clip's ground-truth
text (`tools/autotest/score.py`). Roles: `round_trip`, `speak` (long, for the
self-interrupt/barge windows), `barge` (overlap, not WER-scored), `command`.

**New: three acoustic modes (`--acoustics`).** The assistant's own audio
interacts with the mic, so the right mode depends on the goal:

| mode | echo | sound | for |
|------|------|-------|-----|
| `cable` (default) | no (playback→dead sink) | silent | clean, reproducible **STT/WER** (digital injection = perfect near-field). |
| `delay` | yes (loopback + ~260 ms air-gap) | silent | **self-interrupt + barge-in**, AEC aligned. |
| `speaker` | yes (real speaker+mic) | audible | **true over-the-air** — real ~260 ms delay + room/speaker coloring. Needs `--make-sound`. |

Why split: the assistant's TTS raises the engine's learned **echo floor**, which
drops quiet far-field clips as "echo/ambient". A real near-field user is loud
enough to pass; injected clips must avoid the echo (cable) or out-shout it (gain
boost in delay/speaker). Barge-in/self-interrupt only exist *with* echo. The big
finding: cable's per-launch loopback delay **jitters** (0–120 ms), so a fixed AEC
delay can't reliably align it — hence cable goes echo-free for STT, and the echo
story lives in `delay` (aligned by a 260 ms loopback) and `speaker` (real 260 ms).

**Validated (gemma3:4b, synth voice):**
- `cable` — **STT mean WER ≈ 0.13–0.16**, clips recognized ("sailor in the sea"
  0.08, "capital of France" 0.00); reproducible. The STT-accuracy path.
- `delay` — **self-interrupt = 0 (live), barge-in fires (pass)**; silent.
- `speaker` (real OTA) — **self-interrupt = 0 with the real ~260 ms acoustic
  delay** (the genuine open-speaker result, autonomous). STT degraded (WER ~0.9):
  synth-over-speaker is doubly degraded + far-field — **the owner's real
  recordings will score far better, and `cable` is the STT number that predicts
  the app** (near-field).

Honest limitation: with a single laptop speaker the injected "user" voice and the
assistant's TTS share one speaker (both far-field) — fine for STT/self-interrupt,
an approximation for *simultaneous* barge-in. True near/far separation needs a
second small speaker by the mic (optional).

Files added pt2: `tools/autotest/{acoustics,clips,score}.py`; `voice_loop.py` and
`__main__.py` reworked. Full logic suite still green (1991 passed, 24 skipped).

## Next steps (pick up here)

1. **Owner: record real clips with the guided studio** —
   `.venv/bin/python -m tools.autotest record --out recordings/owner` shows each
   line, counts down `3..2..1..GO`, auto-stops when you finish, trims silence,
   and writes `manifest.json` (gitignored, §9.7). The built-in script covers
   questions / commands / long prompts / barge-in / a memory fact+recall pair /
   natural speech (`--group`/`--limit`/`--review` to tailor). Then
   `.venv/bin/python -m tools.autotest voice --acoustics cable --utterances
   recordings/owner` for the real near-field STT/WER, and `--acoustics speaker
   --make-sound --utterances recordings/owner` for real over-the-air. Real voice
   should score far better than the synth defaults.
2. **The open-speaker P1 still wants ONE human-at-the-mic run** for the final
   gate: `./session.sh --llm echo` on the open speaker → confirm self-interrupt
   gone live, then calibrate `aec_ref_delay_ms` from the new `.ref.wav` via
   `python -m tools.aec_probe …`. The harness now covers the regression side
   headlessly, incl. a real-OTA self-interrupt check (S2=0).
3. **Optional: near/far separation for true simultaneous barge-in** — a second
   small speaker by the mic so the injected "user" isn't co-located with the
   assistant's TTS. Today both share one speaker (fine for STT/self-interrupt).
4. **Optional: CI wiring.** `tools.autotest suite` is CI-safe today; the `voice`
   tier needs PipeWire + models + a small LLM, so it's a self-hosted/manual gate.
5. **Minor finding to watch:** in the echo modes an assistant phrase occasionally
   gets recognized as a *user* final (self-echo-as-input) that the `dropping
   self-echo final` defense doesn't always catch — worth revisiting in the P1.
