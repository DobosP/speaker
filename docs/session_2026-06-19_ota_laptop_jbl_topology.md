# Session 2026-06-19 (pt2) — OTA topology flip + the "really bad sound" root cause

**Headline.** Continued the autonomous over-the-air (OTA) testing with the
external speaker. Per owner, **flipped the topology** to *assistant → bare laptop
speaker, user voice → JBL*. Chased down the owner's "the sound from my recording
is really bad" complaint to its root cause — **the mic's analog ADC gain
(`Capture`) was pinned at +30 dB and PipeWire re-applies it on every
suspend/resume, so the OTA capture clipped at the converter before any digital
volume could help**. With that tamed (plus two harness fixes), STT went from
garbage (WER 0.94) to mostly-correct (subset mean **WER 0.34**, individual clips
**0.00 / 0.08**), and **open-speaker barge-in fires on the bare laptop speaker
with zero self-interrupts** — the P1 goal. The JBL output level was too loud in
the room, so the owner powered it off; **full JBL OTA + level calibration is
deferred** to a follow-up session.

## Topology (owner's decision this session)

- **Assistant → laptop speaker** (`alsa_output.pci-0000_00_1f.3.analog-stereo`),
  the bare Realtek open speaker — this is exactly the open-speaker barge-in P1
  condition (CLAUDE.md hard requirement).
- **User voice → JBL Flip 5** (`bluez_output.D8_37_3B_19_CF_03.1`) via
  `--inject-sink` — the better speaker reproduces the owner's recordings far more
  faithfully than the tinny laptop speaker did.
- **Capture → laptop mic** (`alsa_input.pci-0000_00_1f.3.analog-stereo`).

(Prior session had these reversed: assistant on JBL, user voice on the laptop
speaker — which is *why* the recordings sounded bad: the owner's voice was being
re-played through the tinny laptop speaker next to the mic.)

## Root cause of "the sound is really bad": mic ADC clipping

1. **The source recordings are clean** — `recordings/owner/*.wav` peak 0.6–0.9,
   clip ≈ 0 %. Not the problem.
2. **The OTA *re-capture* clipped at the ADC.** The recorded mic WAV had
   **peak = 1.000, clip% = 16** even after lowering the PipeWire *source* volume.
   The culprit is the hardware control **`amixer -c 1 sset Capture` = 63/100 %
   = +30 dB** (card 1 = PCH). It amplifies the analog signal *before*
   digitization, so it clips at the converter — **no digital/source volume can
   recover a clipped ADC.**
3. **PipeWire re-applies the +30 dB on every source suspend/resume**, so a
   one-shot `amixer` set does not stick (matches the prior session's "the ALSA
   mixer can't be tamed, PipeWire overrides it"). **Workaround used:** a
   background loop re-asserting `amixer -c 1 sset Capture <pct>` +
   `Internal Mic Boost 0` every 0.5 s for the duration of each run. With Capture
   held at ~+4…+7 dB the clip% dropped to ~0.3 % and STT became clean.
4. **The engine's PortAudio/sounddevice capture runs ~10× hotter than the pulse
   path** (`parecord`/`pacat`) at the same ADC gain — so `parecord` calibration
   does NOT predict the engine's level. **Only a real engine run is ground
   truth** for peak/clip.

## Two harness fixes landed (validated live)

Both are general over-the-air improvements, independent of the JBL.

1. **Lead-in silence on injected clips** (`tools/autotest/audio.py`):
   `inject(..., lead_in_ms=)` prepends silence via a temp WAV; new
   `inject_lead_in_ms` on the acoustics classes (Cable/Delay = 0, **Speaker =
   500 ms**), threaded through the 4 inject sites in `voice_loop.py`. Fixes the
   first-word truncation — critical for Bluetooth, which drops the start of a
   freshly-resumed stream. **This is what made S3 barge-in start firing** (the
   talk-over clip's onset was being dropped → 0 barge-ins → now 1).
2. **JBL keep-alive** (`tools/autotest/acoustics.py`,
   `SpeakerAcoustics.session()`): when an `inject_sink` is set, hold it awake
   with a continuous silent `pacat /dev/zero` stream for the whole run, so a
   Bluetooth sink never re-suspends between clips (each `paplay` was otherwise a
   cold A2DP resume that ate the clip's opening). Verified: JBL SUSPENDED →
   RUNNING with the keep-alive.

Logic suite: see `.agents/status.json` (run at session end; harness changes have
no direct tests — `imports`/compile clean, no test references the harness).

## Results (subset = 2 round-trip + 2 speak + 1 barge, owner's real voice)

| run | setup | mean WER | notable | barge-in |
|----|-------|---------|---------|----------|
| 1 (old topo) | user→laptop spk, ADC +30 | 0.94 | hallucinated text | — |
| 2 | user→JBL, ADC +30 | 0.74 | "about the sailor and the sea" | fail |
| 3 | + lead-in | 0.85 | — | **PASS (1)** |
| 4 | ADC +6.75 | 0.73 | sailor clip **0.08** | fail |
| **5** | **+ keepalive + pinner, asst 55 % / JBL 65 %** | **0.34** | capital→"Central Perk" 😄, rainbow refraction correct, sailor 0.08 | **PASS (1)** |
| 6 | quieter (asst 38 / JBL 45) | 0.62 | capital **0.00 / "Paris"**, but 2 clips empty (too quiet) | fail |

Run 5 is the reference: **self-interrupt = 0 (pass), barge-in = 1 (pass)** on the
bare laptop speaker, coherent conversation. STT degrades gracefully — the only
weak clip is the very short "what time is it" (VAD onset). **OTA STT is inherently
degraded vs the `cable` path (0.29); judge STT from `cable`, judge
barge-in/self-interrupt from OTA** (prior session's caveat, re-confirmed).

## AEC calibration finding (laptop-speaker assistant)

`tools.aec_probe` on the louder run-20260619-172921 bundle:
**BEST delay = 40 ms, ERLE +5.0 dB** (well-defined peak at 40–60 ms). The stale
`aec_ref_delay_ms = 260` was for the *JBL-as-assistant* path (Bluetooth latency);
the laptop speaker shares the chassis with the mic so the speaker→mic latency is
just playback/capture buffering (~40 ms). **Set `aec_ref_delay_ms: 40` in
`config.local.json`** (gitignored local override) — re-confirm at the final
calibrated levels.

## Deferred — JBL OTA + level calibration (owner, next session)

The owner powered the JBL off because its output was too loud in the room.

1. **Dial the JBL output level** for a comfortable room that's still audible at
   the mic. Key lever discovered: **room loudness (speaker volume) and mic
   pickup (ADC gain) are decoupled** — keep speakers quiet and turn the *mic* up.
   BUT at very low speaker levels the user clips fall **below the VAD
   speech-floor** (run 6: empty transcripts). Sweet spot is roughly the run-5
   region (asst ~55 %, JBL ~65 %, ADC ~+4 dB) or quiet-room + higher ADC, tuned
   per JBL placement (closer to the mic = quieter room, same mic signal).
2. **A sticky fix for the +30 dB ADC reset.** The gain-pinner loop is a runtime
   hack. Proper fix: a WirePlumber/udev rule that caps the `Capture` ADC gain, or
   keep the pinner only during test runs. Until then, start the pinner before any
   OTA run.
3. **Re-run the full 37-clip OTA** once levels are dialed (the subset already
   proves the pipeline end-to-end).
4. **Re-confirm `aec_ref_delay_ms`** at the final setup.

## Environment notes (this machine)

- `python` is not on PATH — **use `.venv/bin/python`**.
- JBL idles out / disconnects when not streaming: `bluetoothctl connect
  D8:37:3B:19:CF:03`, then `pactl set-default-sink …` as needed.
- Mic ADC controls live on **card 1 (PCH)**: `amixer -c 1 sget Capture` /
  `Internal Mic Boost`. PipeWire resets `Capture` to +30 dB on suspend.
- `recordings/` and `logs/runs/` bundles hold the owner's real voice → **local
  only** (§9.7), not committed.
- ⚠️ Avoid `pkill -f "<text in your own command>"` — it matches the running
  shell and kills it (exit 144). Kill background loops by saved PID or
  `pgrep -x <name>` + parent.

## Next steps (pick up here)

1. Power the JBL on, reconnect, set as default-sink-for-inject; calibrate its
   output level (deferred item 1 above).
2. Start the ADC gain-pinner, run a subset, confirm clean STT + barge-in.
3. Full 37-clip OTA; `aec_probe` the bundle; confirm `aec_ref_delay_ms`.
4. Decide on a sticky ADC-gain fix (WirePlumber rule).
