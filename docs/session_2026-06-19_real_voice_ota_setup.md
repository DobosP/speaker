# Session 2026-06-19 — real-voice tests + over-the-air setup (JBL)

**Headline.** The autonomous harness now runs on the owner's **real recorded
voice**. Cable (near-field) STT scored **WER 0.29** across 37 clips with
self-corrections understood. The **over-the-air** path (assistant on a JBL Flip 5,
user voice on the laptop speaker) is wired up (`--inject-sink`) but needs **level
tuning** before it produces a clean run — left for the owner to run next session.

## The big fix: the built-in mic was clipping everything

The owner's first 37 recordings all clipped (peak 1.0, 5–37% of samples) →
garbled STT. Root cause: the **built-in mic analog gain was maxed** (`Capture`
+30 dB **and** `Internal Mic Boost` +30 dB = **+60 dB**). The ALSA mixer can't be
tamed directly (PipeWire manages/overrides it, and `parecord`/`pw-record` ignore
software volume) — the knob that **sticks and that the recorder honors** is the
**PipeWire source volume**:

```
pactl set-source-volume @DEFAULT_SOURCE@ 40%      # the fix (persists)
```

At 40% the owner's voice records clean (clip 0%, rms ~0.07–0.14). Verify any time
with the new meter: `python -m tools.autotest record --check` (say a sentence →
HOT / QUIET / GOOD). Two gotchas found live:
- the mic **source suspends** when idle, so a one-shot capture reads digital
  silence — hold it awake with a throwaway `parecord … &` during checks; a full
  recording keeps it awake itself.
- `sounddevice` capture (the recorder's path) returns zeros from a non-interactive
  shell unless the source is RUNNING — same suspend issue.

## Cable result (near-field, the STT number)

`python -m tools.autotest voice --acoustics cable --utterances recordings/owner
--llm ollama --model gemma3:4b` → **mean WER 0.29**, 0 dropped, all 5 commands
perfect, self-corrections understood ("call John no James" → *James*; timer →
*10 min*; repeats collapsed). This is the number that predicts the app's
near-field STT. (A harness fix was needed: `CableAcoustics.inject_gain` 100→55,
because the engine captures the null-sink monitor ~2.7× hot and unity injection
clipped it, ratcheting the echo-floor and dropping every clip after ~14 on long
runs. Now committed.)

Remaining STT artifact: the engine clips the **first phoneme** of each injected
clip ("St an alarm", "G me a recipe"). Prepending ~0.3 s lead-in silence to
injections should pull WER toward ~0.18 — not yet done.

## Over-the-air setup (run next session)

Wired up this session: `--inject-sink` lets the user clips play out a **different**
speaker than the assistant, for real near/far separation.

* **Assistant → JBL Flip 5** (the "open speaker"): `bluez_output.D8_37_3B_19_CF_03.1`
  — currently the **default sink**, so the assistant speaks through it. A2DP =
  output-only (no BT mic), so the laptop mic stays the capture device.
* **User voice → laptop speaker** (next to the mic, near-field):
  `alsa_output.pci-0000_00_1f.3.analog-stereo`
* **Capture → laptop mic**: `alsa_input.pci-0000_00_1f.3.analog-stereo` (default source)

Command:
```
.venv/bin/python -m tools.autotest voice --acoustics speaker --make-sound \
  --inject-sink alsa_output.pci-0000_00_1f.3.analog-stereo \
  --utterances recordings/owner --llm ollama --model gemma3:4b
```

### NEEDS TUNING before a full run

A 6-clip OTA subset **failed** (monitor_rms 0.059, STT garbage) — the capture was
too quiet because:
- the **mic is at 40%** (set low to stop the owner's *direct* voice clipping), but
  capturing *laptop-speaker playback* through the mic needs more level;
- the **laptop speaker sink is at 35%** (low), so the injected clips were faint at
  the mic.

Before the full run, raise both and re-check with a small subset until
`monitor_rms ≈ 0.2–0.3`:
```
pactl set-sink-volume alsa_output.pci-0000_00_1f.3.analog-stereo 85%   # laptop speaker louder
pactl set-source-volume @DEFAULT_SOURCE@ 70%                            # mic louder FOR OTA
# (remember to set the mic back to 40% before RE-RECORDING direct voice)
```
The JBL is loud at the mic (echo path confirmed live), so once the user clips are
audible the self-interrupt/barge tests should engage. Note the JBL is **Bluetooth
→ ~150–250 ms extra latency** on top of acoustic delay, so the real reference
delay is well above the configured 260 ms — run `tools.aec_probe` on the OTA
bundle to measure it and set `aec_ref_delay_ms` accordingly.

### Honest caveat

OTA **re-captures** the recordings through a speaker → degraded STT (the subset
garbled clean clips). **Don't judge STT from OTA** — use `cable` for that (0.29).
OTA's value is the **self-interrupt + barge-in under the real JBL echo**.

## Next steps (owner, next session)

1. Raise the laptop-speaker + mic levels (above), run a small OTA subset, confirm
   `monitor_rms ≈ 0.2–0.3` and that clips get recognized → assistant speaks.
2. Run the full OTA: the command above. Then `tools.aec_probe` the bundle for the
   real JBL delay; set `aec_ref_delay_ms`.
3. Set the mic back to 40% before any direct-voice re-recording.
4. Optional STT win: add ~0.3 s lead-in silence to injected clips (fixes the
   first-phoneme drop), re-run cable → expect WER ~0.18.

## Harness changes landed this session

- `record --check` mic meter; `record` guided studio already shipped.
- `CableAcoustics.inject_gain` 100→55 (echo-floor stability on long runs).
- `SpeakerAcoustics(inject_sink=…)` + `--inject-sink` CLI (near/far OTA).
- Correction/disfluency cases + the conversation (final-text→reply) report.

`recordings/` and the per-run `logs/runs/` bundles contain the owner's real voice
→ kept **local only** (§9.7), not committed.
