# Session 2026-06-19→20 — TTS de-click + barge-in P1 validation (locked OTA rig)

**Headline.** Continued the over-the-air autonomous testing with the external
speaker. Locked the owner's **"real conversation" rig** (agent → laptop speaker
@75%, user voice → JBL @20%), **validated open-speaker barge-in (100% talk-over
cut rate)**, and **root-caused + fixed the owner's "strange noise"** — it is the
VITS TTS model emitting impulse spikes, NOT FIFO underruns. All landed on `main`,
1999 tests green.

## What landed on `main`

- **TTS de-click** (`core.audio_frontend.declick` + `sherpa.tts_declick`, default
  on; wired into both `_synthesize` paths). Commit `4c02368` / merge `e0cfb39`.
- **Barge-in stress harness** (`tools/autotest/barge_stress.py`) — many
  self-interrupt + talk-over trials in one engine session → FP/TP rates +
  latency. `_Proc` also counts `barge-in REJECTED`.
- **Locked OTA rig** (`tools/autotest/ota_setup.py`) — device IDs + levels +
  `apply()` + `gain_pinner()`; `barge_stress` applies it automatically.
- Earlier in the thread (merge `5f3cd73`): OTA lead-in silence + JBL keep-alive
  for injection.

## The "strange noise" — root cause (overturns the prior theory)

Prior sessions blamed **FIFO underruns from CPU contention**. That is **wrong**.
A 4-way parallel investigation + a decisive offline experiment showed: the shipped
sherpa-onnx VITS voice (`en_US-libritts_r-medium`) emits **deterministic
sample-level impulse spikes** on certain text (dozens/sentence), reproducible in
**standalone synth with zero engine load**. The run's 5 "underruns" were benign
barge-cut flushes (the producer ran *ahead*; the >2/reply warning never fired).

**Fix:** `declick` (3-point-median impulse repair) on the synthesis/producer
thread — never the real-time `_audio_cb`. Measured: impulse clicks **26→0** and
**70→1**, **no-op on clean speech** (corr 1.0000, 0 samples touched), speech corr
≥0.996. Before/after A/B clips were sent to the owner (they confirmed by ear).
Measurement caveat: count clicks as *impulses* (jump-away-and-return), not raw
`|diff|>0.3` edges (those over-count loud-speech slopes).

Latent follow-up (not the cause): `_audio_cb` does an O(n) resample (22050→16k) +
RMS/EWMA on the hard real-time thread when play_sr≠16k and AEC is on — move it to
the producer if it ever bites on a slow box.

## Barge-in P1 validation (the open-speaker hard requirement)

`barge_stress` on the locked rig (assistant on the bare laptop speaker):

- **Talk-over cut rate: 6/6 = 100%** (every trial where the assistant was actually
  speaking was cut).
- **Self-interrupt: 1 false-positive in 5** (fp_rate 0.20); ~3 s cut latency.
- Open item: re-run with the de-clicker active to see if that 1 FP → 0 (a click
  teed into the echo reference was a likely trigger). **Blocked**: the JBL powered
  off mid-session and won't reconnect from software when off.

## STT clarification (important, prevents a wild-goose chase)

The owner's "recordings sound bad" → the recordings *sound* muffled (built-in mic,
energy <1.5 kHz, ~5-10% hum) but the **recognizer is robust**: fed DIRECTLY (one
hop = real on-device use) they transcribe at ~0.10 WER. **OTA WER (0.3-0.9) is a
double-capture test artifact — judge STT from the direct/cable path, not OTA.**
OTA's value is barge-in/self-interrupt under real echo. (Earlier the real OTA
killer was a **mic ADC clipping bug**: `amixer -c 1 Capture` defaults to +30 dB and
PipeWire re-applies it on every source suspend → clips at the converter; the
gain-pinner holds it down during runs.)

## The locked "real conversation" rig (codified, do not silently change)

- Assistant/LLM/TTS → **bare laptop speaker @ 75%** (`alsa_output.pci-0000_00_1f.3.analog-stereo`, default sink).
- User's injected voice → **JBL Flip 5 @ 20%** (`bluez_output.D8_37_3B_19_CF_03.1`, `--inject-sink`).
- Capture → laptop mic; `sherpa.aec_ref_delay_ms = 40` (laptop-speaker path, NOT the 260 ms BT path).
- `tools.autotest.ota_setup` holds all of this; `barge_stress` auto-applies it +
  pins the mic ADC. Run: `.venv/bin/python -m tools.autotest.barge_stress`.

## Next steps (pick up here)

1. Power the JBL **on** (`bluetoothctl connect D8:37:3B:19:CF:03`), then re-run
   `.venv/bin/python -m tools.autotest.barge_stress` — confirm the de-clicker
   drops the self-interrupt FP toward 0 and the output is audibly clean.
2. If the 1 self-interrupt FP persists, tune the barge gate (the click theory
   would be ruled out) — see the coherence/DTD path in `core/engines/sherpa.py`.
3. Optional: a sticky fix for the +30 dB mic-ADC reset (a WirePlumber rule) to
   retire the gain-pinner hack.
4. Optional: the laptop-only (shared-speaker) `--inject-sink default` path needs
   its own level tuning before it recognizes prompts — only worth it if the JBL
   stays unreliable.

## Env notes

- `.venv/bin/python` (no system `python`). Mic ADC controls on **card 1 (PCH)**.
- JBL **powers off / drops** mid-session; can't reconnect from software when off
  (`le-connection-abort-by-local`) — needs a physical power-on.
- `recordings/` + `logs/runs/` hold the owner's real voice → **local only**, not committed.
- Avoid `pkill -f "<text in your own command>"` — it matches the running shell (exit 144).
