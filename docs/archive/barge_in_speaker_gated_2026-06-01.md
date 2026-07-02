> **SUPERSEDED (2026-07-02):** the speaker-ID gate as the primary barge signal
> was replaced by the self-calibrating `AdaptiveDTD` detector — see
> [`docs/adr/0004`](../adr/0004-adaptive-dtd-device-adaptive-barge-in.md);
> speaker-ID remains an auxiliary gate only. Historical record.

# Speaker-gated barge-in (no-AEC) — calibrated 2026-06-01

Barge-in on open speakers without AEC had two failure modes (earlier sessions): a
level-margin gate **self-interrupts** on the assistant's own TTS leaking into the
mic (134× in one session), and the synthetic-harness 2-stream acoustic test
**crashed** PortAudio. The robust fix is **speaker identity**, not level.

## The mechanism

`core/engines/sherpa._looks_like_user` gates a barge on identity when the speaker
gate is **enrolled AND `speaker_gate_input` is on**: only audio that matches the
enrolled user (cosine ≥ `speaker_threshold`) counts as a barge. The assistant's
own TTS is a *different voice*, so it never clears the gate — no self-interrupt,
no AEC required.

## Calibration (the user's enrollment, AT2020)

| | cosine vs enrollment |
|---|---|
| **your voice** (held-out clips) | 0.586 – 0.71 (floor **0.586**) |
| **assistant TTS** | 0.03 – 0.10 (ceiling **0.096**) |
| margin | **+0.49** |

`speaker_threshold = 0.4` sits well below your floor (robust accept) and well
above the TTS ceiling (firm reject). Engine-verified:
`_looks_like_user(your voice)=True`, `_looks_like_user(assistant TTS)=False`.

## Enabled (config.local.json, machine-local)

```
barge_in_enabled  = true
speaker_gate_input = true     # enables the IDENTITY barge gate (+ answers only you)
speaker_threshold  = 0.4
```

`config.json` already defaults `barge_in_enabled`/`speaker_gate_input` on; the
earlier laptop-mic sessions had disabled them in config.local (the garbled mic
made the gate reject the user, scoring 0.30–0.46). The clean AT2020 + a valid
enrollment reverses that — the user now clears the gate, so the default-on is
correct again. **Requires an enrollment** (`python -m core --enroll`); unenrolled,
barge falls back to the fragile level-margin gate.

NB `speaker_gate_input` also gates ASR **finals** on identity (the assistant
answers only the enrolled user — a TV / another person is dropped). That's coupled
to the identity-barge path in the current code; it's a feature here, but the floor
must stay above the threshold (0.586 > 0.4, with +0.186 headroom).

## The Bluetooth speaker — not usable, not needed

The BT speaker connects as a **PipeWire sink** (`bluez_output…`), but the app's
audio goes through **PortAudio**, which on this box only sees raw ALSA `hw`
devices (no PipeWire/pulse backend) — so it can't address the BT speaker. It
isn't needed: speaker-gated barge fires on *your voice* regardless of where the
assistant plays (laptop speaker, dev 5). A 2nd speaker would only matter for a
*level-margin* acoustic test, which speaker-gating supersedes.

## Live test

```
python -m core --engine sherpa
```
Ask for a long answer ("tell me a long story"); while it's speaking, talk over it
("stop") — it should cut off within a few hundred ms, and must NOT cut itself off
while you're silent. The run's `grade`/metrics carry the stop latency +
self-interrupt count.
