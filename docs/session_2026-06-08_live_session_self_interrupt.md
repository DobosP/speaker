# Session 2026-06-08e — Live session on the WINDOWS side: self-interrupt + SenseVoice cascade

**Run bundle:** `logs/runs/run-20260608-181250.{txt,wav}` (no `summary.json` — the
session was force-stopped, so the bundle didn't finalize; the `.txt` is the full
trace and the `.wav` is a replay fixture). This was the **first live `--engine
sherpa` run on the Windows side** of the box (all prior barge-in validation was on
the Linux side).

## What the owner saw
1. **It interrupts itself.**
2. **Strange interactions — two outputs one after another.**

## Root cause (conclusive from the trace)

### Primary: SELF-INTERRUPT on the open-speaker echo
Every `barge-in detected` fired while `speaking=True` with **avg_rms ≈ 0.005–0.018**
— that is residual TTS *echo*, not a real voice (a genuine talk-over is ~0.2–0.4).
The adaptive z-score DTD (`K=5.0`, weights `(0.2, 1.0, 0.0)`) is tripping its
`z_resid` discriminator on the assistant's own speaker output. Six self-interrupts
in ~90s, e.g.:
```
18:14:05.358 capture heartbeat avg_rms=0.0069 speaking=True
18:14:05.663 barge-in detected  -> cancel_all epoch=1,2
18:14:38.426 barge-in detected  (preceding avg_rms=0.0080, speaking=True)
18:15:11.305 barge-in detected  (preceding avg_rms=0.0051, speaking=True)
```

**Why now (it was validated on Linux):** this is the first run on the **Windows**
side, whose `config.local.json` still carries the *pre-FIFO* `aec_ref_delay_ms=19`
(the Linux side was re-validated at **`ref_delay=0`** after the FIFO far-ref
rewrite — see the device-adaptive-barge-in handoff). A misaligned AEC reference
leaves more residual echo, which trips the DTD. The barge-in's per-device chart
was tuned/validated on the Linux ALC285; the Windows audio stack (different latency
+ `ref_delay=19`) was never re-calibrated. **This is NOT a regression from this
session's code work** (think=false / smart-routing / P2 polish never touch
AEC/barge/ASR).

### Secondary (the amplifier): SenseVoice hallucination cascade → "two outputs"
The SenseVoice 2nd-pass (`asr_final_backend=sense_voice` on this box) hallucinates
plausible-but-wrong sentences from echo / garbled audio. After each self-interrupt
the cancelled tail's echo becomes a spurious short "final", each firing a NEW
response, which self-interrupts again → a runaway loop. Evidence (raw zipformer →
SenseVoice final):
```
'BEING'              -> 'I.'
'THE LOW IS THIS CORDOOR KING' -> 'Hello, is this code working?'
'THIRTEEN'           -> 'Whatt.'
'W STORY OLD'        -> 'O.'
'AH'                 -> 'Okay.'
'COME TO TEN'        -> '1.'
```
The "two outputs one after another" is partly (a) streaming-TTS speaking a 2-sentence
answer (correct), but mostly (b) this cascade: clarification → self-interrupt →
echo-final → another clarification ("I'm not sure what code…", "Could you please
tell me which code…", "I'm still not entirely clear…").

## What WORKED (validated this session)
- **think=false latency fix:** `"Give me a long story"` routed to the **main** tier
  (gemma4:12b) and the first story sentence (*"In a kingdom where the sun never
  set…"*) played ~**3.4 s** after the LLM call started — not the old ~9 s. The fix
  holds live.
- **Prosody endpointing is active** (`endpoint detector: prosody`) — but the
  self-interrupt chaos makes turn-taking quality unreadable; re-assess after the
  self-interrupt is fixed.

## Next-session plan (needs the mic — priority order)

### P0 — kill the self-interrupt on the Windows box (the blocker for everything else)
1. **Re-calibrate AEC + barge-in for THIS box** (it was only ever done on Linux):
   - `python -m tools.echo_probe` (echo-only: the assistant talks to itself). Read
     the ERLE sweep + per-frame `(D, z_raw, z_resid, z_coh)` and the echo-only-D-vs-K
     headroom. Pick the `aec_ref_delay_ms` that maximizes ERLE (Linux peaked at 0;
     don't assume — let the Windows probe decide). Target **self_interruptions=0**.
   - If echo-only D approaches K=5.0 even at the best ref_delay, raise `dtd_k` and/or
     re-check the warm-up seeding (`dtd_*` knobs are `SherpaConfig` defaults; the
     probe shows the headroom).
   - Write the calibrated `aec_ref_delay_ms` (+ any `dtd_k`) into `config.local.json`.
2. **Verify on hardware:** echo-only `self_interruptions=0`, then `python -m core
   --engine sherpa` — talk over a long reply: it MUST still cut (<~1 s) and MUST NOT
   self-interrupt. (Replay `run-20260608-181250.wav` through echo_probe to iterate
   without re-talking.)

### P1 — break the SenseVoice cascade
3. On this box, **disable the 2nd pass** (`config.local.json asr_final_backend=""`)
   OR build the deferred **agreement-guard** (`core/asr_text.py`: accept the 2nd
   pass only when it agrees with / clearly improves the streaming final). NB even
   disabled, the raw zipformer is garbled on echo — the PRIMARY fix is the
   self-interrupt (then no echo-finals are produced at all).
4. Raise `asr_final_min_sec` so sub-threshold echo clips use the streaming final.

### P2 — don't act on echo/spurious finals at all
5. The cascade only exists because echo-tail finals reach the brain. Confirm finals
   aren't produced from echo during/just-after playback (gate finals while
   `_speaking`, or strengthen the addressing/input gate so short echo phrases are
   ingested-not-acted).

## Next steps (pick up here)
1. **P0 self-interrupt recalibration** on the Windows box (echo_probe → ref_delay →
   verify). Everything else is unusable until this is fixed.
2. Then P1 (SenseVoice guard/disable) and re-assess prosody turn-taking live.
3. (Unblocked, separate track) smart-routing audit #4 — cloud-profile lever
   activation + bench evidence (needs cloud keys).
