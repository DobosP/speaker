# TTS and speaker playback

The assistant synthesizes speech locally (Kokoro, Piper, MeloTTS, or Supertonic) and plays it on your speakers. If you see **Assistant:** text but hear nothing, use this guide.

## Quick check (no microphone)

```bash
cd /home/dobo/work/speaker
python scripts/test_tts_playback.py --list-devices
python scripts/test_tts_playback.py --tts-backend kokoro --output-device 4
```

You should see a log line like:

```text
🔊 Playing (kokoro, sounddevice, 1.20s, 4:HDA Intel PCH: ALC285 Analog ...): tmpXXXX.wav
```

If that works but `main.py` does not, the runtime was probably sending audio to a different output than your headphones.

## Root cause on many laptops (ROG Strix, PipeWire/PulseAudio)

PortAudio often sets:

- **Default input** → built-in analog mic (e.g. index **4**, `ALC285 Analog`)
- **Default output** → HDMI/monitor (e.g. index **0**, `BenQ GL2580`)

TTS used to follow the **output** default, so voice went to the monitor while you listened on laptop speakers.

The app now calls `resolve_output_device()` at startup: when `output_device` is unset, it routes TTS to the **same analog card as the mic** (or prints which index it chose).

Override anytime in `config.json`:

```json
{
  "input_device": 4,
  "output_device": 4
}
```

Or on the CLI:

```bash
python main.py --output-device 4 --tts-backend kokoro ...
```

## List devices

```bash
python main.py --list-devices
# or
aplay -l
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Pick an index with `max_output_channels > 0` that matches where you actually hear sound.

## CLI flags

| Flag | Purpose |
|------|---------|
| `--no-tts` | Text-only assistant (no synthesis/playback) |
| `--tts-backend piper` | Force Piper instead of Kokoro |
| `--tts-backend kokoro` | Force Kokoro ONNX |
| `--output-device N` | PortAudio output index for TTS |
| `--playback-backend sounddevice` | PortAudio playback (default for `auto`) |
| `--playback-backend pygame` | SDL/pygame playback |

## Live conversation debug (`SPEAKER_TTS_DEBUG`)

When assistant text prints but you hear no voice during `python main.py`, capture a full TTS trace:

```bash
cd /home/dobo/work/speaker
./scripts/tts_debug_live.sh --llm-model gemma3:latest --stt-model base --runtime-profile balanced
# or:
SPEAKER_TTS_DEBUG=1 python main.py ... 2>&1 | tee /tmp/speaker-tts-debug.log
```

Enable via any of:

| Source | Example |
|--------|---------|
| Environment | `SPEAKER_TTS_DEBUG=1` |
| `config.json` | `"tts_debug": true` |
| CLI | `--tts-debug` |

Console one-liners (when debug is on): `🔊 TTS: enqueued`, `🔊 TTS: playing (...)`, `🔊 TTS: cancelled (barge-in)`, `🔊 TTS: FAILED: ...`

Structured logs use loggers `speaker.tts` and `speaker.audio` on stderr.

### Grep after your test run

```bash
LOG=/tmp/speaker-tts-debug.log

# Did LLM phrases reach the TTS queue?
grep -E 'llm_chunk|queue_enqueue|enqueued' "$LOG"

# Was audio synthesized and played?
grep -E 'synth_|playback_|🔊 Playing|🔊 TTS: playing' "$LOG"

# Cancellations / drops
grep -E 'queue_flush|queue_drop|cancelled|stale_session' "$LOG"

# Failures
grep -E 'FAILED|speak_error|playback_error|silent' "$LOG"

# Echo / barge-in gates (mic path — do not stop TTS, only barge-in)
grep -E 'echo_gate|speech_gate|barge_gate' "$LOG"

# Transport (local_lan does not skip local speakers)
grep transport_broadcast "$LOG"
```

## Logs to look for

- `TTS output: device 4 (...)` — auto-routed playback
- `TTS playback: sounddevice (...)` — engine in use
- `🔊 Playing (...)` — audio actually started
- `❌ TTS playback failed` — engine/device error (details follow)
- `⚠ TTS output appears silent` — synthesis produced a near-empty WAV

## Transport mode

`transport_mode: local_lan` only queues session events; it does **not** disable local speaker playback. Audio still goes through `AudioPlayer` on this machine.

## CPU-only Kokoro

Kokoro ONNX on CPU is expected. Progress lines like `0%/100%` during synthesis are normal. Playback is independent of CUDA.

## Still silent?

1. Run `scripts/test_tts_playback.py` with `--output-device` for each candidate index.
2. Check system volume and that the correct sink is unmuted (`pavucontrol` or `wpctl status`).
3. Try `--playback-backend pygame` if PortAudio routing fails.
4. Save a WAV: `python scripts/test_tts_playback.py --write-wav /tmp/tts.wav --no-play` then `aplay /tmp/tts.wav`.

## Live TTS debug (`main.py` prints text, no audio)

Enable structured logging on the full path: LLM chunks → TTS queue → synthesis → playback.

```bash
chmod +x scripts/tts_debug_live.sh
./scripts/tts_debug_live.sh --llm-model gemma3:latest --stt-model base --runtime-profile balanced

# or
SPEAKER_TTS_DEBUG=1 python main.py ... 2>&1 | tee /tmp/speaker-tts-debug.log
```

Also: `config.json` → `"tts_debug": true`, or `--tts-debug`.

### Log lines to grep after a test run

```bash
LOG=/tmp/speaker-tts-debug.log

# Pipeline
grep -E 'llm_chunk|queue_enqueue|queue_dequeue|speak_start|speak_skipped' "$LOG"

# Synthesis / playback
grep -E 'synth_start|synth_done|playback_start|playback_end|playback_silent|playback_error' "$LOG"

# Cancels
grep -E 'queue_drop|queue_flush|session_invalidate' "$LOG"

# Gates & policy
grep -E 'echo_gate_blocked|speech_gate_blocked|dialogue_barge' "$LOG"

# Console one-liners
grep '🔊 TTS:' "$LOG"
```

| Event | Likely issue |
|-------|----------------|
| `llm_chunk` but no `queue_enqueue` | Cancel flag or queue full |
| `queue_enqueue` but no `queue_dequeue` | Worker stuck or session invalidated |
| `speak_skipped reason=no_player` | TTS init failed at startup |
| `speak_skipped reason=cancel_generation` | Barge-in / new turn before play |
| `synth_error` / `playback_error` | Backend or device failure |
| `playback_silent_wav` | Model produced empty audio |
| No `playback_start` after `speak_start` | Exception before play |
