# Audio pipeline — capture cleanup, echo cancellation & fluid TTS

How the capture and playback paths are cleaned today, **why Microsoft Teams
sounds better than a raw-mic app on the same laptop**, and the concrete knobs to
close that gap on desktop and mobile. This is the current-truth companion to
`docs/open_speaker_barge_in.md` (barge-in internals) and `docs/TTS.md`.

## TL;DR — why Teams is better, and the three highest-leverage moves

Teams doesn't have better DSP. It **captures from the OS *voice-communication*
path**, so the OS/driver runs AEC + noise-suppression + AGC on every frame
*before the app reads a sample*. The desktop core historically opened the **raw
mic** (`core/engines/sherpa.py::_open`) and ran no steady-state cleanup while the
user was just talking. (Notably, this project's own Android app already does the
Teams thing — `mobile/lib/assistant.dart` uses the `voiceCommunication` source.)

1. **Get a clean input signal** — route capture through the OS voice-comm path
   (PipeWire `module-echo-cancel` on Linux, WASAPI Communications on Windows) or
   the in-app WebRTC APM, and/or enable the GTCRN denoiser.
2. **Make TTS fluid** — `tts_target_rms` (per-sentence loudness normalization) is
   now on by default in `config.json`; a raised-cosine fade de-clicks barge-in
   cuts.
3. **Production echo cancellation** — `aec_backend="apm"` runs the WebRTC
   AudioProcessingModule (AEC3 + RES + NS + AGC2 + HPF), which tolerates a
   nonlinear open laptop speaker where the linear NLMS filter measures ~0 dB ERLE.

## The capture chain (per 0.1 s block, `core/engines/sherpa.py`)

```
mic ─► InputAGC / input_gain ─► anti-alias resample (soxr→16 kHz)
     ─► [tee mic_raw → coherence/DTD barge detector]
     ─► AEC / WebRTC APM ─► GTCRN denoiser ─► recorder ─► VAD / ASR / speaker-ID
```

- **Anti-alias resampler** (`core/audio_frontend.py::AudioResampler`) prefers
  `soxr` (stateful, no per-block seam) → `scipy.resample_poly` → naive linear.
  **Install `soxr`** (`pip install soxr`, it's in `requirements.txt`) or every
  0.1 s block is refiltered independently — an aliasing seam that taxes sibilants.
  `python -m tools.doctor` flags a missing `soxr`.
- **AEC runs only while the assistant is speaking** (the far-end reference has
  energy). This is intentional: the deep cancellers distort clean near-end speech,
  and barge-in keys off `mic_raw` (pre-AEC) anyway. The exception is the WebRTC
  APM with `apm_always_on=true`, which runs every block (see below).
- **GTCRN denoiser** is off by default and not shipped. Fetch + enable:
  `python -m tools.setup_models --denoise-model`, then `denoise_enabled=true` +
  `denoise_model=<path>`. **Re-enroll your voice afterward** (the speaker
  embedding shifts post-denoise). Skipped automatically when the always-on APM
  already owns noise suppression.

## Echo cancellation backends (`aec_backend`)

| backend | what it is | open-speaker? | deps |
| --- | --- | --- | --- |
| `nlms` (default) | dependency-free NumPy adaptive filter | ~10–20 dB on a headset/near-field; **~0 dB and diverges on a nonlinear open speaker** | none |
| `dtln` | deep ONNX canceller | +6 dB ERLE, handles nonlinearity | `tools.setup_models --aec-model` |
| **`apm`** | **WebRTC AudioProcessingModule (AEC3 + RES + NS + AGC2 + HPF)** | **40–53 dB ERLE in-vitro; the production path** | `pip install livekit` |

The **APM** is the recommended open-speaker backend. It conforms to the same
`process_16k(near, far)` seam as the other cancellers (`core/engines/_apm.py`),
fed the time-aligned far-end reference from the existing `FarEndRing`. It fails
**open**: if `livekit` is missing, `build_aec` returns `None` and the path is
byte-identical to no-AEC.

- `apm_noise_suppression`, `apm_high_pass_filter`, `apm_gain_control` toggle the
  sub-stages; the echo canceller itself is always on for this backend.
- `apm_always_on=true` runs the whole APM on **every** capture block (not just
  during playback), so its NS/AGC/HPF also clean the user's own idle-path
  utterance — the desktop analog of the OS voice-comm path. When the assistant is
  silent the far reference is ~zeros, so the echo canceller self-cancels to a
  no-op (measured ~93 % idle passthrough).
- `aec_auto_delay=true` (default) feeds the coherence detector's measured
  speaker→mic lag back into the far-end read delay, so a mis-set
  `aec_ref_delay_ms` self-corrects during the run.

**Reproducible open-speaker config:** select the committed `open_speaker` device
profile — `python -m core --engine sherpa --device open_speaker` — instead of
hand-editing `config.local.json`. It turns on `aec_backend="apm"` +
`apm_always_on`.

## The OS voice-comm path (optional, the Teams-equivalent)

An alternative to the in-app APM — let the OS do the DSP:

- **Linux (PipeWire):** load the WebRTC echo-cancel module once and point capture
  at the virtual source:
  ```
  pactl load-module module-echo-cancel aec_method=webrtc \
      source_name=ec_source sink_name=ec_sink
  ```
  Then set `input_device` to the "Echo Cancellation Source" node.
  `python -m tools.doctor` checks the module is loaded (only when
  `capture_voice_comm` is requested).
- **Windows:** set `capture_voice_comm=true` → the engine opens the stream with
  `sd.WasapiSettings(communications=True)` (the AEC/NS Communications category).
- **macOS/iOS:** the `VoiceProcessingIO` AudioUnit / `AVAudioSession .voiceChat`
  mode. Wired on Android (below); desktop macOS not yet.

## Fluid TTS output (`core/engines/sherpa.py::_synthesize`)

The on-device VITS voice has three avoidable artifacts:

- **Uneven / "not fluid" loudness** — the model emits a different amplitude per
  sentence (>2 dB swing). `tts_target_rms=0.12` (now default) normalizes every
  sentence with a soft-knee limiter. Tradeoff: switches to whole-clip synth
  (~0.1 s more first-audio). `0.0` restores the streaming path.
- **Clicks / crackle** — deterministic sample-level impulse spikes.
  `tts_declick=true` (default) repairs them; `tts_declick_threshold=0.22` sits
  safely above dense-consonant energy (~0.14) and below real spikes (0.5–0.95),
  so consonants aren't smeared into a "robotic" timbre.
- **Barge-in click** — a hard FIFO flush is a step discontinuity.
  `barge_fade_ms=4` applies a raised-cosine fade-out to the playback tail.

Residual robotic timbre after this is the **voice model's** intrinsic quality (a
model choice — evaluate `libritts_r-high`, Piper, or Kokoro within the CPU
budget), not a pipeline bug. Note: the "white noise / hiss" once attributed to
FIFO underruns was the impulse spikes (now declicked); the underrun warning has
never fired in a committed run.

## Mobile parity (`mobile/`)

- **Capture is ahead of desktop:** `assistant.dart` already requests the Android
  `voiceCommunication` source + `AcousticEchoCanceler`/`NoiseSuppressor`/
  `AutomaticGainControl`.
- **TTS post-processing** (`tts_isolate.dart`) now ports the desktop `declick` +
  `normalize_rms` onto the worker isolate (parity with the core).
- **Still open:** iOS `AVAudioSession .voiceChat` wiring (iOS mic is currently
  raw), and auto-calibrating the hardcoded `_bargeInRms`.

## Quick reference — enabling everything on a Linux laptop

```bash
pip install soxr livekit                       # resampler + WebRTC APM
python -m tools.setup_models --denoise-model   # optional GTCRN (or rely on the APM)
python -m core --engine sherpa --device open_speaker --enroll   # re-enroll post-cleanup
python -m tools.doctor                          # verify soxr + livekit APM are READY
```
