# Audio pipeline — capture cleanup, echo cancellation & fluid TTS

How the capture and playback paths are cleaned today, **why Microsoft Teams
sounds better than a raw-mic app on the same laptop**, and the concrete knobs to
close that gap on desktop and mobile. This is the current-truth companion to
`docs/TTS.md`; barge-in decisions live in `docs/adr/0004`–`0006`,
`0011`–`0013`, and `0019` (the DTLN-era internals doc is archived at
`docs/archive/open_speaker_barge_in.md`).

> **2026-07-06 (ADR-0013, live-validated):** for the bare-open-speaker path the
> answer to "why Teams is better" below is now the shipped design — capture
> routes through the Linux **OS voice-comm canceller** (PipeWire
> `module-echo-cancel`) **instead of** the in-app APM (`aec_enabled=false`),
> and barge-in is the continuous no-duck **word-cut**
> (`barge_word_cut_enabled=true`). The in-app APM remains the in-app
> fallback and the `open_speaker` profile still selects it; see
> `docs/adr/0013` for the recipe and what is (and isn't) validated. Desktop
> Windows Communications capture is currently unavailable and fails readiness;
> see `docs/adr/0019`.

## TL;DR — why Teams is better, and the three highest-leverage moves

Teams doesn't have better DSP. It **captures from the OS *voice-communication*
path**, so the OS/driver runs AEC + noise-suppression + AGC on every frame
*before the app reads a sample*. The desktop core historically opened the **raw
mic** (`core/engines/sherpa.py::_open`) and ran no steady-state cleanup while the
user was just talking. (Notably, this project's own Android app already does the
Teams thing — `mobile/lib/assistant.dart` uses the `voiceCommunication` source.)

1. **Get a clean input signal** — route capture through PipeWire
   `module-echo-cancel` on Linux or the in-app WebRTC APM, and/or enable the GTCRN
   denoiser. The desktop Windows OS path is blocked by ADR-0019.
2. **Make TTS fluid** — `tts_target_rms` (per-sentence loudness normalization) is
   now on by default in `config.json`; a raised-cosine fade de-clicks barge-in
   cuts.
3. **In-app fallback echo cancellation** — `aec_backend="apm"` runs the WebRTC
   AudioProcessingModule (AEC3 + RES + NS + AGC2 + HPF), which tolerates a
   nonlinear open laptop speaker where the linear NLMS filter measures ~0 dB
   ERLE (use it when the ADR-0013 OS voice-comm path isn't set up).

## The capture chain (per 0.1 s block, `core/engines/sherpa.py`)

```
mic ─► InputAGC / input_gain ─► anti-alias resample (soxr→16 kHz)
     ─► [tee mic_raw → coherence/DTD barge detector]   (in-app AEC paths only)
     ─► AEC / WebRTC APM ─► GTCRN denoiser ─► recorder ─► VAD / ASR / speaker-ID
```

On the ADR-0013 OS-capture path the OS canceller runs *before* "mic" above
(`aec_enabled=false` → the in-app AEC hop and the coherence/DTD tee are
bypassed) and barge-in is the ASR **word-cut** on the OS-cancelled capture — no
acoustic gate is in the loop.

- **Anti-alias resampler** (`core/audio_frontend.py::AudioResampler`) prefers
  `soxr` (stateful, no per-block seam), then `scipy.resample_poly`. Readiness
  accepts either anti-aliased backend and fails only when neither is available;
  linear interpolation is diagnostic fallback, not a ready production path.
- **AEC runs only while the assistant is speaking** (the far-end reference has
  energy). This is intentional: the deep cancellers distort clean near-end speech,
  and on the in-app AEC paths barge-in keys off `mic_raw` (pre-AEC). The
  exceptions: the WebRTC APM with `apm_always_on=true` runs every block (see
  below), and the ADR-0013 OS-capture path has no in-app AEC at all — its
  word-cut barge reads the OS-cancelled stream, not `mic_raw`.
- **GTCRN denoiser** is enabled in tracked config and provisioned by
  `python -m tools.setup_models --denoise-model`; readiness fails if the selected
  model is missing. Speaker enrollment provenance includes the active denoiser,
  so a changed front end requires re-enrollment (ADR-0018). It is skipped when
  the always-on APM already owns noise suppression.

## Smart generic level calibration (`input_calibrate`)

So "low OS gain, never clips" works on any mic without hand-tuning: with
`input_calibrate=true` the engine listens for `input_calibrate_sec` (~1.5 s) of
room tone at startup, measures **this device's** quiet floor
(`compute_input_calibration`, a low-percentile RMS robust to a stray word), and
sets the `InputAGC` noise gate just above it — the device-generic operating point
the AGC otherwise cold-starts wrong. It also measures the ADC clip fraction and
emits an **`input_clipping`** metric into the run bundle (a hot ADC is the #1
silent STT-garbler, and the boost-only AGC can't fix it — the OS level must come
down). Off by default (adds the calibration window to startup); only moves the
AGC floor when `input_agc` is also on, otherwise it just logs the measurement.
Steady-state clipping during a run surfaces the same `input_clipping` metric from
the capture heartbeat. Gain-ownership rules per path: on the **in-app APM
fallback** don't pair `input_agc=true` with open-speaker barge-in — its
time-varying gain perturbs the coherence detector; let the APM's AGC2 own gain.
On the **ADR-0013 Linux OS-capture path** no coherence detector is in the loop:
Linux `module-echo-cancel` is loaded AEC-only (NS+AGC off), so keep
`input_agc=true` + `input_calibrate=true` or the VAD goes deaf. The former Windows
WASAPI guidance is superseded by ADR-0019.

## Echo cancellation backends (`aec_backend`)

| backend | what it is | open-speaker? | deps |
| --- | --- | --- | --- |
| `nlms` (default) | dependency-free NumPy adaptive filter | ~10–20 dB on a headset/near-field; **~0 dB and diverges on a nonlinear open speaker** | none |
| `dtln` | deep ONNX canceller | +6 dB ERLE, handles nonlinearity | `tools.setup_models --aec-model` |
| **`apm`** | **WebRTC AudioProcessingModule (AEC3 + RES + NS + AGC2 + HPF)** | **40–53 dB ERLE in-vitro; the in-app fallback** (the validated open-speaker path is OS voice-comm capture + word-cut, ADR-0013) | `pip install livekit` |

The **APM** is the recommended **in-app fallback** for the open speaker (when
the ADR-0013 OS voice-comm path isn't set up). It conforms to the same
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
- `aec_auto_delay=true` (default) measures the speaker→mic lag **on-device by
  normalized cross-correlation** (`AecDelayCalibrator`, ADR-0012) and feeds it
  back into the far-end read delay, demoting `aec_ref_delay_ms` to a seed —
  a mis-set delay self-corrects during the run (validated 40→~120 ms on
  run-20260702-004345).

**Reproducible open-speaker config:** the live-validated path is the OS
voice-comm capture + no-duck word-cut recipe in **ADR-0013** (`aec_enabled=false`,
`apm_always_on=false`, `barge_word_cut_enabled=true`, plus PipeWire
`module-echo-cancel` on Linux). The
committed `open_speaker` device profile — `python -m core --engine sherpa
--device open_speaker` — remains the **in-app fallback**: it turns on
`aec_backend="apm"` + `apm_always_on` (no OS setup needed, but the APM's
always-on NS is what smeared the near-end user during double-talk; see
ADR-0013's context).

## The OS voice-comm path (the Teams-equivalent — the validated open-speaker path per ADR-0013, still opt-in)

Replaces the in-app APM on the open-speaker path (run with `aec_enabled=false`
so the word-cut barge is live) — let the OS do the DSP:

- **Linux (PipeWire):** load the WebRTC echo-cancel module once and point capture
  at the virtual source:
  ```
  pactl load-module module-echo-cancel aec_method=webrtc \
      source_name=ec_source sink_name=ec_sink
  ```
  Then set `input_device` to the "Echo Cancellation Source" node.
  `python -m tools.doctor` checks the active source and sink when word-cut or
  `capture_voice_comm` selects the route.
- **Windows:** the previously documented sounddevice Communications keyword is
  unsupported. Profiles that select it fail readiness until a verified capture
  implementation exists (ADR-0019).
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
pip install soxr                                # optional faster resampler
# Validated open-speaker path (ADR-0013): OS echo-cancel + no-duck word-cut
pactl load-module module-echo-cancel aec_method=webrtc \
    source_name=ec_source sink_name=ec_sink \
    aec_args="webrtc.noise_suppression=false webrtc.gain_control=false"
# config.local.json: aec_enabled=false, apm_always_on=false,
#                    barge_word_cut_enabled=true, input_agc=true, input_calibrate=true
python -m core --engine sherpa --input-device pipewire --output-device pipewire
python -m tools.doctor                          # checks the EC module is loaded

# In-app APM fallback (no OS setup; NS smears the near-end during double-talk):
pip install livekit
python -m core --engine sherpa --device open_speaker --enroll   # re-enroll post-cleanup
```

On Windows, selecting the former Communications/word-cut recipe reports NOT
READY until a verified OS capture implementation exists; see ADR-0019.
