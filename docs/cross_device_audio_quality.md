# Cross-device audio quality: concepts + implementation plan

> **One-line thesis.** Teams/Zoom don't *calibrate per device* — they defer to a per-device, OEM-tuned **OS voice-communication path** for the device-specific acoustics, then run their **own WebRTC APM** on top as a single device-agnostic loudness/echo policy. This repo today does *neither* cleanly on output: it grabs the **raw** default device and levels the TTS with a per-sentence **linear RMS** hack (`tts_target_rms=0.12`) — no LUFS, no limiter, no per-device target, no OS comm path on render. Closing that two-layer gap is the whole job. Most of the win is configuration + ~200 lines of pure-DSP port, not new infrastructure.

---

## 1. The core diagnosis — why output isn't well calibrated today

The complaint ("speaker output is not well calibrated; I want consistent Teams/Zoom-grade sound across devices, even on old hardware") decomposes into **four concrete defects**, all on the OUTPUT/render side, plus one architectural root cause.

### 1.1 We level with linear RMS, not perceptual loudness

`core/audio_frontend.py::normalize_rms` (called from `sherpa.py:2716-2718`) scales each TTS sentence so its **linear RMS** approaches `tts_target_rms=0.12` (`config.json:462`). This is a crude one-shot cousin of real loudness normalization:

- **Linear RMS, not K-weighted gated loudness.** Two utterances at the same RMS can sound very different in loudness; the ear is not flat. Broadcast/RTC voice uses **ITU-R BS.1770 / EBU R128 LUFS** (K-weighted, gated) or WebRTC AGC2's dBFS speech-level estimator precisely because RMS is perceptually wrong.
- **One-shot, per-sentence, no slew / no memory.** Each sentence is normalized in isolation. There is no inter-sentence state and no bounded gain ramp, so the gain can step between sentences. AGC2 by contrast slews toward target at a bounded rate.
- **A single global target, no per-device value.** `tts_target_rms` is one number. It is the right level for *no* specific speaker. A tinny laptop speaker, a BT speaker, and a USB headset all get the same 0.12.

It does the one thing it was actually built for — stabilizing the *echo* level for open-speaker barge-in (see the docstring) — but it was never a loudness standardizer.

### 1.2 There is no true-peak limiter on the output chain

The *only* peak control on render is the static `tanh` soft knee inside `apply_gain_soft_limit` (`audio_frontend.py:42-64`), reused by `normalize_rms`. That is a **per-sample saturator**, not a look-ahead limiter with attack/release and a true-peak (inter-sample) ceiling. A grep of `sherpa.py`/`config.json` finds **no** `output_gain` / `limiter` / `drc` keys at all. Consequence: loud VITS sentences hard-clip / "buzz" on weak speakers — exactly the "doesn't sound right on old hardware" failure. (Note this is distinct from the VITS **impulse spikes** already handled by `declick`; this is *level*-driven clipping, which `declick` does not address.)

### 1.3 There is no per-device output calibration

`device_profiles` (`config.json:151-310`) shallow-merge per section and are *available* for output tuning, but their `sherpa` blocks only set ASR/TTS threads, `resampler_quality`, and AEC flags. `tts_target_rms` appears in **no** profile. There is no per-device loudness/limiter target, no output EQ, no speaker-class detection.

### 1.4 We open the RAW device, not the OS voice-comm path (the root cause)

`core/engines/sherpa.py:1232-1312` opens the **raw default** device via `sounddevice.InputStream`/`OutputStream` with a hand-rolled fallback-rate chain (`_capture_attempts` + `_RecoveringInputStream`). Device selection is just `config.input_device`/`output_device` (`config.json:616-617`). There is:

- no **role-based** selection (no `eCommunications` "default communication device");
- no **hot-plug / route-change** handling (no `IMMNotificationClient` equivalent);
- no shared cross-platform **connector** seam.

`capture_voice_comm` maps to `sd.WasapiSettings(communications=True)` but **only on the INPUT stream** (`sherpa.py:1283-1293`) and is **OFF by default** (`config.json:630`). The **OUTPUT** stream (`sherpa.py:2537-2540`) is opened with **no** communications category — so the OS render-side **speaker protection / loudness EQ / DRC** that the OEM tuned for that physical speaker never engages. On Linux, PipeWire `module-echo-cancel` is a *documented alternative* (`docs/audio_pipeline.md:97-113`) but never auto-detected/loaded, and `input_device` is never pointed at the EC virtual source. macOS VPIO is unwired (`docs:113`). Mobile already does the Android `voiceCommunication` thing (`docs:136-144`) — desktop lags.

**The architectural mistake:** the two-layer model is *conflated*. The repo grabs the raw device (which on Windows is *forbidden* from doing any AEC/AGC/NS) and bolts its own APM on top, instead of preferring the OS path (Layer 1) and treating its own APM as a device-agnostic safety net (Layer 2).

---

## 2. The target architecture — the two-layer model, mapped to our seams

The pattern every production RTC app uses:

```
                 ┌─────────────────────── Layer 1: OS voice-comm path (per device) ──────────────────────┐
   physical  ──▶ │ Windows: AudioCategory_Communications + eCommunications role  →  AEC/NS/AGC/Deep-NS    │
   device        │ Linux:   PipeWire module-echo-cancel  →  virtual echo-cancel SOURCE (+ ref wired by PW) │ ──▶ near
                 │ Apple:   VoiceProcessingIO / .voiceChat                                                 │
                 │ Android: AudioSource.VOICE_COMMUNICATION (already done in mobile/)                      │
                 │   …plus RENDER-side: OEM speaker protection / loudness EQ / DRC for THAT speaker        │ ◀── play
                 └────────────────────────────────────────────────────────────────────────────────────────┘
                                                     ▲                    │
        ┌────────── Layer 2: app APM (device-agnostic policy) ───────────┴───────────┐
  near ─┤ CAPTURE:  WebRTC APM (AEC3 + RES + NS + HPF)   — already present (_apm.py)  │
        │ OUTPUT:   LUFS/AGC2-style loudness target → true-peak limiter — TO BUILD    │ ─▶ device
        └────────────────────────────────────────────────────────────────────────────┘
```

**Consistency = composition:** Layer 1 fixes the *device* (per-device acoustics, hot-plug, SR negotiation, reference wiring — things you cannot do portably); Layer 2 fixes the *policy* (one echo behaviour, one loudness target).

### Where each piece slots into the existing code

| Concern | New/changed seam | Existing anchor |
|---|---|---|
| **AudioConnector** — "select + open the OS voice-comm path for the chosen device, per platform; honor the comm role; react to route changes" | New small seam in `core/engine.py` (the `AudioEngine` interface lives here) | `sherpa.py:1232-1312` (raw open today), `config.input_device/output_device` |
| **Capture Layer 2** (AEC3+RES+NS+HPF) | *Already exists* | `core/engines/_apm.py` (livekit `rtc.AudioProcessingModule`), wired via `core/engines/_aec.py` `EchoCanceller` + `FarEndRing`; on under `open_speaker` profile (`config.json:248-259`) |
| **Output loudness standardization** (LUFS / AGC2 adaptive-digital target) | New stage in `core/audio_frontend.py`, runs on the synth/producer thread like `declick`/`normalize_rms` today (NOT on the audio callback) | replaces/augments `normalize_rms`; called from `sherpa.py:2716-2718` |
| **Output true-peak limiter** | New final output stage in `core/audio_frontend.py`, after loudness leveling | none today — `apply_gain_soft_limit` (`:42-64`) is the placeholder it replaces |
| **Per-device output target** | New `sherpa` keys in `device_profiles` | `config.json:151-310` profile mechanism (shallow-merge), currently output-blind |
| **OS render comm category** | Add comm category to the OUTPUT stream open | `sherpa.py:2537-2540` |
| **Adaptive playout** | Extend the FIFO read path | `_aec.py::PlaybackFIFO` (`playback_fifo_sec=1.0`, `config.json:646`), underrun metric at `sherpa.py:2599-2606` |

Design rule for the new output DSP: **keep it in `core/audio_frontend.py` on the producer/synth thread**, exactly where `declick` and `normalize_rms` run today, so it never touches the real-time audio callback. The capture APM stays frame-locked to 10 ms via `_apm.py`; the output leveler/limiter operate per synthesized buffer, not per callback block.

---

## 3. Prioritized implementation plan

Ordered highest-leverage / lowest-risk first. Land the AudioConnector refactor (§3.6) *incrementally behind* these wins, not as a big bang.

### P1-A — Output loudness target + true-peak limiter (study-port WebRTC AGC2) — **the single highest-value lift**

**What.** Replace/augment `normalize_rms` with (1) a perceptual **loudness target** and (2) a **look-ahead true-peak limiter** as the final output stage. This is the direct fix for both "not well calibrated" *and* "never clips on old hardware."

**Which OSS.** Do **not** route TTS through the livekit-bound APM for this — the binding exposes only 4 bools + `set_stream_delay_ms` (no headroom/target/max_gain/limiter knobs) and AGC2-on-capture also drives a fragile hardware mic-volume recommendation (PipeWire ADC already stuck at 13% / +30 dB clip on this box). Instead **study-port** the two pure-DSP, allocation-free, frame-locked algorithms from the vendored fork:
- `agc2/adaptive_digital_gain_controller.cc` → `ComputeGainDb`: `target_gain = -headroom_db - speech_level_dbfs`, slew-limited (~6 dB/s), noise-capped.
- `agc2/limiter.cc` + `limiter_db_gain_curve.h`: precomputed soft-knee curve (knee≈1 dB, ratio 5:1, attack power 8), per-subframe envelope → per-sample ramp → clamp; true-peak-safe brickwall.

These are ~200 lines of straightforward float math, trivially real-time on CPU, and map cleanly onto a new output stage. Source files for reference are in `tmp/audio_research/webrtc-audio-processing/`. Switch the target from **linear RMS 0.12** to a **speech-dBFS target (≈ −16 to −18 dBFS)** so output loudness is device-independent and matches the dBFS convention AGC2 uses.

> Optional alternative for the loudness *measure*: bind **`libebur128`** for a true BS.1770 LUFS gated measurement instead of AGC2's dBFS speech-level estimator. Cleaner perceptual match, but adds a native dependency and a measure→correct latency. Given we already need the AGC2 limiter port regardless, the AGC2 dBFS estimator is the lower-friction choice; reach for `libebur128` only if dBFS leveling proves perceptually insufficient in listening tests.

**Config keys (new).** `tts_loudness_target_dbfs` (e.g. `-17`), `tts_output_headroom_db`, `tts_limiter_ceiling_dbfs` (e.g. `-1`), `tts_loudness_slew_db_per_s`. Keep `tts_target_rms` as a deprecated fallback (`<=0` disables the old path).

**Files.** `core/audio_frontend.py` (new `level_to_dbfs` + `true_peak_limiter`, replacing the `normalize_rms`/`apply_gain_soft_limit` call site), `core/engines/sherpa.py:2716-2718` (call the new stage), `config.json:462`.

**Effort.** ~1–1.5 days incl. unit tests over fixture WAVs (assert post-limiter true-peak ≤ ceiling, inter-sentence loudness variance shrinks vs RMS baseline). The limiter alone is the load-bearing anti-clip piece — ship it even if loudness leveling lands second.

### P1-B — Default the OS voice-comm path ON for capture (Windows), resolve via the eCommunications role

**What.** `capture_voice_comm` already maps to `WasapiSettings(communications=True)` on the input stream (`sherpa.py:1283-1293`). **Default it ON for the sherpa engine on Windows**, and resolve the input device via the **eCommunications role** rather than the plain default, so the user's chosen "headset-for-calls" is honored. This engages the OS Communications-mode AEC/AGC/NS/Deep-NS APOs per device — the single biggest "sounds like Teams" lever on Windows, nearly wired already.

**Which OSS.** None — `sounddevice`/WASAPI is enough for the category flag. Role resolution + hot-plug (`IMMNotificationClient`) is the AudioConnector's job (§3.6) and can land after the flag default.

**Config keys.** Flip `capture_voice_comm` default in the Windows profile; add `input_device_role: "communications"`.

**Files.** `config.json` (Windows `device_profiles`), `sherpa.py:1283-1293`.

**Effort.** ~0.5 day for the default flip; role resolution rides with §3.6.

### P1-C — Opt the OUTPUT stream into the comm category (all desktop OSes)

**What.** Open the **render** stream with the Communications category so the OS render-side **speaker protection + loudness EQ + DRC** (OEM-tuned for that physical speaker) engages. This directly addresses the output-calibration complaint and is the cheapest per-device output calibration available — Teams/Zoom rely on exactly this rather than hand-rolling per-device EQ.

**Files.** `sherpa.py:2537-2540` (output stream open; add `WasapiSettings(communications=True)` on Windows, the comm category on the relevant platform).

**Effort.** ~0.5 day on Windows. On Linux this is implicitly handled by routing through the PipeWire EC sink (§3.4).

### P1-D — Linux: wire PipeWire `module-echo-cancel` as a first-class path

**What.** On start, detect (and optionally load) `module-echo-cancel` and **auto-point `input_device` at the virtual echo-cancel SOURCE**; PipeWire wires the playback reference internally. When present, **disable our own Layer-2 AEC** (it's redundant — same WebRTC AEC3 the `apm` backend already wraps, now run by the OS with the reference wired for us), removing the in-app time-alignment burden that drove the open-speaker barge-in P1.

**Which OSS.** None new — it's the same `aec/libspa-aec-webrtc` (WebRTC AEC3). Tune via `webrtc.*` module args (`gain_control`, `noise_suppression`, `high_pass_filter`, `extended_filter`, `voice_detection`).

**Config keys.** `linux_pipewire_ec: "auto" | "load" | "off"`; when active, set `aec_backend` to passthrough.

**Files.** New small detector in the AudioConnector seam (§3.6) or a Linux helper in `core/engines/`; `config.json` (`open_speaker` note `:258`, Linux profile).

**Effort.** ~1 day (detect + point input device + skip-redundant-AEC; tested via the autotest PipeWire virtual-cable rig already in `tools/autotest/`).

### P2-A — Enable APM AGC2 (capture loudness leveling), gated/validated

**What.** APM AGC2 (`apm_gain_control`) is OFF by default (`config.json:255` / `_apm.py:149`), so consistent capture-loudness leveling is unused. Consider enabling it — it composes with the divergence-guard "amplifies" flag (`_aec.py:567-577`). **Validate first:** the livekit binding ties AGC2 to a hardware mic-volume recommendation that has been fragile on this PipeWire box (ADC stuck 13% / +30 dB clip), so don't blind-default it on. The strong capture path is otherwise present and best-in-class — its only other gap is that it's gated behind the non-default `open_speaker` profile.

**Files.** `config.json:255`, `_apm.py:149`. **Effort.** ~0.5 day + a live mic A/B.

### P2-B — Per-device OUTPUT target in `device_profiles`

**What.** Once the §3.1 loudness target + limiter exist, expose the **target** (and optionally a gentle output EQ/tilt) as a per-profile `sherpa` key so laptop speaker vs BT vs headset each get an appropriate setpoint. **Prefer leaning on the OS comm-path render EQ (§3.3) before hand-rolling per-device EQ** — per-device EQ is easy to get wrong. Per-device *target* (a single dBFS number) is low-risk; per-device EQ curves are a stretch.

**Files.** `config.json:151-310` profiles (add `tts_loudness_target_dbfs` per profile). **Effort.** ~0.5 day for the target; EQ deferred.

### P2-C — Adaptive playout buffer (kill underruns under CPU load)

**What.** `_aec.py::PlaybackFIFO` is a fixed-depth (`playback_fifo_sec=1.0`) SPSC ring; the callback zero-fills underruns and emits a `playback_underrun` metric (`sherpa.py:2599-2606`). There is **no** adaptive playout — under CPU load underruns were observed (the documented motivation). Add a modest adaptive depth (grow on observed underruns, shrink toward a latency floor when stable) driven by the existing underrun metric as feedback. Defer time-stretch/PLC concealment (§4).

**Files.** `core/engines/_aec.py` (`PlaybackFIFO`), `config.json:646`. **Effort.** ~1 day; the existing metric gives a built-in test signal.

### P2-D (architecture) — Introduce the `AudioConnector` seam

**What.** A small portable abstraction in `core/engine.py`: *"select + open the OS voice-comm path for the chosen device, per platform; honor the comm role; handle hot-plug/route changes."* It backs the Windows role resolution (§3.2), the Linux EC source pointing (§3.4), and future Apple VPIO. The existing `apm` backend stays the always-on Layer 2. **Land it incrementally behind the P1 wins above** — each P1 item works without the full seam; the connector then consolidates the per-platform device logic out of `sherpa.py`'s hand-rolled `_capture_attempts`/`_RecoveringInputStream`.

**Files.** `core/engine.py` (new seam), `sherpa.py:1232-1312` (delegate device open). **Effort.** ~2–3 days, spread across the P1 landings.

---

## 4. What is TOO MUCH — do not over-build

This is a **single-user local assistant**, not a multi-party conferencing client. Explicitly out of scope:

- **DRC / multiband compression beyond the limiter on output.** Low value for a single, known TTS voice — the limiter is the load-bearing anti-clip piece; a full compressor is overkill. Let the OS comm-path render DRC handle device protection.
- **Hand-rolled per-device output EQ curves.** Teams/Zoom do **not** hand-calibrate per device; they let the OS render-side Communications mode apply OEM-tuned speaker EQ/protection. Lean on §3.3 first. A per-device *target dBFS* is fine; per-device EQ is a trap.
- **A second full APM instance on the render buffer** (or vendoring the fork via the meson build) just to get AGC2 on output. Heavier dependency + build surface than the ~200-line port; the port is the recommended path.
- **`libebur128` true-LUFS** unless dBFS leveling proves insufficient — extra native dep + measure latency for a perceptual delta a single voice may not need.
- **Time-stretch / PLC packet-loss concealment** in playout. We have no network jitter — there are no packets to conceal. A modest adaptive buffer depth is enough; concealment algorithms are conferencing machinery.
- **An additional standalone noise suppressor (e.g. RNNoise).** Capture NS is already covered by the WebRTC APM (`_apm.py`) and the OS comm path. Don't stack a third NS — and note `VOICE_RECOGNITION`-style paths *deliberately* skip NS to feed ASR a cleaner full-band signal, so more NS is not obviously better for an ASR assistant.
- **Transient suppression / ExperimentalAgc/Ns** — removed upstream in WebRTC M131; don't resurrect.

---

## 5. Quick wins this week vs larger bets

**This week (config + ~200 lines of DSP, low risk, directly answers the complaint):**

1. **§3.1-A True-peak limiter** on the output stage (port `limiter.cc`'s curve) — kills clip/"buzz" on weak speakers. *Highest value, lands first.*
2. **§3.1-A Loudness target** replacing RMS (port AGC2 `ComputeGainDb`, target ≈ −17 dBFS, slew-limited) — perceptual, device-independent leveling.
3. **§3.3 Output stream → Communications category** (`sherpa.py:2537-2540`) — OS per-speaker protection/EQ for free.
4. **§3.2 Default `capture_voice_comm` ON on Windows** — OS Communications-mode AEC/AGC/NS per device, one config flip.

**Larger bets (architecture / per-platform, land incrementally):**

- **§3.4 Linux PipeWire `module-echo-cancel` first-class** (auto-point input at the EC source; disable redundant in-app AEC) — also retires the open-speaker barge-in time-alignment burden.
- **§3.6 `AudioConnector` seam** in `core/engine.py` — role-based selection + hot-plug, consolidating `sherpa.py`'s hand-rolled device open. Backs the Windows role resolution and Linux EC pointing; lands piecewise behind the P1 wins.
- **§3.5-B Per-device output targets** in `device_profiles` once the leveler exists.
- **§3.5-C Adaptive playout** driven by the existing `playback_underrun` metric.
- **Apple VPIO** wiring when a macOS/iOS shell needs it (currently unwired, `docs:113`).

**Validation throughout:** use the existing `tools/autotest/` PipeWire virtual-cable rig (digital-loopback AEC caveat noted) for regression, and a live-mic A/B (the `logs/runs/` bundle convention + `--record` replay) for the loudness/limiter listening tests and any AGC2-on-capture default change before it ships on.