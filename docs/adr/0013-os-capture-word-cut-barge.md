# ADR-0013: OS voice-comm capture (module-echo-cancel) + no-duck word-cut barge for the open speaker

Date: 2026-07-06
Status: proposed (Phase B live experiment; opt-in, off by default)

## Decision

For the open-speaker path we route microphone capture through the **OS/PipeWire
echo-canceller** (`module-echo-cancel`, webrtc, noise-suppression + gain-control
OFF) **replacing** the in-app WebRTC APM (`aec_enabled=false`), and we detect
barge-in with a **continuous, no-duck WORD-CUT**: the streaming recognizer is fed
**every** playback block on the OS-cancelled mic and the reply is hard-cut the
instant **≥ `barge_word_cut_min_words` (=4) new non-own-speech words** — or a bare
**"stop"** — are transcribed since the current speech burst started. **Word
content is the discriminant, not level, and there is no duck.**

New knobs (`SherpaConfig`, default OFF → legacy path byte-identical):
`barge_word_cut_enabled` (live only when the flag is set **and** `self._aec is
None`), `barge_word_cut_min_words=4`. Also added this session but superseded by the
above: `barge_confirm_duck_margin_db` (a level gate on the duck-based word-gate —
kept for the `barge_confirm_enabled` path, but that path **pumps** on a nonlinear
speaker, which is why word-cut exists).

## Context / why

Phase B of the permanent plan (`docs/session_2026-07-04_permanent_voice_barge_plan.md`)
— "OS-capture + word-gate" — run live on the bare ROG laptop speaker (Linux Mint,
PipeWire 1.0.5). Findings, each verified live and/or by adversarial multi-agent
review:

- **The OS canceller makes the near-end user CLEAN during playback** — the ASR
  transcribes the owner's talk-over *while the assistant is speaking* (`raw 'STOP'`,
  `raw 'TELL ME A LONG STORY ABOUT THE OCEAN'`). This is the thing the in-app APM
  never gave (always-on NS smeared the near-end; the coherence-veto killed every
  real talk-over — see ADR-0006 / the 2026-06-21 APM-NS-residual P1). Reference is
  wired by playing TTS **into** the echo-cancel sink (`pw-link` confirms).
- **A LEVEL gate cannot discriminate.** The nonlinear laptop speaker leaves
  residual-echo **bursts as loud as the user** (measured run-230222: echo ~−26 dB
  vs a talk-over ~−14 dB below the TTS reference, but bursts reach the user's
  level). With `barge_confirm_duck_margin_db=-18` the duck-based path still fired
  6/7 false ducks. Tightening the dB gate also rejects the user. Dead end.
- **The DUCK is the audible "volume fluctuation."** The duck-then-confirm path
  ducks to 0.15 for 1.5 s on each (flaky) acoustic trigger; on echo that is pure
  pumping. Removing the duck removes the pumping by construction.
- **Word content is the reliable discriminant** now that the near-end is clean, so
  we cut on transcribed words with no volume change until the cut.
- **Three adversarial verifiers** caught that on a nonlinear speaker the residual
  echo transcribes as **garbled 2-word fragments** (e.g. `"YOU'RE ANY"`) that do
  **not** match the clean played text, so `_reads_like_own_speech` (0.6 overlap)
  lets them through → a 2-word floor would false-cut on a silent reply. Fixes
  folded in: a **word-cut-only 4-word floor** (a real sentence clears it; a garbled
  2–3 word echo hallucination does not; a bare "stop" still cuts alone) and a
  **per-speech-burst stream reset** (VAD-quiet block → reset recognizer + base) so
  echo cannot accumulate toward the floor and streaming prefix-revision cannot flip
  the base.

## Consequences

- **Setup (transient, reversible)** on Linux/PipeWire: load the canceller with
  `pactl load-module module-echo-cancel aec_method=webrtc source_name=echo-cancel-source
  sink_name=echo-cancel-sink source_master=<mic> sink_master=<speaker>
  aec_args="webrtc.noise_suppression=false webrtc.gain_control=false"` (returns an
  integer index; `pactl unload-module <idx>` to revert), set both virtual nodes as
  the PipeWire defaults, and launch with `--input-device pipewire --output-device
  pipewire`. Machine-local config (gitignored `config.local.json`):
  `aec_enabled=false, apm_always_on=false, input_agc=true, input_calibrate=true,
  barge_word_cut_enabled=true`. The `pw-cli load-module $(...)` one-shot form is a
  **no-op** (self-unloads on exit) — use `pactl`.
- **Capture level:** keep the PipeWire source volume LOW (~13% → ADC ~6.75 dB) so
  the ADC does not clip; PipeWire maps source volume → hardware ADC gain, and a high
  source volume drives it to +30 dB and clips (see the capture-gain finding, memory
  `capture-gain-source-volume-mechanism-2026-07-05`). The in-app AGC + calibration
  must stay ON (the OS canceller does AEC only, no gain) or the VAD goes deaf.
- **Validated live:** no pumping (zero ducks), no false cut on residual echo (echo
  fragments do not cut), clean near-end STT during playback. **NOT yet validated:**
  the word-cut cut-rate on real talk-overs across a batch (needs more live runs on a
  quiet box; the loopback autotest structurally cannot judge nonlinear open-speaker
  echo).
- **Known limits:** `is_stop_command` is whole-string, so "stop it"/"please stop"
  cut only via the 4-word path (a bare "stop" is fastest); a talk-over that overlaps
  the current sentence ≥60 % is false-rejected by `_reads_like_own_speech` (topic
  miss); cut latency ~1 s (streaming ASR + 4-word requirement).
- Default OFF → every existing path (legacy hard-fire, ADR-0011 duck-confirm, APM)
  is byte-identical; full suite 2295 passed / 24 skipped. Supersedes nothing yet
  (experimental); if it holds across more live runs it becomes the open-speaker
  barge authority and the Windows equivalent is WASAPI communications capture.
