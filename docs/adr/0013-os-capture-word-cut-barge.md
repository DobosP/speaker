# ADR-0013: OS voice-comm capture (module-echo-cancel) + no-duck word-cut barge for the open speaker

Date: 2026-07-06
Status: accepted (Windows addendum superseded-by ADR-0019)

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
- **Windows equivalent (2026-07-06 addendum):** the OS-canceller hop is already
  wired — set `capture_voice_comm=true` (`sounddevice.WasapiSettings(
  communications=True)`, the AEC/NS Communications category; fails open to the
  raw stream) plus the same config flips (`aec_enabled=false,
  apm_always_on=false, barge_word_cut_enabled=true`). Phase-B-on-Windows is a
  config repoint + live measurement, not new core code. The word-cut state
  machine gained its headless regression net the same day
  (`tests/test_barge_word_cut.py`: 4-word floor vs garbled 2-word echo,
  per-burst stream reset, no-duck invariant, suppress guards, `self._aec is
  None` scoping).

## Addendum — 2026-07-06 (evening): first real sustained talk-over batch did NOT reproduce Phase B

The first live **sustained** talk-over batch on the Linux/PipeWire box (owner at the
bare ROG speaker; `module-echo-cancel` webrtc loaded + PipeWire defaults repointed;
`aec_enabled=false, apm_always_on=false, barge_word_cut_enabled=true,
barge_confirm_enabled=false`; LLM gemma3:12b/4b; run `run-20260706-231226`) **did not
fire the word-cut barge.** The assistant played a ~3-minute story and the owner
talked over it repeatedly; the bundle shows **zero multi-word ASR finals for the
entire playback** — only two stray single-word `'AND'` echo fragments (correctly
below the 4-word floor). The word-cut path is a *text* authority, so with no
transcribed words it could not cut. Everything else was healthy: clean pre-playback
STT, `clip=0.0%`, no crash, 2316 tests green.

This directly qualifies the "**Validated live:** ... clean near-end STT during
playback" line above. On a **sustained** talk-over the near-end user voice did
**not** survive capture during playback. The earlier same-day validation observed
`raw 'STOP'` during playback, so the premise appears to hold for **short utterances
in gaps** but **not** across a continuous double-talk batch — which is exactly the
"NOT yet validated" cut-rate item, now answered negatively for this box.

Root cause is **undetermined** because no audio was saved (the run was killed, not
Ctrl-C'd, so the `record_playback_reference` WAV never flushed). Three candidates, to
be distinguished by one diagnostic re-run that **keeps** the cancelled-mic + ref WAV:
(1) the OS webrtc canceller over-suppresses the near-end during double-talk (classic
AEC double-talk suppression) → Phase B is a dead end on this box, consistent with the
2026-07-05 "no clean acoustic fix" conclusion (`barge-voice-no-acoustic-fix`);
(2) the 13% mic is too quiet under double-talk (fixable via level/AGC); (3) the
per-speech-burst streaming reset + `_reads_like_own_speech` swallow the near-end
text. **Until distinguished, Phase B stays experimental/opt-in and is NOT the
open-speaker barge authority.**

### Resolution addendum — 2026-07-06 (late): root cause found in code, class (3)

Code recon (multi-agent, verified against the source) found a **deterministic
state-machine defect** that fully explains the zero-word outcome without any
acoustics: `_barge_word_cut_step` gates all recognizer feeding on
`vad.is_speech_detected()`, but **nothing feeds the VAD during playback** — the
word-cut branch `continue`s before the acoustic path's `accept_waveform` call,
which was the only playback-time VAD update. The VAD therefore stays frozen at
its pre-reply state (quiet, since the user's request just ended), every playback
block short-circuits at the quiet gate, and the recognizer is never fed. The
same-day "validated live" observations (`raw 'STOP'` transcribing) most likely
rode inter-sentence gaps or a stale-true VAD — flaky, not the path working.

**Fixes (branch `fix/barge-wordcut-live-diagnostics`, suite 2341/24):** the step
now feeds the VAD every playback block before consulting it; the burst reset is
debounced (`barge_word_cut_reset_quiet_blocks=3`, knob, 1 = legacy hair-trigger)
because OS-cancelled double-talk flickers the VAD and a single quiet block wiped
a talk-over's accumulated words. Shipped with it so the next live run is
decisive regardless of outcome: word-cut funnel telemetry (`word-cut trace /
burst reset / near-end / funnel` lines + a per-reply summary; decode errors are
now counted+warned instead of swallowed), kill-safe WAV recording (header
patched+flushed every 2 s — survives SIGTERM/SIGKILL), a SIGTERM→Ctrl-C
shutdown bridge, a doctor FAIL when word-cut is configured on Linux with no
`module-echo-cancel` loaded, and a "Word-Cut Funnel (ADR-0013)" section in
`tools.diagnose_run` (flags `fed=0` starvation and voiced-windows-but-zero-words
explicitly). Note the failed run also never passed `--record` — recording
requires the CLI flag, not just `record_playback_reference=true`; the next live
run must launch with `--record`. Whether the OS canceller ALSO suppresses the
near-end during sustained double-talk (root candidates 1/2) is measurable now
and remains open until that run.

### Live validation — 2026-07-07 (run-20260707-002943)

The fixed path **cut a real talk-over on the bare speaker** — the first
successful ADR-0013 word-cut ever: near-end words transcribed during playback,
4-word floor reached ~0.35 s after the first trace, hard cut, user's sentence
preserved as pre-roll and answered. Session funnel (5 replies):
fed=90 / skipped_quiet=164 / resets=1 / dropped_words=0 / own_folds=1 /
decode_errors=0 / cuts=1 / **false cuts=0**; own-echo transcribed ≈zero words
throughout. Root candidates (1)/(2) from the addendum above are thereby
answered for this box: the near-end DOES survive the canceller once the VAD is
actually fed — the failure was purely the class-(3) defect. Known limits
confirmed live: short replies end before the 4-word floor fills (uncuttable
except "stop" — untested), and playback-fed words are dropped at reply end
when no cut fired. The diagnose_run self-interrupt classifier needs a truth-up
(flags a word-cut barge SUSPECT:NO-DTD). Cut-rate over a LARGER batch — and
"stop" — still wanted before default-on is considered.
