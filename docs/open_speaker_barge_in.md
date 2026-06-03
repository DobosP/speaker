# Open-speaker barge-in (laptop mic + laptop speaker, no headphones)

Talk-over barge-in on an **open speaker** is hard: the assistant's own TTS leaks
into the mic and, on a cheap **nonlinear** laptop speaker, looks like a user
barging in -- so the assistant self-interrupts. Prior sessions concluded this was
a hardware limit (reliable path = headphones). This change makes it **usable
without headphones** by stacking four fixes; the live result was "the best
interaction by far -- barge-in felt good" on a Realtek laptop mic+speaker.

## The stack (how it fits together)

1. **DTLN neural echo canceller** (`aec_backend="dtln"`,
   `core/engines/_aec.py::_DTLNEchoCanceller`). A linear filter (NLMS) **cannot**
   cancel a nonlinear speaker's echo -- it leaves a large residual and even
   *diverges* (observed: post-AEC RMS ~7.4, instant self-interrupt). The two-stage
   DTLN deep model handles the nonlinearity and drops the playback residual to
   ~`0.0006` RMS. The ONNX stages ship in `pretrained_models/sherpa/aec/`
   (`tools/setup_models.py --aec-model` fetches+converts them; needs
   `tf2onnx`+`tensorflow-cpu` at build time, only `onnxruntime` at runtime).
2. **AEC stabilization + divergence guard** (`_FDAFAdaptiveFilter`,
   `EchoCanceller.process_16k`). For the NLMS fallback: conservative step
   (`aec_mu=0.3`), leakage (`aec_leak=0.9999`), an in-filter divergence recovery,
   and a canceller-level guard that drops any output louder than the input. An
   echo canceller must never *amplify*; these keep it bounded.
3. **Auto-calibrated residual-floor gate** (`sherpa.py::_update_playback_floor`,
   `_looks_like_user`). The barge gate keys off `_playback_floor_rms` -- the
   post-AEC residual echo+noise floor, learned **online during playback**
   (freeze-on-burst). It adapts to speaker volume + room noise with no manual
   calibration, and the (cancelled) echo sits at the floor so it can't
   self-interrupt. A barge must stand `barge_in_residual_margin_db` above it. This
   is PRIMARY when AEC is on, making the (nonlinear-fooled) coherence detector
   veto-only.
4. **Sustained-barge requirement** (`barge_in_min_speech_sec`). DTLN misses on
   sharp transients, leaving brief (~2-block) residual spikes. Requiring the barge
   to persist (~0.6 s) rejects those momentary spikes while a real talk-over (which
   lasts) still fires.

## Working config (this machine: Realtek Microphone Array + Speakers)

Put in `config.local.json` under `"sherpa"` (it is machine-local / gitignored):

```json
"aec_enabled": true,
"aec_backend": "dtln",
"aec_model": "<abs path>/pretrained_models/sherpa/aec",
"aec_ref_delay_ms": 19,
"barge_in_residual_margin_db": 15.0,
"barge_in_min_speech_sec": 0.6
```

- `aec_ref_delay_ms` is the speaker->mic round-trip; **measure it per device**
  (the coherence detector logs `delay=...ms`; `0` gave ~no cancellation here, the
  true value was ~19 ms). A wrong delay = poor cancellation = self-interrupt.
- Launch with the laptop devices explicitly:
  `python -m core --engine sherpa --record --stream-tts --input-device 1 --output-device 3`
  (`--list-devices` to find the indices).

## Known-remaining (follow-ups, not yet fixed)

- **Latency / sound glitch:** DTLN runs per-block on CPU; under load it can cause
  an audible glitch. Profile the per-block cost; consider a lighter DTLN size
  (256/128) or GPU onnxruntime.
- **Missed / unanswered questions + occasional hallucination:** an LLM-tier /
  addressing-decision issue (fast `gemma3:4b`), independent of barge-in.
- Barge-in is good but **not 100%**; a `"stop"` keyword (KWS) fast-path would be a
  belt-and-suspenders interrupt that sidesteps the echo entirely.
