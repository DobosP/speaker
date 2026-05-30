"""Noise-stress / voice-isolation test for the assistant's *as-built* filtering.

This package stresses the two things the app actually does -- and does NOT do --
about competing sound, and grades them separately:

* **Competing VOICES** are filtered by ONE mechanism: the speaker-ID gate
  (``core/engines/speaker_gate.py``). Once the desired user's voice is enrolled
  and ``sherpa.speaker_gate_input`` is on, a completed ASR final whose speaker
  embedding does not match the enrolled user is *dropped*
  (``on_metric('speaker_rejected_final')``) instead of answered. So an intruder
  TTS voice -- or a noise-born hallucinated final -- SHOULD be rejected.

* **Broadband NOISE** (white / pink / babble energy) is NOT filtered: the app
  has no denoiser, no spectral subtraction, and no AEC. The only level stage is
  ``input_gain`` + an anti-aliased resample to 16 kHz. So broadband noise will
  *degrade STT accuracy* as SNR drops -- this is EXPECTED, not an isolation
  failure, and the report attributes it to "DENOISE = ABSENT".

A correct noise-stress test reveals exactly this split. Two delivery modes:

* ``--mode inject`` (the TRUSTWORTHY measurement): digitally mix the mock user
  with noise (and/or an intruder voice) at a known SNR and feed the recognizer
  through ``InjectingInputStream`` (reusing ``tools/live_session``). The real
  mic/speaker are never touched, so the filtering numbers are acoustic-
  independent and reproducible.
* ``--mode acoustic`` (the realism demo): play the mock user + background noise
  concurrently through the REAL output device while the REAL app captures via
  the REAL mic. On a laptop with a shared speaker+mic and no AEC this garbles
  even clean audio, so its numbers are illustrative only -- the report draws
  the verdict from inject mode and flags acoustic numbers as confounded.

Nothing here edits ``core/*``, ``tools/live_session/*``, or the config files; it
measures the app exactly as configured (writing only a temp enrollment.json to
the run's out-dir).
"""
