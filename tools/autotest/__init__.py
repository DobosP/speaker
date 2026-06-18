"""Autonomous (no-human) test harness for the voice assistant.

The live ``tools.live_session`` path needs a person at the mic; the headless
``--engine replay`` path only re-runs *recorded* audio. This package fills the
gap between them: it drives the **real** runtime end-to-end with **no human and
no real microphone**, so a session can self-verify STT / TTS / barge-in /
memory and produce a committable verdict.

Three independent tiers (run one, or ``all``):

* ``memory`` -- in-process ``fact -> distractors -> recall`` over the real
  capability stack; proves the assistant both *stores* a fact and *uses* the
  recalled block in its answer. No audio, no DB (in-RAM :class:`SessionMemory`).
  Uses a small LLM (``gemma3:4b`` by default; ``--llm echo`` checks plumbing
  only).
* ``voice`` -- stands up a PipeWire **virtual audio cable** (a null sink + its
  monitor), runs the real ``sherpa`` engine routed onto it (the runtime's own
  streams only -- the system default is never touched), and injects
  TTS-synthesized "user" utterances. Exercises the real-time capture thread,
  the ``_audio_cb`` playback FIFO, AEC, and barge-in -- the path the
  replay/sandbox tiers cannot reach. Because the assistant hears its own TTS
  over the loopback, it is also the autonomous reproduction of the open-speaker
  self-interrupt P1.
* ``replay`` -- wraps the existing delay-independent ``tools.replay_barge``
  (coherence self-interrupt) and ``tools.aec_probe`` (ERLE/delay) over a run
  bundle that has a ``.ref.wav`` sibling (the ``voice`` tier records one).

Run: ``python -m tools.autotest all`` (see ``__main__.py`` for options).
"""
