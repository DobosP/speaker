# ADR-0041: Lock the physical TTS speaker across the session

Date: 2026-07-11
Status: accepted

## Decision

Ship `sherpa.tts_lock_speaker_id=true` and treat the configured
`tts_speaker_id` as the physical voice for onboarding, acknowledgements, and
normal replies throughout one runtime session. Continue recognizing and
sanitizing configured voice directives, but prevent them from entering the
runtime's typed `SpeechStyle.voice` or changing Sherpa/FileReplay synthesis
`sid`. Keep emotion and rate as fragment-local speed controls. When the lock is
active, omit named voices from the answering prompt and explicitly prohibit
voice directives. An intentionally multi-character deployment may opt out with
`tts_lock_speaker_id=false` and retains the reply-scoped behavior from ADR-0037.

## Context / why

Main run `run-20260711-154451` resolved the initial reply with speaker 0, then a
story tagged `voice:warm` resolved every fragment with speaker 16. ADR-0037
fixed speaker changes *inside* a streamed reply, but deliberately reset each new
reply to the configured default and allowed a new tag to select another voice.
The owner heard the resulting cross-reply switch and requires one recognizable
assistant voice across onboarding and normal use.

Locking whichever voice the model happened to choose first was rejected because
model output must not choose the assistant's session identity, and onboarding
may already have spoken with the configured default. A prompt-only prohibition
was rejected because direct engine calls and malformed/model-generated markup
still need a fail-closed synthesis boundary. Disabling all expressive markup was
also unnecessary: speed-based emotion and rate do not change physical speaker
identity and remain useful.

## Consequences

- Named voice maps remain sanitizer vocabulary but cannot change the shipped
  session voice; the configured `tts_speaker_id` is deterministic.
- Runtime admission, Sherpa's final synthesis guard, and FileReplay independently
  enforce the same lock, while unsupported control tags remain silent.
- Existing multi-character deployments must explicitly disable the lock and
  accept voice changes; their prior reply-scoped continuity remains available.
- Deterministic tests cover prompt, typed Runtime, Sherpa, and FileReplay paths.
  This is headless evidence only; the owner-ear live A/B is still required, and
  the same live run's missed barge-in remains a separate red gate.
