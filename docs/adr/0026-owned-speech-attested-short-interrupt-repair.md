# ADR-0026: Short interrupt repairs require exact owned-speech attestation

Date: 2026-07-11
Status: accepted

## Decision

At the Sherpa second-pass final-selection seam, admit a rejected short rewrite
only for the exact normalized owner-attested pair `castle death` ->
`cancel that`, only when the configured backend is SenseVoice and an explicit,
positive, finite live-segment speech duration is below the shared 1.2-second
short-clip boundary. That timing may come from VAD observation or confirmed
word-cut PCM; do not use padded PCM array duration as substitute evidence. Keep the
generic agreement guard unchanged, and pass the repaired final through the
existing echo-floor, speaker-identity, runtime self-echo, and intent gates.
Recognize `cancel that` as an immediate stop in the shared Python/Dart control
contract and exempt it from final hold-and-merge.

## Context / why

The owner recording for `cancel that` is transcribed as `CASTLE DEATH` by the
streaming recognizer and correctly as `Cancel that.` by SenseVoice. The generic
agreement guard intentionally rejects this low-overlap short rewrite because
the same model can invent plausible text from short open-speaker echo. Keeping
the streaming text loses a mandatory interrupt; weakening the guard for all
commands, applying phonetic/fuzzy matching, or trusting every short SenseVoice
rewrite would reopen the echo-final cascade the guard prevents.

Backend identity and independently owned speech timing are available only at
the engine seam. VAD observations normally supply it; the word-cut handoff may
supply bounded timestamps for already-confirmed playback-time user PCM. An
exact attested pair there restores the known interrupt without changing
arbitrary agreement-guard callers or padded-array fallback behavior. `Pause.`
remains outside this decision because it is not currently a canonical stop
intent and would require a separate behavior choice.

## Consequences

- The recorded `cancel that` failure reaches the existing deterministic STOP
  path instead of being answered as `castle death`.
- During playback, `cancel that` receives the same below-word-floor word-cut and
  duck-confirm authority as other canonical stop commands.
- Unlisted transcript pairs, other second-pass backends, missing owned-speech
  duration, and utterances at or above 1.2 seconds keep the guarded final.
- The repaired text still needs a valid acoustic carrier and accepted speaker;
  this exception grants no direct command authority.
- Future repair pairs require their own replay evidence and explicit review;
  this is not a general ASR correction dictionary.
- The attestation is owner-recording-derived; deterministic tests cover the
  resulting selection and control paths. No live owner-mic A/B was run.
