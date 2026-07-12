# ADR-0053: Make recorded-owner replay a causal no-device gate

Date: 2026-07-12
Status: superseded-by ADR-0061

## Decision

Run two explicit same-session recorded-owner talk-over cases without
`SPEAKER_LIVE`. Before startup, replace both sounddevice stream constructors,
device queries, and input/output capability checks. Apply one shared inject
profile from the CLI and recorded driver:
in-app AEC, OS voice-communications authority, and production word-cut off;
coherence warm-up/confirmation/delay at 0/1/0. Production 5/2/400 remains
unchanged outside that explicit profile. Use one 100 ms eligible sustain block
for the phase-sensitive synthetic command, but override the longer recorded-
owner cases to production's two-block (0.2 s) temporal policy.

Make injected buffers return sample-consumption receipts in the metrics
`perf_counter` epoch. A recorded cut is green only when one exact metrics token
shows: the hash-pinned base clip drained; a fixed long reply reached
`TTS_FIRST_AUDIO`; the configured onset grace expired; a complete sustain window
plus one capture block of post-pacing floor was actually returned to capture
after both true-first-audio observation and the onset grace, and produced no cut;
owner samples were then
consumed; and ordered `BARGE_IN -> stop_speaking -> BARGE_IN_STOP` landed within
1.0 s of the first owner sample and before the owner buffer drained plus one
bounded processing grace. An old record, enqueue time, synth-start state, late
cut, or no-op stop cannot satisfy the gate. Stop the runtime, make surviving
capture/playback/final/receipt workers a hard failure, then restore the global
sounddevice seams even on failure.

Give `FileReplayEngine` the live engine's guarded SenseVoice/punctuation final
selection. Resample caller-owned PCM once at configured quality before both the
streaming and offline passes, exclude synthetic endpoint tail, and fail closed
when streaming text is empty. A manifest duration attestation may authorize
exactly one endpoint; it cannot be reused after an empty first endpoint or across
multiple utterances. Extend ADR-0026's exact-repair policy with the corpus-attested,
normal-duration SenseVoice pair `DON'T PLAY SPEAK` -> `Stop speaking.` only for
1.4–2.0 s of explicitly owned speech. That interval gives one 0.1 s capture-block
of tolerance around the measured 1.5 s live-VAD duration and the 1.9 s corpus
annotation; missing, shorter, longer, non-finite, or non-SenseVoice evidence
remains rejected. Recognize `stop speaking` as an immediate Python/Dart stop and
exempt it from turn merging.

Pin the reference gate to its exact six labelled utterances, two unique overlap
waveforms, and two promoted same-session pairings. Every clean replay and barge
base must produce exactly one ASR final; a duplicated, missing, blank-labelled,
multi-final, or silently trailing segment makes the gate red.

## Context / why

The repository already carried six local owner utterances and two talk-over
windows, but the WAVs were not extracted in the feature worktree, so earlier
full-suite results only skipped this tier. The talk-over row also required
`SPEAKER_LIVE=1` despite fake streams. Its driver selected fixtures by list
position, used EchoLLM's short echo as the answer, waited only for synth-start,
and combined any post-enqueue stop with the most recent record carrying a barge
stamp. That admitted stale/no-op false greens and did not prove the owner buffer
caused the cut.

After true-first-audio scheduling, a fresh three-repeat inject run still produced
5/6 cuts: the identical 418 ms `Stop.` waveform sometimes retained only one full
eligible block after denoise. Keeping the production two-block requirement in a
capture domain where assistant echo is impossible made the control-flow gate
phase-sensitive without adding echo safety. Lengthening the command or weakening
production was rejected; the explicit profile now uses one block and retains its
floor-only/self-interrupt checks.

Extraction exposed a separate drift: the streaming model now heard the
human-labelled `Stop speaking` clip as `DON'T PLAY SPEAK`, while SenseVoice still
returned `Stop speaking.` The replay engine had never loaded the production
second pass despite describing itself as its measurement twin. Trusting every
offline rewrite or weakening the generic agreement guard was rejected because
that can manufacture STOP semantics from speaker echo. The exact pair plus
owned timing keeps the exception reviewable and still passes through the normal
speech/speaker/control gates.

## Consequences

The strict reference-host command (`SPEAKER_REQUIRE_RECORDED=1 ...pytest
tests/replay_recorded_voice_test.py -q`) now passes exactly 9/9 with zero skips:
six owner utterances, one same-session
multi-turn (including local STOP with no LLM/TTS reply), and two explicit causal
talk-overs through the real capture/VAD/barge/runtime/TTS workers at the two-block
temporal threshold. Owner-first-sample to FIFO stop measured 0.414 s and 0.611 s
(verifier-owned ceiling 1.0 s). Manifest SHA-256 values pin all eight private
waveforms and finite source windows bind the 1.9 s attestation. The audio remains
ignored; the ordinary command self-skips missing clips/models, while strict mode
fails any missing prerequisite.

The superseding inject profile then passed three fresh repetitions at
`logs/live/20260712-101154`: full-duplex coverage 0.972–0.975, all 6/6 intended
FIFO cuts, and zero self-interrupts. The preceding two-block run
`20260712-100610` remains failure evidence (5/6 cuts) rather than being discarded.

This proves those historical waveforms caused cuts in the explicit fake-stream
profile. It does not prove current microphone gain, speaker echo, v5 speaker
authority, physical onset-to-stop latency, voice/timbre stability, or the
production PipeWire word-cut route. Those remain mandatory live bare-speaker
checks; the branch remains unlandable until they pass.
