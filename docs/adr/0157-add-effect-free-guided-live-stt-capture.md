# ADR-0157: Add effect-free guided live STT capture

Date: 2026-08-07
Status: accepted

## Decision

Add `./live.sh --guided-stt-capture` as the only public physical entry for the
owner close/far final-STT comparison. Require one explicit complete
`--final-stt-profile`; reject assistant/model/mode selectors before host setup;
and make speaker identity off only inside that selected process. Keep normal
`./live.sh` behavior, model defaults, enrollment files, tool policy, and
assistant authority unchanged.

Publish a project-owned, immutable 16-case plan and a capture contract into the
private mode-0700 run directory before opening the microphone. The plan has
eight close and eight far cases and includes numerals, vault/Obsidian aliases,
web and reminder commands, an app alias, exclusions, and negative controls.
Bind its exact bytes/order plus the final-STT profile, effective Sherpa
configuration, capture-only configuration, device profile, input gain, and
dry tool-route availability by safe digests. Reopen and verify the publication
after capture; never overwrite an earlier plan, contract, or partial evidence.

Run the core child with a private `stt_capture_only` execution purpose. Build
only the microphone, DSP/AEC, VAD, streaming/final STT, verifier,
punctuation, recorder, and diagnostics needed by that purpose. Warm selected
STT components before microphone and worker startup. Do not construct or query
the LLM factory, `VoiceRuntime`, `VoiceSession`, router, memory, event bus,
screen feed, capabilities, control plane, TTS, keyword spotter, speaker gate,
playback worker, or output device. Capture-specific readiness checks the input
device and selected ASR/frontend path while excluding those inactive
components. Default full-assistant construction and readiness remain
unchanged.

Use a narrow capture session that directly owns the engine. The terminal guide
asks the owner to acknowledge each geometry and case, then arms exactly one
selected final. Accept only typed live-capture acoustic lineage whose speech
starts after the arm, whose capture identity stays constant for the plan, and
whose owner verification is unknown. A timeout, abort, recovery, fatal media
state, duplicate or unarmed final, reused acoustic span, changed capture
identity, incomplete plan, stop-time callback, or failed evidence check makes
the run red. Never inspect, render, retain, or log recognized text in this
path; only the immutable project-owned reference phrase may be displayed.

Return success only after all 16 arms complete, the engine stops, the private
diagnostic manifest validates, and every promised synchronized artifact is
present and bound. Record only aggregate completion, configuration, profile,
and effect-policy facts in the private summary. Preserve partial bundles on
every failure.

## Context / why

The earlier proposed physical comparison used `./live.sh --llm echo`. Echo
still entered the ordinary assistant runtime: deterministic vault, reminder,
web, and trusted-app routes could execute even though a generative model was
not answering. It therefore was neither an effect-free microphone collection
path nor a valid way to promise the owner that spoken evaluation commands
could not touch local state.

The prior two-command procedure also relied on the operator to repeat an
informal plan. Separate run bundles did not prove equal phrase order, equal
capture setup, exact profile selection, one final per phrase, or complete
post-stop diagnostic evidence. A dedicated capture purpose is smaller and
testable without resurrecting a second application entry point.

Enrollment is not a generic prerequisite for STT or open-speaker barge-in.
It remains available for sessions where multiple audible speakers are present
and the owner explicitly wants speaker-specific listening or owner authority.
The capture path intentionally mints no owner authority and has no effect plane
to authorize.

## Consequences

The owner can collect comparable close/far input for SenseVoice and the
Parakeet/Faster-Whisper profile without running the chatbot or any device
capability. The canonical reference phrases are project-owned source; the
owner's audio, recognized text, and run evidence are private and must not be
copied into aggregate reports, normal logs, or committed artifacts.

This change does not select a winning STT profile, change a production
default, prove WER/CER, validate microphone hardware, establish conversational
latency or naturalness, test TTS, exercise tools, validate barge-in, or prove
multi-speaker identity. Those claims require the paired private-bundle
compiler/attestor and the owner's actual close/far runs. The old `--llm echo`
physical commands are history and must not be recommended.

Before landing, the broad non-real-model gate passed 9,094 tests with 15 skips,
25 model-only deselections, and nine dependency warnings. The adjacent
capture/app/readiness/manifest/ASR/playback/streaming gate passed 494 tests
with one skip and two pre-existing SWIG deprecation warnings. The fixed-plan/
public-launcher gate passed 116 tests; the direct purpose-isolation subset
passed 8; and APM/double-talk passed 6. Scoped Ruff, new-file formatting,
whitespace, and the STATUS line cap were green. Independent implementation
and app/readiness reviews returned GO with no code blocker and explicitly
retained owner microphone validation as the next physical gate.
