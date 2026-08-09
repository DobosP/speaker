# ADR-0158: Add paired guided STT bundle attestation

Date: 2026-08-09
Status: accepted

Refines: ADR-0108, ADR-0124, ADR-0125, ADR-0126, ADR-0138, ADR-0144, ADR-0157

## Decision

Add the device-free `tools.guided_stt_pair_attestor` evaluator as the only
supported qualifier for the two completed `guided-stt-capture-v1` bundles.
Its public command accepts only absolute control-bundle, candidate-bundle,
new scratch-root, and new report paths. Do not expose caller-selectable profile,
execution-order, route-gate, plan, case, or geometry controls.

Strictly reopen both project-owned plans, capture contracts, and schema-v2
diagnostic manifests before evaluation and again before publication. Require
complete 16/16 evidence; the exact capture protocol, execution purpose,
side-effect and identity policies, plan bytes and order, route-profile bindings,
capture-configuration digest, device, and input gain; and the fixed
`sense-voice` control and `parakeet-faster-whisper` candidate roles. Validate
each profile-specific effective-Sherpa binding, but do not require those two
profile-dependent digests to equal.

Compile one private schema-v3, 16-case exact-input corpus from each bundle's
receipt-bound final-input PCM. Evaluate the control-capture corpus through
SenseVoice and then Parakeet/Faster-Whisper, followed by the candidate-capture
corpus through SenseVoice and then Parakeet/Faster-Whisper. Both selectors
therefore receive identical ordered PCM within each physical-take comparison;
the two independently spoken capture bundles are not claimed to contain
identical PCM. Run the fixed inert tool-route gate with vault and reminders
enabled, the `obsidian` app alias, and the implicit web route. The gate uses
fail-closed no-invoke/no-open capability and launcher sentinels; construct no
execution-capable provider, voice runtime, capability, reminder store, app
launcher, audio device, or effect path.

Keep references and selected hypotheses private in memory and reduce them
immediately to aggregate counters. Publish only one owner-private, single-link,
mode-0600 schema-v1 report with kind `guided-stt-pair-attestation-v1`, using
atomic no-clobber publication. Do not render or retain transcripts, case rows,
case IDs, tags, aliases, identities, or filesystem paths in stdout or the
report. Never modify, move, overwrite, or delete an input bundle. Exit zero
means that all four fixed cells, the dry route checks, closing revalidation,
and report publication completed; it does not mean that either profile won or
is promotable. Before the terminal report link, Ctrl-C returns 130, SIGHUP or
SIGTERM returns 128 plus the signal number, and ordinary failure returns 2;
all leave no final report. After the terminal link, every outcome returns 0
with the report retained.

## Context / why

ADR-0157 made collection effect-free and bound each run independently, but one
successful bundle could not prove that its counterpart used the same plan,
capture environment, or evidence protocol. Manually exporting and replaying
only one bundle also lets a profile be judged on one physical take while the
other take is ignored. Comparing the selected live transcripts would make the
models under test part of the truth source and would resurrect transcript
retention in the capture path.

The fixed crossover evaluates both selectors on each exact physical take while
preserving the distinction between the two independently spoken recordings.
The deterministic dry route gate measures whether recognized command wording
would reach the intended closed route without turning evaluation into a tool or
device action surface.

## Consequences

The two guided bundles now have one reproducible, fail-closed qualification
path with fixed roles, ordering, source binding, aggregate privacy, and
no-clobber output. This adds an evaluation tool, not another voice-agent or
physical-session entry point. ADR-0157's capture process still never inspects,
renders, retains, or logs recognized text; only the separate offline attestor
holds replay hypotheses transiently for aggregate reduction.

Attestor success cannot verify that the owner spoke every reference correctly
or occupied the prompted physical geometry. Owner review remains mandatory.
The report supplies no model adoption, default change, direct-live or owner
authority, provider/tool execution, multi-speaker identity, native-reader gap,
AEC, playback, TTS, bare-speaker barge-in, comparative live latency, or natural-
conversation conclusion. Those gates remain separate, and no such result exists
until the two real owner bundles are captured and attested.
