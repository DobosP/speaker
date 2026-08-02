Valid until: main advances beyond this task's landing commit — then treat as
history.

# Task result — receipt-bound parakeet.cpp CPU benchmark

## Outcome

Added `parakeet-cpp-realtime-eou-v1` only to the isolated streaming-STT
harness. Manifest schema v8 binds the exact parakeet.cpp v0.5.0 source, build,
model, CPU libraries, Speaker bridge source/library, worker, and Python
executable. Protocol v4 carries typed EOU/EOB observations, exact source/tail
accounting, encoder frame/time telemetry, and a first-observed-EOU policy.

Provisioning verifies two existing closed private roots and publishes one new
private manifest; it does not clone, patch, compile, download, import, execute,
or load candidate code or weights. Execution is one-thread CPU inside
no-network Bubblewrap with no NVIDIA nodes and a verified one-CPU, 2/3-GiB,
zero-swap, 64-task systemd scope.

Upstream text is treated as finalized deltas. Visible text includes the
document containing the first EOU and freezes afterward. EOB, finalize EOU,
and later EOU observations never gain response authority. A first EOU observed
before complete source feed blocks later acceptance and forces full declared
tail exhaustion.

Nothing was wired into setup, `core`, `./live.sh`, the voice-agent
tool/authority plane, command authority, or model defaults. ADR-0119 retains
this as a lower-resource benchmark candidate, not an adopted backend.

## Selected source-bound evidence

The authoritative three-repeat burst report used the exact 57-case public
command/noise corpus, native 1,280-sample chunks, 80 ms partial cadence, and
8,000 tail samples. It completed 171 evaluations with zero final disagreement.
WER/CER were .5745/.5424, exact matches were 63/141, command recall was .8077,
and negative final-case false-positive rate was .2581.

All 2,786,178 offered source samples were consumed and none were dropped.
Twenty-four first-EOU endpoints qualified and 147 evaluations exhausted the
tail. Source-audio/model-input RTF were .4826/.3315 and maximum reported RSS
was 306.027 MiB. The 40,502-byte report has SHA-256
`d73786acbd2b1fe37cd2202fc04ab15f256189e9384015ab7f0435d7dcd028b8`.

Matched one-repeat paced replay preserved exact accuracy. Source-audio and
model-input RTF were .5592/.3841, first-partial p50/p95 were
772.961/1,104.004 ms, six chunk deadlines were missed, maximum backlog was
30.43 ms, and maximum reported RSS was 306.387 MiB. The 40,359-byte report has
SHA-256
`2250da5a5521c5b23328ea3f70a8f86245a46d66ac61b7498a222dee4b5d1723`.

The 0/80/300/400/500 ms sweep selected 500 ms by a strict rule: choose the
shortest cell equal to the repeat-three control across identities, coverage,
source accounting, primitive transcript/command/endpoint/streaming counts, and
every named stratum. At 400 ms one hypothesis word was still absent. If a
future latency policy explicitly relaxes exact fidelity, 300 ms—not 400 ms—is
the useful alternative because 400 ms improved only one character edit.

## Exact receipts and evidence handling

The repeated burst binds config contract
`7cec4d69642bcd1b87f16202781fc361a56c73ed200849399b1b28c0b5a501f4`,
manifest `906d1feb277780056b636f4c837ea7ca7985b5fb02eef9cc24a50e5e84c381c4`,
worker `be12f93f0bb5328d4b44b6be01d6ec1cb5653f6fddbbfc6979355eadfb460df7`,
source bundle `5cec2babe4b353caa4b846176b0a1686bad52be759b0aeefe5636990674384a2`,
and artifact set
`129bc34127eab9d518284a96efde98e0371b486eff21b8328027db32198f74e8`.

Committed evidence is
`docs/evidence/2026-08-02-parakeet-cpp-candidate.json`. It is aggregate and
path-free: no audio, case identifier, local path, reference/hypothesis row,
native transcript, or model output is included. Private reports and artifacts
remain outside Git.

## Verification

The candidate/shared focused gate passed 464 tests plus the isolated
bounded-read race; the broad streaming compatibility gate passed 966, and the
public command/noise gate passed 246. The authoritative host-side non-real
split passed 7,800 broadly plus the isolated race, for 7,801 effective passing
checks; 14 skipped, 24 real-model tests remained deselected, and 11 known
warnings remained. The required APM/DTD gate passed 6/6. The repository lock
passed under its required mode 0600. Evidence JSON, current evaluator-source
hashes, STATUS line count, and `git diff --check` passed after documentation.

## Limits and next work

The .2581 monitored-negative false-positive rate is not safe command/action
evidence. Results begin after PCM and do not validate microphone capture, VAD
or endpoint ground truth, native-reader or mailbox behavior, AEC, barge-in,
enrollment, identity/authority, multiple voices, chatbot/tools/TTS, physical
latency, natural conversation, or training disjointness.

The provisioner verifies but does not recreate the native build. The tested
receipt is Linux x86-64 with fixed AVX-family flags and requires Bubblewrap plus
a systemd user scope; it is not ARM, phone, macOS, Windows, or general
multi-device evidence. Add held-out conversation/Common Voice comparisons,
per-platform build receipts, command-safety policy, capture/AEC replay, and
separate live A/B before considering adoption.
