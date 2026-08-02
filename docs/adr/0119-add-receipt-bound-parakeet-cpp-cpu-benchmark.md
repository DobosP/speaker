# ADR-0119: Add a receipt-bound parakeet.cpp CPU benchmark candidate

Date: 2026-08-02
Status: accepted

## Decision

Add `parakeet-cpp-realtime-eou-v1` only to the isolated streaming-STT
benchmark. Bind parakeet.cpp v0.5.0 at commit
`1bfbebfaaf493866f49597cd3b7901959d395c60`, GGML commit
`e705c5fed490514458bdd2eaddc43bd098fcce9b`, the ordered patch/diff receipt,
Speaker bridge ABI v1 over upstream C API v6, and the F16
`realtime_eou_120m-v1` GGUF in manifest schema v8. The provisioner verifies
only two existing closed private roots and a selected Python executable. It
binds the exact source, build, model, libraries, committed bridge source,
worker, and ELF hardening evidence, then publishes one new mode-0600 manifest
without building, downloading, importing, executing, or loading the candidate.

Execute the candidate only as one-thread CPU at 16 kHz with native
1,280-sample chunks under protocol v4. Bubblewrap exposes read-only
receipt-bound inputs, one writable scratch root, no network, and no NVIDIA
device nodes. Before Ready, verify a systemd-user scope with 2 GiB memory high,
3 GiB memory max, zero swap, 100% CPU quota, 64 tasks, and OOM kill.

Treat every upstream text field as a new delta. Append deltas through the
document containing the first observed EOU, then freeze visible partial and
final text while continuing to validate later output as telemetry. EOB never
freezes text or terminates. Only the first observed EOU may qualify, and only
when it came from `feed` at or after the complete externally bounded source. A
source-early or finalize first EOU permanently prevents a later EOU from
gaining authority and forces declared-tail exhaustion. Always consume the
complete source before a terminal; encoder timestamps never substitute for the
observed feed boundary or become endpoint ground truth.

Select the three-repeat 8,000-tail-sample burst cell plus its matched paced
replay as the retained strict-fidelity evidence. Keep parakeet.cpp
benchmark-only: do not wire it into setup, `core`, `./live.sh`, the voice-agent
tool/authority plane, command authority, or any model default, and do not
promote it from after-PCM results.

## Context / why

ADR-0099 rejected the NeMo Parakeet Realtime EOU reference as too heavy for
the intended portable architecture and named a receipt-bound parakeet.cpp
experiment as the next step. parakeet.cpp supplies a smaller CPU path, but its
JSON is incremental, text fields are deltas, decoder state resets at EOU, and
encoder timestamps do not prove when the controller observed an event relative
to the external source. The first-observed-EOU rule prevents later marker
cherry-picking and makes source/tail accounting independently checkable.

Stopping on any native event initially retained the desired aggregate text but
dropped 91,868 declared source samples. Continuing all text consumed the
source but exposed post-EOU decoder segments as the same turn, adding 39 words
and raising WER to 1.4043. The corrected first-turn freeze consumes the full
source while restoring the valid one-repeat result: WER/CER .5745/.5424,
21/47 exact clips, 21/26 command hits, and 8/31 monitored negative-case false
activations.

The selected burst report completed 171 evaluations over three repeats with
zero final disagreement and exact normalized aggregate and stratum counts. All
2,786,178 offered source samples were consumed with none dropped; 24 first-EOU
endpoints qualified and 147 cases exhausted the tail. Source-audio/model-input
RTF were .4826/.3315, load was 210.096 ms, and maximum reported RSS was 306.027
MiB. The 40,502-byte report has SHA-256
`d73786acbd2b1fe37cd2202fc04ab15f256189e9384015ab7f0435d7dcd028b8`.

Matched paced replay preserved exact accuracy. Source-audio/model-input RTF
were .5592/.3841, first-partial p50/p95 were 772.961/1,104.004 ms, six chunk
deadlines were missed, maximum backlog was 30.43 ms, and maximum reported RSS was
306.387 MiB. Its 40,359-byte report has SHA-256
`2250da5a5521c5b23328ea3f70a8f86245a46d66ac61b7498a222dee4b5d1723`.
The 0/80/300/400/500 ms tail sweep kept WER and command recall equal, but only
500 ms retained the stable 34 hypothesis words, 33 nonempty finals, and 96
character edits. At 400 ms one word was still absent.

## Consequences

Speaker can now reverify one exact lower-memory native candidate without a
Python ML runtime or GPU. The exact bridge was compiled, linked, loaded, and
decoded in the private receipt-bound run. Repository tests validate its source
contract and Python boundaries but do not compile or link C++; libraries and
weights remain outside Git. The tested Linux x86-64 AVX-family build is not
ARM, phone, macOS, Windows, or general multi-device evidence.

The .2581 monitored-negative false-positive rate is unsafe for command/action
authority. No live validation ran. These runs begin after PCM and do not
validate microphone capture, VAD or endpoint ground truth, native-reader or
mailbox behavior, AEC, barge-in, enrollment, identity/authority, multiple
voices, chatbot/tools/TTS, physical latency, natural conversation, or training
disjointness. Held-out conversation/corpus work, per-platform build receipts,
command-safety policy, capture/AEC replay, and separate live A/B remain
required before adoption. Path-free aggregate evidence is in
`docs/evidence/2026-08-02-parakeet-cpp-candidate.json`.
