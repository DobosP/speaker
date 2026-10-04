# ADR-0220: Stage and bundle only the active mobile speech assets

Date: 2026-10-04
Status: accepted

## Decision

Resolve both TTS paths through one Amy-low asset-subtree loader. Preserve nested
espeak paths while excluding unrelated ASR, Whisper, fonts and example assets from TTS
staging. Bundle the exact active four-file hybrid ASR tuple plus available rights/readme
information, instead of its entire release directory. Make unused Whisper download and
three-file bundling explicit with `--with-whisper` on both preparation tools. Preserve
cached weights and examples; reject incomplete selected tuples before rewriting pubspec.

## Context / why

The shipped Flutter assistant constructs English streaming Zipformer and Amy-low
VITS. Its offline Whisper helper has no live call, yet the old build downloaded/bundled
Whisper and TTS copied every bundled asset into support storage. Excluding unused model
variants/examples and staging by engine avoids storage/temporary-memory work without
removing an executed recognition or speech function.

## Consequences

The active ASR tuple remains 74,207,237 bytes. The omitted locked Whisper tuple is
160,626,066 bytes (153.185 MiB uncompressed), before packaging compression; actual APK,
phone RSS, cold-copy latency, thermals and battery were not measured. Explicit optional
Whisper preparation remains available. Full current-lock Flutter gate passed 262 tests,
asset tests 10, analysis clean; final fake packaging gate passed 13 tests. No native/model,
microphone, playback or phone run followed. Startup thread modes are separate ADR-0221.
