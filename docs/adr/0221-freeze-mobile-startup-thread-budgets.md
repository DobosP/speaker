# ADR-0221: Freeze mobile thread requests before service construction

Date: 2026-10-04
Status: accepted

## Decision

Select a strict immutable startup budget with `SPEAKER_PERFORMANCE`: current and
responsive request one ASR/two TTS threads; compact requests one ASR/one TTS thread.
Validate before bindings/plugins/widgets/services. Carry threads/provider through the
ASR initialization payload and reconstruct its native config without dropping them.
Use one pure Piper config builder for direct and worker TTS. Do not hot-switch resident
services or alter models, context, endpoint thresholds, ownership or uncertainty fences.

## Context / why

The owner requested resource modes on phone as well as desktop. The mobile app
already uses INT8 streaming ASR and its own Gemma runtime; Python device profiles do not
configure Flutter. Existing owners close permanently and uncertain native cleanup cannot
justify loading a successor. Startup-only pool requests provide a bounded selection seam
without introducing overlapping model lifetimes or implying missing desktop capabilities.

## Consequences

Use `flutter run` or `flutter build apk` with
`--dart-define=SPEAKER_PERFORMANCE=compact` (or responsive). Current remains the default.
Responsive is an alias of current mobile thread requests, not a new measured speed claim.
The native-free ownership/config gate passed 79 tests; explicit compiled compact tests
passed 13 and analysis was clean. Requested per-engine pools are not an OS/app CPU budget;
phone latency/RSS/thermal/battery and native shutdown remain unmeasured. Asset packaging
is separately governed by ADR-0220. Kitten is not a shipped phone model selection here.
