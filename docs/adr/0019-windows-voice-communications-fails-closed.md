# ADR-0019: Windows voice-communications capture fails closed until verified

Date: 2026-07-10
Status: accepted

## Decision

Treat desktop Windows voice-communications capture as unavailable until the
runtime has a concrete, hardware-verifiable API that opens an AEC/NS
communications stream. When `capture_voice_comm` or the OS-cancelled word-cut
path selects that capability, shared readiness must construct and verify the
requested stream setting before startup. An unsupported setting is NOT READY.
Word-cut must never fall back to raw microphone capture while assistant playback
is its text authority. Linux PipeWire echo-cancel routing remains governed by
ADR-0013 and is unaffected.

## Context / why

ADR-0013's Windows addendum stated that
`sounddevice.WasapiSettings(communications=True)` was already wired and would
activate the Windows Communications AEC/NS category. The installed sounddevice
API does not accept that keyword; construction raises `TypeError`. The previous
best-effort runtime branch could therefore fall back to a raw stream while the
configuration still claimed voice processing. On word-cut, raw playback echo can
be transcribed as authoritative near-end words, so silent degradation is unsafe.
Readiness now exercises the constructor through an injected seam and reports the
missing capability before opening models or devices.

## Consequences

- Normal Windows native-voice startup reports NOT READY when the selected profile
  requires this unsupported path; disabling word-cut or supplying a genuinely
  verified communications-capture implementation is required.
- Bypassing normal readiness still cannot activate word-cut without successful
  post-open route verification.
- The in-app AEC paths remain available as separate, explicitly selected
  fallbacks; they do not prove Windows Communications capture.
- Headless tests cover missing requests, constructor failure, and a future
  constructible implementation. Windows hardware validation remains mandatory
  before this capability can be declared available again.
