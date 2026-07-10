# ADR-0015: Speaker enrollment matches and fingerprints the active capture front end

Date: 2026-07-10
Status: superseded-by ADR-0018

## Decision

`python -m core --enroll` processes every recording blockwise in the same
model-visible order as idle live capture: InputAGC (taking precedence over static
gain), resampling, always-on in-app APM against a zero far-end only when that APM
owns idle processing, then GTCRN unless APM owns noise suppression. OS echo
cancellation remains upstream in the selected/default capture device. Every new
enrollment persists a versioned fingerprint of the processors that actually
built. Runtime loads an embedding only when both its speaker model and front-end
fingerprint match; a mismatch is ignored so speaker gating fails open, with an
actionable re-enrollment warning. A legacy enrollment without provenance is
accepted only on the raw baseline.

## Context / why

GTCRN became the fleet capture default, and InputAGC/APM remain selectable per
machine, but the enrollment CLI still applied only static gain and resampling.
The saved reference therefore occupied a different embedding domain from live
speech, which can make speaker-ID reject the owner. Merely documenting “re-enroll
after denoise” cannot fix this because the old CLI could not create a denoised
reference. Trusting a stale embedding is worse than having no enrollment: an
unenrolled gate admits speech, while a confidently mismatched gate silently locks
out the owner. Fingerprinting requested flags alone was rejected because optional
processors fail open; provenance must describe what actually built.

## Consequences

- Enrollment and runtime share the same stage-selection contract, while injected
  processors keep the ordering and compatibility logic deterministic in headless
  tests.
- A processor becoming available after previously failing open invalidates the old
  enrollment automatically; the assistant continues listening and tells the owner
  to run `python -m core --enroll` again.
- Changing model-visible front-end semantics requires incrementing the enrollment
  front-end version. Legacy files stay compatible only with no AGC, unity static
  gain, no always-on APM, no GTCRN, and no requested OS voice-communications mode.
- Existing machines using denoise/AGC/APM require a real microphone re-enrollment
  after this lands. That live step is still required and is not claimed by the
  headless implementation tests.
