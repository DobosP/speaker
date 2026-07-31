Valid until: this branch lands or is superseded — then treat as history.

# Task result — LiveKit playback ownership

## Summary

LiveKit output now opts into the runtime's tracked playback contract. Each
legacy or tracked admission receives an object-identity ticket and synchronous
generation assignment. STOP fences active, queued, synthesizing, and prepared
work; clears the source before started non-completion receipts; and prevents a
delayed old clear from erasing fresh output.

Completion requires a connected and published transport, an open source, first
frame acceptance, and `wait_for_playout()` followed by a final generation and
source check. Transport loss fences playback. A failed clear disposes the
source before terminal callbacks. Native synthesis remains owned until its real
in-process call returns. A two-slot pipeline pre-generates only one successor,
preserving sentence latency without unbounded audio memory.

## Files changed

- `core/engines/livekit.py`: tracked tickets, bounded ordered playback,
  generation/clear barriers, transport readiness, exact source-drain receipts,
  failure disposal, and owned session/track teardown.
- `tests/test_livekit_engine.py`: event-driven fake-loop/TTS/source coverage for
  drain, STOP, native synthesis, bounded pre-generation, reconnect readiness,
  disposal, clear/capture failure, callback re-entry, and initialization.
- `tests/test_playback_receipts.py`: exact `binding_id` and `TurnHandle`
  playback-drain assertions.
- `requirements-remote.txt`: minimum LiveKit AudioSource contract version.
- `STATUS.md` and ADR-0095: current facts, decision, limits, and follow-up.

## Verification

- Focused LiveKit/receipt/actor/runtime matrix: 150 passed.
- Full repository non-model gate: 6,453 passed, 14 skipped, 22 deselected,
  9 pre-existing warnings.
- APM/double-talk: 6 passed.
- Targeted Ruff and `git diff --check`: passed.
- All tests were headless, single-core, and low-priority. Full-gate run bundles
  were redirected to `/tmp`; no model, network, server, or audio device ran.

## Risks and manual review

`wait_for_playout()` proves only the local LiveKit AudioSource queue drained;
it does not prove a browser or phone rendered the sound. A native TTS call
cannot be killed safely in-process: if it wedges forever, bounded `stop()`
returns degraded while the loop and actor ownership remain retained. A
process-isolated TTS worker and remote-client audible acknowledgement remain
follow-up work. No live WebRTC, microphone, speaker, STT, AEC, or barge-in
quality claim is made.

## Merge recommendation

Green for fast-forward landing on `main`; final read-only review found no
remaining blockers. Do not claim live hardware or remote audible validation.
