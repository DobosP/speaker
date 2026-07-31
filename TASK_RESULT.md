Valid until: this branch lands or is superseded — then treat as history.

# Task result — modular voice-session spine

## Summary

Added the typed session/topology layer behind one public
`python -m core --session` entrypoint. Console, local audio, and replay still
reuse the existing `build_runtime` authority/tool plane; a `VoiceSession` now
owns exactly one injected `VoiceRuntime` and releases it once across normal,
startup-failure, and repeated-cleanup paths.

Audio remains device-only by default. Setup can atomically grant or revoke
encrypted self-hosted trusted-LAN audio, but the canonical trusted-LAN session
stays fail-closed until ADR-0096's concrete publisher-bound LiveKit Agents
wrapper exists and passes live A/B. Participant identity remains transport-only
and remote speech remains action-untrusted. Hidden `--engine` compatibility and
the legacy worker remain rollback paths.

## Main files changed

- `core/voice_session.py`: typed modes, topology and egress policy, bounded
  identifiers, trusted-LAN URL validation, promotion gate, and lifecycle owner.
- `core/app.py`: canonical `--session` selector, pre-profile policy resolution,
  safe run metadata, legacy URL enforcement, and unified teardown.
- `tools/setup_assistant.py` / `config.json`: explicit reversible audio-egress
  grant with the shipped `device_only` default.
- `tools/live_launcher.py` and setup/install helpers: launch `--session local`.
- `tests/test_voice_session.py` and adjacent tests: policy, lifecycle, profile,
  CLI, setup, launcher, and real-process coverage.
- ADR-0097, superseded ADR-0001 status, `STATUS.md`, and current run guides.

## Verification

- Full non-model logic gate: 6,532 passed, 13 skipped, 22 deselected; nine
  pre-existing warnings.
- Final focused session/CLI/lifecycle gate after URL and cleanup hardening:
  55 passed.
- Enrollment/doctor/installer/launcher adjacency: 293 passed.
- Device-profile/origin/tool/LiveKit seam adjacency: 88 passed.
- Required APM/double-talk gate: 6 passed.
- `git diff --check`: passed.
- All checks were headless and low-priority. No microphone, speaker, model,
  network service, or GPU validation is claimed.

## Risks and manual review

This slice does not improve STT or make the remote composition usable. The
concrete LiveKit Agents wrapper, parallel `/chat` removal, stage-separated media
providers, public-corpus harness, bounded Parakeet/Zipformer A/B, and physical
open-speaker validation remain follow-up work. The legacy LiveKit path now
requires the explicit setup grant and is retained only for rollback.

## Merge recommendation

Green for landing after independent review. The full logic, focused, adjacent,
APM, and whitespace gates are green; no live-audio claim is made.
