Valid until: this branch lands or is superseded — then treat as history.

# Task result — publisher-bound LiveKit Agents contract

## Summary

Added a dependency-safe `AudioEngine` adapter contract for a future
publisher-scoped LiveKit Agents session. The trusted Speaker controller remains
the only LLM/tool/authority plane. Raw SDK session/output objects are excluded;
the future wrapper must atomically issue output-route leases and fail closed on
configuration, route mutation, interruption failure, lifecycle races, or close.

This slice is intentionally not selectable: it does not yet implement or install
the concrete LiveKit wrapper and makes no STT, latency, or live-audio claim.

## Files changed

- `core/engines/livekit_agents.py`: typed wrapper policy, participant binding,
  whole-turn commit bridge, action-untrusted remote lineage, generation-safe
  barge evidence, route-lease playback receipts, fatal close fencing, and SDK
  state/close mapping.
- `tests/test_livekit_agents_engine.py`: deterministic fake-loop coverage for
  trust, turn assembly, restart races, interruption episodes, output mutations,
  receipt terminality, failure close, and callback lifecycle.
- `docs/adr/0096-introduce-publisher-bound-livekit-agents-adapter.md`: decision,
  safety boundary, evidence limits, and promotion requirements.
- `STATUS.md`: current unselectable seam and focused evidence.

## Verification

- Focused adapter: 28 passed.
- Adapter plus LiveKit/playback/session/device-tool adjacency: 111 passed.
- Required APM/double-talk regression: 6 passed.
- `git diff --check`: passed.
- All tests were headless, low-priority, and used no model, network, server,
  microphone, speaker, or GPU.

## Risks and manual review

The concrete non-escaping SDK wrapper remains mandatory before wiring. It must
pin LiveKit Agents, own publisher selection and every output mutation, pass
`record=False`, keep framework LLM/tools/MCP empty, use audited local speech
components, and request non-draining close after playout failure. A handle/sink
terminal is not proof that the remote user heard audio; client acknowledgement
and bare-speaker live A/B remain promotion gates.

## Merge recommendation

Green for fast-forward landing. The final independent review found no remaining
P0/P1 issue, and the focused, adjacent, APM, and whitespace gates are green.
Keep the legacy remote path active.
