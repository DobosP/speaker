# Capability catalog + self-aware model (2026-05)

> ⚠️ **Superseded — durable content merged into [`docs/unified_architecture.md`](../unified_architecture.md).** Kept for revision history; do not treat as current. (2026-06-02 consolidation.)

Two of the assistant's goals — *the controller is aware of all its capabilities
and takes the best action*, and *the model knows what it is and what skills it
has* — shared one missing piece: there was no **capability manifest**. The
registry stored only `name -> callable`; the ReAct planner's tool descriptions
were hand-typed in a second place that drifted from the registered set; and the
system prompt never told the model its skills. This adds one manifest as the
single source of truth.

## The manifest

`always_on_agent/capabilities.py` — `CapabilitySpec` describes each capability:

| field | meaning |
|---|---|
| `summary` | user/model-facing "what it does" |
| `when_to_use` | planner-facing guidance (falls back to `summary`) |
| `egress` | `local` / `cloud` — the §9.7 data-boundary class |
| `speaks` | produces spoken output (vs a silent side-effect) |
| `side_effecting` | takes an action / changes state |
| `planner_tool` | exposed to the ReAct planner as a tool |
| `user_facing` | enumerated in the model's self-description |

`CapabilityRegistry.register(name, provider, *, spec=None)` keeps the existing
spec when a provider is **re-registered without one** — so the LLM-backed
`assistant.answer`/`research.local` and the §9.7 `web.search` overrides keep
their metadata. Query it via `spec()`, `manifest()`, `planner_tools()`,
`describe(names, planner=)`.

## What it drives (no more drift)

1. **ReAct tool catalog** (`always_on_agent/react.py`) — `_catalog()` now renders
   from `registry.describe(tools, planner=True)`; the hand-typed
   `_TOOL_DESCRIPTIONS` is deleted. `DEFAULT_TOOLS` is asserted equal to
   `registry.planner_tools()` by a test, and `_reconcile_capabilities`
   (`core/runtime.py`) warns at startup on any configured planner tool that
   isn't registered.
2. **The model's self-description** (`core/persona.py` `build_system_prompt`) —
   the answering model's system prompt enumerates the **user-facing, deliverable**
   skills from the manifest, under a persona-aware identity, with a web-access
   line that reflects the real egress state.
3. **Persona** — the optional `assistant` config block (`name`, `persona`,
   `extra`) gives the model an identity; the ReAct planner's final answer keeps
   the persona name too.

## Honesty rules (so the model doesn't confabulate)

- **Silent, mode-gated side-effects are NOT advertised.** `dictation.clean`,
  `meeting.note`, and `command.stage` are reachable only via an explicit prefix
  or a prior mode switch, chosen deterministically by the analyzer — not by the
  answering LLM. Advertising them would make the model claim it "took a note" /
  "ran the command" when the turn was a plain text reply with no side-effect.
  They stay in the manifest (for the planner + reconciliation) with
  `user_facing=False`.
- **Web is gated on real availability.** `web.search` (egress `cloud`) is only
  listed, and the "you can search the web" limit line only used, when
  `web_search.enabled AND base_url` — matching the condition under which the
  provider actually reaches the network (else it silently falls back to the
  local corpus).

## Layering

`core/persona.py` owns the prompt; `core/capabilities.py` re-exports the
byte-identical `DEFAULT_SYSTEM` (pinned by `tests/test_memory_contract.py`).
The brain (`always_on_agent`) never imports `core`: the persona name reaches the
ReAct planner as a plain string.

## Tests

`tests/test_capability_catalog.py` — manifest/spec/override-preserves-spec,
ReAct-catalog-from-manifest + no-drift, skill enumeration, web gating (unit +
runtime e2e), side-effects-in-manifest-but-not-advertised, persona identity,
`DEFAULT_SYSTEM` byte-identity, `from_dict` robustness.
