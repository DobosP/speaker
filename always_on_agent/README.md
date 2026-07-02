# Always-On Agent (control-plane brain)

This is the control-plane brain of the runtime. It is wired into the `core/`
`VoiceRuntime` via the partial/final STT callbacks (the legacy `main.py`
monolith was deleted 2026-05-26 — see `docs/adr/0002`; the runtime is
`python -m core`).

It proves these pieces independently from the audio stack:

- typed events
- priority event bus
- supervisor-owned modes
- cancellable tasks
- live speech observation and language hints
- deterministic intent decisions before slow LLM/tool work
- assistant/search/research/command/dictation/meeting routing
- local capability providers
- replay-based tests for transcript streams
- integration boundaries for Moonshine, Pipecat, LiveKit, and Wyoming
- TTS request emission after task completion
- command confirmation before action-like tasks
- bounded parallel research with queueing

Run the simulation:

```bash
python -m always_on_agent.app "assistant mode" "search local llm tools"
```

Print machine-readable diagnostics:

```bash
python -m always_on_agent.app --summary-json \
  "assistant mode" \
  "research moonshine pipecat livekit wyoming"
```

Replay JSONL:

```jsonl
{"kind": "stt.final", "text": "assistant mode"}
{"kind": "stt.partial", "text": "research moonshine"}
{"kind": "stt.final", "text": "research moonshine for edge voice"}
```

```bash
python -m always_on_agent.app --jsonl ./replay.jsonl --summary-json
```

Run the tests:

```bash
python -m pytest tests/test_always_on_agent.py
```

Integration: the `core/` runtime publishes `AgentEvent.partial(...)` and
`AgentEvent.final(...)` from its STT callbacks and this supervisor decides
which task runs (see `core/runtime.py` and `docs/unified_architecture.md` §2).

## Current Modules

- `events.py`: event and mode schema
- `models.py`: speech observations and intent decisions
- `speech_analyzer.py`: activation, language hints, intent classification
- `capabilities.py`: local assistant/search/research/command providers
- `tasks.py`: cancellable task runtime
- `planner.py`: explicit task plans and step capability mapping
- `supervisor.py`: mode and task orchestration
- `runtime.py`: public facade (`AlwaysOnAgentRuntime`) for live STT integration
- `bridge.py`: callback adapter for existing partial/final transcript hooks
- `react.py`: bounded ReAct planner (complementary to `planner.py`, not a duplicate)
- `continuation.py` / `followups.py`: gated proactive-turn helpers
- `memory.py`: the `Memory` protocol the brain talks to
- `event_bus.py`: priority event bus
- `replay.py`: deterministic transcript replay harness
- `diagnostics.py`: summaries for test output and debugging
- `app.py`: CLI/demo harness (`python -m always_on_agent.app --jsonl ... --summary-json`)

## Integration Shape

The existing audio app can use:

```python
from always_on_agent.bridge import TranscriptBridge

bridge = TranscriptBridge()
bridge.on_partial_text("search moonshine")
bridge.on_final_text("search moonshine")
```

Command-like actions are staged and require a later `confirm` utterance before
their task starts.
