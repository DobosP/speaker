# Always-On Agent Prototype

This is a separate control-plane prototype for the next architecture. It does
not replace `main.py` yet, but it is designed to be wired into the existing
partial/final STT callbacks.

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

Next integration step: publish `AgentEvent.partial(...)` and
`AgentEvent.final(...)` from the existing partial/final STT callbacks in
`main.py`, then let this supervisor decide which task should run.

## Current Modules

- `events.py`: event and mode schema
- `models.py`: speech observations and intent decisions
- `speech_analyzer.py`: activation, language hints, intent classification
- `capabilities.py`: local assistant/search/research/command providers
- `tasks.py`: cancellable task runtime
- `planner.py`: explicit task plans and step capability mapping
- `supervisor.py`: mode and task orchestration
- `runtime.py`: public facade for live STT integration
- `bridge.py`: callback adapter for existing partial/final transcript hooks
- `adapters.py`: integration boundaries for open-source voice projects
- `replay.py`: deterministic transcript replay harness
- `diagnostics.py`: summaries for test output and debugging

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
