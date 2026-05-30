"""Live on-hardware validation harness.

A synthetic user (a TTS voice distinct from the assistant's) speaks scripted
utterances aloud through the real speakers; the real assistant hears them over
the air via the real mic, thinks, and answers aloud. The harness records an
attributed timeline (every turn labelled user vs assistant, with the exact audio
file + timestamps) and per-turn latency.

This is NOT part of the pytest logic suite -- it needs real ASR/TTS/LLM models
and real audio hardware, and is run ONLY on request:

    python -m tools.live_session --list
    python -m tools.live_session --scenario baseline_latency_single_turn_qa
    python -m tools.live_session --all

See docs/live_validation_2026-05.md.
"""
