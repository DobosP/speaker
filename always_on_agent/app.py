from __future__ import annotations

import argparse
import json
import time

from .diagnostics import summarize
from .events import AgentEvent
from .replay import replay_jsonl
from .supervisor import AgentSupervisor


def run_demo(lines: list[str]) -> AgentSupervisor:
    supervisor = AgentSupervisor()
    for line in lines:
        supervisor.publish(AgentEvent.final(line))
        supervisor.drain()
        time.sleep(0.08)
        supervisor.drain()
    return supervisor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prototype always-on agent supervisor using typed events."
    )
    parser.add_argument(
        "utterances",
        nargs="*",
        help="Text utterances to simulate as final STT transcripts.",
    )
    parser.add_argument("--jsonl", help="Replay a JSONL transcript/event file.")
    parser.add_argument("--summary-json", action="store_true", help="Print machine-readable diagnostics.")
    args = parser.parse_args()

    if args.jsonl:
        supervisor = replay_jsonl(args.jsonl)
    else:
        utterances = args.utterances or [
            "assistant mode",
            "what should I do today",
            "research open source realtime voice agents",
            "stop",
        ]
        supervisor = run_demo(utterances)

    if args.summary_json:
        print(json.dumps(summarize(supervisor), indent=2, sort_keys=True))
        return 0

    print(f"mode={supervisor.state.mode.value}")
    print("transcripts:")
    for transcript in supervisor.state.transcript_log:
        print(f"- {transcript}")
    print("outputs:")
    for output in supervisor.state.spoken_outputs:
        print(f"- {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
