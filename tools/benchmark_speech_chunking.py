"""Deterministic public-text scheduling probe, without inference or audio.

Run: python -m tools.benchmark_speech_chunking > /private/current-result.json
Arrival steps are synthetic word chunks, not model tokens, seconds or DAC latency.
"""

from __future__ import annotations

import json
import re

from core.speech_chunking import SpeechChunker, SpeechChunkingConfig

CASES = {
    "long_clause": "There is a clear way to approach this problem, start with the simplest case and then check each remaining condition before choosing the final approach.",
    "long_unpunctuated": " ".join("example" for _ in range(56)),
    "short": "The answer is forty two.",
    "expression": "[emotion:calm] There is a clear way to approach this problem, start with the simplest case and then review each condition.",
    "numeric": "It is a total of 1,234.56 units at 12:30, and the remaining description follows.",
}


def benchmark() -> dict:
    rows = []
    for name, text in CASES.items():
        parts = re.findall(r"\S+\s*", text)
        modes = {}
        for mode in ("normal", "fast"):
            chunker = SpeechChunker(SpeechChunkingConfig(mode=mode))
            chunks, first = [], None
            for step, part in enumerate(parts, start=1):
                ready = chunker.feed(part)
                if ready and first is None:
                    first = step
                chunks.extend(ready)
            tail = chunker.finish()
            if tail:
                chunks.append(tail)
                if first is None:
                    first = len(parts)  # end-of-stream on last arrival step
            assert " ".join(chunks) == text
            modes[mode] = {
                "first_speakable_arrival_step": first,
                "fragments": len(chunks),
            }
        rows.append({"case": name, "arrival_steps": len(parts), "modes": modes})
    return {
        "schema_version": 1,
        "kind": "synthetic_public_text_scheduling",
        "inference": False,
        "physical_audio": False,
        "settings": {"first_fragment_min_chars": 32, "first_fragment_max_words": 28},
        "limits": "No timing, synthesis, acoustic quality or live latency claim; fast may cost one additional synthesis call.",
        "cases": rows,
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
