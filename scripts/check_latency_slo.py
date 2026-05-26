#!/usr/bin/env python3
"""
Map live ``(latency)`` / diagnostics ``turn_metrics`` JSON to SLO thresholds.

Thresholds align with ``enforce_slo()`` in ``benchmarks/benchmark_realtime.py``.
Use after recording::

    python main.py ... --diagnostics-log-path logs/trace.jsonl

Then::

    python scripts/check_latency_slo.py logs/trace.jsonl
    python scripts/check_latency_slo.py --map
    python scripts/check_latency_slo.py --demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Keep in sync with benchmarks/benchmark_realtime.py enforce_slo()
SLO_MS = {
    "stt_final_to_first_llm_sentence_ms": 2000.0,
    "speech_detected_to_first_audio_ms": 3000.0,
}
# Benchmark names llm_first_sentence_ms for synthetic runs; live streaming uses
# memory_ready_to_first_llm_sentence_ms for the LLM+phrase-buffer segment.
SLO_LLM_FIRST_MS = 2000.0
INTERRUPT_MS_DEFAULT = 300.0
INTERRUPT_MS_EDGE = 380.0


def print_mapping() -> None:
    print("Live/diagnostics field → benchmark_realtime enforce_slo context\n")
    print(
        f"  stt_final_to_first_llm_sentence_ms   ≤ {SLO_MS['stt_final_to_first_llm_sentence_ms']:.0f} ms  "
        "(STT end → first streamed phrase to TTS queue)"
    )
    print(
        f"  speech_detected_to_first_audio_ms    ≤ {SLO_MS['speech_detected_to_first_audio_ms']:.0f} ms  "
        "(utterance start → first TTS audio sample)"
    )
    print(
        f"  memory_ready_to_first_llm_sentence_ms   (diagnostic)  compare to ~{SLO_LLM_FIRST_MS:.0f} ms  "
        "— LLM/phrase buffer only; use with stt_final_to_memory_ready_ms"
    )
    print(
        f"  speech_detected_to_stt_final_ms        (diagnostic)  final STT cost; "
        "if ≫ LLM segment, tune STT model"
    )
    print(
        f"  barge_in_to_stop_ms / estimated_interrupt  ≤ {INTERRUPT_MS_DEFAULT:.0f} ms "
        f"(edge profile estimate {INTERRUPT_MS_EDGE:.0f} ms) — usually from simulated benchmark, not turn_metrics"
    )
    print()


def bottleneck_label(m: dict) -> str:
    """Classify dominant stage using the same signals as main._print_latency_bottleneck_hint."""
    stt = m.get("speech_detected_to_stt_final_ms")
    mem = m.get("stt_final_to_memory_ready_ms")
    llm_s = m.get("memory_ready_to_first_llm_sentence_ms")
    llm_b = m.get("memory_ready_to_llm_response_ready_ms")
    llm = llm_s if llm_s is not None else llm_b

    if mem is not None and mem >= 800.0:
        return "memory/context (embeddings/DB)"
    if stt is not None and llm is not None:
        if stt >= 2500.0 and stt >= 0.55 * max(llm, 1.0):
            return "final STT"
        if llm >= 4000.0 and llm >= 1.8 * max(stt or 0, 1.0):
            return "LLM + phrase buffer"
    if stt is not None and llm is not None:
        if stt > llm * 1.2:
            return "final STT (larger share)"
        if llm > stt * 1.2:
            return "LLM + phrase buffer (larger share)"
    return "mixed / inconclusive"


def check_turn_metrics(row: dict) -> tuple[list[str], list[str]]:
    """Return (failures, passes) human-readable lines."""
    fails: list[str] = []
    ok: list[str] = []
    for key, limit in SLO_MS.items():
        if key not in row:
            continue
        v = row[key]
        if not isinstance(v, (int, float)):
            continue
        if v > limit:
            fails.append(f"FAIL {key}={v:.2f} ms (limit {limit:.0f})")
        else:
            ok.append(f"OK   {key}={v:.2f} ms (limit {limit:.0f})")

    llm_seg = row.get("memory_ready_to_first_llm_sentence_ms")
    if isinstance(llm_seg, (int, float)) and llm_seg > SLO_LLM_FIRST_MS:
        fails.append(
            f"WARN memory_ready_to_first_llm_sentence_ms={llm_seg:.2f} ms "
            f"(>{SLO_LLM_FIRST_MS:.0f} ms — LLM/streaming segment hot)"
        )
    elif isinstance(llm_seg, (int, float)):
        ok.append(
            f"OK   memory_ready_to_first_llm_sentence_ms={llm_seg:.2f} ms "
            f"(guidance ≤{SLO_LLM_FIRST_MS:.0f} for interactive feel)"
        )

    return fails, ok


def parse_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"{path}:{line_no}: JSON skip: {e}", file=sys.stderr)
                continue
            if obj.get("event") == "turn_metrics":
                rows.append(obj)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Check turn_metrics against latency SLOs")
    ap.add_argument(
        "jsonl",
        nargs="?",
        type=Path,
        help="Diagnostics JSONL from --diagnostics-log-path (turn_metrics events)",
    )
    ap.add_argument(
        "--map",
        action="store_true",
        help="Print field → SLO mapping and exit",
    )
    ap.add_argument(
        "--demo",
        action="store_true",
        help="Run checks on synthetic payloads (high-CPU profile example)",
    )
    args = ap.parse_args()

    if args.map:
        print_mapping()
        return 0

    if args.demo:
        print_mapping()
        demo_high = {
            "event": "turn_metrics",
            "speech_detected_to_stt_final_ms": 3449.05,
            "stt_final_to_memory_ready_ms": 12.0,
            "memory_ready_to_first_llm_sentence_ms": 12103.35,
            "stt_final_to_first_llm_sentence_ms": 3449.05 + 12103.35,
            "speech_detected_to_first_audio_ms": 15500.0,
            "runtime_profile": "max_quality",
            "stt_model": "large-v3-turbo",
            "llm_model": "large-local",
        }
        print("Demo payload (illustrative CPU-heavy turn):\n")
        print(json.dumps({k: v for k, v in demo_high.items() if k != "event"}, indent=2))
        print()
        fails, ok = check_turn_metrics(demo_high)
        for x in ok:
            print(x)
        for x in fails:
            print(x)
        print(f"Bottleneck: {bottleneck_label(demo_high)}")
        stt, llm = demo_high.get("speech_detected_to_stt_final_ms"), demo_high.get(
            "memory_ready_to_first_llm_sentence_ms"
        )
        if isinstance(stt, (int, float)) and isinstance(llm, (int, float)):
            tot = stt + llm
            if tot > 0:
                print(
                    f"STT vs LLM segment share: STT {100 * stt / tot:.1f}%  |  "
                    f"LLM {100 * llm / tot:.1f}%"
                )
        return 1 if fails else 0

    if args.jsonl is None:
        ap.print_help()
        print("\nExample: python scripts/check_latency_slo.py logs/trace.jsonl")
        return 2

    if not args.jsonl.is_file():
        print(f"Not found: {args.jsonl}", file=sys.stderr)
        return 2

    rows = parse_jsonl(args.jsonl)
    if not rows:
        print(f"No turn_metrics events in {args.jsonl}", file=sys.stderr)
        return 2

    exit_fail = 0
    for i, row in enumerate(rows):
        print(f"--- turn_metrics #{i + 1} ts={row.get('timestamp', '?')}")
        meta = {
            k: row[k]
            for k in (
                "runtime_profile",
                "stt_model",
                "llm_model",
                "llm_min_phrase_words",
            )
            if k in row
        }
        if meta:
            print(f"    meta: {meta}")
        fails, ok = check_turn_metrics(row)
        for x in ok:
            print(f"    {x}")
        for x in fails:
            print(f"    {x}")
            exit_fail = 1
        print(f"    Bottleneck: {bottleneck_label(row)}")
        stt = row.get("speech_detected_to_stt_final_ms")
        llm = row.get("memory_ready_to_first_llm_sentence_ms")
        if isinstance(stt, (int, float)) and isinstance(llm, (int, float)):
            tot = stt + llm
            if tot > 0:
                print(
                    f"    STT vs LLM share: STT {100 * stt / tot:.1f}%  |  "
                    f"LLM segment {100 * llm / tot:.1f}%"
                )
        print()

    return exit_fail


if __name__ == "__main__":
    raise SystemExit(main())
