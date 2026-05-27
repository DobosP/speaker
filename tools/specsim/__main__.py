"""CLI: python -m tools.specsim [--out PATH] [--json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .report import render
from .simulate import SCENARIOS, simulate_all
from .specs import CATALOG


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulate on-device capability across specs.")
    parser.add_argument(
        "--out",
        default="test-reports/specsim/index.html",
        help="output HTML path (default: test-reports/specsim/index.html)",
    )
    parser.add_argument("--json", action="store_true", help="also print results as JSON to stdout")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(CATALOG, SCENARIOS), encoding="utf-8")

    if args.json:
        results = [
            {
                "spec": r.spec,
                "scenario": r.scenario,
                "first_audio_latency": r.first_audio_latency,
                "response_complete": r.response_complete,
                "barge_in_stop": r.barge_in_stop,
                "total": r.total,
            }
            for r in simulate_all(CATALOG)
        ]
        print(json.dumps(results, indent=2))
    else:
        print(f"Wrote spec-simulation report to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
