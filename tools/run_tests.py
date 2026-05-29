#!/usr/bin/env python3
"""Run staged pytest suites and write structured reports.

    python tools/run_tests.py list           # show stages
    python tools/run_tests.py core            # run one stage
    python tools/run_tests.py fast            # quick dev subset (skips slow/network/llm/backend)
    python tools/run_tests.py full            # run the whole suite
    python tools/run_tests.py e2e             # end-to-end CLI/process tests
    python tools/run_tests.py all-stages      # run every stage in order
    python tools/run_tests.py full --json     # machine-readable summary

Run any stage in parallel by passing pytest-xdist through ``--pytest-arg``
(repeat the flag once per token -- they are forwarded verbatim to pytest):

    python tools/run_tests.py full --pytest-arg=-n --pytest-arg=auto

The suite is parallel-safe (~9s with ``-n auto`` vs ``~35s`` serial). Parallel
is left opt-in here -- the default invocation stays serial for easy debugging.

Reports land in ``test-reports/<run_id>/`` (with a ``latest`` pointer):
per-stage ``summary.json`` / ``failures.json`` / ``llm-summary.md``, plus a
run-level ``summary.json`` and a tabular ``llm-summary.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python tools/run_tests.py` to import the sibling `testing` package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from testing.artifacts import ArtifactStore  # noqa: E402
from testing.runner import PytestRunner  # noqa: E402
from testing.stages import StageRegistry  # noqa: E402
from testing.summary import LLMSummary  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged pytest suites with reports.")
    parser.add_argument("command", help="Stage name, 'all-stages', or 'list'.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a stage fails.")
    parser.add_argument("--allow-failures", action="store_true", help="Exit 0 even if a stage fails.")
    parser.add_argument("--maxfail", type=int, default=None, help="Pass --maxfail=N to pytest.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary.")
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Extra argument passed through to pytest, repeatable "
             "(e.g. parallel: --pytest-arg=-n --pytest-arg=auto; "
             "postgres: --pytest-arg=--postgres).",
    )
    return parser.parse_args()


def _run_stages(args: argparse.Namespace, registry: StageRegistry) -> int:
    stage_names = registry.names() if args.command == "all-stages" else [args.command]
    stages = registry.select(stage_names)  # validates names up front

    store = ArtifactStore()
    store.prepare()
    runner = PytestRunner(store)
    summaries: list[dict[str, object]] = []
    exit_code = 0
    for stage in stages:
        summary = runner.run_stage(
            stage,
            maxfail=args.maxfail,
            allow_failures=args.allow_failures,
            extra_pytest_args=args.pytest_arg,
        )
        summaries.append(summary)
        failed = int(summary.get("returncode", 0)) != 0
        allowed = bool(summary.get("allowed_to_fail")) or args.allow_failures
        if failed and not allowed:
            exit_code = int(summary.get("returncode", 1))
            if not args.keep_going:
                break

    LLMSummary().write_run_index(path=store.root / "llm-summary.md", run_summaries=summaries)
    (store.root / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps({"run": str(store.root), "stages": summaries}, indent=2))
    else:
        print(f"Wrote test reports to {store.root}")
        print(f"Run summary: {store.root / 'llm-summary.md'}")
    return exit_code


def main() -> int:
    args = _parse_args()
    registry = StageRegistry.default()
    if args.command == "list":
        payload = registry.describe()
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for stage in payload:
                print(f"{stage['name']}: {stage['purpose']}")
        return 0
    return _run_stages(args, registry)


if __name__ == "__main__":
    raise SystemExit(main())
