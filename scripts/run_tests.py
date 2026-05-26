#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from testing.artifacts import ArtifactStore
from testing.duplicates import DuplicateScanner
from testing.runner import PytestRunner
from testing.stages import StageRegistry
from testing.summary import LLMSummary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run staged pytest suites with reports.",
        epilog=(
            "Stages with allow_failures (see `list --json`: discovery, backend, all) "
            "mark the run as allowed_to_fail; the process still exits 0 on pytest failure."
        ),
    )
    parser.add_argument(
        "command",
        help="Stage name, or one of: list, analyze",
    )
    parser.add_argument("--keep-going", action="store_true", help="Continue after stage failures.")
    parser.add_argument("--allow-failures", action="store_true", help="Exit 0 even if selected stage fails.")
    parser.add_argument("--maxfail", type=int, default=None, help="Pass --maxfail=N to pytest.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary to stdout.")
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Extra argument passed through to pytest. Repeat as needed.",
    )
    return parser.parse_args()


def _run_stage(args: argparse.Namespace, registry: StageRegistry) -> int:
    stage_names = registry.names() if args.command == "all-stages" else [args.command]
    store = ArtifactStore()
    store.prepare()
    runner = PytestRunner(store)
    summaries = []
    exit_code = 0
    for name in stage_names:
        stage = registry.get(name)
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
        print(f"Latest summary: {store.root / 'llm-summary.md'}")
    return exit_code


def _analyze(args: argparse.Namespace) -> int:
    store = ArtifactStore()
    store.prepare()
    artifacts = store.for_stage("analyze")
    report = DuplicateScanner().write(artifacts.duplicates_path)
    artifacts.summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Test Analysis", ""]
    corpus = report.get("failure_corpus", {})
    lines.extend(
        [
            f"- Failure corpus present: `{corpus.get('present')}`",
            f"- Failure corpus cases: `{corpus.get('case_count', 0)}`",
            f"- Duplicate audio hash groups: `{len(corpus.get('duplicate_audio_hash_groups', []))}`",
            f"- Duplicate corpus names: `{len(corpus.get('duplicate_names', {}))}`",
            f"- Duplicate test short names: `{len(report.get('test_functions', {}).get('duplicate_short_names', {}))}`",
        ]
    )
    artifacts.llm_summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Wrote analysis to {artifacts.stage_dir}")
    return 0


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
    if args.command == "analyze":
        return _analyze(args)
    return _run_stage(args, registry)


if __name__ == "__main__":
    raise SystemExit(main())
