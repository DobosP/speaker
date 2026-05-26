from __future__ import annotations

from pathlib import Path


class LLMSummary:
    def write_stage(self, *, path: Path, summary: dict[str, object], failures: list[dict[str, object]]) -> None:
        lines = [
            f"# Test Stage: {summary.get('stage')}",
            "",
            "## Result",
            "",
            f"- Return code: `{summary.get('returncode')}`",
            f"- Duration: `{summary.get('duration_sec')}s`",
            f"- Passed: `{summary.get('passed', 0)}`",
            f"- Failed: `{summary.get('failed', 0)}`",
            f"- Errors: `{summary.get('errors', 0)}`",
            f"- Skipped: `{summary.get('skipped', 0)}`",
            "",
        ]
        metrics = summary.get("quality_metrics")
        if metrics:
            lines.extend(
                [
                    "## Quality Metrics",
                    "",
                    f"- Failure categories: `{summary.get('failure_categories', {})}`",
                    f"- Failure solvability: `{summary.get('failure_solvability', {})}`",
                    f"- False accept failures: `{metrics.get('false_accept_failures', 0)}`",
                    f"- False reject failures: `{metrics.get('false_reject_failures', 0)}`",
                    f"- False accept failure rate: `{metrics.get('false_accept_failure_rate', 0.0)}`",
                    f"- False reject failure rate: `{metrics.get('false_reject_failure_rate', 0.0)}`",
                    "",
                ]
            )
        if failures:
            lines.extend(["## Failures", ""])
            for failure in failures[:30]:
                lines.extend(
                    [
                        f"### {failure.get('nodeid')}",
                        "",
                        f"- Area: `{failure.get('area')}`",
                        f"- Category: `{failure.get('category', 'unknown')}`",
                        f"- Solvable: `{failure.get('solvable', True)}`",
                        f"- Cause: {failure.get('cause', '(not classified)')}",
                        f"- Suggested fix: {failure.get('suggested_fix', '(none)')}",
                        f"- Message: {failure.get('message') or '(no message)'}",
                        f"- Duration: `{failure.get('duration_sec', 0.0)}s`",
                        "",
                    ]
                )
                excerpt = str(failure.get("traceback_excerpt") or "").strip()
                if excerpt:
                    lines.extend(["```text", excerpt[-1200:], "```", ""])
            if len(failures) > 30:
                lines.append(f"... {len(failures) - 30} more failures omitted. See failures.json.")
        else:
            lines.extend(["## Failures", "", "No failures recorded.", ""])
        path.write_text("\n".join(lines), encoding="utf-8")

    def write_run_index(self, *, path: Path, run_summaries: list[dict[str, object]]) -> None:
        lines = ["# Test Run Summary", ""]
        for summary in run_summaries:
            lines.append(
                f"- `{summary.get('stage')}`: return `{summary.get('returncode')}`, "
                f"passed `{summary.get('passed', 0)}`, failed `{summary.get('failed', 0)}`, "
                f"errors `{summary.get('errors', 0)}`, duration `{summary.get('duration_sec')}s`"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
