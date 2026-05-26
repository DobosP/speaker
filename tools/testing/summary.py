from __future__ import annotations

from pathlib import Path


def _verdict(summary: dict[str, object]) -> str:
    failed = int(summary.get("failed", 0)) + int(summary.get("errors", 0))
    if failed == 0:
        return "PASS"
    return "PASS (allowed)" if summary.get("allowed_to_fail") else "FAIL"


class LLMSummary:
    """Human/LLM-readable markdown reports for stages and the whole run."""

    def write_stage(
        self,
        *,
        path: Path,
        summary: dict[str, object],
        failures: list[dict[str, object]],
    ) -> None:
        lines = [
            f"# Test Stage: {summary.get('stage')}",
            "",
            f"_{summary.get('purpose', '')}_",
            "",
            "## Result",
            "",
            f"- Verdict: **{_verdict(summary)}**",
            f"- Return code: `{summary.get('returncode')}`",
            f"- Duration: `{summary.get('duration_sec')}s`",
            f"- Passed: `{summary.get('passed', 0)}` / "
            f"Failed: `{summary.get('failed', 0)}` / "
            f"Errors: `{summary.get('errors', 0)}` / "
            f"Skipped: `{summary.get('skipped', 0)}` / "
            f"Total: `{summary.get('total', 0)}`",
            "",
        ]
        areas = summary.get("failure_areas") or {}
        if areas:
            lines.extend([f"- Failure areas: `{areas}`", ""])

        if failures:
            lines.extend(["## Failures", ""])
            for failure in failures[:30]:
                lines.extend(
                    [
                        f"### {failure.get('nodeid')}",
                        "",
                        f"- Area: `{failure.get('area')}`",
                        f"- Type: `{failure.get('type')}`",
                        f"- Duration: `{failure.get('duration_sec', 0.0)}s`",
                        f"- Message: {failure.get('message') or '(no message)'}",
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
        totals = {k: 0 for k in ("passed", "failed", "errors", "skipped", "total")}
        duration = 0.0
        overall = "PASS"
        for summary in run_summaries:
            for key in totals:
                totals[key] += int(summary.get(key, 0))
            duration += float(summary.get("duration_sec", 0.0))
            if _verdict(summary) == "FAIL":
                overall = "FAIL"

        lines = [
            "# Test Run Summary",
            "",
            f"**Overall: {overall}** — "
            f"{totals['passed']} passed, {totals['failed']} failed, "
            f"{totals['errors']} errors, {totals['skipped']} skipped "
            f"across {len(run_summaries)} stage(s) in {round(duration, 2)}s.",
            "",
            "| Stage | Verdict | Passed | Failed | Errors | Skipped | Duration |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for summary in run_summaries:
            lines.append(
                f"| {summary.get('stage')} | {_verdict(summary)} "
                f"| {summary.get('passed', 0)} | {summary.get('failed', 0)} "
                f"| {summary.get('errors', 0)} | {summary.get('skipped', 0)} "
                f"| {summary.get('duration_sec')}s |"
            )
        lines.append("")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
