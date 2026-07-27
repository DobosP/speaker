from __future__ import annotations

import json
from pathlib import Path


def _verdict(summary: dict[str, object]) -> str:
    """Stage verdict from a parsed summary.

    PASS requires positive evidence, not just an absence of counted failures:
    a native crash (sherpa's C++ exit(-1), 2026-07 incident) kills pytest
    before any junit/summary exists, so counts come back all-zero -- the old
    counts-only logic called that PASS for 10 days. ``reports.ReportParser``
    stamps ``crash_evidence`` on exactly those runs (non-zero returncode with
    zero counted failures, or zero outcomes at all); here that is a FAIL.
    Allowed-to-fail lanes and preflight-skipped tiers keep their designed
    non-red verdicts."""
    if summary.get("preflight_failed"):
        return "SKIPPED (preflight)"
    allowed = bool(summary.get("allowed_to_fail"))
    if summary.get("crash_evidence"):
        rc = summary.get("returncode")
        return "PASS (allowed)" if allowed else f"FAIL (crash? rc={rc}, no outcomes)"
    failed = int(summary.get("failed", 0)) + int(summary.get("errors", 0))
    if failed == 0 and int(summary.get("returncode", 0) or 0) != 0:
        # Defense in depth: a red exit code is never a PASS, even if the
        # parser produced no explicit evidence record.
        rc = summary.get("returncode")
        return "PASS (allowed)" if allowed else f"FAIL (rc={rc})"
    if failed == 0:
        return "PASS"
    return "PASS (allowed)" if allowed else "FAIL"


def find_interrupted_runs(log_dir: Path, since: float | None = None) -> list[dict[str, object]]:
    """Detect hard interpreter deaths from the ``logs/tests`` run-log contract.

    ``tests/conftest.py`` writes ``tests-<run_id>.started.json`` eagerly (and
    flushed) at session start and removes it after the ``.summary.json`` is
    written at session finish. A started marker with no matching summary is
    therefore positive evidence that a pytest session began and never finished
    -- exactly what a native crash leaves behind (and what used to leave only
    a mysterious zero-byte ``.txt``). Returns one record per orphaned marker,
    newest first; ``since`` (epoch seconds) limits the scan to markers written
    after a given moment, e.g. the current stage's start."""
    orphans: list[dict[str, object]] = []
    if not log_dir.is_dir():
        return orphans
    for marker in sorted(log_dir.glob("tests-*.started.json"), reverse=True):
        if since is not None and marker.stat().st_mtime < since:
            continue
        run_id = marker.name.removeprefix("tests-").removesuffix(".started.json")
        if (log_dir / f"tests-{run_id}.summary.json").exists():
            continue  # finished cleanly; marker removal just hasn't happened
        record: dict[str, object] = {"run_id": run_id, "marker": str(marker)}
        try:
            record.update(json.loads(marker.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass  # the marker's existence is the evidence; contents are a bonus
        orphans.append(record)
    return orphans


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
        if summary.get("crash_evidence"):
            lines.extend(
                [
                    "## NATIVE CRASH SUSPECTED",
                    "",
                    str(summary["crash_evidence"]),
                    "",
                ]
            )
        interrupted = summary.get("interrupted_test_logs") or []
        if interrupted:
            lines.extend(
                [
                    "## Interrupted test sessions (started, never finished)",
                    "",
                ]
            )
            for orphan in interrupted:
                lines.append(
                    f"- run `{orphan.get('run_id')}`: started marker "
                    f"`{orphan.get('marker')}` has no matching summary -- the "
                    "pytest session died before pytest_sessionfinish (hard "
                    "interpreter death)."
                )
            lines.append("")
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
        crashed_stages: list[str] = []
        for summary in run_summaries:
            for key in totals:
                totals[key] += int(summary.get(key, 0))
            duration += float(summary.get("duration_sec", 0.0))
            if _verdict(summary).startswith("FAIL"):
                overall = "FAIL"
                if summary.get("crash_evidence"):
                    crashed_stages.append(str(summary.get("stage")))

        lines = [
            "# Test Run Summary",
            "",
            f"**Overall: {overall}** — "
            f"{totals['passed']} passed, {totals['failed']} failed, "
            f"{totals['errors']} errors, {totals['skipped']} skipped "
            f"across {len(run_summaries)} stage(s) in {round(duration, 2)}s.",
            "",
        ]
        if crashed_stages:
            lines.extend(
                [
                    f"**NATIVE CRASH SUSPECTED** in stage(s): {', '.join(crashed_stages)} "
                    "— pytest died without recording outcomes; see the stage "
                    "llm-summary.md and stdout.txt.",
                    "",
                ]
            )
        lines.extend(
            [
                "| Stage | Verdict | Passed | Failed | Errors | Skipped | Duration |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for summary in run_summaries:
            lines.append(
                f"| {summary.get('stage')} | {_verdict(summary)} "
                f"| {summary.get('passed', 0)} | {summary.get('failed', 0)} "
                f"| {summary.get('errors', 0)} | {summary.get('skipped', 0)} "
                f"| {summary.get('duration_sec')}s |"
            )
        lines.append("")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
