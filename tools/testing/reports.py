from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass
class ParsedReport:
    summary: dict[str, object]
    failures: list[dict[str, object]]


class ReportParser:
    """Turn a pytest run (JUnit XML + stdout) into structured counts + failures.

    Ported from the legacy audio-era runner; the failure classifier is now
    generic (area inference by module name) rather than audio-domain specific.
    """

    _SUMMARY_RE = re.compile(
        r"(?P<failed>\d+) failed|(?P<passed>\d+) passed|(?P<skipped>\d+) skipped|"
        r"(?P<xfailed>\d+) xfailed|(?P<xpassed>\d+) xpassed|(?P<errors>\d+) errors?"
    )

    def parse(
        self,
        *,
        stage: str,
        returncode: int,
        duration_sec: float,
        stdout_path: Path,
        junit_path: Path,
    ) -> ParsedReport:
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
        summary: dict[str, object] = {
            "stage": stage,
            "returncode": returncode,
            "duration_sec": round(duration_sec, 3),
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "xfailed": 0,
            "xpassed": 0,
            "total": 0,
        }
        failures: list[dict[str, object]] = []

        if junit_path.exists():
            self._parse_junit(junit_path, summary, failures)
        else:
            self._parse_stdout_summary(stdout, summary)

        if not failures and returncode != 0:
            failures.extend(self._fallback_failures(stdout))

        summary["total"] = (
            int(summary["passed"])
            + int(summary["failed"])
            + int(summary["errors"])
            + int(summary["skipped"])
        )
        summary["failure_count"] = len(failures)
        summary["failure_areas"] = dict(
            sorted(Counter(str(f.get("area", "general")) for f in failures).items())
        )
        summary["stdout_path"] = str(stdout_path)
        summary["junit_path"] = str(junit_path)
        return ParsedReport(summary=summary, failures=failures)

    def write(self, parsed: ParsedReport, summary_path: Path, failures_path: Path) -> None:
        summary_path.write_text(json.dumps(parsed.summary, indent=2), encoding="utf-8")
        failures_path.write_text(json.dumps(parsed.failures, indent=2), encoding="utf-8")

    def _parse_junit(
        self, junit_path: Path, summary: dict[str, object], failures: list[dict[str, object]]
    ) -> None:
        root = ET.parse(junit_path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        for suite in suites:
            tests = int(float(suite.attrib.get("tests", 0)))
            fails = int(float(suite.attrib.get("failures", 0)))
            errs = int(float(suite.attrib.get("errors", 0)))
            skips = int(float(suite.attrib.get("skipped", 0)))
            summary["passed"] = int(summary["passed"]) + max(0, tests - fails - errs - skips)
            summary["failed"] = int(summary["failed"]) + fails
            summary["errors"] = int(summary["errors"]) + errs
            summary["skipped"] = int(summary["skipped"]) + skips

        for testcase in root.iter("testcase"):
            fail_node = testcase.find("failure")
            err_node = testcase.find("error")
            node = fail_node if fail_node is not None else err_node
            if node is None:
                continue
            classname = testcase.attrib.get("classname", "")
            name = testcase.attrib.get("name", "")
            nodeid = f"{classname}.{name}" if classname else name
            failures.append(
                {
                    "nodeid": nodeid,
                    "file": classname.replace(".", "/") + ".py" if classname else "",
                    "test_name": name,
                    "type": node.tag,
                    "message": node.attrib.get("message", ""),
                    "duration_sec": float(testcase.attrib.get("time", 0.0)),
                    "traceback_excerpt": (node.text or "")[-4000:],
                    "area": self._infer_area(nodeid),
                }
            )

    def _parse_stdout_summary(self, stdout: str, summary: dict[str, object]) -> None:
        for match in self._SUMMARY_RE.finditer(stdout):
            for key, value in match.groupdict().items():
                if value is not None:
                    summary[key] = int(summary.get(key, 0)) + int(value)

    def _fallback_failures(self, stdout: str) -> list[dict[str, object]]:
        lines = [line for line in stdout.splitlines() if line.startswith("FAILED ")]
        return [
            {
                "nodeid": line.removeprefix("FAILED ").split(" - ", 1)[0],
                "file": line.split("::", 1)[0].removeprefix("FAILED "),
                "test_name": line,
                "type": "failure",
                "message": line,
                "duration_sec": 0.0,
                "traceback_excerpt": "",
                "area": self._infer_area(line),
            }
            for line in lines
        ]

    def _infer_area(self, nodeid: str) -> str:
        lowered = nodeid.lower()
        if "core_agent" in lowered or "agent" in lowered:
            return "agent"
        if "core_runtime" in lowered or "runtime" in lowered:
            return "runtime"
        if "sandbox" in lowered or "middle_layer" in lowered:
            return "sandbox"
        if "memory" in lowered:
            return "memory"
        if "speaker" in lowered or "gate" in lowered:
            return "speaker_gate"
        if "always_on" in lowered or "supervisor" in lowered or "planner" in lowered:
            return "brain"
        return "general"
