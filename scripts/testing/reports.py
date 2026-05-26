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
    _SUMMARY_RE = re.compile(
        r"(?P<failed>\d+) failed|(?P<passed>\d+) passed|(?P<skipped>\d+) skipped|"
        r"(?P<xfailed>\d+) xfailed|(?P<xpassed>\d+) xpassed|(?P<errors>\d+) errors?"
    )

    def parse(self, *, stage: str, returncode: int, duration_sec: float, stdout_path: Path, junit_path: Path) -> ParsedReport:
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
        summary = {
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

        summary["total"] = int(summary.get("passed", 0)) + int(summary.get("failed", 0)) + int(summary.get("errors", 0)) + int(summary.get("skipped", 0))
        summary["failure_count"] = len(failures)
        self._add_failure_metrics(summary, failures)
        summary["stdout_path"] = str(stdout_path)
        summary["junit_path"] = str(junit_path)
        return ParsedReport(summary=summary, failures=failures)

    def write(self, parsed: ParsedReport, summary_path: Path, failures_path: Path) -> None:
        summary_path.write_text(json.dumps(parsed.summary, indent=2), encoding="utf-8")
        failures_path.write_text(json.dumps(parsed.failures, indent=2), encoding="utf-8")

    def _parse_junit(self, junit_path: Path, summary: dict[str, object], failures: list[dict[str, object]]) -> None:
        root = ET.parse(junit_path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        for suite in suites:
            summary["passed"] = int(summary["passed"]) + max(
                0,
                int(float(suite.attrib.get("tests", 0)))
                - int(float(suite.attrib.get("failures", 0)))
                - int(float(suite.attrib.get("errors", 0)))
                - int(float(suite.attrib.get("skipped", 0))),
            )
            summary["failed"] = int(summary["failed"]) + int(float(suite.attrib.get("failures", 0)))
            summary["errors"] = int(summary["errors"]) + int(float(suite.attrib.get("errors", 0)))
            summary["skipped"] = int(summary["skipped"]) + int(float(suite.attrib.get("skipped", 0)))

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
                self._classify_failure({
                    "nodeid": nodeid,
                    "file": classname.replace(".", "/") + ".py" if classname else "",
                    "test_name": name,
                    "type": node.tag,
                    "message": node.attrib.get("message", ""),
                    "duration_sec": float(testcase.attrib.get("time", 0.0)),
                    "traceback_excerpt": (node.text or "")[-4000:],
                    "area": self._infer_area(nodeid),
                })
            )

    def _parse_stdout_summary(self, stdout: str, summary: dict[str, object]) -> None:
        for match in self._SUMMARY_RE.finditer(stdout):
            for key, value in match.groupdict().items():
                if value is not None:
                    summary[key] = int(summary.get(key, 0)) + int(value)

    def _fallback_failures(self, stdout: str) -> list[dict[str, object]]:
        lines = [line for line in stdout.splitlines() if line.startswith("FAILED ")]
        return [
            self._classify_failure({
                "nodeid": line.removeprefix("FAILED ").split(" - ", 1)[0],
                "file": line.split("::", 1)[0].removeprefix("FAILED "),
                "test_name": line,
                "type": "failure",
                "message": line,
                "duration_sec": 0.0,
                "traceback_excerpt": "",
                "area": self._infer_area(line),
            })
            for line in lines
        ]

    def _classify_failure(self, failure: dict[str, object]) -> dict[str, object]:
        text = " ".join(
            str(failure.get(key, ""))
            for key in ("nodeid", "message", "traceback_excerpt")
        ).lower()

        category = "unknown"
        cause = "No automatic cause matched; inspect traceback and test metadata."
        solvable = True
        suggested_fix = "Inspect the failing test and implementation."

        if "test_real_usage_open_data" in text and "background_" in text:
            category = "unsolvable_without_intent_gate"
            cause = (
                "The input is real human speech-like background audio. In ungated "
                "listening mode the recorder has no wakeword, speaker, or intent "
                "signal, so raw audio alone cannot prove whether speech is directed "
                "at the assistant."
            )
            solvable = False
            suggested_fix = (
                "Use wakeword/speaker verification/intent gating for background "
                "speech rejection, or change the product requirement for ungated mode."
            )
        elif "echo" in text and ("false interrupt" in text or "self-interrupt" in text or "self_barge" in text):
            category = "echo_alignment_gap"
            cause = (
                "Assistant audio leaked back into the microphone with delay, "
                "reverb, clipping, or sample-rate artifacts that the current "
                "echo similarity gate did not recognize as echo."
            )
            suggested_fix = (
                "Use lag-aware cross-correlation/echo alignment, room echo floor "
                "tracking, or stronger AEC reference handling."
            )
        elif (
            "test_failure_discovery_audio" in text and "background_" in text
        ) or "babble" in text or "tv speech" in text or "environment change" in text:
            category = "background_noise_gate_gap"
            cause = (
                "Speech-like background noise passes VAD/noise gates, often because "
                "the noise floor is missing or stale."
            )
            suggested_fix = (
                "Require calibrated baseline, update noise floor dynamically, and "
                "reject signals already present before assistant speech."
            )
        elif "whisper" in text or "quiet" in text or "soft-spoken" in text:
            category = "weak_user_speech_gap"
            cause = (
                "Low-energy user speech falls below RMS or VAD confidence gates."
            )
            suggested_fix = "Add AGC/normalization or lower VAD threshold for weak speech."
        elif "recorded" in text or "session_" in text:
            category = "recorded_regression"
            cause = (
                "A previously recorded session replays differently under the current "
                "pipeline or needs human annotation."
            )
            suggested_fix = "Inspect turn.json annotation and compare mic/tts replay metrics."

        failure["category"] = category
        failure["cause"] = cause
        failure["solvable"] = solvable
        failure["suggested_fix"] = suggested_fix
        return failure

    def _add_failure_metrics(
        self,
        summary: dict[str, object],
        failures: list[dict[str, object]],
    ) -> None:
        categories = Counter(str(f.get("category", "unknown")) for f in failures)
        areas = Counter(str(f.get("area", "general")) for f in failures)
        solvability = Counter(
            "solvable" if f.get("solvable", True) else "unsolvable"
            for f in failures
        )
        false_accept_categories = {
            "unsolvable_without_intent_gate",
            "background_noise_gate_gap",
            "echo_alignment_gap",
        }
        false_reject_categories = {"weak_user_speech_gap"}
        false_accepts = sum(categories.get(category, 0) for category in false_accept_categories)
        false_rejects = sum(categories.get(category, 0) for category in false_reject_categories)
        total = max(int(summary.get("total", 0)), 1)

        summary["failure_categories"] = dict(sorted(categories.items()))
        summary["failure_areas"] = dict(sorted(areas.items()))
        summary["failure_solvability"] = dict(sorted(solvability.items()))
        summary["quality_metrics"] = {
            "false_accept_failures": false_accepts,
            "false_reject_failures": false_rejects,
            "false_accept_failure_rate": round(false_accepts / total, 4),
            "false_reject_failure_rate": round(false_rejects / total, 4),
            "unsolvable_failure_count": solvability.get("unsolvable", 0),
            "solvable_failure_count": solvability.get("solvable", 0),
        }

    def _infer_area(self, nodeid: str) -> str:
        lowered = nodeid.lower()
        if "real_usage_open_data" in lowered:
            return "real_usage"
        if "barge" in lowered or "echo" in lowered or "audio" in lowered:
            return "audio"
        if "stt" in lowered or "transcrib" in lowered:
            return "stt"
        if "tts" in lowered:
            return "tts"
        if "llm" in lowered:
            return "llm"
        if "record" in lowered or "session" in lowered:
            return "recorded"
        if "profile" in lowered:
            return "profiles"
        return "general"
