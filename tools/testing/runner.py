from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time

from .artifacts import ArtifactStore
from .reports import ReportParser
from .stages import TestStage
from .summary import LLMSummary


def _pytest_invocation() -> list[str]:
    """Return the argv prefix that launches pytest.

    Prefers ``python -m pytest`` (so the test process inherits the same
    interpreter + sys.path as the runner). Falls back to a standalone
    ``pytest`` binary on PATH when pytest isn't importable from
    ``sys.executable`` -- e.g. uv-managed tool installs that ship pytest
    in a separate venv from the project's interpreter."""
    try:
        subprocess.run(
            [sys.executable, "-c", "import pytest"],
            check=True, capture_output=True, timeout=5,
        )
        return [sys.executable, "-m", "pytest"]
    except Exception:
        pass
    pytest_bin = shutil.which("pytest")
    if pytest_bin:
        return [pytest_bin]
    return [sys.executable, "-m", "pytest"]  # surfaces a familiar error message


class PytestRunner:
    """Run one stage as a pytest subprocess and persist its artifacts."""

    def __init__(self, artifact_store: ArtifactStore, parser: ReportParser | None = None):
        self.artifact_store = artifact_store
        self.parser = parser or ReportParser()
        self.summary_writer = LLMSummary()
        self.test_timeout = None
        try:
            with open("config.json", "r", encoding="utf-8") as fh:
                self.test_timeout = json.load(fh).get("test_timeout")
        except Exception:
            pass

    def run_stage(
        self,
        stage: TestStage,
        *,
        maxfail: int | None = None,
        allow_failures: bool = False,
        extra_pytest_args: list[str] | None = None,
    ) -> dict[str, object]:
        artifacts = self.artifact_store.for_stage(stage.name)
        args = [
            *_pytest_invocation(),
            *stage.pytest_args(maxfail=maxfail),
            f"--junitxml={artifacts.junit_path}",
        ]
        if extra_pytest_args:
            args.extend(extra_pytest_args)

        timeout_sec = self.test_timeout if self.test_timeout is not None else stage.timeout_sec
        start = time.time()
        proc = subprocess.run(
            args,
            cwd=".",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
        )
        duration = time.time() - start
        artifacts.stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")

        parsed = self.parser.parse(
            stage=stage.name,
            returncode=proc.returncode,
            duration_sec=duration,
            stdout_path=artifacts.stdout_path,
            junit_path=artifacts.junit_path,
        )
        parsed.summary["command"] = args
        parsed.summary["purpose"] = stage.purpose
        parsed.summary["allowed_to_fail"] = bool(stage.allow_failures or allow_failures)
        self.parser.write(parsed, artifacts.summary_path, artifacts.failures_path)
        self.summary_writer.write_stage(
            path=artifacts.llm_summary_path,
            summary=parsed.summary,
            failures=parsed.failures,
        )
        return parsed.summary
