from __future__ import annotations

import json
import subprocess
import sys
import time

from .artifacts import ArtifactStore
from .duplicates import DuplicateScanner
from .reports import ReportParser
from .stages import TestStage
from .summary import LLMSummary


class PytestRunner:
    def __init__(self, artifact_store: ArtifactStore, parser: ReportParser | None = None):
        self.artifact_store = artifact_store
        self.parser = parser or ReportParser()
        self.summary_writer = LLMSummary()
        self.test_timeout = None
        try:
            with open("config.json", "r") as f:
                cfg = json.load(f)
                self.test_timeout = cfg.get("test_timeout")
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
            sys.executable,
            "-m",
            "pytest",
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

        duplicates = DuplicateScanner().write(artifacts.duplicates_path)
        parsed.summary["duplicates_path"] = str(artifacts.duplicates_path)
        parsed.summary["duplicate_audio_hash_groups"] = len(
            duplicates.get("failure_corpus", {}).get("duplicate_audio_hash_groups", [])
        )
        artifacts.summary_path.write_text(json.dumps(parsed.summary, indent=2), encoding="utf-8")
        self.summary_writer.write_stage(
            path=artifacts.llm_summary_path,
            summary=parsed.summary,
            failures=parsed.failures,
        )
        return parsed.summary
