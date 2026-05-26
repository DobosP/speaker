from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import shutil


@dataclass(frozen=True)
class StageArtifacts:
    root: Path
    stage: str

    @property
    def stage_dir(self) -> Path:
        return self.root / self.stage

    @property
    def stdout_path(self) -> Path:
        return self.stage_dir / "stdout.txt"

    @property
    def junit_path(self) -> Path:
        return self.stage_dir / "junit.xml"

    @property
    def summary_path(self) -> Path:
        return self.stage_dir / "summary.json"

    @property
    def failures_path(self) -> Path:
        return self.stage_dir / "failures.json"

    @property
    def llm_summary_path(self) -> Path:
        return self.stage_dir / "llm-summary.md"

    @property
    def duplicates_path(self) -> Path:
        return self.stage_dir / "duplicates.json"

    def prepare(self) -> None:
        self.stage_dir.mkdir(parents=True, exist_ok=True)


class ArtifactStore:
    def __init__(self, base_dir: str | Path = "test-reports", run_id: str | None = None):
        self.base_dir = Path(base_dir)
        self.run_id = run_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
        self.root = self.base_dir / self.run_id

    def prepare(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        latest = self.base_dir / "latest"
        if latest.exists() or latest.is_symlink():
            if latest.is_symlink() or latest.is_file():
                latest.unlink()
            else:
                shutil.rmtree(latest)
        try:
            latest.symlink_to(self.root.name, target_is_directory=True)
        except OSError:
            latest.write_text(str(self.root), encoding="utf-8")

    def for_stage(self, stage: str) -> StageArtifacts:
        artifacts = StageArtifacts(root=self.root, stage=stage)
        artifacts.prepare()
        return artifacts
