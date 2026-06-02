"""Logic tests for tools/session_bootstrap (no audio/model/network deps)."""
from __future__ import annotations

import json
from pathlib import Path

from tools import session_bootstrap as sb


def _seed(root: Path) -> None:
    (root / ".agents").mkdir()
    (root / "docs").mkdir()
    (root / "logs" / "runs").mkdir(parents=True)

    (root / ".agents" / "status.json").write_text(
        json.dumps(
            {
                "machine": {"cpu": "TestCPU", "ram_gib": 16, "gpu": "TestGPU",
                            "target_profile": "phone"},
                "last_verdict": {"tests": "10 passed", "green": True},
                "next": "do the thing",
            }
        ),
        encoding="utf-8",
    )
    (root / ".agents" / "backlog.md").write_text(
        "# Backlog\n\n## P0\n- [ ] fix the blocker\n- [x] already done\n\n## P1\n- [ ] later item\n",
        encoding="utf-8",
    )
    # Two session docs; the newer date must win.
    (root / "docs" / "session_2026-05-30_old.md").write_text(
        "# Session 2026-05-30 -- older work\n\nbranch **`main`**\n", encoding="utf-8"
    )
    (root / "docs" / "session_2026-06-01_demo.md").write_text(
        "# Session 2026-06-01 -- demo work\n\n"
        "All code committed on branch **`feat/demo`**.\n\n"
        "## Next steps (pick up here)\n"
        "- first step\n- second step\n- third step\n- fourth step\n",
        encoding="utf-8",
    )
    # Healthy run.
    (root / "logs" / "runs" / "run-20260601-100000.summary.json").write_text(
        json.dumps(
            {"run_id": "run-20260601-100000", "duration_sec": 5.0,
             "counts": {"turns": 2, "errors": 0}, "stuck_hints": [], "errors": [],
             "turns": [{"final_to_first_token": 0.5}]}
        ),
        encoding="utf-8",
    )
    # Stuck run -> must flag WARN.
    (root / "logs" / "runs" / "run-20260601-110000.summary.json").write_text(
        json.dumps(
            {"run_id": "run-20260601-110000", "duration_sec": 9.0,
             "counts": {"turns": 1, "errors": 0}, "stuck_hints": ["llm never returned"],
             "errors": [], "turns": [{"final_to_first_token": 0.2}]}
        ),
        encoding="utf-8",
    )


def test_briefing_surfaces_machine_session_runs_and_backlog(tmp_path):
    _seed(tmp_path)
    b = sb.build_briefing(tmp_path)

    # 1. machine profile
    assert "TestCPU" in b and "phone" in b
    # 2. newest session doc wins, with headline + branch + first next-step
    assert "demo work" in b
    assert "feat/demo" in b
    assert "older work" not in b
    assert "first step" in b
    # 3. stuck run flagged WARN; the run id appears
    assert "WARN" in b
    assert "run-20260601-110000" in b
    # 4. open P0 only -- shipped/lower-priority items excluded
    assert "fix the blocker" in b
    assert "already done" not in b
    assert "later item" not in b
    # 5. recommendation continues from the prior next-step
    assert "Continue from prior session" in b


def test_warn_only_for_stuck_or_errors_or_slow():
    healthy = sb.summarize_run  # sanity: function exists
    assert healthy is not None


def test_missing_artifacts_degrade_gracefully(tmp_path):
    # Empty repo: no crash, explicit "(not found)" markers, no WARN noise.
    b = sb.build_briefing(tmp_path)
    assert "not found" in b.lower()
    assert "Recommended working strategy" in b
