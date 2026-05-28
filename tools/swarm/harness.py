#!/usr/bin/env python3
"""Speaker swarm harness -- the build/test/perf engine the agents + loop drive.

One command that, on *this* machine, gives a machine-readable verdict on whether
a code version is good:

    python -m tools.swarm.harness test     # hermetic CI-parity test suite
    python -m tools.swarm.harness perf      # latency smoke (--fake unless --real)
    python -m tools.swarm.harness all       # test + perf, write report (default)
    python -m tools.swarm.harness report    # print the last report

Always writes ``.agents/last_report.json`` (the loop's "is this version green?"
signal) and appends a one-line entry to ``.agents/history.jsonl``.

Hermetic by design: exports ``SPEAKER_NO_LOCAL_CONFIG=1`` so a dev box with real
model paths in ``config.local.json`` behaves like CI (``--engine sherpa`` fails
fast instead of starting the live loop and hanging the suite).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # speaker/
AGENTS = ROOT / ".agents"
PY = os.environ.get("SPEAKER_PY", str(ROOT / ".venv" / "bin" / "python"))
if not Path(PY).exists():
    PY = sys.executable

# LiveKit transport tests need a live server; CI ignores them too.
IGNORE = ["tests/test_livekit_audio.py", "tests/test_livekit_engine.py"]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def run_tests(extra: list[str] | None = None) -> dict:
    """Run the hermetic CI-parity suite; return structured counts."""
    env = dict(os.environ, SPEAKER_NO_LOCAL_CONFIG="1")
    cmd = [
        PY, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
        "--timeout=120", "--timeout-method=signal", "-o", "addopts=",
    ]
    for ig in IGNORE:
        cmd += ["--ignore", ig]
    cmd += extra or []
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    dur = round(time.time() - t0, 2)

    # conftest writes a structured digest under logs/tests/.
    summary = {}
    digests = sorted(glob.glob(str(ROOT / "logs" / "tests" / "tests-*.summary.json")))
    if digests:
        try:
            summary = json.loads(Path(digests[-1]).read_text())
        except Exception:
            summary = {}
    counts = summary.get("counts", {})
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "duration_sec": dur,
        "passed": counts.get("passed"),
        "failed": counts.get("failed"),
        "skipped": counts.get("skipped"),
        "failures": [f["test"] for f in summary.get("failures", [])],
        "digest": digests[-1] if digests else None,
    }


def run_perf(real: bool = False) -> dict:
    """Latency check. Default --fake (no downloads); --real runs the 4090 pipeline."""
    args = [PY, "-m", "tools.bench"]
    args += (["--profile", "desktop_gpu_4090"] if real else ["--fake"])
    t0 = time.time()
    try:
        proc = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
        tail = (proc.stdout or "")[-2000:] + (proc.stderr or "")[-500:]
        return {
            "ok": proc.returncode == 0,
            "mode": "real" if real else "fake",
            "returncode": proc.returncode,
            "duration_sec": round(time.time() - t0, 2),
            "tail": tail.strip(),
        }
    except Exception as exc:  # bench is optional / may need models
        return {"ok": False, "mode": "real" if real else "fake", "error": str(exc)}


def write_report(report: dict) -> None:
    AGENTS.mkdir(parents=True, exist_ok=True)
    (AGENTS / "last_report.json").write_text(json.dumps(report, indent=2) + "\n")
    line = {
        "ts": report["ts"], "sha": report["sha"],
        "tests_ok": report.get("tests", {}).get("ok"),
        "passed": report.get("tests", {}).get("passed"),
        "failed": report.get("tests", {}).get("failed"),
        "perf_ok": report.get("perf", {}).get("ok") if "perf" in report else None,
    }
    with (AGENTS / "history.jsonl").open("a") as fh:
        fh.write(json.dumps(line) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Speaker swarm build/test/perf harness")
    ap.add_argument("command", nargs="?", default="all", choices=["test", "perf", "all", "report"])
    ap.add_argument("--real", action="store_true", help="perf: run the real 4090 pipeline (downloads models)")
    args = ap.parse_args()

    if args.command == "report":
        p = AGENTS / "last_report.json"
        print(p.read_text() if p.exists() else "{}")
        return 0

    report: dict = {"ts": _now(), "sha": _git_sha(), "machine": os.uname().nodename}
    if args.command in ("test", "all"):
        report["tests"] = run_tests()
    if args.command in ("perf", "all"):
        report["perf"] = run_perf(real=args.real)
    report["green"] = bool(report.get("tests", {}).get("ok", True))
    write_report(report)

    print("\n=== swarm harness report ===")
    print(json.dumps({k: v for k, v in report.items() if k != "perf"}, indent=2))
    if "perf" in report:
        print(f"perf: mode={report['perf'].get('mode')} ok={report['perf'].get('ok')}")
    print(f"GREEN={report['green']}  (.agents/last_report.json)")
    return 0 if report["green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
