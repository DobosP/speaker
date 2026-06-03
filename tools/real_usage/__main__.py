"""CLI for the REAL-USAGE test harness.

Runs the user's recordings through the REAL pipeline (real STT -> real LLM ->
real TTS) with the REAL laptop audio OUTPUT -- the assistant SPEAKS ALOUD, so the
real ALSA output path + the playback thread + shutdown ARE exercised (the parts
the headless FileReplayEngine skips, where the live failures live). It feeds each
recorded WAV into the recognizer in place of the mic (patching ONLY
sd.InputStream), leaving sd.OutputStream real.

Runs ONLY on request -- it needs real ASR/TTS/LLM models and real audio hardware.

    python -m tools.real_usage --check               # preflight (models/audio)
    python -m tools.real_usage --list-devices
    python -m tools.real_usage                        # default: logs/runs/run-*.wav
    python -m tools.real_usage --recordings logs/runs --output-device 4 \
        --model gemma3:4b --fast-model gemma3:4b
    python -m tools.real_usage --wav logs/runs/run-20260530-181513.wav

It writes a gradeable per-recording report (report.md + report.json) under
logs/real_usage/<run-id>/ and exits non-zero if any recording FAILs.

By default each fixture runs in its OWN SUBPROCESS so a single hung shutdown
(the ALSA out.write() blocking forever -- the exact failure under test) cannot
wedge the whole batch: the parent times out the child and records the HANG.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from . import report as report_mod

log = logging.getLogger("speaker.real_usage")

DEFAULT_GLOB = "logs/runs/run-*.wav"


def _preflight(config: dict, output_device) -> list[str]:
    """Return a list of human-readable problems (empty == ready to run).
    Mirrors tools/live_session._preflight + checks the output device."""
    problems: list[str] = []
    try:
        import sounddevice as sd  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        problems.append(f"sounddevice not importable ({exc}); pip install sounddevice + a PortAudio backend")
        return problems
    sherpa = config.get("sherpa", {}) or {}
    if not sherpa.get("asr_encoder") or not sherpa.get("asr_tokens"):
        problems.append("sherpa.asr_encoder / asr_tokens not set -- no ASR model (config.json)")
    if not sherpa.get("tts_model"):
        problems.append("sherpa.tts_model not set -- no TTS voice (config.json)")
    dev = output_device if output_device is not None else sherpa.get("output_device")
    if dev is not None:
        try:
            import sounddevice as sd

            info = sd.query_devices(dev if not str(dev).isdigit() else int(dev))
            if int(info.get("max_output_channels", 0)) < 1:
                problems.append(f"output device {dev!r} has no output channels")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"output device {dev!r} not usable: {exc}")
    return problems


def _list_audio_devices() -> None:
    try:
        import sounddevice as sd

        print(sd.query_devices())
    except Exception as exc:  # noqa: BLE001
        print(f"could not list audio devices: {exc}")


def _resolve_wavs(args) -> list[Path]:
    paths: list[str] = []
    if args.wav:
        for w in args.wav:
            paths.extend(sorted(glob.glob(w)) if any(c in w for c in "*?[") else [w])
    if args.recordings:
        for rec in args.recordings:
            p = Path(rec)
            if p.is_dir():
                # Prefer the run-*.wav convention; fall back to every .wav in the
                # dir if there are no run-* recordings.
                in_dir = sorted(glob.glob(str(p / "run-*.wav")))
                if not in_dir:
                    in_dir = sorted(glob.glob(str(p / "*.wav")))
                paths.extend(in_dir)
            else:
                paths.extend(sorted(glob.glob(rec)) if any(c in rec for c in "*?[") else [rec])
    if not paths:
        paths = sorted(glob.glob(DEFAULT_GLOB))
    # De-dup, keep order, drop non-wav.
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        rp = str(Path(p))
        if rp.endswith(".wav") and rp not in seen and Path(rp).exists():
            seen.add(rp)
            out.append(Path(rp))
    return out


def _measure_input(wav: Path):
    """Best-effort level measurement for the silent-skip pre-pass. Returns the
    level dict, or None if the WAV couldn't be read (then we just run it)."""
    try:
        from .runner import load_and_measure

        return load_and_measure(wav)
    except Exception as exc:  # noqa: BLE001 - a bad/missing WAV must not abort the batch
        log.warning("could not measure %s level (%s); running it anyway", wav.name, exc)
        return None


def _load_config(args):
    from core.config import _apply_device_profile, _load_config

    config = _load_config()
    device = args.device or config.get("device", "desktop")
    config = _apply_device_profile(config, device)
    if args.output_device is not None:
        # Force the REAL output device (mirrors app.py CLI override) so the
        # assistant speaks aloud through it and the ALSA output path is exercised.
        config.setdefault("sherpa", {})["output_device"] = args.output_device
    return config


def _run_one_subprocess(wav: Path, args, timeout: float) -> dict:
    """Run ONE fixture in a child process and read back its result JSON. A child
    that won't exit within ``timeout`` IS the SHUTDOWN HANG reproduction -- we
    kill it and synthesize a FAIL result."""
    fd, result_path = tempfile.mkstemp(prefix="real_usage_", suffix=".json")
    os.close(fd)
    cmd = [
        sys.executable, "-m", "tools.real_usage",
        "--_run-one", str(wav),
        "--_result-file", result_path,
        "--llm", args.llm,
        "--response-timeout", str(args.response_timeout),
        "--start-timeout", str(args.start_timeout),
        "--shutdown-timeout", str(args.shutdown_timeout),
    ]
    if args.model:
        cmd += ["--model", args.model]
    if args.fast_model:
        cmd += ["--fast-model", args.fast_model]
    if args.device:
        cmd += ["--device", args.device]
    if args.output_device is not None:
        cmd += ["--output-device", str(args.output_device)]
    if args.open_mic:
        cmd += ["--open-mic"]
    try:
        subprocess.run(cmd, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        log.error("fixture %s subprocess exceeded %.0fs -- SHUTDOWN HANG (killed)", wav.name, timeout)
        try:
            os.unlink(result_path)
        except OSError:
            pass
        # The child wrote no result (or wrote it and then hung in shutdown). A
        # process that won't exit == the hang; record it as a shutdown FAIL.
        return {
            "fixture": wav.name,
            "asr_finals": [],
            "spoken": [],
            "first_audio_latencies": [],
            "barge_in_count": 0,
            "playback_errors": [],
            "playback_loop_dead": False,
            "shutdown_ok": False,
            "shutdown_seconds": timeout,
            "shutdown_timeout": args.shutdown_timeout,
            "error": f"subprocess did not exit within {timeout:.0f}s (shutdown hang)",
        }
    try:
        with open(result_path, "r") as fh:
            result = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        result = {
            "fixture": wav.name, "asr_finals": [], "spoken": [],
            "first_audio_latencies": [], "barge_in_count": 0, "playback_errors": [],
            "playback_loop_dead": False, "shutdown_ok": False,
            "shutdown_seconds": None, "shutdown_timeout": args.shutdown_timeout,
            "error": f"could not read child result: {exc}",
        }
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass
    return result


def _run_one_inprocess(wav: Path, config: dict, args) -> dict:
    from .runner import run_fixture

    return run_fixture(
        config, wav,
        llm_backend=args.llm,
        main_model=args.model,
        fast_model=args.fast_model,
        response_timeout=args.response_timeout,
        shutdown_timeout=args.shutdown_timeout,
        open_mic=args.open_mic,
        start_timeout=args.start_timeout,
    )


def _child_run_one(args) -> int:
    """Internal: execute exactly one fixture and write its result JSON. Invoked
    by the parent in subprocess mode (--_run-one / --_result-file)."""
    from .runner import run_fixture

    config = _load_config(args)
    result = run_fixture(
        config, Path(args._run_one),
        llm_backend=args.llm,
        main_model=args.model,
        fast_model=args.fast_model,
        response_timeout=args.response_timeout,
        shutdown_timeout=args.shutdown_timeout,
        open_mic=args.open_mic,
        start_timeout=args.start_timeout,
    )
    with open(args._result_file, "w") as fh:
        json.dump(result, fh)
    # IMPORTANT: if shutdown hung, run_fixture still returned (its own join timed
    # out) -- but a wedged daemon thread may keep this process alive. Force-exit
    # so the parent's read succeeds and the process actually dies.
    if not result.get("shutdown_ok", True):
        os._exit(0)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tools.real_usage", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recordings", action="append",
                   help="recording dir(s) or glob(s) (default: logs/runs/run-*.wav)")
    p.add_argument("--wav", action="append", help="specific WAV file(s) / glob(s) (repeatable)")
    p.add_argument("--output-device", default=None,
                   help="speaker device index/name the assistant speaks through (e.g. 4)")
    p.add_argument("--device", default=None, help="device profile from config.json (default: desktop)")
    p.add_argument("--llm", default="ollama", choices=["echo", "ollama"])
    p.add_argument("--model", default=None, help="main LLM model override (e.g. gemma3:4b)")
    p.add_argument("--fast-model", default=None, dest="fast_model",
                   help="fast LLM model override (e.g. gemma3:4b)")
    p.add_argument("--response-timeout", type=float, default=60.0,
                   help="per-recording wait for the spoken response (default 60s; clips are 26-44s)")
    p.add_argument("--start-timeout", type=float, default=20.0, dest="start_timeout",
                   help="how long to wait for the assistant to even BEGIN responding before "
                        "moving on (default 20s; raise for long open-mic captures)")
    p.add_argument("--shutdown-timeout", type=float, default=8.0,
                   help="hard timeout on runtime.stop(); exceeding it == the shutdown HANG (default 8s)")
    p.add_argument("--open-mic", action="store_true",
                   help="play the recording aloud through the speakers and leave the REAL mic OPEN "
                        "(don't patch sd.InputStream) -- the only variant that reproduces the "
                        "acoustic-echo barge-in STORM organically. Flakier; needs real speakers+mic.")
    p.add_argument("--in-process", action="store_true",
                   help="run all fixtures in THIS process (default: a subprocess per fixture so a "
                        "single shutdown hang can't wedge the batch)")
    p.add_argument("--barge-in-threshold", type=int,
                   default=report_mod.DEFAULT_BARGE_IN_STORM_THRESHOLD,
                   help="barge-in fires above this in ONE recording == a self-storm FAIL (default 1)")
    p.add_argument("--check", action="store_true", help="preflight (models/audio/device) and exit")
    p.add_argument("--list", "--list-recordings", dest="list_only", action="store_true",
                   help="list the recordings that would run and exit")
    p.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    p.add_argument("--inventory", action="store_true",
                   help="scan the run-log bundles (logs/runs/*.summary.json + WAV level) and write a "
                        "history OVERVIEW (no pipeline run); flags digitally-silent/empty captures")
    p.add_argument("--inventory-out", default=None,
                   help="where --inventory writes the overview markdown (default logs/runs/OVERVIEW.md)")
    p.add_argument("--out-dir", default=None, help="report root (default logs/real_usage/<run-id>)")
    # Internal flags for subprocess-per-fixture mode (not for direct use).
    p.add_argument("--_run-one", default=None, help=argparse.SUPPRESS)
    p.add_argument("--_result-file", default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    # Child worker (subprocess mode): run exactly one fixture, write JSON, exit.
    if args._run_one:
        if not args._result_file:
            print("--_run-one requires --_result-file", file=sys.stderr)
            return 2
        return _child_run_one(args)

    if args.list_devices:
        _list_audio_devices()
        return 0

    if args.inventory:
        from . import inventory as inv

        runs_dir = "logs/runs"
        for rec in (args.recordings or []):
            if Path(rec).is_dir():
                runs_dir = rec
                break
        print(f"scanning run bundles under {runs_dir} ...")
        rows = inv.scan_runs(runs_dir)
        out_path = Path(args.inventory_out) if args.inventory_out else Path(runs_dir) / "OVERVIEW.md"
        inv.write_inventory(rows, out_path)
        empties = inv.empty_wavs(rows)
        n_wav = sum(1 for r in rows if r.get("has_wav"))
        print(f"{len(rows)} run(s); {n_wav} with a WAV; {len(empties)} empty/silent.")
        if empties:
            print("empty captures (prune candidates):")
            for rid in empties:
                print(f"  - {rid}")
        print(f"\nOverview: {out_path}")
        return 0

    config = _load_config(args)
    wavs = _resolve_wavs(args)

    if args.list_only:
        print(f"{len(wavs)} recording(s):")
        for w in wavs:
            print(f"  {w}")
        return 0

    problems = _preflight(config, args.output_device)
    if args.check or problems:
        if problems:
            print("PREFLIGHT PROBLEMS:")
            for prob in problems:
                print(f"  - {prob}")
            print("\nFix these, then re-run. (--list-devices to see audio devices.)")
            return 1
        print("preflight OK: models + audio + output device look ready.")
        return 0

    if not wavs:
        print(f"no recordings found (looked for {DEFAULT_GLOB}). Pass --wav / --recordings.")
        return 2

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("logs/real_usage") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== real-usage validation {run_id} -> {out_dir} ===")
    print(f"{len(wavs)} recording(s); the assistant WILL speak aloud through "
          f"device {args.output_device if args.output_device is not None else '(default)'}.\n")

    # A subprocess gets the response window + shutdown timeout + generous startup
    # slack (model load + warm). Exceeding it == the hang.
    child_timeout = args.response_timeout + args.shutdown_timeout + 90.0

    results: list[dict] = []
    for wav in wavs:
        print(f"--- {wav.name} ---")
        t0 = time.time()
        # Silent pre-pass: a digitally-silent capture (dead/muted mic) is graded
        # EMPTY without building a runtime or speaking aloud -- so a known-bad
        # recording can't masquerade as a "went deaf" pipeline failure.
        level = _measure_input(wav)
        if level is not None and report_mod.is_silent_input(level["rms"], level["peak"]):
            result = report_mod.empty_result(
                wav.name, input_rms=level["rms"], input_peak=level["peak"])
            results.append(result)
            print(f"    EMPTY: digitally-silent capture "
                  f"(rms={level['rms']:.5f}, peak={level['peak']:.5f}) -- skipped, not run")
            continue
        if args.in_process:
            result = _run_one_inprocess(wav, config, args)
        else:
            result = _run_one_subprocess(wav, args, child_timeout)
        result.setdefault("fixture", wav.name)
        if level is not None:
            result["input_rms"] = level["rms"]
            result["input_peak"] = level["peak"]
        results.append(result)
        print(f"    done in {time.time() - t0:.1f}s "
              f"(barge_ins={result.get('barge_in_count')}, "
              f"shutdown_ok={result.get('shutdown_ok')})")

    run = report_mod.grade_run(results, barge_in_storm_threshold=args.barge_in_threshold)
    paths = report_mod.write_reports(run, out_dir, run_id=run_id)
    report_mod.print_summary(run)
    print(f"\nReport: {paths['markdown']}")
    print(f"        {paths['json']}")
    return 0 if run["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
