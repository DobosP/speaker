"""CLI for the autonomous test harness.

    python -m tools.autotest memory   [--llm echo|ollama] [--model gemma3:12b] [--fast-model minicpm5-1b:q8]
    python -m tools.autotest voice    [--llm echo|ollama] [--model gemma3:12b] [--fast-model minicpm5-1b:q8]
    python -m tools.autotest replay   [--bundle logs/runs/run-<id>.wav]
    python -m tools.autotest suite                       # existing pytest gates
    python -m tools.autotest all                         # everything + scorecard

Each tier prints a PASS/FAIL line and writes a JSON report under
``logs/autotest/`` (gitignored alongside other run artifacts). Exit code is
non-zero if any requested tier fails, so it doubles as a CI gate.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict

from .verdicts import (
    DIAGNOSTIC_PASS,
    FAIL,
    INCOMPLETE,
    PASS,
    aggregate_reports,
    evaluate_voice,
    finite_nonnegative,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "logs", "autotest")


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _write(report: dict, name: str) -> str:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"autotest-{_stamp()}-{name}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return path


# --------------------------------------------------------------------------- #
# tiers
# --------------------------------------------------------------------------- #
def tier_memory(args) -> dict:
    from .memory_probe import run_memory_probe

    r = run_memory_probe(
        llm_kind=args.llm,
        model=args.fast_model,
        main_model=args.model,
    )
    rep = {"tier": "memory", **asdict(r)}
    sem = "" if r.answer_uses_fact is None else f" answer_uses_fact={r.answer_uses_fact}"
    label = {
        PASS: "PASS", FAIL: "FAIL", DIAGNOSTIC_PASS: "DIAGNOSTIC PASS",
    }[r.outcome]
    print(f"[memory] {label} "
          f"({r.llm_label}, {r.turns} turns, {r.duration_sec}s) "
          f"recall_available={r.recall_available} "
          f"recall_injected={r.recall_injected} fenced={r.recall_fenced}{sem} "
          f"answer_model={r.answer_model} route={r.answer_route} "
          f"topology_valid={r.topology_valid} controller={r.controller_answer} "
          f"cross_session={r.cross_session} recent_history_clean={r.recent_history_clean}")
    print(f"         answer: {r.answer[:120]!r}")
    return rep


def tier_voice(args) -> dict:
    from core.config import load_config
    from .voice_loop import run_voice_loop

    cfg = load_config()
    sherpa_cfg = cfg.get("sherpa", {})
    out_dir = os.path.join(OUT, f"voice-{_stamp()}")
    r = run_voice_loop(
        repo_root=REPO, sherpa_cfg=sherpa_cfg,
        llm_kind=args.llm,
        main_model=args.model,
        fast_model=args.fast_model,
        out_dir=out_dir,
        acoustics_mode=args.acoustics, latency_ms=args.latency_ms,
        utterances_dir=args.utterances, aec_delay_ms=args.aec_delay_ms,
        make_sound=args.make_sound, inject_sink=args.inject_sink,
    )
    rep = {"tier": "voice", **asdict(r)}
    if r.error:
        print(f"[voice] FAIL ({r.mode}): {r.error}")
        rep.update(
            ok=False,
            complete=False,
            diagnostic_ok=False,
            outcome=FAIL,
            verdict={
                "passed": False,
                "checks_ok": False,
                "complete": False,
                "outcome": FAIL,
                "checks": (),
                "failures": ("runner_error",),
                "not_covered": (),
            },
        )
        return rep

    has_echo = r.mode in ("delay", "speaker")

    # bundle analysis (transcript / turns / barge)
    bun = {}
    bundle_error = ""
    if r.summary_path and os.path.exists(r.summary_path):
        try:
            bun = _analyze_bundle(r.summary_path)
        except (OSError, ValueError, TypeError) as exc:
            bundle_error = f"{type(exc).__name__}: {exc}"
        else:
            rep["bundle"] = bun

    # Grade the exact generation-bearing final selected after each injection,
    # not a best-match sweep over the whole transcript bundle.  The latter can
    # pair a prompt with an unrelated earlier/later turn and false-green both
    # STT and playback causality.
    stt = r.prompt_score or {"mean_wer": 1.0, "n": 0, "pairs": []}
    rep["stt"] = stt
    prompt_evidence = r.prompt_evidence or {}

    sc = r.scenarios
    s1 = sc.get("s1_round_trip", {})
    s2 = sc.get("s2_self_interrupt", {})
    s3 = sc.get("s3_barge_in", {})
    error_count = int(bun.get("error_count", 0) or 0)
    if bundle_error:
        error_count += 1
    verdict = evaluate_voice(
        mode=r.mode,
        engine_ready=r.ready,
        summary_present=bool(bun) and not bundle_error,
        monitor_rms=r.monitor_rms,
        round_trip_clips=int(s1.get("clips", 0) or 0),
        assistant_spoke=s1.get("assistant_spoke") is True,
        # Independent finalized-bundle receipt count.  Same-generation causal
        # binding is enforced by S1/S2/S3; retaining this gate also rejects a
        # malformed/non-finite or incomplete metrics bundle.
        assistant_audio_turns=int(bun.get("first_audio_turns", 0) or 0),
        expected_audio_turns=int(
            prompt_evidence.get("expected_audio_prompts", 0) or 0
        ),
        expected_wer_n=int(
            prompt_evidence.get("expected_labelled_prompts", 0) or 0
        ),
        wer_n=int(stt.get("n", 0) or 0),
        mean_wer=float(stt.get("mean_wer", 1.0)),
        error_count=error_count,
        stuck_hints=tuple(bun.get("stuck_hints", ()) or ()),
        self_interrupt_pass=s2.get("live_pass"),
        barge_pass=s3.get("pass"),
        barge_latency_s=s3.get("barge_latency_s"),
    )
    rep["ok"] = verdict.passed
    rep["complete"] = verdict.complete
    rep["diagnostic_ok"] = verdict.checks_ok
    rep["outcome"] = verdict.outcome
    rep["verdict"] = asdict(verdict)
    if bundle_error:
        rep["bundle_error"] = bundle_error

    label = {
        PASS: "PASS",
        FAIL: "FAIL",
        DIAGNOSTIC_PASS: "DIAGNOSTIC PASS",
    }[verdict.outcome]
    print(f"[voice] {label} (pipeline) mode={r.mode} "
          f"run_id={r.run_id} clips={r.clip_source} monitor_rms={r.monitor_rms:.3f}")
    print(
        f"        causal STT mean WER={stt.get('mean_wer')} "
        f"over {stt.get('n')} clips"
    )
    for pair in stt.get("pairs", []):
        print(
            f"          WER={float(pair.get('wer', 1.0)):.2f}  "
            f"ref={pair.get('ref', '')!r}  hyp={pair.get('hyp', '')!r}"
        )
    # the conversation: the FINAL text the LLM received -> its reply, so messy /
    # self-correcting input can be judged for whether the system made sense of it.
    convo = bun.get("conversation", [])
    if convo:
        print("        conversation (final text fed to LLM -> reply):")
        for role, text in convo[:24]:
            tag = "you " if role == "user" else "asst" if role == "assistant" else role[:4]
            print(f"          {tag}: {text[:90]!r}")
        if len(convo) > 24:
            print(f"          ... (+{len(convo) - 24} more; full transcript in the report)")
    if has_echo:
        print(f"        self-interrupt: live barge-ins during own reply="
              f"{s2.get('barge_ins_during_own_reply')} (pass={s2.get('live_pass')})")
        print(f"        barge-in: {s3.get('barge_ins_after_talkover')} after talk-over "
              f"in {s3.get('barge_latency_s')}s (pass={s3.get('pass')})")
    else:
        print(f"        self-interrupt / barge-in: {s2.get('note','n/a')}")
    if verdict.failures:
        print(f"        failed checks: {', '.join(verdict.failures)}")
    if verdict.not_covered:
        print(f"        not covered: {', '.join(verdict.not_covered)}")
    if not verdict.passed:
        print(f"        (engine stdout: {r.log_path})")
    return rep


def _analyze_bundle(summary_path: str) -> dict:
    with open(summary_path) as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise TypeError("run bundle root must be an object")

    def records(key: str) -> list[dict]:
        value = d.get(key, [])
        if not isinstance(value, list) or any(
            not isinstance(item, dict) for item in value
        ):
            raise TypeError(f"run bundle {key} must be a list of objects")
        return value

    def strings(key: str) -> list[str]:
        value = d.get(key, [])
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise TypeError(f"run bundle {key} must be a list of strings")
        return value

    def count_value(value: object, key: str) -> int:
        if (
            not finite_nonnegative(value)
            or float(value) != int(float(value))
        ):
            raise ValueError(f"run bundle count {key} must be a finite integer")
        return int(float(value))

    tr = records("transcript")
    for entry in tr:
        if not isinstance(entry.get("role", ""), str) or not isinstance(
            entry.get("text", ""), str
        ):
            raise TypeError("run bundle transcript role/text must be strings")
    users = [t for t in tr if t.get("role") == "user"]
    asst = [t for t in tr if t.get("role") == "assistant"]
    turns = records("turns")
    barges = [
        turn
        for turn in turns
        if finite_nonnegative(turn.get("barge_in_latency"))
    ]
    errors = records("errors")
    stuck_hints = strings("stuck_hints")
    counts = d.get("counts", {})
    if not isinstance(counts, dict):
        raise TypeError("run bundle counts must be an object")
    counted_errors = count_value(counts.get("errors", 0), "errors")
    warnings = count_value(counts.get("warnings", 0), "warnings")
    return {
        "user_finals": [t.get("text", "") for t in users],
        "assistant_replies": [t.get("text", "")[:80] for t in asst],
        # ordered transcript -- the final text fed to the LLM next to its reply,
        # so disfluent/correction cases can be judged for sense-making.
        "conversation": [(t.get("role", ""), t.get("text", "")) for t in tr],
        "n_turns": len(turns),
        "n_barge_in_turns": len(barges),
        "stuck_hints": stuck_hints,
        "errors": errors,
        "first_audio_turns": sum(
            finite_nonnegative(turn.get("first_audio_latency"))
            for turn in turns
            if isinstance(turn, dict)
        ),
        "error_count": max(
            counted_errors,
            len(errors),
        ),
        "warnings": warnings,
    }


def _replay_probes(wav_path: str) -> dict:
    """Run the delay-independent coherence + AEC probes over a recorded bundle."""
    out: dict = {}
    summary_bits: list[str] = []
    # coherence self-interrupt (delay-independent)
    try:
        p = subprocess.run(
            [sys.executable, "-m", "tools.replay_barge", wav_path],
            cwd=REPO, capture_output=True, text=True, timeout=120,
        )
        out["replay_barge_stdout"] = p.stdout.strip()
        m = re.search(r"REMAINING after grace=(\d+)", p.stdout)
        if m:
            out["self_interrupts_remaining"] = int(m.group(1))
            summary_bits.append(f"self-interrupts(remaining)={m.group(1)}")
    except Exception as e:  # noqa: BLE001
        out["replay_barge_error"] = str(e)
    # AEC ERLE / best delay
    try:
        p = subprocess.run(
            [sys.executable, "-m", "tools.aec_probe", wav_path,
             "--backend", "dtln", "--max-delay-ms", "400"],
            cwd=REPO, capture_output=True, text=True, timeout=240,
        )
        out["aec_probe_stdout"] = p.stdout.strip()
        m = re.search(r"BEST:\s*delay=(\d+)ms\s*ERLE=([+\-0-9.]+)dB", p.stdout)
        if m:
            out["aec_best_delay_ms"] = int(m.group(1))
            out["aec_best_erle_db"] = float(m.group(2))
            summary_bits.append(f"AEC best={m.group(1)}ms ERLE={m.group(2)}dB")
    except Exception as e:  # noqa: BLE001
        out["aec_probe_error"] = str(e)
    out["summary"] = "; ".join(summary_bits) or "(probes produced no parse)"
    return out


def tier_replay(args) -> dict:
    """Run the replay probes over an explicit bundle or the newest with a ref."""
    wav = args.bundle
    if not wav:
        refs = sorted(glob.glob(os.path.join(REPO, "logs", "runs", "*.ref.wav")))
        if not refs:
            print("[replay] FAIL no bundle with a .ref.wav sibling found")
            return {"tier": "replay", "ok": False, "error": "no .ref.wav bundle"}
        wav = refs[-1].replace(".ref.wav", ".wav")
    rep = {"tier": "replay", "bundle": wav, **_replay_probes(wav)}
    si = rep.get("self_interrupts_remaining")
    ok = si == 0 if si is not None else False
    rep["ok"] = ok
    print(f"[replay] {'PASS' if ok else 'WARN'} {os.path.basename(wav)}: {rep['summary']}")
    return rep


def tier_suite(args) -> dict:
    """Run the existing headless barge/sandbox/memory pytest gates."""
    targets = [
        "tests/test_barge_scorecard.py",
        "tests/test_barge_requirement.py",
        "tests/test_barge_self_interrupt.py",
        "tests/test_sandbox_middle_layer.py",
        "tests/test_memory_contract.py",
    ]
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *targets],
        cwd=REPO, capture_output=True, text=True,
    )
    tail = "\n".join(p.stdout.strip().splitlines()[-6:])
    ok = p.returncode == 0
    print(f"[suite] {'PASS' if ok else 'FAIL'}\n{tail}")
    return {"tier": "suite", "ok": ok, "returncode": p.returncode, "tail": tail}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tools.autotest", description=__doc__)
    ap.add_argument("tier", choices=["memory", "voice", "replay", "suite", "all", "record"])
    ap.add_argument("--llm", choices=["echo", "ollama"], default="ollama",
                    help="LLM backend for the in-loop tiers")
    ap.add_argument("--model", default="gemma3:12b",
                    help="main/reasoning model (default gemma3:12b)")
    ap.add_argument("--fast-model", default="minicpm5-1b:q8", dest="fast_model",
                    help="fast/ordinary answering model (default minicpm5-1b:q8)")
    ap.add_argument("--bundle", default=None, help="replay: explicit run-<id>.wav")
    ap.add_argument("--acoustics", choices=["cable", "delay", "speaker"], default="cable",
                    help="voice: cable=silent null-sink loopback (default); "
                         "delay=silent loopback + ~260ms air-gap (AEC aligned); "
                         "speaker=real over-the-air (needs --make-sound)")
    ap.add_argument("--latency-ms", type=int, default=260, dest="latency_ms",
                    help="voice/delay: emulated air-gap delay (default 260)")
    ap.add_argument("--utterances", default=None,
                    help="voice: dir of recorded clips + manifest.json (real-voice "
                         "injection); default = synth the engine's own voice")
    ap.add_argument("--make-sound", action="store_true", dest="make_sound",
                    help="voice/speaker: REQUIRED to run real over-the-air "
                         "(plays the speaker + records the real mic)")
    ap.add_argument("--inject-sink", default=None, dest="inject_sink",
                    help="voice/speaker: pactl sink to play the 'user' clips into "
                         "(e.g. the laptop speaker for near/far separation); "
                         "default = the engine's output sink")
    ap.add_argument("--aec-delay-ms", type=int, default=None, dest="aec_delay_ms",
                    help="voice: force the AEC reference delay (P1 deep-dive)")
    # record-only
    ap.add_argument("--out", default="recordings/owner",
                    help="record: dir for the clips + manifest.json (gitignored)")
    ap.add_argument("--group", action="append", default=None,
                    help="record: only this group (repeatable): questions, commands, "
                         "long, barge, memory, natural")
    ap.add_argument("--limit", type=int, default=None, help="record: cap the number of clips")
    ap.add_argument("--device", default=None, help="record: input device index/name")
    ap.add_argument("--review", action="store_true", help="record: keep/redo each take")
    ap.add_argument("--dry-run", action="store_true", help="record: print the script, don't record")
    ap.add_argument("--simulate", action="store_true",
                    help="record: synthesize each line instead of recording (self-test)")
    ap.add_argument("--check", action="store_true",
                    help="record: mic level check (say a test sentence -> hot/quiet/good); "
                         "tune gain before recording for real")
    args = ap.parse_args(argv)

    if args.tier == "record":
        from .record import run_record, mic_check
        if args.check:
            mic_check(device=args.device)
            return 0
        sherpa_cfg = {}
        if args.simulate:
            from core.config import load_config
            sherpa_cfg = load_config().get("sherpa", {})
        out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
        run_record(
            out_dir=out, device=args.device, groups=args.group, limit=args.limit,
            review=args.review, dry_run=args.dry_run, simulate=args.simulate,
            sherpa_cfg=sherpa_cfg,
        )
        return 0

    reports: list[dict] = []
    if args.tier in ("memory", "all"):
        reports.append(tier_memory(args))
    if args.tier in ("voice", "all"):
        reports.append(tier_voice(args))
    if args.tier in ("replay",):
        reports.append(tier_replay(args))
    if args.tier in ("suite", "all"):
        reports.append(tier_suite(args))

    overall = aggregate_reports(
        reports,
        # ``all`` advertises the whole autonomous scorecard.  The default
        # echo-free cable remains a useful STT diagnostic, but cannot make that
        # aggregate complete without barge/self-interrupt coverage.
        require_complete=args.tier == "all",
    )
    bundle = {
        "stamp": _stamp(),
        "tier": args.tier,
        "reports": reports,
        "overall": asdict(overall),
    }
    path = _write(bundle, args.tier)
    label = {
        PASS: "PASS",
        FAIL: "FAIL",
        DIAGNOSTIC_PASS: "DIAGNOSTIC PASS",
        INCOMPLETE: "INCOMPLETE",
    }[overall.outcome]
    detail = []
    if overall.failed_tiers:
        detail.append("failed=" + ",".join(overall.failed_tiers))
    if overall.incomplete_tiers:
        detail.append("not-complete=" + ",".join(overall.incomplete_tiers))
    print(f"\nreport: {path}")
    print(f"VERDICT: {label}" + (f" ({'; '.join(detail)})" if detail else ""))
    return overall.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
