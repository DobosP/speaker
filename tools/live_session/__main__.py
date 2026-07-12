"""CLI for the live on-hardware validation harness.

    python -m tools.live_session --list
    python -m tools.live_session --scenario baseline_latency_single_turn_qa
    python -m tools.live_session --all
    python -m tools.live_session --check          # preflight only (models/audio)

Runs only on request. Acoustic mode needs real audio hardware; ``--inject``
patches both device streams and opens no microphone or speaker. Both modes need
the configured ASR/TTS models. See docs/live_validation_2026-05.md.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .driver import LiveConversation, apply_inject_profile
from .report import (
    write_grade,
    write_latency_report,
    write_suite_report,
    write_summary,
    write_timeline,
)
from .scenarios import SCENARIOS, by_name, resolve_suite, suite_names

log = logging.getLogger("speaker.live")


def _expected_machine_grade_counts(scenario) -> tuple[int, int]:
    """Return exact barge opportunities and response rows owned by a scenario."""
    turns = tuple(getattr(scenario, "turns", ()) or ())
    expected_barges = sum(
        1
        for index, turn in enumerate(turns)
        if index > 0
        and getattr(turn, "timing", "") == "barge_in"
        and getattr(turns[index - 1], "timing", "") != "barge_in"
    )
    expected_responses = sum(
        1
        for turn in turns
        if getattr(turn, "timing", "") not in ("immediately", "barge_in")
        and bool(
            tuple(getattr(turn, "expect", ()) or ())
            or tuple(getattr(turn, "forbid", ()) or ())
        )
    )
    return expected_barges, expected_responses


def _inject_grade_failed(
    grade: dict,
    *,
    expected_barges: int,
    expected_responses: int,
) -> bool:
    """Fail closed on every machine-owned verdict in a no-device run."""
    full_duplex = grade.get("full_duplex")
    if not isinstance(full_duplex, dict) or full_duplex.get("ok") is not True:
        return True
    barge = grade.get("barge_in")
    if not isinstance(barge, dict) or barge.get("verdict") != "ok":
        return True
    intended = barge.get("n_intended_barges")
    stopped = barge.get("n_stopped")
    if (
        type(intended) is not int
        or type(stopped) is not int
        or intended != expected_barges
        or stopped != expected_barges
    ):
        return True
    response = grade.get("response")
    if not isinstance(response, dict):
        return True
    aggregate = response.get("aggregate")
    if not isinstance(aggregate, dict):
        return True
    response_count = aggregate.get("n")
    if type(response_count) is not int or response_count < 0:
        return True
    if response_count != expected_responses:
        return True
    if response_count and aggregate.get("verdict") != "ok":
        return True
    return False


def _machine_grade_exit_code(
    rc: int,
    grade: dict,
    *,
    inject: bool,
    expected_barges: int = 0,
    expected_responses: int = 0,
) -> int:
    """Preserve operational errors; make an injected grade failure nonzero."""
    if rc != 0 or not inject:
        return rc
    failed = _inject_grade_failed(
        grade,
        expected_barges=expected_barges,
        expected_responses=expected_responses,
    )
    return 1 if failed else 0


def _preflight(config: dict, *, llm: str = "ollama", **check_kwargs) -> list[str]:
    """Return shared runtime-readiness failures (empty == ready to run).

    ``config`` is already device-profile merged by ``main``.  Reusing doctor's
    selected-profile checks prevents this live gate from saying READY while the
    same devices/word-cut route fail doctor or normal sherpa startup.
    """
    from core.readiness import run_runtime_checks

    checks = run_runtime_checks(
        config, resolved=True, llm_mode=llm, **check_kwargs
    )
    problems: list[str] = []
    for check in checks:
        if check.ok:
            continue
        problem = f"{check.name}: {check.detail}".rstrip()
        if check.hint:
            problem += f"; fix: {check.hint}"
        problems.append(problem)
    return problems


def _list_audio_devices() -> None:
    try:
        import sounddevice as sd

        print(sd.query_devices())
    except Exception as exc:  # noqa: BLE001
        print(f"could not list audio devices: {exc}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tools.live_session", description=__doc__)
    p.add_argument("--scenario", action="append", help="scenario name(s) to run (repeatable)")
    p.add_argument("--all", action="store_true", help="run every scenario")
    p.add_argument("--suite", default=None,
                   help="run a named suite of scenarios: " + ", ".join(suite_names())
                        + " (e.g. --suite acoustic for the over-the-air set; "
                          "--suite latency for the latency profile; --suite realistic)")
    p.add_argument("--repeat", type=int, default=1,
                   help="run each chosen scenario N times -- builds a bigger latency "
                        "sample (the pooled SUITE distribution gets N x the turns)")
    p.add_argument("--list", action="store_true", help="list scenarios and exit")
    p.add_argument("--list-suites", action="store_true", help="list suites and exit")
    p.add_argument("--check", action="store_true", help="preflight (models/audio) and exit")
    p.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    p.add_argument("--device", default=None, help="device profile from config.json (e.g. desktop)")
    p.add_argument("--llm", default="ollama", choices=["echo", "ollama"])
    p.add_argument("--model", default=None, help="main LLM model override")
    p.add_argument("--fast-model", default=None, help="fast LLM model override")
    p.add_argument("--user-speaker-id", type=int, default=None, help="synthetic-user TTS speaker id")
    p.add_argument("--user-speed", type=float, default=None, help="synthetic-user TTS speed")
    p.add_argument("--no-assistant-audio", action="store_true", help="don't re-synthesize the assistant track")
    p.add_argument("--input-device", default=None)
    p.add_argument("--output-device", default=None)
    p.add_argument("--response-timeout", type=float, default=45.0)
    p.add_argument("--no-input-gate", action="store_true",
                   help="disable the ACT/INGEST addressing gate (answer every heard utterance -- "
                        "useful when over-the-air STT is garbled enough that the gate INGESTs it)")
    p.add_argument("--smart-endpoint", action="store_true",
                   help="enable EXPERIMENTAL semantic turn-completion endpointing (core/endpointing.py) "
                        "for a live A/B latency validation: commit a final EARLY when the partial reads "
                        "as a complete turn (reclaims the ~1.2s trailing-silence wait, the #1 first-audio "
                        "latency win) and HOLD past the acoustic timer when it ends mid-phrase. In-memory, "
                        "per-run override only -- the committed config.json default stays OFF. VALIDATE: "
                        "SHORTEN finals MUST match the full-acoustic finals, especially turns ending on a "
                        "preposition ('...capital of France'); a truncated final means min_silence_sec is "
                        "below the decoder lookahead and the feature must NOT be trusted.")
    p.add_argument("--inject", action="store_true",
                   help="feed the synthetic-user audio straight into the recognizer instead of "
                        "playing it over the air -- clean STT->LLM->TTS with no acoustic "
                        "degradation/feedback (the reliable path when built-in speaker+mic is noisy)")
    p.add_argument("--user-volume", type=float, default=None,
                   help="ACOUSTIC ONLY: scale the synthetic-user PLAYBACK amplitude, 0..1 "
                        "(default = native TTS amplitude, byte-identical to before). Lower it "
                        "(e.g. --user-volume 0.4) to reduce the level hitting a near-field mic -- "
                        "one half of the over-the-air SNR knob. The saved user/NN.wav stays "
                        "full-scale; only the audio played over the speakers is attenuated.")
    p.add_argument("--input-gain", type=float, default=None,
                   help="set sherpa.input_gain for this run (default = leave config untouched, "
                        "so the engine keeps its configured gain). The OTHER half of the SNR knob: "
                        "dial it DOWN (e.g. --input-gain 1) to stop a loud near-field speaker from "
                        "SATURATING the open mic and garbling STT. Mirrors the --no-input-gate / "
                        "--smart-endpoint in-memory config override.")
    p.add_argument("--endpoint-min-silence", type=float, default=None,
                   help="override sherpa.endpoint_min_silence_sec for this run -- the EARLIEST "
                        "trailing-silence (since the partial last grew) at which a confident-complete "
                        "turn commits. The dominant latency lever: lowering it cuts the ~900ms endpoint "
                        "wait but risks splitting at a comma pause (a too-early commit is merged back by "
                        "the continuation layer). A/B it (diff the ON finals vs a higher-floor run for "
                        "truncation/splitting). In-memory per-run override only.")
    p.add_argument("--endpoint-max-silence", type=float, default=None,
                   help="override sherpa.endpoint_max_silence_sec for this run (the bounded hold cap "
                        "for a mid-phrase partial). In-memory per-run override only.")
    p.add_argument("--endpoint-high-confidence-floor", type=float, default=None,
                   help="override sherpa.endpoint_high_confidence_floor for this run -- the LOWER "
                        "trailing-silence floor a HIGH-confidence completion (lexical 0.75 bin, a "
                        "normal ending word) commits at, instead of the full --endpoint-min-silence. "
                        "0 disables (uniform floor). The adaptive latency win: A/B 0.55 vs 0 and diff "
                        "the finals for splits/truncation. In-memory per-run override only.")
    p.add_argument("--denoise", action="store_true",
                   help="force sherpa.denoise_enabled=True for this run (the GTCRN speech-denoise "
                        "front-end, default OFF). A/B it against a no-flag run to measure its effect "
                        "on STT -- pair with --noise-snr to inject controlled noise into the played "
                        "audio. NB the denoiser can OVER-suppress clean over-the-air speech, so a "
                        "clean A/B may show it HURTS; its win is under genuine broadband noise. "
                        "In-memory per-run override only (committed config default stays OFF).")
    p.add_argument("--noise-snr", type=float, default=None,
                   help="ACOUSTIC ONLY: overlay deterministic broadband noise on the synthetic-user "
                        "PLAYBACK at this SNR in dB (e.g. --noise-snr 8 = noisy, 20 = mild). The "
                        "controlled-noise half of the denoise A/B: run --noise-snr N with and without "
                        "--denoise and compare STT. Only the played buffer is corrupted; the saved "
                        "user/NN.wav reference clip stays clean.")
    p.add_argument("--barge-in", action="store_true",
                   help="force sherpa.barge_in_enabled=True. In acoustic mode this measures "
                        "talk-over on the real mic and requires physical validation. With --inject "
                        "it exercises the documented echo-free detector/control profile without "
                        "opening audio hardware; that result is not acoustic evidence.")
    p.add_argument("--out-dir", default=None, help="artifact root (default logs/live/<run-id>)")
    args = p.parse_args(argv)

    if args.user_volume is not None and not (0.0 <= args.user_volume <= 1.0):
        p.error("--user-volume must be between 0 and 1")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    if args.list:
        for s in SCENARIOS:
            n_graded = sum(1 for t in s.turns if t.expect or t.forbid)
            tag = f"[{n_graded} graded]" if n_graded else ""
            print(f"  {s.name:42}  {s.capability}  {tag}")
        return 0
    if args.list_suites:
        for name in suite_names():
            members = [s.name for s in resolve_suite(name)]
            print(f"  {name:10} ({len(members):2})  {', '.join(members)}")
        return 0
    if args.list_devices:
        _list_audio_devices()
        return 0

    # Validate a --suite name up front (before touching config/models) so an
    # unknown suite errors fast and identically everywhere.
    if args.suite and args.suite not in suite_names():
        print(f"unknown suite: {args.suite!r}. Use --list-suites. "
              f"Known: {', '.join(suite_names())}")
        return 2

    from core.config import _apply_device_profile, _load_config

    config = _load_config()
    device = args.device or config.get("device", "desktop")
    try:
        config = _apply_device_profile(config, device, strict=True)
    except ValueError as exc:
        print(f"invalid device profile: {exc}")
        return 2
    for key in ("input_device", "output_device"):
        val = getattr(args, key)
        if val is not None:
            config.setdefault("sherpa", {})[key] = val
    if args.no_input_gate:
        config.setdefault("input_gate", {})["enabled"] = False
    if args.smart_endpoint:
        config.setdefault("sherpa", {})["endpoint_enabled"] = True
    if args.endpoint_min_silence is not None:
        config.setdefault("sherpa", {})["endpoint_min_silence_sec"] = float(args.endpoint_min_silence)
    if args.endpoint_max_silence is not None:
        config.setdefault("sherpa", {})["endpoint_max_silence_sec"] = float(args.endpoint_max_silence)
    if args.endpoint_high_confidence_floor is not None:
        config.setdefault("sherpa", {})["endpoint_high_confidence_floor"] = float(
            args.endpoint_high_confidence_floor)
    if args.input_gain is not None:
        config.setdefault("sherpa", {})["input_gain"] = float(args.input_gain)
    if not args.inject:
        # ACOUSTIC mode: the synthetic user AND the assistant both drive the one
        # output device, but exclusive-ALSA hardware allows only ONE open output
        # stream. Have the engine release its TTS stream when idle so the device is
        # free for the synthetic user's next line (turn-taking hands it back).
        config.setdefault("sherpa", {})["release_output_when_idle"] = True
    else:
        # INJECT mode: the assistant's TTS goes to a null sink, so NONE of it
        # reaches the recognizer -- there is no acoustic echo to cancel. Leaving
        # AEC on mis-applies the post-AEC residual-FLOOR barge gate, which expects
        # a real echo floor and rejects a clean injected barge ("0.4s of voiced
        # speech ... did not trip the gate"). Turn AEC off so barge-in gates on the
        # reference-coherence/level path over the clean injected audio -- the
        # injected barge is energy the (teed) playback can't explain, so it fires.
        # Centralized with recorded-owner replay so every no-device caller has
        # identical authority and detector semantics (ADR-0052/0053).
        apply_inject_profile(config)
    if args.barge_in:
        config.setdefault("sherpa", {})["barge_in_enabled"] = True
        if not args.inject:
            # MEASURED 2026-06-01: acoustic (non-inject) --barge-in on a SINGLE
            # exclusive-ALSA output device can hard-CRASH (PortAudio SIGFPE / core
            # dump) -- the synthetic user opening its playback stream while the
            # assistant's just-barged output stream still holds the same device
            # races inside PortAudio's stream-open. It also self-interrupts on the
            # assistant's own TTS leaking into the open mic (no AEC). The self-barge
            # SIGNAL still appears in the log before any crash. For reliable barge
            # grading use --inject; for a REAL acoustic barge use a SEPARATE
            # --output-device for the interrupter, or AEC.
            print("WARNING: acoustic --barge-in on a single output device is FRAGILE "
                  "-- it can self-interrupt on the assistant's own TTS and can crash "
                  "PortAudio (SIGFPE) on the shared exclusive-ALSA device. Prefer "
                  "--inject for barge LOGIC, or a 2nd output device / AEC for real "
                  "acoustic barge-in. Proceeding (best-effort).\n")
    if args.denoise:
        config.setdefault("sherpa", {})["denoise_enabled"] = True

    llm_cfg = config.get("llm", {}) or {}
    requested_models = None
    if args.llm != "echo" and str(llm_cfg.get("backend", "ollama")).lower() == "ollama":
        requested_models = tuple(dict.fromkeys(
            str(value) for value in (
                args.model or llm_cfg.get("main_model"),
                args.fast_model or llm_cfg.get("fast_model"),
                llm_cfg.get("router_model"),
            )
            if value
        ))
    problems = _preflight(
        config,
        llm=args.llm,
        models_needed=requested_models,
        # --inject explicitly replaces both device streams and has no acoustic
        # echo route. It is the intentional headless exception; normal --check
        # and acoustic runs still require real devices + verified OS EC routing.
        require_audio_devices=not args.inject,
        require_os_echo_route=not args.inject,
    )
    if args.check or problems:
        if problems:
            print("PREFLIGHT PROBLEMS:")
            for prob in problems:
                print(f"  - {prob}")
            print("\nFix these, then re-run. (--list-devices to see audio devices.)")
            return 1 if not args.check else 1
        print("preflight OK: models + audio look ready.")
        return 0

    # Resolve which scenarios to run.
    if args.suite:
        try:
            chosen = resolve_suite(args.suite)
        except KeyError:
            print(f"unknown suite: {args.suite!r}. Use --list-suites. "
                  f"Known: {', '.join(suite_names())}")
            return 2
        if not chosen:
            print(f"suite {args.suite!r} resolved to no scenarios.")
            return 2
    elif args.all:
        chosen = list(SCENARIOS)
    elif args.scenario:
        try:
            chosen = [by_name(n) for n in args.scenario]
        except KeyError as exc:
            print(f"unknown scenario: {exc}. Use --list.")
            return 2
    else:
        print("nothing to run: pass --scenario <name>, --suite <name>, --all, --list, or --check.")
        return 2

    repeat = max(1, args.repeat)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    root = Path(args.out_dir) if args.out_dir else Path("logs/live") / run_id
    print(f"\n=== live validation run {run_id} -> {root} ===")
    if args.inject:
        print("Inject mode: physical microphone and speakers will not be opened.")
    else:
        print("Make sure your speakers and mic are on, at a sane volume, near each other.")
    _gain = config.get("sherpa", {}).get("input_gain")
    _uvol = args.user_volume if args.user_volume is not None else 1.0
    _denoise = bool(config.get("sherpa", {}).get("denoise_enabled"))
    print(f"SNR knobs: input_gain={_gain} (--input-gain), "
          f"user_volume={_uvol} (--user-volume; acoustic only), "
          f"noise_snr={args.noise_snr} dB (--noise-snr), denoise={_denoise} (--denoise)\n")

    rc = 0
    runs: list[dict] = []  # collected for the consolidated suite report
    plan = [(s, r) for s in chosen for r in range(repeat)]
    for scenario, rep in plan:
        out_dir = root / scenario.name if repeat == 1 else root / scenario.name / f"rep{rep + 1:02d}"
        label = scenario.name if repeat == 1 else f"{scenario.name} (rep {rep + 1}/{repeat})"
        print(f"\n--- scenario: {label} ---")
        convo = None
        try:
            convo = LiveConversation(
                config,
                llm_backend=args.llm,
                main_model=args.model,
                fast_model=args.fast_model,
                out_dir=out_dir,
                user_speaker_id=args.user_speaker_id,
                user_speed=args.user_speed,
                capture_assistant_audio=not args.no_assistant_audio,
                response_timeout=args.response_timeout,
                inject=args.inject,
                user_volume=args.user_volume,
                noise_snr_db=args.noise_snr,
            )
            convo.start()
            time.sleep(0.5)  # settle the mic
            events = convo.run_scenario(scenario)
        except KeyboardInterrupt:
            print("interrupted.")
            rc = 130
            if convo is not None:
                convo.stop()
            break
        except Exception as exc:  # noqa: BLE001
            log.exception("scenario %s failed", scenario.name)
            print(f"  FAILED: {exc}")
            rc = 1
            if convo is not None:
                convo.stop()
            continue
        else:
            if convo is not None:
                convo.stop()
        capture = getattr(convo, "capture_verdict", None)
        write_timeline(events, out_dir)
        write_latency_report(events, out_dir)
        grade = write_grade(events, out_dir, capture=capture, scenario=scenario)
        write_summary(scenario, events, out_dir, voice=convo.user.voice, capture=capture)
        runs.append({"scenario": scenario, "events": events, "capture": capture})
        expected_barges, expected_responses = _expected_machine_grade_counts(
            scenario
        )
        rc = _machine_grade_exit_code(
            rc,
            grade,
            inject=args.inject,
            expected_barges=expected_barges,
            expected_responses=expected_responses,
        )
        fd = (capture or {}).get("full_duplex", "n/a")
        acc = grade.get("aggregate", {}).get("stt_score_median")
        stt_domain = "injected" if args.inject else "over-the-air"
        print(f"  full_duplex: {fd}  |  {stt_domain} STT accuracy (median): {acc}")
        r = grade.get("response", {}).get("aggregate", {})
        if r.get("n"):
            print(
                f"  response quality: verdict {r.get('verdict')}, median "
                f"{r.get('response_score_median')}, ok {r.get('n_ok')}/{r.get('n')}, "
                f"forbidden hits {r.get('n_forbidden_hits')}"
            )
        b = grade.get("barge_in", {})
        if b:
            rate = b.get("stops_when_barged_rate")
            rate_str = "n/a" if rate is None else f"{rate}"
            print(
                f"  barge-in (inject-mode): stops {b.get('n_stopped')}/"
                f"{b.get('n_intended_barges')} (rate {rate_str}), "
                f"median detector-to-FIFO cut {b.get('stop_latency_ms_median')}ms, "
                f"self-interrupts {b.get('self_interrupt_count')} "
                f"({b.get('verdict')})"
            )
        print(f"  -> {out_dir}/summary.md  (timeline.json, latency.json, grade.json, "
              f"user/, assistant/, heard_over_air.wav)")

    # Consolidated suite report whenever more than one run was produced (a suite,
    # --all, repeated scenarios, or several --scenario flags): one pooled latency
    # distribution + STT + response dashboard across everything that ran.
    if len(runs) > 1:
        suite = write_suite_report(runs, root)
        fa = suite.get("latency", {}).get("first_audio_ms", {})
        resp = suite.get("response", {})
        print(f"\n=== SUITE ({suite.get('n_scenarios')} scenarios) ===")
        if fa:
            print(f"  first_audio: p50 {fa.get('p50')} | p90 {fa.get('p90')} | "
                  f"p99 {fa.get('p99')} ms  (n={fa.get('n')} turns)")
        if resp.get("n"):
            print(f"  response quality: verdict {resp.get('verdict')}, median "
                  f"{resp.get('response_score_median')}, ok {resp.get('n_ok')}/{resp.get('n')}, "
                  f"forbidden hits {resp.get('n_forbidden_hits')}")
        print(f"  -> {root}/SUITE.md  (SUITE.json)")

    print(f"\nDone. Artifacts under {root}. Read each scenario's summary.md to grade it.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
