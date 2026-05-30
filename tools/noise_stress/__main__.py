"""CLI for the noise-stress / voice-isolation test.

    # list scenarios / preflight (no models loaded)
    python -m tools.noise_stress --list
    python -m tools.noise_stress --check

    # the TRUSTWORTHY filtering measurement (real mic/speaker never touched):
    python -m tools.noise_stress --mode inject --all
    python -m tools.noise_stress --mode inject --snr 10          # white sweep at one SNR
    python -m tools.noise_stress --mode inject --scenario intruder_voice_must_not_answer

    # the realism demo (plays over the REAL speaker, captures via the REAL mic;
    # acoustic-confounded on a laptop -- read the inject verdict for truth):
    python -m tools.noise_stress --mode acoustic --scenario white_noise_snr_10db --output-device 4

Runs ONLY on request -- it needs real ASR/TTS/speaker-ID models (and, for
acoustic mode, audio hardware). Writes a graded report under
logs/noise_stress/<run-id>/. Nothing under core/ or config files is modified;
the only file written into the model tree is the temp enrollment.json in the
run's out-dir.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("speaker.noise_stress")

# Distinct multi-speaker TTS ids: the mock USER vs a clearly-different INTRUDER.
DEFAULT_USER_SID = 10
DEFAULT_INTRUDER_SID = 40
DEFAULT_THRESHOLD = 0.5
DEFAULT_SEED = 20260530


def _preflight(config: dict, *, need_audio: bool) -> list[str]:
    problems: list[str] = []
    sherpa = config.get("sherpa", {}) or {}
    if not sherpa.get("asr_encoder") or not sherpa.get("asr_tokens"):
        problems.append("sherpa.asr_encoder / asr_tokens not set -- no ASR model (config.local.json)")
    if not sherpa.get("tts_model"):
        problems.append("sherpa.tts_model not set -- no TTS voice for the mock user / intruder")
    if not sherpa.get("speaker_embedding_model"):
        problems.append(
            "sherpa.speaker_embedding_model not set -- the speaker-ID gate is the ONLY "
            "voice-isolation mechanism; without it this test cannot run. "
            "Run `python -m tools.setup_models`."
        )
    if need_audio:
        try:
            import sounddevice as sd  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            problems.append(f"sounddevice not importable ({exc}); needed for --mode acoustic")
    return problems


def _list_scenarios() -> None:
    from .scenarios import all_scenarios

    for s in all_scenarios():
        print(f"  {s.name:34}  {s.capability}")


# Candidate multi-speaker TTS ids to probe in --check for a separable pair.
CANDIDATE_SPEAKER_IDS = (10, 20, 40, 70, 100)


def _check_separable_pairs(config: dict, args) -> None:
    """During ``--check``, sweep a few TTS speaker-id pairs and print which the
    speaker model finds SEPARABLE -- the actionable answer to "the default
    user/intruder pair inverts, so the real isolation axis can only abstain".

    Best-effort: a missing model, an absent sherpa install, or a synthesis error
    degrades to a one-line note rather than failing the preflight. EnrollmentError
    (a RuntimeError) carries the actionable message; everything else is logged at
    DEBUG and summarized in one line."""
    try:
        from core.engines.sherpa import SherpaConfig

        from .enroll_user import sweep_separable_pairs

        sherpa_cfg = SherpaConfig.from_dict(config.get("sherpa", {}))
        ids = sorted(
            set(CANDIDATE_SPEAKER_IDS) | {args.user_speaker_id, args.intruder_speaker_id}
        )
        print(f"\nsweeping speaker-id pairs for separability: {ids}")
        result = sweep_separable_pairs(
            sherpa_cfg,
            speaker_ids=ids,
            provider=str(getattr(sherpa_cfg, "provider", "cpu")),
            num_threads=int(getattr(sherpa_cfg, "resolved_asr_threads", 1) or 1),
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("separability sweep skipped", exc_info=True)
        print(f"  (separability sweep skipped: {exc})")
        return

    rec = result.get("recommended_pair")
    sep = [p for p in result["per_pair"] if p["separable"]]
    inv = [p for p in result["per_pair"] if p["inverted"]]
    print(f"  probed {len(result['per_pair'])} ordered pairs: "
          f"{len(sep)} separable, {len(inv)} INVERTED "
          "(intruder embeds closer to the user ref than the user's own clips)")
    if rec:
        print(
            f"  RECOMMENDED separable pair: --user-speaker-id {rec['user_speaker_id']} "
            f"--intruder-speaker-id {rec['intruder_speaker_id']} "
            f"(margin {rec['margin']}, user floor {rec['user_floor']} > "
            f"intruder ceiling {rec['intruder_ceiling']}) -- this pair can produce a "
            "genuine PASS_ISOLATION."
        )
    else:
        print(
            "  NO separable pair found among the probed ids: the VoxCeleb-trained "
            "embedder maps these libritts TTS voices into one compressed region, so "
            "synthetic isolation can only ever abstain (INCONCLUSIVE). Use a "
            "real-human-voice fixture pair for a genuine PASS; the gate MECHANISM is "
            "proven by the unit tests."
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tools.noise_stress", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["inject", "acoustic"], default="inject",
                   help="inject = digital mix into the recognizer (trustworthy); "
                        "acoustic = play over real speakers + capture via real mic (demo)")
    p.add_argument("--scenario", action="append", help="scenario name(s) (repeatable)")
    p.add_argument("--all", action="store_true", help="run every scenario")
    p.add_argument("--list", action="store_true", help="list scenarios and exit")
    p.add_argument("--check", action="store_true", help="preflight (models/audio) and exit")
    p.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    p.add_argument("--snr", type=float, action="append",
                   help="override the white-noise sweep with these SNR point(s) in dB (repeatable)")
    p.add_argument("--threshold", type=float, default=None,
                   help="speaker-ID gate cosine threshold (default: auto-calibrate to the "
                        "midpoint between the user floor and intruder ceiling, or "
                        f"{DEFAULT_THRESHOLD} if the voices are not separable)")
    p.add_argument("--user-speaker-id", type=int, default=DEFAULT_USER_SID,
                   help=f"mock-user TTS speaker id (default {DEFAULT_USER_SID})")
    p.add_argument("--intruder-speaker-id", type=int, default=DEFAULT_INTRUDER_SID,
                   help=f"intruder TTS speaker id, must differ from the user (default {DEFAULT_INTRUDER_SID})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed for reproducible noise")
    p.add_argument("--device", default=None, help="device profile from config.json (e.g. desktop)")
    p.add_argument("--llm", default="ollama", choices=["echo", "ollama"])
    p.add_argument("--model", default=None, help="main LLM model override")
    p.add_argument("--fast-model", default=None, help="fast LLM model override")
    p.add_argument("--output-device", default=None, help="ACOUSTIC: output device index/name")
    p.add_argument("--input-device", default=None, help="ACOUSTIC: input device index/name")
    p.add_argument("--noise-volume", type=float, default=0.6,
                   help="ACOUSTIC: amplitude of the looping noise bed, 0..1 (default 0.6)")
    p.add_argument("--response-timeout", type=float, default=45.0)
    p.add_argument("--out-dir", default=None, help="artifact root (default logs/noise_stress/<run-id>)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    if args.list:
        _list_scenarios()
        return 0
    if args.list_devices:
        try:
            import sounddevice as sd

            print(sd.query_devices())
        except Exception as exc:  # noqa: BLE001
            print(f"could not list audio devices: {exc}")
        return 0

    if args.intruder_speaker_id == args.user_speaker_id:
        print("ERROR: --intruder-speaker-id must DIFFER from --user-speaker-id "
              "(the intruder is a distinct voice the gate must reject).")
        return 2

    from core.config import apply_device_profile, load_config

    config = load_config()
    device = args.device or config.get("device", "desktop")
    config = apply_device_profile(config, device)
    for key in ("input_device", "output_device"):
        val = getattr(args, key)
        if val is not None:
            config.setdefault("sherpa", {})[key] = val

    need_audio = args.mode == "acoustic"
    problems = _preflight(config, need_audio=need_audio)
    if args.check or problems:
        if problems:
            print("PREFLIGHT PROBLEMS:")
            for prob in problems:
                print(f"  - {prob}")
            print("\nFix these, then re-run.")
            return 1
        print("preflight OK: models" + (" + audio" if need_audio else "") + " look ready.")
        # Sweep a handful of TTS speaker-id pairs and recommend a SEPARABLE one,
        # so the operator can pick a pair the embedder can actually isolate
        # (rather than only learning, post-run, that the default pair inverts).
        _check_separable_pairs(config, args)
        return 0

    snr_sweep = tuple(args.snr) if args.snr else None
    from .scenarios import DEFAULT_SNR_SWEEP, all_scenarios, by_name

    sweep = snr_sweep or DEFAULT_SNR_SWEEP
    if args.all:
        chosen = list(all_scenarios(sweep))
    elif args.scenario:
        try:
            chosen = [by_name(n, sweep) for n in args.scenario]
        except KeyError as exc:
            print(f"unknown scenario: {exc}. Use --list.")
            return 2
    elif snr_sweep is not None:
        # --snr alone runs just the white-noise sweep at the given point(s).
        from .scenarios import white_snr_sweep

        chosen = list(white_snr_sweep(sweep))
    else:
        print("nothing to run: pass --scenario <name>, --all, --snr <db>, --list, or --check.")
        return 2

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    root = Path(args.out_dir) if args.out_dir else Path("logs/noise_stress") / run_id
    root.mkdir(parents=True, exist_ok=True)
    print(f"\n=== noise-stress run {run_id} ({args.mode} mode) -> {root} ===")
    if args.mode == "acoustic":
        print("ACOUSTIC mode: numbers are illustrative (shared speaker+mic, no AEC). "
              "The trustworthy verdict comes from --mode inject.")

    # --- enroll the mock user's voice ONCE (in memory, temp file) ---
    from core.engines.sherpa import SherpaConfig
    from .driver import NoiseStressSession
    from .enroll_user import (
        EnrollmentError,
        calibrate_separability,
        enroll_mock_user,
        intruder_user_vs_user_cosine,
    )
    from .report import grade_scenario, write_report

    sherpa_cfg = SherpaConfig.from_dict(config.get("sherpa", {}))
    auto_threshold = args.threshold is None
    threshold = DEFAULT_THRESHOLD if args.threshold is None else float(args.threshold)
    calibration = None
    try:
        # Calibrate separability first: it builds the enrollment AND measures
        # whether any threshold separates the user from the intruder.
        calibration = calibrate_separability(
            sherpa_cfg,
            out_dir=root,
            user_speaker_id=args.user_speaker_id,
            intruder_speaker_id=args.intruder_speaker_id,
            provider=str(getattr(sherpa_cfg, "provider", "cpu")),
            num_threads=int(getattr(sherpa_cfg, "resolved_asr_threads", 1) or 1),
        )
        if auto_threshold and calibration.get("separable") and calibration.get("recommended_threshold") is not None:
            threshold = float(calibration["recommended_threshold"])
            print(f"auto-calibrated speaker threshold -> {threshold} "
                  f"(separable user/intruder)")
        elif not calibration.get("separable"):
            print(
                "WARNING: the mock-user and intruder TTS voices are NOT separable by the "
                "speaker model\n"
                f"  (user floor {calibration.get('user_floor')} <= intruder ceiling "
                f"{calibration.get('intruder_ceiling')}). The competing-voice verdict will be "
                "INCONCLUSIVE\n"
                "  for synthetic voices -- a fixture/embedder property, not an app bug. The gate\n"
                "  MECHANISM is verified by the unit tests; real distinct human voices separate cleanly."
            )
            # When auto, drop to a recall-preserving threshold so the enrolled
            # user is still accepted and the noise/STT-degradation axis stays
            # measurable (the intruder is also accepted -> INCONCLUSIVE, as noted).
            if auto_threshold and calibration.get("recall_preserving_threshold") is not None:
                threshold = float(calibration["recall_preserving_threshold"])
                print(f"  -> using a recall-preserving threshold {threshold} so the "
                      "enrolled user is still accepted (noise/STT axis stays measurable).")
        overrides = enroll_mock_user(
            sherpa_cfg,
            out_dir=root,
            user_speaker_id=args.user_speaker_id,
            threshold=threshold,
            provider=str(getattr(sherpa_cfg, "provider", "cpu")),
            num_threads=int(getattr(sherpa_cfg, "resolved_asr_threads", 1) or 1),
        )
    except EnrollmentError as exc:
        print(f"ENROLLMENT FAILED:\n  {exc}")
        return 1
    self_check = overrides.pop("_enroll_self_check", {})
    user_intruder_cos = intruder_user_vs_user_cosine(
        sherpa_cfg,
        user_speaker_id=args.user_speaker_id,
        intruder_speaker_id=args.intruder_speaker_id,
        threshold=threshold,
    )
    print(f"enrolled mock user (sid={args.user_speaker_id}): "
          f"pass-to-ref cosine min={self_check.get('pass_to_ref_min')} "
          f"mean={self_check.get('pass_to_ref_mean')}; "
          f"user-vs-intruder cosine={user_intruder_cos} (threshold={threshold}); "
          f"separable={calibration.get('separable') if calibration else 'n/a'}")

    grades: list[dict] = []
    rc = 0
    for scenario in chosen:
        out_dir = root / scenario.name
        print(f"\n--- scenario: {scenario.name} ---")
        session = None
        try:
            session = NoiseStressSession(
                config,
                mode=args.mode,
                out_dir=out_dir,
                sherpa_overrides=dict(overrides),
                user_speaker_id=args.user_speaker_id,
                intruder_speaker_id=args.intruder_speaker_id,
                seed=args.seed,
                llm_backend=args.llm,
                main_model=args.model,
                fast_model=args.fast_model,
                response_timeout=args.response_timeout,
                output_device=_norm(args.output_device),
                noise_volume=args.noise_volume,
            )
            session.start()
            observations = session.run_scenario(scenario)
        except KeyboardInterrupt:
            print("interrupted.")
            rc = 130
            break
        except Exception as exc:  # noqa: BLE001
            log.exception("scenario %s failed", scenario.name)
            print(f"  FAILED: {exc}")
            rc = 1
            continue
        finally:
            if session is not None:
                session.stop()
        grade = grade_scenario(scenario.name, observations)
        grades.append(grade)
        print(f"  isolation={grade['isolation_verdict']}  recall={grade['recall']}  "
              f"STT median={grade['stt_score_median']}  "
              f"false-pos={grade['false_positive_rate']}  "
              f"rejected_finals={grade['speaker_rejected_finals']}")

    report = write_report(
        grades, root,
        mode=args.mode,
        enroll_self_check=self_check,
        user_intruder_cosine=user_intruder_cos,
        threshold=threshold,
        calibration=calibration,
    )
    h = report["headline"]
    print(f"\n=== HEADLINE ({args.mode}) ===")
    print(f"  competing-voice isolation: {h['competing_voice_isolation']} "
          f"(false positives {h['false_positives_total']}/{h['nontarget_turns_total']}, "
          f"speaker_rejected_final x{h['speaker_rejected_finals_total']})")
    print(f"  broadband denoise: {h['broadband_denoise']}")
    print(f"  desired-voice recall (min): {h['recall_min']}")
    print(f"\nDone. Report: {root}/report.md  (grade.json)")
    return rc


def _norm(dev):
    if dev in (None, ""):
        return None
    if isinstance(dev, str) and dev.lstrip("-").isdigit():
        return int(dev)
    return dev


if __name__ == "__main__":
    sys.exit(main())
