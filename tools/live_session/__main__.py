"""CLI for the live on-hardware validation harness.

    python -m tools.live_session --list
    python -m tools.live_session --scenario baseline_latency_single_turn_qa
    python -m tools.live_session --all
    python -m tools.live_session --check          # preflight only (models/audio)

Runs ONLY on request -- it needs real ASR/TTS/LLM models and real audio hardware.
See docs/live_validation_2026-05.md.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .driver import LiveConversation
from .report import write_latency_report, write_summary, write_timeline
from .scenarios import SCENARIOS, by_name

log = logging.getLogger("speaker.live")


def _preflight(config: dict) -> list[str]:
    """Return a list of human-readable problems (empty == ready to run)."""
    problems: list[str] = []
    try:
        import sounddevice as sd  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        problems.append(f"sounddevice not importable ({exc}); pip install sounddevice + a PortAudio backend")
    sherpa = config.get("sherpa", {}) or {}
    if not sherpa.get("asr_encoder") or not sherpa.get("asr_tokens"):
        problems.append("sherpa.asr_encoder / asr_tokens not set -- no ASR model (config.json)")
    if not sherpa.get("tts_model"):
        problems.append("sherpa.tts_model not set -- no TTS voice for the synthetic user / assistant")
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
    p.add_argument("--list", action="store_true", help="list scenarios and exit")
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
    p.add_argument("--inject", action="store_true",
                   help="feed the synthetic-user audio straight into the recognizer instead of "
                        "playing it over the air -- clean STT->LLM->TTS with no acoustic "
                        "degradation/feedback (the reliable path when built-in speaker+mic is noisy)")
    p.add_argument("--out-dir", default=None, help="artifact root (default logs/live/<run-id>)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    if args.list:
        for s in SCENARIOS:
            print(f"  {s.name:42}  {s.capability}")
        return 0
    if args.list_devices:
        _list_audio_devices()
        return 0

    from core.config import _apply_device_profile, _load_config

    config = _load_config()
    device = args.device or config.get("device", "desktop")
    config = _apply_device_profile(config, device)
    for key in ("input_device", "output_device"):
        val = getattr(args, key)
        if val is not None:
            config.setdefault("sherpa", {})[key] = val
    if args.no_input_gate:
        config.setdefault("input_gate", {})["enabled"] = False

    problems = _preflight(config)
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
    if args.all:
        chosen = list(SCENARIOS)
    elif args.scenario:
        try:
            chosen = [by_name(n) for n in args.scenario]
        except KeyError as exc:
            print(f"unknown scenario: {exc}. Use --list.")
            return 2
    else:
        print("nothing to run: pass --scenario <name>, --all, --list, or --check.")
        return 2

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    root = Path(args.out_dir) if args.out_dir else Path("logs/live") / run_id
    print(f"\n=== live validation run {run_id} -> {root} ===")
    print("Make sure your speakers and mic are on, at a sane volume, near each other.\n")

    rc = 0
    for scenario in chosen:
        out_dir = root / scenario.name
        print(f"\n--- scenario: {scenario.name} ---")
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
            )
            convo.start()
            time.sleep(0.5)  # settle the mic
            events = convo.run_scenario(scenario)
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
            if convo is not None:
                convo.stop()
        write_timeline(events, out_dir)
        write_latency_report(events, out_dir)
        write_summary(scenario, events, out_dir, voice=convo.user.voice)
        print(f"  -> {out_dir}/summary.md  (timeline.json, latency.json, user/, assistant/, heard_over_air.wav)")

    print(f"\nDone. Artifacts under {root}. Read each scenario's summary.md to grade it.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
