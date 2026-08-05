"""CLI: real-model latency benchmark.

  python -m tools.bench --fake                       # no models, plumbing smoke
  python -m tools.bench --profile phone \\
      --fixtures tests/fixture_audio/virtual_real_world --limit 20
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from core.app import _apply_device_profile, _load_config
from core.config import FINAL_STT_PROFILE_NAMES

from . import report, runner

_BUILTIN_FAKE_CASES = [
    ("greeting", "hello there"),
    ("time", "what time is it"),
    ("research", "research local speech to text engines"),
    ("weather", "what is the weather like today"),
]


def _fake_cases(fixtures_dir: str | None, limit: int | None) -> list[tuple[str, str]]:
    if not fixtures_dir:
        return _BUILTIN_FAKE_CASES[: limit or len(_BUILTIN_FAKE_CASES)]
    fixtures = runner.discover_fixtures(fixtures_dir, limit=limit)
    if not fixtures:
        return _BUILTIN_FAKE_CASES[: limit or len(_BUILTIN_FAKE_CASES)]
    # In fake mode there's no real ASR, so feed a placeholder utterance per
    # fixture -- enough to exercise the brain + metrics + report end to end.
    return [(fx.name, "what time is it") for fx in fixtures]


def _build_sherpa_config(config: dict, overrides: dict[str, str]):
    from core.engines.sherpa import SherpaConfig

    merged = {**config.get("sherpa", {}), **overrides}
    return SherpaConfig.from_dict(merged)


def _build_llms(config: dict, paths):
    from core.llm import LlamaCppLLM

    llm_cfg = config.get("llm", {})
    common = dict(
        n_ctx=llm_cfg.get("n_ctx", 2048),
        n_threads=llm_cfg.get("n_threads"),
        n_threads_batch=llm_cfg.get("n_threads_batch"),
        n_gpu_layers=llm_cfg.get("n_gpu_layers", 0),
        chat_format=llm_cfg.get("chat_format"),
        think=llm_cfg.get("think", False),
        options=llm_cfg.get("options"),
        type_k=llm_cfg.get("type_k"),
        type_v=llm_cfg.get("type_v"),
    )
    main = LlamaCppLLM(paths.main_gguf, **common)
    if paths.fast_gguf and os.path.abspath(paths.fast_gguf) == os.path.abspath(
        paths.main_gguf
    ):
        fast = main
    else:
        fast = LlamaCppLLM(paths.fast_gguf, **common) if paths.fast_gguf else None
    return main, fast


def _make_stdio_utf8_safe() -> None:
    """Best-effort: reconfigure stdout/stderr to UTF-8 with replacement so the
    perf summary (which contains non-ASCII glyphs) prints cleanly on any
    console -- notably the Windows cp1252 console, where the default codec
    would otherwise raise UnicodeEncodeError. ``reconfigure`` exists on the
    py3.7+ ``TextIOWrapper``; guard it so non-stream stdout (pytest capture,
    pipes) is left untouched."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    _make_stdio_utf8_safe()
    parser = argparse.ArgumentParser(description="Real-model latency benchmark.")
    parser.add_argument(
        "--mode",
        choices=("legacy", "production"),
        default="legacy",
        help="legacy downloader/fixture bench or strict current-production replay",
    )
    device = parser.add_mutually_exclusive_group()
    device.add_argument(
        "--profile",
        default=None,
        help="legacy device profile (default: phone)",
    )
    device.add_argument(
        "--device",
        default=None,
        help="explicit concrete production device profile",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fixtures", default=None)
    source.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="strict retained schema-v2/v3/v4 corpus.json (production mode)",
    )
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument(
        "--local-config",
        type=Path,
        default=Path("config.local.json"),
    )
    parser.add_argument(
        "--final-stt-profile",
        choices=FINAL_STT_PROFILE_NAMES,
        default=None,
        help="explicit atomic final-STT profile (production mode)",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap number of fixtures")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="plumbing smoke run with ScriptedEngine + EchoLLM (no models/downloads)",
    )
    parser.add_argument("--out", default="test-reports/perf", help="report output dir")
    parser.add_argument("--models-manifest", default=None, help="JSON of model coordinates")
    parser.add_argument("--cache-dir", default=".cache/bench-models")
    parser.add_argument("--stream-tts", action="store_true")
    parser.add_argument(
        "--md-summary",
        default=os.environ.get("GITHUB_STEP_SUMMARY"),
        help="append a markdown summary to this file (CI step summary)",
    )
    args = parser.parse_args(argv)

    if args.mode == "production":
        if args.fake:
            parser.error("--fake is incompatible with --mode production")
        if args.device in (None, "", "auto"):
            parser.error("--mode production requires an explicit concrete --device")
        if args.profile is not None:
            parser.error("--mode production uses --device, not legacy --profile")
        if args.corpus is None:
            parser.error("--mode production requires --corpus")
        if args.fixtures is not None:
            parser.error("--fixtures is legacy-only; production requires --corpus")
        if args.final_stt_profile is None:
            parser.error("--mode production requires --final-stt-profile")
        if args.limit is not None:
            parser.error("--limit cannot reduce a bound production corpus")
        if args.stream_tts:
            parser.error("production replay uses the configured TTS policy")
        if args.models_manifest is not None or args.cache_dir != ".cache/bench-models":
            parser.error("legacy model download flags are invalid in production mode")

        from core.sysinfo import SystemMonitor

        from .production import (
            bind_ollama_identity,
            corpus_identity,
            load_production_corpus,
            prepare_production,
            verify_ollama_identity,
            verify_production_inputs,
        )

        try:
            corpus = load_production_corpus(args.corpus)
            prepared = prepare_production(
                config_path=args.config,
                local_config_path=args.local_config,
                device_profile=args.device,
                final_stt_profile=args.final_stt_profile,
            )
            ollama_identity = bind_ollama_identity(prepared.config)
        except (Exception, SystemExit):
            parser.error("production corpus/config validation failed")

        sample_interval_sec = 1.0
        monitor = None
        system = {}
        production_run = None
        replay_failed = False
        try:
            monitor = SystemMonitor(interval=sample_interval_sec)
            monitor.start()
            production_run = runner.run_production(
                corpus,
                prepared.config,
                device_profile=prepared.device_profile,
                final_stt_profile=prepared.final_stt_profile,
                final_stt_profile_sha256=prepared.final_stt_profile_sha256,
                final_stt_profile_schema_version=(
                    prepared.final_stt_profile_schema_version
                ),
                production_config_sha256=prepared.production_config_sha256,
                execution_config_sha256=prepared.execution_config_sha256,
                replay_safety_schema_version=prepared.replay_safety_schema_version,
                replay_safety_sha256=prepared.replay_safety_sha256,
                active_artifacts_sha256=prepared.active_artifacts_sha256,
                active_artifact_roots=prepared.active_artifact_roots,
                active_artifact_files=prepared.active_artifact_files,
                active_artifact_bytes=prepared.active_artifact_bytes,
                source_revision=prepared.source_revision,
                runtime_identities=prepared.runtime_identities,
                ollama_identity_sha256=ollama_identity.sha256,
                ollama_role_contracts=ollama_identity.role_contracts,
                after_build=lambda: monitor.mark("after_build"),
            )
        except (Exception, SystemExit):
            replay_failed = True
        finally:
            if monitor is not None:
                try:
                    system = monitor.stop()
                except Exception:
                    replay_failed = True
        if replay_failed or production_run is None:
            parser.error("production replay failed")
        try:
            verify_ollama_identity(ollama_identity, prepared.config)
            verify_production_inputs(
                prepared,
                corpus,
                config_path=args.config,
                local_config_path=args.local_config,
            )
        except (Exception, SystemExit):
            parser.error("production inputs changed during replay")

        out_dir = Path(args.out) / args.device
        try:
            index, payload = report.write_production_reports(
                out_dir,
                production_run,
                corpus_identity(corpus),
                system,
                sample_interval_sec=sample_interval_sec,
            )
            summary = report.production_markdown_summary(payload)
            if args.md_summary:
                with open(args.md_summary, "a", encoding="utf-8") as fh:
                    fh.write(summary)
        except (Exception, SystemExit):
            parser.error("production report publication failed")
        print(f"[bench] wrote {index}")
        print(summary)
        return 0

    if args.device is not None:
        parser.error("--device requires --mode production")
    if args.corpus is not None:
        parser.error("--corpus requires --mode production")
    if args.final_stt_profile is not None:
        parser.error("--final-stt-profile requires --mode production")
    if args.config != Path("config.json") or args.local_config != Path(
        "config.local.json"
    ):
        parser.error("--config/--local-config require --mode production")

    profile = args.profile or "phone"
    fixtures_dir = args.fixtures or "tests/fixture_audio/virtual_real_world"
    out_dir = Path(args.out) / profile

    if args.fake:
        cases = _fake_cases(fixtures_dir if os.path.isdir(fixtures_dir) else None, args.limit)
        samples = runner.run_fake(cases)
    else:
        from .models import fetch_models

        config = _apply_device_profile(_load_config(), profile)
        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        paths = fetch_models(args.cache_dir, token=token, manifest_path=args.models_manifest)
        sherpa_cfg = _build_sherpa_config(config, paths.sherpa_overrides())
        main_llm, fast_llm = _build_llms(config, paths)
        fixtures = runner.discover_fixtures(fixtures_dir, limit=args.limit)
        if not fixtures:
            raise SystemExit(f"No fixtures in {fixtures_dir!r}")
        samples = runner.run_real(
            fixtures, sherpa_cfg, main_llm, fast_llm, stream_tts=args.stream_tts
        )

    index = report.write_reports(out_dir, profile, samples)
    print(f"[bench] wrote {index}")
    if args.md_summary:
        with open(args.md_summary, "a", encoding="utf-8") as fh:
            fh.write(report.markdown_summary(profile, samples))
    # Echo the calibration table to stdout for quick reading.
    print(report.markdown_summary(profile, samples))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
